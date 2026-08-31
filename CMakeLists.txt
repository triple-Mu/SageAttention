# SageAttention: single abi3 extension (sageattention/_C.abi3.so) carrying
# every kernel family, with per-source CUDA gencode. Driven by setup.py
# (CMakeExtension) but usable standalone:
#   cmake -S . -B build/cmake -G Ninja -DPython_EXECUTABLE=$(which python) \
#         -DSAGE_CUDA_ARCHS="8.0;8.9;9.0;12.0"
#   cmake --build build/cmake --target _C
cmake_minimum_required(VERSION 3.26)

# We emit --generate-code per source group ourselves (sm_90a/sm_100a mix with
# plain codes); CMake's arch handling must stay fully out of the way, starting
# with the compiler probe.
set(CMAKE_CUDA_ARCHITECTURES OFF)
set(CMAKE_POLICY_DEFAULT_CMP0104 NEW)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/sageattention/_version.py" _sage_version_line
     REGEX "^__version__")
string(REGEX MATCH "\"([^\"]+)\"" _ "${_sage_version_line}")
set(SAGE_VERSION "${CMAKE_MATCH_1}")

project(sageattention VERSION "${SAGE_VERSION}" LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Match the historical torch CUDAExtension semantics: .cu compiled WITHOUT
# -DNDEBUG (device-side assert()s stay live), host .cpp with it.
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# No absolute build-machine paths leaked into the wheel.
set(CMAKE_SKIP_BUILD_RPATH ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH "")

option(SAGE_PTXAS_VERBOSE "Pass -Xptxas=-v (per-kernel register/smem report)" OFF)
option(SAGE_LINEINFO "Pass -lineinfo for ncu source correlation" OFF)
option(SAGE_BUILD_VARLEN "Build the varlen (packed cu_seqlens layout) kernel TUs" ON)
# quant_v_fp8 runs transpose + per-channel fp8 quantization as one kernel by
# default, skipping the fp16 V^T buffer. OFF restores the two-kernel pair. The
# two are bit-identical (the golden dumps gate that), so this switch buys back
# the old timing, not old numbers.
option(SAGE_FUSED_V_QUANT "quant_v_fp8: fuse the V transpose into the fp8 quantization kernel" ON)
option(SAGE_CCACHE "Launch the compilers through ccache when one is installed" ON)

# ccache halves a rebuild of this tree and is bit-exact by construction (a hit
# replays the cached object), so the SASS gate still compares what it did
# before - as long as both sides of a comparison are configured the same way.
# An explicitly set launcher always wins.
if(SAGE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    foreach(_sage_lang C CXX CUDA)
      if(NOT CMAKE_${_sage_lang}_COMPILER_LAUNCHER)
        set(CMAKE_${_sage_lang}_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
      endif()
    endforeach()
    message(STATUS "[sageattention] ccache: ${CCACHE_PROGRAM}")
  else()
    message(STATUS "[sageattention] ccache: not found, compiling directly")
  endif()
endif()

# Arithmetic opt-ins. Each one perturbs kernel results (never the API), so they
# default OFF and the reference build stays bit-exact; turn one on only with
# per-arch accuracy and throughput numbers in hand. The chosen state is recorded
# in the generated sageattn_build_config.h.
option(SAGE_OPT_MAGIC_I2F "sm80 QK dequant: magic-number int->float instead of I2F" OFF)
option(SAGE_OPT_HMUL2_NORM "normalize_d fp16 path: single-rounding __hmul2" OFF)
option(SAGE_OPT_FUSED_EPILOGUE "sm89 epilogue: v_scale*RO+v_mean as one fmaf pass" OFF)
set(SAGE_NVCC_THREADS "8" CACHE STRING "nvcc --threads value (per-TU arch parallelism)")
set(SAGE_CUDA_ARCHS "" CACHE STRING
    "Target compute capabilities, e.g. \"8.0;8.9+PTX;9.0\" (empty: env/auto-detect)")

find_package(CUDAToolkit REQUIRED)
include(cmake/SageTorch.cmake)
include(cmake/SageArch.cmake)

# ---------------------------------------------------------------- nvcc flags
# The -D__CUDA_NO_* / --expt-relaxed-constexpr trio matches what
# torch.utils.cpp_extension always injected (COMMON_NVCC_FLAGS); dropping them
# would silently change half/bf16 overload resolution in the kernels.
set(SAGE_NVCC_FLAGS
  --use_fast_math
  --threads=${SAGE_NVCC_THREADS}
  -diag-suppress=174
  -U__CUDA_NO_HALF_OPERATORS__
  -U__CUDA_NO_HALF_CONVERSIONS__
  -D__CUDA_NO_BFLOAT16_CONVERSIONS__
  -D__CUDA_NO_HALF2_OPERATORS__
  --expt-relaxed-constexpr)
if(SAGE_PTXAS_VERBOSE)
  list(APPEND SAGE_NVCC_FLAGS -Xptxas=-v)
endif()
if(SAGE_LINEINFO)
  list(APPEND SAGE_NVCC_FLAGS -lineinfo)
endif()
separate_arguments(_sage_extra_nvcc NATIVE_COMMAND "$ENV{NVCC_APPEND_FLAGS}")
list(APPEND SAGE_NVCC_FLAGS ${_sage_extra_nvcc})

# ---------------------------------------------------------------- sources
set(SAGE_SRC_BINDING
  csrc/sageattn/python_module.cpp
  csrc/sageattn/ops.cpp
  csrc/sageattn/plan.cpp
  csrc/sageattn/fwd_cuda.cu
  csrc/sageattn/fwd_varlen_cuda.cu
  csrc/sageattn/quant_cuda.cu)
set(SAGE_SRC_FUSED
  csrc/fused/fused.cu
  csrc/fused/quant_per_thread.cu
  csrc/fused/segment_mean.cu)
# The varlen TU is the same kernel body under -DSAGE_VARLEN (set in the file
# itself), so it belongs to the same gencode group as the dense one.
set(SAGE_SRC_SM80 csrc/qattn/qk_int_sv_f16_cuda_sm80.cu)
if(SAGE_BUILD_VARLEN)
  list(APPEND SAGE_SRC_SM80 csrc/qattn/qk_int_sv_f16_cuda_sm80_varlen.cu)
  set(SAGEATTN_BUILD_VARLEN 1)
else()
  set(SAGEATTN_BUILD_VARLEN 0)
  message(STATUS "[sageattention] varlen kernels disabled (SAGE_BUILD_VARLEN=OFF)")
endif()
if(SAGE_FUSED_V_QUANT)
  set(SAGEATTN_FUSED_V_QUANT 1)
else()
  set(SAGEATTN_FUSED_V_QUANT 0)
  message(STATUS "[sageattention] quant_v_fp8 uses the two-kernel V path (SAGE_FUSED_V_QUANT=OFF)")
endif()
# sm90 is a plain source group rather than an OBJECT library, so its varlen TU
# joins the group the way sm80's does; it names its own namespace internally.
set(SAGE_SRC_SM90 csrc/qattn/qk_int_sv_f8_cuda_sm90.cu)
if(SAGE_BUILD_VARLEN)
  list(APPEND SAGE_SRC_SM90 csrc/qattn/qk_int_sv_f8_cuda_sm90_varlen.cu)
endif()
set(SAGE_SRC_SM100
  csrc/qattn/qk_int_sv_f8_cuda_sm100.cu
  csrc/qattn/qk_int_sv_f8_cuda_sm100_ws.cu
  csrc/qattn/qk_int_sv_f8_cuda_sm100_ws_persist.cu)
# Same pattern as sm90: the varlen TU joins the group (and its accel gencode
# list); it names its own namespace internally.
if(SAGE_BUILD_VARLEN)
  list(APPEND SAGE_SRC_SM100 csrc/qattn/qk_int_sv_f8_cuda_sm100_varlen.cu)
endif()
# The sm89 template TUs are compiled twice (OBJECT libraries below): once as
# sage::sm89 with the full fallback arch set, once as sage::sm120 with the
# sm_12x-only set (minus the fp32+fp32 inst_buf variant sm120 never uses).
set(SAGE_SRC_SM89_TUS
  csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn.cu
  csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf.cu
  csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu
  csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn.cu)
set(SAGE_SRC_SM120_TUS
  csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn.cu
  csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu
  csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn.cu)
# The packed-layout TUs of the same family. They compile the same header with
# -DSAGE_VARLEN into <arch>_varlen namespaces, one TU each: the pv_accum_dtype
# plan.cpp resolves to by default for that arch (sm89 fp32+fp16, sm120 fp32).
set(SAGE_SRC_SM89_VARLEN_TUS
  csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_varlen_attn_inst_buf.cu)
set(SAGE_SRC_SM120_VARLEN_TUS
  csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_varlen_attn.cu)

set(SAGE_SOURCES ${SAGE_SRC_BINDING})
set(SAGEATTN_BUILD_SM80 0)
set(SAGEATTN_BUILD_SM89 0)
set(SAGEATTN_BUILD_SM90 0)
set(SAGEATTN_BUILD_SM100 0)
set(SAGEATTN_BUILD_SM120 0)

sage_group_enable(FUSED "${SAGE_SRC_FUSED}")
sage_group_enable(SM80 "${SAGE_SRC_SM80}")
sage_group_enable(SM90 "${SAGE_SRC_SM90}")
sage_group_enable(SM100 "${SAGE_SRC_SM100}")
if(NOT SAGEATTN_BUILD_FUSED)
  message(FATAL_ERROR "internal: the fused quantization kernels serve every "
                      "supported arch and must always be built")
endif()

string(REPLACE ";" "," SAGE_REQUESTED_ARCHS_STR "${SAGE_REQUESTED_ARCHS}")

# Common usage requirements shared by the main target and the OBJECT libs.
add_library(sage_common INTERFACE)
target_include_directories(sage_common INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc
  ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(sage_common INTERFACE sage::torch sage::python_module)
target_compile_definitions(sage_common INTERFACE Py_LIMITED_API=0x03090000)
foreach(_sage_opt MAGIC_I2F HMUL2_NORM FUSED_EPILOGUE)
  # Defined-or-absent, never =0: the kernels guard on #if defined(...).
  # SAGEATTN_OPT_* is the 0/1 mirror that sageattn_build_config.h records.
  set(SAGEATTN_OPT_${_sage_opt} ${SAGE_OPT_${_sage_opt}})
  if(SAGE_OPT_${_sage_opt})
    target_compile_definitions(sage_common INTERFACE SAGE_OPT_${_sage_opt})
    message(STATUS "[sageattention] arithmetic opt-in ON: SAGE_OPT_${_sage_opt}")
  endif()
endforeach()
target_compile_options(sage_common INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:${SAGE_NVCC_FLAGS}>)

# sm89 / sm120: same TUs, different namespace + gencode (single-so ODR rule).
set(SAGE_ARCH_OBJECT_TARGETS "")
if(SAGE_ARCHS_SM89)
  sage_gencode_flags(_sm89_gencode plain "${SAGE_ARCHS_SM89}")
  add_library(sage_sm89_objs OBJECT ${SAGE_SRC_SM89_TUS})
  target_link_libraries(sage_sm89_objs PRIVATE sage_common)
  target_compile_definitions(sage_sm89_objs PRIVATE SAGEATTN_ARCH_NS=sm89)
  target_compile_options(sage_sm89_objs PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${_sm89_gencode}>)
  list(APPEND SAGE_ARCH_OBJECT_TARGETS sage_sm89_objs)
  set(SAGEATTN_BUILD_SM89 1)
  message(STATUS "[sageattention] SM89: ${_sm89_gencode}")
  if(SAGE_BUILD_VARLEN)
    add_library(sage_sm89_varlen_objs OBJECT ${SAGE_SRC_SM89_VARLEN_TUS})
    target_link_libraries(sage_sm89_varlen_objs PRIVATE sage_common)
    target_compile_definitions(sage_sm89_varlen_objs PRIVATE SAGEATTN_ARCH_NS=sm89_varlen)
    target_compile_options(sage_sm89_varlen_objs PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:${_sm89_gencode}>)
    list(APPEND SAGE_ARCH_OBJECT_TARGETS sage_sm89_varlen_objs)
  endif()
else()
  message(STATUS "[sageattention] skip SM89: no requested compute capability is served by it")
endif()
if(SAGE_ARCHS_SM120)
  sage_gencode_flags(_sm120_gencode plain "${SAGE_ARCHS_SM120}")
  add_library(sage_sm120_objs OBJECT ${SAGE_SRC_SM120_TUS})
  target_link_libraries(sage_sm120_objs PRIVATE sage_common)
  target_compile_definitions(sage_sm120_objs PRIVATE SAGEATTN_ARCH_NS=sm120)
  target_compile_options(sage_sm120_objs PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${_sm120_gencode}>)
  list(APPEND SAGE_ARCH_OBJECT_TARGETS sage_sm120_objs)
  set(SAGEATTN_BUILD_SM120 1)
  message(STATUS "[sageattention] SM120: ${_sm120_gencode}")
  if(SAGE_BUILD_VARLEN)
    add_library(sage_sm120_varlen_objs OBJECT ${SAGE_SRC_SM120_VARLEN_TUS})
    target_link_libraries(sage_sm120_varlen_objs PRIVATE sage_common)
    target_compile_definitions(sage_sm120_varlen_objs PRIVATE SAGEATTN_ARCH_NS=sm120_varlen)
    target_compile_options(sage_sm120_varlen_objs PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:${_sm120_gencode}>)
    list(APPEND SAGE_ARCH_OBJECT_TARGETS sage_sm120_varlen_objs)
  endif()
else()
  message(STATUS "[sageattention] skip SM120: no requested compute capability is served by it")
endif()

# Per-family gencode lists for the generated header (backend_serves() input):
# "M.m" -> major*10+minor, cubin entries plus the +PTX subset (plain kinds
# only; the accel families never embed PTX, see sage_gencode_flags).
foreach(_g SM80 SM89 SM90 SM100 SM120)
  set(_cubin_ccs "")
  set(_ptx_ccs "")
  foreach(cc IN LISTS SAGE_ARCHS_${_g})
    string(REPLACE "." "" _n "${cc}")
    list(APPEND _cubin_ccs "${_n}")
    if(NOT SAGE_KIND_${_g} STREQUAL "accel" AND "${cc}" IN_LIST SAGE_PTX_ARCHS)
      list(APPEND _ptx_ccs "${_n}")
    endif()
  endforeach()
  string(REPLACE ";" "," SAGEATTN_${_g}_CUBIN_CCS "${_cubin_ccs}")
  string(REPLACE ";" "," SAGEATTN_${_g}_PTX_CCS "${_ptx_ccs}")
endforeach()

configure_file(cmake/sageattn_build_config.h.in
               "${CMAKE_CURRENT_BINARY_DIR}/generated/sageattn_build_config.h" @ONLY)

# ---------------------------------------------------------------- target
add_library(_C MODULE ${SAGE_SOURCES})
target_link_libraries(_C PRIVATE sage_common)
foreach(objlib IN LISTS SAGE_ARCH_OBJECT_TARGETS)
  target_sources(_C PRIVATE $<TARGET_OBJECTS:${objlib}>)
endforeach()
if(SAGEATTN_BUILD_SM90 OR SAGEATTN_BUILD_SM100)
  target_link_libraries(_C PRIVATE CUDA::cuda_driver)  # cuTensorMapEncodeTiled
endif()

set_target_properties(_C PROPERTIES
  PREFIX ""
  OUTPUT_NAME "_C"
  SUFFIX ".abi3.so"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/sageattention"
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
