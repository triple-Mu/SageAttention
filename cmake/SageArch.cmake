# Arch policy for the single-target build: which source group serves which
# compute capability, per-source --generate-code flags, and the nvcc version
# gates. Byte-compatible with the historical setup.py EXT_SERVES /
# gencode_flags_for logic.
#
# Inputs:  SAGE_CUDA_ARCHS (may be empty -> env/auto-detect inside the resolver)
# Outputs: SAGE_ARCHS_SM80/SM89/SM90/SM100/SM120/FUSED (lists of "M.m")
#          SAGE_PTX_ARCHS (subset of plain archs that also emit PTX)
#          sage_group_enable() to attach gencode flags to a source group

execute_process(
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/resolve_arch_list.py"
          --archs "${SAGE_CUDA_ARCHS}"
  OUTPUT_VARIABLE _sage_arch_spec
  ERROR_VARIABLE _sage_arch_err
  RESULT_VARIABLE _sage_arch_rc
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _sage_arch_rc EQUAL 0)
  message(FATAL_ERROR "arch resolution failed:\n${_sage_arch_err}")
endif()
if(_sage_arch_err)
  message(WARNING "${_sage_arch_err}")
endif()

set(SAGE_REQUESTED_ARCHS "")
set(SAGE_PTX_ARCHS "")
foreach(entry IN LISTS _sage_arch_spec)
  if(entry MATCHES "^([0-9]+\\.[0-9]+)\\+PTX$")
    list(APPEND SAGE_REQUESTED_ARCHS "${CMAKE_MATCH_1}")
    list(APPEND SAGE_PTX_ARCHS "${CMAKE_MATCH_1}")
  else()
    list(APPEND SAGE_REQUESTED_ARCHS "${entry}")
  endif()
endforeach()
message(STATUS "[sageattention] target compute capabilities: ${SAGE_REQUESTED_ARCHS} (PTX: ${SAGE_PTX_ARCHS})")

# ---- group membership (EXT_SERVES equivalents) ------------------------------
set(SAGE_ARCHS_SM80  "")   # cc >= 8.0 (all)          plain
set(SAGE_ARCHS_FUSED "")   # cc >= 8.0 (all)          plain
set(SAGE_ARCHS_SM89  "")   # cc == 8.9 or major >= 10 plain (12.x fallback on purpose)
set(SAGE_ARCHS_SM90  "")   # cc == 9.0                accel (90a only, never PTX)
set(SAGE_ARCHS_SM100 "")   # cc in {10.0, 11.0}       accel (never PTX)
set(SAGE_ARCHS_SM120 "")   # major == 12              plain

# DRAFT (Track D prescreen, do not flip on without sign-off): drop gencode
# entries whose SASS resolve() (plan.cpp:206-266) never reaches when the
# covering group is built from the same arch list:
#   - sm89 group drops major==12 (any requested 12.x also builds the sm120
#     group, which resolve() prefers unconditionally);
#   - sm80 group drops 8.9 when a lower 8.x cubin is requested (same-major
#     binary compatibility serves it) and drops major >= 10 (any requested
#     cc >= 10 joins the sm89 group, so the kSm80F16 fallback is unreachable);
#   - 9.0 always stays in the sm80 group: sm_90a SASS only runs on 9.0
#     exactly, and resolve() sends every 9.x-minor device to kSm80F16.
# OFF because backend_compiled() checks the family flag, not per-arch SASS:
# plan(backend=...) and the raw qattn_sm80_*/qattn_sm89_* ops can still pick a
# family on a device whose SASS was pruned, turning a working launch into
# cudaErrorNoKernelImageForDevice. Pruned entries also lose their +PTX copy.
option(SAGE_PRUNE_GENCODE "Prune per-group gencode entries covered by another group" OFF)

set(_sage_has_sub89_8x FALSE)
foreach(cc IN LISTS SAGE_REQUESTED_ARCHS)
  if(cc MATCHES "^8\\." AND NOT cc STREQUAL "8.9")
    set(_sage_has_sub89_8x TRUE)
  endif()
endforeach()

foreach(cc IN LISTS SAGE_REQUESTED_ARCHS)
  string(REPLACE "." ";" _parts "${cc}")
  list(GET _parts 0 _major)
  list(GET _parts 1 _minor)
  set(_sm80_skip FALSE)
  set(_sm89_skip FALSE)
  if(SAGE_PRUNE_GENCODE)
    if((cc STREQUAL "8.9" AND _sage_has_sub89_8x) OR _major GREATER_EQUAL 10)
      set(_sm80_skip TRUE)
    endif()
    if(_major EQUAL 12)
      set(_sm89_skip TRUE)
    endif()
  endif()
  if(NOT _sm80_skip)
    list(APPEND SAGE_ARCHS_SM80 "${cc}")
  endif()
  list(APPEND SAGE_ARCHS_FUSED "${cc}")
  if((cc STREQUAL "8.9" OR _major GREATER_EQUAL 10) AND NOT _sm89_skip)
    list(APPEND SAGE_ARCHS_SM89 "${cc}")
  endif()
  if(cc STREQUAL "9.0")
    list(APPEND SAGE_ARCHS_SM90 "${cc}")
  endif()
  if(cc STREQUAL "10.0" OR cc STREQUAL "11.0")
    list(APPEND SAGE_ARCHS_SM100 "${cc}")
  endif()
  if(_major EQUAL 12)
    list(APPEND SAGE_ARCHS_SM120 "${cc}")
  endif()
endforeach()

set(SAGE_KIND_SM80  plain)
set(SAGE_KIND_FUSED plain)
set(SAGE_KIND_SM89  plain)
set(SAGE_KIND_SM90  accel)
set(SAGE_KIND_SM100 accel)
set(SAGE_KIND_SM120 plain)

# ---- toolkit gating ----------------------------------------------------------
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
  message(FATAL_ERROR "CUDA 12.0 or higher is required (found ${CMAKE_CUDA_COMPILER_VERSION}).")
endif()
if(SAGE_ARCHS_SM89)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.4)
    message(FATAL_ERROR "CUDA 12.4 or higher is required to build the FP8 (sm89-class) "
                        "kernels for compute capabilities ${SAGE_ARCHS_SM89}.")
  endif()
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
    message(WARNING "CUDA < 12.8: the fp16-accumulator FP8 kernels "
                    "(qk_int8_sv_f8_accum_f16_*) compile to runtime traps on this "
                    "toolkit; upgrade to CUDA 12.8+ for the full sm89 kernel set.")
  endif()
endif()
# sm90 pulls tma.cuh, whose mbarrier/TMA helpers call cuda::ptx::mbarrier_init
# and cuda::ptx::cp_async_bulk_tensor. Those land in CCCL 2.4 = CUDA 12.5;
# CUDA 12.4 has <cuda/ptx> but not those two names. Measured, not inferred.
if(SAGE_ARCHS_SM90 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.5)
  message(FATAL_ERROR "CUDA 12.5 or higher is required for compute capability 9.0 "
                      "(cuda::ptx::mbarrier_init / cp_async_bulk_tensor are CCCL 2.4).")
endif()
# sm100/sm110 pull tcgen05.cuh. Its tcgen05.* wrappers exist from CUDA 12.8,
# but elect_one() calls cuda::ptx::elect_sync, which is CCCL 3.1 = CUDA 13.1.
# nvcc 12.8-13.0 can emit sm_100a SASS, so --list-gpu-code below does NOT catch
# this - the gate has to be explicit.
if(SAGE_ARCHS_SM100 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.1)
  message(FATAL_ERROR "CUDA 13.1 or higher is required to build the tcgen05 kernels for "
                      "compute capabilities ${SAGE_ARCHS_SM100} "
                      "(cuda::ptx::elect_sync is CCCL 3.1).")
endif()

# nvcc --list-gpu-code probe: does this toolkit emit SASS for every requested cc?
execute_process(COMMAND "${CMAKE_CUDA_COMPILER}" --list-gpu-code
                OUTPUT_VARIABLE _sage_gpu_codes OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "\n" ";" _sage_gpu_codes "${_sage_gpu_codes}")
# Toolkit version that first emits each code. These are hints on the
# "nvcc cannot emit sm_XX" error only; the real tcgen05 floor for sm_100/sm_110
# is the CUDA 13.1 gate above, which is stricter.
set(_hint_sm_100 "12.8")
set(_hint_sm_120 "12.8")
set(_hint_sm_103 "12.9")
set(_hint_sm_121 "12.9")
set(_hint_sm_110 "13.0")
foreach(cc IN LISTS SAGE_REQUESTED_ARCHS)
  string(REPLACE "." "" _num "${cc}")
  if(NOT "sm_${_num}" IN_LIST _sage_gpu_codes)
    set(_msg "nvcc ${CMAKE_CUDA_COMPILER_VERSION} cannot emit sm_${_num} (not in nvcc --list-gpu-code).")
    if(DEFINED _hint_sm_${_num})
      string(APPEND _msg " CUDA ${_hint_sm_${_num}} or higher is required.")
    endif()
    message(FATAL_ERROR "${_msg}")
  endif()
endforeach()

# ---- gencode helpers ---------------------------------------------------------
# Single-token --generate-code= form so source-property lists never split.
function(sage_gencode_flags out kind archs)
  set(_f "")
  foreach(cc IN LISTS archs)
    string(REPLACE "." "" n "${cc}")
    if(kind STREQUAL "accel")
      list(APPEND _f "--generate-code=arch=compute_${n}a,code=sm_${n}a")
    else()
      list(APPEND _f "--generate-code=arch=compute_${n},code=sm_${n}")
      if("${cc}" IN_LIST SAGE_PTX_ARCHS)
        list(APPEND _f "--generate-code=arch=compute_${n},code=compute_${n}")
      endif()
    endif()
  endforeach()
  set(${out} "${_f}" PARENT_SCOPE)
endfunction()

# Attach gencode flags to a source group and append it to SAGE_SOURCES.
# NOTE: per-source CUDA_ARCHITECTURES is silently ignored by CMake — only
# COMPILE_OPTIONS works at source granularity. Do not "simplify" this.
function(sage_group_enable name srcs)
  if(NOT SAGE_ARCHS_${name})
    message(STATUS "[sageattention] skip ${name}: no requested compute capability is served by it")
    return()
  endif()
  sage_gencode_flags(_flags "${SAGE_KIND_${name}}" "${SAGE_ARCHS_${name}}")
  set_source_files_properties(${srcs} PROPERTIES COMPILE_OPTIONS "${_flags}")
  set(SAGE_SOURCES ${SAGE_SOURCES} ${srcs} PARENT_SCOPE)
  set(SAGEATTN_BUILD_${name} 1 PARENT_SCOPE)
  message(STATUS "[sageattention] ${name}: ${_flags}")
endfunction()
