# Locate the target Python's torch installation and expose it as INTERFACE
# targets. We deliberately do NOT use find_package(Torch): TorchConfig pulls in
# the full Caffe2/cuDNN/cuSPARSELt discovery and hard-fails on CUDA compiler /
# toolkit version mismatches, while all we need is torch's include dirs, its
# core libraries and the C++ ABI flag — exactly what CUDAExtension used.
#
# Provides:
#   sage::torch          - include dirs + torch libs (never torch_python)
#   sage::python_module  - Python.h include dir only (no libpython link: abi3
#                          extension modules resolve symbols at import time)
#   SAGE_TORCH_CXX11_ABI - 0/1
#   SAGE_TORCH_VERSION   - e.g. 2.13.0+cu132
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
"import json, torch
from torch.utils import cpp_extension as e
print(json.dumps({
    'inc': e.include_paths('cuda'),
    'lib': e.library_paths('cuda'),
    'abi': int(torch._C._GLIBCXX_USE_CXX11_ABI),
    'ver': torch.__version__,
}))"
  OUTPUT_VARIABLE _sage_torch_json
  RESULT_VARIABLE _sage_torch_rc
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT _sage_torch_rc EQUAL 0)
  message(FATAL_ERROR
    "Failed to import torch with ${Python_EXECUTABLE}. Install torch into "
    "this interpreter or build with --no-build-isolation in an environment "
    "that has it.")
endif()

string(JSON _sage_torch_inc_len LENGTH "${_sage_torch_json}" inc)
set(SAGE_TORCH_INCLUDE_DIRS "")
math(EXPR _last "${_sage_torch_inc_len} - 1")
foreach(i RANGE ${_last})
  string(JSON _dir GET "${_sage_torch_json}" inc ${i})
  list(APPEND SAGE_TORCH_INCLUDE_DIRS "${_dir}")
endforeach()

string(JSON _sage_torch_lib_len LENGTH "${_sage_torch_json}" lib)
set(SAGE_TORCH_LIBRARY_DIRS "")
math(EXPR _last "${_sage_torch_lib_len} - 1")
foreach(i RANGE ${_last})
  string(JSON _dir GET "${_sage_torch_json}" lib ${i})
  list(APPEND SAGE_TORCH_LIBRARY_DIRS "${_dir}")
endforeach()

string(JSON SAGE_TORCH_CXX11_ABI GET "${_sage_torch_json}" abi)
string(JSON SAGE_TORCH_VERSION GET "${_sage_torch_json}" ver)

message(STATUS "[sageattention] torch ${SAGE_TORCH_VERSION} "
               "(CXX11_ABI=${SAGE_TORCH_CXX11_ABI}) via ${Python_EXECUTABLE}")

# Same link set as torch.utils.cpp_extension.CUDAExtension, minus torch_python
# (linking it would break the abi3/limited-API contract).
add_library(sage_torch INTERFACE)
add_library(sage::torch ALIAS sage_torch)
target_include_directories(sage_torch INTERFACE ${SAGE_TORCH_INCLUDE_DIRS})
target_link_directories(sage_torch INTERFACE ${SAGE_TORCH_LIBRARY_DIRS})
target_link_libraries(sage_torch INTERFACE
  c10 torch torch_cpu torch_cuda c10_cuda CUDA::cudart)
target_compile_definitions(sage_torch INTERFACE
  _GLIBCXX_USE_CXX11_ABI=${SAGE_TORCH_CXX11_ABI})

add_library(sage_python_module INTERFACE)
add_library(sage::python_module ALIAS sage_python_module)
target_include_directories(sage_python_module INTERFACE ${Python_INCLUDE_DIRS})
