/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// The only translation unit that includes <Python.h>. Importing
// sageattention._C loads the shared object, which runs the TORCH_LIBRARY
// static initializers in ops.cpp and registers every torch.ops.sageattention
// operator. The module itself is an empty shell (extension-cpp pattern) plus
// a few build-info string constants readable without initializing CUDA.
// Py_LIMITED_API is defined project-wide, so only limited-API calls are legal
// here (no pybind11 anywhere in this extension).

#include <Python.h>

#include "sageattn_build_config.h"

extern "C" {

static struct PyModuleDef sageattn_module_def = {
    PyModuleDef_HEAD_INIT,
    "_C",  // module name
    "SageAttention CUDA extension; operators live in torch.ops.sageattention",
    -1,
    nullptr,  // no methods
};

PyMODINIT_FUNC PyInit__C(void)
{
    PyObject* mod = PyModule_Create(&sageattn_module_def);
    if (mod == nullptr) {
        return nullptr;
    }
    // Build metadata (host-only, safe before CUDA init).
    if (PyModule_AddStringConstant(mod, "built_archs", SAGEATTN_BUILT_ARCHS_STR) != 0
        || PyModule_AddStringConstant(mod, "version", SAGEATTN_VERSION) != 0
        || PyModule_AddStringConstant(mod, "cuda_version", SAGEATTN_CUDA_VERSION) != 0
        || PyModule_AddStringConstant(mod, "torch_version", SAGEATTN_TORCH_VERSION) != 0) {
        Py_DECREF(mod);
        return nullptr;
    }
    return mod;
}

}  // extern "C"
