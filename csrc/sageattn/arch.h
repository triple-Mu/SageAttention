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

#pragma once

#include <ATen/cuda/CUDAContext.h>

#include "plan.h"

namespace sage {

// Compute capability of a device. at::cuda::getDeviceProperties caches the
// cudaDeviceProp per device process-wide, so this is a hashmap lookup, not a
// driver call. (The SAGEATTN_SM100_TCGEN05 env gate lives in plan.cpp.)
inline CC device_cc(c10::DeviceIndex device)
{
    const cudaDeviceProp* prop = at::cuda::getDeviceProperties(device);
    return CC{prop->major, prop->minor};
}

}  // namespace sage
