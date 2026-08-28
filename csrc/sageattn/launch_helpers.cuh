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

#include <array>
#include <atomic>
#include <cstddef>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace sage {

// Opt into >48KB dynamic smem once per (kernel instantiation, device) instead
// of on every launch. Launches with <=48KB never need the opt-in. Also turns
// the "this fatbin has no cubin for the current device" failure into an
// actionable error instead of a silent no-op (the old code discarded the
// cudaFuncSetAttribute return value).
template<typename KernelT>
inline void set_max_dynamic_smem_once(KernelT kernel, size_t smem_bytes, int device)
{
    if (smem_bytes <= 48 * 1024) {
        return;
    }
    static std::array<std::atomic<size_t>, C10_COMPILE_TIME_MAX_GPUS> done{};
    if (done[device].load(std::memory_order_acquire) == smem_bytes) {
        return;
    }
    cudaError_t err = cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (err == cudaErrorInvalidDeviceFunction || err == cudaErrorNoKernelImageForDevice) {
        (void)cudaGetLastError();  // clear the sticky error before raising
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        TORCH_CHECK(false,
                    "sageattention: this build contains no kernel image for sm_",
                    major,
                    minor,
                    ". Rebuild with TORCH_CUDA_ARCH_LIST=",
                    major,
                    ".",
                    minor);
    }
    C10_CUDA_CHECK(err);
    done[device].store(smem_bytes, std::memory_order_release);
}

}  // namespace sage
