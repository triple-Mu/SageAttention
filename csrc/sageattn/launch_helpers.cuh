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

// Opt into the launch's dynamic smem size once per (kernel instantiation,
// device) instead of on every launch. Also turns the "this fatbin has no cubin
// for the current device" failure into an actionable error instead of a silent
// no-op (the old code discarded the cudaFuncSetAttribute return value).
//
// Two things this must NOT do, both learned on B200 (2026-08-29):
//
//  - skip the call when smem_bytes <= 48 KB. The 48 KB default budget covers
//    static __shared__ too, so a kernel asking for exactly 49152 dynamic bytes
//    plus a few mbarriers is already over it and fails to launch. sm100 asks
//    for exactly that at head_dim=128.
//  - memoize on smem_bytes. KernelT is only the function-pointer *type*, which
//    every HEAD_DIM/gran/causal/lse instantiation of a family shares, so one
//    static memo serves all of them: a size-keyed cache opts in whichever
//    kernel launches first and silently skips the rest. The kernel address is
//    the right key (smem_bytes is a compile-time constant of the instantiation).
template<typename KernelT>
inline void set_max_dynamic_smem_once(KernelT kernel, size_t smem_bytes, int device)
{
    static std::array<std::atomic<const void*>, C10_COMPILE_TIME_MAX_GPUS> done{};
    const void*                                                           fn = reinterpret_cast<const void*>(kernel);
    if (done[device].load(std::memory_order_acquire) == fn) {
        return;
    }
    cudaError_t err = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
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
    done[device].store(fn, std::memory_order_release);
}

}  // namespace sage
