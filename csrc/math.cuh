/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * This file is based on code from Flashinfer,
 * https://github.com/flashinfer-ai/flashinfer/blob/v0.1.5/include/flashinfer/math.cuh Copyright (c) 2023 by FlashInfer
 * team. Small modifications made by SageAttention team, 2024 (e.g., renamed namespace).
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
#include <cuda_runtime.h>

#include <cuda/std/numbers>

namespace math {

// log2(e); the softmax runs in base 2 (see attn_utils.cuh)
constexpr float log2e = cuda::std::numbers::log2e_v<float>;

/*!
 * \brief Wrapper of PTX ex2.approx instruction, which computes 2^x
 *
 * exp2f lowers to the same ex2.approx.ftz.f32 under --use_fast_math (MUFU
 * count unchanged), but losing the asm volatile lets ptxas move it: measured
 * +20 STL / +16 LDL of spill in the sm120 inner loop. Keep the barrier.
 *
 * \param x input
 */
__device__ __forceinline__ float ptx_exp2(float x)
{
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

/*!
 * \brief log2(x). --use_fast_math lowers __log2f to lg2.approx.ftz.f32.
 * \param x input
 */
__device__ __forceinline__ float ptx_log2(float x)
{
    return __log2f(x);
}

/*!
 * \brief Wrapper of PTX rcp.approx instruction, which computes 1/x
 *
 * No intrinsic exists for the approximate reciprocal (__frcp_rn is the
 * correctly-rounded one); cuda_fp16.hpp writes this same asm by hand.
 *
 * \param x input
 */
__device__ __forceinline__ float ptx_rcp(float x)
{
    float y;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

}  // namespace math