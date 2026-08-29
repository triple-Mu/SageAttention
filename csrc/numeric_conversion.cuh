/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Inspired by CUTLASS, https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/numeric_conversion.h
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
#include <cuda/pipeline>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120400)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890))
#define FP8_CAST_ENABLED
#endif
#endif

#if defined(__CUDA_ARCH__)
#define RUNTIME_ASSERT(x) __brkpt()
#else
#include <assert.h>
#define RUNTIME_ASSERT(x) assert(0 && x)
#endif

__device__ __forceinline__ void unpack_half2_from_uint32_to_float(float* dst, uint32_t src)
{
    const half2 h2 = *reinterpret_cast<const half2*>(&src);
    dst[0]         = __low2float(h2);
    dst[1]         = __high2float(h2);
}

// Hand-written rather than two __nv_cvt_float2_to_fp8x2 calls: the official
// API returns two 16-bit halves, and OR-ing them into one u32 costs a PRMT
// per pack (measured: +64 PRMT and new local spills in the sm100/sm120 inner
// loops). The inline mov.b32 %0, {lo, hi} below is free. cuda_fp8.h would
// also fall back to a software emulation below sm_89, which FP8_CAST_ENABLED
// deliberately does not.
__device__ __forceinline__ void floatx4_to_e4m3x4(uint32_t* dst, float* src0, float* src1)
{
#ifdef FP8_CAST_ENABLED
    asm volatile("{\n"
                 ".reg .b16 lo;\n"
                 ".reg .b16 hi;\n"
                 "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n"
                 "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n"
                 "mov.b32 %0, {lo, hi};\n"
                 "}"
                 : "=r"(dst[0])
                 : "f"(src0[0]), "f"(src0[1]), "f"(src1[0]), "f"(src1[1]));
#else
    RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
#endif
}

// Hand-written: cvt.rni.sat.s8.f32 (round-to-nearest-even *and* saturate in
// one instruction) has no intrinsic; __float2int_rn + clamp is two more.
__device__ __forceinline__ int8_t float_to_int8_rn(float x)
{
    uint32_t dst;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
    return reinterpret_cast<const int8_t&>(dst);
}
// ---------------------------------------------------------------------------
// fp_traits<T>::to_fp32: scalar fp16/bf16 -> fp32 conversion. Single home for
// the dtype converters fused.cu and quant_per_thread.cu used to each carry
// their own copy of; adapted from the user's CTI common.cuh.
// ---------------------------------------------------------------------------

template<typename T>
struct fp_traits;

template<>
struct fp_traits<half> {
    static __device__ __forceinline__ float to_fp32(half v)
    {
        return __half2float(v);
    }
};

template<>
struct fp_traits<__nv_bfloat16> {
    static __device__ __forceinline__ float to_fp32(__nv_bfloat16 v)
    {
        return __bfloat162float(v);
    }
};
