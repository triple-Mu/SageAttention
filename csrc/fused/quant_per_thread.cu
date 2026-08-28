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

#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/types.h>

#include "../dispatch_utils.h"
#include "../gmem_access.cuh"
#include "../numeric_conversion.cuh"
#include "../reduction_utils.cuh"
#include "../sageattn/launch_helpers.cuh"
#include "../utils.cuh"
#include "quant_utils.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// CUDA port of the Triton per-thread quantization kernels
// (sageattention/triton/quant_per_thread.py). The scale layouts match the
// kPerThread expectations of the qk_int8_sv_f8 attention kernels: Q gets one
// scale per mma A-fragment row class (8 per quantization warp), K gets one
// scale per B-fragment column pair class (4 per quantization warp).
//
// Numerics follow the Triton kernels exactly so that outputs are
// bit-identical to the Triton path:
//   scale = amax / 127 + 1e-7; q = x / scale (div.full.f32);
//   round half away from zero; truncate.

namespace {

// Triton lowers fp32 "/" to div.full.f32; nvcc under --use_fast_math does not,
// so spell it explicitly to keep the results bit-identical.
__device__ __forceinline__ float div_full(float a, float b)
{
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ int8_t quantize_round_half_away(float x, float scale)
{
    float q = div_full(x, scale);
    q += (q >= 0.0f) ? 0.5f : -0.5f;
    return static_cast<int8_t>(q);
}

// One warp handles one scale group; lane `l` covers columns [l*VEC, (l+1)*VEC).
template<uint32_t head_dim>
struct PerThreadVec {
    static constexpr uint32_t VEC = head_dim / 32;
    static_assert(VEC == 2 || VEC == 4, "head_dim must be 64 or 128");
};

// The quantize pass re-reads exactly the rows the amax pass read, so holding
// them in registers turns the kernel into a single pass over global memory.
// The re-read is an L1 hit only while a whole SM's worth of warps fits in
// 128 KB, which the K tiles (4 KB per warp) blow past, so this is where the
// traffic actually is: caching them is worth ~1.6x on a 3080 Ti Laptop.
//
// 128 B per thread = 32 registers on top of a ~40 register kernel, which still
// leaves 7 blocks per SM. The 32-row warp_k=128 tiles (sm90/sm100) are held
// out by the row cap: they cost the same 32 registers plus a 2x longer
// unrolled body, and there is no such part here to measure the trade on.
constexpr uint32_t kQuantCacheBytes = 128;
constexpr uint32_t kQuantCacheRows  = 16;

// Query: warp tld of a WARP_BLOCK-token slab owns rows {slab*WARP_BLOCK + r*8 + tld},
// r in [0, WARP_BLOCK/8); one scale over all of them, stored at [slab*8 + tld].
template<uint32_t head_dim, uint32_t warp_block_size, typename T>
__global__ void QuantPerThreadQInt8Kernel(T* __restrict__ input,
                                          int8_t* __restrict__ output,
                                          float* __restrict__ scale,
                                          const uint32_t num_tokens,
                                          const int64_t  stride_bz_input,
                                          const uint32_t stride_seq_input,
                                          const int64_t  stride_h_input,
                                          const int64_t  stride_bz_output,
                                          const uint32_t stride_seq_output,
                                          const int64_t  stride_h_output,
                                          const int64_t  stride_bz_scale,
                                          const int64_t  stride_h_scale)
{
    constexpr uint32_t VEC            = PerThreadVec<head_dim>::VEC;
    using pack_t                      = sage::vec_t<T, VEC>;
    using store_t                     = sage::vec_t<int8_t, VEC>;
    constexpr uint32_t rows_per_group = warp_block_size / 8;
    constexpr bool kCache = rows_per_group <= kQuantCacheRows && rows_per_group * VEC * sizeof(T) <= kQuantCacheBytes;
    // full unroll only pays for the cached form, which needs constant cache
    // indices; unrolling the reload form as well just inflates the live ranges
    constexpr uint32_t kUnroll = kCache ? rows_per_group : 4;

    const uint32_t slab_id         = blockIdx.x;
    const uint32_t head_id         = blockIdx.y;
    const uint32_t batch_id        = blockIdx.z;
    const uint32_t tld             = threadIdx.y;
    const uint32_t lane_id         = threadIdx.x;
    const uint32_t col             = lane_id * VEC;
    const uint32_t slab_base_token = slab_id * warp_block_size;

    // The slab base is the only offset that can reach the n*h*hd range, so it is
    // computed in 64-bit once; the in-slab row offsets stay 32-bit.
    const T* input_base = input + static_cast<int64_t>(batch_id) * stride_bz_input
                          + static_cast<int64_t>(head_id) * stride_h_input
                          + static_cast<int64_t>(slab_base_token) * stride_seq_input + static_cast<int64_t>(col);
    int8_t* output_base = output + static_cast<int64_t>(batch_id) * stride_bz_output
                          + static_cast<int64_t>(head_id) * stride_h_output
                          + static_cast<int64_t>(slab_base_token) * stride_seq_output + static_cast<int64_t>(col);

    pack_t x_cache[kCache ? rows_per_group : 1];

    float amax = 0.0f;
#pragma unroll kUnroll
    for (uint32_t r = 0; r < rows_per_group; r++) {
        const uint32_t local = r * 8 + tld;
        const uint32_t token = slab_base_token + local;
        if (token < num_tokens) {
            pack_t x_val;
            x_val.load_ro(input_base + local * stride_seq_input);
            if constexpr (kCache) {
                x_cache[r] = x_val;
            }
#pragma unroll
            for (uint32_t j = 0; j < VEC; j++) {
                amax = fmaxf(amax, fabsf(fp_traits<T>::to_fp32(x_val[j])));
            }
        }
    }

    amax                    = vllm::warpReduceMax(amax);
    const float group_scale = div_full(amax, 127.0f) + 0.0000001f;

#pragma unroll kUnroll
    for (uint32_t r = 0; r < rows_per_group; r++) {
        const uint32_t local = r * 8 + tld;
        const uint32_t token = slab_base_token + local;
        if (token < num_tokens) {
            pack_t x_val;
            if constexpr (kCache) {
                x_val = x_cache[r];
            }
            else {
                x_val.load_ro(input_base + local * stride_seq_input);
            }
            store_t o_val;
#pragma unroll
            for (uint32_t j = 0; j < VEC; j++) {
                o_val[j] = quantize_round_half_away(fp_traits<T>::to_fp32(x_val[j]), group_scale);
            }
            o_val.store(output_base + local * stride_seq_output);
        }
    }

    if (lane_id == 0) {
        scale[static_cast<int64_t>(batch_id) * stride_bz_scale + static_cast<int64_t>(head_id) * stride_h_scale
              + slab_id * 8 + tld] = group_scale;
    }
}

// Key: warp tld of a WARP_BLOCK-token slab owns row pairs
// {slab*WARP_BLOCK + r*8 + 2*tld, +1}, r in [0, WARP_BLOCK/8); one scale over
// all of them, stored at [slab*4 + tld]. Optionally fuses the k - km
// smoothing subtraction (done in the input dtype, matching the torch sub the
// Triton path performs beforehand).
template<uint32_t head_dim, uint32_t warp_block_size, bool sub_mean, typename T>
__global__ void QuantPerThreadKInt8Kernel(T* __restrict__ input,
                                          T* __restrict__ mean,
                                          int8_t* __restrict__ output,
                                          float* __restrict__ scale,
                                          const uint32_t num_tokens,
                                          const int64_t  stride_bz_input,
                                          const uint32_t stride_seq_input,
                                          const int64_t  stride_h_input,
                                          const int64_t  stride_bz_mean,
                                          const int64_t  stride_h_mean,
                                          const int64_t  stride_bz_output,
                                          const uint32_t stride_seq_output,
                                          const int64_t  stride_h_output,
                                          const int64_t  stride_bz_scale,
                                          const int64_t  stride_h_scale)
{
    constexpr uint32_t VEC             = PerThreadVec<head_dim>::VEC;
    using pack_t                       = sage::vec_t<T, VEC>;
    using store_t                      = sage::vec_t<int8_t, VEC>;
    constexpr uint32_t pairs_per_group = warp_block_size / 8;
    constexpr uint32_t rows_per_thread = pairs_per_group * 2;
    constexpr bool kCache = rows_per_thread <= kQuantCacheRows && rows_per_thread * VEC * sizeof(T) <= kQuantCacheBytes;
    // full unroll only pays for the cached form, which needs constant cache
    // indices; unrolling the reload form as well just inflates the live ranges
    constexpr uint32_t kUnroll = kCache ? pairs_per_group : 4;

    const uint32_t slab_id         = blockIdx.x;
    const uint32_t head_id         = blockIdx.y;
    const uint32_t batch_id        = blockIdx.z;
    const uint32_t tld             = threadIdx.y;
    const uint32_t lane_id         = threadIdx.x;
    const uint32_t col             = lane_id * VEC;
    const uint32_t slab_base_token = slab_id * warp_block_size;

    // The slab base is the only offset that can reach the n*h*hd range, so it is
    // computed in 64-bit once; the in-slab row offsets stay 32-bit.
    const T* input_base = input + static_cast<int64_t>(batch_id) * stride_bz_input
                          + static_cast<int64_t>(head_id) * stride_h_input
                          + static_cast<int64_t>(slab_base_token) * stride_seq_input + static_cast<int64_t>(col);
    int8_t* output_base = output + static_cast<int64_t>(batch_id) * stride_bz_output
                          + static_cast<int64_t>(head_id) * stride_h_output
                          + static_cast<int64_t>(slab_base_token) * stride_seq_output + static_cast<int64_t>(col);

    pack_t mean_val;
    if constexpr (sub_mean) {
        mean_val.load_ro(mean + static_cast<int64_t>(batch_id) * stride_bz_mean
                         + static_cast<int64_t>(head_id) * stride_h_mean + static_cast<int64_t>(col));
    }

    // The smoothing subtraction feeds both amax and the quantization, so the
    // cached rows are the already-subtracted ones and the subtract runs once.
    auto load_row = [&](uint32_t local) {
        pack_t x_val;
        x_val.load_ro(input_base + local * stride_seq_input);
        if constexpr (sub_mean) {
#pragma unroll
            for (uint32_t j = 0; j < VEC; j++) {
                x_val[j] = x_val[j] - mean_val[j];
            }
        }
        return x_val;
    };

    pack_t x_cache[kCache ? rows_per_thread : 1];

    float amax = 0.0f;
#pragma unroll kUnroll
    for (uint32_t r = 0; r < pairs_per_group; r++) {
#pragma unroll
        for (uint32_t p = 0; p < 2; p++) {
            const uint32_t local = r * 8 + tld * 2 + p;
            const uint32_t token = slab_base_token + local;
            if (token < num_tokens) {
                pack_t x_val = load_row(local);
                if constexpr (kCache) {
                    x_cache[r * 2 + p] = x_val;
                }
#pragma unroll
                for (uint32_t j = 0; j < VEC; j++) {
                    amax = fmaxf(amax, fabsf(fp_traits<T>::to_fp32(x_val[j])));
                }
            }
        }
    }

    amax                    = vllm::warpReduceMax(amax);
    const float group_scale = div_full(amax, 127.0f) + 0.0000001f;

#pragma unroll kUnroll
    for (uint32_t r = 0; r < pairs_per_group; r++) {
#pragma unroll
        for (uint32_t p = 0; p < 2; p++) {
            const uint32_t local = r * 8 + tld * 2 + p;
            const uint32_t token = slab_base_token + local;
            if (token < num_tokens) {
                pack_t x_val;
                if constexpr (kCache) {
                    x_val = x_cache[r * 2 + p];
                }
                else {
                    x_val = load_row(local);
                }
                store_t o_val;
#pragma unroll
                for (uint32_t j = 0; j < VEC; j++) {
                    o_val[j] = quantize_round_half_away(fp_traits<T>::to_fp32(x_val[j]), group_scale);
                }
                o_val.store(output_base + local * stride_seq_output);
            }
        }
    }

    if (lane_id == 0) {
        scale[static_cast<int64_t>(batch_id) * stride_bz_scale + static_cast<int64_t>(head_id) * stride_h_scale
              + slab_id * 4 + tld] = group_scale;
    }
}

}  // namespace

void quant_per_thread_int8_q_cuda(torch::Tensor input,
                                  torch::Tensor output,
                                  torch::Tensor scale,
                                  int           block_size,
                                  int           warp_block_size,
                                  int           tensor_layout)
{
    const c10::cuda::CUDAGuard device_guard(input.device());

    QuantLayout l = check_quant_layout(input, output, scale, tensor_layout);
    TORCH_CHECK(warp_block_size % 8 == 0 && block_size % warp_block_size == 0,
                "block_size must be a multiple of warp_block_size, warp_block_size a multiple of 8");

    const int num_slabs =
        static_cast<int>(at::ceil_div<int64_t>(l.num_tokens, block_size)) * (block_size / warp_block_size);
    CHECK_SHAPE(scale, l.batch_size, l.num_heads, num_slabs * 8);

    auto input_dtype = input.scalar_type();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
        DISPATCH_HEAD_DIM(l.head_dim, HEAD_DIM, {
            DISPATCH_WARP_BLOCK_SIZE(warp_block_size, WARP_BLOCK_SIZE, {
                dim3 grid(num_slabs, l.num_heads, l.batch_size);
                dim3 block(32, 8);
                QuantPerThreadQInt8Kernel<HEAD_DIM, WARP_BLOCK_SIZE, c_type>
                    <<<grid, block, 0, stream>>>(reinterpret_cast<c_type*>(input.data_ptr()),
                                                 output.data_ptr<int8_t>(),
                                                 reinterpret_cast<float*>(scale.data_ptr()),
                                                 l.num_tokens,
                                                 l.stride_bz_input,
                                                 l.stride_seq_input,
                                                 l.stride_h_input,
                                                 l.stride_bz_output,
                                                 l.stride_seq_output,
                                                 l.stride_h_output,
                                                 scale.stride(0),
                                                 scale.stride(1));
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
}

void quant_per_thread_int8_k_cuda(torch::Tensor input,
                                  torch::Tensor output,
                                  torch::Tensor scale,
                                  int           block_size,
                                  int           warp_block_size,
                                  int           tensor_layout)
{
    const c10::cuda::CUDAGuard device_guard(input.device());

    QuantLayout l = check_quant_layout(input, output, scale, tensor_layout);
    TORCH_CHECK(warp_block_size % 8 == 0 && block_size % warp_block_size == 0,
                "block_size must be a multiple of warp_block_size, warp_block_size a multiple of 8");

    const int num_slabs =
        static_cast<int>(at::ceil_div<int64_t>(l.num_tokens, block_size)) * (block_size / warp_block_size);
    CHECK_SHAPE(scale, l.batch_size, l.num_heads, num_slabs * 4);

    auto input_dtype = input.scalar_type();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
        DISPATCH_HEAD_DIM(l.head_dim, HEAD_DIM, {
            DISPATCH_WARP_BLOCK_SIZE_K(warp_block_size, WARP_BLOCK_SIZE, {
                dim3 grid(num_slabs, l.num_heads, l.batch_size);
                dim3 block(32, 4);
                QuantPerThreadKInt8Kernel<HEAD_DIM, WARP_BLOCK_SIZE, false, c_type>
                    <<<grid, block, 0, stream>>>(reinterpret_cast<c_type*>(input.data_ptr()),
                                                 nullptr,
                                                 output.data_ptr<int8_t>(),
                                                 reinterpret_cast<float*>(scale.data_ptr()),
                                                 l.num_tokens,
                                                 l.stride_bz_input,
                                                 l.stride_seq_input,
                                                 l.stride_h_input,
                                                 0,
                                                 0,
                                                 l.stride_bz_output,
                                                 l.stride_seq_output,
                                                 l.stride_h_output,
                                                 scale.stride(0),
                                                 scale.stride(1));
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
}

void quant_per_thread_int8_k_fuse_sub_mean_cuda(torch::Tensor input,
                                                torch::Tensor mean,
                                                torch::Tensor output,
                                                torch::Tensor scale,
                                                int           block_size,
                                                int           warp_block_size,
                                                int           tensor_layout)
{
    const c10::cuda::CUDAGuard device_guard(input.device());

    QuantLayout l = check_quant_layout(input, output, scale, tensor_layout);
    TORCH_CHECK(warp_block_size % 8 == 0 && block_size % warp_block_size == 0,
                "block_size must be a multiple of warp_block_size, warp_block_size a multiple of 8");

    CHECK_CUDA(mean);
    CHECK_LASTDIM_CONTIGUOUS(mean);
    CHECK_DIMS(mean, 3);
    CHECK_SHAPE(mean, l.batch_size, l.num_heads, l.head_dim);
    TORCH_CHECK(input.scalar_type() == mean.scalar_type(), "Input and mean must have the same data type");

    const int num_slabs =
        static_cast<int>(at::ceil_div<int64_t>(l.num_tokens, block_size)) * (block_size / warp_block_size);
    CHECK_SHAPE(scale, l.batch_size, l.num_heads, num_slabs * 4);

    auto input_dtype = input.scalar_type();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
        DISPATCH_HEAD_DIM(l.head_dim, HEAD_DIM, {
            DISPATCH_WARP_BLOCK_SIZE_K(warp_block_size, WARP_BLOCK_SIZE, {
                dim3 grid(num_slabs, l.num_heads, l.batch_size);
                dim3 block(32, 4);
                QuantPerThreadKInt8Kernel<HEAD_DIM, WARP_BLOCK_SIZE, true, c_type>
                    <<<grid, block, 0, stream>>>(reinterpret_cast<c_type*>(input.data_ptr()),
                                                 reinterpret_cast<c_type*>(mean.data_ptr()),
                                                 output.data_ptr<int8_t>(),
                                                 reinterpret_cast<float*>(scale.data_ptr()),
                                                 l.num_tokens,
                                                 l.stride_bz_input,
                                                 l.stride_seq_input,
                                                 l.stride_h_input,
                                                 mean.stride(0),
                                                 mean.stride(1),
                                                 l.stride_bz_output,
                                                 l.stride_seq_output,
                                                 l.stride_h_output,
                                                 scale.stride(0),
                                                 scale.stride(1));
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
}
