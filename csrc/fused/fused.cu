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

#include "../cp_async.cuh"
#include "../dispatch_utils.h"
#include "../gmem_access.cuh"
#include "../numeric_conversion.cuh"
#include "../reduction_utils.cuh"
#include "../sageattn/launch_helpers.cuh"
#include "../utils.cuh"
#include "quant_utils.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// sm90/sm100 schedule 2048 threads per SM, so the 1024-thread quant blocks
// only co-schedule two CTAs up to 32 registers; the varlen/int64 address math
// pushed the sub_mean instance to 38-40 registers, halving occupancy (1.8x on
// B200 quant_k). Pin those blocks back to two CTAs: the baseline ran the same
// loop body in 32 registers, so the cap only re-serializes the one-off pointer
// setup. The other archs top out at 1536 threads per SM, where a 1024-thread
// block can never co-schedule anyway - they keep an unannotated kernel.
// The (1024, 1) arm also lowers the register cap of the sub-1024-thread
// instances to 64; that is acceptable collateral because fill_tiles pins the
// 128-token tiles on these two archs, so those instances have no launch path
// here. The other archs get no annotation at all and keep byte-identical SASS.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000)
#define SAGE_QUANT_BOUNDS(threads) __launch_bounds__(1024, (threads) == 1024 ? 2 : 1)
// Under the 32-register pin the fp32 mean image is 8 registers of spill;
// re-converting per use is a handful of free HADD2s instead. The other archs
// keep the precomputed image (and their exact SASS): with registers to spare,
// letting them re-convert is a wash at best.
#define SAGE_QUANT_MEAN_REMAT 1
#else
#define SAGE_QUANT_BOUNDS(threads)
#define SAGE_QUANT_MEAN_REMAT 0
#endif

// kVarlen splits the dense and packed-layout instances: the dense one folds
// seq_base/scale_slot and drops the cu_seqlens sentinel (see the
// QuantPerThreadQInt8Kernel comment in quant_per_thread.cu); the varlen one
// keeps the runtime null test so its instruction stream stays the pre-split one.
template<uint32_t head_dim,
         uint32_t CTA_TOKENS,
         uint32_t num_pack_per_thread = 1,
         bool     has_sm_scale        = false,
         bool     sub_mean            = false,
         bool     kVarlen             = false,
         typename T>
__global__ void SAGE_QUANT_BOUNDS(CTA_TOKENS*(head_dim / 8) / num_pack_per_thread) QuantInt8Kernel(T* __restrict__ input,
                                T* __restrict__ mean,
                                int8_t* __restrict__ output,
                                float* __restrict__ scale,
                                float          sm_scale,
                                const uint32_t num_tokens,
                                const int64_t  stride_batch_input,
                                const uint32_t stride_seq_input,
                                const int64_t  stride_h_input,
                                const int64_t  stride_batch_mean,
                                const int64_t  stride_h_mean,
                                const int64_t  stride_batch_output,
                                const uint32_t stride_seq_output,
                                const int64_t  stride_h_output,
                                const int64_t  stride_batch_scale,
                                const int64_t  stride_h_scale,
                                const int32_t* __restrict__ cu_seqlens,
                                const uint32_t scale_blk_tokens)
{
    static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value,
                  "Only half and bfloat16 are supported");
    static_assert(num_pack_per_thread > 0, "The number of pack per thread must be greater than 0");

    constexpr uint32_t pack_size             = 8;  // float4 contains 8 half or 8 bfloat16
    constexpr uint32_t num_threads_per_token = head_dim / pack_size;

    static_assert(num_threads_per_token <= 32,
                  "The number of threads per token must be less than or equal to warp size");

    T     x_val[num_pack_per_thread][8];
    T     mean_val[8];
    float x_val_float[num_pack_per_thread][8];
    float mean_val_float[8];

    uint32_t token_cta_idx = blockIdx.x;
    uint32_t head_id       = blockIdx.y;
    uint32_t batch_id      = blockIdx.z;
    uint32_t thread_id     = threadIdx.x;

    // Packed varlen layout (cu_seqlens != nullptr): blockIdx.z selects a
    // sequence of the prefix sum instead of a batch entry. Only two things
    // move - the token base, which enters the pointers, and the scale slot -
    // while every predicate below stays in sequence-relative tokens. That is
    // what makes an equal-length batch quantize bit-identically either way.
    // The grid is opened to max_seqlen, so a block past this sequence's own
    // scale blocks leaves here: block-uniform, and before any __syncthreads.
    uint32_t seq_tokens = num_tokens;
    int64_t  seq_base   = 0;
    uint32_t scale_slot = token_cta_idx;
    if constexpr (kVarlen) {
        if (cu_seqlens != nullptr) {
            seq_tokens                  = static_cast<uint32_t>(sage::seq_len(cu_seqlens, batch_id));
            const uint32_t ctas_per_blk = scale_blk_tokens / CTA_TOKENS;
            if (token_cta_idx >= ((seq_tokens + scale_blk_tokens - 1) / scale_blk_tokens) * ctas_per_blk) {
                return;
            }
            seq_base   = sage::seq_offset(cu_seqlens, batch_id);
            scale_slot = static_cast<uint32_t>(sage::blk_offset(cu_seqlens, batch_id, scale_blk_tokens)) * ctas_per_blk
                         + token_cta_idx;
        }
    }

    uint32_t thread_base_token = token_cta_idx * CTA_TOKENS + thread_id / num_threads_per_token;
    T*       input_ptr_base    = input + static_cast<int64_t>(batch_id) * stride_batch_input
                        + static_cast<int64_t>(head_id) * stride_h_input
                        + (seq_base + static_cast<int64_t>(thread_base_token)) * stride_seq_input
                        + static_cast<int64_t>(thread_id % num_threads_per_token * pack_size);
    T* mean_ptr_base = mean + static_cast<int64_t>(batch_id) * stride_batch_mean
                       + static_cast<int64_t>(head_id) * stride_h_mean
                       + static_cast<int64_t>(thread_id % num_threads_per_token * pack_size);
    int8_t* output_ptr_base = output + static_cast<int64_t>(batch_id) * stride_batch_output
                              + static_cast<int64_t>(head_id) * stride_h_output
                              + (seq_base + static_cast<int64_t>(thread_base_token)) * stride_seq_output
                              + static_cast<int64_t>(thread_id % num_threads_per_token * pack_size);
    float* scale_ptr_base = scale + static_cast<int64_t>(batch_id) * stride_batch_scale
                            + static_cast<int64_t>(head_id) * stride_h_scale + static_cast<int64_t>(scale_slot);

    if constexpr (sub_mean) {
        *(float4*)(&mean_val[0]) = *(float4*)(mean_ptr_base);
#if !SAGE_QUANT_MEAN_REMAT
#pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
            mean_val_float[j] = fp_traits<T>::to_fp32(mean_val[j]);
        }
#endif
    }

    constexpr uint32_t iter_stride = CTA_TOKENS / num_pack_per_thread;

// load the data
#pragma unroll
    for (uint32_t i = 0; i < num_pack_per_thread; i++) {
        if (thread_base_token + i * iter_stride < seq_tokens) {
            *(float4*)(&x_val[i][0]) = *(float4*)(input_ptr_base + i * iter_stride * stride_seq_input);
#pragma unroll
            for (uint32_t j = 0; j < 8; j++) {
                x_val_float[i][j] = fp_traits<T>::to_fp32(x_val[i][j]);
            }

            if constexpr (sub_mean) {
#pragma unroll
                for (uint32_t j = 0; j < 8; j++) {
#if SAGE_QUANT_MEAN_REMAT
                    x_val_float[i][j] -= fp_traits<T>::to_fp32(mean_val[j]);
#else
                    x_val_float[i][j] -= mean_val_float[j];
#endif
                }
            }

            if constexpr (has_sm_scale) {
#pragma unroll
                for (uint32_t j = 0; j < 8; j++) {
                    x_val_float[i][j] *= sm_scale;
                }
            }
        }
        else {
#pragma unroll
            for (uint32_t j = 0; j < 8; j++) {
                x_val_float[i][j] = 0.0f;
            }
        }
    }

    float amax = 0.0000001f;  // prevent from dividing by zero

#pragma unroll
    for (uint32_t i = 0; i < num_pack_per_thread; i++) {
#pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
            amax = fmaxf(amax, fabsf(x_val_float[i][j]));
        }
    }

    __shared__ float smem_amax;
    const float      block_amax_val = vllm::blockReduceMax(amax);
    if (thread_id == 0) {
        smem_amax         = block_amax_val;
        scale_ptr_base[0] = smem_amax / 127.0f;
    }

    __syncthreads();

    float recip_scale = 127.0f / smem_amax;

    char4 o_val[num_pack_per_thread][2];

#pragma unroll
    for (uint32_t i = 0; i < num_pack_per_thread; i++) {
#pragma unroll
        for (uint32_t j = 0; j < 2; j += 1) {
            o_val[i][j] = make_char4(float_to_int8_rn(x_val_float[i][j * 4 + 0] * recip_scale),
                                     float_to_int8_rn(x_val_float[i][j * 4 + 1] * recip_scale),
                                     float_to_int8_rn(x_val_float[i][j * 4 + 2] * recip_scale),
                                     float_to_int8_rn(x_val_float[i][j * 4 + 3] * recip_scale));
        }
    }

    // int8 result
#pragma unroll
    for (uint32_t i = 0; i < num_pack_per_thread; i++) {

        if (thread_base_token + i * iter_stride < seq_tokens) {
            *reinterpret_cast<float2*>(output_ptr_base + i * iter_stride * stride_seq_output) =
                *reinterpret_cast<float2*>(&o_val[i][0]);
        }
    }
}

template<uint32_t head_dim, uint32_t CTA_TOKENS, uint32_t num_pack_per_thread = 1, typename T>
__global__ void SubMeanKernel(T* __restrict__ input,
                              T* __restrict__ mean,
                              half* __restrict__ output,
                              const uint32_t num_tokens,
                              const int64_t  stride_batch_input,
                              const uint32_t stride_seq_input,
                              const int64_t  stride_h_input,
                              const int64_t  stride_batch_mean,
                              const int64_t  stride_h_mean,
                              const int64_t  stride_batch_output,
                              const uint32_t stride_seq_output,
                              const int64_t  stride_h_output)
{
    static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value,
                  "Only half and bfloat16 are supported");
    static_assert(num_pack_per_thread > 0, "The number of pack per thread must be greater than 0");

    using T2 = typename std::conditional<std::is_same<T, half>::value, half2, nv_bfloat162>::type;

    constexpr uint32_t pack_size             = 8;  // float4 contains 8 half or 8 bfloat16
    constexpr uint32_t num_threads_per_token = head_dim / pack_size;

    static_assert(num_threads_per_token <= 32,
                  "The number of threads per token must be less than or equal to warp size");

    T2 x_val[num_pack_per_thread][4];
    T2 mean_val[4];

    uint32_t token_cta_idx = blockIdx.x;
    uint32_t head_id       = blockIdx.y;
    uint32_t batch_id      = blockIdx.z;
    uint32_t thread_id     = threadIdx.x;

    uint32_t thread_base_token = token_cta_idx * CTA_TOKENS + thread_id / num_threads_per_token;
    T*       input_ptr_base    = input + static_cast<int64_t>(batch_id) * stride_batch_input
                        + static_cast<int64_t>(head_id) * stride_h_input
                        + static_cast<int64_t>(thread_base_token) * stride_seq_input
                        + static_cast<int64_t>(thread_id % num_threads_per_token * pack_size);
    T* mean_ptr_base = mean + static_cast<int64_t>(batch_id) * stride_batch_mean
                       + static_cast<int64_t>(head_id) * stride_h_mean
                       + static_cast<int64_t>(thread_id % num_threads_per_token * pack_size);
    half* output_ptr_base = output + static_cast<int64_t>(batch_id) * stride_batch_output
                            + static_cast<int64_t>(head_id) * stride_h_output
                            + static_cast<int64_t>(thread_base_token) * stride_seq_output
                            + static_cast<int64_t>(thread_id % num_threads_per_token * pack_size);

    *(float4*)(&mean_val[0]) = *(float4*)(mean_ptr_base);

    constexpr uint32_t iter_stride = CTA_TOKENS / num_pack_per_thread;

// load the data
#pragma unroll
    for (uint32_t i = 0; i < num_pack_per_thread; i++) {
        if (thread_base_token + i * iter_stride < num_tokens) {
            *(float4*)(&x_val[i][0]) = *(float4*)(input_ptr_base + i * iter_stride * stride_seq_input);
#pragma unroll
            for (uint32_t j = 0; j < 4; j++) {
                x_val[i][j] = __hsub2(x_val[i][j], mean_val[j]);

                if constexpr (std::is_same<T, nv_bfloat16>::value) {
                    ((half2*)x_val[i])[j] = __float22half2_rn(__bfloat1622float2(x_val[i][j]));
                }
            }
        }
    }

#pragma unroll
    for (uint32_t i = 0; i < num_pack_per_thread; i++) {
        if (thread_base_token + i * iter_stride < num_tokens) {
            *reinterpret_cast<float4*>(output_ptr_base + i * iter_stride * stride_seq_output) =
                *reinterpret_cast<float4*>(&x_val[i][0]);
        }
    }
}

// kVarlen: same dense/varlen instance split as QuantInt8Kernel above.
template<uint32_t head_dim, uint32_t CTA_TOKENS, bool pad_zero = false, bool permute = true, bool kVarlen = false,
         typename T>
__global__ void TransposePadPermuteKernel(T* __restrict__ input,
                                          T* __restrict__ output,
                                          const uint32_t num_tokens,
                                          const int64_t  stride_batch_input,
                                          const uint32_t stride_seq_input,
                                          const int64_t  stride_h_input,
                                          const int64_t  stride_batch_output,
                                          const uint32_t stride_d_output,
                                          const int64_t  stride_h_output,
                                          const int32_t* __restrict__ cu_seqlens,
                                          const uint32_t pad_tokens)
{

    static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value,
                  "Only half and bfloat16 are supported");

    constexpr uint32_t pack_size             = 8;  // float4 contains 8 half or 8 bfloat16
    uint32_t           num_threads_per_token = head_dim / pack_size;
    uint32_t           num_threads_per_cta   = CTA_TOKENS / pack_size;

    uint32_t token_cta_idx = blockIdx.x;
    uint32_t head_id       = blockIdx.y;
    uint32_t batch_id      = blockIdx.z;
    uint32_t thread_id     = threadIdx.x;

    // Packed varlen layout (cu_seqlens != nullptr): blockIdx.z selects a
    // sequence of the prefix sum instead of a batch entry, and this kernel is
    // where the two coordinate systems of varlen.h meet. The input moves by
    // the sequence's token base, the output by its padded-slab base; the
    // predicate below stays in sequence-relative tokens, which is what makes
    // an equal-length batch transpose bit-identically either way. The grid is
    // opened to max_seqlen, so a block past this sequence leaves here:
    // block-uniform, and before any __syncthreads.
    uint32_t seq_tokens = num_tokens;
    int64_t  seq_base   = 0;
    int64_t  pad_base   = 0;
    if constexpr (kVarlen) {
        if (cu_seqlens != nullptr) {
            seq_tokens = static_cast<uint32_t>(sage::seq_len(cu_seqlens, batch_id));
            if (token_cta_idx * CTA_TOKENS >= seq_tokens) {
                return;
            }
            seq_base = sage::seq_offset(cu_seqlens, batch_id);
            pad_base = sage::pad_offset(cu_seqlens, batch_id, pad_tokens);
        }
    }

    uint32_t thread_base_token = token_cta_idx * CTA_TOKENS + thread_id / num_threads_per_token;

    T* input_ptr_base = input + static_cast<int64_t>(batch_id) * stride_batch_input
                        + static_cast<int64_t>(head_id) * stride_h_input
                        + (seq_base + static_cast<int64_t>(thread_base_token)) * stride_seq_input
                        + static_cast<int64_t>(thread_id % num_threads_per_token * pack_size);
    T* output_ptr_base = output + static_cast<int64_t>(batch_id) * stride_batch_output
                         + static_cast<int64_t>(head_id) * stride_h_output + pad_base
                         + static_cast<int64_t>(token_cta_idx * CTA_TOKENS)
                         + static_cast<int64_t>(thread_id % num_threads_per_cta * pack_size)
                         + static_cast<int64_t>(thread_id / num_threads_per_cta) * stride_d_output;

    // +8 halves of row padding: the transpose loop below reads
    // smem_load_tile[row][col] with col warp-uniform, so an unpadded row stride of
    // head_dim*2 bytes (a multiple of 128) lands all 32 lanes in one bank
    // (32-way conflict, 8 serialized trips per tile). The pad must stay a
    // multiple of 8 halves (16B) to keep the cp.async row starts aligned, which
    // caps the improvement at 4-way instead of conflict-free.
    __shared__ T smem_load_tile[CTA_TOKENS][head_dim + 8];
    __shared__ T smem_store_tile[head_dim][CTA_TOKENS];

    // permute == true: reorder the seq dimension within 16-token groups as
    // 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15 to match the
    // register A-fragment k-order of the fp8 mma.sync / wgmma kernels
    // (sm89/sm90/sm12x). permute == false: keep linear seq order — required by
    // the sm100 tcgen05 kernel, whose P operand is packed in linear k-order.
    uint32_t smem_load_row;
    if constexpr (permute) {
        uint32_t smem_load_row_base = ((thread_id / num_threads_per_token) / 16) * 16;
        uint32_t smem_load_row_mod  = (thread_id / num_threads_per_token) % 16;
        smem_load_row = smem_load_row_base + (smem_load_row_mod / 8) * 2 + ((smem_load_row_mod / 2) % 4) * 4
                        + (smem_load_row_mod % 2);
    }
    else {
        smem_load_row = thread_id / num_threads_per_token;
    }

    constexpr cp_async::SharedMemFillMode fill_mode =
        pad_zero ? cp_async::SharedMemFillMode::kFillZero : cp_async::SharedMemFillMode::kNoFill;
    cp_async::pred_load_128b<cp_async::PrefetchMode::kNoPrefetch, fill_mode>(
        smem_load_tile[smem_load_row] + thread_id % num_threads_per_token * pack_size,
        input_ptr_base,
        thread_base_token < seq_tokens);
    cp_async::commit_group();
    cp_async::wait_group<0>();
    __syncthreads();

    uint32_t smem_row_base   = thread_id % CTA_TOKENS;
    uint32_t smem_col_base   = thread_id / CTA_TOKENS;
    uint32_t smem_col_stride = head_dim / 8;

    // TODO: use ldmatrix to do permutation
#pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        smem_store_tile[smem_col_base + i * smem_col_stride][smem_row_base] =
            smem_load_tile[smem_row_base][smem_col_base + i * smem_col_stride];
    }

    __syncthreads();

    *(float4*)(output_ptr_base) =
        *(float4*)(&smem_store_tile[thread_id / num_threads_per_cta][thread_id % num_threads_per_cta * pack_size]);
}

// No kVarlen split here: the varlen prologue sits outside the token loops
// that dominate this kernel (it only launches on the two-pass V path, i.e.
// padded_tokens > 4096), so a dense instance saves ~6-26 one-off instructions
// per thread while doubling the largest instance bodies of the fused TU.
template<uint32_t pad_size, bool sub_mean = false, typename T>
__global__ void MeanScaleKernel(T* __restrict__ input,
                                int8_t* __restrict__ output,
                                float* __restrict__ mean,
                                float* __restrict__ scale,
                                const float    scale_max,
                                const uint32_t num_tokens,
                                const int64_t  stride_batch_input,
                                const int64_t  stride_d_input,
                                const int64_t  stride_h_input,
                                const int64_t  stride_batch_output,
                                const int64_t  stride_d_output,
                                const int64_t  stride_h_output,
                                const int64_t  stride_batch_mean,
                                const int64_t  stride_h_mean,
                                const int64_t  stride_batch_scale,
                                const int64_t  stride_h_scale,
                                const int32_t* __restrict__ cu_seqlens,
                                const uint32_t pad_tokens)
{
    static_assert(std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value,
                  "Only half and bfloat16 are supported");

    constexpr uint32_t pack_size = 8;  // float4 contains 8 half or 8 bfloat16
    using pack_t                 = sage::vec_t<T, pack_size>;

    // Both passes walk the same addresses (same base, same gmem_stride), so when
    // their per-thread iteration counts agree the second global read is pure
    // duplicated traffic and the tile can be kept in registers instead. The
    // counts only disagree for the handful of threads that straddle the
    // ceil16 / ceil64 padding boundary, where the quantize pass touches pad
    // tokens the statistics pass never read; those threads take the reload path.
    // kCache packs cost kCache * 16B of registers per thread.
    constexpr uint32_t kCache = 4;

    uint32_t head_id   = blockIdx.x;
    uint32_t batch_id  = blockIdx.y;
    uint32_t d_id      = blockIdx.z;
    uint32_t thread_id = threadIdx.x;

    uint32_t num_threads = blockDim.x;
    uint32_t gmem_stride = num_threads * pack_size;

    // Packed varlen layout (cu_seqlens != nullptr): the transposed tensors
    // lose their batch dimension and blockIdx.y selects a sequence of the
    // prefix sum; its slab starts at varlen.h's pad_offset, the same
    // expression the transpose kernel wrote it with. num_tokens becomes the
    // sequence's own length, and the ceil16 statistics divisor below is taken
    // over that - an equal-length batch then quantizes bit-identically.
    uint32_t seq_tokens = num_tokens;
    int64_t  pad_base   = 0;
    if (cu_seqlens != nullptr) {
        seq_tokens = static_cast<uint32_t>(sage::seq_len(cu_seqlens, batch_id));
        // An empty sequence has no statistic to reduce and no fp8 token to
        // write. Leaving here (block-uniform, before the block reductions)
        // also keeps a 0/0 mean out of the output; the scale and mean entries
        // stay the zeros the allocation put there.
        if (seq_tokens == 0) {
            return;
        }
        pad_base = sage::pad_offset(cu_seqlens, batch_id, pad_tokens);
    }

    // pad the number of tokens to 16 to deal with fp8 permute in previous kernel
    uint32_t stat_padded_num_tokens = at::round_up<uint32_t>(seq_tokens, 16);
    uint32_t num_iters =
        stat_padded_num_tokens / gmem_stride + ((stat_padded_num_tokens % gmem_stride) > thread_id * pack_size);
    // the quantize pass covers all fp8 output tokens to prevent nan in random initialization
    uint32_t padded_num_tokens = at::round_up<uint32_t>(seq_tokens, pad_size);
    uint32_t num_quant_iters =
        padded_num_tokens / gmem_stride + ((padded_num_tokens % gmem_stride) > thread_id * pack_size);

    const bool cached = (num_iters == num_quant_iters) && (num_iters <= kCache);

    T* input_ptr_base = input + static_cast<int64_t>(batch_id) * stride_batch_input
                        + static_cast<int64_t>(head_id) * stride_h_input + static_cast<int64_t>(d_id) * stride_d_input
                        + pad_base + static_cast<int64_t>(thread_id * pack_size);
    int8_t* output_ptr_base =
        output + static_cast<int64_t>(batch_id) * stride_batch_output + static_cast<int64_t>(head_id) * stride_h_output
        + static_cast<int64_t>(d_id) * stride_d_output + pad_base + static_cast<int64_t>(thread_id * pack_size);

    pack_t x_cache[kCache];

    float max_val = -1000000.0f;
    float min_val = 1000000.0f;
    float sum_val = 0.0f;

    auto accumulate = [&](const pack_t& x_val) {
#pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
            float x_temp = fp_traits<T>::to_fp32(x_val[j]);
            max_val      = fmaxf(max_val, x_temp);
            min_val      = fminf(min_val, x_temp);

            if constexpr (sub_mean) {
                sum_val += x_temp;
            }
        }
    };

    if (cached) {
        // constant trip count so x_cache stays in registers
#pragma unroll
        for (uint32_t i = 0; i < kCache; i++) {
            if (i < num_iters) {
                x_cache[i].load_ro(input_ptr_base + i * gmem_stride);
                accumulate(x_cache[i]);
            }
        }
    }
    else {
        for (uint32_t i = 0; i < num_iters; i++) {
            pack_t x_val;
            x_val.load_ro(input_ptr_base + i * gmem_stride);
            accumulate(x_val);
        }
    }

    // reduce
    __shared__ float smem_amax;
    __shared__ float smem_mean;

    float block_max_val = vllm::blockReduceMax(max_val);
    float block_min_val = vllm::blockReduceMin(min_val);
    float block_sum_val;

    if constexpr (sub_mean) {
        block_sum_val = vllm::blockReduceSum(sum_val);
    }

    if (thread_id == 0) {
        // smem_mean is only defined (and only consumed) on the sub_mean path;
        // computing it unconditionally used to read the uninitialized
        // block_sum_val (UB, harmless in practice but wrong).
        if constexpr (sub_mean) {
            smem_mean = block_sum_val / stat_padded_num_tokens;
            smem_amax = fmaxf(fabsf(block_max_val - smem_mean), fabsf(block_min_val - smem_mean));
            mean[static_cast<int64_t>(batch_id) * stride_batch_mean + static_cast<int64_t>(head_id) * stride_h_mean
                 + static_cast<int64_t>(d_id)] = smem_mean;
        }
        else {
            smem_amax = fmaxf(fabsf(block_max_val), fabsf(block_min_val));
        }

        scale[static_cast<int64_t>(batch_id) * stride_batch_scale + static_cast<int64_t>(head_id) * stride_h_scale
              + static_cast<int64_t>(d_id)] = smem_amax / scale_max;
    }

    __syncthreads();

    float mean_val = 0.0f;
    if constexpr (sub_mean) {
        mean_val = smem_mean;
    }
    // amax == 0 (an all-zero channel, or a constant channel under sub_mean)
    // makes this division inf, and quantizing 0 * inf = NaN emits fp8 0x7f,
    // which poisons the whole output channel through the PV mma. A subnormal
    // amax (bf16 input) reaches inf too: ftz flushes it to zero, or without
    // ftz the division overflows. The stored scale is (near) zero in both
    // cases, so quantizing the channel to (signed) zero is exact.
    float recip_scale = scale_max / smem_amax;
    if (!isfinite(recip_scale)) {
        recip_scale = 0.0f;
    }

    auto quantize_store = [&](const pack_t& x_val, uint32_t i) {
        float    x_val_float[8];
        uint32_t x_val_fp8[2];
#pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
            x_val_float[j] = fp_traits<T>::to_fp32(x_val[j]);
            if constexpr (sub_mean) {
                x_val_float[j] = (x_val_float[j] - mean_val) * recip_scale;
            }
            else {
                x_val_float[j] *= recip_scale;
            }
        }

        floatx4_to_e4m3x4(x_val_fp8, x_val_float, x_val_float + 2);
        floatx4_to_e4m3x4(x_val_fp8 + 1, x_val_float + 4, x_val_float + 6);

        *(uint2*)(output_ptr_base + i * gmem_stride) = *(uint2*)(&x_val_fp8[0]);
    };

    if (cached) {
#pragma unroll
        for (uint32_t i = 0; i < kCache; i++) {
            if (i < num_quant_iters) {
                quantize_store(x_cache[i], i);
            }
        }
    }
    else {
        for (uint32_t i = 0; i < num_quant_iters; i++) {
            pack_t x_val;
            x_val.load_ro(input_ptr_base + i * gmem_stride);
            quantize_store(x_val, i);
        }
    }
}

// Channel chunk one CTA of the fused kernel below owns. 16 halves is 32 bytes:
// a whole DRAM sector of every token row it touches, so the gathered read
// wastes no bandwidth, while 3 * 16 float accumulators still leave the CTA two
// resident blocks' worth of registers.
constexpr int kVQuantChannelChunk = 16;

// Block size of the fused kernel. Not a tuning knob: it is MeanScaleKernel's,
// and with the 8-token pack it decides which transposed tokens each reduction
// leaf holds, so changing it changes the fp8 output.
constexpr int kVQuantThreads = 256;

// One-kernel form of TransposePadPermuteKernel + MeanScaleKernel: the same fp8
// output without the fp16 V^T buffer in between, so V's bytes are read twice
// instead of being read, written transposed and read back (a write and a read
// of the whole value tensor, ~28% of what the pair moves).
//
// The thread mapping is what makes the result bit-identical rather than merely
// equal. The block keeps MeanScaleKernel's shape - 256 threads, thread t owning
// the transposed token packs [i * 2048 + t * 8, +8) of its channel - so every
// reduction leaf and every step of the blockReduce tree is the one the two-pass
// version took, statistics still run over the ceil16 axis and the quantize pass
// still over the ceil64 one. Only where the eight values come from changes:
// instead of reading them contiguously out of V^T, thread t gathers them from
// V's own layout through src_token below. What the block gives up is the coalesced
// row - it covers chunk_size channels rather than one, so a gather is a 32-byte
// piece of a token row (a whole DRAM sector, no waste) instead of a 128-byte
// line.
// No kVarlen split here: the varlen-only block is 13 instructions of a
// 2216-3272 instruction body (0.6%), all outside the token loops, so a dense
// instance buys nothing measurable and doubles the biggest bodies of the TU.
template<uint32_t chunk_size, uint32_t pad_size, bool sub_mean = false, bool permute = true, typename T>
__global__ void TransposeQuantFp8Kernel(const T* __restrict__ input,
                                        int8_t* __restrict__ output,
                                        float* __restrict__ mean,
                                        float* __restrict__ scale,
                                        const float    scale_max,
                                        const uint32_t num_tokens,
                                        const int64_t  stride_batch_input,
                                        const uint32_t stride_seq_input,
                                        const int64_t  stride_h_input,
                                        const int64_t  stride_batch_output,
                                        const int64_t  stride_d_output,
                                        const int64_t  stride_h_output,
                                        const int64_t  stride_batch_mean,
                                        const int64_t  stride_h_mean,
                                        const int64_t  stride_batch_scale,
                                        const int64_t  stride_h_scale,
                                        const int32_t* __restrict__ cu_seqlens,
                                        const uint32_t pad_tokens)
{
    static_assert(std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value,
                  "Only half and bfloat16 are supported");

    constexpr uint32_t pack_size = 8;  // transposed tokens per thread per round
    constexpr uint32_t load_size = 8;  // one 16B LDG of T
    static_assert(chunk_size % load_size == 0, "the channel chunk must be whole LDGs wide");
    constexpr uint32_t num_loads = chunk_size / load_size;
    using load_t                 = sage::vec_t<T, load_size>;

    // Compile-time, not blockDim.x: gmem_stride is what a reduction leaf's
    // token pack is derived from, so it belongs to the bit-exactness contract.
    constexpr uint32_t num_threads = kVQuantThreads;
    constexpr uint32_t gmem_stride = num_threads * pack_size;
    constexpr uint32_t num_warps   = num_threads / 32;

    // The channel chunk is grid.x, not grid.z as in MeanScaleKernel: the
    // head_dim / chunk_size blocks of one (batch, head) each take a
    // chunk_size-wide slice of the *same* token rows, so they have to be
    // co-resident for the slices to add back up to whole lines in L2. Putting
    // them next to each other in launch order is what makes that happen.
    const uint32_t d_base    = blockIdx.x * chunk_size;
    const uint32_t head_id   = blockIdx.y;
    const uint32_t batch_id  = blockIdx.z;
    const uint32_t thread_id = threadIdx.x;
    const uint32_t lane_id   = thread_id & 31;
    const uint32_t warp_id   = thread_id >> 5;

    // Packed varlen layout (cu_seqlens != nullptr): both coordinate systems of
    // varlen.h meet here, as they did across the two kernels this replaces. The
    // input moves by the sequence's token base, the output by its padded-slab
    // base, and every predicate stays in sequence-relative tokens. An empty
    // sequence leaves (block-uniform, before any barrier), exactly as
    // MeanScaleKernel did: its scale and mean entries stay the zeros the
    // allocation put there and its slab stays zero fp8.
    uint32_t seq_tokens = num_tokens;
    int64_t  seq_base   = 0;
    int64_t  pad_base   = 0;
    if (cu_seqlens != nullptr) {
        seq_tokens = static_cast<uint32_t>(sage::seq_len(cu_seqlens, batch_id));
        if (seq_tokens == 0) {
            return;
        }
        seq_base = sage::seq_offset(cu_seqlens, batch_id);
        pad_base = sage::pad_offset(cu_seqlens, batch_id, pad_tokens);
    }

    // pad the number of tokens to 16 to deal with fp8 permute
    const uint32_t stat_padded_num_tokens = at::round_up<uint32_t>(seq_tokens, 16);
    const uint32_t num_iters =
        stat_padded_num_tokens / gmem_stride + ((stat_padded_num_tokens % gmem_stride) > thread_id * pack_size);
    // the quantize pass covers all fp8 output tokens to prevent nan in random initialization
    const uint32_t padded_num_tokens = at::round_up<uint32_t>(seq_tokens, pad_size);
    const uint32_t num_quant_iters =
        padded_num_tokens / gmem_stride + ((padded_num_tokens % gmem_stride) > thread_id * pack_size);

    const T* input_ptr_base = input + static_cast<int64_t>(batch_id) * stride_batch_input
                              + static_cast<int64_t>(head_id) * stride_h_input + seq_base * stride_seq_input
                              + static_cast<int64_t>(d_base);
    int8_t* output_ptr_base =
        output + static_cast<int64_t>(batch_id) * stride_batch_output + static_cast<int64_t>(head_id) * stride_h_output
        + static_cast<int64_t>(d_base) * stride_d_output + pad_base + static_cast<int64_t>(thread_id * pack_size);

    // Source token of transposed position (pack_base + j), for a pack that
    // starts 8-aligned. TransposePadPermuteKernel stores input row r at output
    // row perm[r] = {0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15} within a 16-token
    // group; this is the inverse of that, and only its first half is tabulated
    // because inv[8 + j] == inv[j] + 4.
    auto src_token = [](uint32_t pack_base, uint32_t j) -> uint32_t {
        if constexpr (permute) {
            constexpr uint32_t inv_permute[8] = {0, 1, 8, 9, 2, 3, 10, 11};
            return pack_base + inv_permute[j] - 4u * ((pack_base >> 3) & 1u);
        }
        else {
            return pack_base + j;
        }
    };

    float max_val[chunk_size];
    float min_val[chunk_size];
    float sum_val[chunk_size];
#pragma unroll
    for (uint32_t c = 0; c < chunk_size; c++) {
        max_val[c] = -1000000.0f;
        min_val[c] = 1000000.0f;
        sum_val[c] = 0.0f;
    }

    for (uint32_t i = 0; i < num_iters; i++) {
        const uint32_t pack_base = i * gmem_stride + thread_id * pack_size;
#pragma unroll
        for (uint32_t j = 0; j < pack_size; j++) {
            const uint32_t tok   = src_token(pack_base, j);
            const bool     valid = tok < seq_tokens;
            load_t         x_val[num_loads];
#pragma unroll
            for (uint32_t l = 0; l < num_loads; l++) {
                if (valid) {
                    x_val[l].load_ro(input_ptr_base + static_cast<int64_t>(tok) * stride_seq_input + l * load_size);
                }
                else {
#pragma unroll
                    for (uint32_t e = 0; e < load_size; e++) {
                        x_val[l][e] = fp_traits<T>::from_fp32(0.0f);
                    }
                }
            }
#pragma unroll
            for (uint32_t c = 0; c < chunk_size; c++) {
                const float x_temp = fp_traits<T>::to_fp32(x_val[c / load_size][c % load_size]);
                max_val[c]         = fmaxf(max_val[c], x_temp);
                min_val[c]         = fminf(min_val[c], x_temp);

                if constexpr (sub_mean) {
                    sum_val[c] += x_temp;
                }
            }
        }
    }

    // The reduction blockReduceMax/Min/Sum would do, unrolled over the chunk so
    // that the chunk_size channels share one barrier instead of taking one
    // each. Same operations in the same order, so the same bits.
    constexpr uint32_t sum_warps = sub_mean ? num_warps : 1;
    __shared__ float   smem_max[chunk_size][num_warps];
    __shared__ float   smem_min[chunk_size][num_warps];
    __shared__ float   smem_sum[chunk_size][sum_warps];
    __shared__ float   smem_amax[chunk_size];
    __shared__ float   smem_mean[chunk_size];

#pragma unroll
    for (uint32_t c = 0; c < chunk_size; c++) {
        const float v = vllm::warpReduceMax(max_val[c]);
        if (lane_id == 0) {
            smem_max[c][warp_id] = v;
        }
    }
#pragma unroll
    for (uint32_t c = 0; c < chunk_size; c++) {
        const float v = vllm::warpReduceMin(min_val[c]);
        if (lane_id == 0) {
            smem_min[c][warp_id] = v;
        }
    }
    if constexpr (sub_mean) {
#pragma unroll
        for (uint32_t c = 0; c < chunk_size; c++) {
            const float v = vllm::warpReduceSum(sum_val[c]);
            if (lane_id == 0) {
                smem_sum[c][warp_id] = v;
            }
        }
    }

    __syncthreads();

    if (warp_id == 0) {
#pragma unroll
        for (uint32_t c = 0; c < chunk_size; c++) {
            const bool live = lane_id < num_warps;

            float partial_max = -1e20f;
            float partial_min = 1e20f;
            float partial_sum = 0.0f;
            if (live) {
                partial_max = smem_max[c][lane_id];
                partial_min = smem_min[c][lane_id];
                if constexpr (sub_mean) {
                    partial_sum = smem_sum[c][lane_id];
                }
            }

            const float block_max_val = vllm::warpReduceMax(partial_max);
            const float block_min_val = vllm::warpReduceMin(partial_min);
            float       block_sum_val = 0.0f;
            if constexpr (sub_mean) {
                block_sum_val = vllm::warpReduceSum(partial_sum);
            }

            if (lane_id == 0) {
                float amax;
                if constexpr (sub_mean) {
                    const float m = block_sum_val / stat_padded_num_tokens;
                    amax          = fmaxf(fabsf(block_max_val - m), fabsf(block_min_val - m));
                    smem_mean[c]  = m;
                    mean[static_cast<int64_t>(batch_id) * stride_batch_mean
                         + static_cast<int64_t>(head_id) * stride_h_mean + static_cast<int64_t>(d_base + c)] = m;
                }
                else {
                    amax = fmaxf(fabsf(block_max_val), fabsf(block_min_val));
                }
                smem_amax[c] = amax;
                scale[static_cast<int64_t>(batch_id) * stride_batch_scale
                      + static_cast<int64_t>(head_id) * stride_h_scale + static_cast<int64_t>(d_base + c)] =
                    amax / scale_max;
            }
        }
    }

    __syncthreads();

    for (uint32_t i = 0; i < num_quant_iters; i++) {
        const uint32_t pack_base = i * gmem_stride + thread_id * pack_size;
#pragma unroll
        for (uint32_t l = 0; l < num_loads; l++) {
            load_t x_val[pack_size];
#pragma unroll
            for (uint32_t j = 0; j < pack_size; j++) {
                const uint32_t tok = src_token(pack_base, j);
                if (tok < seq_tokens) {
                    x_val[j].load_ro(input_ptr_base + static_cast<int64_t>(tok) * stride_seq_input + l * load_size);
                }
                else {
#pragma unroll
                    for (uint32_t e = 0; e < load_size; e++) {
                        x_val[j][e] = fp_traits<T>::from_fp32(0.0f);
                    }
                }
            }
#pragma unroll
            for (uint32_t e = 0; e < load_size; e++) {
                const uint32_t c = l * load_size + e;

                float mean_val = 0.0f;
                if constexpr (sub_mean) {
                    mean_val = smem_mean[c];
                }
                // amax == 0 (an all-zero channel, or a constant channel under
                // sub_mean) makes this division inf, and quantizing 0 * inf =
                // NaN emits fp8 0x7f, which poisons the whole output channel
                // through the PV mma. A subnormal amax (bf16 input) reaches inf
                // too: ftz flushes it to zero, or without ftz the division
                // overflows. The stored scale is (near) zero in both cases, so
                // quantizing the channel to (signed) zero is exact.
                float recip_scale = scale_max / smem_amax[c];
                if (!isfinite(recip_scale)) {
                    recip_scale = 0.0f;
                }

                float    x_val_float[8];
                uint32_t x_val_fp8[2];
#pragma unroll
                for (uint32_t j = 0; j < pack_size; j++) {
                    x_val_float[j] = fp_traits<T>::to_fp32(x_val[j][e]);
                    if constexpr (sub_mean) {
                        x_val_float[j] = (x_val_float[j] - mean_val) * recip_scale;
                    }
                    else {
                        x_val_float[j] *= recip_scale;
                    }
                }

                floatx4_to_e4m3x4(x_val_fp8, x_val_float, x_val_float + 2);
                floatx4_to_e4m3x4(x_val_fp8 + 1, x_val_float + 4, x_val_float + 6);

                *(uint2*)(output_ptr_base + static_cast<int64_t>(c) * stride_d_output + i * gmem_stride) =
                    *(uint2*)(&x_val_fp8[0]);
            }
        }
    }
}

void quant_per_block_int8_cuda(torch::Tensor        input,
                               torch::Tensor        output,
                               torch::Tensor        scale,
                               int                  block_size,
                               int                  tensor_layout,
                               const QuantVarlen&   varlen,
                               std::optional<float> sm_scale)
{
    const c10::cuda::CUDAGuard device_guard(input.device());

    const bool  is_varlen = varlen.cu_seqlens != nullptr;
    QuantLayout layout    = is_varlen ? parse_quant_varlen_layout(input, output, scale, varlen) :
                                        parse_quant_layout(input, output, scale, tensor_layout);
    SAGEATTN_QUANT_LAYOUT_LOCALS(layout);
    SAGEATTN_QUANT_SCALE_STRIDES(scale, is_varlen);

    auto input_dtype = input.scalar_type();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

// The checks, launch geometry and argument list the two sm_scale flavours
// share; only the has_sm_scale template argument and the sm_scale value
// differ (the kernel ignores sm_scale when has_sm_scale is false).
#define SAGEATTN_QUANT_PER_BLOCK_BODY(HAS_SM_SCALE, SM_SCALE)                                                          \
    if (is_varlen) {                                                                                                   \
        CHECK_SHAPE(scale, num_heads, sage::blk_total(input.size(0), batch_size, CTA_TOKENS));                         \
    }                                                                                                                  \
    else {                                                                                                             \
        CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));                               \
        CHECK_SHAPE(scale, batch_size, num_heads, at::ceil_div<int64_t>(num_tokens, CTA_TOKENS));                      \
    }                                                                                                                  \
    dim3 grid(at::ceil_div<int64_t>(num_tokens, CTA_TOKENS), num_heads, batch_size);                                   \
    constexpr int num_pack_per_thread = (CTA_TOKENS * (HEAD_DIM / 8) + 1023) / 1024;                                   \
    dim3 block(CTA_TOKENS * (HEAD_DIM / 8) / num_pack_per_thread);                                                     \
    auto* kernel = is_varlen ? QuantInt8Kernel<HEAD_DIM, CTA_TOKENS, num_pack_per_thread, HAS_SM_SCALE, false, true,   \
                                               c_type> :                                                               \
                               QuantInt8Kernel<HEAD_DIM, CTA_TOKENS, num_pack_per_thread, HAS_SM_SCALE, false, false,  \
                                               c_type>;                                                                \
    kernel<<<grid, block, 0, stream>>>(reinterpret_cast<c_type*>(input.data_ptr()),                                    \
                                     nullptr,                                                                          \
                                     output.data_ptr<int8_t>(),                                                        \
                                     reinterpret_cast<float*>(scale.data_ptr()),                                       \
                                     SM_SCALE,                                                                         \
                                     num_tokens,                                                                       \
                                     stride_batch_input,                                                               \
                                     stride_seq_input,                                                                 \
                                     stride_h_input,                                                                   \
                                     0,                                                                                \
                                     0,                                                                                \
                                     stride_batch_output,                                                              \
                                     stride_seq_output,                                                                \
                                     stride_h_output,                                                                  \
                                     stride_batch_scale,                                                               \
                                     stride_h_scale,                                                                   \
                                     varlen.cu_seqlens,                                                                \
                                     block_size);                                                                      \
    C10_CUDA_KERNEL_LAUNCH_CHECK()

    // Two dispatch nests rather than one branch inside the innermost body:
    // kernels land in the cubin in (reverse) first-instantiation order, so
    // this keeps the pre-merge order - every has_sm_scale=true instance
    // before every false one - and with it byte-identical SASS.
    if (sm_scale.has_value()) {
        DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
            DISPATCH_BLOCK_SIZE(block_size, CTA_TOKENS, {
                DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, { SAGEATTN_QUANT_PER_BLOCK_BODY(true, *sm_scale); });
            });
        });
    }
    else {
        DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
            DISPATCH_BLOCK_SIZE(block_size, CTA_TOKENS, {
                DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, { SAGEATTN_QUANT_PER_BLOCK_BODY(false, 0.0f); });
            });
        });
    }
#undef SAGEATTN_QUANT_PER_BLOCK_BODY
}

void quant_per_block_int8_fuse_sub_mean_cuda(torch::Tensor      input,
                                             torch::Tensor      mean,
                                             torch::Tensor      output,
                                             torch::Tensor      scale,
                                             int                block_size,
                                             int                tensor_layout,
                                             const QuantVarlen& varlen)
{
    const c10::cuda::CUDAGuard device_guard(input.device());

    const bool  is_varlen = varlen.cu_seqlens != nullptr;
    QuantLayout layout    = is_varlen ? parse_quant_varlen_layout(input, output, scale, varlen) :
                                        parse_quant_layout(input, output, scale, tensor_layout, &mean);
    SAGEATTN_QUANT_LAYOUT_LOCALS(layout);
    SAGEATTN_QUANT_SCALE_STRIDES(scale, is_varlen);

    if (is_varlen) {  // parse_quant_layout checks these on the dense path
        CHECK_CUDA(mean);
        CHECK_CONTIGUOUS(mean);
        CHECK_DIMS(mean, 3);
    }

    auto input_dtype = input.scalar_type();
    auto mean_dtype  = mean.scalar_type();

    TORCH_CHECK(input_dtype == mean_dtype, "Input and mean must have the same data type");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
        DISPATCH_BLOCK_SIZE(block_size, CTA_TOKENS, {
            DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
                // the per-sequence mean keeps the dense [batch, heads, head_dim] shape
                CHECK_SHAPE(mean, batch_size, num_heads, head_dim);
                if (is_varlen) {
                    CHECK_SHAPE(scale, num_heads, sage::blk_total(input.size(0), batch_size, CTA_TOKENS));
                }
                else {
                    CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
                    CHECK_SHAPE(scale, batch_size, num_heads, at::ceil_div<int64_t>(num_tokens, CTA_TOKENS));
                }

                dim3 grid(at::ceil_div<int64_t>(num_tokens, CTA_TOKENS), num_heads, batch_size);

                constexpr int num_pack_per_thread = (CTA_TOKENS * (HEAD_DIM / 8) + 1023) / 1024;

                dim3 block(CTA_TOKENS * (HEAD_DIM / 8) / num_pack_per_thread);

                auto* kernel =
                    is_varlen ? QuantInt8Kernel<HEAD_DIM, CTA_TOKENS, num_pack_per_thread, false, true, true, c_type> :
                                QuantInt8Kernel<HEAD_DIM, CTA_TOKENS, num_pack_per_thread, false, true, false, c_type>;
                kernel<<<grid, block, 0, stream>>>(reinterpret_cast<c_type*>(input.data_ptr()),
                                                 reinterpret_cast<c_type*>(mean.data_ptr()),
                                                 output.data_ptr<int8_t>(),
                                                 reinterpret_cast<float*>(scale.data_ptr()),
                                                 0.0f,
                                                 num_tokens,
                                                 stride_batch_input,
                                                 stride_seq_input,
                                                 stride_h_input,
                                                 mean.stride(0),
                                                 mean.stride(1),
                                                 stride_batch_output,
                                                 stride_seq_output,
                                                 stride_h_output,
                                                 stride_batch_scale,
                                                 stride_h_scale,
                                                 varlen.cu_seqlens,
                                                 block_size);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
}

// use block size 128 and warp_block size 32
void quant_per_warp_int8_cuda(torch::Tensor      input,
                              torch::Tensor      output,
                              torch::Tensor      scale,
                              int                block_size,
                              int                warp_block_size,
                              int                tensor_layout,
                              const QuantVarlen& varlen)
{
    const c10::cuda::CUDAGuard device_guard(input.device());

    const bool  is_varlen = varlen.cu_seqlens != nullptr;
    QuantLayout layout    = is_varlen ? parse_quant_varlen_layout(input, output, scale, varlen) :
                                        parse_quant_layout(input, output, scale, tensor_layout);
    SAGEATTN_QUANT_LAYOUT_LOCALS(layout);
    SAGEATTN_QUANT_SCALE_STRIDES(scale, is_varlen);

    auto input_dtype = input.scalar_type();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
        DISPATCH_BLOCK_SIZE(block_size, CTA_TOKENS, {
            DISPATCH_WARP_BLOCK_SIZE(warp_block_size, WARP_TOKENS, {
                DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
                    if (is_varlen) {
                        CHECK_SHAPE(scale,
                                    num_heads,
                                    sage::blk_total(input.size(0), batch_size, CTA_TOKENS)
                                        * (CTA_TOKENS / WARP_TOKENS));
                    }
                    else {
                        CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
                        CHECK_SHAPE(scale,
                                    batch_size,
                                    num_heads,
                                    at::ceil_div<int64_t>(num_tokens, CTA_TOKENS) * (CTA_TOKENS / WARP_TOKENS));
                    }

                    dim3 grid(at::ceil_div<int64_t>(num_tokens, CTA_TOKENS) * (CTA_TOKENS / WARP_TOKENS),
                              num_heads,
                              batch_size);

                    constexpr int num_pack_per_thread = (WARP_TOKENS * (HEAD_DIM / 8) + 1023) / 1024;

                    dim3 block(WARP_TOKENS * (HEAD_DIM / 8) / num_pack_per_thread);

                    auto* kernel =
                        is_varlen ?
                            QuantInt8Kernel<HEAD_DIM, WARP_TOKENS, num_pack_per_thread, false, false, true, c_type> :
                            QuantInt8Kernel<HEAD_DIM, WARP_TOKENS, num_pack_per_thread, false, false, false, c_type>;
                    kernel<<<grid, block, 0, stream>>>(reinterpret_cast<c_type*>(input.data_ptr()),
                                                     nullptr,
                                                     output.data_ptr<int8_t>(),
                                                     reinterpret_cast<float*>(scale.data_ptr()),
                                                     0.0f,
                                                     num_tokens,
                                                     stride_batch_input,
                                                     stride_seq_input,
                                                     stride_h_input,
                                                     0,
                                                     0,
                                                     stride_batch_output,
                                                     stride_seq_output,
                                                     stride_h_output,
                                                     stride_batch_scale,
                                                     stride_h_scale,
                                                     varlen.cu_seqlens,
                                                     block_size);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    });
}

void sub_mean_cuda(torch::Tensor input, torch::Tensor mean, torch::Tensor output, int tensor_layout)
{
    const c10::cuda::CUDAGuard device_guard(input.device());

    CHECK_CUDA(input);
    CHECK_CUDA(mean);
    CHECK_CUDA(output);

    CHECK_LASTDIM_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(mean);
    CHECK_CONTIGUOUS(output);

    CHECK_DIMS(input, 4);
    CHECK_DIMS(mean, 3);
    CHECK_DIMS(output, 4);

    CHECK_DTYPE(output, torch::kHalf);

    const int64_t batch_size = input.size(0);
    const int64_t head_dim   = input.size(3);

    int64_t stride_batch_input  = input.stride(0);
    int64_t stride_batch_output = output.stride(0);

    int64_t num_tokens, num_heads;
    int64_t stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;

    if (tensor_layout == 0) {
        num_tokens        = input.size(1);
        num_heads         = input.size(2);
        stride_seq_input  = input.stride(1);
        stride_h_input    = input.stride(2);
        stride_seq_output = output.stride(1);
        stride_h_output   = output.stride(2);
    }
    else {
        num_tokens        = input.size(2);
        num_heads         = input.size(1);
        stride_seq_input  = input.stride(2);
        stride_h_input    = input.stride(1);
        stride_seq_output = output.stride(2);
        stride_h_output   = output.stride(1);
    }

    auto input_dtype = input.scalar_type();
    auto mean_dtype  = mean.scalar_type();

    TORCH_CHECK(input_dtype == mean_dtype, "Input and mean must have the same data type");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
        DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
            CHECK_SHAPE(mean, batch_size, num_heads, head_dim);
            CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));

            constexpr int CTA_TOKENS = (HEAD_DIM == 128) ? 64 : 128;

            dim3 grid(at::ceil_div<int64_t>(num_tokens, CTA_TOKENS), num_heads, batch_size);

            constexpr int num_pack_per_thread = (CTA_TOKENS * (HEAD_DIM / 8) + 1023) / 1024;

            dim3 block(CTA_TOKENS * (HEAD_DIM / 8) / num_pack_per_thread);

            SubMeanKernel<HEAD_DIM, CTA_TOKENS, num_pack_per_thread>
                <<<grid, block, 0, stream>>>(reinterpret_cast<c_type*>(input.data_ptr()),
                                             reinterpret_cast<c_type*>(mean.data_ptr()),
                                             reinterpret_cast<half*>(output.data_ptr()),
                                             num_tokens,
                                             stride_batch_input,
                                             stride_seq_input,
                                             stride_h_input,
                                             mean.stride(0),
                                             mean.stride(1),
                                             stride_batch_output,
                                             stride_seq_output,
                                             stride_h_output);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

// Like CHECK_SHAPE, but the seq (last) dim only has to be *at least* the
// CTA-aligned token count, and CTA-aligned itself. quant_v_fp8(pad_multiple=128)
// allocates its output at the 128-aligned length while this kernel writes the
// 64-aligned prefix; the caller owns the [seq, output.size(3)) tail.
#define CHECK_SHAPE_PADDED_SEQ(x, d0, d1, d2, seq)                                                                     \
    TORCH_CHECK(x.size(0) == (d0) && x.size(1) == (d1) && x.size(2) == (d2) && x.size(3) >= (seq)                      \
                    && x.size(3) % CTA_TOKENS == 0,                                                                    \
                "Tensor " #x " must have shape (" #d0 ", " #d1 ", " #d2 ", n) "                                        \
                "with n a multiple of ",                                                                               \
                CTA_TOKENS,                                                                                            \
                " and n >= " #seq " (",                                                                                \
                seq,                                                                                                   \
                "), got (",                                                                                            \
                x.size(0),                                                                                             \
                ", ",                                                                                                  \
                x.size(1),                                                                                             \
                ", ",                                                                                                  \
                x.size(2),                                                                                             \
                ", ",                                                                                                  \
                x.size(3),                                                                                             \
                ")")

static void transpose_pad_impl(
    torch::Tensor input, torch::Tensor output, int tensor_layout, bool permute, const QuantVarlen& varlen)
{
    const c10::cuda::CUDAGuard device_guard(input.device());

    CHECK_CUDA(input);
    CHECK_CUDA(output);

    CHECK_LASTDIM_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(output);

    constexpr int CTA_TOKENS = 64;

    int64_t batch_size, head_dim, num_heads, num_tokens, grid_blocks;
    int64_t stride_batch_input, stride_seq_input, stride_h_input;
    int64_t stride_batch_output, stride_d_output, stride_h_output;

    if (varlen.cu_seqlens != nullptr) {
        // packed [total_tokens, heads, head_dim] -> [heads, head_dim, padded_total]
        CHECK_DIMS(input, 3);

        const VTLayout out_layout = parse_vt_varlen_layout(output, varlen);

        batch_size          = varlen.batch_size;
        num_heads           = input.size(1);
        head_dim            = input.size(2);
        num_tokens          = varlen.max_seqlen;  // sizes the grid; the kernel reads its own length
        stride_batch_input  = 0;
        stride_seq_input    = input.stride(0);
        stride_h_input      = input.stride(1);
        stride_batch_output = 0;
        stride_d_output     = out_layout.stride_d;
        stride_h_output     = out_layout.stride_h;

        TORCH_CHECK(out_layout.num_heads == num_heads && out_layout.head_dim == head_dim,
                    "transposed value must be (heads, head_dim, padded_total) = (",
                    num_heads,
                    ", ",
                    head_dim,
                    ", n), got (",
                    output.size(0),
                    ", ",
                    output.size(1),
                    ", ",
                    output.size(2),
                    ")");
        const int64_t padded_total = sage::blk_total(input.size(0), batch_size, varlen.pad_tokens) * varlen.pad_tokens;
        TORCH_CHECK(out_layout.padded_num_tokens == padded_total,
                    "transposed value last dim (",
                    out_layout.padded_num_tokens,
                    ") must be blk_total(total_tokens, batch_size, ",
                    varlen.pad_tokens,
                    ") * ",
                    varlen.pad_tokens,
                    " (",
                    padded_total,
                    ")");
        TORCH_CHECK(varlen.pad_tokens % CTA_TOKENS == 0,
                    "pad_multiple (",
                    varlen.pad_tokens,
                    ") must be a multiple of ",
                    CTA_TOKENS);
        // Each sequence owns ceil(len / pad) padded blocks at most (varlen.h,
        // Property 1); the CTAs below cover ceil(len / CTA_TOKENS) of them, so
        // for pad_multiple > CTA_TOKENS the slab's tail stays unwritten. The
        // fp8 quantization pass stops at the same bound and the fp8 output is
        // allocated zeroed, exactly as the dense pad_multiple=128 path.
        grid_blocks = at::ceil_div<int64_t>(num_tokens, CTA_TOKENS);
    }
    else {
        CHECK_DIMS(input, 4);
        CHECK_DIMS(output, 4);

        batch_size = input.size(0);
        head_dim   = input.size(3);

        stride_batch_input = input.stride(0);

        // The output is the transposed-value layout. Only its strides come from the
        // parsed layout: its sizes are what the CHECK_SHAPE below validates against
        // the input-derived ones, so they must stay input-derived.
        const VTLayout out_layout = parse_vt_layout(output, tensor_layout);
        stride_batch_output       = out_layout.stride_batch;
        stride_d_output           = out_layout.stride_d;
        stride_h_output           = out_layout.stride_h;

        int64_t padded_num_tokens;

        if (tensor_layout == 0) {
            num_tokens       = input.size(1);
            num_heads        = input.size(2);
            stride_seq_input = input.stride(1);
            stride_h_input   = input.stride(2);

            padded_num_tokens = at::round_up<int64_t>(num_tokens, CTA_TOKENS);

            CHECK_SHAPE_PADDED_SEQ(output, batch_size, head_dim, num_heads, padded_num_tokens);
        }
        else {
            num_tokens       = input.size(2);
            num_heads        = input.size(1);
            stride_seq_input = input.stride(2);
            stride_h_input   = input.stride(1);

            padded_num_tokens = at::round_up<int64_t>(num_tokens, CTA_TOKENS);
            CHECK_SHAPE_PADDED_SEQ(output, batch_size, num_heads, head_dim, padded_num_tokens);
        }

        grid_blocks = padded_num_tokens / CTA_TOKENS;
    }

    auto input_dtype  = input.scalar_type();
    auto output_dtype = output.scalar_type();

    TORCH_CHECK(input_dtype == output_dtype, "Input and output must have the same data type");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
        DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
            dim3 grid(grid_blocks, num_heads, batch_size);

            static_assert(CTA_TOKENS * HEAD_DIM <= 8192);

            dim3 block(CTA_TOKENS * (HEAD_DIM / 8));

            const bool is_varlen = varlen.cu_seqlens != nullptr;
            if (permute) {
                auto* kernel = is_varlen ? TransposePadPermuteKernel<HEAD_DIM, CTA_TOKENS, true, true, true, c_type> :
                                           TransposePadPermuteKernel<HEAD_DIM, CTA_TOKENS, true, true, false, c_type>;
                kernel<<<grid, block, 0, stream>>>(reinterpret_cast<c_type*>(input.data_ptr()),
                                                 reinterpret_cast<c_type*>(output.data_ptr()),
                                                 num_tokens,
                                                 stride_batch_input,
                                                 stride_seq_input,
                                                 stride_h_input,
                                                 stride_batch_output,
                                                 stride_d_output,
                                                 stride_h_output,
                                                 varlen.cu_seqlens,
                                                 static_cast<uint32_t>(varlen.pad_tokens));
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
            else {
                auto* kernel = is_varlen ? TransposePadPermuteKernel<HEAD_DIM, CTA_TOKENS, true, false, true, c_type> :
                                           TransposePadPermuteKernel<HEAD_DIM, CTA_TOKENS, true, false, false, c_type>;
                kernel<<<grid, block, 0, stream>>>(reinterpret_cast<c_type*>(input.data_ptr()),
                                                 reinterpret_cast<c_type*>(output.data_ptr()),
                                                 num_tokens,
                                                 stride_batch_input,
                                                 stride_seq_input,
                                                 stride_h_input,
                                                 stride_batch_output,
                                                 stride_d_output,
                                                 stride_h_output,
                                                 varlen.cu_seqlens,
                                                 static_cast<uint32_t>(varlen.pad_tokens));
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        });
    });
}

void transpose_pad_permute_cuda(torch::Tensor input, torch::Tensor output, int tensor_layout, const QuantVarlen& varlen)
{
    transpose_pad_impl(input, output, tensor_layout, /*permute=*/true, varlen);
}

void transpose_pad_cuda(torch::Tensor input, torch::Tensor output, int tensor_layout, const QuantVarlen& varlen)
{
    transpose_pad_impl(input, output, tensor_layout, /*permute=*/false, varlen);
}

// The transposed-value sizes, strides and output shape check that the two fp8
// V quantization launchers share. Dense is [B, H, D, padded] (HND) /
// [B, D, H, padded] (NHD); packed drops the batch dimension to
// [H, D, padded_total], and which slab a sequence owns is the kernel's
// business (varlen.h pad_offset), not the host's.
struct VTQuantLayout {
    int64_t batch_size, num_heads, head_dim;
    int64_t stride_batch_input, stride_d_input, stride_h_input;
    int64_t stride_batch_output, stride_d_output, stride_h_output;
};

static VTQuantLayout parse_vt_quant_layout(const torch::Tensor& input,
                                           const torch::Tensor& output,
                                           int                  tensor_layout,
                                           const QuantVarlen&   varlen)
{
    VTQuantLayout l;
    if (varlen.cu_seqlens != nullptr) {
        const VTLayout in_layout  = parse_vt_varlen_layout(input, varlen);
        const VTLayout out_layout = parse_vt_varlen_layout(output, varlen);

        CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2));

        l.batch_size          = varlen.batch_size;
        l.num_heads           = in_layout.num_heads;
        l.head_dim            = in_layout.head_dim;
        l.stride_batch_input  = 0;
        l.stride_d_input      = in_layout.stride_d;
        l.stride_h_input      = in_layout.stride_h;
        l.stride_batch_output = 0;
        l.stride_d_output     = out_layout.stride_d;
        l.stride_h_output     = out_layout.stride_h;
        return l;
    }

    const VTLayout in_layout  = parse_vt_layout(input, tensor_layout);
    const VTLayout out_layout = parse_vt_layout(output, tensor_layout);

    CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));

    l.batch_size          = in_layout.batch_size;
    l.num_heads           = in_layout.num_heads;
    l.head_dim            = in_layout.head_dim;
    l.stride_batch_input  = in_layout.stride_batch;
    l.stride_d_input      = in_layout.stride_d;
    l.stride_h_input      = in_layout.stride_h;
    l.stride_batch_output = out_layout.stride_batch;
    l.stride_d_output     = out_layout.stride_d;
    l.stride_h_output     = out_layout.stride_h;
    return l;
}

// Shared body of the two fp8 V quantization launchers below: they differ only
// in the mean tensor. An undefined mean selects the plain scale path, a
// defined one the sub_mean kernel (the transpose_quant_v_fp8_cuda convention).
static void scale_fuse_quant_impl(torch::Tensor      input,
                                  torch::Tensor      output,
                                  torch::Tensor      mean,
                                  torch::Tensor      scale,
                                  int                num_tokens,
                                  float              scale_max,
                                  int                tensor_layout,
                                  const QuantVarlen& varlen)
{
    const c10::cuda::CUDAGuard device_guard(input.device());

    // The fp8 conversion (cvt.rn.satfinite.e4m3x2) requires sm_89+; on older
    // parts the kernel guard compiles to a trap that kills the CUDA context.
    // Fail loudly on the host instead.
    {
        const cudaDeviceProp* prop = at::cuda::getDeviceProperties(input.device().index());
        TORCH_CHECK(prop->major > 8 || (prop->major == 8 && prop->minor >= 9),
                    "fp8 per-channel V quantization requires compute capability 8.9+, got sm_",
                    prop->major,
                    prop->minor);
    }

    const bool sub_mean = mean.defined();

    CHECK_CUDA(input);
    CHECK_CUDA(output);
    if (sub_mean) {
        CHECK_CUDA(mean);
    }
    CHECK_CUDA(scale);

    // CHECK_DTYPE(output, torch::kInt8);
    if (sub_mean) {
        CHECK_DTYPE(mean, torch::kFloat);
    }
    CHECK_DTYPE(scale, torch::kFloat);

    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(output);
    if (sub_mean) {
        CHECK_CONTIGUOUS(mean);
    }
    CHECK_CONTIGUOUS(scale);

    if (varlen.cu_seqlens == nullptr) {
        CHECK_DIMS(input, 4);
        CHECK_DIMS(output, 4);
    }  // the packed rank is checked by parse_vt_varlen_layout
    if (sub_mean) {
        CHECK_DIMS(mean, 3);
    }
    CHECK_DIMS(scale, 3);

    const VTQuantLayout l = parse_vt_quant_layout(input, output, tensor_layout, varlen);

    const int64_t batch_size = l.batch_size;
    const int64_t num_heads  = l.num_heads;
    const int64_t head_dim   = l.head_dim;

    const int64_t stride_batch_input = l.stride_batch_input;
    const int64_t stride_d_input     = l.stride_d_input;
    const int64_t stride_h_input     = l.stride_h_input;

    const int64_t stride_batch_output = l.stride_batch_output;
    const int64_t stride_d_output     = l.stride_d_output;
    const int64_t stride_h_output     = l.stride_h_output;

    if (sub_mean) {
        CHECK_SHAPE(mean, batch_size, num_heads, head_dim);
    }
    CHECK_SHAPE(scale, batch_size, num_heads, head_dim);

    constexpr int CTA_THREADS = 256;

    dim3 grid(num_heads, batch_size, head_dim);
    dim3 block(CTA_THREADS);

    auto input_dtype = input.scalar_type();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

// The launch the two flavours share; only the sub_mean template argument and
// the mean pointer/strides differ.
#define SAGEATTN_LAUNCH_MEAN_SCALE(SUB_MEAN, MEAN_PTR, STRIDE_BATCH_MEAN, STRIDE_H_MEAN)                               \
    MeanScaleKernel<64, SUB_MEAN, c_type><<<grid, block, 0, stream>>>(reinterpret_cast<c_type*>(input.data_ptr()),     \
                                                                      reinterpret_cast<int8_t*>(output.data_ptr()),    \
                                                                      MEAN_PTR,                                        \
                                                                      reinterpret_cast<float*>(scale.data_ptr()),      \
                                                                      scale_max,                                       \
                                                                      num_tokens,                                      \
                                                                      stride_batch_input,                              \
                                                                      stride_d_input,                                  \
                                                                      stride_h_input,                                  \
                                                                      stride_batch_output,                             \
                                                                      stride_d_output,                                 \
                                                                      stride_h_output,                                 \
                                                                      STRIDE_BATCH_MEAN,                               \
                                                                      STRIDE_H_MEAN,                                   \
                                                                      scale.stride(0),                                 \
                                                                      scale.stride(1),                                 \
                                                                      varlen.cu_seqlens,                               \
                                                                      static_cast<uint32_t>(varlen.pad_tokens));       \
    C10_CUDA_KERNEL_LAUNCH_CHECK()

    // Two dispatch bodies rather than one branch inside the dispatch: kernels
    // land in the cubin in (reverse) first-instantiation order, so this keeps
    // the pre-merge order - both sub_mean=false instances before the true
    // ones - and with it byte-identical SASS.
    if (!sub_mean) {
        DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(
            input_dtype, c_type, { SAGEATTN_LAUNCH_MEAN_SCALE(false, nullptr, 0, 0); });
    }
    else {
        DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
            SAGEATTN_LAUNCH_MEAN_SCALE(true, reinterpret_cast<float*>(mean.data_ptr()), mean.stride(0), mean.stride(1));
        });
    }
#undef SAGEATTN_LAUNCH_MEAN_SCALE
}

void scale_fuse_quant_cuda(torch::Tensor      input,
                           torch::Tensor      output,
                           torch::Tensor      scale,
                           int                num_tokens,
                           float              scale_max,
                           int                tensor_layout,
                           const QuantVarlen& varlen)
{
    scale_fuse_quant_impl(input, output, torch::Tensor(), scale, num_tokens, scale_max, tensor_layout, varlen);
}

void mean_scale_fuse_quant_cuda(torch::Tensor      input,
                                torch::Tensor      output,
                                torch::Tensor      mean,
                                torch::Tensor      scale,
                                int                num_tokens,
                                float              scale_max,
                                int                tensor_layout,
                                const QuantVarlen& varlen)
{
    scale_fuse_quant_impl(input, output, mean, scale, num_tokens, scale_max, tensor_layout, varlen);
}

bool transpose_quant_v_fp8_supported(int64_t head_dim)
{
    return head_dim % kVQuantChannelChunk == 0;
}

void transpose_quant_v_fp8_cuda(torch::Tensor      input,
                                torch::Tensor      output,
                                torch::Tensor      mean,
                                torch::Tensor      scale,
                                float              scale_max,
                                int                tensor_layout,
                                bool               permute,
                                const QuantVarlen& varlen)
{
    const c10::cuda::CUDAGuard device_guard(input.device());

    // The fp8 conversion (cvt.rn.satfinite.e4m3x2) requires sm_89+; on older
    // parts the kernel guard compiles to a trap that kills the CUDA context.
    // Fail loudly on the host instead.
    {
        const cudaDeviceProp* prop = at::cuda::getDeviceProperties(input.device().index());
        TORCH_CHECK(prop->major > 8 || (prop->major == 8 && prop->minor >= 9),
                    "fp8 per-channel V quantization requires compute capability 8.9+, got sm_",
                    prop->major,
                    prop->minor);
    }

    const bool sub_mean = mean.defined();

    CHECK_CUDA(input);
    CHECK_CUDA(output);
    CHECK_CUDA(scale);

    CHECK_DTYPE(scale, torch::kFloat);

    CHECK_LASTDIM_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(output);
    CHECK_CONTIGUOUS(scale);

    CHECK_DIMS(scale, 3);
    if (sub_mean) {
        CHECK_CUDA(mean);
        CHECK_DTYPE(mean, torch::kFloat);
        CHECK_CONTIGUOUS(mean);
        CHECK_DIMS(mean, 3);
    }

    // The alignment of the transposed axis the fp8 pass covers, and the block
    // size the two-pass transpose kernel wrote in. A pad_multiple=128 buffer's
    // [64-aligned, 128-aligned) tail stays untouched here exactly as it did
    // there; the caller zero-fills it.
    constexpr int CTA_TOKENS = 64;

    int64_t batch_size, head_dim, num_heads, num_tokens;
    int64_t stride_batch_input, stride_seq_input, stride_h_input;
    int64_t stride_batch_output, stride_d_output, stride_h_output;

    if (varlen.cu_seqlens != nullptr) {
        // packed [total_tokens, heads, head_dim] -> [heads, head_dim, padded_total]
        CHECK_DIMS(input, 3);

        const VTLayout out_layout = parse_vt_varlen_layout(output, varlen);

        batch_size          = varlen.batch_size;
        num_heads           = input.size(1);
        head_dim            = input.size(2);
        num_tokens          = 0;  // the kernel reads each sequence's own length
        stride_batch_input  = 0;
        stride_seq_input    = input.stride(0);
        stride_h_input      = input.stride(1);
        stride_batch_output = 0;
        stride_d_output     = out_layout.stride_d;
        stride_h_output     = out_layout.stride_h;

        TORCH_CHECK(out_layout.num_heads == num_heads && out_layout.head_dim == head_dim,
                    "quantized value must be (heads, head_dim, padded_total) = (",
                    num_heads,
                    ", ",
                    head_dim,
                    ", n), got (",
                    output.size(0),
                    ", ",
                    output.size(1),
                    ", ",
                    output.size(2),
                    ")");
        const int64_t padded_total = sage::blk_total(input.size(0), batch_size, varlen.pad_tokens) * varlen.pad_tokens;
        TORCH_CHECK(out_layout.padded_num_tokens == padded_total,
                    "quantized value last dim (",
                    out_layout.padded_num_tokens,
                    ") must be blk_total(total_tokens, batch_size, ",
                    varlen.pad_tokens,
                    ") * ",
                    varlen.pad_tokens,
                    " (",
                    padded_total,
                    ")");
        TORCH_CHECK(varlen.pad_tokens % CTA_TOKENS == 0,
                    "pad_multiple (",
                    varlen.pad_tokens,
                    ") must be a multiple of ",
                    CTA_TOKENS);
    }
    else {
        CHECK_DIMS(input, 4);
        CHECK_DIMS(output, 4);

        batch_size         = input.size(0);
        head_dim           = input.size(3);
        stride_batch_input = input.stride(0);

        const VTLayout out_layout = parse_vt_layout(output, tensor_layout);
        stride_batch_output       = out_layout.stride_batch;
        stride_d_output           = out_layout.stride_d;
        stride_h_output           = out_layout.stride_h;

        if (tensor_layout == 0) {
            num_tokens       = input.size(1);
            num_heads        = input.size(2);
            stride_seq_input = input.stride(1);
            stride_h_input   = input.stride(2);

            CHECK_SHAPE_PADDED_SEQ(
                output, batch_size, head_dim, num_heads, at::round_up<int64_t>(num_tokens, CTA_TOKENS));
        }
        else {
            num_tokens       = input.size(2);
            num_heads        = input.size(1);
            stride_seq_input = input.stride(2);
            stride_h_input   = input.stride(1);

            CHECK_SHAPE_PADDED_SEQ(
                output, batch_size, num_heads, head_dim, at::round_up<int64_t>(num_tokens, CTA_TOKENS));
        }

        CHECK_LEN_I32(num_tokens, num_tokens);
    }

    CHECK_SHAPE(scale, batch_size, num_heads, head_dim);
    if (sub_mean) {
        CHECK_SHAPE(mean, batch_size, num_heads, head_dim);
    }

    TORCH_CHECK(transpose_quant_v_fp8_supported(head_dim),
                "the fused V transpose+quantization kernel walks head_dim in chunks of ",
                kVQuantChannelChunk,
                ", got head_dim ",
                head_dim);

    float*        mean_ptr          = sub_mean ? reinterpret_cast<float*>(mean.data_ptr()) : nullptr;
    const int64_t stride_batch_mean = sub_mean ? mean.stride(0) : 0;
    const int64_t stride_h_mean     = sub_mean ? mean.stride(1) : 0;

    dim3 grid(head_dim / kVQuantChannelChunk, num_heads, batch_size);
    dim3 block(kVQuantThreads);

    auto input_dtype = input.scalar_type();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

// Preprocessor directives cannot live inside a macro argument, so the four
// (sub_mean, permute) instantiations are spelled out here rather than in the
// DISPATCH body below. c_type is the dispatch's, bound at the expansion site.
#define SAGEATTN_LAUNCH_FUSED_V_QUANT(SUB_MEAN, PERMUTE)                                                               \
    TransposeQuantFp8Kernel<kVQuantChannelChunk, CTA_TOKENS, SUB_MEAN, PERMUTE, c_type>                                \
        <<<grid, block, 0, stream>>>(reinterpret_cast<const c_type*>(input.data_ptr()),                                \
                                     reinterpret_cast<int8_t*>(output.data_ptr()),                                     \
                                     mean_ptr,                                                                         \
                                     reinterpret_cast<float*>(scale.data_ptr()),                                       \
                                     scale_max,                                                                        \
                                     static_cast<uint32_t>(num_tokens),                                                \
                                     stride_batch_input,                                                               \
                                     static_cast<uint32_t>(stride_seq_input),                                          \
                                     stride_h_input,                                                                   \
                                     stride_batch_output,                                                              \
                                     stride_d_output,                                                                  \
                                     stride_h_output,                                                                  \
                                     stride_batch_mean,                                                                \
                                     stride_h_mean,                                                                    \
                                     scale.stride(0),                                                                  \
                                     scale.stride(1),                                                                  \
                                     varlen.cu_seqlens,                                                                \
                                     static_cast<uint32_t>(varlen.pad_tokens))

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
        if (sub_mean) {
            if (permute) {
                SAGEATTN_LAUNCH_FUSED_V_QUANT(true, true);
            }
            else {
                SAGEATTN_LAUNCH_FUSED_V_QUANT(true, false);
            }
        }
        else {
            if (permute) {
                SAGEATTN_LAUNCH_FUSED_V_QUANT(false, true);
            }
            else {
                SAGEATTN_LAUNCH_FUSED_V_QUANT(false, false);
            }
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

#undef SAGEATTN_LAUNCH_FUSED_V_QUANT