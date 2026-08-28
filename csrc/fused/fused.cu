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

enum class QuantType {
    kInt8,
    kInt4,
};

template<uint32_t head_dim,
         uint32_t BLOCK_SIZE,
         uint32_t num_pack_per_thread = 1,
         bool     has_sm_scale        = false,
         bool     sub_mean            = false,
         typename T>
__global__ void QuantInt8Kernel(T* __restrict__ input,
                                T* __restrict__ mean,
                                int8_t* __restrict__ output,
                                float* __restrict__ scale,
                                float          sm_scale,
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

    uint32_t bx        = blockIdx.x;
    uint32_t head_id   = blockIdx.y;
    uint32_t batch_id  = blockIdx.z;
    uint32_t thread_id = threadIdx.x;

    uint32_t thread_base_token = bx * BLOCK_SIZE + thread_id / num_threads_per_token;
    T*       input_ptr_base    = input + static_cast<int64_t>(batch_id) * stride_bz_input
                        + static_cast<int64_t>(head_id) * stride_h_input
                        + static_cast<int64_t>(thread_base_token) * stride_seq_input
                        + static_cast<int64_t>(thread_id % num_threads_per_token * pack_size);
    T* mean_ptr_base = mean + static_cast<int64_t>(batch_id) * stride_bz_mean
                       + static_cast<int64_t>(head_id) * stride_h_mean
                       + static_cast<int64_t>(thread_id % num_threads_per_token * pack_size);
    int8_t* output_ptr_base = output + static_cast<int64_t>(batch_id) * stride_bz_output
                              + static_cast<int64_t>(head_id) * stride_h_output
                              + static_cast<int64_t>(thread_base_token) * stride_seq_output
                              + static_cast<int64_t>(thread_id % num_threads_per_token * pack_size);
    float* scale_ptr_base = scale + static_cast<int64_t>(batch_id) * stride_bz_scale
                            + static_cast<int64_t>(head_id) * stride_h_scale + static_cast<int64_t>(bx);

    if constexpr (sub_mean) {
        *(float4*)(&mean_val[0]) = *(float4*)(mean_ptr_base);
#pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
            mean_val_float[j] = fp_traits<T>::to_fp32(mean_val[j]);
        }
    }

    constexpr uint32_t iter_stride = BLOCK_SIZE / num_pack_per_thread;

// load the data
#pragma unroll
    for (uint32_t i = 0; i < num_pack_per_thread; i++) {
        if (thread_base_token + i * iter_stride < num_tokens) {
            *(float4*)(&x_val[i][0]) = *(float4*)(input_ptr_base + i * iter_stride * stride_seq_input);
#pragma unroll
            for (uint32_t j = 0; j < 8; j++) {
                x_val_float[i][j] = fp_traits<T>::to_fp32(x_val[i][j]);
            }

            if constexpr (sub_mean) {
#pragma unroll
                for (uint32_t j = 0; j < 8; j++) {
                    x_val_float[i][j] -= mean_val_float[j];
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

    float amax_val = 0.0000001f;  // prevent from dividing by zero

#pragma unroll
    for (uint32_t i = 0; i < num_pack_per_thread; i++) {
#pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
            amax_val = fmaxf(amax_val, fabsf(x_val_float[i][j]));
        }
    }

    __shared__ float s_amax;
    const float      block_amax_val = vllm::blockReduceMax(amax_val);
    if (thread_id == 0) {
        s_amax            = block_amax_val;
        scale_ptr_base[0] = s_amax / 127.0f;
    }

    __syncthreads();

    float tmp_scale = 127.0f / s_amax;

    char4 o_val[num_pack_per_thread][2];

#pragma unroll
    for (uint32_t i = 0; i < num_pack_per_thread; i++) {
#pragma unroll
        for (uint32_t j = 0; j < 2; j += 1) {
            o_val[i][j] = make_char4(float_to_int8_rn(x_val_float[i][j * 4 + 0] * tmp_scale),
                                     float_to_int8_rn(x_val_float[i][j * 4 + 1] * tmp_scale),
                                     float_to_int8_rn(x_val_float[i][j * 4 + 2] * tmp_scale),
                                     float_to_int8_rn(x_val_float[i][j * 4 + 3] * tmp_scale));
        }
    }

    // int8 result
#pragma unroll
    for (uint32_t i = 0; i < num_pack_per_thread; i++) {

        if (thread_base_token + i * iter_stride < num_tokens) {
            *reinterpret_cast<float2*>(output_ptr_base + i * iter_stride * stride_seq_output) =
                *reinterpret_cast<float2*>(&o_val[i][0]);
        }
    }
}

template<uint32_t head_dim, uint32_t BLOCK_SIZE, uint32_t num_pack_per_thread = 1, typename T>
__global__ void SubMeanKernel(T* __restrict__ input,
                              T* __restrict__ mean,
                              half* __restrict__ output,
                              const uint32_t num_tokens,
                              const int64_t  stride_bz_input,
                              const uint32_t stride_seq_input,
                              const int64_t  stride_h_input,
                              const int64_t  stride_bz_mean,
                              const int64_t  stride_h_mean,
                              const int64_t  stride_bz_output,
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

    uint32_t bx        = blockIdx.x;
    uint32_t head_id   = blockIdx.y;
    uint32_t batch_id  = blockIdx.z;
    uint32_t thread_id = threadIdx.x;

    uint32_t thread_base_token = bx * BLOCK_SIZE + thread_id / num_threads_per_token;
    T*       input_ptr_base    = input + static_cast<int64_t>(batch_id) * stride_bz_input
                        + static_cast<int64_t>(head_id) * stride_h_input
                        + static_cast<int64_t>(thread_base_token) * stride_seq_input
                        + static_cast<int64_t>(thread_id % num_threads_per_token * pack_size);
    T* mean_ptr_base = mean + static_cast<int64_t>(batch_id) * stride_bz_mean
                       + static_cast<int64_t>(head_id) * stride_h_mean
                       + static_cast<int64_t>(thread_id % num_threads_per_token * pack_size);
    half* output_ptr_base = output + static_cast<int64_t>(batch_id) * stride_bz_output
                            + static_cast<int64_t>(head_id) * stride_h_output
                            + static_cast<int64_t>(thread_base_token) * stride_seq_output
                            + static_cast<int64_t>(thread_id % num_threads_per_token * pack_size);

    *(float4*)(&mean_val[0]) = *(float4*)(mean_ptr_base);

    constexpr uint32_t iter_stride = BLOCK_SIZE / num_pack_per_thread;

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

template<uint32_t head_dim, uint32_t CTA_SIZE, bool pad_zero = false, bool permute = true, typename T>
__global__ void TransposePadPermuteKernel(T* __restrict__ input,
                                          T* __restrict__ output,
                                          const uint32_t num_tokens,
                                          const int64_t  stride_bz_input,
                                          const uint32_t stride_seq_input,
                                          const int64_t  stride_h_input,
                                          const int64_t  stride_bz_output,
                                          const uint32_t stride_d_output,
                                          const int64_t  stride_h_output)
{

    static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value,
                  "Only half and bfloat16 are supported");

    constexpr uint32_t pack_size             = 8;  // float4 contains 8 half or 8 bfloat16
    uint32_t           num_threads_per_token = head_dim / pack_size;
    uint32_t           num_threads_per_cta   = CTA_SIZE / pack_size;

    uint32_t bx        = blockIdx.x;
    uint32_t head_id   = blockIdx.y;
    uint32_t batch_id  = blockIdx.z;
    uint32_t thread_id = threadIdx.x;

    uint32_t thread_base_token = bx * CTA_SIZE + thread_id / num_threads_per_token;

    T* input_ptr_base = input + static_cast<int64_t>(batch_id) * stride_bz_input
                        + static_cast<int64_t>(head_id) * stride_h_input
                        + static_cast<int64_t>(thread_base_token) * stride_seq_input
                        + static_cast<int64_t>(thread_id % num_threads_per_token * pack_size);
    T* output_ptr_base = output + static_cast<int64_t>(batch_id) * stride_bz_output
                         + static_cast<int64_t>(head_id) * stride_h_output + static_cast<int64_t>(bx * CTA_SIZE)
                         + static_cast<int64_t>(thread_id % num_threads_per_cta * pack_size)
                         + static_cast<int64_t>(thread_id / num_threads_per_cta) * stride_d_output;

    // +8 halves of row padding: the transpose loop below reads
    // shared_load[row][col] with col warp-uniform, so an unpadded row stride of
    // head_dim*2 bytes (a multiple of 128) lands all 32 lanes in one bank
    // (32-way conflict, 8 serialized trips per tile). The pad must stay a
    // multiple of 8 halves (16B) to keep the cp.async row starts aligned, which
    // caps the improvement at 4-way instead of conflict-free.
    __shared__ T shared_load[CTA_SIZE][head_dim + 8];
    __shared__ T shared_store[head_dim][CTA_SIZE];

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
        shared_load[smem_load_row] + thread_id % num_threads_per_token * pack_size,
        input_ptr_base,
        thread_base_token < num_tokens);
    cp_async::commit_group();
    cp_async::wait_group<0>();
    __syncthreads();

    uint32_t smem_row_base   = thread_id % CTA_SIZE;
    uint32_t smem_col_base   = thread_id / CTA_SIZE;
    uint32_t smem_col_stride = head_dim / 8;

    // TODO: use ldmatrix to do permutation
#pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        shared_store[smem_col_base + i * smem_col_stride][smem_row_base] =
            shared_load[smem_row_base][smem_col_base + i * smem_col_stride];
    }

    __syncthreads();

    *(float4*)(output_ptr_base) =
        *(float4*)(&shared_store[thread_id / num_threads_per_cta][thread_id % num_threads_per_cta * pack_size]);
}

template<uint32_t pad_size, bool sub_mean = false, typename T>
__global__ void MeanScaleKernel(T* __restrict__ input,
                                int8_t* __restrict__ output,
                                float* __restrict__ mean,
                                float* __restrict__ scale,
                                const float    scale_max,
                                const uint32_t num_tokens,
                                const int64_t  stride_bz_input,
                                const int64_t  stride_d_input,
                                const int64_t  stride_h_input,
                                const int64_t  stride_bz_output,
                                const int64_t  stride_d_output,
                                const int64_t  stride_h_output,
                                const int64_t  stride_bz_mean,
                                const int64_t  stride_h_mean,
                                const int64_t  stride_bz_scale,
                                const int64_t  stride_h_scale)
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
    // pad the number of tokens to 16 to deal with fp8 permute in previous kernel
    uint32_t fp8_padded_num_tokens = at::round_up<uint32_t>(num_tokens, 16);
    uint32_t num_iters =
        fp8_padded_num_tokens / gmem_stride + ((fp8_padded_num_tokens % gmem_stride) > thread_id * pack_size);
    // the quantize pass covers all fp8 output tokens to prevent nan in random initialization
    uint32_t padded_num_tokens = at::round_up<uint32_t>(num_tokens, pad_size);
    uint32_t num_quant_iters =
        padded_num_tokens / gmem_stride + ((padded_num_tokens % gmem_stride) > thread_id * pack_size);

    const bool cached = (num_iters == num_quant_iters) && (num_iters <= kCache);

    T* input_ptr_base = input + static_cast<int64_t>(batch_id) * stride_bz_input
                        + static_cast<int64_t>(head_id) * stride_h_input + static_cast<int64_t>(d_id) * stride_d_input
                        + static_cast<int64_t>(thread_id * pack_size);
    int8_t* output_ptr_base =
        output + static_cast<int64_t>(batch_id) * stride_bz_output + static_cast<int64_t>(head_id) * stride_h_output
        + static_cast<int64_t>(d_id) * stride_d_output + static_cast<int64_t>(thread_id * pack_size);

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
    __shared__ float s_amax_val;
    __shared__ float s_mean_val;

    float block_max_val = vllm::blockReduceMax(max_val);
    float block_min_val = vllm::blockReduceMin(min_val);
    float block_sum_val;

    if constexpr (sub_mean) {
        block_sum_val = vllm::blockReduceSum(sum_val);
    }

    if (thread_id == 0) {
        // s_mean_val is only defined (and only consumed) on the sub_mean path;
        // computing it unconditionally used to read the uninitialized
        // block_sum_val (UB, harmless in practice but wrong).
        if constexpr (sub_mean) {
            s_mean_val = block_sum_val / fp8_padded_num_tokens;
            s_amax_val = fmaxf(fabsf(block_max_val - s_mean_val), fabsf(block_min_val - s_mean_val));
            mean[static_cast<int64_t>(batch_id) * stride_bz_mean + static_cast<int64_t>(head_id) * stride_h_mean
                 + static_cast<int64_t>(d_id)] = s_mean_val;
        }
        else {
            s_amax_val = fmaxf(fabsf(block_max_val), fabsf(block_min_val));
        }

        scale[static_cast<int64_t>(batch_id) * stride_bz_scale + static_cast<int64_t>(head_id) * stride_h_scale
              + static_cast<int64_t>(d_id)] = s_amax_val / scale_max;
    }

    __syncthreads();

    float mean_val = 0.0f;
    if constexpr (sub_mean) {
        mean_val = s_mean_val;
    }
    float recp_scale = scale_max / s_amax_val;

    auto quantize_store = [&](const pack_t& x_val, uint32_t i) {
        float    x_val_float[8];
        uint32_t x_val_fp8[2];
#pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
            x_val_float[j] = fp_traits<T>::to_fp32(x_val[j]);
            if constexpr (sub_mean) {
                x_val_float[j] = (x_val_float[j] - mean_val) * recp_scale;
            }
            else {
                x_val_float[j] *= recp_scale;
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

void quant_per_block_int8_cuda(
    torch::Tensor input, torch::Tensor output, torch::Tensor scale, float sm_scale, int block_size, int tensor_layout)
{
    const c10::cuda::CUDAGuard device_guard(input.device());

    QuantLayout ql = parse_quant_layout(input, output, scale, tensor_layout);
    SAGEATTN_QUANT_LAYOUT_LOCALS(ql);

    auto input_dtype = input.scalar_type();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
        DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
            DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
                CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
                CHECK_SHAPE(scale, batch_size, num_heads, at::ceil_div<int64_t>(num_tokens, BLOCK_SIZE));

                dim3 grid(at::ceil_div<int64_t>(num_tokens, BLOCK_SIZE), num_heads, batch_size);

                constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

                dim3 block(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

                QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, true, false, c_type>
                    <<<grid, block, 0, stream>>>(reinterpret_cast<c_type*>(input.data_ptr()),
                                                 nullptr,
                                                 output.data_ptr<int8_t>(),
                                                 reinterpret_cast<float*>(scale.data_ptr()),
                                                 sm_scale,
                                                 num_tokens,
                                                 stride_bz_input,
                                                 stride_seq_input,
                                                 stride_h_input,
                                                 0,
                                                 0,
                                                 stride_bz_output,
                                                 stride_seq_output,
                                                 stride_h_output,
                                                 scale.stride(0),
                                                 scale.stride(1));
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
}

void quant_per_block_int8_cuda(
    torch::Tensor input, torch::Tensor output, torch::Tensor scale, int block_size, int tensor_layout)
{
    const c10::cuda::CUDAGuard device_guard(input.device());

    QuantLayout ql = parse_quant_layout(input, output, scale, tensor_layout);
    SAGEATTN_QUANT_LAYOUT_LOCALS(ql);

    auto input_dtype = input.scalar_type();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
        DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
            DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
                CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
                CHECK_SHAPE(scale, batch_size, num_heads, at::ceil_div<int64_t>(num_tokens, BLOCK_SIZE));

                dim3 grid(at::ceil_div<int64_t>(num_tokens, BLOCK_SIZE), num_heads, batch_size);

                constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

                dim3 block(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

                QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, false, false, c_type>
                    <<<grid, block, 0, stream>>>(reinterpret_cast<c_type*>(input.data_ptr()),
                                                 nullptr,
                                                 output.data_ptr<int8_t>(),
                                                 reinterpret_cast<float*>(scale.data_ptr()),
                                                 0.0f,
                                                 num_tokens,
                                                 stride_bz_input,
                                                 stride_seq_input,
                                                 stride_h_input,
                                                 0,
                                                 0,
                                                 stride_bz_output,
                                                 stride_seq_output,
                                                 stride_h_output,
                                                 scale.stride(0),
                                                 scale.stride(1));
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
}

void quant_per_block_int8_fuse_sub_mean_cuda(torch::Tensor input,
                                             torch::Tensor mean,
                                             torch::Tensor output,
                                             torch::Tensor scale,
                                             int           block_size,
                                             int           tensor_layout)
{
    const c10::cuda::CUDAGuard device_guard(input.device());

    QuantLayout ql = parse_quant_layout(input, output, scale, tensor_layout, &mean);
    SAGEATTN_QUANT_LAYOUT_LOCALS(ql);

    auto input_dtype = input.scalar_type();
    auto mean_dtype  = mean.scalar_type();

    TORCH_CHECK(input_dtype == mean_dtype, "Input and mean must have the same data type");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
        DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
            DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
                CHECK_SHAPE(mean, batch_size, num_heads, head_dim);
                CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
                CHECK_SHAPE(scale, batch_size, num_heads, at::ceil_div<int64_t>(num_tokens, BLOCK_SIZE));

                dim3 grid(at::ceil_div<int64_t>(num_tokens, BLOCK_SIZE), num_heads, batch_size);

                constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

                dim3 block(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

                QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, false, true, c_type>
                    <<<grid, block, 0, stream>>>(reinterpret_cast<c_type*>(input.data_ptr()),
                                                 reinterpret_cast<c_type*>(mean.data_ptr()),
                                                 output.data_ptr<int8_t>(),
                                                 reinterpret_cast<float*>(scale.data_ptr()),
                                                 0.0f,
                                                 num_tokens,
                                                 stride_bz_input,
                                                 stride_seq_input,
                                                 stride_h_input,
                                                 mean.stride(0),
                                                 mean.stride(1),
                                                 stride_bz_output,
                                                 stride_seq_output,
                                                 stride_h_output,
                                                 scale.stride(0),
                                                 scale.stride(1));
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
}

// use block size 128 and warp_block size 32
void quant_per_warp_int8_cuda(torch::Tensor input,
                              torch::Tensor output,
                              torch::Tensor scale,
                              int           block_size,
                              int           warp_block_size,
                              int           tensor_layout)
{
    const c10::cuda::CUDAGuard device_guard(input.device());

    QuantLayout ql = parse_quant_layout(input, output, scale, tensor_layout);
    SAGEATTN_QUANT_LAYOUT_LOCALS(ql);

    auto input_dtype = input.scalar_type();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
        DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
            DISPATCH_WARP_BLOCK_SIZE(warp_block_size, WARP_BLOCK_SIZE, {
                DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
                    CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
                    CHECK_SHAPE(scale,
                                batch_size,
                                num_heads,
                                at::ceil_div<int64_t>(num_tokens, BLOCK_SIZE) * (BLOCK_SIZE / WARP_BLOCK_SIZE));

                    dim3 grid(at::ceil_div<int64_t>(num_tokens, BLOCK_SIZE) * (BLOCK_SIZE / WARP_BLOCK_SIZE),
                              num_heads,
                              batch_size);

                    constexpr int num_pack_per_thread = (WARP_BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

                    dim3 block(WARP_BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

                    QuantInt8Kernel<HEAD_DIM, WARP_BLOCK_SIZE, num_pack_per_thread, false, false, c_type>
                        <<<grid, block, 0, stream>>>(reinterpret_cast<c_type*>(input.data_ptr()),
                                                     nullptr,
                                                     output.data_ptr<int8_t>(),
                                                     reinterpret_cast<float*>(scale.data_ptr()),
                                                     0.0,
                                                     num_tokens,
                                                     stride_bz_input,
                                                     stride_seq_input,
                                                     stride_h_input,
                                                     0,
                                                     0,
                                                     stride_bz_output,
                                                     stride_seq_output,
                                                     stride_h_output,
                                                     scale.stride(0),
                                                     scale.stride(1));
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

    int64_t stride_bz_input  = input.stride(0);
    int64_t stride_bz_output = output.stride(0);

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

            constexpr int BLOCK_SIZE = (HEAD_DIM == 128) ? 64 : 128;

            dim3 grid(at::ceil_div<int64_t>(num_tokens, BLOCK_SIZE), num_heads, batch_size);

            constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

            dim3 block(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

            SubMeanKernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread>
                <<<grid, block, 0, stream>>>(reinterpret_cast<c_type*>(input.data_ptr()),
                                             reinterpret_cast<c_type*>(mean.data_ptr()),
                                             reinterpret_cast<half*>(output.data_ptr()),
                                             num_tokens,
                                             stride_bz_input,
                                             stride_seq_input,
                                             stride_h_input,
                                             mean.stride(0),
                                             mean.stride(1),
                                             stride_bz_output,
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
                    && x.size(3) % CTA_SIZE == 0,                                                                      \
                "Tensor " #x " must have shape (" #d0 ", " #d1 ", " #d2 ", n) "                                        \
                "with n a multiple of ",                                                                               \
                CTA_SIZE,                                                                                              \
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

static void transpose_pad_impl(torch::Tensor input, torch::Tensor output, int tensor_layout, bool permute)
{
    const c10::cuda::CUDAGuard device_guard(input.device());

    CHECK_CUDA(input);
    CHECK_CUDA(output);

    CHECK_LASTDIM_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(output);

    CHECK_DIMS(input, 4);
    CHECK_DIMS(output, 4);

    constexpr int CTA_SIZE = 64;

    const int64_t batch_size = input.size(0);
    const int64_t head_dim   = input.size(3);

    int64_t stride_bz_input = input.stride(0);

    // The output is the transposed-value layout. Only its strides come from the
    // parsed layout: its sizes are what the CHECK_SHAPE below validates against
    // the input-derived ones, so they must stay input-derived.
    const VTLayout ol               = parse_vt_layout(output, tensor_layout);
    const int64_t  stride_bz_output = ol.stride_bz;
    const int64_t  stride_d_output  = ol.stride_d;
    const int64_t  stride_h_output  = ol.stride_h;

    int64_t num_tokens, padded_num_tokens, num_heads;
    int64_t stride_seq_input, stride_h_input;

    if (tensor_layout == 0) {
        num_tokens       = input.size(1);
        num_heads        = input.size(2);
        stride_seq_input = input.stride(1);
        stride_h_input   = input.stride(2);

        padded_num_tokens = at::round_up<int64_t>(num_tokens, CTA_SIZE);

        CHECK_SHAPE_PADDED_SEQ(output, batch_size, head_dim, num_heads, padded_num_tokens);
    }
    else {
        num_tokens       = input.size(2);
        num_heads        = input.size(1);
        stride_seq_input = input.stride(2);
        stride_h_input   = input.stride(1);

        padded_num_tokens = at::round_up<int64_t>(num_tokens, CTA_SIZE);
        CHECK_SHAPE_PADDED_SEQ(output, batch_size, num_heads, head_dim, padded_num_tokens);
    }

    auto input_dtype  = input.scalar_type();
    auto output_dtype = output.scalar_type();

    TORCH_CHECK(input_dtype == output_dtype, "Input and output must have the same data type");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
        DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
            dim3 grid(padded_num_tokens / CTA_SIZE, num_heads, batch_size);

            static_assert(CTA_SIZE * HEAD_DIM <= 8192);

            dim3 block(CTA_SIZE * (HEAD_DIM / 8));

            if (permute) {
                TransposePadPermuteKernel<HEAD_DIM, CTA_SIZE, true, true, c_type>
                    <<<grid, block, 0, stream>>>(reinterpret_cast<c_type*>(input.data_ptr()),
                                                 reinterpret_cast<c_type*>(output.data_ptr()),
                                                 num_tokens,
                                                 stride_bz_input,
                                                 stride_seq_input,
                                                 stride_h_input,
                                                 stride_bz_output,
                                                 stride_d_output,
                                                 stride_h_output);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
            else {
                TransposePadPermuteKernel<HEAD_DIM, CTA_SIZE, true, false, c_type>
                    <<<grid, block, 0, stream>>>(reinterpret_cast<c_type*>(input.data_ptr()),
                                                 reinterpret_cast<c_type*>(output.data_ptr()),
                                                 num_tokens,
                                                 stride_bz_input,
                                                 stride_seq_input,
                                                 stride_h_input,
                                                 stride_bz_output,
                                                 stride_d_output,
                                                 stride_h_output);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        });
    });
}

void transpose_pad_permute_cuda(torch::Tensor input, torch::Tensor output, int tensor_layout)
{
    transpose_pad_impl(input, output, tensor_layout, /*permute=*/true);
}

void transpose_pad_cuda(torch::Tensor input, torch::Tensor output, int tensor_layout)
{
    transpose_pad_impl(input, output, tensor_layout, /*permute=*/false);
}

void scale_fuse_quant_cuda(
    torch::Tensor input, torch::Tensor output, torch::Tensor scale, int num_tokens, float scale_max, int tensor_layout)
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

    CHECK_CUDA(input);
    CHECK_CUDA(output);
    CHECK_CUDA(scale);

    // CHECK_DTYPE(output, torch::kInt8);
    CHECK_DTYPE(scale, torch::kFloat);

    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(output);
    CHECK_CONTIGUOUS(scale);

    CHECK_DIMS(input, 4);
    CHECK_DIMS(output, 4);
    CHECK_DIMS(scale, 3);

    const VTLayout il = parse_vt_layout(input, tensor_layout);
    const VTLayout ol = parse_vt_layout(output, tensor_layout);

    const int64_t batch_size = il.batch_size;
    const int64_t num_heads  = il.num_heads;
    const int64_t head_dim   = il.head_dim;

    const int64_t stride_bz_input = il.stride_bz;
    const int64_t stride_d_input  = il.stride_d;
    const int64_t stride_h_input  = il.stride_h;

    const int64_t stride_bz_output = ol.stride_bz;
    const int64_t stride_d_output  = ol.stride_d;
    const int64_t stride_h_output  = ol.stride_h;

    CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
    CHECK_SHAPE(scale, batch_size, num_heads, head_dim);

    constexpr int CTA_SIZE = 256;

    dim3 grid(num_heads, batch_size, head_dim);
    dim3 block(CTA_SIZE);

    auto input_dtype = input.scalar_type();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
        MeanScaleKernel<64, false, c_type><<<grid, block, 0, stream>>>(reinterpret_cast<c_type*>(input.data_ptr()),
                                                                       reinterpret_cast<int8_t*>(output.data_ptr()),
                                                                       nullptr,
                                                                       reinterpret_cast<float*>(scale.data_ptr()),
                                                                       scale_max,
                                                                       num_tokens,
                                                                       stride_bz_input,
                                                                       stride_d_input,
                                                                       stride_h_input,
                                                                       stride_bz_output,
                                                                       stride_d_output,
                                                                       stride_h_output,
                                                                       0,
                                                                       0,
                                                                       scale.stride(0),
                                                                       scale.stride(1));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

void mean_scale_fuse_quant_cuda(torch::Tensor input,
                                torch::Tensor output,
                                torch::Tensor mean,
                                torch::Tensor scale,
                                int           num_tokens,
                                float         scale_max,
                                int           tensor_layout)
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

    CHECK_CUDA(input);
    CHECK_CUDA(output);
    CHECK_CUDA(mean);
    CHECK_CUDA(scale);

    // CHECK_DTYPE(output, torch::kInt8);
    CHECK_DTYPE(mean, torch::kFloat);
    CHECK_DTYPE(scale, torch::kFloat);

    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(output);
    CHECK_CONTIGUOUS(mean);
    CHECK_CONTIGUOUS(scale);

    CHECK_DIMS(input, 4);
    CHECK_DIMS(output, 4);
    CHECK_DIMS(mean, 3);
    CHECK_DIMS(scale, 3);

    const VTLayout il = parse_vt_layout(input, tensor_layout);
    const VTLayout ol = parse_vt_layout(output, tensor_layout);

    const int64_t batch_size = il.batch_size;
    const int64_t num_heads  = il.num_heads;
    const int64_t head_dim   = il.head_dim;

    const int64_t stride_bz_input = il.stride_bz;
    const int64_t stride_d_input  = il.stride_d;
    const int64_t stride_h_input  = il.stride_h;

    const int64_t stride_bz_output = ol.stride_bz;
    const int64_t stride_d_output  = ol.stride_d;
    const int64_t stride_h_output  = ol.stride_h;

    CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
    CHECK_SHAPE(mean, batch_size, num_heads, head_dim);
    CHECK_SHAPE(scale, batch_size, num_heads, head_dim);

    constexpr int CTA_SIZE = 256;

    dim3 grid(num_heads, batch_size, head_dim);
    dim3 block(CTA_SIZE);

    auto input_dtype = input.scalar_type();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
        MeanScaleKernel<64, true, c_type><<<grid, block, 0, stream>>>(reinterpret_cast<c_type*>(input.data_ptr()),
                                                                      reinterpret_cast<int8_t*>(output.data_ptr()),
                                                                      reinterpret_cast<float*>(mean.data_ptr()),
                                                                      reinterpret_cast<float*>(scale.data_ptr()),
                                                                      scale_max,
                                                                      num_tokens,
                                                                      stride_bz_input,
                                                                      stride_d_input,
                                                                      stride_h_input,
                                                                      stride_bz_output,
                                                                      stride_d_output,
                                                                      stride_h_output,
                                                                      mean.stride(0),
                                                                      mean.stride(1),
                                                                      scale.stride(0),
                                                                      scale.stride(1));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}