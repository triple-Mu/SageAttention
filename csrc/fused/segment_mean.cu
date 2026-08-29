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

// The per-sequence K mean of the packed [total_tokens, heads, head_dim]
// layout: smooth_k's production side. The consumption side already exists -
// the fuse_sub_mean quantization kernels take the mean as an input - but the
// reduction itself used to be an ATen composite (repeat_interleave segment ids
// + a float32 index_add_ over the whole packed tensor), which cost more than
// the attention win of packing on fast parts. This kernel reads each token
// once and writes [batch_size, heads, head_dim].
//
// Work split: blockIdx walks (token chunk, head, sequence), so the grid never
// depends on how the tokens are split between sequences - grid.x is opened to
// ceil(max_seqlen / chunk) like every other varlen launch, and blocks past
// their own sequence exit up front. One block reduces one chunk of
// kSegmentMeanChunkTokens tokens for one (sequence, head); a single-chunk grid
// divides and writes the mean directly, a multi-chunk grid writes float32
// partial sums for a second, fixed-order pass over exactly the chunks that
// were written. Every reduction order is fixed, so unlike the index_add_
// composite the result is deterministic run to run.

#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include "../dispatch_utils.h"
#include "../gmem_access.cuh"
#include "../numeric_conversion.cuh"
#include "../sageattn/varlen.h"
#include "../utils.cuh"
#include "quant_utils.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

// Tokens reduced per block. Also the bound under which the single-kernel path
// applies: a batch whose max_seqlen fits in one chunk skips the partial-sum
// round trip entirely.
constexpr uint32_t kSegmentMeanChunkTokens = 1024;

template<uint32_t head_dim, bool kSingleChunk, typename T>
__global__ void SegmentMeanKernel(T* __restrict__ input,
                                  T* __restrict__ mean,        // [batch, heads, head_dim], kSingleChunk only
                                  float* __restrict__ partial,  // [batch, heads, chunks, head_dim] otherwise
                                  const uint32_t stride_seq_input,
                                  const int64_t  stride_h_input,
                                  const int32_t* __restrict__ cu_seqlens)
{
    static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value,
                  "Only half and bfloat16 are supported");

    constexpr uint32_t pack_size             = 8;  // float4 contains 8 half or 8 bfloat16
    constexpr uint32_t num_threads_per_token = head_dim / pack_size;
    // 512 threads: a latency-bound reduction is paid for in loads in flight
    // per SM, and the smem staging below stays 16 KB for either head_dim.
    constexpr uint32_t block_threads   = 512;
    constexpr uint32_t tokens_per_iter = block_threads / num_threads_per_token;

    const uint32_t chunk_idx = blockIdx.x;
    const uint32_t head_id   = blockIdx.y;
    const uint32_t batch_id  = blockIdx.z;
    const uint32_t thread_id = threadIdx.x;

    const uint32_t seq_tokens = static_cast<uint32_t>(sage::seq_len(cu_seqlens, batch_id));
    const int64_t  seq_base   = sage::seq_offset(cu_seqlens, batch_id);

    const uint32_t chunk_begin = chunk_idx * kSegmentMeanChunkTokens;
    const uint32_t chunk_end   = min(chunk_begin + kSegmentMeanChunkTokens, seq_tokens);

    // A chunk past this sequence's own length leaves without writing, and the
    // finish kernel folds exactly the chunks that were written, so the cost of
    // a loose max_seqlen bound is idle blocks, not workspace traffic. The
    // at::empty workspace stays deterministic because the unwritten slots are
    // never read. Block-uniform, before any __syncthreads. The single-chunk
    // grid stays: an empty sequence must still write its zero mean.
    if constexpr (!kSingleChunk) {
        if (chunk_begin >= seq_tokens) {
            return;
        }
    }

    const uint32_t phase    = thread_id / num_threads_per_token;
    const uint32_t col_base = thread_id % num_threads_per_token * pack_size;

    float acc[pack_size];
#pragma unroll
    for (uint32_t j = 0; j < pack_size; j++) {
        acc[j] = 0.0f;
    }

    T* input_ptr_base =
        input + seq_base * static_cast<int64_t>(stride_seq_input) + static_cast<int64_t>(head_id) * stride_h_input
        + static_cast<int64_t>(col_base);

    // kUnroll loads into distinct registers, addresses computed up front, then
    // the accumulate chain: kUnroll independent LDGs stay in flight per thread
    // (a plain `#pragma unroll` here reuses one register quad and serializes
    // every load behind the previous iteration's adds).
    constexpr uint32_t kUnroll = 4;
    uint32_t           token   = chunk_begin + phase;
    for (; token + (kUnroll - 1) * tokens_per_iter < chunk_end; token += kUnroll * tokens_per_iter) {
        sage::vec_t<T, pack_size> x_val[kUnroll];
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; u++) {
            x_val[u].load_ro(input_ptr_base
                             + static_cast<int64_t>(token + u * tokens_per_iter) * stride_seq_input);
        }
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; u++) {
#pragma unroll
            for (uint32_t j = 0; j < pack_size; j++) {
                acc[j] += fp_traits<T>::to_fp32(x_val[u][j]);
            }
        }
    }
    for (; token < chunk_end; token += tokens_per_iter) {
        sage::vec_t<T, pack_size> x_val;
        x_val.load_ro(input_ptr_base + static_cast<int64_t>(token) * stride_seq_input);
#pragma unroll
        for (uint32_t j = 0; j < pack_size; j++) {
            acc[j] += fp_traits<T>::to_fp32(x_val[j]);
        }
    }

    // Cross-phase reduction, fixed order (phase 0, 1, ...): deterministic.
    __shared__ float smem[tokens_per_iter][head_dim];
#pragma unroll
    for (uint32_t j = 0; j < pack_size; j++) {
        smem[phase][col_base + j] = acc[j];
    }
    __syncthreads();

    if (thread_id < head_dim) {
        float sum_val = 0.0f;
#pragma unroll
        for (uint32_t p = 0; p < tokens_per_iter; p++) {
            sum_val += smem[p][thread_id];
        }

        if constexpr (kSingleChunk) {
            // An empty sequence gets a zero mean (clamped divisor, zero sum),
            // matching the ATen composite this kernel replaces.
            const float divisor = seq_tokens > 0 ? static_cast<float>(seq_tokens) : 1.0f;
            mean[(static_cast<int64_t>(batch_id) * gridDim.y + head_id) * head_dim + thread_id] =
                fp_traits<T>::from_fp32(sum_val / divisor);
        }
        else {
            partial[((static_cast<int64_t>(batch_id) * gridDim.y + head_id) * gridDim.x + chunk_idx) * head_dim
                    + thread_id] = sum_val;
        }
    }
}

// Second pass of the multi-chunk grid: fold the [batch, heads, chunks,
// head_dim] partial sums chunk by chunk (fixed order again), divide by the
// sequence's own length, and cast once - the same accumulate-in-float32
// contract as the dense k.mean(dim=seq_dim). blockDim.x is head_dim. Only the
// chunks the first pass wrote for this sequence are read, so the fold's cost
// follows the tokens present, not the max_seqlen bound.
template<typename T>
__global__ void SegmentMeanFinishKernel(const float* __restrict__ partial,
                                        T* __restrict__ mean,
                                        const uint32_t num_chunks,
                                        const int32_t* __restrict__ cu_seqlens)
{
    const uint32_t head_id   = blockIdx.x;
    const uint32_t batch_id  = blockIdx.y;
    const uint32_t col       = threadIdx.x;
    const uint32_t head_dim  = blockDim.x;

    const uint32_t seq_tokens = static_cast<uint32_t>(sage::seq_len(cu_seqlens, batch_id));
    const float    divisor    = seq_tokens > 0 ? static_cast<float>(seq_tokens) : 1.0f;
    const uint32_t seq_chunks =
        min(num_chunks, (seq_tokens + kSegmentMeanChunkTokens - 1) / kSegmentMeanChunkTokens);

    const float* partial_ptr_base =
        partial + (static_cast<int64_t>(batch_id) * gridDim.x + head_id) * num_chunks * head_dim
        + static_cast<int64_t>(col);
    float sum_val = 0.0f;
    for (uint32_t c = 0; c < seq_chunks; c++) {
        sum_val += partial_ptr_base[c * head_dim];
    }
    mean[(static_cast<int64_t>(batch_id) * gridDim.x + head_id) * head_dim + col] =
        fp_traits<T>::from_fp32(sum_val / divisor);
}

}  // namespace

void segment_mean_cuda(torch::Tensor input, torch::Tensor mean, const QuantVarlen& varlen)
{
    const c10::cuda::CUDAGuard device_guard(input.device());

    CHECK_CUDA(input);
    CHECK_CUDA(mean);
    CHECK_LASTDIM_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(mean);
    CHECK_DIMS(input, 3);
    CHECK_DIMS(mean, 3);
    TORCH_CHECK(mean.scalar_type() == input.scalar_type(), "mean must have the same dtype as input");
    CHECK_SHAPE(mean, varlen.batch_size, input.size(1), input.size(2));
    CHECK_LEN_I32(total_tokens, input.size(0));
    CHECK_STRIDE_LOOP32(input, input.stride(0));

    const int64_t num_heads = input.size(1);
    const int64_t head_dim  = input.size(2);
    // max(1, ...): a batch of only empty sequences still writes its zero means.
    const int64_t num_chunks =
        at::ceil_div<int64_t>(std::max<int64_t>(varlen.max_seqlen, 1), kSegmentMeanChunkTokens);

    const uint32_t stride_seq_input = static_cast<uint32_t>(input.stride(0));
    const int64_t  stride_h_input   = input.stride(1);

    auto         input_dtype = input.scalar_type();
    cudaStream_t stream      = at::cuda::getCurrentCUDAStream();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
        DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
            dim3 grid(num_chunks, num_heads, varlen.batch_size);
            dim3 block(512);
            if (num_chunks == 1) {
                SegmentMeanKernel<HEAD_DIM, true, c_type>
                    <<<grid, block, 0, stream>>>(reinterpret_cast<c_type*>(input.data_ptr()),
                                                 reinterpret_cast<c_type*>(mean.data_ptr()),
                                                 nullptr,
                                                 stride_seq_input,
                                                 stride_h_input,
                                                 varlen.cu_seqlens);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
            else {
                // Scratch, not an output: freed with the call. at::empty is
                // safe because the finish kernel reads exactly the slots the
                // first pass wrote and nothing else.
                at::Tensor partial = at::empty({varlen.batch_size, num_heads, num_chunks, head_dim},
                                               input.options().dtype(at::kFloat));
                SegmentMeanKernel<HEAD_DIM, false, c_type>
                    <<<grid, block, 0, stream>>>(reinterpret_cast<c_type*>(input.data_ptr()),
                                                 nullptr,
                                                 partial.data_ptr<float>(),
                                                 stride_seq_input,
                                                 stride_h_input,
                                                 varlen.cu_seqlens);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                dim3 finish_grid(num_heads, varlen.batch_size);
                dim3 finish_block(HEAD_DIM);
                SegmentMeanFinishKernel<c_type>
                    <<<finish_grid, finish_block, 0, stream>>>(partial.data_ptr<float>(),
                                                               reinterpret_cast<c_type*>(mean.data_ptr()),
                                                               static_cast<uint32_t>(num_chunks),
                                                               varlen.cu_seqlens);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        });
    });
}
