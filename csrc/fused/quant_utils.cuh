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
#include <torch/types.h>

#include "../sageattn/varlen.h"
#include "../utils.cuh"

// Shared input/output/scale prelude for the int8 quantization launchers
// (fused.cu per-block / per-warp paths and quant_per_thread.cu per-thread
// paths). The CHECK_* macros stringify their arguments into the error
// message, so the parameter names below must not be renamed.

// Fields are int64_t (no narrowing from size()/stride()); the LOCALS macro
// below re-applies the width contract: batch/head strides stay 64-bit (base
// offset, computed once), seq strides re-narrow to uint32_t (loop-carried,
// bounds-checked in parse_quant_layout).
struct QuantLayout {
    int64_t batch_size;
    int64_t head_dim;
    int64_t num_tokens;
    int64_t num_heads;
    int64_t stride_batch_input, stride_seq_input, stride_h_input;
    int64_t stride_batch_output, stride_seq_output, stride_h_output;
};

// The packed [total_tokens, heads, head_dim] varlen layout, passed to the
// quantization launchers as a trailing argument. A null cu_seqlens means
// dense, and every launcher then behaves exactly as it did before varlen
// existed (FA2 passes its varlen pointers the same way).
//
// The kernels are shared, not duplicated: the sequence's tokens are addressed
// relative to its own start, so an equal-length batch quantizes to bit-identical
// output either way. See csrc/sageattn/varlen.h for the addressing.
struct QuantVarlen {
    const int32_t* cu_seqlens = nullptr;  // [batch_size + 1] int32, device
    int64_t        batch_size = 0;
    int64_t        max_seqlen = 0;  // grid.x is opened to this and blocks past
                                    // their own sequence exit immediately
    int64_t pad_tokens = 0;         // transposed-value family only: the block
                                    // size the padded V^T axis is laid out
                                    // with (varlen.h pad_offset). The int8 QK
                                    // launchers take their block size as an
                                    // explicit argument instead.
};

// The CHECK_* sequence and tensor_layout-dependent sizes/strides common to
// every int8 quant launcher. `mean_opt`, when non-null, is checked at the
// positions quant_per_block_int8_fuse_sub_mean_cuda checked its mean tensor
// (CUDA / contiguous / dims; dtype is checked against the input dtype at the
// launcher).
inline QuantLayout parse_quant_layout(const torch::Tensor& input,
                                      const torch::Tensor& output,
                                      const torch::Tensor& scale,
                                      int                  tensor_layout,
                                      const torch::Tensor* mean_opt = nullptr)
{
    CHECK_CUDA(input);
    if (mean_opt) {
        const torch::Tensor& mean = *mean_opt;
        CHECK_CUDA(mean);
    }
    CHECK_CUDA(output);
    CHECK_CUDA(scale);

    CHECK_DTYPE(output, torch::kInt8);
    CHECK_DTYPE(scale, torch::kFloat);

    CHECK_LASTDIM_CONTIGUOUS(input);
    if (mean_opt) {
        const torch::Tensor& mean = *mean_opt;
        CHECK_CONTIGUOUS(mean);
    }
    CHECK_CONTIGUOUS(output);
    CHECK_CONTIGUOUS(scale);

    CHECK_DIMS(input, 4);
    if (mean_opt) {
        const torch::Tensor& mean = *mean_opt;
        CHECK_DIMS(mean, 3);
    }
    CHECK_DIMS(output, 4);
    CHECK_DIMS(scale, 3);

    QuantLayout l;
    l.batch_size          = input.size(0);
    l.head_dim            = input.size(3);
    l.stride_batch_input  = input.stride(0);
    l.stride_batch_output = output.stride(0);

    if (tensor_layout == 0) {
        l.num_tokens        = input.size(1);
        l.num_heads         = input.size(2);
        l.stride_seq_input  = input.stride(1);
        l.stride_h_input    = input.stride(2);
        l.stride_seq_output = output.stride(1);
        l.stride_h_output   = output.stride(2);
    }
    else {
        l.num_tokens        = input.size(2);
        l.num_heads         = input.size(1);
        l.stride_seq_input  = input.stride(2);
        l.stride_h_input    = input.stride(1);
        l.stride_seq_output = output.stride(2);
        l.stride_h_output   = output.stride(1);
    }

    CHECK_LEN_I32(num_tokens, l.num_tokens);
    CHECK_STRIDE_LOOP32(input, l.stride_seq_input);
    CHECK_STRIDE_LOOP32(output, l.stride_seq_output);

    return l;
}

// parse_quant_layout's varlen counterpart. The packed tensors are 3-D, so the
// batch stride is 0 and blockIdx.z indexes cu_seqlens; num_tokens carries
// max_seqlen, which sizes grid.x only (the kernel reads the sequence's own
// length out of cu_seqlens).
inline QuantLayout parse_quant_varlen_layout(const torch::Tensor& input,
                                             const torch::Tensor& output,
                                             const torch::Tensor& scale,
                                             const QuantVarlen&   varlen)
{
    CHECK_CUDA(input);
    CHECK_CUDA(output);
    CHECK_CUDA(scale);

    CHECK_DTYPE(output, torch::kInt8);
    CHECK_DTYPE(scale, torch::kFloat);

    CHECK_LASTDIM_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(output);
    CHECK_CONTIGUOUS(scale);

    CHECK_DIMS(input, 3);
    CHECK_DIMS(output, 3);
    CHECK_DIMS(scale, 2);
    CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2));

    QuantLayout l;
    l.batch_size          = varlen.batch_size;
    l.head_dim            = input.size(2);
    l.num_tokens          = varlen.max_seqlen;
    l.num_heads           = input.size(1);
    l.stride_batch_input  = 0;
    l.stride_seq_input    = input.stride(0);
    l.stride_h_input      = input.stride(1);
    l.stride_batch_output = 0;
    l.stride_seq_output   = output.stride(0);
    l.stride_h_output     = output.stride(1);

    CHECK_LEN_I32(num_tokens, l.num_tokens);
    CHECK_LEN_I32(total_tokens, input.size(0));
    CHECK_STRIDE_LOOP32(input, l.stride_seq_input);
    CHECK_STRIDE_LOOP32(output, l.stride_seq_output);

    return l;
}

// Layout prelude for the transposed-value family ([B, H, D, padded_n] (HND) /
// [B, D, H, padded_n] (NHD)): transpose_pad_v / scale_fuse_quant /
// mean_scale_fuse_quant. Here the per-CTA loops advance along the padded
// token axis with unit stride, and blockIdx.z walks head_dim, so stride_d is
// the loop-adjacent one to bound.
struct VTLayout {
    int64_t batch_size;
    int64_t head_dim;
    int64_t num_heads;
    int64_t padded_num_tokens;
    int64_t stride_batch, stride_d, stride_h;
};

inline VTLayout parse_vt_layout(const torch::Tensor& t, int tensor_layout)
{
    VTLayout l;
    l.batch_size        = t.size(0);
    l.padded_num_tokens = t.size(3);
    l.stride_batch      = t.stride(0);
    if (tensor_layout == 0) {
        l.head_dim  = t.size(1);
        l.num_heads = t.size(2);
        l.stride_d  = t.stride(1);
        l.stride_h  = t.stride(2);
    }
    else {
        l.num_heads = t.size(1);
        l.head_dim  = t.size(2);
        l.stride_h  = t.stride(1);
        l.stride_d  = t.stride(2);
    }
    CHECK_LEN_I32(padded_num_tokens, l.padded_num_tokens);
    return l;
}

// parse_vt_layout's varlen counterpart. The packed transposed value is
// [heads, head_dim, padded_total]: the batch dimension is gone, and a
// sequence's slab starts at varlen.h's pad_offset instead of a batch stride.
inline VTLayout parse_vt_varlen_layout(const torch::Tensor& t, const QuantVarlen& varlen)
{
    CHECK_DIMS(t, 3);

    VTLayout l;
    l.batch_size        = varlen.batch_size;
    l.num_heads         = t.size(0);
    l.head_dim          = t.size(1);
    l.padded_num_tokens = t.size(2);
    l.stride_batch      = 0;
    l.stride_h          = t.stride(0);
    l.stride_d          = t.stride(1);
    CHECK_LEN_I32(padded_num_tokens, l.padded_num_tokens);
    return l;
}

// The per-thread quant launchers additionally front-load the output shape
// check and the head_dim divisibility check (fused.cu keeps its equivalent
// CHECK_SHAPE inside the dispatch macros, preserving its error ordering).
inline QuantLayout check_quant_layout(const torch::Tensor& input,
                                      const torch::Tensor& output,
                                      const torch::Tensor& scale,
                                      int                  tensor_layout,
                                      const QuantVarlen&   varlen = {})
{
    if (varlen.cu_seqlens != nullptr) {
        QuantLayout l = parse_quant_varlen_layout(input, output, scale, varlen);
        TORCH_CHECK(l.head_dim % 32 == 0, "head_dim must be a multiple of 32");
        return l;
    }

    QuantLayout l = parse_quant_layout(input, output, scale, tensor_layout);

    CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
    TORCH_CHECK(l.head_dim % 32 == 0, "head_dim must be a multiple of 32");
    return l;
}

// Rebind the parsed layout to the local names fused.cu's DISPATCH bodies were
// written against (CHECK_SHAPE stringifies these spellings into its error
// messages).
// The scale tensor is [batch, heads, blocks] when dense and [heads, blocks]
// when packed, and the kernels take its two strides; varlen therefore has no
// batch stride and reads the head stride out of dim 0.
#define SAGEATTN_QUANT_SCALE_STRIDES(scale, is_varlen)                                                                 \
    const int64_t stride_batch_scale = (is_varlen) ? 0 : (scale).stride(0);                                            \
    const int64_t stride_h_scale     = (is_varlen) ? (scale).stride(0) : (scale).stride(1)

#define SAGEATTN_QUANT_LAYOUT_LOCALS(L)                                                                                \
    const int64_t  batch_size          = (L).batch_size;                                                               \
    const int64_t  head_dim            = (L).head_dim;                                                                 \
    const int64_t  num_tokens          = (L).num_tokens;                                                               \
    const int64_t  num_heads           = (L).num_heads;                                                                \
    const int64_t  stride_batch_input  = (L).stride_batch_input;                                                       \
    const uint32_t stride_seq_input    = static_cast<uint32_t>((L).stride_seq_input);                                  \
    const int64_t  stride_h_input      = (L).stride_h_input;                                                           \
    const int64_t  stride_batch_output = (L).stride_batch_output;                                                      \
    const uint32_t stride_seq_output   = static_cast<uint32_t>((L).stride_seq_output);                                 \
    const int64_t  stride_h_output     = (L).stride_h_output
