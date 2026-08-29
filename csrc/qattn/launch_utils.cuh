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
#include "../sageattn/launch_helpers.cuh"
#include "../sageattn/varlen.h"
#include "../sageattn/varlen_check.h"
#include "../utils.cuh"

// Shared tensor-check / layout prelude for the qk_int* attention launchers.
//
// Every launcher historically repeated the same ~90-line prelude: the CHECK_*
// sequence over query/key/value/output and the scale tensors, the
// tensor_layout-dependent sizes and strides, the GQA divisibility error and
// the lse allocation. The prelude comes in three family flavors, selected by
// QKVFamily below; the optional value_scale / value_mean tensors are passed
// as pointers (nullptr when the variant does not take them) and are checked
// at the exact positions the original launchers checked them.
//
// The CHECK_* macros stringify their arguments into the error message, so the
// parameter and local variable names in this file must not be renamed: they
// are part of the user-visible error strings.

enum class QKVFamily {
    kSVF8CudaCore,  // sm89/sm120 fp8 mma path: query/key lastdim-contiguous,
                    // value fully contiguous ([B, D, H, padded_kv] transposed)
                    // with unchecked dtype (fp8; see the historical TODO below)
    kSVF8TMA,       // sm90/sm100 fp8 TMA path: value lastdim-contiguous
                    // ([B, D, H, padded_kv] transposed), dtype Float8_e4m3fn
    kSVF16,         // sm80 fp16 path: query/key fully contiguous, value
                    // lastdim-contiguous with the same [B, N, H, D] /
                    // [B, H, N, D] layout as key, dtype half
};

// All fields are int64_t (no narrowing from Tensor::size()/stride()). The
// width contract is applied where the values are consumed: batch/head level
// strides stay 64-bit all the way into the kernels (used once for the base
// offset), loop-carried strides are re-narrowed to uint32_t by the LOCALS
// macros below after qkv_layout_parse() has bounds-checked them.
struct QKVLayout {
    int64_t       batch_size;
    int64_t       head_dim;
    int64_t       qo_len, kv_len;
    int64_t       num_qo_heads, num_kv_heads, qo_per_kv_head;
    int64_t       stride_batch_q, stride_batch_k, stride_batch_v, stride_batch_o;
    int64_t       stride_seq_q, stride_h_q;
    int64_t       stride_seq_k, stride_h_k;
    int64_t       stride_h_v;
    int64_t       stride_d_v;    // transposed-value families (kSVF8CudaCore / kSVF8TMA)
    int64_t       stride_seq_v;  // seq-major-value family (kSVF16)
    int64_t       stride_seq_o, stride_h_o;
    torch::Tensor lse;  // empty unless return_lse
};

template<QKVFamily kFamily>
inline QKVLayout qkv_layout_parse(const torch::Tensor& query,
                                  const torch::Tensor& key,
                                  const torch::Tensor& value,
                                  const torch::Tensor& output,
                                  const torch::Tensor& query_scale,
                                  const torch::Tensor& key_scale,
                                  const torch::Tensor* value_scale_opt,
                                  const torch::Tensor* value_mean_opt,
                                  int                  tensor_layout,
                                  int                  return_lse)
{
    CHECK_CUDA(query);
    CHECK_CUDA(key);
    CHECK_CUDA(value);
    CHECK_CUDA(output);
    CHECK_CUDA(query_scale);
    CHECK_CUDA(key_scale);
    if (value_scale_opt) {
        const torch::Tensor& value_scale = *value_scale_opt;
        CHECK_CUDA(value_scale);
    }
    if (value_mean_opt) {
        const torch::Tensor& value_mean = *value_mean_opt;
        CHECK_CUDA(value_mean);
    }

    if constexpr (kFamily == QKVFamily::kSVF16) {
        CHECK_CONTIGUOUS(query);
        CHECK_CONTIGUOUS(key);
    }
    else {
        CHECK_LASTDIM_CONTIGUOUS(query);
        CHECK_LASTDIM_CONTIGUOUS(key);
    }
    if constexpr (kFamily == QKVFamily::kSVF8CudaCore) {
        CHECK_CONTIGUOUS(value);  // ensure value is contiguous to prevent troubles in the kernel
    }
    else {
        CHECK_LASTDIM_CONTIGUOUS(value);
    }
    CHECK_LASTDIM_CONTIGUOUS(output);
    CHECK_CONTIGUOUS(query_scale);
    CHECK_CONTIGUOUS(key_scale);
    if (value_scale_opt) {
        const torch::Tensor& value_scale = *value_scale_opt;
        CHECK_CONTIGUOUS(value_scale);
    }
    if (value_mean_opt) {
        const torch::Tensor& value_mean = *value_mean_opt;
        CHECK_CONTIGUOUS(value_mean);
    }

    CHECK_DTYPE(query, torch::kInt8);
    CHECK_DTYPE(key, torch::kInt8);
    if constexpr (kFamily == QKVFamily::kSVF8TMA) {
        CHECK_DTYPE(value, at::ScalarType::Float8_e4m3fn);
    }
    else if constexpr (kFamily == QKVFamily::kSVF16) {
        CHECK_DTYPE(value, torch::kHalf);
    }
    else {
        // TODO: how to check fp8 data type?
        // CHECK_DTYPE(value, torch::kHalf);
    }
    CHECK_DTYPE(query_scale, torch::kFloat32);
    CHECK_DTYPE(key_scale, torch::kFloat32);
    if (value_scale_opt) {
        const torch::Tensor& value_scale = *value_scale_opt;
        CHECK_DTYPE(value_scale, torch::kFloat32);
    }
    if (value_mean_opt) {
        if constexpr (kFamily == QKVFamily::kSVF8CudaCore) {
            const torch::Tensor& value_mean = *value_mean_opt;
            CHECK_DTYPE(value_mean, torch::kFloat32);
        }
        // kSVF16 instead checks value_mean's dtype against the output dtype at
        // the launcher (TORCH_CHECK(value_mean_dtype == out_dtype, ...)).
    }

    CHECK_DIMS(query, 4);
    CHECK_DIMS(key, 4);
    CHECK_DIMS(value, 4);
    CHECK_DIMS(output, 4);
    CHECK_DIMS(query_scale, 3);
    CHECK_DIMS(key_scale, 3);
    if (value_scale_opt) {
        const torch::Tensor& value_scale = *value_scale_opt;
        CHECK_DIMS(value_scale, 3);
    }
    if (value_mean_opt) {
        const torch::Tensor& value_mean = *value_mean_opt;
        CHECK_DIMS(value_mean, 3);
    }

    const int64_t batch_size = query.size(0);
    const int64_t head_dim   = query.size(3);

    QKVLayout layout;
    layout.batch_size     = batch_size;
    layout.head_dim       = head_dim;
    layout.stride_batch_q = query.stride(0);
    layout.stride_batch_k = key.stride(0);
    layout.stride_batch_v = value.stride(0);
    layout.stride_batch_o = output.stride(0);
    layout.stride_d_v     = 0;
    layout.stride_seq_v   = 0;

    int64_t qo_len, kv_len, num_qo_heads, num_kv_heads;

    if constexpr (kFamily != QKVFamily::kSVF16) {
        // value is [B, D, H, padded_kv] transposed; its padded last dim is
        // workload-dependent, so the fixed dims are TORCH_CHECKed individually
        // (plain assert() disappears under -DNDEBUG release builds).
        if constexpr (kFamily == QKVFamily::kSVF8TMA) {
            TORCH_CHECK(value.size(0) == batch_size,
                        "value batch dim (",
                        value.size(0),
                        ") must match query batch dim (",
                        batch_size,
                        ")");
        }

        if (tensor_layout == 0) {
            qo_len       = query.size(1);
            kv_len       = key.size(1);
            num_qo_heads = query.size(2);
            num_kv_heads = key.size(2);

            layout.stride_seq_q = query.stride(1);
            layout.stride_h_q   = query.stride(2);
            layout.stride_seq_k = key.stride(1);
            layout.stride_h_k   = key.stride(2);
            layout.stride_h_v   = value.stride(2);
            layout.stride_d_v   = value.stride(1);
            layout.stride_seq_o = output.stride(1);
            layout.stride_h_o   = output.stride(2);

            CHECK_SHAPE(key, batch_size, kv_len, num_kv_heads, head_dim);
            CHECK_SHAPE(output, batch_size, qo_len, num_qo_heads, head_dim);
            TORCH_CHECK(value.size(1) == head_dim,
                        "transposed value dim 1 (",
                        value.size(1),
                        ") must be head_dim (",
                        head_dim,
                        ")");
            TORCH_CHECK(value.size(2) == num_kv_heads,
                        "transposed value dim 2 (",
                        value.size(2),
                        ") must be num_kv_heads (",
                        num_kv_heads,
                        ")");
        }
        else if (tensor_layout == 1) {
            qo_len       = query.size(2);
            kv_len       = key.size(2);
            num_qo_heads = query.size(1);
            num_kv_heads = key.size(1);

            layout.stride_seq_q = query.stride(2);
            layout.stride_h_q   = query.stride(1);
            layout.stride_seq_k = key.stride(2);
            layout.stride_h_k   = key.stride(1);
            layout.stride_h_v   = value.stride(1);
            layout.stride_d_v   = value.stride(2);
            layout.stride_seq_o = output.stride(2);
            layout.stride_h_o   = output.stride(1);

            CHECK_SHAPE(key, batch_size, num_kv_heads, kv_len, head_dim);
            CHECK_SHAPE(output, batch_size, num_qo_heads, qo_len, head_dim);
            TORCH_CHECK(value.size(2) == head_dim,
                        "transposed value dim 2 (",
                        value.size(2),
                        ") must be head_dim (",
                        head_dim,
                        ")");
            TORCH_CHECK(value.size(1) == num_kv_heads,
                        "transposed value dim 1 (",
                        value.size(1),
                        ") must be num_kv_heads (",
                        num_kv_heads,
                        ")");
        }
        else {
            TORCH_CHECK_VALUE(false, "tensor_layout must be 0 or 1");
        }
    }
    else {
        // tensor_layout 0 for [B, N, H, D], 1 for [B, H, N, D]
        if (tensor_layout == 0) {
            qo_len       = query.size(1);
            kv_len       = key.size(1);
            num_qo_heads = query.size(2);
            num_kv_heads = key.size(2);
            CHECK_SHAPE(key, batch_size, kv_len, num_kv_heads, head_dim);
            CHECK_SHAPE(value, batch_size, kv_len, num_kv_heads, head_dim);

            layout.stride_seq_q = query.stride(1);
            layout.stride_seq_k = key.stride(1);
            layout.stride_seq_v = value.stride(1);
            layout.stride_seq_o = output.stride(1);

            layout.stride_h_q = query.stride(2);
            layout.stride_h_k = key.stride(2);
            layout.stride_h_v = value.stride(2);
            layout.stride_h_o = output.stride(2);
        }
        else if (tensor_layout == 1) {
            qo_len       = query.size(2);
            kv_len       = key.size(2);
            num_qo_heads = query.size(1);
            num_kv_heads = key.size(1);
            CHECK_SHAPE(key, batch_size, num_kv_heads, kv_len, head_dim);
            CHECK_SHAPE(value, batch_size, num_kv_heads, kv_len, head_dim);

            layout.stride_seq_q = query.stride(2);
            layout.stride_seq_k = key.stride(2);
            layout.stride_seq_v = value.stride(2);
            layout.stride_seq_o = output.stride(2);

            layout.stride_h_q = query.stride(1);
            layout.stride_h_k = key.stride(1);
            layout.stride_h_v = value.stride(1);
            layout.stride_h_o = output.stride(1);
        }
        else {
            TORCH_CHECK_VALUE(false, "tensor_layout must be 0 or 1");
        }
    }

    TORCH_CHECK_VALUE(num_qo_heads % num_kv_heads == 0,
                      "num_qo_heads (",
                      num_qo_heads,
                      ") must be divisible by num_kv_heads (",
                      num_kv_heads,
                      ")");

    layout.qo_len         = qo_len;
    layout.kv_len         = kv_len;
    layout.num_qo_heads   = num_qo_heads;
    layout.num_kv_heads   = num_kv_heads;
    layout.qo_per_kv_head = num_qo_heads / num_kv_heads;

    // Width contract (see QKVLayout): sequence lengths index in 32-bit, and the
    // loop-carried strides feed single-IMAD 32-bit pointer bumps inside the
    // kernels. Batch/head strides are 64-bit and need no bound.
    CHECK_LEN_I32(qo_len, qo_len);
    CHECK_LEN_I32(kv_len, kv_len);
    CHECK_STRIDE_LOOP32(query, layout.stride_seq_q);
    CHECK_STRIDE_LOOP32(key, layout.stride_seq_k);
    CHECK_STRIDE_LOOP32(output, layout.stride_seq_o);
    if constexpr (kFamily == QKVFamily::kSVF16) {
        CHECK_STRIDE_LOOP32(value, layout.stride_seq_v);
    }
    else {
        CHECK_STRIDE_LOOP32(value, layout.stride_d_v);
    }

    layout.lse = torch::empty({0});
    if (return_lse) {
        layout.lse = torch::empty({batch_size, num_qo_heads, qo_len}, query.options().dtype(torch::kFloat32));
    }

    return layout;
}

// ------------------------------------------------------------------ varlen
// The packed [total_tokens, heads, head_dim] counterpart of the above. It is a
// separate parse rather than a flag on qkv_layout_parse: the ranks differ, the
// scale tensors lose their batch dimension, and the dense error strings are
// pinned by tests. Only the fp16 (seq-major value) family exists so far - the
// transposed-value families get their value branch with their varlen TUs.
struct QKVVarlenLayout {
    int64_t       batch_size;
    int64_t       head_dim;
    int64_t       total_q, total_k;
    int64_t       max_seqlen_q, max_seqlen_k;
    int64_t       num_qo_heads, num_kv_heads, qo_per_kv_head;
    int64_t       stride_seq_q, stride_h_q;
    int64_t       stride_seq_k, stride_h_k;
    int64_t       stride_seq_v, stride_h_v;
    int64_t       stride_seq_o, stride_h_o;
    int64_t       q_scale_stride_h, k_scale_stride_h;
    torch::Tensor lse;  // empty unless return_lse
};

inline QKVVarlenLayout qkv_varlen_layout_parse(const torch::Tensor& query,
                                               const torch::Tensor& key,
                                               const torch::Tensor& value,
                                               const torch::Tensor& output,
                                               const torch::Tensor& query_scale,
                                               const torch::Tensor& key_scale,
                                               const torch::Tensor& cu_seqlens_q,
                                               const torch::Tensor& cu_seqlens_k,
                                               int64_t              max_seqlen_q,
                                               int64_t              max_seqlen_k,
                                               int                  return_lse)
{
    CHECK_CUDA(query);
    CHECK_CUDA(key);
    CHECK_CUDA(value);
    CHECK_CUDA(output);
    CHECK_CUDA(query_scale);
    CHECK_CUDA(key_scale);

    CHECK_CONTIGUOUS(query);
    CHECK_CONTIGUOUS(key);
    CHECK_LASTDIM_CONTIGUOUS(value);
    CHECK_LASTDIM_CONTIGUOUS(output);
    CHECK_CONTIGUOUS(query_scale);
    CHECK_CONTIGUOUS(key_scale);

    CHECK_DTYPE(query, torch::kInt8);
    CHECK_DTYPE(key, torch::kInt8);
    CHECK_DTYPE(value, torch::kHalf);
    CHECK_DTYPE(query_scale, torch::kFloat32);
    CHECK_DTYPE(key_scale, torch::kFloat32);

    CHECK_DIMS(query, 3);
    CHECK_DIMS(key, 3);
    CHECK_DIMS(value, 3);
    CHECK_DIMS(output, 3);
    CHECK_DIMS(query_scale, 2);
    CHECK_DIMS(key_scale, 2);

    sage::check_cu_seqlens(cu_seqlens_q, query, "cu_seqlens_q");
    sage::check_cu_seqlens(cu_seqlens_k, key, "cu_seqlens_k");
    TORCH_CHECK(cu_seqlens_q.size(0) == cu_seqlens_k.size(0),
                "cu_seqlens_q and cu_seqlens_k must describe the same batch, got lengths ",
                cu_seqlens_q.size(0),
                " and ",
                cu_seqlens_k.size(0));

    QKVVarlenLayout layout;
    layout.batch_size   = cu_seqlens_q.size(0) - 1;
    layout.head_dim     = query.size(2);
    layout.total_q      = query.size(0);
    layout.total_k      = key.size(0);
    layout.max_seqlen_q = max_seqlen_q;
    layout.max_seqlen_k = max_seqlen_k;
    layout.num_qo_heads = query.size(1);
    layout.num_kv_heads = key.size(1);

    const int64_t total_q      = layout.total_q;
    const int64_t total_k      = layout.total_k;
    const int64_t head_dim     = layout.head_dim;
    const int64_t num_qo_heads = layout.num_qo_heads;
    const int64_t num_kv_heads = layout.num_kv_heads;

    CHECK_SHAPE(key, total_k, num_kv_heads, head_dim);
    CHECK_SHAPE(value, total_k, num_kv_heads, head_dim);
    CHECK_SHAPE(output, total_q, num_qo_heads, head_dim);

    TORCH_CHECK_VALUE(num_qo_heads % num_kv_heads == 0,
                      "num_qo_heads (",
                      num_qo_heads,
                      ") must be divisible by num_kv_heads (",
                      num_kv_heads,
                      ")");
    layout.qo_per_kv_head = num_qo_heads / num_kv_heads;

    layout.stride_seq_q = query.stride(0);
    layout.stride_h_q   = query.stride(1);
    layout.stride_seq_k = key.stride(0);
    layout.stride_h_k   = key.stride(1);
    layout.stride_seq_v = value.stride(0);
    layout.stride_h_v   = value.stride(1);
    layout.stride_seq_o = output.stride(0);
    layout.stride_h_o   = output.stride(1);

    TORCH_CHECK(query_scale.size(0) == num_qo_heads,
                "query_scale must be [num_qo_heads, blocks], got dim 0 = ",
                query_scale.size(0));
    TORCH_CHECK(
        key_scale.size(0) == num_kv_heads, "key_scale must be [num_kv_heads, blocks], got dim 0 = ", key_scale.size(0));
    layout.q_scale_stride_h = query_scale.stride(0);
    layout.k_scale_stride_h = key_scale.stride(0);

    // Same width contract as the dense parse; total_tokens is the quantity
    // that has to fit in int32 here, since it is what the kernels index with.
    CHECK_LEN_I32(total_q, total_q);
    CHECK_LEN_I32(total_k, total_k);
    CHECK_LEN_I32(max_seqlen_q, max_seqlen_q);
    CHECK_LEN_I32(max_seqlen_k, max_seqlen_k);
    CHECK_STRIDE_LOOP32(query, layout.stride_seq_q);
    CHECK_STRIDE_LOOP32(key, layout.stride_seq_k);
    CHECK_STRIDE_LOOP32(value, layout.stride_seq_v);
    CHECK_STRIDE_LOOP32(output, layout.stride_seq_o);
    CHECK_STRIDE_LOOP32(query_scale, layout.q_scale_stride_h);
    CHECK_STRIDE_LOOP32(key_scale, layout.k_scale_stride_h);

    // head-major, one entry per packed token (documented on sageattn_varlen:
    // this is not flash-attention's [total_q, heads])
    layout.lse = torch::empty({0});
    if (return_lse) {
        layout.lse = torch::empty({num_qo_heads, total_q}, query.options().dtype(torch::kFloat32));
    }

    return layout;
}

// Rebind the parsed layout to the local names the DISPATCH bodies were
// written against. CHECK_SHAPE stringifies its arguments into the error
// message, so the spellings below must stay identical to the original
// launcher locals.
// Width contract: batch/head strides and lengths stay int64_t (consumed once
// per kernel for the 64-bit base offset), loop-carried strides re-narrow to
// uint32_t (bounds-checked in qkv_layout_parse above).
#define SAGEATTN_QKV_LAYOUT_LOCALS_COMMON(L)                                                                           \
    const int64_t  batch_size     = (L).batch_size;                                                                    \
    const int64_t  head_dim       = (L).head_dim;                                                                      \
    const int64_t  qo_len         = (L).qo_len;                                                                        \
    const int64_t  kv_len         = (L).kv_len;                                                                        \
    const int64_t  num_qo_heads   = (L).num_qo_heads;                                                                  \
    const int64_t  num_kv_heads   = (L).num_kv_heads;                                                                  \
    const int64_t  qo_per_kv_head = (L).qo_per_kv_head;                                                                \
    const int64_t  stride_batch_q = (L).stride_batch_q;                                                                \
    const int64_t  stride_batch_k = (L).stride_batch_k;                                                                \
    const int64_t  stride_batch_v = (L).stride_batch_v;                                                                \
    const int64_t  stride_batch_o = (L).stride_batch_o;                                                                \
    const uint32_t stride_seq_q   = static_cast<uint32_t>((L).stride_seq_q);                                           \
    const int64_t  stride_h_q     = (L).stride_h_q;                                                                    \
    const uint32_t stride_seq_k   = static_cast<uint32_t>((L).stride_seq_k);                                           \
    const int64_t  stride_h_k     = (L).stride_h_k;                                                                    \
    const int64_t  stride_h_v     = (L).stride_h_v;                                                                    \
    const uint32_t stride_seq_o   = static_cast<uint32_t>((L).stride_seq_o);                                           \
    const int64_t  stride_h_o     = (L).stride_h_o;                                                                    \
    torch::Tensor  lse            = (L).lse

// Transposed-value (fp8) launchers additionally use stride_d_v.
#define SAGEATTN_QKV_LAYOUT_LOCALS_FP8(L)                                                                              \
    SAGEATTN_QKV_LAYOUT_LOCALS_COMMON(L);                                                                              \
    const uint32_t stride_d_v = static_cast<uint32_t>((L).stride_d_v)

// Seq-major-value (fp16) launchers additionally use stride_seq_v.
#define SAGEATTN_QKV_LAYOUT_LOCALS_F16(L)                                                                              \
    SAGEATTN_QKV_LAYOUT_LOCALS_COMMON(L);                                                                              \
    const uint32_t stride_seq_v = static_cast<uint32_t>((L).stride_seq_v)

// The varlen counterpart. Same width contract: the per-head strides stay
// 64-bit (used once for the base offset), the loop-carried token strides
// re-narrow to uint32_t.
#define SAGEATTN_QKV_VARLEN_LOCALS_F16(L)                                                                              \
    const int64_t  batch_size       = (L).batch_size;                                                                  \
    const int64_t  head_dim         = (L).head_dim;                                                                    \
    const int64_t  total_q          = (L).total_q;                                                                     \
    const int64_t  total_k          = (L).total_k;                                                                     \
    const int64_t  max_seqlen_q     = (L).max_seqlen_q;                                                                \
    const int64_t  max_seqlen_k     = (L).max_seqlen_k;                                                                \
    const int64_t  num_qo_heads     = (L).num_qo_heads;                                                                \
    const int64_t  num_kv_heads     = (L).num_kv_heads;                                                                \
    const int64_t  qo_per_kv_head   = (L).qo_per_kv_head;                                                              \
    const uint32_t stride_seq_q     = static_cast<uint32_t>((L).stride_seq_q);                                         \
    const int64_t  stride_h_q       = (L).stride_h_q;                                                                  \
    const uint32_t stride_seq_k     = static_cast<uint32_t>((L).stride_seq_k);                                         \
    const int64_t  stride_h_k       = (L).stride_h_k;                                                                  \
    const uint32_t stride_seq_v     = static_cast<uint32_t>((L).stride_seq_v);                                         \
    const int64_t  stride_h_v       = (L).stride_h_v;                                                                  \
    const uint32_t stride_seq_o     = static_cast<uint32_t>((L).stride_seq_o);                                         \
    const int64_t  stride_h_o       = (L).stride_h_o;                                                                  \
    const uint32_t q_scale_stride_h = static_cast<uint32_t>((L).q_scale_stride_h);                                     \
    const uint32_t k_scale_stride_h = static_cast<uint32_t>((L).k_scale_stride_h);                                     \
    torch::Tensor  lse              = (L).lse
