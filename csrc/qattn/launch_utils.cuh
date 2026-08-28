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
#include "../utils.cuh"
#include <torch/extension.h>

#include <cassert>
#include <sstream>
#include <stdexcept>

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

struct QKVLayout {
  int batch_size;
  int head_dim;
  int qo_len, kv_len;
  int num_qo_heads, num_kv_heads, num_kv_groups;
  int stride_bz_q, stride_bz_k, stride_bz_v, stride_bz_o;
  int stride_seq_q, stride_h_q;
  int stride_seq_k, stride_h_k;
  int stride_h_v;
  int stride_d_v;    // transposed-value families (kSVF8CudaCore / kSVF8TMA)
  int stride_seq_v;  // seq-major-value family (kSVF16)
  int stride_seq_o, stride_h_o;
  torch::Tensor lse;  // empty unless return_lse
};

template <QKVFamily kFamily>
inline QKVLayout qkv_layout_parse(const torch::Tensor &query, const torch::Tensor &key,
                                  const torch::Tensor &value, const torch::Tensor &output,
                                  const torch::Tensor &query_scale, const torch::Tensor &key_scale,
                                  const torch::Tensor *value_scale_opt,
                                  const torch::Tensor *value_mean_opt,
                                  int tensor_layout, int return_lse)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);
  if (value_scale_opt)
  {
    const torch::Tensor &value_scale = *value_scale_opt;
    CHECK_CUDA(value_scale);
  }
  if (value_mean_opt)
  {
    const torch::Tensor &value_mean = *value_mean_opt;
    CHECK_CUDA(value_mean);
  }

  if constexpr (kFamily == QKVFamily::kSVF16)
  {
    CHECK_CONTIGUOUS(query);
    CHECK_CONTIGUOUS(key);
  }
  else
  {
    CHECK_LASTDIM_CONTIGUOUS(query);
    CHECK_LASTDIM_CONTIGUOUS(key);
  }
  if constexpr (kFamily == QKVFamily::kSVF8CudaCore)
  {
    CHECK_CONTIGUOUS(value); // ensure value is contiguous to prevent troubles in the kernel
  }
  else
  {
    CHECK_LASTDIM_CONTIGUOUS(value);
  }
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);
  if (value_scale_opt)
  {
    const torch::Tensor &value_scale = *value_scale_opt;
    CHECK_CONTIGUOUS(value_scale);
  }
  if (value_mean_opt)
  {
    const torch::Tensor &value_mean = *value_mean_opt;
    CHECK_CONTIGUOUS(value_mean);
  }

  CHECK_DTYPE(query, torch::kInt8);
  CHECK_DTYPE(key, torch::kInt8);
  if constexpr (kFamily == QKVFamily::kSVF8TMA)
  {
    CHECK_DTYPE(value, at::ScalarType::Float8_e4m3fn);
  }
  else if constexpr (kFamily == QKVFamily::kSVF16)
  {
    CHECK_DTYPE(value, torch::kHalf);
  }
  else
  {
    // TODO: how to check fp8 data type?
    // CHECK_DTYPE(value, torch::kHalf);
  }
  CHECK_DTYPE(query_scale, torch::kFloat32);
  CHECK_DTYPE(key_scale, torch::kFloat32);
  if (value_scale_opt)
  {
    const torch::Tensor &value_scale = *value_scale_opt;
    CHECK_DTYPE(value_scale, torch::kFloat32);
  }
  if (value_mean_opt)
  {
    if constexpr (kFamily == QKVFamily::kSVF8CudaCore)
    {
      const torch::Tensor &value_mean = *value_mean_opt;
      CHECK_DTYPE(value_mean, torch::kFloat32);
    }
    // kSVF16 instead checks value_mean's dtype against the output dtype at
    // the launcher (TORCH_CHECK(value_mean_dtype == output_dtype, ...)).
  }

  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);
  if (value_scale_opt)
  {
    const torch::Tensor &value_scale = *value_scale_opt;
    CHECK_DIMS(value_scale, 3);
  }
  if (value_mean_opt)
  {
    const torch::Tensor &value_mean = *value_mean_opt;
    CHECK_DIMS(value_mean, 3);
  }

  const int batch_size = query.size(0);
  const int head_dim = query.size(3);

  QKVLayout layout;
  layout.batch_size = batch_size;
  layout.head_dim = head_dim;
  layout.stride_bz_q = query.stride(0);
  layout.stride_bz_k = key.stride(0);
  layout.stride_bz_v = value.stride(0);
  layout.stride_bz_o = output.stride(0);
  layout.stride_d_v = 0;
  layout.stride_seq_v = 0;

  int qo_len, kv_len, num_qo_heads, num_kv_heads;

  if constexpr (kFamily != QKVFamily::kSVF16)
  {
    // value is [B, D, H, padded_kv] transposed; its sizes are asserted rather
    // than CHECK_SHAPEd (the padded last dim is workload-dependent).
    if constexpr (kFamily == QKVFamily::kSVF8TMA)
    {
      assert(value.size(0) == batch_size);
    }

    if (tensor_layout == 0)
    {
      qo_len = query.size(1);
      kv_len = key.size(1);
      num_qo_heads = query.size(2);
      num_kv_heads = key.size(2);

      layout.stride_seq_q = query.stride(1);
      layout.stride_h_q = query.stride(2);
      layout.stride_seq_k = key.stride(1);
      layout.stride_h_k = key.stride(2);
      layout.stride_h_v = value.stride(2);
      layout.stride_d_v = value.stride(1);
      layout.stride_seq_o = output.stride(1);
      layout.stride_h_o = output.stride(2);

      CHECK_SHAPE(key, batch_size, kv_len, num_kv_heads, head_dim);
      CHECK_SHAPE(output, batch_size, qo_len, num_qo_heads, head_dim);
      assert(value.size(1) == head_dim);
      assert(value.size(2) == num_kv_heads);
    }
    else
    {
      qo_len = query.size(2);
      kv_len = key.size(2);
      num_qo_heads = query.size(1);
      num_kv_heads = key.size(1);

      layout.stride_seq_q = query.stride(2);
      layout.stride_h_q = query.stride(1);
      layout.stride_seq_k = key.stride(2);
      layout.stride_h_k = key.stride(1);
      layout.stride_h_v = value.stride(1);
      layout.stride_d_v = value.stride(2);
      layout.stride_seq_o = output.stride(2);
      layout.stride_h_o = output.stride(1);

      CHECK_SHAPE(key, batch_size, num_kv_heads, kv_len, head_dim);
      CHECK_SHAPE(output, batch_size, num_qo_heads, qo_len, head_dim);
      assert(value.size(2) == head_dim);
      assert(value.size(1) == num_kv_heads);
    }
  }
  else
  {
    // tensor_layout 0 for [B, N, H, D], 1 for [B, H, N, D]
    if (tensor_layout == 0)
    {
      qo_len = query.size(1);
      kv_len = key.size(1);
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
    else if (tensor_layout == 1)
    {
      qo_len = query.size(2);
      kv_len = key.size(2);
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
    else
    {
      throw std::invalid_argument("tensor_layout must be 0 or 1");
    }
  }

  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads (" << num_qo_heads << ") must be divisible by num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(err_msg.str());
  }

  layout.qo_len = qo_len;
  layout.kv_len = kv_len;
  layout.num_qo_heads = num_qo_heads;
  layout.num_kv_heads = num_kv_heads;
  layout.num_kv_groups = num_qo_heads / num_kv_heads;

  layout.lse = torch::empty({0});
  if (return_lse)
  {
    layout.lse = torch::empty({batch_size, num_qo_heads, qo_len}, query.options().dtype(torch::kFloat32));
  }

  return layout;
}

// Rebind the parsed layout to the local names the DISPATCH bodies were
// written against. CHECK_SHAPE stringifies its arguments into the error
// message, so the spellings below must stay identical to the original
// launcher locals.
#define SAGEATTN_QKV_LAYOUT_LOCALS_COMMON(L)   \
  const int batch_size = (L).batch_size;       \
  const int head_dim = (L).head_dim;           \
  const int qo_len = (L).qo_len;               \
  const int kv_len = (L).kv_len;               \
  const int num_qo_heads = (L).num_qo_heads;   \
  const int num_kv_heads = (L).num_kv_heads;   \
  const int num_kv_groups = (L).num_kv_groups; \
  const int stride_bz_q = (L).stride_bz_q;     \
  const int stride_bz_k = (L).stride_bz_k;     \
  const int stride_bz_v = (L).stride_bz_v;     \
  const int stride_bz_o = (L).stride_bz_o;     \
  const int stride_seq_q = (L).stride_seq_q;   \
  const int stride_h_q = (L).stride_h_q;       \
  const int stride_seq_k = (L).stride_seq_k;   \
  const int stride_h_k = (L).stride_h_k;       \
  const int stride_h_v = (L).stride_h_v;       \
  const int stride_seq_o = (L).stride_seq_o;   \
  const int stride_h_o = (L).stride_h_o;       \
  torch::Tensor lse = (L).lse

// Transposed-value (fp8) launchers additionally use stride_d_v.
#define SAGEATTN_QKV_LAYOUT_LOCALS_FP8(L) \
  SAGEATTN_QKV_LAYOUT_LOCALS_COMMON(L);   \
  const int stride_d_v = (L).stride_d_v

// Seq-major-value (fp16) launchers additionally use stride_seq_v.
#define SAGEATTN_QKV_LAYOUT_LOCALS_F16(L) \
  SAGEATTN_QKV_LAYOUT_LOCALS_COMMON(L);   \
  const int stride_seq_v = (L).stride_seq_v
