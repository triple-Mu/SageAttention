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
#include <torch/extension.h>

#include "../utils.cuh"

// Shared input/output/scale prelude for the int8 quantization launchers
// (fused.cu per-block / per-warp paths and quant_per_thread.cu per-thread
// paths). The CHECK_* macros stringify their arguments into the error
// message, so the parameter names below must not be renamed.

struct QuantLayout
{
  int batch_size;
  int head_dim;
  int num_tokens;
  int num_heads;
  int stride_bz_input, stride_seq_input, stride_h_input;
  int stride_bz_output, stride_seq_output, stride_h_output;
};

// The CHECK_* sequence and tensor_layout-dependent sizes/strides common to
// every int8 quant launcher. `mean_opt`, when non-null, is checked at the
// positions quant_per_block_int8_fuse_sub_mean_cuda checked its mean tensor
// (CUDA / contiguous / dims; dtype is checked against the input dtype at the
// launcher).
inline QuantLayout parse_quant_layout(const torch::Tensor &input, const torch::Tensor &output,
                                      const torch::Tensor &scale, int tensor_layout,
                                      const torch::Tensor *mean_opt = nullptr)
{
  CHECK_CUDA(input);
  if (mean_opt)
  {
    const torch::Tensor &mean = *mean_opt;
    CHECK_CUDA(mean);
  }
  CHECK_CUDA(output);
  CHECK_CUDA(scale);

  CHECK_DTYPE(output, torch::kInt8);
  CHECK_DTYPE(scale, torch::kFloat);

  CHECK_LASTDIM_CONTIGUOUS(input);
  if (mean_opt)
  {
    const torch::Tensor &mean = *mean_opt;
    CHECK_CONTIGUOUS(mean);
  }
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  if (mean_opt)
  {
    const torch::Tensor &mean = *mean_opt;
    CHECK_DIMS(mean, 3);
  }
  CHECK_DIMS(output, 4);
  CHECK_DIMS(scale, 3);

  QuantLayout l;
  l.batch_size = input.size(0);
  l.head_dim = input.size(3);
  l.stride_bz_input = input.stride(0);
  l.stride_bz_output = output.stride(0);

  if (tensor_layout == 0)
  {
    l.num_tokens = input.size(1);
    l.num_heads = input.size(2);
    l.stride_seq_input = input.stride(1);
    l.stride_h_input = input.stride(2);
    l.stride_seq_output = output.stride(1);
    l.stride_h_output = output.stride(2);
  }
  else
  {
    l.num_tokens = input.size(2);
    l.num_heads = input.size(1);
    l.stride_seq_input = input.stride(2);
    l.stride_h_input = input.stride(1);
    l.stride_seq_output = output.stride(2);
    l.stride_h_output = output.stride(1);
  }

  return l;
}

// The per-thread quant launchers additionally front-load the output shape
// check and the head_dim divisibility check (fused.cu keeps its equivalent
// CHECK_SHAPE inside the dispatch macros, preserving its error ordering).
inline QuantLayout check_quant_layout(const torch::Tensor &input, const torch::Tensor &output,
                                      const torch::Tensor &scale, int tensor_layout)
{
  QuantLayout l = parse_quant_layout(input, output, scale, tensor_layout);

  CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
  TORCH_CHECK(l.head_dim % 32 == 0, "head_dim must be a multiple of 32");
  return l;
}

// Rebind the parsed layout to the local names fused.cu's DISPATCH bodies were
// written against (CHECK_SHAPE stringifies these spellings into its error
// messages).
#define SAGEATTN_QUANT_LAYOUT_LOCALS(L)                 \
  const int batch_size = (L).batch_size;                \
  const int head_dim = (L).head_dim;                    \
  const int num_tokens = (L).num_tokens;                \
  const int num_heads = (L).num_heads;                  \
  const int stride_bz_input = (L).stride_bz_input;      \
  const int stride_seq_input = (L).stride_seq_input;    \
  const int stride_h_input = (L).stride_h_input;        \
  const int stride_bz_output = (L).stride_bz_output;    \
  const int stride_seq_output = (L).stride_seq_output;  \
  const int stride_h_output = (L).stride_h_output
