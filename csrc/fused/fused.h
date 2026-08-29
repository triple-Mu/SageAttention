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

#include "quant_utils.cuh"

// The int8 launchers whose kernels the packed varlen layout shares take a
// trailing QuantVarlen; a default-constructed one (null cu_seqlens) is the
// dense behaviour, unchanged.

void quant_per_block_int8_cuda(
    torch::Tensor input, torch::Tensor output, torch::Tensor scale, float sm_scale, int block_size, int tensor_layout);

void quant_per_block_int8_cuda(torch::Tensor      input,
                               torch::Tensor      output,
                               torch::Tensor      scale,
                               int                block_size,
                               int                tensor_layout,
                               const QuantVarlen& varlen = {});

void quant_per_block_int8_fuse_sub_mean_cuda(torch::Tensor      input,
                                             torch::Tensor      mean,
                                             torch::Tensor      output,
                                             torch::Tensor      scale,
                                             int                block_size,
                                             int                tensor_layout,
                                             const QuantVarlen& varlen = {});

void quant_per_warp_int8_cuda(torch::Tensor      input,
                              torch::Tensor      output,
                              torch::Tensor      scale,
                              int                block_size,
                              int                warp_block_size,
                              int                tensor_layout,
                              const QuantVarlen& varlen = {});

void sub_mean_cuda(torch::Tensor input, torch::Tensor mean, torch::Tensor output, int tensor_layout);

// Per-sequence mean of a packed [total_tokens, heads, head_dim] input into a
// [batch_size, heads, head_dim] mean of the same dtype (float32 accumulation,
// one cast at the end, an empty sequence gets zeros): smooth_k's production
// side, feeding the fuse_sub_mean launchers above. Unlike them this one is
// varlen-only, so the QuantVarlen is not optional; its max_seqlen sizes the
// grid exactly like theirs.
void segment_mean_cuda(torch::Tensor input, torch::Tensor mean, const QuantVarlen& varlen);

// The transposed-value family takes the same trailing QuantVarlen; it
// additionally uses its pad_tokens (the padded V^T axis block size).
// num_tokens is the dense sequence length and is unused when packed - the
// kernel reads each sequence's own length out of cu_seqlens.

void transpose_pad_permute_cuda(torch::Tensor      input,
                                torch::Tensor      output,
                                int                tensor_layout,
                                const QuantVarlen& varlen = {});

void transpose_pad_cuda(torch::Tensor input, torch::Tensor output, int tensor_layout, const QuantVarlen& varlen = {});

void scale_fuse_quant_cuda(torch::Tensor      input,
                           torch::Tensor      output,
                           torch::Tensor      scale,
                           int                num_tokens,
                           float              scale_max,
                           int                tensor_layout,
                           const QuantVarlen& varlen = {});

void mean_scale_fuse_quant_cuda(torch::Tensor      input,
                                torch::Tensor      output,
                                torch::Tensor      mean,
                                torch::Tensor      scale,
                                int                num_tokens,
                                float              scale_max,
                                int                tensor_layout,
                                const QuantVarlen& varlen = {});

void quant_per_thread_int8_q_cuda(torch::Tensor      input,
                                  torch::Tensor      output,
                                  torch::Tensor      scale,
                                  int                block_size,
                                  int                warp_block_size,
                                  int                tensor_layout,
                                  const QuantVarlen& varlen = {});

void quant_per_thread_int8_k_cuda(torch::Tensor      input,
                                  torch::Tensor      output,
                                  torch::Tensor      scale,
                                  int                block_size,
                                  int                warp_block_size,
                                  int                tensor_layout,
                                  const QuantVarlen& varlen = {});

void quant_per_thread_int8_k_fuse_sub_mean_cuda(torch::Tensor      input,
                                                torch::Tensor      mean,
                                                torch::Tensor      output,
                                                torch::Tensor      scale,
                                                int                block_size,
                                                int                warp_block_size,
                                                int                tensor_layout,
                                                const QuantVarlen& varlen = {});