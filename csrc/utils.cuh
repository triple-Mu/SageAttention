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
// torch/types.h provides torch::Tensor / torch::kInt8 / torch::IntArrayRef /
// TORCH_CHECK without pulling in pybind11 (torch/extension.h would, which
// conflicts with Py_LIMITED_API).
#include <torch/types.h>

#include <climits>
#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), "Tensor " #x " must be on CUDA")
#define CHECK_DTYPE(x, true_dtype)                                                                                     \
    TORCH_CHECK(x.dtype() == true_dtype, "Tensor " #x " must have dtype (" #true_dtype ")")
#define CHECK_DIMS(x, true_dim)                                                                                        \
    TORCH_CHECK(x.dim() == true_dim, "Tensor " #x " must have dimension number (" #true_dim ")")
// Loop-carried strides stay 32-bit inside the kernels (single-IMAD pointer
// bumps); their per-iteration coefficient never exceeds 256, so 2^24 is the
// safe bound. Block-level (batch/head) strides are 64-bit and unbounded.
#define CHECK_STRIDE_LOOP32(x, s)                                                                                      \
    TORCH_CHECK((int64_t)(s) >= 0 && (int64_t)(s) < (int64_t(1) << 24),                                                \
                "Tensor " #x " stride " #s " (",                                                                       \
                (int64_t)(s),                                                                                          \
                ") exceeds the 2^24 limit of the 32-bit in-kernel addressing path")
#define CHECK_LEN_I32(name, n)                                                                                         \
    TORCH_CHECK((int64_t)(n) >= 0 && (int64_t)(n) <= INT32_MAX, #name " (", (int64_t)(n), ") must fit in int32")
#define CHECK_SHAPE(x, ...)                                                                                            \
    TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), "Tensor " #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "Tensor " #x " must be contiguous")
#define CHECK_LASTDIM_CONTIGUOUS(x)                                                                                    \
    TORCH_CHECK(x.stride(-1) == 1, "Tensor " #x " must be contiguous at the last dimension")