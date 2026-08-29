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
#include <cstdint>
#include <torch/types.h>

#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)                                                                     \
    if (head_dim == 64) {                                                                                              \
        constexpr int HEAD_DIM = 64;                                                                                   \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else if (head_dim == 128) {                                                                                        \
        constexpr int HEAD_DIM = 128;                                                                                  \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else {                                                                                                             \
        TORCH_CHECK_VALUE(false, "Unsupported head dim: ", int(head_dim));                                             \
    }

#define DISPATCH_CAUSAL(is_causal, IS_CAUSAL, ...)                                                                     \
    if (is_causal == 1) {                                                                                              \
        constexpr bool IS_CAUSAL = true;                                                                               \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else if (is_causal == 0) {                                                                                         \
        constexpr bool IS_CAUSAL = false;                                                                              \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else {                                                                                                             \
        TORCH_CHECK_VALUE(false, "Unsupported causal mode: ", int(is_causal));                                         \
    }

#define DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, ...)                                                      \
    if (qk_quant_gran == 2) {                                                                                          \
        constexpr int QK_QUANT_GRAN = 2;                                                                               \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else if (qk_quant_gran == 3) {                                                                                     \
        constexpr int QK_QUANT_GRAN = 3;                                                                               \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else {                                                                                                             \
        TORCH_CHECK_VALUE(false, "Unsupported qk_quant_gran: ", int(qk_quant_gran));                                   \
    }

#define DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, ...)                                                               \
    if (return_lse == 1) {                                                                                             \
        constexpr bool RETURN_LSE = true;                                                                              \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else if (return_lse == 0) {                                                                                        \
        constexpr bool RETURN_LSE = false;                                                                             \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else {                                                                                                             \
        TORCH_CHECK_VALUE(false, "Unsupported causal mode: ", int(return_lse));                                        \
    }

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(pytorch_dtype, c_type, ...)                                               \
    if (pytorch_dtype == at::ScalarType::Half) {                                                                       \
        using c_type = half;                                                                                           \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else if (pytorch_dtype == at::ScalarType::BFloat16) {                                                              \
        using c_type = nv_bfloat16;                                                                                    \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else {                                                                                                             \
        TORCH_CHECK(false, __PRETTY_FUNCTION__, " failed to dispatch data type ", pytorch_dtype);                      \
    }

#define DISPATCH_BLOCK_SIZE(block_size, CTA_TOKENS, ...)                                                               \
    if (block_size == 64) {                                                                                            \
        constexpr int CTA_TOKENS = 64;                                                                                 \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else if (block_size == 128) {                                                                                      \
        constexpr int CTA_TOKENS = 128;                                                                                \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else {                                                                                                             \
        TORCH_CHECK_VALUE(false, "Unsupported block_size ", int(block_size));                                          \
    }

#define DISPATCH_WARP_BLOCK_SIZE(warp_block_size, WARP_TOKENS, ...)                                                    \
    if (warp_block_size == 16) {                                                                                       \
        constexpr int WARP_TOKENS = 16;                                                                                \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else if (warp_block_size == 32) {                                                                                  \
        constexpr int WARP_TOKENS = 32;                                                                                \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else {                                                                                                             \
        TORCH_CHECK_VALUE(false, "Unsupported warp_block_size ", int(warp_block_size));                                \
    }

// The per-thread K quantization warp owns row *pairs*, so its warp block is
// twice the Q one; 64 and 128 are the only values plan.cpp's fill_tiles()
// produces for warp_k. Kept separate from DISPATCH_WARP_BLOCK_SIZE so the
// per-warp Q path does not instantiate tiles it never launches.
#define DISPATCH_WARP_BLOCK_SIZE_K(warp_block_size, WARP_TOKENS, ...)                                                  \
    if (warp_block_size == 64) {                                                                                       \
        constexpr int WARP_TOKENS = 64;                                                                                \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else if (warp_block_size == 128) {                                                                                 \
        constexpr int WARP_TOKENS = 128;                                                                               \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else {                                                                                                             \
        TORCH_CHECK_VALUE(false, "Unsupported warp_block_size ", int(warp_block_size));                                \
    }
