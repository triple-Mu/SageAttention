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

// Addressing for the flash-attention style varlen layout, where q/k/v are
// flattened to [total_tokens, heads, head_dim] and the sequence boundaries
// live in a cu_seqlens prefix sum of shape [batch_size + 1] (int32,
// cu_seqlens[0] == 0, cu_seqlens[batch_size] == total_tokens).
//
// Two coordinate systems are in play. Tokens address q / k / o / lse directly.
// Blocks address everything stored one entry per CTA-sized tile: the int8 QK
// scales, and the zero-padded V^T slabs the fp8 path swizzles. Both are closed
// forms over cu_seqlens - there is no second, host-computed prefix-sum tensor -
// and the quantization kernels, the attention kernels and the host-side shape
// checks all call these functions rather than re-deriving the arithmetic.
//
// Property 1: every sequence owns enough blocks for itself.
//   blk_offset(b + 1) - blk_offset(b) = cu[b+1]/C - cu[b]/C + 1
//   Write cu[b] = q*C + r and cu[b+1] = p*C + s with 0 <= r, s < C. Then
//   ceil(seq_len(b) / C) = (p - q) + [s > r] <= (p - q) + 1, which is exactly
//   that difference. So a sequence's blocks never spill into the next
//   sequence's range, and at most one block per sequence goes unused.
//
// Property 2: the total block count is a static shape.
//   blk_total(total, B, C) = total/C + B, and it equals blk_offset(cu, B, C)
//   whenever cu[B] == total. It depends on total_tokens and batch_size only,
//   never on how the tokens are split between sequences. So a cudagraph replay
//   that rewrites the *contents* of cu_seqlens cannot change any tensor shape,
//   and the fake kernels need no unbacked SymInt.

#if defined(__CUDACC__)
#define SAGE_VARLEN_HOST_DEVICE __host__ __device__ __forceinline__
#else
#define SAGE_VARLEN_HOST_DEVICE inline
#endif

namespace sage {

// First token of sequence batch_idx in the flattened [total_tokens, ...] q/k/v.
SAGE_VARLEN_HOST_DEVICE constexpr int32_t seq_offset(const int32_t* cu_seqlens, int32_t batch_idx)
{
    return cu_seqlens[batch_idx];
}

SAGE_VARLEN_HOST_DEVICE constexpr int32_t seq_len(const int32_t* cu_seqlens, int32_t batch_idx)
{
    return cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx];
}

// Index of sequence batch_idx's first cta_tokens-sized block (Property 1).
SAGE_VARLEN_HOST_DEVICE constexpr int32_t blk_offset(const int32_t* cu_seqlens, int32_t batch_idx, int32_t cta_tokens)
{
    return cu_seqlens[batch_idx] / cta_tokens + batch_idx;
}

// First token slot of sequence batch_idx in a block-padded buffer (the fp8
// V^T slabs): the block index above, expanded back to tokens.
SAGE_VARLEN_HOST_DEVICE constexpr int32_t pad_offset(const int32_t* cu_seqlens, int32_t batch_idx, int32_t cta_tokens)
{
    return blk_offset(cu_seqlens, batch_idx, cta_tokens) * cta_tokens;
}

// Blocks needed by the whole batch (Property 2). int64_t because it is
// multiplied by the per-block scale count on the way to a tensor size.
SAGE_VARLEN_HOST_DEVICE constexpr int64_t blk_total(int64_t total_tokens, int64_t batch_size, int64_t cta_tokens)
{
    return total_tokens / cta_tokens + batch_size;
}

// Both properties on one fixture, including an empty sequence and a segment
// length that is not a multiple of the block size. constexpr at namespace
// scope has internal linkage, so this costs nothing at runtime.
constexpr int32_t kVarlenCheckCu[4] = {0, 100, 100, 333};

static_assert(seq_len(kVarlenCheckCu, 1) == 0, "an empty sequence must have length 0");
static_assert(blk_offset(kVarlenCheckCu, 0, 64) == 0 && blk_offset(kVarlenCheckCu, 1, 64) == 2
                  && blk_offset(kVarlenCheckCu, 2, 64) == 3,
              "block bases must follow cu[b]/C + b");
static_assert(blk_offset(kVarlenCheckCu, 1, 64) - blk_offset(kVarlenCheckCu, 0, 64) >= (100 + 63) / 64,
              "Property 1: a sequence must own at least ceil(len / C) blocks");
static_assert(blk_offset(kVarlenCheckCu, 3, 64) - blk_offset(kVarlenCheckCu, 2, 64) >= (233 + 63) / 64,
              "Property 1: a sequence must own at least ceil(len / C) blocks");
static_assert(blk_offset(kVarlenCheckCu, 3, 64) == blk_total(333, 3, 64),
              "Property 2: the one-past-the-end block base is the static total");
static_assert(pad_offset(kVarlenCheckCu, 2, 64) == 3 * 64, "padded slabs start on a block boundary");

}  // namespace sage
