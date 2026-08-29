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

#include "varlen.h"

namespace sage {

// Everything a kernel needs to know about one sequence of the batch, in the
// two coordinate systems varlen.h defines. The dense and varlen kernels are
// separate translation units of the same body (see the *_impl.cuh headers), so
// kVarlen is fixed at compile time: on the dense side every offset below is
// the literal 0 and folds away, leaving qo_len / kv_len exactly where the
// pre-varlen code read them.
//
// This is the sage counterpart of flash-attention's BlockInfo (FA2
// csrc/flash_attn/src/block_info.h), minus leftpad_k and seqused_k, which no
// sage entry point exposes.
//
// delta is the bottom-right causal alignment FA uses: row q of a sequence
// attends to keys up to q + delta. It is signed on purpose - kv shorter than
// qo is legal and makes it negative, and the masked-tile arithmetic derived
// from it underflows if it is ever carried in an unsigned type.
template<bool kVarlen, int kBlockQ, int kBlockK>
struct SeqlenInfo {
    const int32_t offset_q, offset_k;      // this sequence's first q / k token
    const int32_t seqlen_q, seqlen_k;      // its length in tokens
    const int32_t blk_q_base, blk_k_base;  // its first scale block
    const int32_t delta;                   // seqlen_k - seqlen_q, may be negative

    // Dense: one sequence per batch entry, so token 0 is block 0 is offset 0.
    SAGE_VARLEN_HOST_DEVICE constexpr SeqlenInfo(int32_t qo_len, int32_t kv_len):
        offset_q(0),
        offset_k(0),
        seqlen_q(qo_len),
        seqlen_k(kv_len),
        blk_q_base(0),
        blk_k_base(0),
        delta(kv_len - qo_len)
    {
        static_assert(!kVarlen, "the (qo_len, kv_len) constructor is the dense one");
    }

    // Varlen: batch_idx is blockIdx.z, the cu_seqlens are [batch_size + 1].
    SAGE_VARLEN_HOST_DEVICE SeqlenInfo(const int32_t* cu_seqlens_q, const int32_t* cu_seqlens_k, int32_t batch_idx):
        offset_q(seq_offset(cu_seqlens_q, batch_idx)),
        offset_k(seq_offset(cu_seqlens_k, batch_idx)),
        seqlen_q(seq_len(cu_seqlens_q, batch_idx)),
        seqlen_k(seq_len(cu_seqlens_k, batch_idx)),
        blk_q_base(blk_offset(cu_seqlens_q, batch_idx, kBlockQ)),
        blk_k_base(blk_offset(cu_seqlens_k, batch_idx, kBlockK)),
        delta(seqlen_k - seqlen_q)
    {
        static_assert(kVarlen, "the cu_seqlens constructor is the varlen one");
    }
};

// The dense form is a compile-time constant end to end (sm80 geometry).
static_assert(SeqlenInfo<false, 128, 64>{300, 300}.blk_q_base == 0, "dense block bases are 0");
static_assert(SeqlenInfo<false, 128, 64>{100, 300}.delta == 200, "delta is kv - qo");
static_assert(SeqlenInfo<false, 128, 64>{1000, 1}.delta == -999, "delta must stay signed");

}  // namespace sage
