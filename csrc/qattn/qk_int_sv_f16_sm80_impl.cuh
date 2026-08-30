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

// The sm80 int8-QK / fp16-PV attention kernel. Split out of
// qk_int_sv_f16_cuda_sm80.cu (which keeps the host launchers) so that a
// second translation unit can include the same body; the split is a pure
// move, the kernel text below is unchanged.
//
// That second translation unit is qk_int_sv_f16_cuda_sm80_varlen.cu, which
// compiles this body with SAGE_VARLEN defined: q/k/v are packed to
// [total_tokens, heads, head_dim] and blockIdx.z selects a sequence of a
// cu_seqlens prefix sum instead of a batch entry. The choice of #ifdef over a
// `bool kVarlen` template parameter is what keeps the dense kernel's SASS
// byte-identical - a template parameter changes the kernel's parameter space
// and its constant-bank layout, so the dense instantiation could not be
// unchanged even where the code is.

#include "../utils.cuh"
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <torch/types.h>

#include "../cp_async.cuh"
#include "../dispatch_utils.h"
#include "../math.cuh"
#include "../mma.cuh"
#include "../permuted_smem.cuh"
#include "../sageattn/seqlen_info.cuh"

#include "attn_utils.cuh"

// All sm80 kernels and launchers live in sage::sm80: the arch families are
// linked into a single _C extension, so same-named launchers/templates from
// different arches must not share symbols (ODR). The varlen translation unit
// compiles a *different* body under the same template name and needs its own
// namespace for the same reason (the sm89 family solves it identically, with
// SAGEATTN_ARCH_NS).
#ifndef SAGEATTN_SM80_NS
#define SAGEATTN_SM80_NS sm80
#endif

namespace sage {
namespace SAGEATTN_SM80_NS {

#define PACK_SIZE_QK 16  // as if it is int8
#define PACK_SIZE_V 8    // fp16
#define PACK_SIZE_O 8    // fp16

// treat as if int8 tensor core
#define MMA_QK_M 16
#define MMA_QK_N 16
#define MMA_QK_K 32

// fp16 tensor core
#define MMA_SV_M 16
#define MMA_SV_N 16
#define MMA_SV_K 16

// Which flavour of the KV-tile pipeline stage to instantiate; see the
// process_tile lambda in the kernel below.
enum class TileKind {
    kBulk,
    kMask,
    kLast
};
template<TileKind K>
using TileTag = std::integral_constant<TileKind, K>;

template<uint32_t         CTA_Q,
         uint32_t         CTA_K,
         uint32_t         WARP_Q,
         uint32_t         WARP_K,
         uint32_t         head_dim,
         DataType         DTypeQK,
         QuantGranularity Q_GRAN,
         QuantGranularity K_GRAN,
         typename DTypeSVAccum = float,
         bool use_inst_buf     = false,
         typename DTypeOut     = half,
         ComputeUnit DenominatorAccumUnit,
         MaskMode    mask_mode   = MaskMode::kNone,
         bool        return_lse  = false,
         bool        fuse_v_mean = false>
__global__ void qk_int_sv_f16_attn_kernel(const int8_t* __restrict__ Q,
                                          const int8_t* __restrict__ K,
                                          const half* __restrict__ V,
                                          DTypeOut* __restrict__ O,
                                          float* __restrict__ Lse,
                                          const float* __restrict__ Q_scale,
                                          const float* __restrict__ K_scale,
                                          const DTypeOut* __restrict__ V_mean,
#ifdef SAGE_VARLEN
                                          // [batch_size + 1] prefix sums; the sequence lengths and the
                                          // scale block bases are derived from them per block. The
                                          // *_stride_h are the packed tensors' per-head extents, which
                                          // the dense kernel instead reads back out of gridDim.
                                          const int32_t* __restrict__ cu_seqlens_q,
                                          const int32_t* __restrict__ cu_seqlens_k,
                                          const uint32_t q_scale_stride_h,
                                          const uint32_t k_scale_stride_h,
                                          const uint32_t lse_stride_h,
#else
                                          const uint32_t qo_len,
                                          const uint32_t kv_len,
#endif
                                          const uint32_t qo_per_kv_head,
                                          const int64_t  stride_batch_q,
                                          const uint32_t stride_seq_q,
                                          const int64_t  stride_h_q,
                                          const int64_t  stride_batch_k,
                                          const uint32_t stride_seq_k,
                                          const int64_t  stride_h_k,
                                          const int64_t  stride_batch_v,
                                          const uint32_t stride_seq_v,
                                          const int64_t  stride_h_v,
                                          const int64_t  stride_batch_o,
                                          const uint32_t stride_seq_o,
                                          const int64_t  stride_h_o,
                                          float          sm_scale)
{
    // compile time check
    static_assert(DTypeQK == DataType::kInt8, "DTypeQK must be int8");
    static_assert(Q_GRAN == QuantGranularity::kPerWarp || Q_GRAN == QuantGranularity::kPerThread,
                  "Q_GRAN must be kPerWarp or kPerThread");
    static_assert(K_GRAN == QuantGranularity::kPerWarp || K_GRAN == QuantGranularity::kPerThread,
                  "K_GRAN must be kPerWarp or kPerThread");
    static_assert(std::is_same<DTypeSVAccum, float>::value || !use_inst_buf,
                  "use_inst_buf only supports DTypeSVAccum as float");
    static_assert(std::is_same<DTypeSVAccum, float>::value || std::is_same<DTypeSVAccum, half>::value,
                  "DTypeSVAccum must be float or half");
    static_assert(std::is_same<DTypeOut, half>::value || std::is_same<DTypeOut, nv_bfloat16>::value,
                  "DTypeOut must be half or nv_bfloat16");
    static_assert(head_dim % 64 == 0, "head_dim must be a multiple of 64");
    static_assert(!fuse_v_mean || std::is_same<DTypeSVAccum, half>::value, "fuse_v_mean only supports half");
    static_assert(CTA_Q / CTA_K <= 2);  // for efficient causal implementation

    using DTypeOut2 = typename std::conditional<std::is_same<DTypeOut, half>::value, half2, nv_bfloat162>::type;

    constexpr uint32_t num_warps_q = CTA_Q / WARP_Q;
    constexpr uint32_t num_warps_k = CTA_K / WARP_K;
    constexpr uint32_t num_warps   = num_warps_q * num_warps_k;
    constexpr uint32_t num_tiles_q = WARP_Q / MMA_QK_M;
    constexpr uint32_t num_tiles_k = WARP_K / MMA_QK_N;
    constexpr uint32_t num_tiles_qk_inner = head_dim / MMA_QK_K;
    constexpr uint32_t num_tiles_v = head_dim / MMA_SV_N;

    constexpr uint32_t QK_SMEM_STRIDE = head_dim;
    constexpr uint32_t O_SMEM_STRIDE  = head_dim;
    constexpr uint32_t V_SMEM_STRIDE  = head_dim;

    extern __shared__ int8_t smem[];

    const uint32_t lane_id = get_lane_id_2d();
    const uint32_t warp_id = get_warp_id_2d();

    // maximize L2 hit rate
    const uint32_t batch_id = blockIdx.z;
    // Under a causal mask a CTA's cost grows with its query-block index, so walk
    // blockIdx.x backwards: the heaviest CTAs are scheduled first and the tail of
    // the grid is filled with the cheap ones instead of the other way round. Each
    // CTA owns a disjoint slice of O, so the permutation changes no results.
    const uint32_t cta_idx_q    = (mask_mode == MaskMode::kCausal) ? (gridDim.x - 1 - blockIdx.x) : blockIdx.x;
    const uint32_t num_qo_heads = gridDim.y;
    const uint32_t head_id      = blockIdx.y;

#ifdef SAGE_VARLEN
    // blockIdx.z is a sequence of the prefix sums, not a batch entry. From
    // here down every length and every index is sequence-relative, exactly as
    // the dense kernel's are relative to its batch entry; the sequence's
    // absolute position enters the pointer bases and the scale block bases
    // and nowhere else.
    const SeqlenInfo<true, CTA_Q, CTA_K> seq_info(cu_seqlens_q, cu_seqlens_k, batch_id);
    const uint32_t                       qo_len = seq_info.seqlen_q;
    const uint32_t                       kv_len = seq_info.seqlen_k;
    // The grid is opened to max_seqlen_q, so most sequences leave part of it
    // empty. Under a causal mask cta_idx_q is the reversed blockIdx.x, so the
    // test has to be on the logical index, and it is block-uniform either way.
    if (cta_idx_q * CTA_Q >= qo_len) {
        return;
    }
#endif

    // transfer to base 2 instead of base e with better numerical efficiency
    sm_scale *= math::log2e;

    // RS holds the fragment of S
    int32_t      RS[num_tiles_q][num_tiles_k][8];
    DTypeSVAccum RO[num_tiles_q][num_tiles_v][8];
    float        row_max[num_tiles_q][2];  // max
    float        denom[num_tiles_q][2];    // denominator

    // q_scale is read once, so the whole index can stay 64-bit. k_scale is read
    // once per iteration, so it is split into a 64-bit base pointer plus a
    // 32-bit running offset. The offset's live range crosses every
    // process_tile expansion, and at 255 registers ptxas rematerializes its
    // lane term as an S2R from SR_TID inside the loop - on the per-iteration
    // dequant_scale chain, the same pattern the sm89 kernel hit - so the
    // plain non-causal instances fold it into the pointer below.
    int64_t      q_scale_idx;
    const float* K_scale_base_ptr;
    uint32_t     k_scale_off;

    constexpr uint32_t k_scale_advance_offset = (K_GRAN == QuantGranularity::kPerWarp)  ? (CTA_K / WARP_K) :
                                                                                          (CTA_K / WARP_K) * 4;

#ifdef SAGE_VARLEN
    // The packed scales are [heads, blocks]: the batch dimension is gone and
    // the sequence's own blocks start at varlen.h's blk_offset, which is the
    // same expression the quantization kernel wrote them with. Deriving the
    // per-head extent from gridDim (the dense form below) would be wrong here
    // twice over - the grid covers max_seqlen, and the blocks of the earlier
    // sequences sit in front of this one's.
    constexpr uint32_t q_scale_per_cta = (Q_GRAN == QuantGranularity::kPerWarp)  ? num_warps_q :
                                                                                   (num_warps_q * 8);
    q_scale_idx                        = static_cast<int64_t>(head_id) * q_scale_stride_h
                  + static_cast<int64_t>(seq_info.blk_q_base + cta_idx_q) * q_scale_per_cta;
    if constexpr (Q_GRAN == QuantGranularity::kPerWarp) {
        q_scale_idx += get_warp_idx_q<num_warps_q, num_warps_k>();
    }
    else if constexpr (Q_GRAN == QuantGranularity::kPerThread) {
        q_scale_idx += get_warp_idx_q<num_warps_q, num_warps_k>() * 8 + lane_id / 4;
    }

    K_scale_base_ptr = K_scale + static_cast<int64_t>(head_id / qo_per_kv_head) * k_scale_stride_h
                       + static_cast<int64_t>(seq_info.blk_k_base) * k_scale_advance_offset;
    if constexpr (K_GRAN == QuantGranularity::kPerWarp) {
        k_scale_off = get_warp_idx_k<num_warps_q, num_warps_k>();
    }
    else if constexpr (K_GRAN == QuantGranularity::kPerThread) {
        k_scale_off = get_warp_idx_k<num_warps_q, num_warps_k>() * 4 + lane_id % 4;
    }
#else
    if constexpr (Q_GRAN == QuantGranularity::kPerWarp) {
        const uint32_t num_warp_tiles_q = gridDim.x * num_warps_q;
        q_scale_idx                     = static_cast<int64_t>(batch_id) * num_qo_heads * num_warp_tiles_q
                      + static_cast<int64_t>(head_id) * num_warp_tiles_q + cta_idx_q * num_warps_q
                      + get_warp_idx_q<num_warps_q, num_warps_k>();
    }
    else if constexpr (Q_GRAN == QuantGranularity::kPerThread) {
        const uint32_t num_warp_tiles_q = gridDim.x * num_warps_q;
        q_scale_idx                     = static_cast<int64_t>(batch_id) * num_qo_heads * (num_warp_tiles_q * 8)
                      + static_cast<int64_t>(head_id) * (num_warp_tiles_q * 8) + cta_idx_q * (num_warps_q * 8)
                      + get_warp_idx_q<num_warps_q, num_warps_k>() * 8 + lane_id / 4;
    }

    if constexpr (K_GRAN == QuantGranularity::kPerWarp) {
        const uint32_t num_warp_tiles_k = div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);
        K_scale_base_ptr = K_scale + static_cast<int64_t>(batch_id) * (num_qo_heads / qo_per_kv_head) * num_warp_tiles_k
                           + static_cast<int64_t>(head_id / qo_per_kv_head) * num_warp_tiles_k;
        k_scale_off = get_warp_idx_k<num_warps_q, num_warps_k>();
    }
    else if constexpr (K_GRAN == QuantGranularity::kPerThread) {
        const uint32_t num_warp_tiles_k = div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);
        K_scale_base_ptr                = K_scale
                           + static_cast<int64_t>(batch_id) * (num_qo_heads / qo_per_kv_head) * (num_warp_tiles_k * 4)
                           + static_cast<int64_t>(head_id / qo_per_kv_head) * (num_warp_tiles_k * 4);
        k_scale_off = get_warp_idx_k<num_warps_q, num_warps_k>() * 4 + lane_id % 4;
    }
#endif

    // initialize o, row_max, denom
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
            if constexpr (std::is_same<DTypeSVAccum, float>::value) {
#pragma unroll
                for (uint32_t e = 0; e < 8; e++) {
                    RO[fq][fv][e] = 0.0f;
                }
            }
            else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
#pragma unroll
                for (uint32_t e = 0; e < 4; e++) {
                    ((int32_t*)RO[fq][fv])[e] = 0;
                }
            }
        }
    }
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t e = 0; e < 2; e++) {
            row_max[fq][e] = -5000000.0f;
            denom[fq][e]   = 1.0f;
        }
    }

    // A predicated V load must zero the rows past the sequence: the out-of-bound
    // mask drives their probabilities to exactly 0, and 0 * NaN is NaN, so a
    // stale half left in shared memory by the previous CTA poisons the whole
    // output row. Every varlen sequence ends mid-tile, which makes that the
    // normal case there; the dense kernel runs the same risk on its last tile.
    // The Q/K loads keep kNoFill -- their product is integer and masked before
    // any fp use.
    constexpr cp_async::SharedMemFillMode kVFillMode = cp_async::SharedMemFillMode::kFillZero;

    constexpr uint32_t K_smem_row_offset = CTA_Q;
    constexpr uint32_t V_smem_row_offset = CTA_Q + CTA_K;

    constexpr SwizzleMode swizzle_mode_QK = (QK_SMEM_STRIDE == 32) ? SwizzleMode::k32B :
                                            (QK_SMEM_STRIDE == 64) ? SwizzleMode::k64B :
                                                                     SwizzleMode::k128B;
    smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_Q(smem);
    smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_K(smem + K_smem_row_offset * QK_SMEM_STRIDE);
    constexpr SwizzleMode swizzle_mode_V = (V_SMEM_STRIDE == 32) ? SwizzleMode::k64B : SwizzleMode::k128B;
    smem_t<swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V> smem_V(smem + V_smem_row_offset * QK_SMEM_STRIDE);
    constexpr SwizzleMode swizzle_mode_O = (O_SMEM_STRIDE == 32) ? SwizzleMode::k64B : SwizzleMode::k128B;
    smem_t<swizzle_mode_O, O_SMEM_STRIDE / PACK_SIZE_O> smem_O(smem);

    constexpr uint32_t global_to_shared_line_lanes_QK = (QK_SMEM_STRIDE == 32) ? 2 : (QK_SMEM_STRIDE == 64) ? 4 : 8;
    constexpr uint32_t global_to_shared_copy_lines_per_warp_QK = (QK_SMEM_STRIDE == 32) ? 16 :
                                                                 (QK_SMEM_STRIDE == 64) ? 8 :
                                                                                          4;
    constexpr uint32_t global_to_shared_line_lanes_V           = (V_SMEM_STRIDE == 32) ? 4 : 8;
    constexpr uint32_t global_to_shared_copy_lines_per_warp_V  = (V_SMEM_STRIDE == 32) ? 8 : 4;
    constexpr uint32_t global_to_shared_line_lanes_O           = (O_SMEM_STRIDE == 32) ? 4 : 8;
    constexpr uint32_t global_to_shared_copy_lines_per_warp_O  = (O_SMEM_STRIDE == 32) ? 8 : 4;

    constexpr uint32_t QK_smem_iters_row = QK_SMEM_STRIDE / (global_to_shared_line_lanes_QK * PACK_SIZE_QK);
    constexpr uint32_t Q_smem_iters_col  = CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_QK);
    constexpr uint32_t K_smem_iters_col  = CTA_K / (num_warps * global_to_shared_copy_lines_per_warp_QK);
    constexpr uint32_t V_smem_iters_row  = V_SMEM_STRIDE / (global_to_shared_line_lanes_V * PACK_SIZE_V);
    constexpr uint32_t V_smem_iters_col  = CTA_K / (num_warps * global_to_shared_copy_lines_per_warp_V);
    constexpr uint32_t O_smem_iters_row  = O_SMEM_STRIDE / (global_to_shared_line_lanes_O * PACK_SIZE_O);
    constexpr uint32_t O_smem_iters_col  = CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_O);

    // The base offsets are spelled out twice rather than hoisted into a shared
    // local: the dense text below has to stay character for character what it
    // was, or nvcc reschedules it and the SASS gate (dense instruction streams
    // byte-identical across this split) fails.
#ifdef SAGE_VARLEN
    // The sequence's first token replaces the batch stride; the packed tensors
    // have no batch dimension at all.
    const int8_t* Q_lane_base_ptr = Q + static_cast<int64_t>(seq_info.offset_q) * stride_seq_q
                                    + static_cast<int64_t>(head_id) * stride_h_q
                                    + static_cast<int64_t>(cta_idx_q * CTA_Q + CTA_Q / num_warps * warp_id
                                                           + lane_id / global_to_shared_line_lanes_QK)
                                          * stride_seq_q
                                    + static_cast<int64_t>((lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK);
    const int8_t* K_lane_base_ptr =
        K + static_cast<int64_t>(seq_info.offset_k) * stride_seq_k
        + static_cast<int64_t>(head_id / qo_per_kv_head) * stride_h_k
        + static_cast<int64_t>(CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK) * stride_seq_k
        + static_cast<int64_t>((lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK);
    const half* V_lane_base_ptr =
        V + static_cast<int64_t>(seq_info.offset_k) * stride_seq_v
        + static_cast<int64_t>(head_id / qo_per_kv_head) * stride_h_v
        + static_cast<int64_t>(CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_V) * stride_seq_v
        + static_cast<int64_t>((lane_id % global_to_shared_line_lanes_V) * PACK_SIZE_V);
#else
    const int8_t* Q_lane_base_ptr = Q + static_cast<int64_t>(batch_id) * stride_batch_q
                                    + static_cast<int64_t>(head_id) * stride_h_q
                                    + static_cast<int64_t>(cta_idx_q * CTA_Q + CTA_Q / num_warps * warp_id
                                                           + lane_id / global_to_shared_line_lanes_QK)
                                          * stride_seq_q
                                    + static_cast<int64_t>((lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK);
    const int8_t* K_lane_base_ptr =
        K + static_cast<int64_t>(batch_id) * stride_batch_k
        + static_cast<int64_t>(head_id / qo_per_kv_head) * stride_h_k
        + static_cast<int64_t>(CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK) * stride_seq_k
        + static_cast<int64_t>((lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK);
    const half* V_lane_base_ptr =
        V + static_cast<int64_t>(batch_id) * stride_batch_v
        + static_cast<int64_t>(head_id / qo_per_kv_head) * stride_h_v
        + static_cast<int64_t>(CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_V) * stride_seq_v
        + static_cast<int64_t>((lane_id % global_to_shared_line_lanes_V) * PACK_SIZE_V);
#endif
    uint32_t Q_smem_offset_load = smem_Q.get_permuted_offset(
        warp_id * global_to_shared_copy_lines_per_warp_QK * Q_smem_iters_col + lane_id / global_to_shared_line_lanes_QK,
        lane_id % global_to_shared_line_lanes_QK);
    uint32_t K_smem_offset_load = smem_K.get_permuted_offset(
        warp_id * global_to_shared_copy_lines_per_warp_QK * K_smem_iters_col + lane_id / global_to_shared_line_lanes_QK,
        lane_id % global_to_shared_line_lanes_QK);
    uint32_t V_smem_offset_load = smem_V.get_permuted_offset(
        warp_id * global_to_shared_copy_lines_per_warp_V * V_smem_iters_col + lane_id / global_to_shared_line_lanes_V,
        lane_id % global_to_shared_line_lanes_V);

    uint32_t Q_smem_offset_mma =
        smem_Q.get_permuted_offset(get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id % 16, lane_id / 16);
    uint32_t K_smem_offset_mma = smem_K.get_permuted_offset(
        get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
    uint32_t V_smem_offset_mma =
        smem_V.get_permuted_offset(get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 16, lane_id / 16);

    // for causal masking
#ifdef SAGE_VARLEN
    // Bottom-right alignment (flash-attention semantics): row q of the sequence
    // attends to keys up to q + delta. Q_idx_lane_base serves the mask and
    // nothing else here, so the shift can live in it - but it has to become
    // signed, because kv shorter than qo makes delta negative and an unsigned
    // row index would wrap to "mask nothing" exactly where everything is masked.
    int32_t Q_idx_lane_base =
        static_cast<int32_t>(cta_idx_q * CTA_Q + get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4)
        + seq_info.delta;
#else
    uint32_t Q_idx_lane_base    = cta_idx_q * CTA_Q + get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
#endif
    uint32_t K_idx_lane_base = get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + 2 * (lane_id % 4);

    // for loading
    uint32_t Q_load_idx_lane_base =
        cta_idx_q * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;
    uint32_t K_load_idx_lane_base = CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;
    uint32_t V_load_idx_lane_base = CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_V;

#ifdef SAGE_VARLEN
    // The last key this CTA can see is (cta_idx_q + 1) * CTA_Q - 1 + delta.
    // delta is signed and the bound can go negative (kv much shorter than qo),
    // which is why the whole computation is int32: the dense uint32 form would
    // wrap it into an enormous trip count.
    const int32_t causal_bound   = static_cast<int32_t>((cta_idx_q + 1) * CTA_Q) + seq_info.delta;
    const int32_t kv_bound       = (mask_mode == MaskMode::kCausal) ? min(static_cast<int32_t>(kv_len), causal_bound) :
                                                                      static_cast<int32_t>(kv_len);
    const int32_t num_iterations = div_ceil(kv_bound, static_cast<int32_t>(CTA_K));

    // The first KV tile holding a masked element. With delta not a multiple of
    // CTA_K the diagonal band crosses three tiles, not the two the dense code
    // hardcodes. Clamping to num_iterations - 2 makes delta == 0 reproduce the
    // dense tile structure exactly: kBulk folds the K dequant scale into
    // sm_scale while kMask multiplies it into S, so masking one tile more or
    // fewer is a numerical difference and not only a speed one.
    const int32_t first_masked_tile =
        (mask_mode == MaskMode::kCausal) ?
            max(0,
                min(num_iterations - 2,
                    (static_cast<int32_t>(cta_idx_q * CTA_Q) + seq_info.delta) / static_cast<int32_t>(CTA_K))) :
            num_iterations - 2;

    if (num_iterations <= 0) {
        // No KV tile at all: an empty key sequence, or a causal CTA whose rows
        // admit no key. O is a zero row and lse is -inf. Leaving from here,
        // before the first cp.async is issued, keeps nothing in flight.
        DTypeOut* O_zero_ptr = O + static_cast<int64_t>(seq_info.offset_q) * stride_seq_o
                               + static_cast<int64_t>(head_id) * stride_h_o
                               + static_cast<int64_t>(cta_idx_q * CTA_Q + CTA_Q / num_warps * warp_id
                                                      + lane_id / global_to_shared_line_lanes_O)
                                     * stride_seq_o
                               + static_cast<int64_t>(lane_id % global_to_shared_line_lanes_O * PACK_SIZE_O);
        uint32_t O_zero_idx = cta_idx_q * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_O;
#pragma unroll
        for (uint32_t i = 0; i < O_smem_iters_col; i++) {
#pragma unroll
            for (uint32_t j = 0; j < O_smem_iters_row; j++) {
                if (O_zero_idx < qo_len) {
                    *reinterpret_cast<uint4*>(O_zero_ptr + j * global_to_shared_line_lanes_O * PACK_SIZE_O) =
                        make_uint4(0, 0, 0, 0);
                }
            }
            O_zero_ptr += global_to_shared_copy_lines_per_warp_O * stride_seq_o;
            O_zero_idx += global_to_shared_copy_lines_per_warp_O;
        }

        if constexpr (return_lse) {
            const uint32_t lse_idx = cta_idx_q * CTA_Q + lane_id / 4 + 8 * (lane_id % 4)
                                     + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>();
            if (lse_idx < qo_len && (lane_id % 4) < 2 * num_tiles_q) {
                Lse[static_cast<int64_t>(head_id) * lse_stride_h + seq_info.offset_q + lse_idx] = -INFINITY;
            }
        }
        return;
    }
#else
    const uint32_t num_iterations =
        div_ceil(mask_mode == MaskMode::kCausal ? min(kv_len, (cta_idx_q + 1) * CTA_Q) : kv_len, CTA_K);
#endif

    // load Q with predicate
    load_global_to_shared<global_to_shared_line_lanes_QK,
                          global_to_shared_copy_lines_per_warp_QK,
                          QK_smem_iters_row,
                          Q_smem_iters_col,
                          swizzle_mode_QK,
                          QK_SMEM_STRIDE / PACK_SIZE_QK,
                          CTA_Q>(
        &Q_lane_base_ptr, Q_smem_offset_load, stride_seq_q, smem_Q, Q_load_idx_lane_base, qo_len);
    cp_async::commit_group();

    // for num_tiles_qk_inner = 1, we load all Qs in register
    uint32_t RQ[num_tiles_q][4];
    if constexpr (num_tiles_qk_inner == 1) {
        // Only this branch touches smem_Q before the main loop, so the drain lives
        // here. Otherwise Q stays in flight and the loop's wait_group<1> (which
        // retires the older Q group together with K) is what makes it visible --
        // that way Q/K/V are three back-to-back cp.async groups.
        cp_async::wait_group<0>();
        __syncthreads();
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
            smem_Q.ldmatrix_m8n8x4(Q_smem_offset_mma, RQ[fq]);
            Q_smem_offset_mma = smem_Q.advance_offset_by_row<16>(Q_smem_offset_mma);
        }
    }

    // load K with predicate
    load_global_to_shared<global_to_shared_line_lanes_QK,
                          global_to_shared_copy_lines_per_warp_QK,
                          QK_smem_iters_row,
                          K_smem_iters_col,
                          swizzle_mode_QK,
                          QK_SMEM_STRIDE / PACK_SIZE_QK,
                          CTA_K>(
        &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K, K_load_idx_lane_base, kv_len);
    cp_async::commit_group();

    // Fold the lane/warp term into the pointer for the plain non-causal
    // instances only, mirroring the sm89 kernel: their KV loop is almost all
    // kBulk tiles, where the extra S2R sits on the per-iteration dequant_scale
    // chain. The causal and return_lse instances keep the split form - folding
    // there reshuffles their register allocation for the worse (measured on
    // sm89: causal gives half its win back, non-causal+lse spills 39 -> 90).
    if constexpr (mask_mode != MaskMode::kCausal && !return_lse) {
        K_scale_base_ptr += k_scale_off;
        k_scale_off = 0;
    }

    float q_scale = Q_scale[q_scale_idx];

    float original_sm_scale = sm_scale;
    float dequant_scale     = q_scale * K_scale_base_ptr[k_scale_off + 0 * k_scale_advance_offset];

    sm_scale = original_sm_scale * dequant_scale;

    // load V with predicate
    load_global_to_shared<global_to_shared_line_lanes_V,
                          global_to_shared_copy_lines_per_warp_V,
                          V_smem_iters_row,
                          V_smem_iters_col,
                          swizzle_mode_V,
                          V_SMEM_STRIDE / PACK_SIZE_V,
                          CTA_K,
                          kVFillMode>(
        &V_lane_base_ptr, V_smem_offset_load, stride_seq_v, smem_V, V_load_idx_lane_base, kv_len);
    cp_async::commit_group();

    K_load_idx_lane_base += CTA_K;
    V_load_idx_lane_base += CTA_K;

    // ------------------------------------------------------------- KV tile body
    // The three call sites below run the same pipeline stage and differ only in
    // compile-time bits:
    //   kBulk: no mask. The K dequant scale is folded into sm_scale, so S stays
    //          unscaled here; the next K/V tile is prefetched unpredicated.
    //   kMask: causal mask, so S is dequantized explicitly and update_mdo gets the
    //          raw sm_scale instead; the next K/V tile needs bound predicates.
    //   kLast: kMask plus the out-of-bound mask, and no prefetch -- V is then the
    //          only cp.async group still in flight, hence wait_group<0>.
    // next_iter indexes the K scale of the tile being prefetched (unused by kLast).
    auto process_tile = [&](auto kind_t, uint32_t next_iter) {
        constexpr TileKind kind = decltype(kind_t)::value;

        // ensure K is ready
        cp_async::wait_group<1>();
        __syncthreads();

        // compute QK^T
        if constexpr (num_tiles_qk_inner == 1) {
            compute_int_qk<num_warps_q,
                           num_warps_k,
                           num_tiles_q,
                           num_tiles_k,
                           num_tiles_qk_inner,
                           swizzle_mode_QK,
                           QK_SMEM_STRIDE / PACK_SIZE_QK,
                           DTypeQK>(smem_K, RS, RQ, K_smem_offset_mma);
        }
        else {
            compute_int_qk<num_warps_q,
                           num_warps_k,
                           num_tiles_q,
                           num_tiles_k,
                           num_tiles_qk_inner,
                           swizzle_mode_QK,
                           QK_SMEM_STRIDE / PACK_SIZE_QK,
                           DTypeQK>(smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
        }

        float RS_f32[num_tiles_q][num_tiles_k][8];

#if defined(SAGE_OPT_MAGIC_I2F)
        // Magic-number int->float. |S| <= head_dim * 127^2 = 2.06e6 < 2^22, so
        // __int_as_float(S + 0x4B400000) is exactly 12582912.0f + S: the conversion
        // itself is lossless and I2F (quarter rate on sm8x) becomes IADD. kBulk
        // subtracts the bias back and stays bit-exact; kMask folds the bias into the
        // scale multiply as one FFMA, so its only numerical difference from the
        // reference is the single rounding of 12582912.0f * dequant_scale -- at most
        // 0.75 * dequant_scale absolute, i.e. 2^-24 of the representable S range.
        const float magic_bias = (kind == TileKind::kBulk) ? -12582912.0f : -12582912.0f * dequant_scale;
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
                for (uint32_t e = 0; e < 8; e++) {
                    const float S_magic = __int_as_float(RS[fq][fk][e] + 0x4B400000);
                    if constexpr (kind == TileKind::kBulk) {
                        RS_f32[fq][fk][e] = S_magic + magic_bias;
                    }
                    else {
                        RS_f32[fq][fk][e] = __fmaf_rn(S_magic, dequant_scale, magic_bias);
                    }
                }
            }
        }
#else
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
                for (uint32_t e = 0; e < 8; e++) {
                    if constexpr (kind == TileKind::kBulk) {
                        RS_f32[fq][fk][e] = __int2float_rz(RS[fq][fk][e]);
                    }
                    else {
                        RS_f32[fq][fk][e] = __int2float_rz(RS[fq][fk][e]) * dequant_scale;
                    }
                }
            }
        }
#endif

        if constexpr (kind != TileKind::kBulk && mask_mode == MaskMode::kCausal) {
#ifdef SAGE_VARLEN
            apply_causal_mask_bottom_right<num_tiles_q, num_tiles_k>(Q_idx_lane_base, K_idx_lane_base, RS_f32);
#else
            apply_causal_mask<num_tiles_q, num_tiles_k>(Q_idx_lane_base, K_idx_lane_base, RS_f32);
#endif
        }
        if constexpr (kind == TileKind::kLast) {
            // check out of bound in the last iter
            apply_out_of_bounds_mask<num_tiles_q, num_tiles_k>(K_idx_lane_base, RS_f32, kv_len);
        }
        K_idx_lane_base += CTA_K;

        // kBulk already carries dequant_scale inside sm_scale; the masked tiles
        // applied it to RS_f32 above and so pass the unscaled softmax scale.
        const float mdo_sm_scale = (kind == TileKind::kBulk) ? sm_scale : original_sm_scale;

        if constexpr (std::is_same<DTypeSVAccum, float>::value) {
            update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, false, false>(
                RS_f32, RO, row_max, denom, mdo_sm_scale);
        }
        else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
            update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, true, false, false>(
                RS_f32, RO, row_max, denom, mdo_sm_scale);
        }

        if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore) {
            accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, denom);
        }

        uint32_t RS_f16[num_tiles_q][num_tiles_k][4];
        RS_f32_to_f16<num_tiles_q, num_tiles_k>(RS_f32, RS_f16);

        if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore) {
            accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kTensorCore>(RS_f16, denom);
        }

        if constexpr (kind != TileKind::kLast) {
            __syncthreads();

            // load K
            if constexpr (kind == TileKind::kBulk) {
                load_global_to_shared<global_to_shared_line_lanes_QK,
                                      global_to_shared_copy_lines_per_warp_QK,
                                      QK_smem_iters_row,
                                      K_smem_iters_col,
                                      swizzle_mode_QK,
                                      QK_SMEM_STRIDE / PACK_SIZE_QK,
                                      CTA_K>(&K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K);
            }
            else {
                load_global_to_shared<global_to_shared_line_lanes_QK,
                                      global_to_shared_copy_lines_per_warp_QK,
                                      QK_smem_iters_row,
                                      K_smem_iters_col,
                                      swizzle_mode_QK,
                                      QK_SMEM_STRIDE / PACK_SIZE_QK,
                                      CTA_K>(
                    &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K, K_load_idx_lane_base, kv_len);
            }
            cp_async::commit_group();

            dequant_scale = q_scale * K_scale_base_ptr[k_scale_off + next_iter * k_scale_advance_offset];
            sm_scale      = original_sm_scale * dequant_scale;

            // ensure V is ready
            cp_async::wait_group<1>();
        }
        else {
            // ensure V is ready
            cp_async::wait_group<0>();
        }
        __syncthreads();

        if constexpr (!use_inst_buf) {
            compute_fp16_sv_permuted<num_warps_q,
                                     num_warps_k,
                                     num_tiles_q,
                                     num_tiles_k,
                                     num_tiles_v,
                                     swizzle_mode_V,
                                     V_SMEM_STRIDE / PACK_SIZE_V,
                                     4>(smem_V, RS_f16, RO, denom, V_smem_offset_mma);
        }
        else {
            compute_fp16_sv_permuted_inst_buf<num_warps_q,
                                              num_warps_k,
                                              num_tiles_q,
                                              num_tiles_k,
                                              num_tiles_v,
                                              swizzle_mode_V,
                                              V_SMEM_STRIDE / PACK_SIZE_V,
                                              4>(smem_V, RS_f16, RO, denom, V_smem_offset_mma);
        }

        __syncthreads();

        if constexpr (kind != TileKind::kLast) {
            // load V
            if constexpr (kind == TileKind::kBulk) {
                load_global_to_shared<global_to_shared_line_lanes_V,
                                      global_to_shared_copy_lines_per_warp_V,
                                      V_smem_iters_row,
                                      V_smem_iters_col,
                                      swizzle_mode_V,
                                      V_SMEM_STRIDE / PACK_SIZE_V,
                                      CTA_K>(&V_lane_base_ptr, V_smem_offset_load, stride_seq_v, smem_V);
            }
            else {
                load_global_to_shared<global_to_shared_line_lanes_V,
                                      global_to_shared_copy_lines_per_warp_V,
                                      V_smem_iters_row,
                                      V_smem_iters_col,
                                      swizzle_mode_V,
                                      V_SMEM_STRIDE / PACK_SIZE_V,
                                      CTA_K,
                                      kVFillMode>(
                    &V_lane_base_ptr, V_smem_offset_load, stride_seq_v, smem_V, V_load_idx_lane_base, kv_len);
            }
            cp_async::commit_group();
            K_load_idx_lane_base += CTA_K;
            V_load_idx_lane_base += CTA_K;
        }
    };

#ifdef SAGE_VARLEN
    // A call processes the tile before the one it prefetches, so `tile` is the
    // index of the tile being processed. Two to three of them are masked
    // (first_masked_tile above); with delta == 0 the first loop ends at
    // num_iterations - 2 and the second runs once, which is the dense
    // structure below, instruction for instruction.
    int32_t tile = 0;
    for (; tile + 1 < num_iterations && tile < first_masked_tile; tile++) {
        process_tile(TileTag<TileKind::kBulk>{}, tile + 1);
    }
    for (; tile + 1 < num_iterations; tile++) {
        process_tile(TileTag<TileKind::kMask>{}, tile + 1);
    }
    process_tile(TileTag<TileKind::kLast>{}, 0);
#else
#pragma unroll
    for (uint32_t iter = 1; iter < num_iterations - 1; iter++) {
        process_tile(TileTag<TileKind::kBulk>{}, iter);
    }

    // second last iter, apply causal mask
    if (num_iterations > 1) {
        process_tile(TileTag<TileKind::kMask>{}, num_iterations - 1);
    }

    // last iter, apply causal mask and out of bound mask
    process_tile(TileTag<TileKind::kLast>{}, 0);
#endif

    // TODO: thread block sync mdo state for num_warps_k > 0

#ifdef SAGE_VARLEN
    // A row that admitted no key at all - bottom-right causal with kv shorter
    // than qo - saw nothing but the -5000000 mask sentinel. The sentinel is
    // finite, so it *became* the row max and every masked entry got the weight
    // exp2(0) = 1: without this the row would come out as the plain average of
    // a KV tile. flash-attention's answer for such a row is a zero output and
    // an -inf lse, which is what is forced here. The row is identified by its
    // index rather than by comparing against the scaled sentinel:
    // Q_idx_lane_base already carries + delta, so "no admissible key" is
    // exactly a negative shifted row index. (row_max is not read by
    // normalize_d; setting it to -inf is what makes the lse below -inf.)
    if constexpr (mask_mode == MaskMode::kCausal) {
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t e = 0; e < 2; e++) {
                if (Q_idx_lane_base + static_cast<int32_t>(fq * 16 + 8 * e) < 0) {
#pragma unroll
                    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
#pragma unroll
                        for (uint32_t sub = 0; sub < 2; sub++) {
                            RO[fq][fv][e * 2 + sub]     = DTypeSVAccum(0.0f);
                            RO[fq][fv][e * 2 + sub + 4] = DTypeSVAccum(0.0f);
                        }
                    }
                    row_max[fq][e] = -INFINITY;
                }
            }
        }
    }
#endif

    normalize_d<num_tiles_q, num_tiles_v, DenominatorAccumUnit>(RO, row_max, denom);

    // save the result
    // if (get_warp_idx_k<num_warps_q, num_warps_k>() == 0)
    // {

    // convert half to bfloat16
    if constexpr (std::is_same<DTypeSVAccum, half>::value && std::is_same<DTypeOut, nv_bfloat16>::value) {
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
                ((nv_bfloat162*)RO[fq][fv])[0] = __float22bfloat162_rn(__half22float2(((half2*)RO[fq][fv])[0]));
                ((nv_bfloat162*)RO[fq][fv])[1] = __float22bfloat162_rn(__half22float2(((half2*)RO[fq][fv])[1]));
                ((nv_bfloat162*)RO[fq][fv])[2] = __float22bfloat162_rn(__half22float2(((half2*)RO[fq][fv])[2]));
                ((nv_bfloat162*)RO[fq][fv])[3] = __float22bfloat162_rn(__half22float2(((half2*)RO[fq][fv])[3]));
            }
        }
    }

    // add v_mean
    if constexpr (fuse_v_mean) {
        DTypeOut2       v_mean[2];
        const DTypeOut* V_mean_base_ptr = V_mean
                                          + static_cast<int64_t>(batch_id) * (num_qo_heads / qo_per_kv_head) * head_dim
                                          + static_cast<int64_t>(head_id / qo_per_kv_head) * head_dim + lane_id % 4 * 2;
#pragma unroll
        for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
            v_mean[0] = *((DTypeOut2*)(V_mean_base_ptr + fv * 16));
            v_mean[1] = *((DTypeOut2*)(V_mean_base_ptr + 8 + fv * 16));
#pragma unroll
            for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
                ((DTypeOut2*)RO[fq][fv])[0] = __hadd2(((DTypeOut2*)RO[fq][fv])[0], v_mean[0]);
                ((DTypeOut2*)RO[fq][fv])[1] = __hadd2(((DTypeOut2*)RO[fq][fv])[1], v_mean[0]);
                ((DTypeOut2*)RO[fq][fv])[2] = __hadd2(((DTypeOut2*)RO[fq][fv])[2], v_mean[1]);
                ((DTypeOut2*)RO[fq][fv])[3] = __hadd2(((DTypeOut2*)RO[fq][fv])[3], v_mean[1]);
            }
        }
    }

    // save the result to shared memory
    uint32_t O_smem_row_base = get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
            uint32_t O_smem_offset =
                smem_O.get_permuted_offset(O_smem_row_base + fq * MMA_QK_M, fv * (MMA_SV_N / PACK_SIZE_O));

            if constexpr (std::is_same<DTypeSVAccum, float>::value) {
                // convert RO to half
                uint32_t RO_f16[4];
#pragma unroll
                for (uint32_t e = 0; e < 4; e++) {
                    if constexpr (std::is_same<DTypeOut, half>::value) {
                        ((half2*)RO_f16)[e] = __float22half2_rn(((float2*)RO[fq][fv])[e]);
                    }
                    else if constexpr (std::is_same<DTypeOut, nv_bfloat16>::value) {
                        ((nv_bfloat162*)RO_f16)[e] = __float22bfloat162_rn(((float2*)RO[fq][fv])[e]);
                    }
                }

                ((uint32_t*)(smem_O.base + O_smem_offset))[lane_id % 4]                                     = RO_f16[0];
                ((uint32_t*)(smem_O.base + O_smem_offset + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[1];

                // ! permuted, make sure you know what you are doing
                ((uint32_t*)(smem_O.base + (O_smem_offset ^ 0x1)))[lane_id % 4] = RO_f16[2];
                ((uint32_t*)(smem_O.base + (O_smem_offset ^ 0x1) + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] =
                    RO_f16[3];
            }
            else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
                ((uint32_t*)(smem_O.base + O_smem_offset))[lane_id % 4] = ((uint32_t*)RO[fq][fv])[0];
                ((uint32_t*)(smem_O.base + O_smem_offset + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] =
                    ((uint32_t*)RO[fq][fv])[1];

                // ! permuted, make sure you know what you are doing
                ((uint32_t*)(smem_O.base + (O_smem_offset ^ 0x1)))[lane_id % 4] = ((uint32_t*)RO[fq][fv])[2];
                ((uint32_t*)(smem_O.base + (O_smem_offset ^ 0x1) + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] =
                    ((uint32_t*)RO[fq][fv])[3];
            }
        }
    }

    // Yes, this sync is required: the O rows each warp just wrote to smem are
    // re-read below by the same warp (num_warps_k == 1 and CTA_Q / num_warps ==
    // WARP_Q make the row sets identical), so __syncwarp is exactly sufficient
    // and __syncthreads would be overkill. The static_assert pins the premise.
    static_assert(num_warps_k == 1 && CTA_Q / num_warps == WARP_Q,
                  "the __syncwarp-only O smem handoff assumes warp-private rows");
    __syncwarp();

    // shared memory to global memory
#ifdef SAGE_VARLEN
    DTypeOut* O_lane_ptr =
        O + static_cast<int64_t>(seq_info.offset_q) * stride_seq_o
        + static_cast<int64_t>(head_id) * stride_h_o
        + static_cast<int64_t>(cta_idx_q * CTA_Q + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>()
                               + lane_id / global_to_shared_line_lanes_O)
              * stride_seq_o
        + static_cast<int64_t>(lane_id % global_to_shared_line_lanes_O * PACK_SIZE_O);
#else
    DTypeOut* O_lane_ptr =
        O + static_cast<int64_t>(batch_id) * stride_batch_o + static_cast<int64_t>(head_id) * stride_h_o
        + static_cast<int64_t>(cta_idx_q * CTA_Q + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>()
                               + lane_id / global_to_shared_line_lanes_O)
              * stride_seq_o
        + static_cast<int64_t>(lane_id % global_to_shared_line_lanes_O * PACK_SIZE_O);
#endif
    uint32_t O_smem_offset = smem_O.get_permuted_offset(get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q
                                                            + lane_id / global_to_shared_line_lanes_O,
                                                        lane_id % global_to_shared_line_lanes_O);
    uint32_t O_load_idx_lane_base =
        cta_idx_q * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_O;

#pragma unroll
    for (uint32_t i = 0; i < O_smem_iters_col; i++) {
#pragma unroll
        for (uint32_t j = 0; j < O_smem_iters_row; j++) {
            if (O_load_idx_lane_base < qo_len) {
                smem_O.store_128b(O_smem_offset, O_lane_ptr);
            }
            O_lane_ptr += (global_to_shared_line_lanes_O * PACK_SIZE_O);
            O_smem_offset = smem_O.advance_offset_by_column<global_to_shared_line_lanes_O>(O_smem_offset);
        }

        O_smem_offset = smem_O.advance_offset_by_row<global_to_shared_copy_lines_per_warp_O>(
            O_smem_offset - (O_smem_iters_row * global_to_shared_line_lanes_O));
        O_lane_ptr += ((global_to_shared_copy_lines_per_warp_O * stride_seq_o)
                       - (O_smem_iters_row * global_to_shared_line_lanes_O * PACK_SIZE_O));
        O_load_idx_lane_base += global_to_shared_copy_lines_per_warp_O;
    }

    if constexpr (return_lse) {
        uint32_t lse_idx =
            cta_idx_q * CTA_Q + lane_id / 4 + 8 * (lane_id % 4) + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>();
#ifdef SAGE_VARLEN
        // lse is [heads, total_tokens]: head-major, and the sequence's rows sit
        // at its token offset.
        float* lse_lane_ptr = Lse + static_cast<int64_t>(head_id) * lse_stride_h + seq_info.offset_q + lse_idx;
#else
        float* lse_lane_ptr = Lse + static_cast<int64_t>(batch_id) * (qo_len * num_qo_heads)
                              + static_cast<int64_t>(head_id) * qo_len + lse_idx;
#endif
        uint32_t fq = (lane_id % 4) / 2;
        uint32_t e  = (lane_id % 4) % 2;

        if (lse_idx < qo_len && (lane_id % 4) < 2 * num_tiles_q) {
            lse_lane_ptr[0] = (math::ptx_log2(denom[fq][e]) + row_max[fq][e]);
        }
    }

    // }
}

}  // namespace SAGEATTN_SM80_NS
}  // namespace sage
