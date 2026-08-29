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
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <torch/types.h>

#include "../cp_async.cuh"
#include "../dispatch_utils.h"
#include "../math.cuh"
#include "../mma.cuh"
#include "../permuted_smem.cuh"

#include "attn_utils.cuh"

#define PACK_SIZE_QK 16  // as if it is int8
#define PACK_SIZE_V 16   // fp8
#define PACK_SIZE_O 8    // fp16

// treat as if int8 tensor core
#define MMA_QK_M 16
#define MMA_QK_N 16
#define MMA_QK_K 32

// fp8 tensor core
#define MMA_SV_M 16
#define MMA_SV_N 16
#define MMA_SV_K 32

// This header (and the launcher built on it) is compiled twice into the
// single _C extension: once per arch namespace (sm89 with sm_89/sm_10x/sm_12x
// SASS as the generic fallback, sm120 with sm_12x-only SASS). The macro
// namespace keeps the two instantiation sets from colliding (ODR).
#ifndef SAGEATTN_ARCH_NS
#define SAGEATTN_ARCH_NS sm89
#endif
namespace sage {
namespace SAGEATTN_ARCH_NS {

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
         MaskMode    mask_mode         = MaskMode::kNone,
         bool        return_lse        = false,
         bool        fuse_v_scale      = false,
         bool        fuse_v_mean       = false,
         bool        use_pv_fp16_accum = false>
__global__ void qk_int_sv_f8_attn_kernel(const int8_t* __restrict__ Q,
                                         const int8_t* __restrict__ K,
                                         const int8_t* __restrict__ V,
                                         DTypeOut* __restrict__ O,
                                         float* __restrict__ Lse,
                                         const float* __restrict__ Q_scale,
                                         const float* __restrict__ K_scale,
                                         const float* __restrict__ V_scale,
                                         const float* __restrict__ V_mean,
                                         const uint32_t qo_len,
                                         const uint32_t kv_len,
                                         const uint32_t qo_per_kv_head,
                                         const int64_t  stride_batch_q,
                                         const uint32_t stride_seq_q,
                                         const int64_t  stride_h_q,
                                         const int64_t  stride_batch_k,
                                         const uint32_t stride_seq_k,
                                         const int64_t  stride_h_k,
                                         const int64_t  stride_batch_v,
                                         const int64_t  stride_h_v,
                                         const uint32_t stride_d_v,
                                         const int64_t  stride_batch_o,
                                         const uint32_t stride_seq_o,
                                         const int64_t  stride_h_o,
                                         float          sm_scale)
{
    // compile time check
    static_assert(DTypeQK == DataType::kInt8, "DTypeQK must be int8");
    static_assert(Q_GRAN == QuantGranularity::kPerBlock || Q_GRAN == QuantGranularity::kPerWarp
                      || Q_GRAN == QuantGranularity::kPerThread,
                  "Q_GRAN must be kPerBlock, kPerWarp or kPerThread");
    static_assert(K_GRAN == QuantGranularity::kPerBlock || K_GRAN == QuantGranularity::kPerWarp
                      || K_GRAN == QuantGranularity::kPerThread,
                  "K_GRAN must be kPerBlock, kPerWarp or kPerThread");
    static_assert(head_dim % 64 == 0, "head_dim must be a multiple of 64");
    static_assert(std::is_same<DTypeSVAccum, float>::value, "DTypeSVAccum must be float, half is WIP");
    static_assert(std::is_same<DTypeOut, half>::value || std::is_same<DTypeOut, nv_bfloat16>::value,
                  "DTypeOut must be half or nv_bfloat16");
    static_assert(CTA_K % 64 == 0);
    static_assert(CTA_Q / CTA_K <= 2);  // for efficient causal implementation

    constexpr uint32_t num_warps_q        = CTA_Q / WARP_Q;
    constexpr uint32_t num_warps_k        = CTA_K / WARP_K;
    constexpr uint32_t num_warps          = num_warps_q * num_warps_k;
    constexpr uint32_t num_tiles_q        = WARP_Q / MMA_QK_M;
    constexpr uint32_t num_tiles_k        = WARP_K / MMA_QK_N;
    constexpr uint32_t num_tiles_qk_inner = head_dim / MMA_QK_K;
    constexpr uint32_t num_tiles_v        = head_dim / MMA_SV_N;

    constexpr uint32_t QK_SMEM_STRIDE = head_dim;
    constexpr uint32_t O_SMEM_STRIDE  = head_dim;
    //                       for fp16: head_dim
    constexpr uint32_t V_SMEM_STRIDE = CTA_K;

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

    // transfer to base 2 instead of base e with better numerical efficiency
    sm_scale *= math::log2e;

    // RS holds the fragment of S
    int32_t      RS[num_tiles_q][num_tiles_k][8];
    DTypeSVAccum RO[num_tiles_q][num_tiles_v][8];
    float        row_max[num_tiles_q][2];  // max
    float        denom[num_tiles_q][2];    // denominator

    // q_scale is read once, so the whole index can stay 64-bit. k_scale is read
    // once per iteration, so it is split into a 64-bit base pointer plus a
    // 32-bit running offset.
    int64_t      q_scale_idx;
    const float* K_scale_base_ptr;
    uint32_t     k_scale_off;

    if constexpr (Q_GRAN == QuantGranularity::kPerBlock) {
        const uint32_t num_ctas_q = gridDim.x;
        q_scale_idx               = static_cast<int64_t>(batch_id) * num_qo_heads * num_ctas_q
                      + static_cast<int64_t>(head_id) * num_ctas_q + cta_idx_q;
    }
    else if constexpr (Q_GRAN == QuantGranularity::kPerWarp) {
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

    if constexpr (K_GRAN == QuantGranularity::kPerBlock) {
        const uint32_t num_ctas_k = div_ceil(kv_len, CTA_K);
        K_scale_base_ptr = K_scale + static_cast<int64_t>(batch_id) * (num_qo_heads / qo_per_kv_head) * num_ctas_k
                           + static_cast<int64_t>(head_id / qo_per_kv_head) * num_ctas_k;
        k_scale_off = 0;
    }
    else if constexpr (K_GRAN == QuantGranularity::kPerWarp) {
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

    constexpr uint32_t k_scale_advance_offset = (K_GRAN == QuantGranularity::kPerBlock) ? 1 :
                                                (K_GRAN == QuantGranularity::kPerWarp)  ? (CTA_K / WARP_K) :
                                                                                          (CTA_K / WARP_K) * 4;

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

    constexpr uint32_t K_smem_row_offset = CTA_Q;
    constexpr uint32_t V_smem_row_offset = CTA_Q + CTA_K;

    constexpr SwizzleMode swizzle_mode_QK = (QK_SMEM_STRIDE == 32) ? SwizzleMode::k32B :
                                            (QK_SMEM_STRIDE == 64) ? SwizzleMode::k64B :
                                                                     SwizzleMode::k128B;
    smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_Q(smem);
    smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_K(smem + K_smem_row_offset * QK_SMEM_STRIDE);
    //                                             for fp16: 32
    constexpr SwizzleMode swizzle_mode_V = (V_SMEM_STRIDE == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
    smem_t<swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V> smem_V(smem + V_smem_row_offset * QK_SMEM_STRIDE);
    constexpr SwizzleMode swizzle_mode_O = (O_SMEM_STRIDE == 32) ? SwizzleMode::k64B : SwizzleMode::k128B;
    smem_t<swizzle_mode_O, O_SMEM_STRIDE / PACK_SIZE_O> smem_O(smem);

    constexpr uint32_t global_to_shared_line_lanes_QK = (QK_SMEM_STRIDE == 32) ? 2 : (QK_SMEM_STRIDE == 64) ? 4 : 8;
    constexpr uint32_t global_to_shared_copy_lines_per_warp_QK = (QK_SMEM_STRIDE == 32) ? 16 :
                                                                 (QK_SMEM_STRIDE == 64) ? 8 :
                                                                                          4;
    //                                                         for fp16: 32
    constexpr uint32_t global_to_shared_line_lanes_V = (V_SMEM_STRIDE == 64) ? 4 : 8;
    //                                                                  for fp16: 32
    constexpr uint32_t global_to_shared_copy_lines_per_warp_V = (V_SMEM_STRIDE == 64) ? 8 : 4;
    constexpr uint32_t global_to_shared_line_lanes_O          = (O_SMEM_STRIDE == 32) ? 4 : 8;
    constexpr uint32_t global_to_shared_copy_lines_per_warp_O = (O_SMEM_STRIDE == 32) ? 8 : 4;

    constexpr uint32_t QK_smem_iters_row = QK_SMEM_STRIDE / (global_to_shared_line_lanes_QK * PACK_SIZE_QK);
    constexpr uint32_t Q_smem_iters_col  = CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_QK);
    constexpr uint32_t K_smem_iters_col  = CTA_K / (num_warps * global_to_shared_copy_lines_per_warp_QK);
    constexpr uint32_t V_smem_iters_row  = V_SMEM_STRIDE / (global_to_shared_line_lanes_V * PACK_SIZE_V);
    //                          for fp16: CTA_K
    constexpr uint32_t V_smem_iters_col = head_dim / (num_warps * global_to_shared_copy_lines_per_warp_V);
    constexpr uint32_t O_smem_iters_row = O_SMEM_STRIDE / (global_to_shared_line_lanes_O * PACK_SIZE_O);
    constexpr uint32_t O_smem_iters_col = CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_O);

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
    //                                                                for fp16: CTA_K / num_warps * warp_id *
    //                                                                stride_seq_v + lane_id /
    //                                                                global_to_shared_line_lanes_V * stride_seq_v
    const int8_t* V_lane_base_ptr =
        V + static_cast<int64_t>(batch_id) * stride_batch_v
        + static_cast<int64_t>(head_id / qo_per_kv_head) * stride_h_v
        + static_cast<int64_t>(head_dim / num_warps * warp_id + lane_id / global_to_shared_line_lanes_V) * stride_d_v
        + static_cast<int64_t>((lane_id % global_to_shared_line_lanes_V) * PACK_SIZE_V);
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
    // for fp 16:
    // uint32_t V_smem_offset_mma = smem_V.get_permuted_offset(get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K +
    // lane_id % 16, lane_id / 16);
    uint32_t V_smem_offset_mma = smem_V.get_permuted_offset(
        lane_id % 8 + (lane_id / 16) * 8,
        get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K / PACK_SIZE_V + (lane_id / 8) % 2);

    // for causal masking
    uint32_t Q_idx_lane_base = cta_idx_q * CTA_Q + get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
    uint32_t K_idx_lane_base = get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + 2 * (lane_id % 4);

    // for loading
    uint32_t Q_load_idx_lane_base =
        cta_idx_q * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;
    uint32_t K_load_idx_lane_base = CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;

    const uint32_t num_iterations =
        div_ceil(mask_mode == MaskMode::kCausal ? min(kv_len, (cta_idx_q + 1) * CTA_Q) : kv_len, CTA_K);

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

    // No drain here: nothing reads smem_Q before the main loop, and the loop's
    // wait_group<1> retires the older Q group together with K. Q/K/V therefore
    // go out as three back-to-back cp.async groups.

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

    float q_scale = Q_scale[q_scale_idx];

    float original_sm_scale = sm_scale;
    float dequant_scale     = q_scale * K_scale_base_ptr[k_scale_off + 0 * k_scale_advance_offset];

    sm_scale = original_sm_scale * dequant_scale;

    // load V
    // ! we assume that V is padded. If not, there might be illegal memory access or nan issue.
    // for fp16:
    // load_global_to_shared                stride_seq_v
    load_fp8_V_global_to_shared<global_to_shared_line_lanes_V,
                                global_to_shared_copy_lines_per_warp_V,
                                V_smem_iters_row,
                                V_smem_iters_col,
                                swizzle_mode_V,
                                V_SMEM_STRIDE / PACK_SIZE_V,
                                CTA_K>(&V_lane_base_ptr, V_smem_offset_load, stride_d_v, smem_V);
    cp_async::commit_group();

    K_load_idx_lane_base += CTA_K;

    // ------------------------------------------------------------- KV tile body
    // The three call sites below run the same pipeline stage and differ only in
    // compile-time bits:
    //   kBulk: no mask. The K dequant scale is folded into sm_scale, so S stays
    //          unscaled here; the next K tile is prefetched unpredicated.
    //   kMask: causal mask, so S is dequantized explicitly and update_mdo gets the
    //          raw sm_scale instead; the next K tile needs bound predicates.
    //   kLast: kMask plus the out-of-bound mask, and no prefetch -- V is then the
    //          only cp.async group still in flight, hence wait_group<0>.
    // next_iter indexes the K scale of the tile being prefetched (unused by kLast).
    auto process_tile = [&](auto kind_t, uint32_t next_iter) {
        constexpr TileKind kind = decltype(kind_t)::value;

        // ensure K is ready
        cp_async::wait_group<1>();
        __syncthreads();

        // compute QK^T
        compute_int_qk<num_warps_q,
                       num_warps_k,
                       num_tiles_q,
                       num_tiles_k,
                       num_tiles_qk_inner,
                       swizzle_mode_QK,
                       QK_SMEM_STRIDE / PACK_SIZE_QK,
                       DTypeQK>(smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);

        float RS_f32[num_tiles_q][num_tiles_k][8];

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

        if constexpr (kind != TileKind::kBulk && mask_mode == MaskMode::kCausal) {
            apply_causal_mask<num_tiles_q, num_tiles_k>(Q_idx_lane_base, K_idx_lane_base, RS_f32);
        }
        if constexpr (kind == TileKind::kLast) {
            apply_out_of_bounds_mask<num_tiles_q, num_tiles_k>(K_idx_lane_base, RS_f32, kv_len);
        }
        K_idx_lane_base += CTA_K;

        // kBulk already carries dequant_scale inside sm_scale; the masked tiles
        // applied it to RS_f32 above and so pass the unscaled softmax scale.
        const float mdo_sm_scale = (kind == TileKind::kBulk) ? sm_scale : original_sm_scale;

        if constexpr (std::is_same<DTypeSVAccum, float>::value) {
            update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(
                RS_f32, RO, row_max, denom, mdo_sm_scale);
        }
        else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
            update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, true, true, false>(
                RS_f32, RO, row_max, denom, mdo_sm_scale);
        }

        if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore) {
            accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, denom);
        }

        uint32_t RS_f8[num_tiles_q][num_tiles_k / 2][4];
        RS_f32_to_f8<num_tiles_q, num_tiles_k>(RS_f32, RS_f8);

        if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore) {
            accumulate_d_f8<num_tiles_q, num_tiles_k>(RS_f8, denom);
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

        // for fp16:
        // compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V,
        // V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        //   smem_V, RS_f16, RO, denom, V_smem_offset_mma);
        if constexpr (!use_inst_buf) {
            compute_fp8_sv<num_warps_q,
                           num_warps_k,
                           num_tiles_q,
                           num_tiles_k,
                           num_tiles_v,
                           swizzle_mode_V,
                           V_SMEM_STRIDE / PACK_SIZE_V>(smem_V, RS_f8, RO, denom);
        }
        else {
            if constexpr (!use_pv_fp16_accum) {
                compute_fp8_sv_inst_buf<num_warps_q,
                                        num_warps_k,
                                        num_tiles_q,
                                        num_tiles_k,
                                        num_tiles_v,
                                        swizzle_mode_V,
                                        V_SMEM_STRIDE / PACK_SIZE_V>(smem_V, RS_f8, RO, denom);
            }
            else {
                compute_fp8_sv_inst_buf_fp16_accum<num_warps_q,
                                                   num_warps_k,
                                                   num_tiles_q,
                                                   num_tiles_k,
                                                   num_tiles_v,
                                                   swizzle_mode_V,
                                                   V_SMEM_STRIDE / PACK_SIZE_V>(smem_V, RS_f8, RO, denom);
            }
        }

        __syncthreads();

        if constexpr (kind != TileKind::kLast) {
            // load V
            // for fp16:
            // load_global_to_shared                stride_seq_v
            load_fp8_V_global_to_shared<global_to_shared_line_lanes_V,
                                        global_to_shared_copy_lines_per_warp_V,
                                        V_smem_iters_row,
                                        V_smem_iters_col,
                                        swizzle_mode_V,
                                        V_SMEM_STRIDE / PACK_SIZE_V,
                                        CTA_K>(&V_lane_base_ptr, V_smem_offset_load, stride_d_v, smem_V);
            cp_async::commit_group();

            K_load_idx_lane_base += CTA_K;
        }
    };

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

    // TODO: thread block sync mdo state for num_warps_k > 0. Then only one thread block needs to do the final saving.

    normalize_d<num_tiles_q, num_tiles_v, ComputeUnit::kCudaCore>(RO, row_max, denom);

    // ! here we just implement the case for fp32 acumulation
#if defined(SAGE_OPT_FUSED_EPILOGUE)
    // One pass over RO instead of two. With both scales live this replaces
    // 8 FMUL + 8 FADD per tile with 8 FFMA and drops one rounding (the reference
    // rounds RO * v_scale before adding v_mean), so it is strictly more accurate.
    // The scale-only variants keep the same arithmetic, just merged loop nests.
    // v_scale index per e is (e / 4) * 2 + (e % 2): 0,1,0,1,2,3,2,3.
    if constexpr (fuse_v_scale || fuse_v_mean) {
        const int64_t v_row_off = static_cast<int64_t>(batch_id) * (num_qo_heads / qo_per_kv_head) * head_dim
                                  + static_cast<int64_t>(head_id / qo_per_kv_head) * head_dim + (lane_id % 4) * 2;
        const float* V_scale_base_ptr = fuse_v_scale ? V_scale + v_row_off : nullptr;
        const float* V_mean_base_ptr  = fuse_v_mean ? V_mean + v_row_off : nullptr;
        float        v_scale[4];
        float        v_mean[4];
#pragma unroll
        for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
            if constexpr (fuse_v_scale) {
                ((float2*)v_scale)[0] = *((float2*)(V_scale_base_ptr + fv * 16));
                ((float2*)v_scale)[1] = *((float2*)(V_scale_base_ptr + fv * 16 + 8));
            }
            if constexpr (fuse_v_mean) {
                ((float2*)v_mean)[0] = *((float2*)(V_mean_base_ptr + fv * 16));
                ((float2*)v_mean)[1] = *((float2*)(V_mean_base_ptr + fv * 16 + 8));
            }
#pragma unroll
            for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
                for (uint32_t e = 0; e < 8; e++) {
                    const uint32_t vi = (e / 4) * 2 + (e % 2);
                    if constexpr (fuse_v_scale && fuse_v_mean) {
                        RO[fq][fv][e] = __fmaf_rn(RO[fq][fv][e], v_scale[vi], v_mean[vi]);
                    }
                    else if constexpr (fuse_v_scale) {
                        RO[fq][fv][e] *= v_scale[vi];
                    }
                    else {
                        RO[fq][fv][e] += v_mean[vi];
                    }
                }
            }
        }
    }
#else
    if constexpr (fuse_v_scale) {
        float        v_scale[4];
        const float* V_scale_base_ptr = V_scale
                                        + static_cast<int64_t>(batch_id) * (num_qo_heads / qo_per_kv_head) * head_dim
                                        + static_cast<int64_t>(head_id / qo_per_kv_head) * head_dim + (lane_id % 4) * 2;
#pragma unroll
        for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
            ((float2*)v_scale)[0] = *((float2*)(V_scale_base_ptr + fv * 16));
            ((float2*)v_scale)[1] = *((float2*)(V_scale_base_ptr + fv * 16 + 8));
#pragma unroll
            for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
                RO[fq][fv][0] *= v_scale[0];
                RO[fq][fv][1] *= v_scale[1];
                RO[fq][fv][2] *= v_scale[0];
                RO[fq][fv][3] *= v_scale[1];
                RO[fq][fv][4] *= v_scale[2];
                RO[fq][fv][5] *= v_scale[3];
                RO[fq][fv][6] *= v_scale[2];
                RO[fq][fv][7] *= v_scale[3];
            }
        }
    }

    if constexpr (fuse_v_mean) {
        float        v_mean[4];
        const float* V_mean_base_ptr = V_mean
                                       + static_cast<int64_t>(batch_id) * (num_qo_heads / qo_per_kv_head) * head_dim
                                       + static_cast<int64_t>(head_id / qo_per_kv_head) * head_dim + (lane_id % 4) * 2;
#pragma unroll
        for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
            ((float2*)v_mean)[0] = *((float2*)(V_mean_base_ptr + fv * 16));
            ((float2*)v_mean)[1] = *((float2*)(V_mean_base_ptr + fv * 16 + 8));
#pragma unroll
            for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
                RO[fq][fv][0] += v_mean[0];
                RO[fq][fv][1] += v_mean[1];
                RO[fq][fv][2] += v_mean[0];
                RO[fq][fv][3] += v_mean[1];
                RO[fq][fv][4] += v_mean[2];
                RO[fq][fv][5] += v_mean[3];
                RO[fq][fv][6] += v_mean[2];
                RO[fq][fv][7] += v_mean[3];
            }
        }
    }
#endif

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
                    else {
                        ((nv_bfloat162*)RO_f16)[e] = __float22bfloat162_rn(((float2*)RO[fq][fv])[e]);
                    }
                }

                ((uint32_t*)(smem_O.base + O_smem_offset))[lane_id % 4]                                     = RO_f16[0];
                ((uint32_t*)(smem_O.base + O_smem_offset + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[1];

                O_smem_offset =
                    smem_O.get_permuted_offset(O_smem_row_base + fq * MMA_QK_M, fv * (MMA_SV_N / PACK_SIZE_O) + 1);
                ((uint32_t*)(smem_O.base + O_smem_offset))[lane_id % 4]                                     = RO_f16[2];
                ((uint32_t*)(smem_O.base + O_smem_offset + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[3];
            }
            else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
                // TODO: not implement
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
    DTypeOut* O_lane_ptr =
        O + static_cast<int64_t>(batch_id) * stride_batch_o + static_cast<int64_t>(head_id) * stride_h_o
        + static_cast<int64_t>(cta_idx_q * CTA_Q + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>()
                               + lane_id / global_to_shared_line_lanes_O)
              * stride_seq_o
        + static_cast<int64_t>(lane_id % global_to_shared_line_lanes_O * PACK_SIZE_O);
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
        // ! this only works for num_tiles_q = 2
        uint32_t lse_idx =
            cta_idx_q * CTA_Q + lane_id / 4 + 8 * (lane_id % 4) + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>();
        float* lse_lane_ptr = Lse + static_cast<int64_t>(batch_id) * (qo_len * num_qo_heads)
                              + static_cast<int64_t>(head_id) * qo_len + lse_idx;
        uint32_t fq = (lane_id % 4) / 2;
        uint32_t e  = (lane_id % 4) % 2;

        if (lse_idx < qo_len) {
            lse_lane_ptr[0] = (math::ptx_log2(denom[fq][e]) + row_max[fq][e]);
        }
    }
}

}  // namespace SAGEATTN_ARCH_NS
}  // namespace sage
