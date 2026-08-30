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

// The sm90 int8-QK / fp8-PV attention kernel. Split out of
// qk_int_sv_f8_cuda_sm90.cu (which keeps the host launchers) so that a
// second translation unit can include the same body; the split is a pure
// move, the kernel text below is unchanged.
//
// That second translation unit is qk_int_sv_f8_cuda_sm90_varlen.cu, which
// compiles this body with SAGE_VARLEN defined: q/k/v are packed to
// [total_tokens, heads, head_dim] and blockIdx.z selects a sequence of a
// cu_seqlens prefix sum instead of a batch entry. The choice of #ifdef over a
// `bool kVarlen` template parameter is what keeps the dense kernel's SASS
// byte-identical - a template parameter changes the kernel's parameter space
// and its constant-bank layout, so the dense instantiation could not be
// unchanged even where the code is.

#include "../utils.cuh"
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/types.h>

#include "../dispatch_utils.h"
#include "../math.cuh"
#include "../wgmma.cuh"

#include "../sageattn/seqlen_info.cuh"
#include "../tma.cuh"
#include "attn_utils.cuh"

// All sm90 kernels and launchers live in sage::sm90 (single-so ODR rule). The
// varlen translation unit compiles a *different* body under the same template
// name and needs its own namespace for the same reason.
#ifndef SAGEATTN_SM90_NS
#define SAGEATTN_SM90_NS sm90
#endif

namespace sage {
namespace SAGEATTN_SM90_NS {

template<uint32_t         CTA_Q,
         uint32_t         CTA_K,
         uint32_t         NUM_THREADS,
         uint32_t         head_dim,
         QuantGranularity Q_GRAN,
         QuantGranularity K_GRAN,
         typename DTypeOut,
         MaskMode mask_mode    = MaskMode::kNone,
         bool     return_lse   = false,
         bool     fuse_v_scale = false>
__global__ void qk_int8_sv_f8_attn_kernel(const __grid_constant__ CUtensorMap tensorMapQ,
                                          const __grid_constant__ CUtensorMap tensorMapK,
                                          const __grid_constant__ CUtensorMap tensorMapV,
                                          const float* __restrict__ Q_scale,
                                          const float* __restrict__ K_scale,
                                          const float* __restrict__ V_scale,
                                          DTypeOut* O,
                                          float* __restrict__ Lse,
                                          const int64_t  stride_batch_o,
                                          const int64_t  stride_h_o,
                                          uint32_t       stride_seq_o,
#ifdef SAGE_VARLEN
                                          // [batch_size + 1] prefix sums; the sequence lengths, the scale
                                          // block bases and the padded V^T slab base are derived from them
                                          // per block. The *_stride_h are the packed tensors' per-head
                                          // extents, which the dense kernel instead reads out of gridDim
                                          // (query scale) or recomputes from kv_len (key scale).
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
                                          float          sm_scale)
{
    static_assert(NUM_THREADS == 128);
    static_assert(CTA_Q <= CTA_K);

    const uint32_t warp_idx = (threadIdx.x % 128) / 32;
    const uint32_t lane_id  = threadIdx.x % 32;

    constexpr uint32_t num_tiles_q        = CTA_Q / 64;
    constexpr uint32_t num_tiles_k        = CTA_K / 16;
    constexpr uint32_t num_tiles_qk_inner = head_dim / 32;
    constexpr uint32_t num_tiles_v        = head_dim / 16;
    constexpr uint32_t num_tiles_pv_inner = CTA_K / 32;

    const uint32_t batch_id     = blockIdx.z;
    const uint32_t cta_idx_q    = blockIdx.x;
    const uint32_t head_id      = blockIdx.y;
    const uint32_t num_qo_heads = gridDim.y;
    const uint32_t kv_head_id   = head_id / qo_per_kv_head;

#ifdef SAGE_VARLEN
    // blockIdx.z is a sequence of the prefix sums, not a batch entry. From here
    // down every length and every index is sequence-relative, exactly as the
    // dense kernel's are relative to its batch entry; the sequence's absolute
    // position enters the TMA coordinates, the scale block bases and the O / lse
    // pointers, and nowhere else.
    const SeqlenInfo<true, CTA_Q, CTA_K> seq_info(cu_seqlens_q, cu_seqlens_k, batch_id);
    const uint32_t                       qo_len = seq_info.seqlen_q;
    const uint32_t                       kv_len = seq_info.seqlen_k;
    // The grid is opened to max_seqlen_q, so most sequences leave part of it
    // empty. Block-uniform, and ahead of the barrier init below.
    if (cta_idx_q * CTA_Q >= qo_len) {
        return;
    }

    // Bottom-right alignment (flash-attention semantics): row q of the sequence
    // attends to keys up to q + delta. delta is signed - kv shorter than qo is
    // legal - so the whole trip-count computation is int32; the dense uint32
    // form would wrap a negative bound into an enormous number of iterations.
    const int32_t causal_bound = static_cast<int32_t>((cta_idx_q + 1) * CTA_Q) + seq_info.delta;
    const int32_t kv_bound     = (mask_mode == MaskMode::kCausal) ? min(static_cast<int32_t>(kv_len), causal_bound) :
                                                                    static_cast<int32_t>(kv_len);
    const int32_t num_iterations = div_ceil(kv_bound, static_cast<int32_t>(CTA_K));

    // The first KV tile holding a masked element. The dense kernel masks
    // exactly one tile (the epilogue's) because CTA_Q <= CTA_K puts the whole
    // 64-wide diagonal band inside one 128-wide tile; with delta not a multiple
    // of CTA_K the band straddles two. Clamping to num_iterations - 1 makes
    // delta == 0 reproduce the dense tile structure exactly, which is a
    // numerical requirement and not only a speed one: an unmasked tile folds
    // the K dequant scale into sm_scale while a masked one multiplies it into
    // S, so masking one tile more or fewer changes the result bits.
    const int32_t first_masked_tile =
        (mask_mode == MaskMode::kCausal) ?
            max(0,
                min(num_iterations - 1,
                    (static_cast<int32_t>(cta_idx_q * CTA_Q) + seq_info.delta) / static_cast<int32_t>(CTA_K))) :
            num_iterations - 1;

    if (num_iterations <= 0) {
        // No KV tile at all: an empty key sequence, or a causal CTA whose rows
        // admit no key. O is a zero row and lse is -inf. Leaving from here,
        // before the barriers are initialized, keeps no TMA in flight.
        DTypeOut* O_zero_ptr = O + static_cast<int64_t>(seq_info.offset_q) * stride_seq_o
                               + static_cast<int64_t>(head_id) * stride_h_o
                               + static_cast<int64_t>(cta_idx_q * CTA_Q + warp_idx * 16 + (lane_id / 4)) * stride_seq_o
                               + static_cast<int64_t>((lane_id % 4) * 2);
        const uint32_t O_zero_idx = cta_idx_q * CTA_Q + warp_idx * 16 + lane_id / 4;
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fv = 0; fv < head_dim / 16; fv++) {
                // DTypeOut is half or nv_bfloat16; a zero of either is a zero
                // bit pattern, and the pair store matches the dense epilogue's
                // 4-byte alignment.
                if (O_zero_idx + fq * 64 < qo_len) {
                    *reinterpret_cast<uint32_t*>(O_zero_ptr + fq * 64 * stride_seq_o + fv * 16)     = 0u;
                    *reinterpret_cast<uint32_t*>(O_zero_ptr + fq * 64 * stride_seq_o + fv * 16 + 8) = 0u;
                }
                if (O_zero_idx + fq * 64 + 8 < qo_len) {
                    *reinterpret_cast<uint32_t*>(O_zero_ptr + fq * 64 * stride_seq_o + fv * 16 + 8 * stride_seq_o) = 0u;
                    *reinterpret_cast<uint32_t*>(O_zero_ptr + fq * 64 * stride_seq_o + fv * 16 + 8 + 8 * stride_seq_o) =
                        0u;
                }
            }
        }

        if constexpr (return_lse) {
            const uint32_t lse_idx = cta_idx_q * CTA_Q + lane_id / 4 + 8 * (lane_id % 4) + 16 * warp_idx;
            if (lse_idx < qo_len && (lane_id % 4) < 2) {
                Lse[static_cast<int64_t>(head_id) * lse_stride_h + seq_info.offset_q + lse_idx] = -INFINITY;
            }
        }
        return;
    }
#endif

    sm_scale *= math::log2e;

    extern __shared__ __align__(128) int8_t smem_[];

    int8_t* sQ = (int8_t*)smem_;
    int8_t* sK = (int8_t*)(smem_ + CTA_Q * head_dim * sizeof(int8_t));
    int8_t* sV = (int8_t*)(smem_ + CTA_Q * head_dim * sizeof(int8_t) + CTA_K * head_dim * sizeof(int8_t));
    half*   sO = (half*)smem_;

    int32_t RS[num_tiles_q][num_tiles_k][8];
    float   RO[num_tiles_q][num_tiles_v][8];
    float   row_max[num_tiles_q][2];
    float   denom[num_tiles_q][2];

    // Q scale is read once: the whole index may live in 64-bit. K scale is read
    // once per KV tile, so it is split into a 64-bit base pointer plus a 32-bit
    // running offset that the main loop advances.
    int64_t      q_scale_idx;
    const float* K_scale_base_ptr;
    uint32_t     k_scale_off;

#ifdef SAGE_VARLEN
    // The packed scales are [heads, blocks]: the batch dimension is gone and
    // the sequence's own blocks start at varlen.h's blk_offset, which is the
    // same expression the quantization kernel wrote them with. Deriving the
    // per-head extent from gridDim (the dense form below) or from kv_len would
    // be wrong here twice over - the grid covers max_seqlen, and the blocks of
    // the earlier sequences sit in front of this one's.
    constexpr uint32_t q_scale_per_cta = (Q_GRAN == QuantGranularity::kPerWarp)   ? (NUM_THREADS / 32) :
                                                                                    (NUM_THREADS / 32) * 8;
    constexpr uint32_t k_scale_per_cta = (K_GRAN == QuantGranularity::kPerWarp) ? 1 : 4;

    q_scale_idx = static_cast<int64_t>(head_id) * q_scale_stride_h
                  + static_cast<int64_t>(seq_info.blk_q_base + cta_idx_q) * q_scale_per_cta;
    if constexpr (Q_GRAN == QuantGranularity::kPerWarp) {
        q_scale_idx += warp_idx;
    }
    else if constexpr (Q_GRAN == QuantGranularity::kPerThread) {
        q_scale_idx += warp_idx * 8 + lane_id / 4;
    }

    K_scale_base_ptr = K_scale + static_cast<int64_t>(kv_head_id) * k_scale_stride_h
                       + static_cast<int64_t>(seq_info.blk_k_base) * k_scale_per_cta;
    k_scale_off = (K_GRAN == QuantGranularity::kPerThread) ? (lane_id % 4) : 0;
#else
    if constexpr (Q_GRAN == QuantGranularity::kPerWarp) {
        const uint32_t num_warp_tiles_q = gridDim.x * 4;
        q_scale_idx                     = static_cast<int64_t>(batch_id) * num_qo_heads * num_warp_tiles_q
                      + static_cast<int64_t>(head_id) * num_warp_tiles_q + cta_idx_q * 4 + warp_idx;
    }
    else if constexpr (Q_GRAN == QuantGranularity::kPerThread) {
        const uint32_t num_warp_tiles_q = gridDim.x * 4;
        q_scale_idx                     = static_cast<int64_t>(batch_id) * num_qo_heads * (num_warp_tiles_q * 8)
                      + static_cast<int64_t>(head_id) * (num_warp_tiles_q * 8) + cta_idx_q * (4 * 8) + warp_idx * 8
                      + lane_id / 4;
    }

    if constexpr (K_GRAN == QuantGranularity::kPerWarp) {
        const uint32_t num_ctas_k = div_ceil(kv_len, CTA_K);
        K_scale_base_ptr = K_scale + static_cast<int64_t>(batch_id) * (num_qo_heads / qo_per_kv_head) * num_ctas_k
                           + static_cast<int64_t>(head_id / qo_per_kv_head) * num_ctas_k;
        k_scale_off = 0;
    }
    else if constexpr (K_GRAN == QuantGranularity::kPerThread) {
        const uint32_t num_ctas_k = div_ceil(kv_len, CTA_K);
        K_scale_base_ptr = K_scale + static_cast<int64_t>(batch_id) * (num_qo_heads / qo_per_kv_head) * (num_ctas_k * 4)
                           + static_cast<int64_t>(head_id / qo_per_kv_head) * (num_ctas_k * 4);
        k_scale_off = lane_id % 4;
    }
#endif

    constexpr uint32_t k_scale_advance_offset = (K_GRAN == QuantGranularity::kPerWarp) ? 1 : 4;

    uint32_t Q_idx_lane_base = cta_idx_q * CTA_Q + warp_idx * 16 + lane_id / 4;
#ifdef SAGE_VARLEN
    // Q_idx_lane_base serves three readers on sm90 - the causal mask, the O
    // store bound and (through lse_idx) the lse store - and only the mask wants
    // the bottom-right shift. So the shifted row index is a second, signed
    // variable and the store bounds keep using the sequence-relative one.
    const int32_t Q_idx_mask_base = static_cast<int32_t>(Q_idx_lane_base) + seq_info.delta;
#endif

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        row_max[fq][0] = -5000000.0f;
        row_max[fq][1] = -5000000.0f;
        denom[fq][0]   = 1.0f;
        denom[fq][1]   = 1.0f;
    }

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
#pragma unroll
            for (uint32_t e = 0; e < 8; e++) {
                RO[fq][fv][e] = 0.0f;
            }
        }
    }

    __shared__ __align__(8) uint64_t barrier_Q;
    __shared__ __align__(8) uint64_t barrier_K;
    __shared__ __align__(8) uint64_t barrier_V;

    if (threadIdx.x == 0) {
        init_barrier(&barrier_Q, 1);
        init_barrier(&barrier_K, 1);
        init_barrier(&barrier_V, 1);
    }

    __syncthreads();

    // load Q, K, V
    if (threadIdx.x == 0) {
        expect_bytes<(CTA_Q * head_dim) * sizeof(int8_t)>(&barrier_Q);
        expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_K);
        expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_V);
#ifdef SAGE_VARLEN
        // The tensor maps are rank-4 with a batch extent of 1, so the batch
        // coordinate is always 0 and the sequence's offset rides in the token
        // coordinate instead. That is what keeps the maps free of any
        // cu_seqlens value and therefore safe to capture in a cudagraph.
        //
        // Q and K move in tokens; V is the block-padded V^T slab, so it moves
        // in varlen.h's pad_offset - which is blk_k_base * CTA_K exactly
        // because fwd_cuda pins v_pad_multiple == blk_k.
        load_async_4D(sQ, &tensorMapQ, &barrier_Q, 0, seq_info.offset_q + cta_idx_q * CTA_Q, head_id, 0);
        load_async_4D(sK, &tensorMapK, &barrier_K, 0, seq_info.offset_k, kv_head_id, 0);
        load_async_4D(sV, &tensorMapV, &barrier_V, seq_info.blk_k_base * CTA_K, 0, kv_head_id, 0);
#else
        load_async_4D(sQ, &tensorMapQ, &barrier_Q, 0, cta_idx_q * CTA_Q, head_id, batch_id);
        load_async_4D(sK, &tensorMapK, &barrier_K, 0, 0, kv_head_id, batch_id);
        load_async_4D(sV, &tensorMapV, &barrier_V, 0, 0, kv_head_id, batch_id);
#endif
    }

    float q_scale           = Q_scale[q_scale_idx];
    float original_sm_scale = sm_scale;

    // wait for Q
    wait(&barrier_Q, 0);

#ifndef SAGE_VARLEN
    const uint32_t num_iterations =
        div_ceil(mask_mode == MaskMode::kCausal ? min(kv_len, (cta_idx_q + 1) * CTA_Q) : kv_len, CTA_K);
#endif

    int phase = 1;
#ifdef SAGE_VARLEN
    for (int32_t iter = 1; iter < num_iterations; iter++) {
#else
    for (uint32_t iter = 1; iter < num_iterations; iter++) {
#endif
        phase ^= 1;

#ifdef SAGE_VARLEN
        // Iteration `iter` processes tile iter - 1 and prefetches tile iter.
        // With delta == 0 first_masked_tile is num_iterations - 1, so this is
        // false throughout and the loop is the dense one: the same single FMUL
        // folding the dequant scale into sm_scale, the same unmasked S.
        const bool tile_masked = (iter - 1 >= first_masked_tile);

        float dequant_scale = q_scale * K_scale_base_ptr[k_scale_off + (iter - 1) * k_scale_advance_offset];
        sm_scale            = tile_masked ? original_sm_scale : original_sm_scale * dequant_scale;
#else
        float dequant_scale = q_scale * K_scale_base_ptr[k_scale_off + (iter - 1) * k_scale_advance_offset];
        sm_scale            = original_sm_scale * dequant_scale;
#endif

        // wait for K
        wait(&barrier_K, phase);

        // compute QK^T
        wgmma::warpgroup_arrive();
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
            int8_t* sQ_local = sQ + fq * 64 * head_dim;
            wgmma::wgmma_s8s8s32<CTA_K, 0, head_dim>(RS[fq], sQ_local, sK);
#pragma unroll
            for (int k_it = 1; k_it < num_tiles_qk_inner; k_it++) {
                wgmma::wgmma_s8s8s32<CTA_K, 1, head_dim>(RS[fq], &sQ_local[k_it * 32], &sK[k_it * 32]);
            }
        }
        wgmma::warpgroup_commit_batch();
        wgmma::warpgroup_wait<0>();

        // load K
        if (threadIdx.x == 0) {
            expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_K);
#ifdef SAGE_VARLEN
            // A tile that runs off the end of this sequence reads the next
            // one's keys; the epilogue's kv_idx >= kv_len test masks them, and
            // past total_tokens the tensor map's out-of-bounds fill takes over.
            load_async_4D(sK, &tensorMapK, &barrier_K, 0, seq_info.offset_k + iter * CTA_K, kv_head_id, 0);
#else
            load_async_4D(sK, &tensorMapK, &barrier_K, 0, iter * CTA_K, kv_head_id, batch_id);
#endif
        }

        // convert RS to float
        float RS_f32[num_tiles_q][num_tiles_k][8];
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
                for (uint32_t e = 0; e < 8; e++) {
                    RS_f32[fq][fk][e] = __int2float_rz(RS[fq][fk][e]);
                }
            }
        }

#ifdef SAGE_VARLEN
        // The extra masked tile bottom-right alignment can produce. It is the
        // epilogue's arithmetic minus the kv_len bound, which cannot bite here:
        // every key of a tile before the last is below (num_iterations - 1) *
        // CTA_K <= kv_bound <= kv_len. Reached only when delta pushes the
        // diagonal band across a CTA_K boundary; with delta == 0 the branch is
        // dead and this loop is the dense one.
        if constexpr (mask_mode == MaskMode::kCausal) {
            if (tile_masked) {
#pragma unroll
                for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
                    for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
                        for (uint32_t e = 0; e < 8; e++) {
                            const int32_t q_idx  = Q_idx_mask_base + static_cast<int32_t>(fq * 64 + 8 * ((e % 4) / 2));
                            const int32_t kv_idx = (iter - 1) * static_cast<int32_t>(CTA_K)
                                                   + static_cast<int32_t>(fk * 16 + 2 * (lane_id % 4) + 8 * (e / 4)
                                                                          + e % 2);

                            RS_f32[fq][fk][e] =
                                (kv_idx > q_idx) ? -5000000.0f : RS_f32[fq][fk][e] * dequant_scale;
                        }
                    }
                }
            }
        }
#endif

        update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(RS_f32, RO, row_max, denom, sm_scale);

        // accumulate denom on thread basis
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
                denom[fq][0] += (RS_f32[fq][fk][0] + RS_f32[fq][fk][1] + RS_f32[fq][fk][4] + RS_f32[fq][fk][5]);
                denom[fq][1] += (RS_f32[fq][fk][2] + RS_f32[fq][fk][3] + RS_f32[fq][fk][6] + RS_f32[fq][fk][7]);
            }
        }

        uint32_t RS_f8[num_tiles_q][num_tiles_pv_inner][4];
        RS_f32_to_f8<num_tiles_q, num_tiles_k>(RS_f32, RS_f8);

        // wait for V
        wait(&barrier_V, phase);

        float RO_tmp[num_tiles_q][num_tiles_v][8];
        wgmma::warpgroup_arrive();
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
            wgmma::wgmma_f8f8f32<head_dim, 0, CTA_K>(RO_tmp[fq], RS_f8[fq][0], &sV[0]);
#pragma unroll
            for (uint32_t v_it = 1; v_it < num_tiles_pv_inner; v_it++) {
                wgmma::wgmma_f8f8f32<head_dim, 1, CTA_K>(RO_tmp[fq], RS_f8[fq][v_it], &sV[v_it * 32]);
            }
        }

        wgmma::warpgroup_commit_batch();
        wgmma::warpgroup_wait<0>();

#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
#pragma unroll
                for (uint32_t e = 0; e < 8; e++) {
                    RO[fq][fv][e] += RO_tmp[fq][fv][e];
                }
            }
        }

        // load V
        if (threadIdx.x == 0) {
            expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_V);
#ifdef SAGE_VARLEN
            // Unlike K, V never crosses into the next sequence: the slab of
            // sequence b spans blk_offset(b + 1) - blk_offset(b) >=
            // ceil(kv_len / CTA_K) blocks (varlen.h, Property 1), and iter
            // never reaches that many.
            load_async_4D(sV, &tensorMapV, &barrier_V, (seq_info.blk_k_base + iter) * CTA_K, 0, kv_head_id, 0);
#else
            load_async_4D(sV, &tensorMapV, &barrier_V, iter * CTA_K, 0, kv_head_id, batch_id);
#endif
        }
    }

    {
        phase ^= 1;

        float dequant_scale = q_scale * K_scale_base_ptr[k_scale_off + (num_iterations - 1) * k_scale_advance_offset];
        sm_scale            = original_sm_scale;

        // wait for K
        wait(&barrier_K, phase);

        // compute QK^T
        wgmma::warpgroup_arrive();
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
            int8_t* sQ_local = sQ + fq * 64 * head_dim;
            wgmma::wgmma_s8s8s32<CTA_K, 0, head_dim>(RS[fq], sQ_local, sK);
#pragma unroll
            for (int k_it = 1; k_it < num_tiles_qk_inner; k_it++) {
                wgmma::wgmma_s8s8s32<CTA_K, 1, head_dim>(RS[fq], &sQ_local[k_it * 32], &sK[k_it * 32]);
            }
        }
        wgmma::warpgroup_commit_batch();
        wgmma::warpgroup_wait<0>();

        // convert RS to float
        float RS_f32[num_tiles_q][num_tiles_k][8];
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
                for (uint32_t e = 0; e < 8; e++) {
                    RS_f32[fq][fk][e] = __int2float_rz(RS[fq][fk][e]) * dequant_scale;
                }
            }
        }

        // masking
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
                for (uint32_t e = 0; e < 8; e++) {
#ifdef SAGE_VARLEN
                    // Signed throughout: the bottom-right shift can push the
                    // row bound below zero, and an unsigned compare would wrap
                    // it into "mask nothing" exactly where everything is masked.
                    const int32_t q_idx  = Q_idx_mask_base + static_cast<int32_t>(fq * 64 + 8 * ((e % 4) / 2));
                    const int32_t kv_idx = (num_iterations - 1) * static_cast<int32_t>(CTA_K)
                                           + static_cast<int32_t>(fk * 16 + 2 * (lane_id % 4) + 8 * (e / 4) + e % 2);

                    bool is_out_of_bounds;

                    if constexpr (mask_mode == MaskMode::kCausal) {
                        is_out_of_bounds = (kv_idx > q_idx) || (kv_idx >= static_cast<int32_t>(kv_len));
                    }
                    else {
                        is_out_of_bounds = (kv_idx >= static_cast<int32_t>(kv_len));
                    }
#else
                    const uint32_t q_idx = Q_idx_lane_base + fq * 64 + 8 * ((e % 4) / 2);
                    const uint32_t kv_idx =
                        (num_iterations - 1) * CTA_K + fk * 16 + 2 * (lane_id % 4) + 8 * (e / 4) + e % 2;

                    bool is_out_of_bounds;

                    if constexpr (mask_mode == MaskMode::kCausal) {
                        is_out_of_bounds = (kv_idx > q_idx) || (kv_idx >= kv_len);
                    }
                    else {
                        is_out_of_bounds = (kv_idx >= kv_len);
                    }
#endif

                    if (is_out_of_bounds) {
                        RS_f32[fq][fk][e] = -5000000.0f;
                    }
                }
            }
        }

        update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(RS_f32, RO, row_max, denom, sm_scale);

        // accumulate denom on thread basis
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
                denom[fq][0] += (RS_f32[fq][fk][0] + RS_f32[fq][fk][1] + RS_f32[fq][fk][4] + RS_f32[fq][fk][5]);
                denom[fq][1] += (RS_f32[fq][fk][2] + RS_f32[fq][fk][3] + RS_f32[fq][fk][6] + RS_f32[fq][fk][7]);
            }
        }

        uint32_t RS_f8[num_tiles_q][num_tiles_pv_inner][4];
        RS_f32_to_f8<num_tiles_q, num_tiles_k>(RS_f32, RS_f8);

        // wait for V
        wait(&barrier_V, phase);

        float RO_tmp[num_tiles_q][num_tiles_v][8];
        wgmma::warpgroup_arrive();
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
            wgmma::wgmma_f8f8f32<head_dim, 0, CTA_K>(RO_tmp[fq], RS_f8[fq][0], &sV[0]);
#pragma unroll
            for (uint32_t v_it = 1; v_it < num_tiles_pv_inner; v_it++) {
                wgmma::wgmma_f8f8f32<head_dim, 1, CTA_K>(RO_tmp[fq], RS_f8[fq][v_it], &sV[v_it * 32]);
            }
        }

        wgmma::warpgroup_commit_batch();
        wgmma::warpgroup_wait<0>();

#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
#pragma unroll
                for (uint32_t e = 0; e < 8; e++) {
                    RO[fq][fv][e] += RO_tmp[fq][fv][e];
                }
            }
        }
    }

#ifdef SAGE_VARLEN
    // A row that admitted no key at all - bottom-right causal with kv shorter
    // than qo - saw nothing but the -5000000 mask sentinel. The sentinel is
    // finite, so it *became* the row max and every masked entry got the weight
    // exp2(0) = 1: without this the row would come out as the plain average of
    // a KV tile. flash-attention's answer for such a row is a zero output and
    // an -inf lse, which is what is forced here. The row is identified by its
    // index rather than by comparing against the scaled sentinel:
    // Q_idx_mask_base already carries + delta, so "no admissible key" is
    // exactly a negative shifted row index. (row_max is not read by
    // normalize_d; setting it to -inf is what makes the lse below -inf.)
    //
    // e indexes RO as the store below reads it: {0, 1, 4, 5} are the row at
    // + 0, {2, 3, 6, 7} the row at + 8, which is the same pairing row_max and
    // denom use.
    if constexpr (mask_mode == MaskMode::kCausal) {
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t e = 0; e < 2; e++) {
                if (Q_idx_mask_base + static_cast<int32_t>(fq * 64 + 8 * e) < 0) {
#pragma unroll
                    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
#pragma unroll
                        for (uint32_t sub = 0; sub < 2; sub++) {
                            RO[fq][fv][e * 2 + sub]     = 0.0f;
                            RO[fq][fv][e * 2 + sub + 4] = 0.0f;
                        }
                    }
                    row_max[fq][e] = -INFINITY;
                }
            }
        }
    }
#endif

    normalize_d<num_tiles_q, num_tiles_v, ComputeUnit::kCudaCore>(RO, row_max, denom);

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

#ifdef SAGE_VARLEN
    // The sequence's first token replaces the batch stride; the packed tensor
    // has no batch dimension at all.
    DTypeOut* O_lane_ptr = O + static_cast<int64_t>(seq_info.offset_q) * stride_seq_o
                           + static_cast<int64_t>(head_id) * stride_h_o
                           + static_cast<int64_t>(cta_idx_q * CTA_Q + warp_idx * 16 + (lane_id / 4)) * stride_seq_o
                           + static_cast<int64_t>((lane_id % 4) * 2);
#else
    DTypeOut* O_lane_ptr = O + static_cast<int64_t>(batch_id) * stride_batch_o
                           + static_cast<int64_t>(head_id) * stride_h_o
                           + static_cast<int64_t>(cta_idx_q * CTA_Q + warp_idx * 16 + (lane_id / 4)) * stride_seq_o
                           + static_cast<int64_t>((lane_id % 4) * 2);
#endif
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t fv = 0; fv < head_dim / 16; fv++) {
            if (Q_idx_lane_base + fq * 64 < qo_len) {
                if constexpr (std::is_same<DTypeOut, half>::value) {
                    ((half2*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16))[0] =
                        __float22half2_rn(((float2*)(RO[fq][fv]))[0]);
                    ((half2*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8))[0] =
                        __float22half2_rn(((float2*)(RO[fq][fv]))[2]);
                }
                else {
                    ((nv_bfloat162*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16))[0] =
                        __float22bfloat162_rn(((float2*)(RO[fq][fv]))[0]);
                    ((nv_bfloat162*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8))[0] =
                        __float22bfloat162_rn(((float2*)(RO[fq][fv]))[2]);
                }
            }

            if (Q_idx_lane_base + fq * 64 + 8 < qo_len) {
                if constexpr (std::is_same<DTypeOut, half>::value) {
                    ((half2*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8 * stride_seq_o))[0] =
                        __float22half2_rn(((float2*)(RO[fq][fv]))[1]);
                    ((half2*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8 + 8 * stride_seq_o))[0] =
                        __float22half2_rn(((float2*)(RO[fq][fv]))[3]);
                }
                else {
                    ((nv_bfloat162*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8 * stride_seq_o))[0] =
                        __float22bfloat162_rn(((float2*)(RO[fq][fv]))[1]);
                    ((nv_bfloat162*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8 + 8 * stride_seq_o))[0] =
                        __float22bfloat162_rn(((float2*)(RO[fq][fv]))[3]);
                }
            }
        }

        if constexpr (return_lse) {
            // only works for CTA_Q = 64
            uint32_t lse_idx = cta_idx_q * CTA_Q + lane_id / 4 + 8 * (lane_id % 4) + 16 * warp_idx;
#ifdef SAGE_VARLEN
            // lse is [heads, total_tokens]: head-major, and the sequence's rows
            // sit at its token offset. The lane dance above is unchanged.
            float* lse_lane_ptr =
                Lse + static_cast<int64_t>(head_id) * lse_stride_h + seq_info.offset_q + lse_idx;
#else
            float*   lse_lane_ptr = Lse + static_cast<int64_t>(batch_id) * (static_cast<int64_t>(qo_len) * num_qo_heads)
                                  + static_cast<int64_t>(head_id) * qo_len + lse_idx;
#endif
            uint32_t fq = (lane_id % 4) / 2;
            uint32_t e  = (lane_id % 4) % 2;

            if (lse_idx < qo_len && (lane_id % 4) < 2) {
                lse_lane_ptr[0] = (math::ptx_log2(denom[fq][e]) + row_max[fq][e]);
            }
        }
    }
}

}  // namespace sm90
}  // namespace sage
