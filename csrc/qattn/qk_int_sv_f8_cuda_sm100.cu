/*
 * Copyright (c) 2025 by SageAttention team.
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

// SM100 / SM110 (Blackwell, arch-specific sm_100a/sm_110a) tcgen05 attention
// kernel: INT8 QK^T via tcgen05.mma.kind::i8 (SS, s32 accumulator in TMEM) +
// FP8 PV via tcgen05.mma.kind::f8f6f4 e4m3 (TS: P fed straight from TMEM;
// a compile-flagged SS twin stages P through smem as the on-device oracle,
// see PV_FROM_SMEM). O accumulates in true fp32, resident in TMEM for the
// whole sequence (no inst_buf splitting needed on Blackwell).
//
// MVP stage-1 shape (see design plan): CTA_Q = CTA_K = 128, 128 threads (one
// warpgroup, no warp specialization). Thread t owns S/O row t == TMEM lane t.
// TMA + mbarrier pipeline carried over from qk_int_sv_f8_cuda_sm90.cu;
// wgmma sync (commit_batch/wait_group) replaced by tcgen05.commit + mbarrier
// parity waits (bar_S_done / bar_O_done).
//
// TMEM column plan (128 lanes x 512 cols x 32b, one alloc of all 512 cols):
//   S [0,128)          128x128 s32 QK accumulator (1 col = 1 s32 per lane)
//   P [128,160)        128x128 e4m3, packed 4-per-word (K elems 4c..4c+3 in
//                      word col c, byte b = K elem 4c+b) - the kind::f8f6f4
//                      TS A-operand layout (PTX ISA 9.7.17.10.4.3 packing +
//                      Layout D 9.7.17.10.5.4: lane = M row)
//   O [160,160+HD)     128xHD f32 PV accumulator
//   [160+HD,512)       spare (future S0/S1 ping-pong stage)
//
// V layout contract: value must come from per_channel_fp8(..., permute=False)
// — transposed (head_dim x padded_kv) and padded, but with LINEAR kv order.
// The sm90-style within-16 seq permutation exists to match the register
// A-fragment k-order of mma.sync/wgmma; here P (the A operand) is packed into
// TMEM in linear k-order (TS layout), so V's k-order must be linear as well.
// Both the TS path and the PV_FROM_SMEM SS twin consume the same linear sV.
//
// Compile modes:
//   default                     - kernel + torch launchers (extension build)
//   -DSAGE_SM100_DEVICE_ONLY    - kernel only, torch-free (ptxas probe TUs)
//   -DSAGE_SM100_PV_FROM_SMEM   - launchers instantiate the PV-SS twin path

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <type_traits>

#include "../math.cuh"
#include "../numeric_conversion.cuh"
#include "../tcgen05.cuh"

#ifdef SAGE_SM100_DEVICE_ONLY

// Torch-free replicas of the attn_utils.cuh definitions the device code needs
// (values must match attn_utils.cuh exactly; keep in sync).
enum class MaskMode {
    kNone   = 0,
    kCausal = 1,
};

enum class QuantGranularity {
    kPerTensor = 0,
    kPerBlock  = 1,
    kPerWarp   = 2,
    kPerThread = 3,
};

#ifndef S_FP8_OFFSET
#define S_FP8_OFFSET 8.807f
#endif

#ifndef div_ceil
#define div_ceil(M, N) (((M) + (N)-1) / (N))
#endif

#else  // !SAGE_SM100_DEVICE_ONLY

#include <torch/types.h>

#include "../dispatch_utils.h"
#include "../utils.cuh"
#include "attn_utils.cuh"  // MaskMode, QuantGranularity, S_FP8_OFFSET, div_ceil
#include "launch_utils.cuh"

#endif  // SAGE_SM100_DEVICE_ONLY

// All sm100 kernels and launchers live in sage::sm100 (single-so ODR rule:
// the fuse_v_scale launcher shares its public name with the sm89 family).
namespace sage {
namespace sm100 {

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
//
// Divergence audit invariants (design plan 5.6, grep-verifiable):
//   * every tcgen05::mma_* and tcgen05::tcgen05_commit call sits inside
//     `if (warp_idx == 0 && tcgen05::elect_one())` (warp 0 fully active,
//     one elected issuing thread);
//   * every tcgen05::tmem_ld/st/wait/fence call is in uniform control flow:
//     unconditional, or under `if (iter > 0)` / peeled-iteration selection,
//     both uniform across the CTA (iter is a grid-uniform loop counter);
//   * every TMEM producer->consumer handoff across threads is bracketed by
//     tcgen05_fence_before_sync(); __syncthreads(); tcgen05_fence_after_sync().

template<uint32_t         CTA_Q,
         uint32_t         CTA_K,
         uint32_t         NUM_THREADS,
         uint32_t         head_dim,
         QuantGranularity Q_GRAN,
         QuantGranularity K_GRAN,
         typename DTypeOut,
         MaskMode mask_mode    = MaskMode::kNone,
         bool     return_lse   = false,
         bool     fuse_v_scale = false,
         bool     PV_FROM_SMEM = false>
__global__ void __launch_bounds__(NUM_THREADS)
    qk_int8_sv_f8_attn_kernel_sm100(const __grid_constant__ CUtensorMap tensorMapQ,
                                    const __grid_constant__ CUtensorMap tensorMapK,
                                    const __grid_constant__ CUtensorMap tensorMapV,
                                    const float* __restrict__ Q_scale,
                                    const float* __restrict__ K_scale,
                                    const float* __restrict__ V_scale,
                                    DTypeOut* O,
                                    float* __restrict__ Lse,
                                    const int64_t  stride_bz_o,
                                    const int64_t  stride_h_o,
                                    uint32_t       stride_seq_o,
                                    const uint32_t qo_len,
                                    const uint32_t kv_len,
                                    const uint32_t num_kv_groups,
                                    float          sm_scale)
{
    // --- static shape checks (design plan section 5) ---
    static_assert(NUM_THREADS == 128, "MVP: one warpgroup; thread id == S/O row == TMEM lane");
    static_assert(CTA_Q == 128, "M=128 is the native cta_group::1 MMA shape");
    static_assert(CTA_K == 128, "N=128 tile; V^T rows are CTA_K bytes (128B swizzle)");
    static_assert(head_dim == 64 || head_dim == 128, "dispatched head dims");
    static_assert(head_dim % 32 == 0, "QK MMA is chained in K=32 steps");

    // --- TMEM column plan; regions disjoint by construction ---
    constexpr uint32_t TMEM_COLS_TOTAL = 512;
    constexpr uint32_t TMEM_COL_S      = 0;
    constexpr uint32_t TMEM_COLS_S     = CTA_K;  // s32: one column per element
    constexpr uint32_t TMEM_COL_P      = TMEM_COL_S + TMEM_COLS_S;
    constexpr uint32_t TMEM_COLS_P     = CTA_K / 4;  // e4m3 packed 4 per 32b word
    constexpr uint32_t TMEM_COL_O      = TMEM_COL_P + TMEM_COLS_P;
    constexpr uint32_t TMEM_COLS_O     = head_dim;  // f32: one column per element
    static_assert(TMEM_COL_O + TMEM_COLS_O <= TMEM_COLS_TOTAL, "TMEM budget exceeded");
    static_assert(TMEM_COL_P >= TMEM_COL_S + TMEM_COLS_S && TMEM_COL_O >= TMEM_COL_P + TMEM_COLS_P,
                  "TMEM regions must not overlap");

    // --- derived tile counts ---
    constexpr uint32_t num_k_steps_qk = head_dim / 32;  // K=32 elems per kind::i8 step
    constexpr uint32_t num_k_steps_pv = CTA_K / 32;     // K=32 elems per kind::f8f6f4 step
    constexpr uint32_t num_chunks_s   = CTA_K / 32;     // 32x32b.x32 loads per S row
    constexpr uint32_t num_chunks_o   = head_dim / 32;  // 32x32b.x32 loads per O row

    // --- smem plan (mirrors the launcher's sMemSize computation) ---
    constexpr uint32_t SMEM_Q_BYTES = CTA_Q * head_dim;
    constexpr uint32_t SMEM_K_BYTES = CTA_K * head_dim;
    constexpr uint32_t SMEM_V_BYTES = CTA_K * head_dim;  // V^T: head_dim rows x CTA_K cols
    constexpr uint32_t SMEM_P_BYTES = PV_FROM_SMEM ? CTA_Q * CTA_K : 0;
    static_assert(SMEM_Q_BYTES + SMEM_K_BYTES + SMEM_V_BYTES + SMEM_P_BYTES <= 227 * 1024,
                  "smem budget exceeded (sm100 CTA limit)");

    // --- smem descriptor parameters (K-major canonical layouts; parity-tested
    //     against cute::UMMA::make_umma_desc in tests/test_desc_parity.cpp) ---
    constexpr tcgen05::SmemSwizzleMode QK_SWIZZLE =
        (head_dim == 128) ? tcgen05::SmemSwizzleMode::kSwizzle128B : tcgen05::SmemSwizzleMode::kSwizzle64B;
    constexpr uint32_t                 QK_SBO    = 8 * head_dim;  // 8-row swizzle-atom pitch (bytes)
    constexpr tcgen05::SmemSwizzleMode V_SWIZZLE = tcgen05::SmemSwizzleMode::kSwizzle128B;
    constexpr uint32_t                 V_SBO     = 8 * CTA_K;  // V^T rows are CTA_K = 128 bytes

    // --- MMA instruction descriptors ---
    constexpr uint32_t idesc_qk = tcgen05::make_instr_desc<CTA_Q,
                                                           CTA_K,
                                                           tcgen05::kMmaFmtS8,
                                                           tcgen05::kMmaFmtS8,
                                                           tcgen05::kMmaCFmtS32,
                                                           /*AMajorK=*/true,
                                                           /*BMajorK=*/true>();
    constexpr uint32_t idesc_pv = tcgen05::make_instr_desc<CTA_Q,
                                                           head_dim,
                                                           tcgen05::kMmaFmtE4M3,
                                                           tcgen05::kMmaFmtE4M3,
                                                           tcgen05::kMmaCFmtF32,
                                                           /*AMajorK=*/true,
                                                           /*BMajorK=*/true>();

    const uint32_t warp_idx = threadIdx.x / 32;
    const uint32_t row      = threadIdx.x;  // S/O row owned by this thread == TMEM lane

    const uint32_t batch_id     = blockIdx.z;
    const uint32_t bx           = blockIdx.x;
    const uint32_t head_id      = blockIdx.y;
    const uint32_t num_qo_heads = gridDim.y;
    const uint32_t kv_head_id   = head_id / num_kv_groups;

    sm_scale *= math::log2e;  // softmax runs in base 2 (attn_utils.cuh convention)

    extern __shared__ __align__(1024) int8_t smem_[];
    int8_t*                                  sQ = smem_;
    int8_t*                                  sK = smem_ + SMEM_Q_BYTES;
    int8_t*                                  sV = smem_ + SMEM_Q_BYTES + SMEM_K_BYTES;
    int8_t* sP = smem_ + SMEM_Q_BYTES + SMEM_K_BYTES + SMEM_V_BYTES;  // PV_FROM_SMEM only

    // --- quant scale indexing (row -> scale by pure index math; audit 1.6) ---
    // Q scale is read once: the whole index may live in 64-bit. K scale is read
    // once per KV tile, so it is split into a 64-bit base pointer plus the 32-bit
    // running offset the tile loop advances (no lane term here, unlike sm90).
    int64_t      q_scale_idx;
    const float* K_scale_base;

    if constexpr (Q_GRAN == QuantGranularity::kPerBlock) {
        const uint32_t num_block_q = gridDim.x;
        q_scale_idx                = static_cast<int64_t>(batch_id) * num_qo_heads * num_block_q
                      + static_cast<int64_t>(head_id) * num_block_q + bx;
    }
    else if constexpr (Q_GRAN == QuantGranularity::kPerWarp) {
        // per_warp_int8_cuda(BLKQ=128, WARPQ=32): 4 scales per block, one per 32 rows
        const uint32_t num_warp_block_q = gridDim.x * (CTA_Q / 32);
        q_scale_idx                     = static_cast<int64_t>(batch_id) * num_qo_heads * num_warp_block_q
                      + static_cast<int64_t>(head_id) * num_warp_block_q + bx * (CTA_Q / 32) + row / 32;
    }
    else if constexpr (Q_GRAN == QuantGranularity::kPerThread) {
        // per_thread quant (WARPQ=32): 8 scales per 32-row group, class = row % 8
        const uint32_t num_warp_block_q = gridDim.x * (CTA_Q / 32);
        q_scale_idx                     = static_cast<int64_t>(batch_id) * num_qo_heads * (num_warp_block_q * 8)
                      + static_cast<int64_t>(head_id) * (num_warp_block_q * 8) + bx * ((CTA_Q / 32) * 8)
                      + (row / 32) * 8 + (row % 8);
    }

    if constexpr (K_GRAN == QuantGranularity::kPerBlock || K_GRAN == QuantGranularity::kPerWarp) {
        const uint32_t num_block_k = div_ceil(kv_len, CTA_K);
        K_scale_base = K_scale + static_cast<int64_t>(batch_id) * (num_qo_heads / num_kv_groups) * num_block_k
                       + static_cast<int64_t>(head_id / num_kv_groups) * num_block_k;
    }
    else if constexpr (K_GRAN == QuantGranularity::kPerThread) {
        // per_thread K quant: 4 scales per KV tile; S column j uses class (j%8)/2
        const uint32_t num_block_k = div_ceil(kv_len, CTA_K);
        K_scale_base = K_scale + static_cast<int64_t>(batch_id) * (num_qo_heads / num_kv_groups) * (num_block_k * 4)
                       + static_cast<int64_t>(head_id / num_kv_groups) * (num_block_k * 4);
    }

    constexpr uint32_t k_scale_advance_offset =
        (K_GRAN == QuantGranularity::kPerBlock || K_GRAN == QuantGranularity::kPerWarp) ? 1 : 4;

    // --- flash-attention state (thread-local; this thread's row) ---
    float m = -5000000.0f;
    float d = 1.0f;

    // --- barriers: TMA trio (Q/K/V) + tcgen05.commit trackers (S/O) ---
    __shared__ __align__(8) uint64_t barrier_Q;
    __shared__ __align__(8) uint64_t barrier_K;
    __shared__ __align__(8) uint64_t barrier_V;
    __shared__ __align__(8) uint64_t barrier_S_done;
    __shared__ __align__(8) uint64_t barrier_O_done;
    __shared__ __align__(4) uint32_t tmem_addr_slot;

    if (threadIdx.x == 0) {
        tcgen05::init_barrier(&barrier_Q, 1);
        tcgen05::init_barrier(&barrier_K, 1);
        tcgen05::init_barrier(&barrier_V, 1);
        tcgen05::init_barrier(&barrier_S_done, 1);
        tcgen05::init_barrier(&barrier_O_done, 1);
    }

    __syncthreads();

    // --- issue Q (once) + first K/V tiles ---
    if (threadIdx.x == 0) {
        tcgen05::expect_bytes<(CTA_Q * head_dim) * sizeof(int8_t)>(&barrier_Q);
        tcgen05::expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_K);
        tcgen05::expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_V);
        tcgen05::load_async_4D(sQ, &tensorMapQ, &barrier_Q, 0, bx * CTA_Q, head_id, batch_id);
        tcgen05::load_async_4D(sK, &tensorMapK, &barrier_K, 0, 0, kv_head_id, batch_id);
        tcgen05::load_async_4D(sV, &tensorMapV, &barrier_V, 0, 0, kv_head_id, batch_id);
    }

    const float q_scale = Q_scale[q_scale_idx];

    // --- TMEM allocation: warp-collective alloc of the whole 512 cols, then
    //     immediately relinquish the alloc permit (CUTLASS ordering, risk R7) ---
    if (warp_idx == 0) {
        tcgen05::tmem_alloc(&tmem_addr_slot, TMEM_COLS_TOTAL);
        tcgen05::tmem_relinquish();
    }
    __syncthreads();
    const uint32_t tmem_base = tmem_addr_slot;
    // MMA D/A operands address lane 0; ld/st address this warp's lane quadrant
    // (warp w may only touch lanes [32w, 32w+32), PTX ISA 9.7.17.8.1).
    const uint32_t tmem_S_mma    = tmem_base + TMEM_COL_S;
    const uint32_t tmem_P_mma    = tmem_base + TMEM_COL_P;
    const uint32_t tmem_O_mma    = tmem_base + TMEM_COL_O;
    const uint32_t tmem_row_base = tmem_base + ((warp_idx * 32) << 16);

    // wait for Q (phase 0: single Q load, barrier never reused)
    tcgen05::wait(&barrier_Q, 0);

    const uint32_t num_iterations =
        div_ceil(mask_mode == MaskMode::kCausal ? min(kv_len, (bx + 1) * CTA_Q) : kv_len, CTA_K);

    const uint32_t q_idx = bx * CTA_Q + row;

    // mbarrier phase bits (all barriers complete exactly once per KV tile;
    // see docs/barrier_ledger.md)
    int k_phase = 0, v_phase = 0, s_phase = 0, o_phase = 0;

    // -------------------------------------------------------------------------
    // Per-KV-tile body. is_last_t selects the peeled last iteration (fused
    // causal/OOB mask, no next-tile TMA issues). All control flow inside is
    // uniform across the CTA.
    // -------------------------------------------------------------------------
    auto process_tile = [&](auto is_last_t, uint32_t iter) {
        constexpr bool is_last = decltype(is_last_t)::value;

        // ---- QK^T: S = Q K^T (kind::i8 SS, chained K=32 steps) ----
        tcgen05::wait(&barrier_K, k_phase);
        k_phase ^= 1;

        if (warp_idx == 0 && tcgen05::elect_one()) {
#pragma unroll
            for (uint32_t k_it = 0; k_it < num_k_steps_qk; k_it++) {
                const uint64_t desc_q =
                    tcgen05::make_smem_desc_sm100<QK_SWIZZLE, tcgen05::kKMajorLBO, QK_SBO>(&sQ[k_it * 32]);
                const uint64_t desc_k =
                    tcgen05::make_smem_desc_sm100<QK_SWIZZLE, tcgen05::kKMajorLBO, QK_SBO>(&sK[k_it * 32]);
                // enable_D = 0 on the first K step: zero-init S every KV tile
                tcgen05::mma_i8_ss(tmem_S_mma, desc_q, desc_k, idesc_qk, k_it > 0);
            }
            tcgen05::tcgen05_commit(&barrier_S_done);
        }

        tcgen05::wait(&barrier_S_done, s_phase);
        s_phase ^= 1;

        // QK retired -> K smem free: prefetch next K tile
        if constexpr (!is_last) {
            if (threadIdx.x == 0) {
                tcgen05::expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_K);
                tcgen05::load_async_4D(sK, &tensorMapK, &barrier_K, 0, (iter + 1) * CTA_K, kv_head_id, batch_id);
            }
        }

        // ---- softmax: pull this thread's full S row out of TMEM ----
        // K-scale(s) for this tile (per-thread K: 4 scales, column class (j%8)/2)
        float dq[(K_GRAN == QuantGranularity::kPerThread) ? 4 : 1];
        if constexpr (K_GRAN == QuantGranularity::kPerThread) {
#pragma unroll
            for (uint32_t cls = 0; cls < 4; cls++) {
                dq[cls] = q_scale * K_scale_base[iter * k_scale_advance_offset + cls];
            }
        }
        else {
            dq[0] = q_scale * K_scale_base[iter * k_scale_advance_offset];
        }

        float s_f32[CTA_K];
        float m_local = -5000000.0f;
#pragma unroll
        for (uint32_t c = 0; c < num_chunks_s; c++) {
            uint32_t s_u32[32];
            tcgen05::tmem_ld_32x32b_x32(s_u32, tmem_row_base + TMEM_COL_S + c * 32);
            tcgen05::tmem_ld_wait();
#pragma unroll
            for (uint32_t jj = 0; jj < 32; jj++) {
                const uint32_t j = c * 32 + jj;
                float          dq_j;
                if constexpr (K_GRAN == QuantGranularity::kPerThread) {
                    dq_j = dq[(j % 8) / 2];
                }
                else {
                    dq_j = dq[0];
                }
                float s = __int2float_rz(static_cast<int32_t>(s_u32[jj])) * dq_j;

                if constexpr (is_last) {
                    // fused causal/OOB mask: column j <-> kv position iter*CTA_K + j
                    const uint32_t k_idx = iter * CTA_K + j;
                    bool           is_out_of_bounds;
                    if constexpr (mask_mode == MaskMode::kCausal) {
                        is_out_of_bounds = (k_idx > q_idx) || (k_idx >= kv_len);
                    }
                    else {
                        is_out_of_bounds = (k_idx >= kv_len);
                    }
                    if (is_out_of_bounds) {
                        s = -5000000.0f;
                    }
                }

                m_local  = max(m_local, s);
                s_f32[j] = s;
            }
        }

        // ---- online softmax update (thread-local; no shuffles) ----
        // m folds sm_scale (already x log2e) and the e4m3-saturation offset:
        // p[j] = exp2(s[j]*sm_scale - m) = exp2(s' - max' + S_FP8_OFFSET) in (0, 448]
        const float m_prev = m;
        m                  = max(m, fmaf(m_local, sm_scale, -S_FP8_OFFSET));
        const float alpha  = math::ptx_exp2(m_prev - m);  // o_scale
        d *= alpha;
        const float neg_m = -m;
        float       d_sum = 0.0f;
#pragma unroll
        for (uint32_t j = 0; j < CTA_K; j++) {
            const float p = math::ptx_exp2(fmaf(s_f32[j], sm_scale, neg_m));
            s_f32[j]      = p;
            d_sum += p;
        }
        d += d_sum;

        // ---- P re-quant: pack e4m3 in LINEAR row order (word c = K elems
        //      4c..4c+3, byte b = elem 4c+b) - the TS A-operand TMEM layout ----
        uint32_t p_u32[CTA_K / 4];
#pragma unroll
        for (uint32_t w = 0; w < CTA_K / 4; w++) {
            floatx4_to_e4m3x4(&p_u32[w], &s_f32[4 * w], &s_f32[4 * w + 2]);
        }

        // ---- O correction: O *= alpha (skip on first tile: O uninitialized;
        //      `iter > 0` is uniform across the CTA) ----
        if (iter > 0) {
#pragma unroll
            for (uint32_t c = 0; c < num_chunks_o; c++) {
                uint32_t o_u32[32];
                tcgen05::tmem_ld_32x32b_x32(o_u32, tmem_row_base + TMEM_COL_O + c * 32);
                tcgen05::tmem_ld_wait();
#pragma unroll
                for (uint32_t jj = 0; jj < 32; jj++) {
                    o_u32[jj] = __float_as_uint(__uint_as_float(o_u32[jj]) * alpha);
                }
                tcgen05::tmem_st_32x32b_x32(tmem_row_base + TMEM_COL_O + c * 32, o_u32);
            }
            tcgen05::tmem_st_wait();
        }

        // ---- feed P to the PV MMA ----
        if constexpr (!PV_FROM_SMEM) {
            // TS path: store P into TMEM (32 cols, one warp-collective st)
            tcgen05::tmem_st_32x32b_x32(tmem_row_base + TMEM_COL_P, p_u32);
            tcgen05::tmem_st_wait();
        }
        else {
            // SS twin (on-device oracle for the TS TMEM layout, risk R1): stage P
            // through smem in the same 128B-swizzled K-major layout TMA produces
            // for sK/sV: word w of row r lands at word ((w/4 ^ r%8)*4 + w%4).
            uint32_t* sP_row = reinterpret_cast<uint32_t*>(sP + row * CTA_K);
#pragma unroll
            for (uint32_t w = 0; w < CTA_K / 4; w++) {
                sP_row[(((w >> 2) ^ (row & 7)) << 2) | (w & 3)] = p_u32[w];
            }
        }

        // ---- TMEM/smem producer (128 threads) -> MMA issuer handoff ----
        tcgen05::tcgen05_fence_before_sync();
        __syncthreads();
        tcgen05::tcgen05_fence_after_sync();

        // ---- PV: O += P V (kind::f8f6f4 e4m3, chained K=32 steps) ----
        tcgen05::wait(&barrier_V, v_phase);
        v_phase ^= 1;

        if (warp_idx == 0 && tcgen05::elect_one()) {
#pragma unroll
            for (uint32_t v_it = 0; v_it < num_k_steps_pv; v_it++) {
                const uint64_t desc_v =
                    tcgen05::make_smem_desc_sm100<V_SWIZZLE, tcgen05::kKMajorLBO, V_SBO>(&sV[v_it * 32]);
                // enable_D = 0 only on (first KV tile, first K step): O zero-init
                // once globally, then accumulates across the whole sequence
                const bool enable_d = (iter > 0) || (v_it > 0);
                if constexpr (!PV_FROM_SMEM) {
                    // A from TMEM: K step advances 32 e4m3 elems = 8 columns
                    tcgen05::mma_f8f8f32_ts(tmem_O_mma, tmem_P_mma + v_it * 8, desc_v, idesc_pv, enable_d);
                }
                else {
                    const uint64_t desc_p =
                        tcgen05::make_smem_desc_sm100<V_SWIZZLE, tcgen05::kKMajorLBO, V_SBO>(&sP[v_it * 32]);
                    tcgen05::mma_f8f8f32_ss(tmem_O_mma, desc_p, desc_v, idesc_pv, enable_d);
                }
            }
            tcgen05::tcgen05_commit(&barrier_O_done);
        }

        // PV retired -> V smem (and P TMEM/smem) free, O valid in TMEM
        tcgen05::wait(&barrier_O_done, o_phase);
        o_phase ^= 1;

        if constexpr (!is_last) {
            if (threadIdx.x == 0) {
                tcgen05::expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_V);
                tcgen05::load_async_4D(sV, &tensorMapV, &barrier_V, (iter + 1) * CTA_K, 0, kv_head_id, batch_id);
            }
        }
    };

    for (uint32_t iter = 0; iter + 1 < num_iterations; iter++) {
        process_tile(std::false_type{}, iter);
    }
    process_tile(std::true_type{}, num_iterations - 1);  // peeled: fused mask, no prefetch

    // -------------------------------------------------------------------------
    // Epilogue: O = O * v_scale / d, cvt to fp16/bf16, direct global stores
    // (thread = row; MVP keeps the sm90-style register->global epilogue).
    // -------------------------------------------------------------------------
    const float d_rcp     = math::ptx_rcp(d);
    const bool  row_valid = q_idx < qo_len;
    DTypeOut* O_row_ptr = O + static_cast<int64_t>(batch_id) * stride_bz_o + static_cast<int64_t>(head_id) * stride_h_o
                          + static_cast<int64_t>(q_idx) * stride_seq_o;
    const float* V_scale_ptr = fuse_v_scale ?
                                   V_scale + static_cast<int64_t>(batch_id) * (num_qo_heads / num_kv_groups) * head_dim
                                       + static_cast<int64_t>(kv_head_id) * head_dim :
                                   nullptr;

#pragma unroll
    for (uint32_t c = 0; c < num_chunks_o; c++) {
        uint32_t o_u32[32];
        tcgen05::tmem_ld_32x32b_x32(o_u32, tmem_row_base + TMEM_COL_O + c * 32);
        tcgen05::tmem_ld_wait();

        float o_f32[32];
#pragma unroll
        for (uint32_t jj = 0; jj < 32; jj++) {
            o_f32[jj] = __uint_as_float(o_u32[jj]) * d_rcp;
            if constexpr (fuse_v_scale) {
                o_f32[jj] *= __ldg(V_scale_ptr + c * 32 + jj);  // per-channel v_scale
            }
        }

        if (row_valid) {
#pragma unroll
            for (uint32_t jj = 0; jj < 16; jj++) {
                const float2 o2 = make_float2(o_f32[2 * jj], o_f32[2 * jj + 1]);
                if constexpr (std::is_same<DTypeOut, half>::value) {
                    ((half2*)(O_row_ptr + c * 32 + 2 * jj))[0] = __float22half2_rn(o2);
                }
                else {
                    ((nv_bfloat162*)(O_row_ptr + c * 32 + 2 * jj))[0] = __float22bfloat162_rn(o2);
                }
            }
        }
    }

    if constexpr (return_lse) {
        // one thread-local store per row (correct for any CTA_Q, unlike the
        // sm90 CTA_Q=64-only lane dance)
        if (row_valid) {
            Lse[static_cast<int64_t>(batch_id) * (static_cast<int64_t>(qo_len) * num_qo_heads)
                + static_cast<int64_t>(head_id) * qo_len + q_idx] = math::ptx_log2(d) + m;
        }
    }

    // --- TMEM dealloc (all TMEM reads above are complete per-thread via
    //     tcgen05.wait::ld; order them before warp 0's dealloc) ---
    tcgen05::tcgen05_fence_before_sync();
    __syncthreads();
    tcgen05::tcgen05_fence_after_sync();
    if (warp_idx == 0) {
        tcgen05::tmem_dealloc(tmem_base, TMEM_COLS_TOTAL);
    }
}

#ifndef SAGE_SM100_DEVICE_ONLY

// ---------------------------------------------------------------------------
// Host launchers (clone of the sm90 launcher bodies; CTA_Q 64 -> 128; the
// kernels accumulate PV in true fp32 so there is no _inst_buf variant).
// ---------------------------------------------------------------------------

#ifdef SAGE_SM100_PV_FROM_SMEM
constexpr bool kPVFromSmem = true;  // SS cross-check build (hardware-day oracle)
#else
constexpr bool kPVFromSmem = false;  // default: TS (P fed from TMEM)
#endif

torch::Tensor qk_int8_sv_f8_accum_f32_attn(torch::Tensor query,
                                           torch::Tensor key,
                                           torch::Tensor value,
                                           torch::Tensor output,
                                           torch::Tensor query_scale,
                                           torch::Tensor key_scale,
                                           int           tensor_layout,
                                           int           is_causal,
                                           int           qk_quant_gran,
                                           float         sm_scale,
                                           int           return_lse)
{
    const c10::cuda::CUDAGuard device_guard(query.device());
    cudaStream_t               stream = at::cuda::getCurrentCUDAStream();

    QKVLayout qkv = qkv_layout_parse<QKVFamily::kSVF8TMA>(query,
                                                          key,
                                                          value,
                                                          output,
                                                          query_scale,
                                                          key_scale,
                                                          /*value_scale_opt=*/nullptr,
                                                          /*value_mean_opt=*/nullptr,
                                                          tensor_layout,
                                                          return_lse);
    SAGEATTN_QKV_LAYOUT_LOCALS_FP8(qkv);

    auto output_type = output.scalar_type();

    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
            DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
                DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
                    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_type, DTypeOut, {
                        constexpr int CTA_Q       = 128;
                        constexpr int CTA_K       = 128;
                        constexpr int NUM_THREADS = 128;

                        constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

                        TORCH_CHECK(value.size(3) >= div_ceil(kv_len, CTA_K) * CTA_K,
                                    "value.size(3) >= div_ceil(kv_len, CTA_K) * CTA_K failed: value.size(3)=",
                                    value.size(3),
                                    ", kv_len=",
                                    kv_len,
                                    ", CTA_K=",
                                    CTA_K);

                        if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp)) {
                            CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / 32));
                            CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K));
                        }
                        else if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread)) {
                            CHECK_SHAPE(
                                query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / 32) * 8);
                            CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * 4);
                        }
                        else {
                            static_assert(QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp)
                                              || QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread),
                                          "Unsupported quantization granularity");
                        }

                        CUtensorMap tma_map_Q =
                            create_tensor_map_4D<CTA_Q, HEAD_DIM>(reinterpret_cast<int8_t*>(query.data_ptr()),
                                                                  batch_size,
                                                                  num_qo_heads,
                                                                  qo_len,
                                                                  HEAD_DIM,
                                                                  stride_bz_q,
                                                                  stride_h_q,
                                                                  stride_seq_q);
                        CUtensorMap tma_map_K =
                            create_tensor_map_4D<CTA_K, HEAD_DIM>(reinterpret_cast<int8_t*>(key.data_ptr()),
                                                                  batch_size,
                                                                  num_kv_heads,
                                                                  kv_len,
                                                                  HEAD_DIM,
                                                                  stride_bz_k,
                                                                  stride_h_k,
                                                                  stride_seq_k);
                        CUtensorMap tma_map_V =
                            create_tensor_map_4D<HEAD_DIM, CTA_K>(reinterpret_cast<int8_t*>(value.data_ptr()),
                                                                  batch_size,
                                                                  num_kv_heads,
                                                                  HEAD_DIM,
                                                                  value.size(3),
                                                                  stride_bz_v,
                                                                  stride_h_v,
                                                                  stride_d_v);

                        auto*  kernel   = qk_int8_sv_f8_attn_kernel_sm100<CTA_Q,
                                                                       CTA_K,
                                                                       NUM_THREADS,
                                                                       HEAD_DIM,
                                                                       static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                                       static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                                       DTypeOut,
                                                                       mask_mode,
                                                                       RETURN_LSE,
                                                                       false,
                                                                       kPVFromSmem>;
                        size_t sMemSize = CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t)
                                          + CTA_K * HEAD_DIM * sizeof(int8_t)
                                          + (kPVFromSmem ? CTA_Q * CTA_K * sizeof(int8_t) : 0);
                        sage::set_max_dynamic_smem_once(kernel, sMemSize, query.get_device());

                        dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
                        kernel<<<grid, NUM_THREADS, sMemSize, stream>>>(
                            tma_map_Q,
                            tma_map_K,
                            tma_map_V,
                            reinterpret_cast<float*>(query_scale.data_ptr()),
                            reinterpret_cast<float*>(key_scale.data_ptr()),
                            nullptr,
                            reinterpret_cast<DTypeOut*>(output.data_ptr()),
                            (RETURN_LSE) ? reinterpret_cast<float*>(lse.data_ptr()) : nullptr,
                            stride_bz_o,
                            stride_h_o,
                            stride_seq_o,
                            qo_len,
                            kv_len,
                            num_kv_groups,
                            sm_scale);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    });
                });
            });
        });
    });

    return lse;
}

torch::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(torch::Tensor query,
                                                        torch::Tensor key,
                                                        torch::Tensor value,
                                                        torch::Tensor output,
                                                        torch::Tensor query_scale,
                                                        torch::Tensor key_scale,
                                                        torch::Tensor value_scale,
                                                        int           tensor_layout,
                                                        int           is_causal,
                                                        int           qk_quant_gran,
                                                        float         sm_scale,
                                                        int           return_lse)
{
    const c10::cuda::CUDAGuard device_guard(query.device());
    cudaStream_t               stream = at::cuda::getCurrentCUDAStream();

    QKVLayout qkv = qkv_layout_parse<QKVFamily::kSVF8TMA>(query,
                                                          key,
                                                          value,
                                                          output,
                                                          query_scale,
                                                          key_scale,
                                                          &value_scale,
                                                          /*value_mean_opt=*/nullptr,
                                                          tensor_layout,
                                                          return_lse);
    SAGEATTN_QKV_LAYOUT_LOCALS_FP8(qkv);

    auto output_dtype = output.scalar_type();

    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
            DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
                DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
                    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
                        constexpr int CTA_Q       = 128;
                        constexpr int CTA_K       = 128;
                        constexpr int NUM_THREADS = 128;

                        constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

                        TORCH_CHECK(value.size(3) >= div_ceil(kv_len, CTA_K) * CTA_K,
                                    "value.size(3) >= div_ceil(kv_len, CTA_K) * CTA_K failed: value.size(3)=",
                                    value.size(3),
                                    ", kv_len=",
                                    kv_len,
                                    ", CTA_K=",
                                    CTA_K);

                        if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp)) {
                            CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / 32));
                            CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K));
                        }
                        else if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread)) {
                            CHECK_SHAPE(
                                query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / 32) * 8);
                            CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * 4);
                        }
                        else {
                            static_assert(QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp)
                                              || QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread),
                                          "Unsupported quantization granularity");
                        }

                        CHECK_SHAPE(value_scale, batch_size, num_kv_heads, head_dim);

                        CUtensorMap tma_map_Q =
                            create_tensor_map_4D<CTA_Q, HEAD_DIM>(reinterpret_cast<int8_t*>(query.data_ptr()),
                                                                  batch_size,
                                                                  num_qo_heads,
                                                                  qo_len,
                                                                  HEAD_DIM,
                                                                  stride_bz_q,
                                                                  stride_h_q,
                                                                  stride_seq_q);
                        CUtensorMap tma_map_K =
                            create_tensor_map_4D<CTA_K, HEAD_DIM>(reinterpret_cast<int8_t*>(key.data_ptr()),
                                                                  batch_size,
                                                                  num_kv_heads,
                                                                  kv_len,
                                                                  HEAD_DIM,
                                                                  stride_bz_k,
                                                                  stride_h_k,
                                                                  stride_seq_k);
                        CUtensorMap tma_map_V =
                            create_tensor_map_4D<HEAD_DIM, CTA_K>(reinterpret_cast<int8_t*>(value.data_ptr()),
                                                                  batch_size,
                                                                  num_kv_heads,
                                                                  HEAD_DIM,
                                                                  value.size(3),
                                                                  stride_bz_v,
                                                                  stride_h_v,
                                                                  stride_d_v);

                        auto*  kernel   = qk_int8_sv_f8_attn_kernel_sm100<CTA_Q,
                                                                       CTA_K,
                                                                       NUM_THREADS,
                                                                       HEAD_DIM,
                                                                       static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                                       static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                                       DTypeOut,
                                                                       mask_mode,
                                                                       RETURN_LSE,
                                                                       true,
                                                                       kPVFromSmem>;
                        size_t sMemSize = CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t)
                                          + CTA_K * HEAD_DIM * sizeof(int8_t)
                                          + (kPVFromSmem ? CTA_Q * CTA_K * sizeof(int8_t) : 0);
                        sage::set_max_dynamic_smem_once(kernel, sMemSize, query.get_device());

                        dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
                        kernel<<<grid, NUM_THREADS, sMemSize, stream>>>(
                            tma_map_Q,
                            tma_map_K,
                            tma_map_V,
                            reinterpret_cast<float*>(query_scale.data_ptr()),
                            reinterpret_cast<float*>(key_scale.data_ptr()),
                            reinterpret_cast<float*>(value_scale.data_ptr()),
                            reinterpret_cast<DTypeOut*>(output.data_ptr()),
                            (RETURN_LSE) ? reinterpret_cast<float*>(lse.data_ptr()) : nullptr,
                            stride_bz_o,
                            stride_h_o,
                            stride_seq_o,
                            qo_len,
                            kv_len,
                            num_kv_groups,
                            sm_scale);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    });
                });
            });
        });
    });

    return lse;
}

#endif  // !SAGE_SM100_DEVICE_ONLY

}  // namespace sm100
}  // namespace sage
