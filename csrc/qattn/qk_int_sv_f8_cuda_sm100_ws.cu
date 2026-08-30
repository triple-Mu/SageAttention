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

// SM100/SM110 warp-specialized (C1) tcgen05 attention kernel: 512 threads /
// 16 warps / 1 CTA per SM, dual Q tile (CTA covers 2*CTA_Q = 256 rows).
//
// Blueprint: cutedsl-sage-sm100:cutedsl_sage/core_sm100.py (itself derived
// from the CUTLASS sm100 FMHA example). Warp roles (core_sm100.py L141-148):
//   warps  0- 3  softmax0    (tile 0: S drain, online softmax, P pack)
//   warps  4- 7  softmax1    (tile 1)
//   warps  8-11  correction  (O rescale by o_scale; final epilogue math)
//   warp  12     mma         (single elected thread issues every tcgen05.mma)
//   warp  13     load        (single elected thread issues every TMA)
//   warp  14     epilogue    (idle until M3: TMA-store epilogue)
//   warp  15     empty
// setmaxnreg budgets 192/192/88/40 per warpgroup: 128*(192+192+88+40)
// = 65536 registers, exactly one CTA's worth (SM register file); see the
// rationale at kNumRegs* below for the 8-reg shift off the cutedsl split.
//
// TMEM plan (512 columns, always alloc 512; lane = attention row within a
// tile; both tiles use all 128 lanes at different columns):
//   S0 [  0,128)   128x128 s32 QK acc, tile 0
//     vec0 [0,2)   f32 (m_prev, row_max) softmax->correction hand-off, and
//                  (denom, row_max) as the final item; aliases S0 cols 0-1
//     P0 [32,64)   e4m3 packed 4/word (TS A-operand layout); aliases S0
//   S1 [128,256)   tile 1, same internal structure (vec1 @128, P1 @160)
//   O0 [256,256+HD)          128xHD f32 PV acc, tile 0
//   O1 [256+HD,256+2*HD)     tile 1 (d64 leaves [384,512) unused)
//
// Bit-exactness contract (M0-M2): every float op sequence of softmax /
// correction / epilogue is copied verbatim from qk_int_sv_f8_cuda_sm100.cu
// (line references at each block). Copies are annotated instead of extracted
// into a shared header so the existing kernel's TU keeps byte-identical
// device text (SASS gate). Softmax reads each S row once - two x64
// tcgen05.ld per KV block, the widest form ptxas accepts under the
// 128-register entry target - and keeps the whole dequanted row in
// registers (the 128-thread kernel's value sequence, :407-449). History: a
// two-pass variant (re-reading TMEM for the exp2 pass) doubled the
// tcgen05.ld traffic and lost 11% end to end; the round-2 single-pass
// version chained four x32 loads and still stalled on long_scoreboard -
// the analyses and the register accounts are in
// bench/sm100_review/C1_DESIGN.md.
//
// Pipeline/barrier design and the deadlock-freedom + TMEM alias-hazard
// proofs live in bench/sm100_review/barrier_ledger.md (16-warp section).
//
// Compile modes mirror qk_int_sv_f8_cuda_sm100.cu:
//   default                   - kernel + torch launcher (extension build)
//   -DSAGE_SM100_DEVICE_ONLY  - kernel only, torch-free (ptxas probe TU)

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

namespace sage {
namespace sm100 {

// File-local wrappers the ws kernel needs beyond tcgen05.cuh. Kept here (not
// in the shared header) so the existing sm100 TU's preprocessed device text
// is untouched; promote to tcgen05.cuh once the ws kernel is hardware-proven.
namespace ws {

// setmaxnreg needs an immediate; ptxas honors it only when the kernel entry
// register count is determinable => the kernel must carry
// __launch_bounds__(512, 1) (verified nvcc 13.3 / sm_100a: without minBlocks
// ptxas emits C7508 "'setmaxnreg' ignored").
template<int N>
__device__ __forceinline__ void warpgroup_reg_alloc()
{
#ifdef SAGE_TCGEN05_ENABLED
    cuda::ptx::setmaxnreg_inc(cuda::ptx::n32_t<N>{});
#else
    __trap();
#endif
}

template<int N>
__device__ __forceinline__ void warpgroup_reg_dealloc()
{
#ifdef SAGE_TCGEN05_ENABLED
    cuda::ptx::setmaxnreg_dec(cuda::ptx::n32_t<N>{});
#else
    __trap();
#endif
}

// ---------------------------------------------------------------------------
// 32-bit shared-space barrier/TMA helpers. The tma.cuh helpers take generic
// pointers and convert per call; through the warp-role lambdas that keeps the
// 64-bit generic smem base live across the setmaxnreg regions, which ptxas
// then carries on the stack (measured spills). Converting once per branch to
// u32 shared-space addresses keeps everything in single registers. The asm
// bodies are the u32 cores of the tma.cuh / cuda::ptx equivalents.
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t smem_u32(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// tma.cuh wait(), u32 form (same opaque single-asm spin loop, same rationale)
__device__ __forceinline__ void wait_bar(uint32_t bar, int kPhaseBit)
{
    asm volatile("{\n"
                 ".reg .pred                P1;\n"
                 "LAB_WAIT:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                 "@P1                       bra.uni DONE;\n"
                 "bra.uni                   LAB_WAIT;\n"
                 "DONE:\n"
                 "}\n" ::"r"(bar),
                 "r"(kPhaseBit));
}

// mbarrier.arrive, count 1 (cuda::ptx::mbarrier_arrive u32 core)
__device__ __forceinline__ void arrive_bar(uint32_t bar)
{
    asm volatile("{\n"
                 ".reg .b64 state;\n"
                 "mbarrier.arrive.release.cta.shared::cta.b64 state, [%0];\n"
                 "}\n" ::"r"(bar)
                 : "memory");
}

// tma.cuh expect_bytes(), u32 form
template<uint32_t bytes>
__device__ __forceinline__ void expect_bytes_bar(uint32_t bar)
{
    asm volatile("{\n"
                 ".reg .b64 state;\n"
                 "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 state, [%0], %1;\n"
                 "}\n" ::"r"(bar),
                 "n"(bytes)
                 : "memory");
}

// tma.cuh load_async_4D(), u32 dst + u32 barrier form
__device__ __forceinline__ void
load_async_4D(uint32_t dst, void const* const src_tma_map, uint32_t bar, int c0, int c1, int c2, int c3)
{
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3, %4, %5}], [%6];\n" ::"r"(dst),
                 "l"(src_tma_map),
                 "r"(c0),
                 "r"(c1),
                 "r"(c2),
                 "r"(c3),
                 "r"(bar)
                 : "memory");
}

// tcgen05.cuh tcgen05_commit(), u32 form
__device__ __forceinline__ void commit_bar(uint32_t bar)
{
#ifdef SAGE_TCGEN05_ENABLED
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(bar) : "memory");
#else
    (void)bar;
    __trap();
#endif
}

// 2-column tcgen05.ld/st for the vec hand-off (tcgen05.cuh starts at x4).
__device__ __forceinline__ void tmem_ld_32x32b_x2(uint32_t r[2], uint32_t tmem_addr)
{
#ifdef SAGE_TCGEN05_ENABLED
    cuda::ptx::tcgen05_ld_32x32b(*reinterpret_cast<uint32_t(*)[2]>(r), tmem_addr);
#else
    r[0] = 0u;
    r[1] = 0u;
    (void)tmem_addr;
    __trap();
#endif
}

// 64-column tcgen05.ld: the softmax S-row drain (tcgen05.cuh stops at x32).
// One LDTM.x64 moves half the 128-col row, so each KV block exposes two
// TMEM-load latencies instead of four chained x32 round trips, still with
// exactly one outstanding tcgen05.ld (stays clear of the batch-issue
// pattern that hung in the A5 experiment). x64 is the widest feasible
// form here: the x128 variant is a single instruction whose destination
// block alone (128 regs + the address operand) exceeds the 128-register
// entry target that __launch_bounds__(512, 1) pins, and ptxas rejects it
// with C7602 "Insufficient registers" regardless of the setmaxnreg region
// budget (measured, nvcc 13.3).
__device__ __forceinline__ void tmem_ld_32x32b_x64(uint32_t r[64], uint32_t tmem_addr)
{
#ifdef SAGE_TCGEN05_ENABLED
    cuda::ptx::tcgen05_ld_32x32b(*reinterpret_cast<uint32_t(*)[64]>(r), tmem_addr);
#else
#pragma unroll
    for (uint32_t i = 0; i < 64; ++i) {
        r[i] = 0u;
    }
    (void)tmem_addr;
    __trap();
#endif
}

__device__ __forceinline__ void tmem_st_32x32b_x2(uint32_t tmem_addr, const uint32_t r[2])
{
#ifdef SAGE_TCGEN05_ENABLED
    cuda::ptx::tcgen05_st_32x32b(tmem_addr, *reinterpret_cast<const uint32_t(*)[2]>(r));
#else
    (void)tmem_addr;
    (void)r;
    __trap();
#endif
}

}  // namespace ws

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
//
// Divergence audit invariants (same discipline as the 128-thread kernel):
//   * every tcgen05::mma_* / tcgen05_commit sits under
//     `warp_idx == kMmaWarp && elect_one()` (single issuing thread);
//   * every tcgen05::tmem_ld/st/wait call is warpgroup-uniform: the softmax /
//     correction loops iterate grid-uniform trip counts;
//   * every cross-warp TMEM producer->consumer hand-off is bracketed by
//     tcgen05_fence_before_sync(); mbarrier arrive / wait;
//     tcgen05_fence_after_sync().

template<uint32_t         CTA_Q,  // Q rows per tile; the CTA covers 2*CTA_Q
         uint32_t         CTA_K,
         uint32_t         NUM_THREADS,
         uint32_t         head_dim,
         QuantGranularity Q_GRAN,
         QuantGranularity K_GRAN,
         typename DTypeOut,
         MaskMode mask_mode    = MaskMode::kNone,
         bool     return_lse   = false,
         bool     fuse_v_scale = false>
__global__ void __launch_bounds__(NUM_THREADS, 1)
    qk_int8_sv_f8_attn_kernel_sm100_ws(const __grid_constant__ CUtensorMap tensorMapQ,
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
                                       const uint32_t qo_len,
                                       const uint32_t kv_len,
                                       const uint32_t qo_per_kv_head,
                                       float          sm_scale)
{
    // --- static shape checks ---
    static_assert(NUM_THREADS == 512, "C1: 16 warps, full warp specialization");
    static_assert(CTA_Q == 128, "M=128 is the native cta_group::1 MMA shape (per tile)");
    static_assert(CTA_K == 128, "N=128 tile; V^T rows are CTA_K bytes (128B swizzle)");
    static_assert(head_dim == 64 || head_dim == 128, "dispatched head dims");
    static_assert(head_dim % 32 == 0, "QK MMA is chained in K=32 steps");

    // --- warp roles ---
    constexpr uint32_t kMmaWarp  = 12;
    constexpr uint32_t kLoadWarp = 13;
    // warps 14 (epilogue, reserved for the M3 TMA-store path) and 15 (empty)
    // dealloc registers and exit.
    // Budgets sum to exactly 512 regs/thread-column = 64K/SM. The cutedsl
    // split is 192/192/96/32; this kernel moves 8 regs from correction (ptxas
    // ceiling 77) to the mma/load group (raw-mbarrier bookkeeping needs ~38
    // in the d64 build; at 32 ptxas held the overflow on the stack).
    constexpr int kNumRegsSoftmax    = 192;
    constexpr int kNumRegsCorrection = 88;
    constexpr int kNumRegsOther      = 40;
    static_assert(2 * 128 * kNumRegsSoftmax + 128 * kNumRegsCorrection + 128 * kNumRegsOther == 64 * 1024,
                  "warpgroup register budgets must exactly cover the SM register file");

    // --- TMEM column plan (see file header) ---
    constexpr uint32_t TMEM_COL_S0     = 0;
    constexpr uint32_t TMEM_COL_S1     = 128;
    constexpr uint32_t TMEM_COL_VEC    = 0;   // offset inside each S tile
    constexpr uint32_t TMEM_COL_P      = 32;  // offset inside each S tile
    constexpr uint32_t TMEM_COL_O0     = 256;
    constexpr uint32_t TMEM_COL_O1     = 256 + head_dim;
    constexpr uint32_t TMEM_COLS_TOTAL = 512;  // one CTA/SM by construction; alloc all
    static_assert(TMEM_COL_O1 + head_dim <= TMEM_COLS_TOTAL, "TMEM budget exceeded");

    // --- derived tile counts (identical to the 128-thread kernel; the
    //     softmax S drain is a single x128 tcgen05.ld, not tiled) ---
    constexpr uint32_t num_tiles_qk_inner = head_dim / 32;
    constexpr uint32_t num_tiles_pv_inner = CTA_K / 32;
    constexpr uint32_t num_tiles_o        = head_dim / 32;

    // --- smem plan: Q double tile + shared K/V 4-slot ring ---
    constexpr uint32_t SMEM_Q_BYTES  = CTA_Q * head_dim;  // per tile
    constexpr uint32_t SMEM_KV_BYTES = CTA_K * head_dim;  // per ring slot (K tile or V^T tile)
    constexpr uint32_t kKvStages     = 4;                 // cutedsl kv_stage=4 (8-bit inputs)
    static_assert(2 * SMEM_Q_BYTES + kKvStages * SMEM_KV_BYTES <= 227 * 1024, "smem budget exceeded");

    // --- smem descriptors + MMA instruction descriptors (same as the
    //     128-thread kernel; parity-tested in bench/sm100_review) ---
    constexpr tcgen05::SmemSwizzleMode QK_SWIZZLE =
        (head_dim == 128) ? tcgen05::SmemSwizzleMode::kSwizzle128B : tcgen05::SmemSwizzleMode::kSwizzle64B;
    constexpr uint32_t                 QK_SBO    = 8 * head_dim;
    constexpr tcgen05::SmemSwizzleMode V_SWIZZLE = tcgen05::SmemSwizzleMode::kSwizzle128B;
    constexpr uint32_t                 V_SBO     = 8 * CTA_K;

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

    const uint32_t warp_idx   = threadIdx.x / 32;
    const uint32_t warp_group = warp_idx / 4;

    extern __shared__ __align__(1024) int8_t smem_[];
    int8_t*                                  sQ      = smem_;                     // 2 tiles
    int8_t*                                  sKVring = smem_ + 2 * SMEM_Q_BYTES;  // 4 slots

    // --- barriers (roles/counts: bench/sm100_review/barrier_ledger.md).
    //     One array so every role addresses them as a single u32 base plus
    //     compile-time byte offsets - separate arrays kept several live base
    //     registers in the 32-reg mma/load region and spilled (measured). ---
    enum BarId : uint32_t {
        kBarQFull     = 0,   // +tile: TMA expect_tx, count 1, used once each
        kBarKvFull    = 2,   // +slot: TMA expect_tx, count 1, 4-slot ring
        kBarKvEmpty   = 6,   // +slot: tcgen05.commit by mma warp, count 1
        kBarSFull     = 10,  // +tile: tcgen05.commit after QK chain
        kBarSEmpty    = 12,  // +tile: 128 softmax arrivals = S drained + P stored
        kBarVecFull   = 14,  // +tile: 128 softmax arrivals = vec stored
        kBarVecEmpty  = 16,  // +tile: 128 correction arrivals = vec read
        kBarCorrFull  = 18,  // +tile: tcgen05.commit after PV chain
        kBarCorrEmpty = 20,  // +tile: 128 correction arrivals = rescale stored
        kBarDealloc   = 22,  // 384 arrivals (softmax0/1 + correction)
        kNumBars      = 23,
    };
    __shared__ __align__(8) uint64_t bars[kNumBars];
    __shared__ __align__(4) uint32_t tmem_addr_slot;
    __shared__ __align__(16) float sV_scale[fuse_v_scale ? head_dim : 1];

    if (threadIdx.x == 0) {
#pragma unroll
        for (uint32_t t = 0; t < 2; t++) {
            tcgen05::init_barrier(&bars[kBarQFull + t], 1);
            tcgen05::init_barrier(&bars[kBarSFull + t], 1);
            tcgen05::init_barrier(&bars[kBarSEmpty + t], 128);
            tcgen05::init_barrier(&bars[kBarVecFull + t], 128);
            tcgen05::init_barrier(&bars[kBarVecEmpty + t], 128);
            tcgen05::init_barrier(&bars[kBarCorrFull + t], 1);
            tcgen05::init_barrier(&bars[kBarCorrEmpty + t], 128);
        }
#pragma unroll
        for (uint32_t s = 0; s < 4; s++) {
            tcgen05::init_barrier(&bars[kBarKvFull + s], 1);
            tcgen05::init_barrier(&bars[kBarKvEmpty + s], 1);
        }
        tcgen05::init_barrier(&bars[kBarDealloc], 3 * 128);
    }

    // --- stage v_scale in smem (published by the __syncthreads below) ---
    if constexpr (fuse_v_scale) {
        if (threadIdx.x < head_dim) {
            const float* V_scale_base_ptr = V_scale
                                            + static_cast<int64_t>(blockIdx.z) * (gridDim.y / qo_per_kv_head) * head_dim
                                            + static_cast<int64_t>(blockIdx.y / qo_per_kv_head) * head_dim;
            sV_scale[threadIdx.x] = V_scale_base_ptr[threadIdx.x];
        }
    }

    // --- TMEM: warp-collective alloc of all 512 cols by the mma warp, then
    //     relinquish (CUTLASS ordering); published by the __syncthreads ---
    if (warp_idx == kMmaWarp) {
        tcgen05::tmem_alloc(&tmem_addr_slot, TMEM_COLS_TOTAL);
        tcgen05::tmem_relinquish();
    }

    __syncthreads();  // the only CTA-wide barrier; all cross-warp sync below is mbarrier-based

    // Everything role-specific (trip counts, tmem base, block ids, row ids) is
    // (re)computed inside each warpgroup branch: a value computed before the
    // setmaxnreg boundary and used after it must be carried across, and in
    // the 32-register region that shows up as stack spills (measured).
    // KV trip counts per tile (differentiated causal trips; per 128-row tile
    // this equals the 128-thread kernel's num_iterations with the q-block
    // index 2*bx / 2*bx+1):
    auto trip_count = [&](uint32_t tile) {
        const uint32_t kblk = div_ceil(kv_len, CTA_K);
        if constexpr (mask_mode == MaskMode::kCausal) {
            return min(2 * blockIdx.x + 1 + tile, kblk);
        }
        return kblk;
    };

    // =========================================================================
    // softmax0 / softmax1 warpgroups
    // =========================================================================
    if (warp_group <= 1) {
        ws::warpgroup_reg_alloc<kNumRegsSoftmax>();

        const uint32_t tile     = warp_group;  // 0 or 1
        const uint32_t trip     = trip_count(tile);
        const uint32_t lane_row = threadIdx.x % 128;  // S row within the tile == TMEM lane

        const uint32_t batch_id     = blockIdx.z;
        const uint32_t cta_idx_q    = blockIdx.x;  // covers Q rows [256*bx, 256*bx + 256)
        const uint32_t head_id      = blockIdx.y;
        const uint32_t num_qo_heads = gridDim.y;

        const uint32_t bars_u32   = ws::smem_u32(bars);
        const uint32_t bar_full   = bars_u32 + (kBarSFull + tile) * 8;
        const uint32_t bar_empty  = bars_u32 + (kBarSEmpty + tile) * 8;
        const uint32_t bar_vfull  = bars_u32 + (kBarVecFull + tile) * 8;
        const uint32_t bar_vempty = bars_u32 + (kBarVecEmpty + tile) * 8;
        const uint32_t tmem_row   = tmem_addr_slot + ((warp_idx % 4) * 32 << 16) + (tile ? TMEM_COL_S1 : TMEM_COL_S0);

        float local_sm_scale = sm_scale * math::log2e;  // :237 (softmax runs in base 2)

        const uint32_t q_idx = cta_idx_q * (2 * CTA_Q) + tile * CTA_Q + lane_row;

        // --- quant scale indexing (mirrors :249-284, with the q-block index
        //     rebuilt from qo_len: grid.x covers two 128-row blocks, so
        //     gridDim.x is no longer the number of quant blocks). The block
        //     index is clamped for the tail CTA's fully-OOB tile 1 (rows are
        //     never stored; the clamp only prevents an OOB scale read). ---
        const uint32_t num_qblocks = div_ceil(qo_len, CTA_Q);
        const uint32_t qblk        = min(2 * cta_idx_q + tile, num_qblocks - 1);
        int64_t        q_scale_idx;
        if constexpr (Q_GRAN == QuantGranularity::kPerWarp) {
            const uint32_t num_warp_tiles_q = num_qblocks * (CTA_Q / 32);
            q_scale_idx                     = static_cast<int64_t>(batch_id) * num_qo_heads * num_warp_tiles_q
                          + static_cast<int64_t>(head_id) * num_warp_tiles_q + qblk * (CTA_Q / 32) + lane_row / 32;
        }
        else if constexpr (Q_GRAN == QuantGranularity::kPerThread) {
            const uint32_t num_warp_tiles_q = num_qblocks * (CTA_Q / 32);
            q_scale_idx                     = static_cast<int64_t>(batch_id) * num_qo_heads * (num_warp_tiles_q * 8)
                          + static_cast<int64_t>(head_id) * (num_warp_tiles_q * 8) + qblk * ((CTA_Q / 32) * 8)
                          + (lane_row / 32) * 8 + (lane_row % 8);
        }

        const float* K_scale_base_ptr;
        if constexpr (K_GRAN == QuantGranularity::kPerWarp) {
            const uint32_t num_ctas_k = div_ceil(kv_len, CTA_K);
            K_scale_base_ptr = K_scale + static_cast<int64_t>(batch_id) * (num_qo_heads / qo_per_kv_head) * num_ctas_k
                               + static_cast<int64_t>(head_id / qo_per_kv_head) * num_ctas_k;
        }
        else if constexpr (K_GRAN == QuantGranularity::kPerThread) {
            const uint32_t num_ctas_k = div_ceil(kv_len, CTA_K);
            K_scale_base_ptr          = K_scale
                               + static_cast<int64_t>(batch_id) * (num_qo_heads / qo_per_kv_head) * (num_ctas_k * 4)
                               + static_cast<int64_t>(head_id / qo_per_kv_head) * (num_ctas_k * 4);
        }
        constexpr uint32_t k_scale_advance_offset = (K_GRAN == QuantGranularity::kPerWarp) ? 1 : 4;

        const float q_scale = Q_scale[q_scale_idx];

        // --- flash-attention state (mirrors :287-288) ---
        float row_max = -5000000.0f;
        float denom   = 1.0f;

        int s_full_phase    = 0;
        int vec_empty_phase = 0;

        // -------------------------------------------------------------------
        // Per-KV-block softmax step. One pass over the S row: two x64
        // tcgen05.ld + wait round trips drain the whole row (two exposed
        // TMEM-load latencies per KV block; round 2 chained four x32 loads
        // and long_scoreboard stayed the top stall - see C1_DESIGN.md round
        // 3, which also records why x128 is rejected by ptxas). Still
        // exactly one outstanding tcgen05.ld, NOT the batch-issue/
        // collective-wait pattern that hung in the A5 experiment. Each raw
        // word is dequanted+masked in place and folded into m_local; the
        // whole row then stays in registers, so nothing touches TMEM
        // between the vec store (aliases S cols [0,2)) and the P store.
        // The per-element dequant/max/exp2/d_sum/pack value sequences are
        // those of the 128-thread kernel (:407-449, :459-474; see file
        // header). is_last folds the causal/OOB mask exactly like :431-444.
        // -------------------------------------------------------------------
        auto softmax_step = [&](auto is_last_t, uint32_t iter) {
            constexpr bool is_last = decltype(is_last_t)::value;

            // K-scale(s) for this tile (mirrors :400-410)
            float dequant_scale[(K_GRAN == QuantGranularity::kPerThread) ? 4 : 1];
            if constexpr (K_GRAN == QuantGranularity::kPerThread) {
#pragma unroll
                for (uint32_t cls = 0; cls < 4; cls++) {
                    dequant_scale[cls] = q_scale * K_scale_base_ptr[iter * k_scale_advance_offset + cls];
                }
            }
            else {
                dequant_scale[0] = q_scale * K_scale_base_ptr[iter * k_scale_advance_offset];
            }

            // dequant + fused mask for S column j (mirrors :420-444)
            auto s_of = [&](uint32_t raw, uint32_t j) -> float {
                float dequant_scale_j;
                if constexpr (K_GRAN == QuantGranularity::kPerThread) {
                    dequant_scale_j = dequant_scale[(j % 8) / 2];
                }
                else {
                    dequant_scale_j = dequant_scale[0];
                }
                float s = __int2float_rz(static_cast<int32_t>(raw)) * dequant_scale_j;
                if constexpr (is_last) {
                    const uint32_t kv_idx = iter * CTA_K + j;
                    bool           is_out_of_bounds;
                    if constexpr (mask_mode == MaskMode::kCausal) {
                        is_out_of_bounds = (kv_idx > q_idx) || (kv_idx >= kv_len);
                    }
                    else {
                        is_out_of_bounds = (kv_idx >= kv_len);
                    }
                    if (is_out_of_bounds) {
                        s = -5000000.0f;
                    }
                }
                return s;
            };

            ws::wait_bar(bar_full, s_full_phase);
            s_full_phase ^= 1;

            // ---- load + dequant + mask + row max, whole row retained
            //      (:407-449; per-element op order unchanged - same ascending
            //      column order as the four-chunk version, so every float op
            //      sees the same inputs in the same order). The row array is
            //      uint32_t so each f32 result can overwrite its raw word in
            //      place: the ld's in-flight block and the retained row share
            //      registers instead of stacking a separate staging array. ----
            uint32_t RS_row[CTA_K];
            float    m_local = -5000000.0f;
#pragma unroll
            for (uint32_t c = 0; c < CTA_K / 64; c++) {
                ws::tmem_ld_32x32b_x64(&RS_row[c * 64], tmem_row + c * 64);
                tcgen05::tmem_ld_wait();
#pragma unroll
                for (uint32_t jj = 0; jj < 64; jj++) {
                    const float s       = s_of(RS_row[c * 64 + jj], c * 64 + jj);
                    m_local             = max(m_local, s);
                    RS_row[c * 64 + jj] = __float_as_uint(s);
                }
            }

            // ---- online softmax update (mirrors :454-458) ----
            const float m_prev  = row_max;
            row_max             = max(row_max, fmaf(m_local, local_sm_scale, -S_FP8_OFFSET));
            const float o_scale = math::ptx_exp2(m_prev - row_max);
            denom *= o_scale;

            // ---- vec = (m_prev, row_max) -> correction, sent before the exp2
            //      segment so the O rescale overlaps it. Slot held from the
            //      previous step's tail acquire (virgin on step 0). All S
            //      reads are done, so aliasing cols [0,2) is harmless. ----
            uint32_t vec_u32[2];
            vec_u32[0] = __float_as_uint(m_prev);
            vec_u32[1] = __float_as_uint(row_max);
            ws::tmem_st_32x32b_x2(tmem_row + TMEM_COL_VEC, vec_u32);
            tcgen05::tmem_st_wait();
            tcgen05::tcgen05_fence_before_sync();
            ws::arrive_bar(bar_vfull);

            // ---- p = exp2(fma(s, scale, -row_max)), d_sum, e4m3 pack from
            //      the retained row (p / d_sum / pack value sequences match
            //      :459-474; the pack is fused per 4 columns so each RS_f32
            //      quad dies as its RP word is born - liveness only falls
            //      from here). ----
            const float neg_row_max = -row_max;
            float       d_sum       = 0.0f;
            uint32_t    RP_u32[CTA_K / 4];
#pragma unroll
            for (uint32_t w = 0; w < CTA_K / 4; w++) {
                float p[4];
#pragma unroll
                for (uint32_t b = 0; b < 4; b++) {
                    p[b] = math::ptx_exp2(fmaf(__uint_as_float(RS_row[4 * w + b]), local_sm_scale, neg_row_max));
                    d_sum += p[b];
                }
                floatx4_to_e4m3x4(&RP_u32[w], &p[0], &p[2]);
            }
            denom += d_sum;

            // ---- P -> TMEM (TS A-operand layout, mirrors :493-499; cols
            //      [32,64) alias S but the row is already in registers), then
            //      release the S tile: this single arrival tells the mma warp
            //      both "S drained" (next QK may overwrite) and "P ready"
            //      (PV may read). ----
            tcgen05::tmem_st_32x32b_x32(tmem_row + TMEM_COL_P, RP_u32);
            tcgen05::tmem_st_wait();
            tcgen05::tcgen05_fence_before_sync();
            ws::arrive_bar(bar_empty);

            // acquire the vec slot for the next store (correction read it)
            ws::wait_bar(bar_vempty, vec_empty_phase);
            vec_empty_phase ^= 1;
        };

        for (uint32_t iter = 0; iter + 1 < trip; iter++) {
            softmax_step(std::false_type{}, iter);
        }
        softmax_step(std::true_type{}, trip - 1);  // peeled: fused causal/OOB mask

        // ---- final vec = (denom, row_max): correction derives the epilogue
        //      d_rcp from denom (bit-identical: f32 round-trips TMEM exactly).
        //      The slot was acquired at the tail of the last step, so the
        //      store cannot clobber a vec correction has not read. ----
        uint32_t vec_u32[2];
        vec_u32[0] = __float_as_uint(denom);
        vec_u32[1] = __float_as_uint(row_max);
        ws::tmem_st_32x32b_x2(tmem_row + TMEM_COL_VEC, vec_u32);
        tcgen05::tmem_st_wait();
        tcgen05::tcgen05_fence_before_sync();
        ws::arrive_bar(bar_vfull);

        if constexpr (return_lse) {
            // mirrors :601-608 (one thread-local store per row)
            if (q_idx < qo_len) {
                Lse[static_cast<int64_t>(batch_id) * (static_cast<int64_t>(qo_len) * num_qo_heads)
                    + static_cast<int64_t>(head_id) * qo_len + q_idx] = math::ptx_log2(denom) + row_max;
            }
        }

        ws::arrive_bar(bars_u32 + kBarDealloc * 8);  // prior tcgen05 ops already fenced above
    }

    // =========================================================================
    // correction warpgroup (warps 8-11): O rescale per KV block, then the
    // register->global epilogue (M0-M2; M3 moves the store to smem+TMA).
    // =========================================================================
    else if (warp_group == 2) {
        ws::warpgroup_reg_dealloc<kNumRegsCorrection>();

        const uint32_t trip0     = trip_count(0);
        const uint32_t trip1     = trip_count(1);
        const uint32_t lane_row  = threadIdx.x % 128;  // O row within a tile == TMEM lane
        const uint32_t batch_id  = blockIdx.z;
        const uint32_t head_id   = blockIdx.y;
        const uint32_t tmem_lane = tmem_addr_slot + ((warp_idx % 4) * 32 << 16);

        // per-tile pipeline state as scalars; the lambdas take everything by
        // reference so no local array is ever dynamically indexed (spills)
        int vec0_full_phase = 0, vec1_full_phase = 0;
        int corr0_full_phase = 0, corr1_full_phase = 0;

        // consume a vec the O tile does not need (step 0: O is overwritten by
        // the first PV with enable_D=0, mirroring the `iter > 0` skip at :478)
        auto discard_vec = [&](uint32_t bar_vfull, uint32_t bar_vempty, int& vphase) {
            ws::wait_bar(bar_vfull, vphase);
            vphase ^= 1;
            ws::arrive_bar(bar_vempty);
        };

        // read vec -> value pair (fenced tcgen05.ld, releases the vec slot)
        auto read_vec =
            [&](uint32_t bar_vfull, uint32_t bar_vempty, int& vphase, uint32_t vec_addr, float& v0, float& v1) {
                ws::wait_bar(bar_vfull, vphase);
                vphase ^= 1;
                tcgen05::tcgen05_fence_after_sync();
                uint32_t vec_u32[2];
                ws::tmem_ld_32x32b_x2(vec_u32, vec_addr);
                tcgen05::tmem_ld_wait();
                v0 = __uint_as_float(vec_u32[0]);
                v1 = __uint_as_float(vec_u32[1]);
                ws::arrive_bar(bar_vempty);
            };

        // O_t *= o_scale (float op order mirrors :479-491)
        auto rescale = [&](uint32_t bar_vfull,
                           uint32_t bar_vempty,
                           int&     vphase,
                           uint32_t vec_addr,
                           uint32_t bar_cfull,
                           uint32_t bar_cempty,
                           int&     cphase,
                           uint32_t tmem_o_row) {
            float m_prev, rmax;
            read_vec(bar_vfull, bar_vempty, vphase, vec_addr, m_prev, rmax);
            const float o_scale = math::ptx_exp2(m_prev - rmax);  // same expr as :456 -> same bits
            ws::wait_bar(bar_cfull, cphase);
            cphase ^= 1;
#pragma unroll
            for (uint32_t c = 0; c < num_tiles_o; c++) {
                uint32_t RO_u32[32];
                tcgen05::tmem_ld_32x32b_x32(RO_u32, tmem_o_row + c * 32);
                tcgen05::tmem_ld_wait();
#pragma unroll
                for (uint32_t jj = 0; jj < 32; jj++) {
                    RO_u32[jj] = __float_as_uint(__uint_as_float(RO_u32[jj]) * o_scale);
                }
                tcgen05::tmem_st_32x32b_x32(tmem_o_row + c * 32, RO_u32);
            }
            tcgen05::tmem_st_wait();
            tcgen05::tcgen05_fence_before_sync();
            ws::arrive_bar(bar_cempty);
        };

        // final O_t = O_t * d_rcp (* v_scale) -> global (mirrors :567-599)
        auto epilog = [&](uint32_t bar_vfull,
                          uint32_t bar_vempty,
                          int&     vphase,
                          uint32_t vec_addr,
                          uint32_t bar_cfull,
                          int&     cphase,
                          uint32_t tmem_o_row,
                          uint32_t q_idx) {
            float denom_t, rmax_unused;
            read_vec(bar_vfull, bar_vempty, vphase, vec_addr, denom_t, rmax_unused);
            ws::wait_bar(bar_cfull, cphase);
            cphase ^= 1;

            const float d_rcp      = math::ptx_rcp(denom_t);
            const bool  row_valid  = q_idx < qo_len;
            DTypeOut*   O_lane_ptr = O + static_cast<int64_t>(batch_id) * stride_batch_o
                                   + static_cast<int64_t>(head_id) * stride_h_o
                                   + static_cast<int64_t>(q_idx) * stride_seq_o;

            // The scale/convert/store loops of :572-599 are fused per element
            // pair here so only one 32-word array is live (the second array
            // pushed the region past the 96-reg budget - measured spills).
            // Per-element op sequence (mul d_rcp, mul v_scale, cvt) is
            // unchanged and elements are independent, so bits are unchanged.
#pragma unroll
            for (uint32_t c = 0; c < num_tiles_o; c++) {
                uint32_t RO_u32[32];
                tcgen05::tmem_ld_32x32b_x32(RO_u32, tmem_o_row + c * 32);
                tcgen05::tmem_ld_wait();

#pragma unroll
                for (uint32_t jj = 0; jj < 16; jj++) {
                    float lo = __uint_as_float(RO_u32[2 * jj]) * d_rcp;
                    float hi = __uint_as_float(RO_u32[2 * jj + 1]) * d_rcp;
                    if constexpr (fuse_v_scale) {
                        lo *= sV_scale[c * 32 + 2 * jj];
                        hi *= sV_scale[c * 32 + 2 * jj + 1];
                    }
                    if (row_valid) {
                        const float2 o2 = make_float2(lo, hi);
                        if constexpr (std::is_same<DTypeOut, half>::value) {
                            ((half2*)(O_lane_ptr + c * 32 + 2 * jj))[0] = __float22half2_rn(o2);
                        }
                        else {
                            ((nv_bfloat162*)(O_lane_ptr + c * 32 + 2 * jj))[0] = __float22bfloat162_rn(o2);
                        }
                    }
                }
            }
        };

        const uint32_t vec0_addr   = tmem_lane + TMEM_COL_S0 + TMEM_COL_VEC;
        const uint32_t vec1_addr   = tmem_lane + TMEM_COL_S1 + TMEM_COL_VEC;
        const uint32_t tmem_o0_row = tmem_lane + TMEM_COL_O0;
        const uint32_t tmem_o1_row = tmem_lane + TMEM_COL_O1;
        const uint32_t q_idx0      = blockIdx.x * (2 * CTA_Q) + lane_row;

        const uint32_t bars_u32 = ws::smem_u32(bars);
        const uint32_t vf0 = bars_u32 + kBarVecFull * 8, ve0 = bars_u32 + kBarVecEmpty * 8;
        const uint32_t vf1 = vf0 + 8, ve1 = ve0 + 8;
        const uint32_t cf0 = bars_u32 + kBarCorrFull * 8, ce0 = bars_u32 + kBarCorrEmpty * 8;
        const uint32_t cf1 = cf0 + 8, ce1 = ce0 + 8;

        discard_vec(vf0, ve0, vec0_full_phase);
        discard_vec(vf1, ve1, vec1_full_phase);
        for (uint32_t j = 1; j < trip0; j++) {
            rescale(vf0, ve0, vec0_full_phase, vec0_addr, cf0, ce0, corr0_full_phase, tmem_o0_row);
            rescale(vf1, ve1, vec1_full_phase, vec1_addr, cf1, ce1, corr1_full_phase, tmem_o1_row);
        }
        for (uint32_t j = trip0; j < trip1; j++) {  // causal S1-only rounds
            rescale(vf1, ve1, vec1_full_phase, vec1_addr, cf1, ce1, corr1_full_phase, tmem_o1_row);
        }
        epilog(vf0, ve0, vec0_full_phase, vec0_addr, cf0, corr0_full_phase, tmem_o0_row, q_idx0);
        epilog(vf1, ve1, vec1_full_phase, vec1_addr, cf1, corr1_full_phase, tmem_o1_row, q_idx0 + CTA_Q);

        tcgen05::tcgen05_fence_before_sync();  // order the epilogue tmem_lds before dealloc
        ws::arrive_bar(bars_u32 + kBarDealloc * 8);
    }

    // =========================================================================
    // mma / load / epilogue / empty warpgroup (warps 12-15)
    // =========================================================================
    else {
        ws::warpgroup_reg_dealloc<kNumRegsOther>();

        if (warp_idx == kMmaWarp) {
            if (tcgen05::elect_one()) {
                const uint32_t trip0 = trip_count(0);
                const uint32_t trip1 = trip_count(1);
                // one live TMEM base; per-call +CONST rematerializes freely
                const uint32_t tb = tmem_addr_slot;

                // pipeline state as scalars passed by reference: dynamically
                // indexed local phase arrays would spill (32-reg budget)
                int s0_empty_phase = 0, s1_empty_phase = 0;
                int corr0_empty_phase = 0, corr1_empty_phase = 0;

                // one u32 base for every barrier and smem operand address
                // (see the ws:: helper comment; sKVring == sQ + 2 Q tiles)
                const uint32_t bars_u32 = ws::smem_u32(bars);
                const uint32_t sQ_u32   = ws::smem_u32(sQ);
                const uint32_t ring_u32 = sQ_u32 + 2 * SMEM_Q_BYTES;

                // KV ring item n (K_i = item 2i, V_i = item 2i+1) -> slot n%4;
                // the full barrier of slot s completes once per lap, so the
                // wait phase is derivable from the item: (n/4) & 1.
                auto kv_slot_u32 = [&](uint32_t item) { return ring_u32 + (item & 3) * SMEM_KV_BYTES; };
                auto wait_kv     = [&](uint32_t item) {
                    ws::wait_bar(bars_u32 + (kBarKvFull + (item & 3)) * 8, (item >> 2) & 1);
                };
                // slot free once every prior MMA (incl. its last reader) retired
                auto release_kv = [&](uint32_t item) { ws::commit_bar(bars_u32 + (kBarKvEmpty + (item & 3)) * 8); };

                // S_t = Q_t K_i^T (kind::i8 SS chain, mirrors :375-385)
                auto qk = [&](uint32_t tmem_S_t, uint32_t sQ_t, uint32_t sK_slot, uint32_t bar_full_t) {
#pragma unroll
                    for (uint32_t k_it = 0; k_it < num_tiles_qk_inner; k_it++) {
                        const uint64_t desc_q =
                            tcgen05::make_smem_desc_sm100(sQ_t + k_it * 32, tcgen05::kKMajorLBO, QK_SBO, QK_SWIZZLE);
                        const uint64_t desc_k =
                            tcgen05::make_smem_desc_sm100(sK_slot + k_it * 32, tcgen05::kKMajorLBO, QK_SBO, QK_SWIZZLE);
                        tcgen05::mma_i8_ss(tmem_S_t, desc_q, desc_k, idesc_qk, k_it > 0);
                    }
                    ws::commit_bar(bar_full_t);
                };

                // O_t (+)= P_t V_i (kind::f8f6f4 TS chain, mirrors :525-543).
                // The bar_s_empty wait is the P-ready acquire: softmax arrives
                // only after its P tcgen05.st completed (and it doubles as the
                // S-drained signal for the next QK on this tile).
                auto pv = [&](uint32_t tmem_O_t,
                              uint32_t tmem_P_t,
                              uint32_t bar_p_ready,
                              int&     p_ready_phase,
                              uint32_t bar_o_done,
                              uint32_t sV_slot,
                              bool     accumulate) {
                    ws::wait_bar(bar_p_ready, p_ready_phase);
                    p_ready_phase ^= 1;
                    tcgen05::tcgen05_fence_after_sync();
#pragma unroll
                    for (uint32_t v_it = 0; v_it < num_tiles_pv_inner; v_it++) {
                        const uint64_t desc_v =
                            tcgen05::make_smem_desc_sm100(sV_slot + v_it * 32, tcgen05::kKMajorLBO, V_SBO, V_SWIZZLE);
                        tcgen05::mma_f8f8f32_ts(
                            tmem_O_t, tmem_P_t + v_it * 8, desc_v, idesc_pv, accumulate || (v_it > 0));
                    }
                    ws::commit_bar(bar_o_done);
                };
                // O-slot acquire: correction finished the rescale that must
                // precede this PV (and has therefore read the matching vec -
                // the alias-hazard link, see barrier_ledger.md)
                auto wait_corr_empty = [&](uint32_t bar, int& phase) {
                    ws::wait_bar(bar, phase);
                    phase ^= 1;
                    tcgen05::tcgen05_fence_after_sync();
                };

                // ---- prologue: QK00, QK10, PV00 (cutedsl L621-658) ----
                ws::wait_bar(bars_u32 + kBarQFull * 8, 0);
                wait_kv(0);  // K_0
                qk(tb + TMEM_COL_S0, sQ_u32, kv_slot_u32(0), bars_u32 + kBarSFull * 8);
                ws::wait_bar(bars_u32 + (kBarQFull + 1) * 8, 0);
                qk(tb + TMEM_COL_S1, sQ_u32 + SMEM_Q_BYTES, kv_slot_u32(0), bars_u32 + (kBarSFull + 1) * 8);
                release_kv(0);  // K_0 (fires when QK10 retired)
                wait_kv(1);     // V_0
                pv(tb + TMEM_COL_O0,
                   tb + TMEM_COL_S0 + TMEM_COL_P,
                   bars_u32 + kBarSEmpty * 8,
                   s0_empty_phase,
                   bars_u32 + kBarCorrFull * 8,
                   kv_slot_u32(1),
                   /*accumulate=*/false);

                // ---- steady loop (cutedsl L661-704): per i, in issue order
                //      QK0(i) | PV1(i-1) | QK1(i) | PV0(i) ----
                bool     pv1_started = false;
                uint32_t v_item      = 1;  // ring item of the V the next PV1 consumes
                for (uint32_t i = 1; i < trip0; i++) {
                    wait_kv(2 * i);  // K_i
                    qk(tb + TMEM_COL_S0, sQ_u32, kv_slot_u32(2 * i), bars_u32 + kBarSFull * 8);

                    if (pv1_started) {
                        wait_corr_empty(bars_u32 + (kBarCorrEmpty + 1) * 8, corr1_empty_phase);
                    }
                    pv(tb + TMEM_COL_O1,
                       tb + TMEM_COL_S1 + TMEM_COL_P,
                       bars_u32 + (kBarSEmpty + 1) * 8,
                       s1_empty_phase,
                       bars_u32 + (kBarCorrFull + 1) * 8,
                       kv_slot_u32(v_item),
                       pv1_started);  // PV1(i-1)
                    pv1_started = true;
                    release_kv(v_item);  // V_{i-1}

                    qk(tb + TMEM_COL_S1, sQ_u32 + SMEM_Q_BYTES, kv_slot_u32(2 * i), bars_u32 + (kBarSFull + 1) * 8);
                    release_kv(2 * i);  // K_i

                    wait_kv(2 * i + 1);  // V_i
                    v_item = 2 * i + 1;
                    wait_corr_empty(bars_u32 + kBarCorrEmpty * 8, corr0_empty_phase);
                    pv(tb + TMEM_COL_O0,
                       tb + TMEM_COL_S0 + TMEM_COL_P,
                       bars_u32 + kBarSEmpty * 8,
                       s0_empty_phase,
                       bars_u32 + kBarCorrFull * 8,
                       kv_slot_u32(v_item),
                       /*accumulate=*/true);  // PV0(i)
                }

                // ---- causal S1-only rounds (cutedsl L707-728): PV1 + QK1 ----
                for (uint32_t i = trip0; i < trip1; i++) {
                    wait_kv(2 * i);  // K_i
                    if (pv1_started) {
                        wait_corr_empty(bars_u32 + (kBarCorrEmpty + 1) * 8, corr1_empty_phase);
                    }
                    pv(tb + TMEM_COL_O1,
                       tb + TMEM_COL_S1 + TMEM_COL_P,
                       bars_u32 + (kBarSEmpty + 1) * 8,
                       s1_empty_phase,
                       bars_u32 + (kBarCorrFull + 1) * 8,
                       kv_slot_u32(v_item),
                       pv1_started);  // PV1(i-1)
                    pv1_started = true;
                    release_kv(v_item);
                    qk(tb + TMEM_COL_S1, sQ_u32 + SMEM_Q_BYTES, kv_slot_u32(2 * i), bars_u32 + (kBarSFull + 1) * 8);
                    release_kv(2 * i);
                    wait_kv(2 * i + 1);  // V_i
                    v_item = 2 * i + 1;
                }

                // ---- tail: PV1(trip1-1) (cutedsl L733-743) ----
                if (pv1_started) {
                    wait_corr_empty(bars_u32 + (kBarCorrEmpty + 1) * 8, corr1_empty_phase);
                }
                pv(tb + TMEM_COL_O1,
                   tb + TMEM_COL_S1 + TMEM_COL_P,
                   bars_u32 + (kBarSEmpty + 1) * 8,
                   s1_empty_phase,
                   bars_u32 + (kBarCorrFull + 1) * 8,
                   kv_slot_u32(v_item),
                   pv1_started);
                release_kv(v_item);
            }

            // --- TMEM dealloc: whole warp waits for the 12 TMEM-reading warps
            //     (their arrivals are fenced), then deallocs collectively ---
            ws::wait_bar(ws::smem_u32(&bars[kBarDealloc]), 0);
            tcgen05::tcgen05_fence_after_sync();
            tcgen05::tmem_dealloc(tmem_addr_slot, TMEM_COLS_TOTAL);
        }
        else if (warp_idx == kLoadWarp) {
            if (tcgen05::elect_one()) {
                const uint32_t trip1      = trip_count(1);
                const uint32_t batch_id   = blockIdx.z;
                const uint32_t cta_idx_q  = blockIdx.x;
                const uint32_t head_id    = blockIdx.y;
                const uint32_t kv_head_id = head_id / qo_per_kv_head;
                // issue order Q0, K0, Q1, V0, (K,V)* mirrors cutedsl L589-611;
                // TMA coords/boxes are those of the 128-thread kernel
                // (:314-321, :393-396, :550-554). Ring item n -> slot n%4;
                // laps >= 1 wait for the mma warp's retire-commit, whose
                // phase is derivable from the item: slot n%4 has completed
                // n/4 empty phases before item n needs it -> wait (n/4-1)&1.
                const uint32_t bars_u32      = ws::smem_u32(bars);
                const uint32_t sQ_u32        = ws::smem_u32(sQ);
                const uint32_t ring_u32      = sQ_u32 + 2 * SMEM_Q_BYTES;
                auto           kv_full_bar   = [&](uint32_t item) { return bars_u32 + (kBarKvFull + (item & 3)) * 8; };
                auto           kv_empty_wait = [&](uint32_t item) {
                    if (item >= 4) {
                        ws::wait_bar(bars_u32 + (kBarKvEmpty + (item & 3)) * 8, ((item >> 2) - 1) & 1);
                    }
                };

                ws::expect_bytes_bar<SMEM_Q_BYTES>(bars_u32 + kBarQFull * 8);
                ws::load_async_4D(
                    sQ_u32, &tensorMapQ, bars_u32 + kBarQFull * 8, 0, cta_idx_q * (2 * CTA_Q), head_id, batch_id);
                ws::expect_bytes_bar<SMEM_KV_BYTES>(kv_full_bar(0));
                ws::load_async_4D(ring_u32, &tensorMapK, kv_full_bar(0), 0, 0, kv_head_id, batch_id);
                ws::expect_bytes_bar<SMEM_Q_BYTES>(bars_u32 + (kBarQFull + 1) * 8);
                ws::load_async_4D(sQ_u32 + SMEM_Q_BYTES,
                                  &tensorMapQ,
                                  bars_u32 + (kBarQFull + 1) * 8,
                                  0,
                                  cta_idx_q * (2 * CTA_Q) + CTA_Q,
                                  head_id,
                                  batch_id);
                ws::expect_bytes_bar<SMEM_KV_BYTES>(kv_full_bar(1));
                ws::load_async_4D(ring_u32 + SMEM_KV_BYTES, &tensorMapV, kv_full_bar(1), 0, 0, kv_head_id, batch_id);

                for (uint32_t i = 1; i < trip1; i++) {
                    const uint32_t k_item = 2 * i;
                    kv_empty_wait(k_item);
                    ws::expect_bytes_bar<SMEM_KV_BYTES>(kv_full_bar(k_item));
                    ws::load_async_4D(ring_u32 + (k_item & 3) * SMEM_KV_BYTES,
                                      &tensorMapK,
                                      kv_full_bar(k_item),
                                      0,
                                      i * CTA_K,
                                      kv_head_id,
                                      batch_id);

                    const uint32_t v_item = 2 * i + 1;
                    kv_empty_wait(v_item);
                    ws::expect_bytes_bar<SMEM_KV_BYTES>(kv_full_bar(v_item));
                    ws::load_async_4D(ring_u32 + (v_item & 3) * SMEM_KV_BYTES,
                                      &tensorMapV,
                                      kv_full_bar(v_item),
                                      i * CTA_K,
                                      0,
                                      kv_head_id,
                                      batch_id);
                }
            }
        }
        // warps 14 (epilogue, M3) and 15 (empty): nothing to do, exit
    }
}

#ifndef SAGE_SM100_DEVICE_ONLY

// ---------------------------------------------------------------------------
// Host launcher: clone of qk_int_sv_f8_cuda_sm100.cu's fuse_v_scale launcher
// (:767-902) with the ws kernel, 512 threads, grid.x covering 256 rows, and
// the Q-double + KV-ring smem plan. Selected via SAGEATTN_SM100_WS there.
// ---------------------------------------------------------------------------

torch::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_ws(torch::Tensor query,
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

    auto out_dtype = output.scalar_type();

    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
            DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
                DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
                    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(out_dtype, DTypeOut, {
                        constexpr int CTA_Q       = 128;  // per tile; the CTA covers 2 tiles
                        constexpr int CTA_K       = 128;
                        constexpr int NUM_THREADS = 512;

                        constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

                        TORCH_CHECK(value.size(3) >= div_ceil(kv_len, CTA_K) * CTA_K,
                                    "value.size(3) >= div_ceil(kv_len, CTA_K) * CTA_K failed: value.size(3)=",
                                    value.size(3),
                                    ", kv_len=",
                                    kv_len,
                                    ", CTA_K=",
                                    CTA_K);

                        // quant blocking stays 128-row (independent of grid.x)
                        SAGEATTN_CHECK_QK_SCALE_SHAPES(div_ceil(qo_len, CTA_Q) * (CTA_Q / 32), div_ceil(kv_len, CTA_K));

                        CHECK_SHAPE(value_scale, batch_size, num_kv_heads, head_dim);

                        QKVTensorMaps tma_maps = make_qkv_tensor_maps<CTA_Q, CTA_K, HEAD_DIM>(query, key, value, qkv);

                        auto* kernel = qk_int8_sv_f8_attn_kernel_sm100_ws<CTA_Q,
                                                                          CTA_K,
                                                                          NUM_THREADS,
                                                                          HEAD_DIM,
                                                                          static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                                          static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                                          DTypeOut,
                                                                          mask_mode,
                                                                          RETURN_LSE,
                                                                          true>;
                        // 2 Q tiles + 4-slot KV ring (barriers/v_scale are static smem)
                        size_t smem_bytes =
                            2 * CTA_Q * HEAD_DIM * sizeof(int8_t) + 4 * CTA_K * HEAD_DIM * sizeof(int8_t);
                        sage::set_max_dynamic_smem_once(kernel, smem_bytes, query.get_device());

                        dim3 grid(div_ceil(qo_len, 2 * CTA_Q), num_qo_heads, batch_size);
                        kernel<<<grid, NUM_THREADS, smem_bytes, stream>>>(
                            tma_maps.q,
                            tma_maps.k,
                            tma_maps.v,
                            reinterpret_cast<float*>(query_scale.data_ptr()),
                            reinterpret_cast<float*>(key_scale.data_ptr()),
                            reinterpret_cast<float*>(value_scale.data_ptr()),
                            reinterpret_cast<DTypeOut*>(output.data_ptr()),
                            (RETURN_LSE) ? reinterpret_cast<float*>(lse.data_ptr()) : nullptr,
                            stride_batch_o,
                            stride_h_o,
                            stride_seq_o,
                            qo_len,
                            kv_len,
                            qo_per_kv_head,
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
