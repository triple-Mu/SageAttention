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

// Persistent variant of the SM100/SM110 warp-specialized (C1) tcgen05
// attention kernel (Phase B, BEYOND_CUDNN_PLAN.md section 4.4). Same 512
// threads / 16 warps / dual Q tile body as qk_int_sv_f8_cuda_sm100_ws.cu --
// warp roles, TMEM plan, register budgets, per-step math and the G1/G2
// numerics contract are all THAT file's (read its header first; per-block
// commentary is kept only where this file diverges). This is a separate TU
// (not a template flag) so the non-persistent kernel's device text stays
// byte-identical - SASS gate.
//
// What "persistent" changes (and nothing else):
//   * grid = min(total_tiles, #SMs); every warp role runs a grid-stride
//     loop over work items e (one work item = one 256-row Q tile x head x
//     batch, the old blockIdx). Decode: qblk2 = nq2-1 - e%nq2 (reversed so
//     causal's heavy tiles are issued first - static LPT), head = (e/nq2) %
//     num_qo_heads, batch = e/nq2/num_qo_heads. Consecutive e share the
//     head, so concurrent CTAs keep the old scheme's K/V L2 overlap.
//   * The per-CTA prologue (TMEM alloc handshake, 4 tensormap prefetches,
//     mbarrier inits, setmaxnreg, __syncthreads) runs ONCE; work items reuse
//     everything. No pipeline teardown between work items: every mbarrier
//     keeps completing/waiting with its phase variables carried across the
//     loop, and the KV ring keeps counting items globally (item0 running
//     base), so ring slot/phase derivations are unchanged expressions.
//   * Cross-work-item reuse is guarded by 3 new barriers + 2 new arrivals:
//     q_empty (mma commits after a work item's last QK issue; load waits it
//     before overwriting sQ), epi_empty[t] (epilogue warp arrives after the
//     bulk-store group drained; correction waits before restaging sO[t]),
//     and correction's epilog additionally arrives corr_empty[t] so the mma
//     warp can acquire "epilog read O_t and vec_t" before QK_t(0) of the
//     next work item (H14). softmax additionally consumes the final
//     vec_empty completion (previously dangling) before its next step-0 vec
//     store. Full extended ledger: barrier_ledger.md, persistent section.
//   * The one-shot per-(batch, kv-head) smem preloads do not survive a
//     persistent CTA crossing heads. sK_scale is DROPPED: softmax reads
//     k-scales via LDG (the pre-G2 transport; same f32 words, so per-tile
//     bits are unchanged - known cost: G2's L2 saving is given back, small
//     on the s1024-16384 target shapes; upgrade path: restage like
//     sV_scale, at the price of a softmax0<->softmax1 rendezvous).
//     sV_scale is RESTAGED per work item by the correction warpgroup (its
//     only reader) under a bar.sync sandwich - a per-element LDG in the
//     epilog loop was tried first and spilled the 88-reg region.
//   * Per work item the roles re-derive q/k/v scale pointers, trip counts
//     and TMA coords from the decoded (batch, head, qblk2); flash state
//     (row_max, denom) resets; O tiles reset through the existing
//     enable_D=0 first-PV mechanism. Per-tile compute order is untouched,
//     so outputs stay bitwise those of the non-persistent ws kernel
//     (golden double-track gate).
//
// Compile modes mirror qk_int_sv_f8_cuda_sm100_ws.cu:
//   default                   - kernel + torch launcher (extension build)
//   -DSAGE_SM100_DEVICE_ONLY  - kernel only, torch-free (ptxas probe TU)

#include <cfloat>
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

// TMA descriptor prefetch (cutedsl L432-436): warms the descriptor cache so
// the first UTMALDG/UTMASTG on each tensor map does not stall on a cold
// descriptor fetch. Pure prefetch - no data or ordering effect.
__device__ __forceinline__ void prefetch_tensormap(void const* desc)
{
    asm volatile("prefetch.tensormap [%0];" ::"l"(desc) : "memory");
}

// TMA bulk-tensor store (S2G), u32 smem source. Unlike tma.cuh's
// store_async_4D this takes the inner coordinate too: the O staging tile is
// stored as head_dim/64 boxes of 64 columns (the widest inner box a
// 128B-swizzled fp16/bf16 tensor map admits), so c0 is 0 or 64.
// Completion is tracked through the bulk async-group (commit/wait below),
// not an mbarrier.
__device__ __forceinline__ void
store_async_4D(void const* dst_tma_map, uint32_t src, int c0, int c1, int c2, int c3)
{
    asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
                 " [%0, {%2, %3, %4, %5}], [%1];\n" ::"l"(dst_tma_map),
                 "r"(src),
                 "r"(c0),
                 "r"(c1),
                 "r"(c2),
                 "r"(c3)
                 : "memory");
}

__device__ __forceinline__ void store_commit_group()
{
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

// Wait until at most N committed bulk store groups are still *reading* their
// smem source. The epilogue warp must drain to 0 before the CTA exits: smem
// is reclaimed at CTA exit, so an in-flight bulk read would race it (same
// reason CUTLASS epilogues end with tma_store_wait<0>). Global visibility of
// the stores themselves is covered by the kernel-completion fence.
template<int N>
__device__ __forceinline__ void store_wait_group_read()
{
    asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(N) : "memory");
}

// st.shared.v4 with a u32 shared-space address (16B unit of the sO staging
// tile; see the ws:: header comment for why u32 addresses).
__device__ __forceinline__ void sts_128b(uint32_t addr, uint4 v)
{
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(addr),
                 "r"(v.x),
                 "r"(v.y),
                 "r"(v.z),
                 "r"(v.w));
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
// One LDTM.x64 moves half the 128-col row; the softmax step issues the two
// of them back to back under one collective wait::ld (2 outstanding, r4 -
// the schedule ptxas already picked for r3's peeled instance and the r3
// stress runs covered; the A5 hang was 4+ outstanding). x64 is the widest
// feasible form here: the x128 variant is a single instruction whose
// destination block alone (128 regs + the address operand) exceeds the
// 128-register entry target that __launch_bounds__(512, 1) pins, and ptxas
// rejects it with C7602 "Insufficient registers" regardless of the
// setmaxnreg region budget (measured, nvcc 13.3).
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

// ---------------------------------------------------------------------------
// Packed f32x2 ALU (PTX ISA 8.6+). Each lane of mul/fma.rn.ftz.f32x2 is an
// independent IEEE fp32 op with round-to-nearest-even and ftz - exactly what
// the scalar mul.ftz.f32 / fma.rn.ftz.f32 this TU emits under
// --use_fast_math compute - so results are bit-identical per element to the
// scalar sequence they replace. (The explicit .rn also forbids mul->fma
// contraction, which never fires on these muls anyway: their consumers are
// max chains and fma multiplicands, not adds; FMUL.FTZ verified in SASS.)
// ptxas lowering, measured with nvcc 13.3: sm_100a -> one FMUL2.FTZ /
// FFMA2.FTZ with splat operands folded to a single broadcast register;
// sm_110a -> split back into two scalar FMUL/FFMA (no packed ALU there),
// correct either way. max.f32x2 does not exist (ptxas rejects it), so the
// row-max chain stays scalar.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void f32x2_mul(float& d0, float& d1, float a0, float a1, float b0, float b1)
{
#ifdef SAGE_TCGEN05_ENABLED
    asm("{\n"
        ".reg .b64 a, b, d;\n"
        "mov.b64 a, {%2, %3};\n"
        "mov.b64 b, {%4, %5};\n"
        "mul.rn.ftz.f32x2 d, a, b;\n"
        "mov.b64 {%0, %1}, d;\n"
        "}"
        : "=f"(d0), "=f"(d1)
        : "f"(a0), "f"(a1), "f"(b0), "f"(b1));
#else
    d0 = a0 * b0;
    d1 = a1 * b1;
#endif
}

__device__ __forceinline__ void
f32x2_fma(float& d0, float& d1, float a0, float a1, float b0, float b1, float c0, float c1)
{
#ifdef SAGE_TCGEN05_ENABLED
    asm("{\n"
        ".reg .b64 a, b, c, d;\n"
        "mov.b64 a, {%2, %3};\n"
        "mov.b64 b, {%4, %5};\n"
        "mov.b64 c, {%6, %7};\n"
        "fma.rn.ftz.f32x2 d, a, b, c;\n"
        "mov.b64 {%0, %1}, d;\n"
        "}"
        : "=f"(d0), "=f"(d1)
        : "f"(a0), "f"(a1), "f"(b0), "f"(b1), "f"(c0), "f"(c1));
#else
    d0 = fmaf(a0, b0, c0);
    d1 = fmaf(a1, b1, c1);
#endif
}

__device__ __forceinline__ void f32x2_add(float& d0, float& d1, float a0, float a1, float b0, float b1)
{
#ifdef SAGE_TCGEN05_ENABLED
    asm("{\n"
        ".reg .b64 a, b, d;\n"
        "mov.b64 a, {%2, %3};\n"
        "mov.b64 b, {%4, %5};\n"
        "add.rn.ftz.f32x2 d, a, b;\n"
        "mov.b64 {%0, %1}, d;\n"
        "}"
        : "=f"(d0), "=f"(d1)
        : "f"(a0), "f"(a1), "f"(b0), "f"(b1));
#else
    d0 = a0 + b0;
    d1 = a1 + b1;
#endif
}

// ---------------------------------------------------------------------------
// Balanced fmax reduction trees over the raw S row (G1). fmax over floats is
// exact (no rounding), associative and commutative, so the tree result equals
// the serial fold for any operand order; the raw row holds only exact
// int32->f32 values and the -inf mask sentinel (no NaN source), and FMNMX
// drops a NaN operand anyway. Compile-time recursion so ptxas sees one
// balanced expression tree (critical path log2(N) instead of N).
// ---------------------------------------------------------------------------

// Contiguous tree over N words holding f32 bits.
template<uint32_t N>
__device__ __forceinline__ float tree_fmax(const uint32_t* r)
{
    if constexpr (N == 1) {
        return __uint_as_float(r[0]);
    }
    else {
        return fmaxf(tree_fmax<N / 2>(r), tree_fmax<N / 2>(r + N / 2));
    }
}

// Tree over one per-thread k-scale class: class c owns columns
// {8k + 2c, 8k + 2c + 1}; the caller passes r offset by 2c and the template
// walks class-element indices [I0, I0+N) at column offset 8*(i/2) + i%2.
template<uint32_t I0, uint32_t N>
__device__ __forceinline__ float tree_fmax_cls(const uint32_t* r)
{
    if constexpr (N == 1) {
        return __uint_as_float(r[8 * (I0 / 2) + (I0 % 2)]);
    }
    else {
        return fmaxf(tree_fmax_cls<I0, N / 2>(r), tree_fmax_cls<I0 + N / 2, N / 2>(r));
    }
}

// Integer twins of the two trees, over the row's original int32 S values
// (wave24 = the wave22 vec_full delivery fix revived for d64 ONLY,
// C1_DESIGN.md section 14 / D64_DESIGN.md section 8.5): int32 -> f32
// conversion is exact and strictly monotone for |S| < 2^24, so one I2F of
// the integer tree max is bit-identical to the fmax tree over the converted
// row - and the row's 128 I2Fs leave the s_full -> vec_full window
// correction spins on (they run fused into the exp2 arguments instead,
// after the vec hand-off). d64 steady (non-peeled) steps only: the peeled
// step's -inf mask sentinel needs the f32 trees above, and d128 keeps the
// f32 trees on every step (the ungated change lost there, section 15.4).
template<uint32_t N>
__device__ __forceinline__ int32_t tree_imax(const uint32_t* r)
{
    if constexpr (N == 1) {
        return static_cast<int32_t>(r[0]);
    }
    else {
        return max(tree_imax<N / 2>(r), tree_imax<N / 2>(r + N / 2));
    }
}

template<uint32_t I0, uint32_t N>
__device__ __forceinline__ int32_t tree_imax_cls(const uint32_t* r)
{
    if constexpr (N == 1) {
        return static_cast<int32_t>(r[8 * (I0 / 2) + (I0 % 2)]);
    }
    else {
        return max(tree_imax_cls<I0, N / 2>(r), tree_imax_cls<I0 + N / 2, N / 2>(r));
    }
}

}  // namespace ws

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
//
// Divergence audit invariants (same discipline as the 128-thread kernel):
//   * every tcgen05::mma_* / tcgen05_commit sits under
//     `warp_idx == kMmaWarp && elect_one()` (single issuing thread);
//   * every tcgen05::tmem_ld/st/wait call is at least warp-uniform (the
//     .aligned contract): the softmax / correction loops iterate grid-uniform
//     trip counts, and the one data-dependent branch around such calls - the
//     correction rescale's ballot skip - is guarded by a warp vote;
//   * every cross-warp TMEM producer->consumer hand-off is bracketed by
//     tcgen05_fence_before_sync(); mbarrier arrive / wait;
//     tcgen05_fence_after_sync().

// Extra parameters vs the non-persistent kernel: the grid is 1-D (work
// stealing is static), so num_qo_heads can no longer be read off gridDim.y
// and the work-item count is passed explicitly.
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
    qk_int8_sv_f8_attn_kernel_sm100_ws_persist(const __grid_constant__ CUtensorMap tensorMapQ,
                                               const __grid_constant__ CUtensorMap tensorMapK,
                                               const __grid_constant__ CUtensorMap tensorMapV,
                                               const __grid_constant__ CUtensorMap tensorMapO,
                                               const float* __restrict__ Q_scale,
                                               const float* __restrict__ K_scale,
                                               const float* __restrict__ V_scale,
                                               float* __restrict__ Lse,
                                               const uint32_t qo_len,
                                               const uint32_t kv_len,
                                               const uint32_t qo_per_kv_head,
                                               const uint32_t num_qo_heads,
                                               const uint32_t total_tiles,
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
    constexpr uint32_t kEpiWarp  = 14;  // TMA-stores the sO staging tiles
    // warp 15 (empty) deallocs registers and exits.
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

    // --- smem plan: Q double tile + shared K/V 4-slot ring + O staging ---
    constexpr uint32_t SMEM_Q_BYTES  = CTA_Q * head_dim;  // per tile
    constexpr uint32_t SMEM_KV_BYTES = CTA_K * head_dim;  // per ring slot (K tile or V^T tile)
    constexpr uint32_t kKvStages     = 4;                 // cutedsl kv_stage=4 (8-bit inputs)
    // O staging (r5 lever B / M3): per tile, head_dim/64 boxes of
    // CTA_Q x 64 DTypeOut in the TMA 128B-swizzle layout, TMA-stored by the
    // epilogue warp. Used once per tile per CTA (no double-buffer pressure).
    constexpr uint32_t SMEM_O_BOX_BYTES = CTA_Q * 64 * sizeof(DTypeOut);  // 128B rows
    constexpr uint32_t SMEM_O_BYTES     = (head_dim / 64) * SMEM_O_BOX_BYTES;  // per tile
    // K-scale layout constants (indexing only). The persistent kernel has no
    // sK_scale smem preload: the row is per (batch, kv-head), which changes
    // per work item, so softmax reads scales straight from gmem (see the
    // file header for the G2 trade-off).
    constexpr uint32_t k_scale_advance_offset = (K_GRAN == QuantGranularity::kPerWarp) ? 1 : 4;
    constexpr uint32_t kNumKScales            = (K_GRAN == QuantGranularity::kPerThread) ? 4 : 1;
    static_assert(2 * SMEM_Q_BYTES + kKvStages * SMEM_KV_BYTES + 2 * SMEM_O_BYTES <= 227 * 1024,
                  "smem budget exceeded");

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

    // Dynamic smem layout: sQ (2 tiles) | KV ring (4 slots) | sO (2 staging
    // tiles). Each role derives u32 addresses from smem_ with these
    // compile-time offsets (no named pointers: a 64-bit generic base carried
    // across the setmaxnreg boundary spills, see the ws:: helper comment).
    extern __shared__ __align__(1024) int8_t smem_[];
    int8_t*                                  sQ = smem_;

    // --- barriers (roles/counts: bench/sm100_review/barrier_ledger.md).
    //     One array so every role addresses them as a single u32 base plus
    //     compile-time byte offsets - separate arrays kept several live base
    //     registers in the 32-reg mma/load region and spilled (measured). ---
    enum BarId : uint32_t {
        kBarQFull     = 0,   // +tile: TMA expect_tx, count 1, once per work item
        kBarKvFull    = 2,   // +slot: TMA expect_tx, count 1, 4-slot ring
        kBarKvEmpty   = 6,   // +slot: tcgen05.commit by mma warp, count 1
        kBarSFull     = 10,  // +tile: tcgen05.commit after QK chain
        kBarSEmpty    = 12,  // +tile: 128 softmax arrivals = S drained + P stored
        kBarVecFull   = 14,  // +tile: 128 softmax arrivals = vec stored
        kBarVecEmpty  = 16,  // +tile: 128 correction arrivals = vec read
        kBarCorrFull  = 18,  // +tile: tcgen05.commit after PV chain
        kBarCorrEmpty = 20,  // +tile: 128 correction arrivals = rescale stored,
                             //        or (persistent) epilog read O_t + vec_t
        kBarDealloc   = 22,  // 384 arrivals (softmax0/1 + correction)
        kBarEpiFull   = 23,  // +tile: 128 correction arrivals = O_t staged to sO
        // --- persistent-only barriers (barrier_ledger.md persistent section)
        kBarQEmpty    = 25,  // tcgen05.commit by mma after a work item's last QK
                             // issue; load waits before overwriting sQ
        kBarEpiEmpty  = 26,  // +tile: 1 arrival by the epilogue warp after the
                             // bulk-store group drained; correction waits
                             // before restaging sO[t]
        kNumBars      = 28,
    };
    __shared__ __align__(8) uint64_t bars[kNumBars];
    __shared__ __align__(4) uint32_t tmem_addr_slot;
    // v_scale stage, restaged PER WORK ITEM by the correction warpgroup (its
    // only reader) under a bar.sync sandwich - see the correction branch. A
    // straight per-element LDG in the epilog conversion loop was tried first
    // and spilled (~570B: ptxas hoists the long-latency loads and the whole
    // row's liveness lands on the 88-reg region's stack).
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
            tcgen05::init_barrier(&bars[kBarEpiFull + t], 128);
            tcgen05::init_barrier(&bars[kBarEpiEmpty + t], 1);
        }
#pragma unroll
        for (uint32_t s = 0; s < 4; s++) {
            tcgen05::init_barrier(&bars[kBarKvFull + s], 1);
            tcgen05::init_barrier(&bars[kBarKvEmpty + s], 1);
        }
        tcgen05::init_barrier(&bars[kBarDealloc], 3 * 128);
        tcgen05::init_barrier(&bars[kBarQEmpty], 1);
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
    //
    // Work-item schedule (identical expressions in every role; the roles never
    // rendezvous, so they must decode the same e -> tile mapping): grid-stride
    // e = blockIdx.x, +gridDim.x, ... < total_tiles; qblk2 runs fastest and
    // REVERSED (static LPT: under causal the trip count grows with qblk2, so
    // heavy tiles are issued first and stride-scattered across CTAs), then
    // head, then batch. Non-causal trips are uniform and the order is inert.
    // The decode reads kernel params only, so each setmaxnreg region
    // rematerializes it locally instead of carrying entry-region registers
    // across the boundary (the region-discipline note below); outputs a role
    // does not use are dead-code-eliminated after inlining.
    auto tile_decode = [&](uint32_t e, uint32_t& qblk2, uint32_t& head_id, uint32_t& batch_id) {
        const uint32_t nq2 = div_ceil(qo_len, 2 * CTA_Q);
        qblk2              = nq2 - 1 - (e % nq2);
        head_id            = (e / nq2) % num_qo_heads;
        batch_id           = e / nq2 / num_qo_heads;
    };

    // KV trip counts per tile (differentiated causal trips; per 128-row tile
    // this equals the 128-thread kernel's num_iterations with the q-block
    // index 2*qblk2 / 2*qblk2+1):
    auto trip_count = [&](uint32_t qblk2, uint32_t tile) {
        const uint32_t kblk = div_ceil(kv_len, CTA_K);
        if constexpr (mask_mode == MaskMode::kCausal) {
            return min(2 * qblk2 + 1 + tile, kblk);
        }
        return kblk;
    };

    // =========================================================================
    // softmax0 / softmax1 warpgroups
    // =========================================================================
    if (warp_group <= 1) {
        ws::warpgroup_reg_alloc<kNumRegsSoftmax>();

        const uint32_t tile     = warp_group;  // 0 or 1
        const uint32_t lane_row = threadIdx.x % 128;  // S row within the tile == TMEM lane

        const uint32_t bars_u32   = ws::smem_u32(bars);
        const uint32_t bar_full   = bars_u32 + (kBarSFull + tile) * 8;
        const uint32_t bar_empty  = bars_u32 + (kBarSEmpty + tile) * 8;
        const uint32_t bar_vfull  = bars_u32 + (kBarVecFull + tile) * 8;
        const uint32_t bar_vempty = bars_u32 + (kBarVecEmpty + tile) * 8;
        const uint32_t tmem_row   = tmem_addr_slot + ((warp_idx % 4) * 32 << 16) + (tile ? TMEM_COL_S1 : TMEM_COL_S0);

        float local_sm_scale = sm_scale * math::log2e;  // :237 (softmax runs in base 2)

        const uint32_t num_qblocks = div_ceil(qo_len, CTA_Q);

        // Pipeline phases are loop-carried: no barrier is ever re-initialized,
        // each pipe just keeps toggling across work items (persistent ledger).
        int s_full_phase    = 0;
        int vec_empty_phase = 0;

#pragma unroll 1
        for (uint32_t e = blockIdx.x; e < total_tiles; e += gridDim.x) {
        uint32_t qblk2, head_id, batch_id;
        tile_decode(e, qblk2, head_id, batch_id);
        const uint32_t trip = trip_count(qblk2, tile);

        const uint32_t q_idx = qblk2 * (2 * CTA_Q) + tile * CTA_Q + lane_row;

        // --- quant scale indexing (mirrors :249-284, q-block index from the
        //     decoded qblk2). The block index is clamped for the tail work
        //     item's fully-OOB tile 1 (rows are never stored; the clamp only
        //     prevents an OOB scale read). ---
        const uint32_t qblk = min(2 * qblk2 + tile, num_qblocks - 1);
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

        // Per-work-item k-scale row base (persistent: the hot path IS gmem;
        // no smem prefix exists because the row changes with the kv head).
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
        const float q_scale = Q_scale[q_scale_idx];

        // K-scale software prefetch (r5 lever A structure). The k-scale
        // word(s) of block iter+1 are loaded inside step iter's tcgen05.ld
        // shadow (below) - per-thread gmem broadcast LDG here (see the file
        // header for why the G2 smem copy does not survive persistence).
        // Same f32 words as the smem path, and the q_scale multiply stays at
        // the top of the consuming step, so the float op sequence - and the
        // per-tile output bits - are unchanged.
        float k_scale_pref[kNumKScales];
#pragma unroll
        for (uint32_t cls = 0; cls < kNumKScales; cls++) {
            k_scale_pref[cls] = K_scale_base_ptr[cls];  // block 0
        }

        // --- flash-attention state (mirrors :287-288), reset per work item ---
        float row_max = -5000000.0f;
        float denom   = 1.0f;

        // -------------------------------------------------------------------
        // Per-KV-block softmax step. One pass over the S row: two x64
        // tcgen05.ld issued back to back, then ONE collective wait::ld
        // (r4; C1_DESIGN.md 6.4). tcgen05.wait::ld has no per-operation
        // form - PTX ISA defines it as blocking on ALL prior tcgen05.ld of
        // the thread - so the r3 ld/wait/ld/wait chain cannot overlap the
        // two loads at PTX level; fusing the waits lets the second TMEM
        // round trip overlap the first (round 2's four chained x32 and
        // round 3's two chained x64 both kept long_scoreboard on top -
        // see C1_DESIGN.md 6.2/6.3, which also records why x128 is
        // rejected by ptxas; NB the overlap is pinned at PTX only, 6.4
        // records where ptxas re-sinks ld1 at SASS level). Two
        // outstanding tcgen05.ld is NOT the A5 pattern
        // (4+ outstanding, hang root cause unlocated): it is the exact
        // schedule ptxas already emitted for r3's peeled instance, which
        // the r3 stress runs (2x8000 launches) covered clean; this change
        // only pins that schedule at PTX level.
        //
        // G1 (C1_DESIGN.md section 9): the row stays in the RAW integer
        // domain - no dequanted row is materialized. The block row max is a
        // balanced max tree over the raw row (integer on d64 steady steps
        // since wave24, D64_DESIGN.md 8.5) scaled back per k-scale class
        // (bit-identical to the old serial dequant+fold, see below), and the
        // exp2 argument is ONE packed fma per pair from the raw word:
        //   p = exp2(raw * (local_sm_scale*dequant) - row_max)
        // where -row_max already carries the S_FP8_OFFSET fold. d_sum
        // accumulates in 4 independent packed f32x2 chains. This replaces
        // the 128-thread kernel's per-element value sequence (:420-474):
        // row_max/vec/o_scale stay bit-identical, but P moves by the
        // rounding-placement difference of the fused dequant and d_sum is
        // reassociated - the kernel is accuracy-gated, not golden-bitwise
        // gated, from G1 on. The whole row stays in registers, so nothing
        // touches TMEM between the vec store (aliases S cols [0,2)) and the
        // P store. is_last folds the causal/OOB mask like :431-444, in the
        // raw domain (-inf sentinel).
        // -------------------------------------------------------------------
        auto softmax_step = [&](auto is_last_t, uint32_t iter) {
            constexpr bool is_last = decltype(is_last_t)::value;
            // wave24 (D64_DESIGN.md section 8.5): the wave22 vec_full
            // delivery fix (int-domain row max + issue wall, C1_DESIGN.md
            // section 14) is revived for d64 ONLY - ungated it lost on the
            // d128 main battlefield (section 15.4) while carrying the whole
            // d64 gain of the wave23 fused tree (section 15.5). Every gated
            // branch below keeps the baseline text for d128, so d128
            // instantiations compile to byte-identical SASS.
            constexpr bool kVecFullD64 = (head_dim == 64);

            // K-scale(s) for this tile (mirrors :400-410; the raw word(s) were
            // prefetched into k_scale_pref during the previous step's ld shadow,
            // same value = same product bits)
            float dequant_scale[kNumKScales];
#pragma unroll
            for (uint32_t cls = 0; cls < kNumKScales; cls++) {
                dequant_scale[cls] = q_scale * k_scale_pref[cls];
            }

            ws::wait_bar(bar_full, s_full_phase);
            s_full_phase ^= 1;

            // ---- load + convert + mask, whole raw row retained. The row
            //      array is uint32_t so each f32 word overwrites its raw
            //      word in place: the ld's in-flight block and the retained
            //      row share registers. r4: both x64 lds issue back to back
            //      under one collective wait so their TMEM-load latencies
            //      can overlap (PTX-level structure; SASS reality per arch
            //      in C1_DESIGN.md 6.4). Masked lanes become -inf, which
            //      every consumer folds correctly: the fmax tree ignores it
            //      (unless the whole class is masked, absorbed by the -5e6
            //      floor below), and the exp2 argument becomes -inf -> p = 0
            //      exactly (c_raw > 0 by the FLT_MIN clamp below). ----
            uint32_t RS_row[CTA_K];
#pragma unroll
            for (uint32_t c = 0; c < CTA_K / 64; c++) {
                ws::tmem_ld_32x32b_x64(&RS_row[c * 64], tmem_row + c * 64);
            }
            // r5 lever A site: the next block's k-scale word(s) load inside
            // the tcgen05.ld shadow (after both LDTMs, before the collective
            // wait). Persistent kernel: straight gmem broadcast LDG (no smem
            // prefix; file header).
            if constexpr (!is_last) {
#pragma unroll
                for (uint32_t cls = 0; cls < kNumKScales; cls++) {
                    k_scale_pref[cls] = K_scale_base_ptr[(iter + 1) * k_scale_advance_offset + cls];
                }
            }
            tcgen05::tmem_ld_wait();
            // d64 steady steps skip this loop: the row STAYS int32 - its 128
            // I2Fs would otherwise sit inside the s_full -> vec_full window
            // that correction spins on (wave24, C1_DESIGN.md section 14 /
            // D64_DESIGN.md section 8.5); they run fused into the exp2
            // arguments below instead, after the vec hand-off. d128 and the
            // peeled step keep the baseline convert(+mask) here.
            if constexpr (!kVecFullD64 || is_last) {
#pragma unroll
                for (uint32_t j = 0; j < CTA_K; j++) {
                    float raw = __int2float_rz(static_cast<int32_t>(RS_row[j]));  // exact, |S| < 2^24
                    if constexpr (is_last) {
                        const uint32_t kv_idx = iter * CTA_K + j;
                        bool           oob;
                        if constexpr (mask_mode == MaskMode::kCausal) {
                            oob = (kv_idx > q_idx) || (kv_idx >= kv_len);
                        }
                        else {
                            oob = kv_idx >= kv_len;
                        }
                        if (oob) {
                            raw = __uint_as_float(0xff800000u);  // -inf
                        }
                    }
                    RS_row[j] = __float_as_uint(raw);
                }
            }

            // ---- block row max (G1): balanced fmax tree in the raw domain,
            //      one multiply per k-scale class back to the dequant domain.
            //      rnd(x*d) is monotone in x for d >= 0, so the class max of
            //      rounded products equals the rounded product of the class
            //      max: m_deq is bit-identical to the old serial fold over
            //      the dequanted row. The -5e6 floor reproduces the old
            //      m_local init (a masked/-5e6-sentinel lane could win the
            //      old fold the same way), and also absorbs the one NaN
            //      corner - d * (-inf) with a zero-amax d = 0 - because
            //      fmaxf drops a NaN operand. row_max / o_scale / the vec
            //      hand-off therefore stay bit-identical to the 128-thread
            //      kernel (:454-458). ----
            float m_deq;
            if constexpr (kVecFullD64 && !is_last) {
                // d64 steady steps: integer max tree, one exact I2F at each
                // tree root (ws::tree_imax note) - bit-identical to the f32
                // tree over the converted row, with the vec store's
                // dependency chain shrunk from 128 I2F + FMNMX tree to an
                // IMNMX tree + 1 I2F per root.
                if constexpr (K_GRAN == QuantGranularity::kPerThread) {
                    m_deq = dequant_scale[0] * __int2float_rz(ws::tree_imax_cls<0, 32>(&RS_row[0]));
#pragma unroll
                    for (uint32_t cls = 1; cls < kNumKScales; cls++) {
                        m_deq =
                            fmaxf(m_deq, dequant_scale[cls] * __int2float_rz(ws::tree_imax_cls<0, 32>(&RS_row[2 * cls])));
                    }
                }
                else {
                    m_deq = dequant_scale[0] * __int2float_rz(ws::tree_imax<CTA_K>(RS_row));
                }
            }
            else if constexpr (K_GRAN == QuantGranularity::kPerThread) {
                m_deq = dequant_scale[0] * ws::tree_fmax_cls<0, 32>(&RS_row[0]);
#pragma unroll
                for (uint32_t cls = 1; cls < kNumKScales; cls++) {
                    m_deq = fmaxf(m_deq, dequant_scale[cls] * ws::tree_fmax_cls<0, 32>(&RS_row[2 * cls]));
                }
            }
            else {
                m_deq = dequant_scale[0] * ws::tree_fmax<CTA_K>(RS_row);
            }
            m_deq = fmaxf(m_deq, -5000000.0f);

            // ---- online softmax update (expressions of :454-458). denom's
            //      o_scale rescale is folded into the d_sum accumulation
            //      below as an EXPLICIT fmaf: nvcc contracted the old
            //      `denom *= o_scale; ...; denom += d_sum` pair into one FFMA
            //      inside the straight-line step, and control flow inserted
            //      between the two statements (the reverted wave22 issue
            //      wall / moved vec_empty wait were both instances) splits
            //      the basic block, which killed the contraction and turned
            //      golden bit-exactness into 1-ulp denominator drift (wave23
            //      B200 golden, 132 diffs, all multi-block shapes). fmaf
            //      keeps the FFMA in every code shape. ----
            const float m_prev  = row_max;
            row_max             = max(row_max, fmaf(m_deq, local_sm_scale, -S_FP8_OFFSET));
            const float o_scale = math::ptx_exp2(m_prev - row_max);

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

            // ---- issue wall (d64 only; wave24, C1_DESIGN.md section 14 /
            //      D64_DESIGN.md section 8.5). Without it ptxas sinks the
            //      vec STTM/arrive into the exp2/pack tail (baseline SASS:
            //      STTM.x2 amid the F2FP burst, ~25 instructions before the
            //      P store), so the hand-off correction spins on was
            //      delivered a whole XU burst late - in-order issue makes
            //      textual order issue order, and a straight-line block
            //      lets the scheduler interleave freely. The wall is a
            //      control dependence: an opaque never-taken early-out
            //      (vec_gate is bar_vfull's shared-window address, nonzero
            //      by the bars layout; the volatile asm hides the value
            //      from NVVM, and ptxas cannot fold a runtime address),
            //      which neither compiler can delete, and no instruction
            //      below it can issue above the branch - or above the
            //      arrive before it (hoisting the branch would make the
            //      arrive conditional). Warp-uniform by construction
            //      (bar_vfull is warpgroup-uniform), so the tcgen05
            //      .aligned contract below holds. It splits the basic block
            //      between the o_scale rescale and the d_sum accumulation -
            //      bit-safe ONLY because denom's FFMA is an explicit fmaf
            //      (the wave23 contraction lesson, see the o_scale note
            //      above). Runtime cost: ISETP+BRA per step. ----
            if constexpr (kVecFullD64) {
                uint32_t vec_gate = bar_vfull;
                asm volatile("" : "+r"(vec_gate));
                if (vec_gate == 0) {
                    return;
                }
            }

            // ---- p = exp2(fma(raw, c_raw, -row_max)), d_sum, e4m3 pack from
            //      the retained raw row (G1 domain fold). With
            //      c_raw = local_sm_scale * dequant_scale the exp2 argument
            //      is ONE packed fma straight from the raw word - the old
            //      per-element dequant multiply is gone - and -row_max
            //      already carries the S_FP8_OFFSET fold (row_max =
            //      c*m - offset, so -row_max = offset - c*m: the same
            //      constant-domain form as cutedsl's LOG2_448 neg_off,
            //      L1111-1113). P moves by the rounding placement -
            //      rnd(c*d) then fma, vs rnd(raw*d) then fma - i.e. <= 1 ulp
            //      of the argument; quantified in C1_DESIGN.md section 9.
            //      The FLT_MIN clamp (peeled step only, where masked -inf
            //      lanes exist) keeps c_raw > 0 so -inf * c_raw stays -inf
            //      (a zero-amax block has dequant = 0, and -inf * 0 = NaN);
            //      for live lanes the clamped product |raw|*FLT_MIN <= 2e-32
            //      vanishes into the addend, and c_raw is only ever 0 when
            //      every product underflows anyway.
            //      d_sum: 4 independent packed f32x2 chains, quad w feeding
            //      chain pair (w & 1), then 3 packed folds + 1 scalar fold -
            //      add depth 16+3 instead of 128 (cutedsl L1160-1183 same
            //      structure); pure reassociation of the same IEEE adds.
            //      The pack is fused per 4 columns so each raw quad dies as
            //      its RP word is born - liveness only falls from here. ----
            float c_raw[kNumKScales];
#pragma unroll
            for (uint32_t cls = 0; cls < kNumKScales; cls++) {
                c_raw[cls] = local_sm_scale * dequant_scale[cls];
                if constexpr (is_last) {
                    c_raw[cls] = fmaxf(c_raw[cls], FLT_MIN);
                }
            }
            const float neg_row_max = -row_max;
            // Retained-row word -> f32 exp2 operand: d128 and the peeled
            // step hold f32 bits; d64 steady steps hold the raw int32 and
            // convert here - the same exact I2F the baseline convert loop
            // ran before the vec hand-off, just past it now (wave24).
            auto col_f32 = [&](uint32_t rw) {
                if constexpr (kVecFullD64 && !is_last) {
                    return __int2float_rz(static_cast<int32_t>(rw));
                }
                else {
                    return __uint_as_float(rw);
                }
            };
            float acc[8];  // 4 packed f32x2 d_sum accumulators
#pragma unroll
            for (uint32_t k = 0; k < 8; k++) {
                acc[k] = 0.0f;
            }
            uint32_t RP_u32[CTA_K / 4];
#pragma unroll
            for (uint32_t w = 0; w < CTA_K / 4; w++) {
                // quad w = columns 4w..4w+3; pairs share a k-scale class
                // ((j%8)/2 equal for j, j+1 with j even), classes {0,1} on
                // even w and {2,3} on odd w
                float c01, c23;
                if constexpr (K_GRAN == QuantGranularity::kPerThread) {
                    c01 = c_raw[((4 * w) % 8) / 2];
                    c23 = c_raw[((4 * w + 2) % 8) / 2];
                }
                else {
                    c01 = c_raw[0];
                    c23 = c_raw[0];
                }
                float a[4];
                ws::f32x2_fma(a[0],
                              a[1],
                              col_f32(RS_row[4 * w]),
                              col_f32(RS_row[4 * w + 1]),
                              c01,
                              c01,
                              neg_row_max,
                              neg_row_max);
                ws::f32x2_fma(a[2],
                              a[3],
                              col_f32(RS_row[4 * w + 2]),
                              col_f32(RS_row[4 * w + 3]),
                              c23,
                              c23,
                              neg_row_max,
                              neg_row_max);
                float p[4];
#pragma unroll
                for (uint32_t b = 0; b < 4; b++) {
                    p[b] = math::ptx_exp2(a[b]);
                }
                const uint32_t k0 = (w & 1) * 4;
                ws::f32x2_add(acc[k0], acc[k0 + 1], acc[k0], acc[k0 + 1], p[0], p[1]);
                ws::f32x2_add(acc[k0 + 2], acc[k0 + 3], acc[k0 + 2], acc[k0 + 3], p[2], p[3]);
                floatx4_to_e4m3x4(&RP_u32[w], &p[0], &p[2]);
            }
            ws::f32x2_add(acc[0], acc[1], acc[0], acc[1], acc[4], acc[5]);
            ws::f32x2_add(acc[2], acc[3], acc[2], acc[3], acc[6], acc[7]);
            ws::f32x2_add(acc[0], acc[1], acc[0], acc[1], acc[2], acc[3]);
            // denom = o_scale * denom + d_sum, one FFMA - the exact op nvcc
            // contracted the old  *= / +=  pair into (see the o_scale note).
            denom = fmaf(o_scale, denom, acc[0] + acc[1]);

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

        // ---- persistent: consume the final vec_empty completion (epilog read
        //      the (denom, row_max) pair) so the next work item's step-0 vec
        //      store has its slot. In the one-shot kernel this completion
        //      dangled harmlessly; here waiting it unconditionally keeps the
        //      vec pipe's completions and waits 1:1 (the last work item's
        //      wait is satisfied by its own epilog before the CTA exits). ----
        ws::wait_bar(bar_vempty, vec_empty_phase);
        vec_empty_phase ^= 1;
        }  // persistent work-item loop

        ws::arrive_bar(bars_u32 + kBarDealloc * 8);  // prior tcgen05 ops already fenced above
    }

    // =========================================================================
    // correction warpgroup (warps 8-11): O rescale per KV block, then the
    // final epilogue math staged to sO (TMA-stored by the epilogue warp, r5
    // lever B; M0-M2 stored register->global here directly).
    // =========================================================================
    else if (warp_group == 2) {
        ws::warpgroup_reg_dealloc<kNumRegsCorrection>();

        const uint32_t lane_row  = threadIdx.x % 128;  // O row within a tile == TMEM lane
        const uint32_t tmem_lane = tmem_addr_slot + ((warp_idx % 4) * 32 << 16);

        // per-tile pipeline state as scalars; the lambdas take everything by
        // reference so no local array is ever dynamically indexed (spills).
        // All phases are loop-carried across work items (persistent ledger).
        int vec0_full_phase = 0, vec1_full_phase = 0;
        int corr0_full_phase = 0, corr1_full_phase = 0;
        int epi0_empty_phase = 0, epi1_empty_phase = 0;

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
            // Ballot skip (cudnn all_alpha_one, prefill_d128_f16_sm100.py
            // :1719-1731): when every lane of this warp has o_scale == 1.0f
            // the whole TMEM read-multiply-write below is the identity and is
            // skipped; the barrier traffic (corr_full wait / corr_empty
            // arrive) is kept, so the pipeline is unchanged - corr_empty now
            // means "rescale stored or skipped as the identity". A block that
            // does not raise the running max hits this exactly: row_max =
            // max(m_prev, challenger) returns m_prev bit-for-bit, so
            // m_prev - rmax is +/-0 and ex2.approx(+/-0) = +1.0 by its
            // special-value table (ex2.approx results rounding to 1.0 for
            // tiny negative arguments also skip - the multiplier IS 1.0f).
            // Bit-exactness of the skip: x * 1.0f under mul.rn.ftz.f32x2
            // returns x for every value x this accumulator can hold. The one
            // FTZ divergence - a denormal x is flushed by the mul but kept by
            // the skip - cannot arise: e4m3 values are multiples of 2^-9, so
            // every PV product is a multiple of 2^-18, every f32 rounding of
            // a partial sum stays on that grid (|x| < 2^6 is exact, above it
            // ulp >= 2^-17), the rescale mul itself is FTZ (never stores a
            // denormal), and adding a non-denormal to a 2^-18-grid value
            // cannot land in (0, 2^-126) - so O never holds a denormal.
            // Inf/NaN cannot arise either (|O| <= trips * 128 * 448^2 << f32
            // max). The branch is warp-uniform (vote), which satisfies the
            // .aligned contract of the tcgen05.ld/st inside.
            const bool all_one = __all_sync(0xffffffffu, o_scale == 1.0f);
            ws::wait_bar(bar_cfull, cphase);
            cphase ^= 1;
            if (!all_one) {
#pragma unroll
                for (uint32_t c = 0; c < num_tiles_o; c++) {
                    uint32_t RO_u32[32];
                    tcgen05::tmem_ld_32x32b_x32(RO_u32, tmem_o_row + c * 32);
                    tcgen05::tmem_ld_wait();
                    // G5 (cutedsl L1237-1239 same): adjacent columns share the
                    // splat o_scale, so the rescale packs as mul.rn.ftz.f32x2 -
                    // per lane bit-identical to the scalar mul it replaces (see
                    // the f32x2 helper comment), halving the FMUL issue count.
#pragma unroll
                    for (uint32_t jj = 0; jj < 32; jj += 2) {
                        float lo, hi;
                        ws::f32x2_mul(
                            lo, hi, __uint_as_float(RO_u32[jj]), __uint_as_float(RO_u32[jj + 1]), o_scale, o_scale);
                        RO_u32[jj]     = __float_as_uint(lo);
                        RO_u32[jj + 1] = __float_as_uint(hi);
                    }
                    tcgen05::tmem_st_32x32b_x32(tmem_o_row + c * 32, RO_u32);
                }
                tcgen05::tmem_st_wait();
            }
            tcgen05::tcgen05_fence_before_sync();
            ws::arrive_bar(bar_cempty);
        };

        // final O_t = O_t * d_rcp (* v_scale) -> sO staging tile in the TMA
        // 128B-swizzle layout; the epilogue warp TMA-stores it (r5 lever B).
        // Value sequence per element (mul d_rcp, mul v_scale, cvt rn) is that
        // of :572-599 - only the destination changed, and the TMA store moves
        // the converted bytes verbatim, so the output bits are unchanged. The
        // old per-row q_idx < qo_len guard is now the tensor map's bounding
        // box (dim1 = qo_len): OOB rows of a box are clipped by the TMA store,
        // never written - same observable effect.
        // Persistent deltas: (a) sO[t] is reused every work item, so before
        // restaging the lambda waits epi_empty[t] = the epilogue warp drained
        // the previous work item's bulk-store group (reuse=false skips it on
        // the virgin first work item); (b) after the O/vec TMEM reads it ALSO
        // arrives corr_empty[t]: that completion is the mma warp's license to
        // overwrite O_t (PV(0) enable_D=0) and the vec cols (QK(0) over
        // S[0,2)) in the NEXT work item (ledger H14).
        auto epilog = [&](uint32_t bar_vfull,
                          uint32_t bar_vempty,
                          int&     vphase,
                          uint32_t vec_addr,
                          uint32_t bar_cfull,
                          int&     cphase,
                          uint32_t bar_cempty,
                          uint32_t tmem_o_row,
                          uint32_t sO_tile,
                          uint32_t bar_efull,
                          uint32_t bar_eempty,
                          int&     ephase,
                          bool     reuse) {
            if (reuse) {
                ws::wait_bar(bar_eempty, ephase);
                ephase ^= 1;
            }
            float denom_t, rmax_unused;
            read_vec(bar_vfull, bar_vempty, vphase, vec_addr, denom_t, rmax_unused);
            ws::wait_bar(bar_cfull, cphase);
            cphase ^= 1;

            const float d_rcp = math::ptx_rcp(denom_t);
            // 128B-swizzle addressing: the 16B unit u of row r lives at
            // r*128 + (u ^ (r%8))*16 within its 64-column box (the standard
            // swizzle atom the tensor map's CU_TENSOR_MAP_SWIZZLE_128B
            // expects; conflict-free STS.128 across the warp's 32 rows).
            const uint32_t swz_phase = lane_row & 7;
            const uint32_t row_base  = sO_tile + lane_row * 128;

#pragma unroll
            for (uint32_t c = 0; c < num_tiles_o; c++) {
                uint32_t RO_u32[32];
                tcgen05::tmem_ld_32x32b_x32(RO_u32, tmem_o_row + c * 32);
                tcgen05::tmem_ld_wait();

                // 32 columns -> 4 16B units, each 4 half2/bf162 words
#pragma unroll
                for (uint32_t u = 0; u < 4; u++) {
                    uint4 unit;
                    uint32_t* w = reinterpret_cast<uint32_t*>(&unit);
#pragma unroll
                    for (uint32_t p = 0; p < 4; p++) {
                        const uint32_t jj = u * 4 + p;  // half2 index within the 32-col chunk
                        float          lo = __uint_as_float(RO_u32[2 * jj]) * d_rcp;
                        float          hi = __uint_as_float(RO_u32[2 * jj + 1]) * d_rcp;
                        if constexpr (fuse_v_scale) {
                            lo *= sV_scale[c * 32 + 2 * jj];
                            hi *= sV_scale[c * 32 + 2 * jj + 1];
                        }
                        const float2 o2 = make_float2(lo, hi);
                        if constexpr (std::is_same<DTypeOut, half>::value) {
                            const half2 h2 = __float22half2_rn(o2);
                            w[p]           = *reinterpret_cast<const uint32_t*>(&h2);
                        }
                        else {
                            const nv_bfloat162 b2 = __float22bfloat162_rn(o2);
                            w[p]                  = *reinterpret_cast<const uint32_t*>(&b2);
                        }
                    }
                    // box = 64-col half of the tile; unit index within the box row
                    const uint32_t box   = c / 2;
                    const uint32_t u_box = (c % 2) * 4 + u;
                    ws::sts_128b(row_base + box * SMEM_O_BOX_BYTES + ((u_box ^ swz_phase) << 4), unit);
                }
            }
            // O_t and vec_t are fully read (every tmem_ld above waited):
            // release them to the mma warp for the next work item (H14).
            tcgen05::tcgen05_fence_before_sync();
            ws::arrive_bar(bar_cempty);
            // generic-proxy sO writes -> async-proxy bulk-store visibility
            // (SS-twin lesson), then hand the tile to the epilogue warp.
            tcgen05::fence_async_shared();
            ws::arrive_bar(bar_efull);
        };

        const uint32_t vec0_addr   = tmem_lane + TMEM_COL_S0 + TMEM_COL_VEC;
        const uint32_t vec1_addr   = tmem_lane + TMEM_COL_S1 + TMEM_COL_VEC;
        const uint32_t tmem_o0_row = tmem_lane + TMEM_COL_O0;
        const uint32_t tmem_o1_row = tmem_lane + TMEM_COL_O1;
        const uint32_t sO_u32      = ws::smem_u32(smem_) + 2 * SMEM_Q_BYTES + kKvStages * SMEM_KV_BYTES;

        const uint32_t bars_u32 = ws::smem_u32(bars);
        const uint32_t vf0 = bars_u32 + kBarVecFull * 8, ve0 = bars_u32 + kBarVecEmpty * 8;
        const uint32_t vf1 = vf0 + 8, ve1 = ve0 + 8;
        const uint32_t cf0 = bars_u32 + kBarCorrFull * 8, ce0 = bars_u32 + kBarCorrEmpty * 8;
        const uint32_t cf1 = cf0 + 8, ce1 = ce0 + 8;
        const uint32_t ef0 = bars_u32 + kBarEpiFull * 8, ef1 = bars_u32 + (kBarEpiFull + 1) * 8;
        const uint32_t ee0 = bars_u32 + kBarEpiEmpty * 8, ee1 = bars_u32 + (kBarEpiEmpty + 1) * 8;

#pragma unroll 1
        for (uint32_t e = blockIdx.x; e < total_tiles; e += gridDim.x) {
        uint32_t qblk2, head_id, batch_id;
        tile_decode(e, qblk2, head_id, batch_id);
        const uint32_t trip0 = trip_count(qblk2, 0);
        const uint32_t trip1 = trip_count(qblk2, 1);
        const bool     reuse = e != blockIdx.x;  // sO[t] holds the previous work item's stage

        // Restage the (batch, kv-head) v_scale row for this work item.
        // Correction is sV_scale's ONLY reader, so a named-barrier sandwich
        // inside the warpgroup is a complete ordering argument: bar.sync #1 -
        // every corr thread arrived, i.e. its previous work item's epilog
        // reads of the old row are program-order complete; write; bar.sync #2
        // - the new row is visible to all 128 threads before any epilog read.
        // Plain CTA-scope smem + bar.sync: no mbarrier, no ledger impact.
        // (barrier 0 is the kernel-entry __syncthreads; use id 1.)
        if constexpr (fuse_v_scale) {
            asm volatile("bar.sync 1, 128;" ::: "memory");
            if (lane_row < head_dim) {
                const float* V_scale_base_ptr =
                    V_scale
                    + (static_cast<int64_t>(batch_id) * (num_qo_heads / qo_per_kv_head) + head_id / qo_per_kv_head)
                          * head_dim;
                sV_scale[lane_row] = V_scale_base_ptr[lane_row];
            }
            asm volatile("bar.sync 1, 128;" ::: "memory");
        }

        discard_vec(vf0, ve0, vec0_full_phase);
        discard_vec(vf1, ve1, vec1_full_phase);
        // unroll 1: with the packed rescale body (G5) the frontend otherwise
        // unrolls these runtime-trip loops ~4x (+45% static instructions for
        // zero dynamic issues saved - the trip count is data-dependent)
#pragma unroll 1
        for (uint32_t j = 1; j < trip0; j++) {
            rescale(vf0, ve0, vec0_full_phase, vec0_addr, cf0, ce0, corr0_full_phase, tmem_o0_row);
            rescale(vf1, ve1, vec1_full_phase, vec1_addr, cf1, ce1, corr1_full_phase, tmem_o1_row);
        }
#pragma unroll 1
        for (uint32_t j = trip0; j < trip1; j++) {  // causal S1-only rounds
            rescale(vf1, ve1, vec1_full_phase, vec1_addr, cf1, ce1, corr1_full_phase, tmem_o1_row);
        }
        epilog(vf0, ve0, vec0_full_phase, vec0_addr, cf0, corr0_full_phase, ce0,
               tmem_o0_row, sO_u32, ef0, ee0, epi0_empty_phase, reuse);
        epilog(vf1, ve1, vec1_full_phase, vec1_addr, cf1, corr1_full_phase, ce1,
               tmem_o1_row, sO_u32 + SMEM_O_BYTES, ef1, ee1, epi1_empty_phase, reuse);
        }  // persistent work-item loop

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
                // one live TMEM base; per-call +CONST rematerializes freely
                const uint32_t tb = tmem_addr_slot;

                // pipeline state as scalars passed by reference: dynamically
                // indexed local phase arrays would spill (32-reg budget).
                // All loop-carried across work items, plus the running KV
                // ring item base (the ring never resets: slot/phase stay the
                // same expressions of the ABSOLUTE item) and the Q-tile
                // phase (q_full completes once per tile per work item).
                int s0_empty_phase = 0, s1_empty_phase = 0;
                int corr0_empty_phase = 0, corr1_empty_phase = 0;
                int      q_full_phase = 0;
                uint32_t item0        = 0;

                // one u32 base for every barrier and smem operand address
                // (see the ws:: helper comment; the ring sits at sQ + 2 Q tiles)
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

#pragma unroll 1
                for (uint32_t e = blockIdx.x; e < total_tiles; e += gridDim.x) {
                uint32_t qblk2, head_unused, batch_unused;
                tile_decode(e, qblk2, head_unused, batch_unused);
                // Fake data dependence: route kv_len through qblk2 so the
                // kblk math is emitted AFTER the decode's udiv sequence.
                // Without it kblk is computed at the loop head and has to
                // survive the udiv's register peak - ptxas parks it on the
                // stack (measured: this was the last 4B spill at d64).
                uint32_t kv_dep = kv_len;
                asm("" : "+r"(kv_dep) : "r"(qblk2));
                const uint32_t kblk = div_ceil(kv_dep, CTA_K);
                uint32_t       trip0, trip1;
                if constexpr (mask_mode == MaskMode::kCausal) {
                    trip0 = min(2 * qblk2 + 1, kblk);
                    trip1 = min(2 * qblk2 + 2, kblk);
                }
                else {
                    trip0 = kblk;
                    trip1 = kblk;
                }

                // ---- prologue: QK00, QK10, PV00 (cutedsl L621-658). From
                //      the second work item on, QK_t(0) must first acquire
                //      corr_empty[t] = "correction's epilog read O_t and
                //      vec_t" (H14): QK_t(0) overwrites the vec cols of S_t
                //      and PV_t(0) (enable_D=0, later in program order)
                //      overwrites O_t. One completion covers both. ----
                ws::wait_bar(bars_u32 + kBarQFull * 8, q_full_phase);
                if (e != blockIdx.x) {
                    wait_corr_empty(bars_u32 + kBarCorrEmpty * 8, corr0_empty_phase);
                }
                wait_kv(item0);  // K_0
                qk(tb + TMEM_COL_S0, sQ_u32, kv_slot_u32(item0), bars_u32 + kBarSFull * 8);
                ws::wait_bar(bars_u32 + (kBarQFull + 1) * 8, q_full_phase);
                q_full_phase ^= 1;
                if (e != blockIdx.x) {
                    wait_corr_empty(bars_u32 + (kBarCorrEmpty + 1) * 8, corr1_empty_phase);
                }
                qk(tb + TMEM_COL_S1, sQ_u32 + SMEM_Q_BYTES, kv_slot_u32(item0), bars_u32 + (kBarSFull + 1) * 8);
                release_kv(item0);   // K_0 (fires when QK10 retired)
                wait_kv(item0 + 1);  // V_0
                pv(tb + TMEM_COL_O0,
                   tb + TMEM_COL_S0 + TMEM_COL_P,
                   bars_u32 + kBarSEmpty * 8,
                   s0_empty_phase,
                   bars_u32 + kBarCorrFull * 8,
                   kv_slot_u32(item0 + 1),
                   /*accumulate=*/false);

                // ---- steady loop (cutedsl L661-704): per i, in issue order
                //      QK0(i) | PV1(i-1) | QK1(i) | PV0(i). pv1_started resets
                //      per work item: PV1(0)'s corr acquire was already done
                //      by the QK1(0)-side wait above (or is vacuous on the
                //      first work item). ----
                bool     pv1_started = false;
                uint32_t v_item      = item0 + 1;  // ring item of the V the next PV1 consumes
                // unroll 1 (persistent): inside the work-item loop the frontend
                // otherwise unrolls these runtime-trip issue loops and the extra
                // per-copy state tips the 40-reg region into spills (d64)
#pragma unroll 1
                for (uint32_t i = 1; i < trip0; i++) {
                    wait_kv(item0 + 2 * i);  // K_i
                    qk(tb + TMEM_COL_S0, sQ_u32, kv_slot_u32(item0 + 2 * i), bars_u32 + kBarSFull * 8);

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

                    qk(tb + TMEM_COL_S1,
                       sQ_u32 + SMEM_Q_BYTES,
                       kv_slot_u32(item0 + 2 * i),
                       bars_u32 + (kBarSFull + 1) * 8);
                    release_kv(item0 + 2 * i);  // K_i

                    wait_kv(item0 + 2 * i + 1);  // V_i
                    v_item = item0 + 2 * i + 1;
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
#pragma unroll 1
                for (uint32_t i = trip0; i < trip1; i++) {
                    wait_kv(item0 + 2 * i);  // K_i
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
                    qk(tb + TMEM_COL_S1,
                       sQ_u32 + SMEM_Q_BYTES,
                       kv_slot_u32(item0 + 2 * i),
                       bars_u32 + (kBarSFull + 1) * 8);
                    release_kv(item0 + 2 * i);
                    wait_kv(item0 + 2 * i + 1);  // V_i
                    v_item = item0 + 2 * i + 1;
                }

                // ---- every QK of this work item is issued: release sQ to
                //      the load warp for the next work item's Q TMA. The
                //      commit completes once all prior MMAs retired, which
                //      includes both tiles' last QK (in-order UMMA queue). ----
                ws::commit_bar(bars_u32 + kBarQEmpty * 8);

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

                item0 += 2 * trip1;  // ring items consumed by this work item
                }  // persistent work-item loop
            }

            // --- TMEM dealloc: whole warp waits for the 12 TMEM-reading warps
            //     (their arrivals are fenced), then deallocs collectively ---
            ws::wait_bar(ws::smem_u32(&bars[kBarDealloc]), 0);
            tcgen05::tcgen05_fence_after_sync();
            tcgen05::tmem_dealloc(tmem_addr_slot, TMEM_COLS_TOTAL);
        }
        else if (warp_idx == kLoadWarp) {
            if (tcgen05::elect_one()) {
                // issue order Q0, K0, Q1, V0, (K,V)* mirrors cutedsl L589-611;
                // TMA coords/boxes are those of the 128-thread kernel
                // (:314-321, :393-396, :550-554). Ring item n -> slot n%4;
                // n is ABSOLUTE across work items (item0 running base, same
                // counting as the mma warp), so the full/empty phase
                // derivations are untouched: item n waits the empty
                // completion #(n/4 - 1) of its slot -> phase (n/4-1)&1.
                const uint32_t bars_u32      = ws::smem_u32(bars);
                const uint32_t sQ_u32        = ws::smem_u32(sQ);
                const uint32_t ring_u32      = sQ_u32 + 2 * SMEM_Q_BYTES;
                auto           kv_full_bar   = [&](uint32_t item) { return bars_u32 + (kBarKvFull + (item & 3)) * 8; };
                auto           kv_empty_wait = [&](uint32_t item) {
                    if (item >= 4) {
                        ws::wait_bar(bars_u32 + (kBarKvEmpty + (item & 3)) * 8, ((item >> 2) - 1) & 1);
                    }
                };

                // G4: prefetch the load-side descriptors before the first use
                ws::prefetch_tensormap(&tensorMapQ);
                ws::prefetch_tensormap(&tensorMapK);
                ws::prefetch_tensormap(&tensorMapV);

                int      q_empty_phase = 0;
                uint32_t item0         = 0;
#pragma unroll 1
                for (uint32_t e = blockIdx.x; e < total_tiles; e += gridDim.x) {
                uint32_t qblk2, head_id, batch_id;
                tile_decode(e, qblk2, head_id, batch_id);
                const uint32_t trip1      = trip_count(qblk2, 1);
                const uint32_t kv_head_id = head_id / qo_per_kv_head;

                // sQ reuse acquire: the mma warp's q_empty commit certifies
                // every QK of the previous work item retired (its last
                // reader). Work item 0's sQ is virgin - no wait.
                if (e != blockIdx.x) {
                    ws::wait_bar(bars_u32 + kBarQEmpty * 8, q_empty_phase);
                    q_empty_phase ^= 1;
                }

                ws::expect_bytes_bar<SMEM_Q_BYTES>(bars_u32 + kBarQFull * 8);
                ws::load_async_4D(
                    sQ_u32, &tensorMapQ, bars_u32 + kBarQFull * 8, 0, qblk2 * (2 * CTA_Q), head_id, batch_id);
                kv_empty_wait(item0);  // ring reuse from lap 1 on (absolute item)
                ws::expect_bytes_bar<SMEM_KV_BYTES>(kv_full_bar(item0));
                ws::load_async_4D(
                    ring_u32 + (item0 & 3) * SMEM_KV_BYTES, &tensorMapK, kv_full_bar(item0), 0, 0, kv_head_id, batch_id);
                ws::expect_bytes_bar<SMEM_Q_BYTES>(bars_u32 + (kBarQFull + 1) * 8);
                ws::load_async_4D(sQ_u32 + SMEM_Q_BYTES,
                                  &tensorMapQ,
                                  bars_u32 + (kBarQFull + 1) * 8,
                                  0,
                                  qblk2 * (2 * CTA_Q) + CTA_Q,
                                  head_id,
                                  batch_id);
                kv_empty_wait(item0 + 1);
                ws::expect_bytes_bar<SMEM_KV_BYTES>(kv_full_bar(item0 + 1));
                ws::load_async_4D(ring_u32 + ((item0 + 1) & 3) * SMEM_KV_BYTES,
                                  &tensorMapV,
                                  kv_full_bar(item0 + 1),
                                  0,
                                  0,
                                  kv_head_id,
                                  batch_id);

                for (uint32_t i = 1; i < trip1; i++) {
                    const uint32_t k_item = item0 + 2 * i;
                    kv_empty_wait(k_item);
                    ws::expect_bytes_bar<SMEM_KV_BYTES>(kv_full_bar(k_item));
                    ws::load_async_4D(ring_u32 + (k_item & 3) * SMEM_KV_BYTES,
                                      &tensorMapK,
                                      kv_full_bar(k_item),
                                      0,
                                      i * CTA_K,
                                      kv_head_id,
                                      batch_id);

                    const uint32_t v_item = item0 + 2 * i + 1;
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
                item0 += 2 * trip1;
                }  // persistent work-item loop
            }
        }
        else if (warp_idx == kEpiWarp) {
            // Epilogue warp (r5 lever B): one TMA bulk store per staged O box.
            // The producers (correction) issued fence.proxy.async before their
            // arrivals, so the bulk-store engine sees the staged bytes.
            // Persistent: per work item, drain the bulk-store group with
            // wait_group.read 0 and only then arrive epi_empty[t] - that
            // release/acquire chain (correction waits epi_empty before its
            // next STS) is what makes the sO reuse WAR-safe; the last work
            // item's drain doubles as the CTA-exit smem-lifetime wait (H13).
            if (tcgen05::elect_one()) {
                ws::prefetch_tensormap(&tensorMapO);  // G4: cold-fetch off the store path
                const uint32_t bars_u32 = ws::smem_u32(bars);
                const uint32_t sO_u32   = ws::smem_u32(smem_) + 2 * SMEM_Q_BYTES + kKvStages * SMEM_KV_BYTES;

                int epi_full_phase = 0;
#pragma unroll 1
                for (uint32_t e = blockIdx.x; e < total_tiles; e += gridDim.x) {
                    uint32_t qblk2, head_id, batch_id;
                    tile_decode(e, qblk2, head_id, batch_id);
                    const uint32_t q_base = qblk2 * (2 * CTA_Q);
#pragma unroll
                    for (uint32_t t = 0; t < 2; t++) {
                        ws::wait_bar(bars_u32 + (kBarEpiFull + t) * 8, epi_full_phase);
#pragma unroll
                        for (uint32_t b = 0; b < head_dim / 64; b++) {
                            ws::store_async_4D(&tensorMapO,
                                               sO_u32 + t * SMEM_O_BYTES + b * SMEM_O_BOX_BYTES,
                                               b * 64,
                                               q_base + t * CTA_Q,
                                               head_id,
                                               batch_id);
                        }
                        ws::store_commit_group();
                    }
                    epi_full_phase ^= 1;
                    ws::store_wait_group_read<0>();
                    ws::arrive_bar(bars_u32 + kBarEpiEmpty * 8);
                    ws::arrive_bar(bars_u32 + (kBarEpiEmpty + 1) * 8);
                }
            }
        }
        // warp 15 (empty): nothing to do, exit
    }
}

#ifndef SAGE_SM100_DEVICE_ONLY

// ---------------------------------------------------------------------------
// Host launcher: clone of the ws launcher with a 1-D persistent grid
// (min(total work items, #SMs); __launch_bounds__(512, 1) pins 1 CTA/SM) and
// the two extra kernel scalars. Selected in qk_int_sv_f8_cuda_sm100.cu when
// the ws path is active AND SAGEATTN_SM100_WS_PERSIST is truthy (default
// off; auto integration is deferred until B200 acceptance).
// ---------------------------------------------------------------------------

torch::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_ws_persist(torch::Tensor query,
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

                        // O tensor map for the TMA-store epilogue: 64-column
                        // boxes (128B rows = the widest inner box a
                        // 128B-swizzled 2-byte tensor map admits), dim1 bound
                        // qo_len so the tail CTA's OOB rows are clipped by the
                        // store itself.
                        CUtensorMap tma_map_o =
                            create_tensor_map_4D<CTA_Q, 64>(reinterpret_cast<DTypeOut*>(output.data_ptr()),
                                                            batch_size,
                                                            num_qo_heads,
                                                            qo_len,
                                                            HEAD_DIM,
                                                            stride_batch_o,
                                                            stride_h_o,
                                                            stride_seq_o);

                        auto* kernel =
                            qk_int8_sv_f8_attn_kernel_sm100_ws_persist<CTA_Q,
                                                                       CTA_K,
                                                                       NUM_THREADS,
                                                                       HEAD_DIM,
                                                                       static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                                       static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                                       DTypeOut,
                                                                       mask_mode,
                                                                       RETURN_LSE,
                                                                       true>;
                        // 2 Q tiles + 4-slot KV ring + 2 O staging tiles
                        // (barriers are static smem)
                        size_t smem_bytes = 2 * CTA_Q * HEAD_DIM * sizeof(int8_t)
                                            + 4 * CTA_K * HEAD_DIM * sizeof(int8_t)
                                            + 2 * CTA_Q * HEAD_DIM * sizeof(DTypeOut);
                        sage::set_max_dynamic_smem_once(kernel, smem_bytes, query.get_device());

                        // persistent 1-D grid: one CTA per SM (or fewer when
                        // the work does not fill a wave)
                        const uint32_t total_tiles = div_ceil(qo_len, 2 * CTA_Q) * num_qo_heads * batch_size;
                        const uint32_t num_sms =
                            static_cast<uint32_t>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
                        dim3 grid(total_tiles < num_sms ? total_tiles : num_sms);
                        kernel<<<grid, NUM_THREADS, smem_bytes, stream>>>(
                            tma_maps.q,
                            tma_maps.k,
                            tma_maps.v,
                            tma_map_o,
                            reinterpret_cast<float*>(query_scale.data_ptr()),
                            reinterpret_cast<float*>(key_scale.data_ptr()),
                            reinterpret_cast<float*>(value_scale.data_ptr()),
                            (RETURN_LSE) ? reinterpret_cast<float*>(lse.data_ptr()) : nullptr,
                            qo_len,
                            kv_len,
                            qo_per_kv_head,
                            num_qo_heads,
                            total_tiles,
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
