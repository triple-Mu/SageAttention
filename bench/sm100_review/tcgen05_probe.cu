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

// ptxas probe TU for tcgen05.cuh: a dummy kernel that exercises EVERY wrapper
// so every asm string is machine-checked by ptxas. Torch-free by design; not
// meant to produce meaningful results, only to compile.
//
//   positive gates: -cubin for sm_100a and sm_110a must be clean
//   negative gates: -cubin for sm_90a / sm_120a must compile via the trap()
//                   stubs with zero tcgen05 instructions in SASS

#include <cuda.h>

#include "tcgen05.cuh"

namespace {

constexpr uint32_t kCtaQ = 128;
constexpr uint32_t kCtaK = 128;
constexpr uint32_t kHeadDim = 128;

// TMEM column plan from the sm100 design: S 0-127, P 128-159, O 160-287.
constexpr uint32_t kTmemCols = 512;
constexpr uint32_t kTmemColS = 0;
constexpr uint32_t kTmemColP = 128;
constexpr uint32_t kTmemColO = 160;

}  // namespace

__global__ void __launch_bounds__(128) tcgen05_probe_kernel(
    const __grid_constant__ CUtensorMap tensorMapQ,
    const __grid_constant__ CUtensorMap tensorMapK, uint32_t* out, uint32_t enable_d) {
  __shared__ alignas(1024) uint8_t smem_Q[kCtaQ * kHeadDim];
  __shared__ alignas(1024) uint8_t smem_K[kCtaK * kHeadDim];
  __shared__ alignas(8) uint64_t bar_Q;
  __shared__ alignas(8) uint64_t bar_K;
  __shared__ alignas(8) uint64_t bar_S_done;
  __shared__ alignas(16) uint32_t tmem_addr_slot;

  // --- mbarrier + TMA helpers (shared with the sm90 kernel, sm_90+) ---
  if (threadIdx.x == 0) {
    tcgen05::init_barrier(&bar_Q, 1);
    tcgen05::init_barrier(&bar_K, 1);
    tcgen05::init_barrier(&bar_S_done, 1);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    tcgen05::expect_bytes<kCtaQ * kHeadDim>(&bar_Q);
    tcgen05::load_async_4D(smem_Q, &tensorMapQ, &bar_Q, 0, 0, 0, 0);
    tcgen05::expect_bytes<kCtaK * kHeadDim>(&bar_K);
    tcgen05::load_async_4D(smem_K, &tensorMapK, &bar_K, 0, 0, 0, 0);
  }
  tcgen05::wait(&bar_Q, 0);
  tcgen05::wait(&bar_K, 0);

  // --- TMEM allocation protocol (alloc -> relinquish at start) ---
  if (threadIdx.x < 32) {
    tcgen05::tmem_alloc(&tmem_addr_slot, kTmemCols);
    tcgen05::tmem_relinquish();
  }
  __syncthreads();
  const uint32_t tmem_base = tmem_addr_slot;
  const uint32_t tmem_S = tmem_base + kTmemColS;
  const uint32_t tmem_P = tmem_base + kTmemColP;
  const uint32_t tmem_O = tmem_base + kTmemColO;

  // --- descriptors (both the raw-address and pointer builders) ---
  constexpr uint32_t kSbo128 = 8 * kHeadDim;  // 1024B: 8-row swizzle-atom pitch
  const uint64_t desc_Q = tcgen05::make_smem_desc_sm100<tcgen05::SmemSwizzleMode::kSwizzle128B,
                                                        tcgen05::kKMajorLBO, kSbo128>(smem_Q);
  const uint64_t desc_K = tcgen05::make_smem_desc_sm100(
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_K)), tcgen05::kKMajorLBO, kSbo128,
      tcgen05::SmemSwizzleMode::kSwizzle128B);

  constexpr uint32_t idesc_qk =
      tcgen05::make_instr_desc<kCtaQ, kCtaK, tcgen05::kMmaFmtS8, tcgen05::kMmaFmtS8,
                               tcgen05::kMmaCFmtS32, /*AMajorK=*/true, /*BMajorK=*/true>();
  constexpr uint32_t idesc_pv =
      tcgen05::make_instr_desc<kCtaQ, kHeadDim, tcgen05::kMmaFmtE4M3, tcgen05::kMmaFmtE4M3,
                               tcgen05::kMmaCFmtF32, /*AMajorK=*/true, /*BMajorK=*/true>();
  constexpr uint32_t idesc_pv64 =
      tcgen05::make_instr_desc<kCtaQ, 64, tcgen05::kMmaFmtE4M3, tcgen05::kMmaFmtE4M3,
                               tcgen05::kMmaCFmtF32, /*AMajorK=*/true, /*BMajorK=*/true>();

  // --- MMA kinds, issued by one elected thread of warp 0 ---
  if (threadIdx.x < 32 && tcgen05::elect_one()) {
    tcgen05::mma_i8_ss(tmem_S, desc_Q, desc_K, idesc_qk, enable_d != 0);   // QK^T
    tcgen05::mma_f8f8f32_ts(tmem_O, tmem_P, desc_K, idesc_pv, true);       // PV (TS)
    tcgen05::mma_f8f8f32_ss(tmem_O, desc_Q, desc_K, idesc_pv64, false);    // PV cross-check (SS)
    tcgen05::tcgen05_commit(&bar_S_done);
  }
  tcgen05::wait(&bar_S_done, 0);

  // --- every tcgen05.ld / tcgen05.st shape, warp-collective ---
  uint32_t r4[4], r8[8], r16[16], r32[32];
  tcgen05::tmem_ld_32x32b_x4(r4, tmem_S);
  tcgen05::tmem_ld_32x32b_x8(r8, tmem_S + 4);
  tcgen05::tmem_ld_32x32b_x16(r16, tmem_S + 16);
  tcgen05::tmem_ld_32x32b_x32(r32, tmem_S + 32);
  tcgen05::tmem_ld_wait();

  tcgen05::tmem_st_32x32b_x4(tmem_O, r4);
  tcgen05::tmem_st_32x32b_x8(tmem_O + 4, r8);
  tcgen05::tmem_st_32x32b_x16(tmem_O + 16, r16);
  tcgen05::tmem_st_32x32b_x32(tmem_P, r32);
  tcgen05::tmem_st_wait();

  // --- TMEM producer -> consumer handoff fencing across __syncthreads() ---
  tcgen05::tcgen05_fence_before_sync();
  __syncthreads();
  tcgen05::tcgen05_fence_after_sync();

  // Keep results live so nothing is dead-code eliminated.
  uint32_t acc = r4[0] ^ r8[7] ^ r16[15] ^ r32[31];
#pragma unroll
  for (int i = 0; i < 4; ++i) acc ^= r32[8 * i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc ^ uint32_t(desc_Q) ^ uint32_t(desc_K);

  // --- generic mbarrier arrive + TMA store helpers ---
  if (threadIdx.x == 0) {
    tcgen05::arrive<1>(&bar_Q);
    tcgen05::store_async_4D(&tensorMapQ, smem_Q, 0, 0, 0);
  }

  // --- TMEM dealloc at exit ---
  __syncthreads();
  if (threadIdx.x < 32) {
    tcgen05::tmem_dealloc(tmem_base, kTmemCols);
  }
}
