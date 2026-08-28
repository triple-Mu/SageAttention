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

// Hand-written PTX wrappers for the sm100/sm110 tcgen05 (5th-gen tensor core)
// instruction family, sibling of mma.cuh / wgmma.cuh. The mbarrier + TMA
// bulk-tensor helpers shared by the sm90 and sm100 attention kernels live in
// tma.cuh and are re-exported into this namespace below.
//
// tcgen05 is only available on arch-specific targets sm_100a and sm_110a
// (kind::i8 is rejected on sm_103a / sm_100f / sm_110f / base sm_100). Everything
// tcgen05-specific is guarded by SAGE_TCGEN05_ENABLED; on other targets the
// wrappers compile to trap() stubs (mma.cuh RUNTIME_ASSERT convention).

#pragma once

#include <stdint.h>

#if defined(__CUDACC__)
#include <cuda.h>
#endif

#include "tma.cuh"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1100) && \
    defined(__CUDA_ARCH_SPECIFIC__)
#define SAGE_TCGEN05_ENABLED
#endif

#if defined(__CUDACC__)
#define SAGE_TC05_HOST_DEVICE __host__ __device__ __forceinline__
#else
#define SAGE_TC05_HOST_DEVICE inline
#endif

namespace tcgen05 {

// ---------------------------------------------------------------------------
// SM100 shared-memory matrix descriptor (UMMA SmemDescriptor).
//
// Bit layout (cute/arch/mma_sm100_desc.hpp, union SmemDescriptor):
//   [ 0,14)  start_address        smem byte address >> 4
//   [16,30)  leading_byte_offset  LBO bytes >> 4
//   [32,46)  stride_byte_offset   SBO bytes >> 4
//   [46,48)  version              1 on Blackwell (cute make_umma_desc)
//   [49,52)  base_offset          0 (cute make_umma_desc)
//   [52,53)  lbo_mode             0 = legacy (cute make_umma_desc default)
//   [61,64)  layout_type          3-bit swizzle mode (SmemSwizzleMode below)
//
// This is NOT the Hopper wgmma descriptor (wgmma.cuh matrix_descriptor_encode):
// sm100 adds version/lbo_mode and widens layout_type to 3 bits.
// ---------------------------------------------------------------------------

enum class SmemSwizzleMode : uint8_t {
  kNone = 0,               // SWIZZLE_NONE
  kSwizzle128BBase32b = 1, // SWIZZLE_128B_BASE32B (MN-major only)
  kSwizzle128B = 2,        // SWIZZLE_128B
  kSwizzle64B = 4,         // SWIZZLE_64B
  kSwizzle32B = 6,         // SWIZZLE_32B
};

// K-major canonical LBO/SBO (bytes) for the byte-sized (int8/e4m3) row-major
// token x dim tiles this kernel uses (cute make_umma_desc<Major::K>):
//   LBO = 16 (one uint128_t; ignored/unit stride inside a swizzle atom)
//   SBO = 8 rows * row_bytes (8-row swizzle atom pitch), e.g. 1024 for 128B
//   rows (SWIZZLE_128B), 512 for 64B rows (SWIZZLE_64B).
constexpr uint32_t kKMajorLBO = 16;

SAGE_TC05_HOST_DEVICE constexpr uint64_t make_smem_desc_sm100(uint32_t smem_addr,
                                                              uint32_t lbo_bytes,
                                                              uint32_t sbo_bytes,
                                                              SmemSwizzleMode layout,
                                                              uint32_t base_offset = 0,
                                                              uint32_t lbo_mode = 0) {
  return (uint64_t((smem_addr >> 4) & 0x3FFFu))              // start_address [0,14)
         | (uint64_t((lbo_bytes >> 4) & 0x3FFFu) << 16)      // leading_byte_offset [16,30)
         | (uint64_t((sbo_bytes >> 4) & 0x3FFFu) << 32)      // stride_byte_offset [32,46)
         | (uint64_t(1) << 46)                               // version = 1 (Blackwell)
         | (uint64_t(base_offset & 0x7u) << 49)              // base_offset [49,52)
         | (uint64_t(lbo_mode & 0x1u) << 52)                 // lbo_mode [52,53)
         | (uint64_t(static_cast<uint8_t>(layout) & 0x7u) << 61); // layout_type [61,64)
}

// ---------------------------------------------------------------------------
// SM100 MMA instruction descriptor (UMMA InstrDescriptor).
//
// Bit layout (cute/arch/mma_sm100_desc.hpp, union InstrDescriptor); the fields
// we never use (sparse, saturate, negate, max_shift) are left 0, matching the
// cute::UMMA::make_instr_desc defaults:
//   [ 4, 6)  c_format   0 = F16, 1 = F32, 2 = S32
//   [ 7,10)  a_format   kind::f8f6f4: 0=E4M3 1=E5M2; kind::i8: 0=U8 1=S8
//   [10,13)  b_format   same encoding as a_format
//   [15,16)  a_major    0 = K-major, 1 = MN-major
//   [16,17)  b_major    0 = K-major, 1 = MN-major
//   [17,23)  n_dim      N >> 3
//   [24,29)  m_dim      M >> 4
// ---------------------------------------------------------------------------

// a/b_format encodings (per tcgen05.mma kind)
constexpr uint32_t kMmaFmtE4M3 = 0;  // kind::f8f6f4  (MXF8F6F4Format::E4M3)
constexpr uint32_t kMmaFmtE5M2 = 1;  // kind::f8f6f4  (MXF8F6F4Format::E5M2)
constexpr uint32_t kMmaFmtU8 = 0;    // kind::i8      (S8Format::UINT8)
constexpr uint32_t kMmaFmtS8 = 1;    // kind::i8      (S8Format::INT8)
// c_format encodings
constexpr uint32_t kMmaCFmtF16 = 0;
constexpr uint32_t kMmaCFmtF32 = 1;
constexpr uint32_t kMmaCFmtS32 = 2;

template <uint32_t M, uint32_t N, uint32_t AFmt, uint32_t BFmt, uint32_t CFmt,
          bool AMajorK, bool BMajorK>
SAGE_TC05_HOST_DEVICE constexpr uint32_t make_instr_desc() {
  static_assert(M == 64 || M == 128, "tcgen05 idesc: M must be 64 or 128 (cta_group::1)");
  static_assert(N % 16 == 0 && N >= 16 && N <= 256, "tcgen05 idesc: invalid N");
  static_assert(AFmt <= 7 && BFmt <= 7, "tcgen05 idesc: a/b_format is a 3-bit field");
  static_assert(CFmt <= 2, "tcgen05 idesc: c_format must be F16(0)/F32(1)/S32(2)");
  return (CFmt << 4)                       // c_format [4,6)
         | (AFmt << 7)                     // a_format [7,10)
         | (BFmt << 10)                    // b_format [10,13)
         | ((AMajorK ? 0u : 1u) << 15)     // a_major [15,16)
         | ((BMajorK ? 0u : 1u) << 16)     // b_major [16,17)
         | ((N >> 3) << 17)                // n_dim [17,23)
         | ((M >> 4) << 24);               // m_dim [24,29)
}

#if defined(__CUDACC__)

// ---------------------------------------------------------------------------
// mbarrier + TMA bulk-tensor helpers (tma.cuh); re-exported here so the sm100
// kernel keeps addressing them as tcgen05::. Require sm_90+.
// ---------------------------------------------------------------------------

using ::init_barrier;
using ::expect_bytes;
using ::load_async_4D;
using ::store_async_4D;
using ::wait;
using ::arrive;

// Device-side smem descriptor builder (LBO/SBO/swizzle as compile-time knobs).
template <SmemSwizzleMode layout, uint32_t lbo_bytes, uint32_t sbo_bytes, typename T>
__device__ __forceinline__ uint64_t make_smem_desc_sm100(T* smem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  return make_smem_desc_sm100(addr, lbo_bytes, sbo_bytes, layout);
}

// elect.sync: returns true for exactly one active lane of the warp (sm_90+).
__device__ __forceinline__ bool elect_one() {
  uint32_t pred = 0;
  asm volatile(
      "{\n"
      ".reg .pred P;\n"
      "elect.sync _|P, 0xffffffff;\n"
      "selp.b32 %0, 1, 0, P;\n"
      "}\n"
      : "=r"(pred));
  return pred != 0;
}

// ---------------------------------------------------------------------------
// TMEM allocation (warp-collective; nCols power of two >= 32).
// ---------------------------------------------------------------------------

__device__ __forceinline__ void tmem_alloc(uint32_t* smem_dst, uint32_t ncols) {
#ifdef SAGE_TCGEN05_ENABLED
  uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(
                   dst_ptr),
               "r"(ncols)
               : "memory");
#else
  (void)smem_dst;
  (void)ncols;
  __trap();
#endif
}

__device__ __forceinline__ void tmem_dealloc(uint32_t tmem_addr, uint32_t ncols) {
#ifdef SAGE_TCGEN05_ENABLED
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(tmem_addr),
               "r"(ncols)
               : "memory");
#else
  (void)tmem_addr;
  (void)ncols;
  __trap();
#endif
}

__device__ __forceinline__ void tmem_relinquish() {
#ifdef SAGE_TCGEN05_ENABLED
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
#else
  __trap();
#endif
}

// ---------------------------------------------------------------------------
// tcgen05.mma wrappers (single elected thread issues these; async, tracked by
// tcgen05_commit + mbarrier). d_tmem is the accumulator TMEM address;
// enable_input_d = false zero-initializes the accumulator (ScaleD=0).
// ---------------------------------------------------------------------------

// QK^T: s8 x s8 -> s32 (A and B from smem descriptors).
__device__ __forceinline__ void mma_i8_ss(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
                                          uint32_t i_desc, bool enable_input_d) {
#ifdef SAGE_TCGEN05_ENABLED
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::i8 [%0], %1, %2, %3, p;\n"
      "}\n"
      :
      : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(uint32_t(enable_input_d))
      : "memory");
#else
  (void)d_tmem;
  (void)a_desc;
  (void)b_desc;
  (void)i_desc;
  (void)enable_input_d;
  __trap();
#endif
}

// PV: e4m3 x e4m3 -> f32, A from TMEM (TS; A must be K-major).
__device__ __forceinline__ void mma_f8f8f32_ts(uint32_t d_tmem, uint32_t a_tmem,
                                               uint64_t b_desc, uint32_t i_desc,
                                               bool enable_input_d) {
#ifdef SAGE_TCGEN05_ENABLED
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p;\n"
      "}\n"
      :
      : "r"(d_tmem), "r"(a_tmem), "l"(b_desc), "r"(i_desc), "r"(uint32_t(enable_input_d))
      : "memory");
#else
  (void)d_tmem;
  (void)a_tmem;
  (void)b_desc;
  (void)i_desc;
  (void)enable_input_d;
  __trap();
#endif
}

// PV cross-check variant (P staged through smem) + future fp8-QK family variant:
// e4m3 x e4m3 -> f32, A from smem descriptor (SS).
__device__ __forceinline__ void mma_f8f8f32_ss(uint32_t d_tmem, uint64_t a_desc,
                                               uint64_t b_desc, uint32_t i_desc,
                                               bool enable_input_d) {
#ifdef SAGE_TCGEN05_ENABLED
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n"
      "}\n"
      :
      : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(uint32_t(enable_input_d))
      : "memory");
#else
  (void)d_tmem;
  (void)a_desc;
  (void)b_desc;
  (void)i_desc;
  (void)enable_input_d;
  __trap();
#endif
}

// ---------------------------------------------------------------------------
// tcgen05.ld / tcgen05.st, 32x32b shape (warp-collective; each thread owns one
// TMEM lane = one attention row; .xN = N consecutive 32-bit columns/thread).
// ---------------------------------------------------------------------------

// clang-format off
#define SAGE_TC05_R_0_7    "%0, %1, %2, %3, %4, %5, %6, %7"
#define SAGE_TC05_R_8_15   "%8, %9, %10, %11, %12, %13, %14, %15"
#define SAGE_TC05_R_16_23  "%16, %17, %18, %19, %20, %21, %22, %23"
#define SAGE_TC05_R_24_31  "%24, %25, %26, %27, %28, %29, %30, %31"
#define SAGE_TC05_R_1_8    "%1, %2, %3, %4, %5, %6, %7, %8"
#define SAGE_TC05_R_9_16   "%9, %10, %11, %12, %13, %14, %15, %16"
#define SAGE_TC05_R_17_24  "%17, %18, %19, %20, %21, %22, %23, %24"
#define SAGE_TC05_R_25_32  "%25, %26, %27, %28, %29, %30, %31, %32"
// clang-format on

#define SAGE_TC05_OUT4(r, i) \
  "=r"((r)[(i) + 0]), "=r"((r)[(i) + 1]), "=r"((r)[(i) + 2]), "=r"((r)[(i) + 3])
#define SAGE_TC05_OUT8(r, i) SAGE_TC05_OUT4(r, (i) + 0), SAGE_TC05_OUT4(r, (i) + 4)
#define SAGE_TC05_OUT16(r, i) SAGE_TC05_OUT8(r, (i) + 0), SAGE_TC05_OUT8(r, (i) + 8)
#define SAGE_TC05_OUT32(r, i) SAGE_TC05_OUT16(r, (i) + 0), SAGE_TC05_OUT16(r, (i) + 16)

#define SAGE_TC05_IN4(r, i) \
  "r"((r)[(i) + 0]), "r"((r)[(i) + 1]), "r"((r)[(i) + 2]), "r"((r)[(i) + 3])
#define SAGE_TC05_IN8(r, i) SAGE_TC05_IN4(r, (i) + 0), SAGE_TC05_IN4(r, (i) + 4)
#define SAGE_TC05_IN16(r, i) SAGE_TC05_IN8(r, (i) + 0), SAGE_TC05_IN8(r, (i) + 8)
#define SAGE_TC05_IN32(r, i) SAGE_TC05_IN16(r, (i) + 0), SAGE_TC05_IN16(r, (i) + 16)

#define SAGE_TC05_STUB_LD(r, n) \
  _Pragma("unroll") for (uint32_t _i = 0; _i < (n); ++_i)(r)[_i] = 0u; \
  (void)tmem_addr; \
  __trap();

__device__ __forceinline__ void tmem_ld_32x32b_x4(uint32_t r[4], uint32_t tmem_addr) {
#ifdef SAGE_TCGEN05_ENABLED
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];\n"
               : SAGE_TC05_OUT4(r, 0)
               : "r"(tmem_addr));
#else
  SAGE_TC05_STUB_LD(r, 4)
#endif
}

__device__ __forceinline__ void tmem_ld_32x32b_x8(uint32_t r[8], uint32_t tmem_addr) {
#ifdef SAGE_TCGEN05_ENABLED
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {" SAGE_TC05_R_0_7 "}, [%8];\n"
               : SAGE_TC05_OUT8(r, 0)
               : "r"(tmem_addr));
#else
  SAGE_TC05_STUB_LD(r, 8)
#endif
}

__device__ __forceinline__ void tmem_ld_32x32b_x16(uint32_t r[16], uint32_t tmem_addr) {
#ifdef SAGE_TCGEN05_ENABLED
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {" SAGE_TC05_R_0_7 ", " SAGE_TC05_R_8_15
               "}, [%16];\n"
               : SAGE_TC05_OUT16(r, 0)
               : "r"(tmem_addr));
#else
  SAGE_TC05_STUB_LD(r, 16)
#endif
}

__device__ __forceinline__ void tmem_ld_32x32b_x32(uint32_t r[32], uint32_t tmem_addr) {
#ifdef SAGE_TCGEN05_ENABLED
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {" SAGE_TC05_R_0_7 ", " SAGE_TC05_R_8_15
               ", " SAGE_TC05_R_16_23 ", " SAGE_TC05_R_24_31 "}, [%32];\n"
               : SAGE_TC05_OUT32(r, 0)
               : "r"(tmem_addr));
#else
  SAGE_TC05_STUB_LD(r, 32)
#endif
}

__device__ __forceinline__ void tmem_st_32x32b_x4(uint32_t tmem_addr, const uint32_t r[4]) {
#ifdef SAGE_TCGEN05_ENABLED
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};\n"
               :
               : "r"(tmem_addr), SAGE_TC05_IN4(r, 0)
               : "memory");
#else
  (void)tmem_addr;
  (void)r;
  __trap();
#endif
}

__device__ __forceinline__ void tmem_st_32x32b_x8(uint32_t tmem_addr, const uint32_t r[8]) {
#ifdef SAGE_TCGEN05_ENABLED
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {" SAGE_TC05_R_1_8 "};\n"
               :
               : "r"(tmem_addr), SAGE_TC05_IN8(r, 0)
               : "memory");
#else
  (void)tmem_addr;
  (void)r;
  __trap();
#endif
}

__device__ __forceinline__ void tmem_st_32x32b_x16(uint32_t tmem_addr, const uint32_t r[16]) {
#ifdef SAGE_TCGEN05_ENABLED
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {" SAGE_TC05_R_1_8
               ", " SAGE_TC05_R_9_16 "};\n"
               :
               : "r"(tmem_addr), SAGE_TC05_IN16(r, 0)
               : "memory");
#else
  (void)tmem_addr;
  (void)r;
  __trap();
#endif
}

__device__ __forceinline__ void tmem_st_32x32b_x32(uint32_t tmem_addr, const uint32_t r[32]) {
#ifdef SAGE_TCGEN05_ENABLED
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {" SAGE_TC05_R_1_8
               ", " SAGE_TC05_R_9_16 ", " SAGE_TC05_R_17_24 ", " SAGE_TC05_R_25_32 "};\n"
               :
               : "r"(tmem_addr), SAGE_TC05_IN32(r, 0)
               : "memory");
#else
  (void)tmem_addr;
  (void)r;
  __trap();
#endif
}

// ---------------------------------------------------------------------------
// Completion / ordering.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void tmem_ld_wait() {
#ifdef SAGE_TCGEN05_ENABLED
  asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
#else
  __trap();
#endif
}

__device__ __forceinline__ void tmem_st_wait() {
#ifdef SAGE_TCGEN05_ENABLED
  asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
#else
  __trap();
#endif
}

// Makes `bar` track completion of all prior tcgen05.mma issued by this CTA;
// consumer spins on wait(bar, phase).
__device__ __forceinline__ void tcgen05_commit(uint64_t* bar) {
#ifdef SAGE_TCGEN05_ENABLED
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(
          bar_ptr)
      : "memory");
#else
  (void)bar;
  __trap();
#endif
}

// Order TMEM accesses across __syncthreads(): producer issues
// tcgen05_fence_before_sync() before the barrier, consumer issues
// tcgen05_fence_after_sync() after it.
__device__ __forceinline__ void tcgen05_fence_before_sync() {
#ifdef SAGE_TCGEN05_ENABLED
  asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
#else
  __trap();
#endif
}

__device__ __forceinline__ void tcgen05_fence_after_sync() {
#ifdef SAGE_TCGEN05_ENABLED
  asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
#else
  __trap();
#endif
}

#undef SAGE_TC05_STUB_LD
#undef SAGE_TC05_OUT4
#undef SAGE_TC05_OUT8
#undef SAGE_TC05_OUT16
#undef SAGE_TC05_OUT32
#undef SAGE_TC05_IN4
#undef SAGE_TC05_IN8
#undef SAGE_TC05_IN16
#undef SAGE_TC05_IN32
#undef SAGE_TC05_R_0_7
#undef SAGE_TC05_R_8_15
#undef SAGE_TC05_R_16_23
#undef SAGE_TC05_R_24_31
#undef SAGE_TC05_R_1_8
#undef SAGE_TC05_R_9_16
#undef SAGE_TC05_R_17_24
#undef SAGE_TC05_R_25_32

#endif  // defined(__CUDACC__)

}  // namespace tcgen05
