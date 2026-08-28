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

// Host-only, GPU-free parity test: our hand-rolled SM100 descriptor builders
// (tcgen05::make_smem_desc_sm100 / tcgen05::make_instr_desc) vs the CUTLASS
// cute::UMMA reference builders, bit-for-bit.
//
// CUTLASS is used here ONLY as a CPU test oracle; no CUTLASS header is
// included by any shipped kernel source.
//
// Build (container, CUTLASS from the vendored pytorch third_party checkout):
//   g++ -std=c++17 -O2 -I<cutlass>/include -I/usr/local/cuda/include \
//       tests/test_desc_parity.cpp -o test_desc_parity     (or nvcc -x cu)
//
// Note: cute::UMMA::make_umma_desc calls cast_smem_ptr_to_uint, whose host
// fallback prints "ERROR: cast_smem_ptr_to_uint not supported but used." and
// returns 0. That is expected here: those cases compare descriptors with
// start_address = 0; the start_address bit placement is covered separately by
// the SmemDescriptor bitfield-union oracle cases.

#include <cstdint>
#include <cstdio>

#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm100.hpp>

#include "../csrc/tcgen05.cuh"

using cute::UMMA::Major;

static int g_pass = 0;
static int g_fail = 0;

static void check64(const char* name, uint64_t ours, uint64_t cutlass_ref) {
  const bool ok = (ours == cutlass_ref);
  (ok ? g_pass : g_fail)++;
  std::printf("[%s] %-58s ours=0x%016llx cute=0x%016llx\n", ok ? "PASS" : "FAIL", name,
              (unsigned long long)ours, (unsigned long long)cutlass_ref);
}

static void check32(const char* name, uint32_t ours, uint32_t cutlass_ref) {
  const bool ok = (ours == cutlass_ref);
  (ok ? g_pass : g_fail)++;
  std::printf("[%s] %-58s ours=0x%08x cute=0x%08x\n", ok ? "PASS" : "FAIL", name, ours,
              cutlass_ref);
}

// ---------------------------------------------------------------------------
// Instruction descriptor parity vs cute::UMMA::make_instr_desc.
// ---------------------------------------------------------------------------

template <int M, int N, class AType, class BType, class CType, Major AMaj, Major BMaj,
          uint32_t AFmt, uint32_t BFmt, uint32_t CFmt>
static void check_idesc(const char* name) {
  auto ref = cute::UMMA::make_instr_desc<AType, BType, CType, M, N, AMaj, BMaj>();
  const uint32_t ours = tcgen05::make_instr_desc<M, N, AFmt, BFmt, CFmt,
                                                 /*AMajorK=*/AMaj == Major::K,
                                                 /*BMajorK=*/BMaj == Major::K>();
  check32(name, ours, ref.desc_);
}

template <int M, int N>
static void check_idesc_all_majors() {
  static char name[128];

  // (s8, s8, s32) -- QK^T, kind::i8
  #define CASE_S8(AM, BM)                                                                   \
    std::snprintf(name, sizeof(name), "idesc M=%d N=%d s8s8s32 a_major=%s b_major=%s", M, N, \
                  (AM == Major::K ? "K" : "MN"), (BM == Major::K ? "K" : "MN"));            \
    check_idesc<M, N, int8_t, int8_t, int32_t, AM, BM, tcgen05::kMmaFmtS8,                  \
                tcgen05::kMmaFmtS8, tcgen05::kMmaCFmtS32>(name)
  CASE_S8(Major::K, Major::K);
  CASE_S8(Major::K, Major::MN);
  CASE_S8(Major::MN, Major::K);
  CASE_S8(Major::MN, Major::MN);
  #undef CASE_S8

  // (e4m3, e4m3, f32) -- PV, kind::f8f6f4
  #define CASE_F8(AM, BM)                                                                 \
    std::snprintf(name, sizeof(name), "idesc M=%d N=%d e4m3e4m3f32 a_major=%s b_major=%s", \
                  M, N, (AM == Major::K ? "K" : "MN"), (BM == Major::K ? "K" : "MN"));    \
    check_idesc<M, N, cute::float_e4m3_t, cute::float_e4m3_t, float, AM, BM,              \
                tcgen05::kMmaFmtE4M3, tcgen05::kMmaFmtE4M3, tcgen05::kMmaCFmtF32>(name)
  CASE_F8(Major::K, Major::K);
  CASE_F8(Major::K, Major::MN);
  CASE_F8(Major::MN, Major::K);
  CASE_F8(Major::MN, Major::MN);
  #undef CASE_F8
}

// ---------------------------------------------------------------------------
// Smem descriptor parity vs cute::UMMA::make_umma_desc for the canonical
// K-major swizzled tiles the sm100 kernel uses (byte-sized elements, row-major
// token x dim smem). start_address is 0 on host (see file header note).
// ---------------------------------------------------------------------------

template <class T, class SwAtom, int Rows, int RowBytes>
static uint64_t cute_kmajor_desc() {
  auto layout = cute::tile_to_shape(SwAtom{}, cute::Shape<cute::Int<Rows>, cute::Int<RowBytes>>{});
  auto t = cute::make_tensor(cute::make_smem_ptr(static_cast<T const*>(nullptr)), layout);
  cute::UMMA::SmemDescriptor d = cute::UMMA::make_umma_desc<Major::K>(t);
  return d.desc_;
}

static void check_smem_desc_semantics() {
  // Q/K tile, HEAD_DIM=128: 128 rows x 128B rows, SWIZZLE_128B, LBO=16B, SBO=1024B.
  check64("smem 128x128 s8 K-major SW128 (LBO=16,SBO=1024)",
          tcgen05::make_smem_desc_sm100(0, tcgen05::kKMajorLBO, 1024,
                                        tcgen05::SmemSwizzleMode::kSwizzle128B),
          cute_kmajor_desc<int8_t, cute::UMMA::Layout_K_SW128_Atom<int8_t>, 128, 128>());

  // V^T tile, HEAD_DIM=128: 128 rows x CTA_K=128 cols e4m3, same geometry.
  check64("smem 128x128 e4m3 K-major SW128 (LBO=16,SBO=1024)",
          tcgen05::make_smem_desc_sm100(0, tcgen05::kKMajorLBO, 1024,
                                        tcgen05::SmemSwizzleMode::kSwizzle128B),
          cute_kmajor_desc<cute::float_e4m3_t,
                           cute::UMMA::Layout_K_SW128_Atom<cute::float_e4m3_t>, 128, 128>());

  // Q/K tile, HEAD_DIM=64: 128 rows x 64B rows, SWIZZLE_64B, LBO=16B, SBO=512B.
  check64("smem 128x64 s8 K-major SW64 (LBO=16,SBO=512)",
          tcgen05::make_smem_desc_sm100(0, tcgen05::kKMajorLBO, 512,
                                        tcgen05::SmemSwizzleMode::kSwizzle64B),
          cute_kmajor_desc<int8_t, cute::UMMA::Layout_K_SW64_Atom<int8_t>, 128, 64>());

  // V^T tile, HEAD_DIM=64: 64 rows x CTA_K=128 cols e4m3 -> still SWIZZLE_128B.
  check64("smem 64x128 e4m3 K-major SW128 (LBO=16,SBO=1024)",
          tcgen05::make_smem_desc_sm100(0, tcgen05::kKMajorLBO, 1024,
                                        tcgen05::SmemSwizzleMode::kSwizzle128B),
          cute_kmajor_desc<cute::float_e4m3_t,
                           cute::UMMA::Layout_K_SW128_Atom<cute::float_e4m3_t>, 64, 128>());
}

// ---------------------------------------------------------------------------
// Smem descriptor bit-placement parity (incl. nonzero start_address) vs the
// cute::UMMA::SmemDescriptor bitfield union, populated exactly the way
// make_umma_desc populates it (version=1, base_offset=0, lbo_mode=0).
// ---------------------------------------------------------------------------

static uint64_t union_oracle(uint32_t addr, uint32_t lbo_bytes, uint32_t sbo_bytes,
                             uint8_t layout_type) {
  cute::UMMA::SmemDescriptor d;
  d.desc_ = 0;
  d.version_ = 1;
  d.lbo_mode_ = 0;
  d.base_offset_ = 0;
  d.layout_type_ = layout_type;
  d.start_address_ = static_cast<uint16_t>(addr >> 4);
  d.leading_byte_offset_ = static_cast<uint16_t>(lbo_bytes >> 4);
  d.stride_byte_offset_ = static_cast<uint16_t>(sbo_bytes >> 4);
  return d.desc_;
}

static void check_smem_desc_bits() {
  struct Case {
    const char* name;
    uint32_t addr, lbo, sbo;
    tcgen05::SmemSwizzleMode mode;
  };
  const Case cases[] = {
      {"smem bits addr=0x0000 SW128", 0x0000u, 16u, 1024u, tcgen05::SmemSwizzleMode::kSwizzle128B},
      {"smem bits addr=0x0010 SW128", 0x0010u, 16u, 1024u, tcgen05::SmemSwizzleMode::kSwizzle128B},
      {"smem bits addr=0x4000 SW128", 0x4000u, 16u, 1024u, tcgen05::SmemSwizzleMode::kSwizzle128B},
      {"smem bits addr=0xbff0 SW128", 0xbff0u, 16u, 1024u, tcgen05::SmemSwizzleMode::kSwizzle128B},
      {"smem bits addr=0x8400 SW64", 0x8400u, 16u, 512u, tcgen05::SmemSwizzleMode::kSwizzle64B},
      {"smem bits addr=0x2a30 SW32", 0x2a30u, 16u, 256u, tcgen05::SmemSwizzleMode::kSwizzle32B},
      {"smem bits addr=0x1230 NONE", 0x1230u, 128u, 256u, tcgen05::SmemSwizzleMode::kNone},
  };
  for (const Case& c : cases) {
    check64(c.name, tcgen05::make_smem_desc_sm100(c.addr, c.lbo, c.sbo, c.mode),
            union_oracle(c.addr, c.lbo, c.sbo, static_cast<uint8_t>(c.mode)));
  }
}

int main() {
  std::printf("== instruction descriptor parity (cute::UMMA::make_instr_desc) ==\n");
  check_idesc_all_majors<128, 128>();
  check_idesc_all_majors<128, 64>();

  std::printf("== smem descriptor parity (cute::UMMA::make_umma_desc, K-major tiles) ==\n");
  check_smem_desc_semantics();

  std::printf("== smem descriptor bit placement (cute::UMMA::SmemDescriptor union) ==\n");
  check_smem_desc_bits();

  std::printf("== summary: %d PASS, %d FAIL ==\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
