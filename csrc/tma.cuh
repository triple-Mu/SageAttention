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

// Single home for the host-side TMA tensor-map builder and the mbarrier /
// bulk-tensor PTX helpers shared by the sm90 (wgmma) and sm100 (tcgen05)
// attention kernels. These previously existed in three copies: file-local
// definitions in qk_int_sv_f8_cuda_sm90.cu, the tcgen05.cuh namespace copies,
// and a local create_tensor_map_4D in qk_int_sv_f8_cuda_sm100.cu.
//
// Everything here requires sm_90+ at runtime; create_tensor_map_4D requires
// linking against the CUDA driver (-lcuda, for cuTensorMapEncodeTiled).

#pragma once

#if defined(__CUDACC__)

#include <cuda.h>
#include <stdint.h>
#include <cassert>

template <int BlockMajorSize, int BlockMinorSize, bool swizzle = true,
          CUtensorMapL2promotion_enum promotion_mode = CU_TENSOR_MAP_L2_PROMOTION_NONE,
          typename T>
CUtensorMap create_tensor_map_4D(T *gmem_ptr, int d1, int d2, int d3, int d4, int stride1,
                                 int stride2, int stride3) {
    constexpr int smem_stride = BlockMinorSize * sizeof(T);
    static_assert(sizeof(T) == 2 || sizeof(T) == 1);
    static_assert(smem_stride == 32 || smem_stride == 64 || smem_stride == 128);

    CUtensorMap tma_map;
    void *gmem_address = (void *)gmem_ptr;
    uint64_t gmem_prob_shape[5] = {(uint64_t)d4, (uint64_t)d3, (uint64_t)d2, (uint64_t)d1, 1};
    uint64_t gmem_prob_stride[5] = {(uint64_t)stride3 * sizeof(T), (uint64_t)stride2 * sizeof(T),
                                    (uint64_t)stride1 * sizeof(T), 0, 0};
    uint32_t smem_box_shape[5] = {uint32_t(BlockMinorSize), uint32_t(BlockMajorSize), 1, 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &tma_map,
        (sizeof(T) == 2) ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
        gmem_address, gmem_prob_shape, gmem_prob_stride, smem_box_shape, smem_box_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        (swizzle == false)        ? CU_TENSOR_MAP_SWIZZLE_NONE
        : (smem_stride == 128)    ? CU_TENSOR_MAP_SWIZZLE_128B
        : (smem_stride == 64)     ? CU_TENSOR_MAP_SWIZZLE_64B
                                  : CU_TENSOR_MAP_SWIZZLE_32B,
        promotion_mode, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    assert(result == CUDA_SUCCESS);

    return tma_map;
}

__device__ __forceinline__ void init_barrier(uint64_t* bar, int thread_count) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr), "r"(thread_count));
}

template <uint32_t bytes>
__device__ __forceinline__ void expect_bytes(uint64_t* bar) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar_ptr),
               "n"(bytes));
}

template <typename T>
__device__ __forceinline__ void load_async_4D(T* dst, void const* const src_tma_map,
                                              uint64_t* bar, int s0, int s1, int s2, int s3) {
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5, %6}], [%2];"
      :
      : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "r"(s0), "r"(s1), "r"(s2), "r"(s3)
      : "memory");
}

template <typename T>
__device__ __forceinline__ void store_async_4D(void const* dst_tma_map, T* src,
                                               int global_token_idx, int global_head_idx,
                                               int global_batch_idx) {
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst_tma_map);
  uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));

  asm volatile(
      "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
      " [%0, {%2, %3, %4, %5}], [%1];"
      :
      : "l"(tma_ptr), "r"(src_ptr), "n"(0), "r"(global_token_idx), "r"(global_head_idx),
        "r"(global_batch_idx)
      : "memory");
}

__device__ __forceinline__ void wait(uint64_t* bar, int kPhaseBit) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "{\n"
      ".reg .pred                P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1                       bra.uni DONE;\n"
      "bra.uni                   LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(mbar_ptr),
      "r"(kPhaseBit));
}

template <uint32_t count = 1>
__device__ __forceinline__ void arrive(uint64_t* bar) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
               :
               : "r"(mbar_ptr), "n"(count)
               : "memory");
}

#endif  // defined(__CUDACC__)
