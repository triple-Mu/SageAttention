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
//
// The device-side helpers wrap cuda::ptx (CCCL, needs CUDA 12.3+) rather than
// raw asm. wait() is the one exception; see the comment on it.

#pragma once

#if defined(__CUDACC__)

#include <cuda.h>
#include <cuda/ptx>
#include <stdexcept>
#include <stdint.h>
#include <string>

// dims/strides are int64_t: a tensor stride in [2^31, 2^32) passed as int
// used to sign-extend into a ~1.8e19 byte stride and fail encoding with an
// opaque driver error (the long-seq "cuTensorMapEncodeTiled failed" symptom).
template<int                         BlockMajorSize,
         int                         BlockMinorSize,
         bool                        swizzle        = true,
         CUtensorMapL2promotion_enum promotion_mode = CU_TENSOR_MAP_L2_PROMOTION_NONE,
         typename T>
CUtensorMap create_tensor_map_4D(T*      gmem_ptr,
                                 int64_t dim_batch,
                                 int64_t dim_head,
                                 int64_t dim_major,
                                 int64_t dim_minor,
                                 int64_t stride_batch,
                                 int64_t stride_head,
                                 int64_t stride_major)
{
    constexpr int smem_stride = BlockMinorSize * sizeof(T);
    static_assert(sizeof(T) == 2 || sizeof(T) == 1);
    static_assert(smem_stride == 32 || smem_stride == 64 || smem_stride == 128);

    CUtensorMap tma_map;
    void*       gmem_address       = (void*)gmem_ptr;
    uint64_t    gmem_prob_shape[5] = {
           (uint64_t)dim_minor, (uint64_t)dim_major, (uint64_t)dim_head, (uint64_t)dim_batch, 1};
    uint64_t gmem_prob_stride[5] = {(uint64_t)stride_major * sizeof(T),
                                    (uint64_t)stride_head * sizeof(T),
                                    (uint64_t)stride_batch * sizeof(T),
                                    0,
                                    0};
    uint32_t smem_box_shape[5]   = {uint32_t(BlockMinorSize), uint32_t(BlockMajorSize), 1, 1, 1};
    uint32_t smem_box_stride[5]  = {1, 1, 1, 1, 1};

    // Pre-validate the driver's documented limits so an oversized tensor
    // fails with the actual number instead of a bare CUDA_ERROR_INVALID_VALUE.
    for (int i = 0; i < 4; i++) {
        if (gmem_prob_shape[i] > (uint64_t(1) << 32)) {
            throw std::runtime_error("create_tensor_map_4D: dim " + std::to_string(i) + " ("
                                     + std::to_string(gmem_prob_shape[i]) + " elements) exceeds the TMA 2^32 limit");
        }
    }
    for (int i = 0; i < 3; i++) {
        if (gmem_prob_stride[i] >= (uint64_t(1) << 40) || gmem_prob_stride[i] % 16 != 0) {
            throw std::runtime_error("create_tensor_map_4D: stride " + std::to_string(i) + " ("
                                     + std::to_string(gmem_prob_stride[i])
                                     + " bytes) must be 16-byte aligned and below the TMA 2^40 limit");
        }
    }

    CUresult result =
        cuTensorMapEncodeTiled(&tma_map,
                               (sizeof(T) == 2) ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_UINT8,
                               4,
                               gmem_address,
                               gmem_prob_shape,
                               gmem_prob_stride,
                               smem_box_shape,
                               smem_box_stride,
                               CU_TENSOR_MAP_INTERLEAVE_NONE,
                               (swizzle == false)   ? CU_TENSOR_MAP_SWIZZLE_NONE :
                               (smem_stride == 128) ? CU_TENSOR_MAP_SWIZZLE_128B :
                               (smem_stride == 64)  ? CU_TENSOR_MAP_SWIZZLE_64B :
                                                      CU_TENSOR_MAP_SWIZZLE_32B,
                               promotion_mode,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    // plain assert() vanishes under -DNDEBUG; a failed encode would then reach
    // the kernel as a garbage tensor map. Keep this header torch-free, so
    // throw std::runtime_error (converted to a Python RuntimeError upstream).
    if (result != CUDA_SUCCESS) {
        const char* msg = nullptr;
        cuGetErrorString(result, &msg);
        throw std::runtime_error(std::string("cuTensorMapEncodeTiled failed: ")
                                 + (msg ? msg : "unknown CUDA driver error"));
    }

    return tma_map;
}

__device__ __forceinline__ void init_barrier(uint64_t* bar, int thread_count)
{
    cuda::ptx::mbarrier_init(bar, static_cast<uint32_t>(thread_count));
}

// mbarrier.arrive.expect_tx without an explicit .sem/.scope defaults to
// .release.cta, which is what cuda::ptx spells out. The returned arrival state
// is unused here (the hand-written form discarded it with `_`).
template<uint32_t bytes>
__device__ __forceinline__ void expect_bytes(uint64_t* bar)
{
    cuda::ptx::mbarrier_arrive_expect_tx(
        cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, bar, bytes);
}

template<typename T>
__device__ __forceinline__ void
load_async_4D(T* dst, void const* const src_tma_map, uint64_t* bar, int coord0, int coord1, int coord2, int coord3)
{
    const int32_t coords[4] = {coord0, coord1, coord2, coord3};
    cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_cluster, cuda::ptx::space_global, dst, src_tma_map, coords, bar);
}

// reserved: currently unused, planned for the sm90 TMA-store epilogue (perf
// batch C item H6) — do not remove as dead code.
template<typename T>
__device__ __forceinline__ void store_async_4D(void const* dst_tma_map, T* src, int coord1, int coord2, int coord3)
{
    const int32_t coords[4] = {0, coord1, coord2, coord3};
    cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_global, cuda::ptx::space_shared, dst_tma_map, coords, src);
}

// Kept as hand-written asm on purpose. The equivalent
// `while (!cuda::ptx::mbarrier_try_wait_parity(bar, kPhaseBit)) {}` is correct,
// but it exposes the spin loop to NVVM as a real CFG loop instead of one opaque
// asm block. That splits the surrounding basic blocks and costs +6% SASS
// instructions in the sm90 kernel (measured: 108928 -> 115456, with 6144 FFMA
// de-contracted into FADD+FMUL). The clobber list is not the cause — adding
// `: "memory"` to the loop below reproduces the baseline exactly.
__device__ __forceinline__ void wait(uint64_t* bar, int kPhaseBit)
{
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("{\n"
                 ".reg .pred                P1;\n"
                 "LAB_WAIT:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                 "@P1                       bra.uni DONE;\n"
                 "bra.uni                   LAB_WAIT;\n"
                 "DONE:\n"
                 "}\n" ::"r"(mbar_addr),
                 "r"(kPhaseBit));
}

template<uint32_t count = 1>
__device__ __forceinline__ void arrive(uint64_t* bar)
{
    cuda::ptx::mbarrier_arrive(cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, bar, count);
}

#endif  // defined(__CUDACC__)
