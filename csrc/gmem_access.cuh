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

// Global-memory access primitives shared by the quantization kernels:
// how a global address is computed (64-bit base + 32-bit in-CTA offsets,
// see the width contract in qattn/launch_utils.cuh) and how wide one
// load/store moves data.
//
// vec_t is adapted from the user's CTI common.cuh, widened to every
// LDG/STG size the kernels actually use (2/4/8/16 bytes). 128-bit is the
// hardware ceiling for a single global load/store on every supported arch
// (there is no ld.global.v8.b32); anything wider goes through cp.async
// (../cp_async.cuh) or TMA (../tma.cuh), not through this header.

#pragma once

#include <cstdint>

namespace sage {

// Block-level base offset in one 64-bit expression. __host__ __device__ so a
// host-side unit test can sweep the 2^32 boundary without touching a GPU.
__host__ __device__ __forceinline__ int64_t gmem_base_offset(uint32_t batch_id,
                                                             int64_t  stride_batch,
                                                             uint32_t head_id,
                                                             int64_t  stride_h,
                                                             uint32_t seq_idx,
                                                             uint32_t stride_seq,
                                                             uint32_t lane_off)
{
    return static_cast<int64_t>(batch_id) * stride_batch + static_cast<int64_t>(head_id) * stride_h
           + static_cast<int64_t>(seq_idx) * stride_seq + static_cast<int64_t>(lane_off);
}

template<int BYTES>
struct vec_storage;
template<>
struct vec_storage<2> {
    using type = uint16_t;
};
template<>
struct vec_storage<4> {
    using type = uint32_t;
};
template<>
struct vec_storage<8> {
    using type = uint2;
};
template<>
struct vec_storage<16> {
    using type = uint4;
};

// One LDG/STG worth of elements. alignas guarantees the vector width is legal
// whenever the pointer arithmetic that produced `p` is element-aligned.
template<typename T, int pack_size>
struct alignas(pack_size * sizeof(T)) vec_t {
    static constexpr int kBytes = pack_size * static_cast<int>(sizeof(T));
    static_assert(kBytes == 2 || kBytes == 4 || kBytes == 8 || kBytes == 16,
                  "vec_t must map to exactly one LDG/STG (2/4/8/16 bytes)");
    using storage_t = typename vec_storage<kBytes>::type;

    T data[pack_size];

    __device__ __forceinline__ T& operator[](int i)
    {
        return data[i];
    }
    __device__ __forceinline__ const T& operator[](int i) const
    {
        return data[i];
    }

    // reinterpret_cast, not cuda::std::bit_cast: bit_cast can only re-type the
    // in-register half of each of these (the caller's T* -> storage_t* cast has
    // no well-typed spelling), and doing so perturbed MeanScaleKernel's SASS
    // (same instruction count and registers, different NOP/SHF/LOP3 mix) for no
    // aliasing gain.
    __device__ __forceinline__ void load(const T* p)
    {
        *reinterpret_cast<storage_t*>(data) = *reinterpret_cast<const storage_t*>(p);
    }
    // Read-only path (ld.global.nc): the quant kernels take non-const
    // T* __restrict__ inputs, so nvcc cannot prove read-only on its own.
    __device__ __forceinline__ void load_ro(const T* p)
    {
        *reinterpret_cast<storage_t*>(data) = __ldg(reinterpret_cast<const storage_t*>(p));
    }
    __device__ __forceinline__ void store(T* p) const
    {
        *reinterpret_cast<storage_t*>(p) = *reinterpret_cast<const storage_t*>(data);
    }
};

}  // namespace sage
