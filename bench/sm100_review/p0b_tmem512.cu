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

// M0 ignition probe B: the ws kernel's TMEM lifecycle at 512 columns / 512
// threads / 1 CTA per SM. The mma warp (12) allocs all 512 columns and
// relinquishes; the 12 TMEM-reading warps (0-11) each st/ld their lane
// quadrant at per-warpgroup column blocks (incl. the top columns 508-511),
// verify the round-trip, then arrive on the dealloc mbarrier (count 384);
// the mma warp waits on it and deallocs collectively - exactly the ws
// kernel's shutdown protocol.
//
// Compile gate (no GPU needed):
//   nvcc -std=c++17 -O3 -cubin -Xptxas -v -arch=sm_100a \
//        -I csrc -o /tmp/p0b.cubin bench/sm100_review/p0b_tmem512.cu
// Hardware run:
//   nvcc -std=c++17 -O3 -arch=sm_100a -I csrc -o /tmp/p0b bench/sm100_review/p0b_tmem512.cu
//   /tmp/p0b   (prints PASS/FAIL; a hang here means the alloc/dealloc
//               handshake is broken - see the triage plan in C1_DESIGN.md)

#include <cstdio>
#include <stdint.h>

#include "tcgen05.cuh"

namespace {

constexpr int      kThreads   = 512;
constexpr uint32_t kTmemCols  = 512;
constexpr int      kMmaWarp   = 12;
constexpr int      kReaders   = 12 * 32;  // warps 0-11

__global__ void __launch_bounds__(kThreads, 1) p0b_kernel(uint32_t* err)
{
    __shared__ __align__(8) uint64_t bar_dealloc;
    __shared__ __align__(4) uint32_t tmem_addr_slot;

    const uint32_t warp = threadIdx.x / 32;

    if (threadIdx.x == 0) {
        tcgen05::init_barrier(&bar_dealloc, kReaders);
    }
    if (warp == kMmaWarp) {
        tcgen05::tmem_alloc(&tmem_addr_slot, kTmemCols);
        tcgen05::tmem_relinquish();
    }
    __syncthreads();

    if (warp < 12) {
        // warpgroup g in {0,1,2} probes columns {g*168, g*168+164} and, for
        // g == 2, the top block 508-511; lane quadrant = warp%4 (all three
        // warps of a quadrant use disjoint columns).
        const uint32_t g         = warp / 4;
        const uint32_t lane      = (warp % 4) * 32 + (threadIdx.x % 32);
        const uint32_t tmem_row  = tmem_addr_slot + ((warp % 4) * 32 << 16);
        const uint32_t col_base  = g * 168;
        const uint32_t col_extra = (g == 2) ? 508 : (col_base + 164);

        uint32_t w[4], r[4];
#pragma unroll
        for (uint32_t i = 0; i < 4; i++) {
            w[i] = (lane << 16) ^ (g << 12) ^ (col_base + i);
        }
        tcgen05::tmem_st_32x32b_x4(tmem_row + col_base, w);
        tcgen05::tmem_st_32x32b_x4(tmem_row + col_extra, w);
        tcgen05::tmem_st_wait();

        tcgen05::tmem_ld_32x32b_x4(r, tmem_row + col_base);
        tcgen05::tmem_ld_wait();
#pragma unroll
        for (uint32_t i = 0; i < 4; i++) {
            if (r[i] != w[i]) {
                atomicOr(err, 1u << g);
            }
        }
        tcgen05::tmem_ld_32x32b_x4(r, tmem_row + col_extra);
        tcgen05::tmem_ld_wait();
#pragma unroll
        for (uint32_t i = 0; i < 4; i++) {
            if (r[i] != w[i]) {
                atomicOr(err, 1u << (g + 4));
            }
        }

        tcgen05::tcgen05_fence_before_sync();
        tcgen05::arrive<1>(&bar_dealloc);
    }
    else if (warp == kMmaWarp) {
        tcgen05::wait(&bar_dealloc, 0);
        tcgen05::tcgen05_fence_after_sync();
        tcgen05::tmem_dealloc(tmem_addr_slot, kTmemCols);
    }
    // warps 13-15: exit
}

}  // namespace

int main()
{
    uint32_t* err_d;
    if (cudaMalloc(&err_d, sizeof(uint32_t)) != cudaSuccess) {
        printf("FAIL: cudaMalloc\n");
        return 1;
    }
    cudaMemset(err_d, 0, sizeof(uint32_t));

    // two back-to-back launches: the second re-allocs the columns the first
    // dealloc'ed, so a leaked alloc shows up as a hang/error here
    for (int rep = 0; rep < 2; rep++) {
        p0b_kernel<<<1, kThreads>>>(err_d);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        printf("FAIL: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }

    uint32_t err_h = 0;
    cudaMemcpy(&err_h, err_d, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err_h) {
        printf("FAIL: err mask 0x%x\n", err_h);
        return 1;
    }
    printf("PASS\n");
    return 0;
}
