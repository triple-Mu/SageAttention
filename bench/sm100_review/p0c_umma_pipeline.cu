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

// M0 ignition probe C: the ws kernel's mma_s pipeline in isolation. A single
// elected thread of the mma warp issues the i8 QK chain and commits to a
// full mbarrier; 128 consumer threads (one warpgroup) wait parity, drain S
// from TMEM (4 x 32-column tcgen05.ld like the softmax), validate every
// element against a recomputed reference, and arrive on the empty mbarrier
// (count 128), which the mma thread waits on before re-issuing - repeated
// for kIters phase flips. Q/K are hand-packed into the 128B-swizzled K-major
// smem layout (the qk_int_sv_f8_cuda_sm100.cu PV_FROM_SMEM word formula),
// so the generic-proxy fill needs fence.proxy.async.shared::cta before the
// MMA reads it (the SS-twin hang lesson; see tcgen05::fence_async_shared).
//
// Compile gate (no GPU needed):
//   nvcc -std=c++17 -O3 -cubin -Xptxas -v -arch=sm_100a \
//        -I csrc -o /tmp/p0c.cubin bench/sm100_review/p0c_umma_pipeline.cu
// Hardware run:
//   nvcc -std=c++17 -O3 -arch=sm_100a -I csrc -o /tmp/p0c bench/sm100_review/p0c_umma_pipeline.cu
//   /tmp/p0c   (prints PASS/FAIL; a hang means the commit/parity machinery
//               or the smem fill fence is broken)

#include <cstdio>
#include <stdint.h>

#include "tcgen05.cuh"

namespace {

constexpr int      kThreads  = 160;  // warps 0-3 consumers, warp 4 mma
constexpr int      kMmaWarp  = 4;
constexpr uint32_t kM        = 128;
constexpr uint32_t kN        = 128;
constexpr uint32_t kK        = 128;
constexpr uint32_t kIters    = 8;
constexpr uint32_t kTmemCols = 128;  // S only

__host__ __device__ int8_t q_val(uint32_t m, uint32_t k)
{
    return int8_t(int((m + 2 * k) % 5) - 2);
}

__host__ __device__ int8_t k_val(uint32_t n, uint32_t k)
{
    return int8_t(int((3 * n + k) % 7) - 3);
}

// pack 4 int8 elements into the 128B-swizzle K-major word slot (word w of
// row r lands at word ((w/4 ^ r%8)*4 + w%4); same formula as the sm100
// kernel's PV_FROM_SMEM staging)
__device__ void fill_swizzled(int8_t* smem, uint32_t word_idx, bool is_q)
{
    const uint32_t r = word_idx / 32;  // row (128B = 32 words per row)
    const uint32_t w = word_idx % 32;
    uint32_t       v = 0;
#pragma unroll
    for (uint32_t b = 0; b < 4; b++) {
        const int8_t e = is_q ? q_val(r, 4 * w + b) : k_val(r, 4 * w + b);
        v |= uint32_t(uint8_t(e)) << (8 * b);
    }
    reinterpret_cast<uint32_t*>(smem + r * kK)[((w >> 2) ^ (r & 7)) << 2 | (w & 3)] = v;
}

__global__ void __launch_bounds__(kThreads, 1) p0c_kernel(const int32_t* __restrict__ ref, uint32_t* err)
{
    __shared__ __align__(1024) int8_t sQ[kM * kK];
    __shared__ __align__(1024) int8_t sK[kN * kK];
    __shared__ __align__(8) uint64_t  bar_full;
    __shared__ __align__(8) uint64_t  bar_empty;
    __shared__ __align__(8) uint64_t  bar_done;  // one-shot: consumers finished
    __shared__ __align__(4) uint32_t  tmem_addr_slot;

    const uint32_t warp = threadIdx.x / 32;

    if (threadIdx.x == 0) {
        tcgen05::init_barrier(&bar_full, 1);
        tcgen05::init_barrier(&bar_empty, 128);
        tcgen05::init_barrier(&bar_done, 128);
    }
    for (uint32_t idx = threadIdx.x; idx < kM * kK / 4; idx += kThreads) {
        fill_swizzled(sQ, idx, true);
        fill_swizzled(sK, idx, false);
    }
    // generic-proxy stores feeding tcgen05.mma: fence before the CTA barrier
    tcgen05::fence_async_shared();
    if (warp == kMmaWarp) {
        tcgen05::tmem_alloc(&tmem_addr_slot, kTmemCols);
        tcgen05::tmem_relinquish();
    }
    __syncthreads();

    constexpr uint32_t kSbo     = 8 * kK;  // 1024B: 8-row swizzle-atom pitch
    constexpr uint32_t idesc_qk = tcgen05::make_instr_desc<kM,
                                                           kN,
                                                           tcgen05::kMmaFmtS8,
                                                           tcgen05::kMmaFmtS8,
                                                           tcgen05::kMmaCFmtS32,
                                                           /*AMajorK=*/true,
                                                           /*BMajorK=*/true>();

    if (warp == kMmaWarp) {
        if (tcgen05::elect_one()) {
            for (uint32_t iter = 0; iter < kIters; iter++) {
                if (iter > 0) {
                    tcgen05::wait(&bar_empty, (iter - 1) & 1);  // consumers drained S
                    tcgen05::tcgen05_fence_after_sync();
                }
#pragma unroll
                for (uint32_t k_it = 0; k_it < kK / 32; k_it++) {
                    const uint64_t desc_q =
                        tcgen05::make_smem_desc_sm100<tcgen05::SmemSwizzleMode::kSwizzle128B, tcgen05::kKMajorLBO,
                                                      kSbo>(&sQ[k_it * 32]);
                    const uint64_t desc_k =
                        tcgen05::make_smem_desc_sm100<tcgen05::SmemSwizzleMode::kSwizzle128B, tcgen05::kKMajorLBO,
                                                      kSbo>(&sK[k_it * 32]);
                    tcgen05::mma_i8_ss(tmem_addr_slot, desc_q, desc_k, idesc_qk, k_it > 0);
                }
                tcgen05::tcgen05_commit(&bar_full);
            }
        }
    }
    else {
        const uint32_t m        = threadIdx.x;  // consumer thread = S row = TMEM lane
        const uint32_t tmem_row = tmem_addr_slot + ((warp % 4) * 32 << 16);
        for (uint32_t iter = 0; iter < kIters; iter++) {
            tcgen05::wait(&bar_full, iter & 1);
#pragma unroll
            for (uint32_t c = 0; c < kN / 32; c++) {
                uint32_t rs[32];
                tcgen05::tmem_ld_32x32b_x32(rs, tmem_row + c * 32);
                tcgen05::tmem_ld_wait();
#pragma unroll
                for (uint32_t jj = 0; jj < 32; jj++) {
                    const uint32_t n = c * 32 + jj;
                    if (int32_t(rs[jj]) != ref[m * kN + n]) {
                        atomicOr(err, 1u << iter);
                    }
                }
            }
            tcgen05::tcgen05_fence_before_sync();
            tcgen05::arrive<1>(&bar_empty);
        }
        tcgen05::arrive<1>(&bar_done);
    }

    // shutdown: dedicated one-shot barrier (a parity wait on the per-iter
    // bar_empty would alias an earlier odd phase and dealloc too early)
    if (warp == kMmaWarp) {
        tcgen05::wait(&bar_done, 0);
        tcgen05::tcgen05_fence_after_sync();
        tcgen05::tmem_dealloc(tmem_addr_slot, kTmemCols);
    }
}

}  // namespace

int main()
{
    // reference S on the host (small ints, exact)
    int32_t* ref_h = new int32_t[kM * kN];
    for (uint32_t m = 0; m < kM; m++) {
        for (uint32_t n = 0; n < kN; n++) {
            int32_t acc = 0;
            for (uint32_t k = 0; k < kK; k++) {
                acc += int32_t(q_val(m, k)) * int32_t(k_val(n, k));
            }
            ref_h[m * kN + n] = acc;
        }
    }

    int32_t*  ref_d;
    uint32_t* err_d;
    if (cudaMalloc(&ref_d, kM * kN * sizeof(int32_t)) != cudaSuccess
        || cudaMalloc(&err_d, sizeof(uint32_t)) != cudaSuccess) {
        printf("FAIL: cudaMalloc\n");
        return 1;
    }
    cudaMemcpy(ref_d, ref_h, kM * kN * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemset(err_d, 0, sizeof(uint32_t));

    p0c_kernel<<<1, kThreads>>>(ref_d, err_d);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        printf("FAIL: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }

    uint32_t err_h = 0;
    cudaMemcpy(&err_h, err_d, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err_h) {
        printf("FAIL: iter mask 0x%x\n", err_h);
        return 1;
    }
    printf("PASS\n");
    return 0;
}
