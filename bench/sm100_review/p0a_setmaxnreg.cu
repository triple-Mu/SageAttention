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

// M0 ignition probe A: setmaxnreg on sm_100a with the exact warpgroup shape
// and budgets of qk_int_sv_f8_cuda_sm100_ws.cu (192/192/88/40, 512 threads,
// 1 CTA/SM). Each region holds a near-budget number of registers live (per
// value asm barriers + a reversed second use defeat rematerialization) and
// writes an exactly-representable checksum; the host validates every thread.
//
// Compile gate (no GPU needed):
//   nvcc -std=c++17 -O3 -cubin -Xptxas -v -arch=sm_100a \
//        -o /tmp/p0a.cubin bench/sm100_review/p0a_setmaxnreg.cu
//   expect: no C7508 ("'setmaxnreg' ignored"), 0 spills, entry 128 regs,
//   SASS USETMAXREG TRY_ALLOC 0xc0 / DEALLOC 0x58 / DEALLOC 0x28.
// Hardware run:
//   nvcc -std=c++17 -O3 -arch=sm_100a -o /tmp/p0a bench/sm100_review/p0a_setmaxnreg.cu
//   /tmp/p0a   (prints PASS/FAIL; exit code 0 on PASS)

#include <cuda/ptx>
#include <cstdio>
#include <stdint.h>

namespace {

constexpr int kThreads = 512;
// budgets must match qk_int_sv_f8_cuda_sm100_ws.cu (sum*128 == 64K)
constexpr int kRegsSoftmax = 192;
constexpr int kRegsCorr    = 88;
constexpr int kRegsOther   = 40;
static_assert(2 * 128 * kRegsSoftmax + 128 * kRegsCorr + 128 * kRegsOther == 64 * 1024, "");

// live-register load per region (leaves headroom for indices/pointers)
constexpr int kLiveSoftmax = 150;
constexpr int kLiveCorr    = 60;
constexpr int kLiveOther   = 24;

// Holds N floats live at once: each element is pinned by an asm barrier, and
// the second accumulation walks them in reverse so none can die early.
template<int N>
__device__ float hold_live(const float* in, uint32_t tid)
{
    float v[N];
#pragma unroll
    for (int i = 0; i < N; i++) {
        v[i] = in[tid + i * kThreads];
        asm volatile("" : "+f"(v[i]));  // opaque: no refetch, no fold
    }
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < N; i++) {
        s += v[i];
    }
#pragma unroll
    for (int i = N - 1; i >= 0; i--) {
        s += v[i] * 2.0f;
    }
    return s;
}

__global__ void __launch_bounds__(kThreads, 1) p0a_kernel(const float* in, float* out)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t wg  = tid / 128;

    float s;
    if (wg <= 1) {
        cuda::ptx::setmaxnreg_inc(cuda::ptx::n32_t<kRegsSoftmax>{});
        s = hold_live<kLiveSoftmax>(in, tid);
    }
    else if (wg == 2) {
        cuda::ptx::setmaxnreg_dec(cuda::ptx::n32_t<kRegsCorr>{});
        s = hold_live<kLiveCorr>(in, tid);
    }
    else {
        cuda::ptx::setmaxnreg_dec(cuda::ptx::n32_t<kRegsOther>{});
        // mma/load warps take different sub-branches like the real kernel
        const uint32_t warp = tid / 32;
        if (warp == 12 || warp == 13) {
            s = hold_live<kLiveOther>(in, tid);
        }
        else {
            s = hold_live<8>(in, tid);
        }
    }
    out[tid] = s;
}

}  // namespace

int main()
{
    const int n_in = kThreads * kLiveSoftmax;
    float*    in_h = new float[n_in];
    for (int i = 0; i < n_in; i++) {
        in_h[i] = float(i % 7);  // small ints: fp32 sums are exact
    }

    float *in_d, *out_d;
    if (cudaMalloc(&in_d, n_in * sizeof(float)) != cudaSuccess
        || cudaMalloc(&out_d, kThreads * sizeof(float)) != cudaSuccess) {
        printf("FAIL: cudaMalloc\n");
        return 1;
    }
    cudaMemcpy(in_d, in_h, n_in * sizeof(float), cudaMemcpyHostToDevice);

    p0a_kernel<<<1, kThreads>>>(in_d, out_d);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        printf("FAIL: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }

    float* out_h = new float[kThreads];
    cudaMemcpy(out_h, out_d, kThreads * sizeof(float), cudaMemcpyDeviceToHost);

    int bad = 0;
    for (int tid = 0; tid < kThreads; tid++) {
        const int wg   = tid / 128;
        const int warp = tid / 32;
        const int n    = (wg <= 1) ? kLiveSoftmax : (wg == 2) ? kLiveCorr : (warp <= 13) ? kLiveOther : 8;
        float     ref  = 0.f;
        for (int i = 0; i < n; i++) {
            ref += 3.0f * float((tid + i * kThreads) % 7);
        }
        if (out_h[tid] != ref) {
            if (bad < 8) {
                printf("tid %d: got %f want %f\n", tid, out_h[tid], ref);
            }
            bad++;
        }
    }
    if (bad) {
        printf("FAIL: %d mismatches\n", bad);
        return 1;
    }
    printf("PASS\n");
    return 0;
}
