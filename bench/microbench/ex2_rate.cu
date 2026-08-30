// SPDX-License-Identifier: Apache-2.0
// E1 prescreen: sustained per-thread throughput of the softmax XU chain
// candidates, one kernel per instruction so nothing cross-contaminates:
//   ex2_f32    ex2.approx.ftz.f32   (current softmax exp2, 1 elem/instr)
//   ex2_f16x2  ex2.approx.f16x2     (E1 candidate, 2 elems/instr)
//   cvt_f16x2  cvt.rn.f16x2.f32     (the f32->half2 pack E1 inserts before ex2)
//   ffma_f32   plain f32 FFMA       (FMA-pipe reference)
//
// Timing method follows mma_rate.cu: kChains independent dependency chains
// per thread, all in registers, results folded into an impossible compare so
// nothing is dead-code eliminated. Rates are normalized by prop.clockRate;
// boost drift shifts absolute numbers, not the ratios the go/no-go uses.
//
// Build (run, sm86):   nvcc -O3 -std=c++17 -arch=sm_86 ex2_rate.cu -o ex2_rate_86
// Legality check only: nvcc -O3 -std=c++17 -cubin -arch=sm_89   ex2_rate.cu -o ex2_rate.sm89.cubin
//                      nvcc -O3 -std=c++17 -cubin -arch=sm_90a  ex2_rate.cu -o ex2_rate.sm90a.cubin
//                      nvcc -O3 -std=c++17 -cubin -arch=sm_100a ex2_rate.cu -o ex2_rate.sm100a.cubin

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

constexpr int kIters = 16384;
constexpr int kChains = 8;
constexpr int kThreadsPerBlock = 256;
constexpr int kBlocksPerSm = 6;  // 1536 threads = sm86 max residency

__global__ void rate_ex2_f32(float *sink)
{
  float y[kChains];
#pragma unroll
  for (int c = 0; c < kChains; c++) y[c] = 0.5f + 0.001f * c;
  for (int i = 0; i < kIters; i++) {
#pragma unroll
    for (int c = 0; c < kChains; c++)
      asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(y[c]));
  }
  float acc = 0.f;
#pragma unroll
  for (int c = 0; c < kChains; c++) acc += y[c];
  if (acc == 12345.678f) *sink = acc;  // never true; defeats DCE
}

__global__ void rate_ex2_f16x2(float *sink)
{
  uint32_t y[kChains];
#pragma unroll
  for (int c = 0; c < kChains; c++) y[c] = 0x38003800u + c;  // ~(0.5, 0.5) packed
  for (int i = 0; i < kIters; i++) {
#pragma unroll
    for (int c = 0; c < kChains; c++)
      asm volatile("ex2.approx.f16x2 %0, %0;" : "+r"(y[c]));
  }
  uint32_t acc = 0;
#pragma unroll
  for (int c = 0; c < kChains; c++) acc ^= y[c];
  if (acc == 0xDEADBEEFu) *sink = 1.f;
}

__global__ void rate_cvt_f16x2(float *sink)
{
  uint32_t y[kChains];
#pragma unroll
  for (int c = 0; c < kChains; c++) y[c] = 0x3F000000u + c;
  const float b = 1.5f;
  for (int i = 0; i < kIters; i++) {
#pragma unroll
    for (int c = 0; c < kChains; c++) {
      // chain via bit-reinterpret of the packed result (free, no extra instr)
      float a = __uint_as_float(y[c]);
      asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(y[c]) : "f"(a), "f"(b));
    }
  }
  uint32_t acc = 0;
#pragma unroll
  for (int c = 0; c < kChains; c++) acc ^= y[c];
  if (acc == 0xDEADBEEFu) *sink = 1.f;
}

__global__ void rate_ffma_f32(float *sink)
{
  float y[kChains];
#pragma unroll
  for (int c = 0; c < kChains; c++) y[c] = 0.5f + 0.001f * c;
  const float a = 0.9999f, b = 1e-7f;
  for (int i = 0; i < kIters; i++) {
#pragma unroll
    for (int c = 0; c < kChains; c++)
      asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(y[c]) : "f"(a), "f"(b));
  }
  float acc = 0.f;
#pragma unroll
  for (int c = 0; c < kChains; c++) acc += y[c];
  if (acc == 12345.678f) *sink = acc;
}

template <typename K>
void run(K kernel, const char *name, int num_sms, double clock_ghz, int elems_per_instr)
{
  float *sink;
  cudaMalloc(&sink, sizeof(float));
  dim3 grid(num_sms * kBlocksPerSm), block(kThreadsPerBlock);
  kernel<<<grid, block>>>(sink);  // warmup
  cudaDeviceSynchronize();
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);
  kernel<<<grid, block>>>(sink);
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, t0, t1);
  const double instrs =
      static_cast<double>(grid.x) * block.x * kChains * static_cast<double>(kIters);
  const double ginstr = instrs / (ms * 1e-3) / 1e9;
  const double instr_per_sm_clk = instrs / (ms * 1e-3) / num_sms / (clock_ghz * 1e9);
  printf("%-10s: %8.3f ms  %8.1f Ginstr/s  %6.2f instr/SM/clk  %6.2f elem/SM/clk\n",
         name, ms, ginstr, instr_per_sm_clk, instr_per_sm_clk * elems_per_instr);
  cudaFree(sink);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
}

int main()
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int clock_khz = 0;  // cudaDeviceProp::clockRate is gone in CUDA 13
  cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
  const double clock_ghz = clock_khz / 1e6;
  printf("device: %s (sm_%d%d, %d SMs, %.2f GHz boost)\n",
         prop.name, prop.major, prop.minor, prop.multiProcessorCount, clock_ghz);

  run(rate_ffma_f32, "ffma_f32", prop.multiProcessorCount, clock_ghz, 1);
  run(rate_ex2_f32, "ex2_f32", prop.multiProcessorCount, clock_ghz, 1);
  run(rate_ex2_f16x2, "ex2_f16x2", prop.multiProcessorCount, clock_ghz, 2);
  run(rate_cvt_f16x2, "cvt_f16x2", prop.multiProcessorCount, clock_ghz, 2);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    return 1;
  }
  return 0;
}
