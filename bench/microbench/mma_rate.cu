// SPDX-License-Identifier: Apache-2.0
// Sustained tensor-core mma throughput on sm12x, per instruction:
// s8s8s32 / f8f8f32 / f8f8f16 / QMMA.SF (sm_120a build).
//
// Each warp iterates 8 independent accumulation chains fully in registers
// (no memory traffic in the timed loop). FLOPs = 2*M*N*K per mma.
//
// Build (plain):  nvcc -O3 -std=c++17 -gencode arch=compute_120,code=sm_120  mma_rate.cu -o rate_120
// Build (120a):   nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -DSAGE_SM120A mma_rate.cu -o rate_120a

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

constexpr int kIters = 16384;
constexpr int kChains = 8;
constexpr int kWarpsPerBlock = 8;

#define MMA_F8F8F32(d, a, b)                                                     \
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "           \
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"        \
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                  \
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]))

#define MMA_F8F8F16(d, a, b)                                                     \
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "           \
               "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"                     \
               : "+r"(d[0]), "+r"(d[1])                                          \
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]))

#define MMA_S8S8S32(d, a, b)                                                     \
  asm volatile("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "     \
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"        \
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])                  \
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]))

#define MMA_QMMA_SF(d, a, b, sfa, sfb)                                           \
  asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X"      \
               ".m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "                     \
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "         \
               "{%10}, {0, 0}, {%11}, {0, 0};\n"                                \
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                  \
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), \
                 "r"(sfa), "r"(sfb))

__global__ void rate_f8f8f32(float *sink)
{
  uint32_t a[4] = {0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u};
  uint32_t b[2] = {0x38383838u, 0x38383838u};
  float d[kChains][4] = {};
  for (int i = 0; i < kIters; i++) {
#pragma unroll
    for (int c = 0; c < kChains; c++) MMA_F8F8F32(d[c], a, b);
  }
  float acc = 0.f;
#pragma unroll
  for (int c = 0; c < kChains; c++) acc += d[c][0] + d[c][1] + d[c][2] + d[c][3];
  if (acc == 12345.678f) *sink = acc; // never true; defeats DCE
}

__global__ void rate_f8f8f16(float *sink)
{
  uint32_t a[4] = {0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u};
  uint32_t b[2] = {0x38383838u, 0x38383838u};
  uint32_t d[kChains][2] = {};
  for (int i = 0; i < kIters; i++) {
#pragma unroll
    for (int c = 0; c < kChains; c++) MMA_F8F8F16(d[c], a, b);
  }
  uint32_t acc = 0;
#pragma unroll
  for (int c = 0; c < kChains; c++) acc ^= d[c][0] ^ d[c][1];
  if (acc == 0xDEADBEEFu) *sink = 1.f;
}

__global__ void rate_s8s8s32(float *sink)
{
  uint32_t a[4] = {0x01010101u, 0x01010101u, 0x01010101u, 0x01010101u};
  uint32_t b[2] = {0x01010101u, 0x01010101u};
  int d[kChains][4] = {};
  for (int i = 0; i < kIters; i++) {
#pragma unroll
    for (int c = 0; c < kChains; c++) MMA_S8S8S32(d[c], a, b);
  }
  int acc = 0;
#pragma unroll
  for (int c = 0; c < kChains; c++) acc ^= d[c][0] ^ d[c][3];
  if (acc == 0x7FFFFFFF) *sink = 1.f;
}

#ifdef SAGE_SM120A
__global__ void rate_qmma_sf(float *sink)
{
  uint32_t a[4] = {0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u};
  uint32_t b[2] = {0x38383838u, 0x38383838u};
  uint32_t sfa = 0x7F7F7F7Fu, sfb = 0x7F7F7F7Fu;
  float d[kChains][4] = {};
  for (int i = 0; i < kIters; i++) {
#pragma unroll
    for (int c = 0; c < kChains; c++) MMA_QMMA_SF(d[c], a, b, sfa, sfb);
  }
  float acc = 0.f;
#pragma unroll
  for (int c = 0; c < kChains; c++) acc += d[c][0] + d[c][3];
  if (acc == 12345.678f) *sink = acc;
}
#endif

template <typename K>
double run(K kernel, const char *name, int num_sms)
{
  float *sink;
  cudaMalloc(&sink, sizeof(float));
  dim3 grid(num_sms * 4), block(32 * kWarpsPerBlock);
  kernel<<<grid, block>>>(sink); // warmup
  cudaDeviceSynchronize();
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0);
  kernel<<<grid, block>>>(sink);
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, t0, t1);
  const double mmas = static_cast<double>(grid.x) * kWarpsPerBlock * kChains * kIters;
  const double flops = mmas * 2.0 * 16 * 8 * 32;
  const double tflops = flops / (ms * 1e-3) / 1e12;
  printf("%-10s: %8.1f T%s\n", name, tflops, name[0] == 's' ? "OPS" : "FLOPS");
  cudaFree(sink);
  cudaEventDestroy(t0); cudaEventDestroy(t1);
  return tflops;
}

int main()
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("device: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

  run(rate_s8s8s32, "s8s8s32", prop.multiProcessorCount);
  double f32r = run(rate_f8f8f32, "f8f8f32", prop.multiProcessorCount);
  double f16r = run(rate_f8f8f16, "f8f8f16", prop.multiProcessorCount);
#ifdef SAGE_SM120A
  double sfr = run(rate_qmma_sf, "qmma.sf", prop.multiProcessorCount);
  printf("\nv2 GO if qmma.sf >= 1.7x f8f8f32-fp32acc: ratio = %.2f\n", sfr / f32r);
#else
  printf("\n(QMMA.SF needs the sm_120a binary)\n");
#endif
  printf("f8f8f16 / f8f8f32 ratio = %.2f  [Ada was ~2.0; decides how costly plain-fp32 default is]\n", f16r / f32r);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }
  return 0;
}
