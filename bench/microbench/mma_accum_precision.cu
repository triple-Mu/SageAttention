// SPDX-License-Identifier: Apache-2.0
// Accumulator-precision microbench for sm12x tensor-core mma paths.
//
// Method (arXiv 2512.07004 style): initialize the mma accumulator C to 2^k and
// add a single product of exactly 1.0. If the instruction keeps f fractional
// bits in its accumulator, the +1 survives while k <= f and is lost after.
// Reports the largest k with an exact result per instruction:
//   - s8s8s32 (sanity: exact up to 2^31)
//   - f8f8f32 mma.sync           (Ada keeps ~13 bits -> the "fp22" issue;
//                                 sm12x is expected to keep the full 23)
//   - f8f8f16 mma.sync           (fp16 accum: expect 10)
//   - QMMA.SF kind::mxf8f6f4     (block-scaled, sm_120a build only; the v2
//                                 premise is that this one is full fp32)
//
// Build (plain):  nvcc -O2 -std=c++17 -gencode arch=compute_120,code=sm_120  mma_accum_precision.cu -o precision_120
// Build (120a):   nvcc -O2 -std=c++17 -gencode arch=compute_120a,code=sm_120a -DSAGE_SM120A mma_accum_precision.cu -o precision_120a
// (also compiles/runs on sm_89/sm_90 for cross-checking the Ada/Hopper numbers)

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// e4m3 1.0 = sign 0, exp 7 (bias 7), mantissa 0 -> 0x38
#define E4M3_ONE 0x38u

// One warp; A(i,k) = 1.0 for all i,k held by each lane; B = 0 except B(0,0) = 1.0
// -> D(i,0) += 1.0. Lane 0 register d0 holds D(0,0).
__global__ void probe_f8f8f32(const float c_init, float *out)
{
  const int lane = threadIdx.x & 31;
  uint32_t a[4] = {0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u};
  uint32_t b[2] = {0u, 0u};
  if (lane == 0) b[0] = E4M3_ONE; // B(k=0, n=0)
  float d[4] = {c_init, c_init, c_init, c_init};
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  if (lane == 0) *out = d[0];
}

__global__ void probe_f8f8f16(const float c_init, float *out)
{
  const int lane = threadIdx.x & 31;
  uint32_t a[4] = {0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u};
  uint32_t b[2] = {0u, 0u};
  if (lane == 0) b[0] = E4M3_ONE;
  __half2 ci = __float2half2_rn(c_init);
  uint32_t d[2] = {*reinterpret_cast<uint32_t *>(&ci), *reinterpret_cast<uint32_t *>(&ci)};
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
      : "+r"(d[0]), "+r"(d[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  if (lane == 0) *out = __low2float(*reinterpret_cast<__half2 *>(&d[0]));
}

__global__ void probe_s8s8s32(const int c_init, int *out)
{
  const int lane = threadIdx.x & 31;
  uint32_t a[4] = {0x01010101u, 0x01010101u, 0x01010101u, 0x01010101u};
  uint32_t b[2] = {0u, 0u};
  if (lane == 0) b[0] = 1u;
  int d[4] = {c_init, c_init, c_init, c_init};
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  if (lane == 0) *out = d[0];
}

#ifdef SAGE_SM120A
// Block-scaled QMMA.SF with both scales = ue8m0 1.0 (biased exponent 127).
__global__ void probe_qmma_sf(const float c_init, float *out)
{
  const int lane = threadIdx.x & 31;
  uint32_t a[4] = {0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u};
  uint32_t b[2] = {0u, 0u};
  if (lane == 0) b[0] = E4M3_ONE;
  float d[4] = {c_init, c_init, c_init, c_init};
  uint32_t sfa = 0x7F7F7F7Fu, sfb = 0x7F7F7F7Fu; // ue8m0: 2^(127-127) = 1.0
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%11}, {0, 0};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "r"(sfa), "r"(sfb));
  if (lane == 0) *out = d[0];
}
#endif

template <typename LaunchF>
int measure_frac_bits(LaunchF launch, int max_k)
{
  float *d_out;
  cudaMalloc(&d_out, sizeof(float));
  int last_exact = -1;
  for (int k = 0; k <= max_k; k++) {
    const float c = static_cast<float>(1ull << k);
    launch(c, d_out);
    float r = 0.f;
    cudaMemcpy(&r, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    const double expect = static_cast<double>(c) + 1.0;
    if (static_cast<double>(r) == expect) last_exact = k;
    else break;
  }
  cudaFree(d_out);
  return last_exact;
}

int main()
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

  // s8s8s32 sanity
  {
    int *d_out; cudaMalloc(&d_out, sizeof(int));
    int last = -1;
    for (int k = 0; k <= 30; k++) {
      probe_s8s8s32<<<1, 32>>>(1 << k, d_out);
      int r = 0; cudaMemcpy(&r, d_out, sizeof(int), cudaMemcpyDeviceToHost);
      if (r == (1 << k) + 1) last = k; else break;
    }
    cudaFree(d_out);
    printf("s8s8s32   : exact through 2^%d (+1)  [expect 30: exact int32]\n", last);
  }

  int f32bits = measure_frac_bits([](float c, float *o) { probe_f8f8f32<<<1, 32>>>(c, o); }, 26);
  printf("f8f8f32   : keeps %d fractional bits  [Ada: ~13 (the fp22 issue); full fp32: 23]\n", f32bits);

  int f16bits = measure_frac_bits([](float c, float *o) { probe_f8f8f16<<<1, 32>>>(c, o); }, 15);
  printf("f8f8f16   : keeps %d fractional bits  [fp16 accum: 10]\n", f16bits);

#ifdef SAGE_SM120A
  int sfbits = measure_frac_bits([](float c, float *o) { probe_qmma_sf<<<1, 32>>>(c, o); }, 26);
  printf("QMMA.SF   : keeps %d fractional bits  [v2 GO requires 23]\n", sfbits);
#else
  printf("QMMA.SF   : (not built; use the sm_120a binary)\n");
#endif

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }
  printf("\nDecision: if f8f8f32 keeps 23 bits, sm120 default pv_accum_dtype='fp32' is exact;\n"
         "if QMMA.SF also keeps 23 bits at full rate (see mma_rate), v2 block-scale is GO.\n");
  return 0;
}
