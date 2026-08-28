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

// Device-only ptxas gate TU for qk_int_sv_f8_cuda_sm100.cu (torch-free):
// explicitly instantiates the MVP kernel for HEAD_DIM {64,128} x both PV
// paths (TS from TMEM / SS twin from smem) so every asm string, descriptor
// and TMEM access is machine-checked by ptxas at sm_100a and sm_110a.
//
//   nvcc -std=c++17 -O3 --use_fast_math -cubin -Xptxas -v \
//        -gencode arch=compute_100a,code=sm_100a  (and compute_110a/sm_110a)

#define SAGE_SM100_DEVICE_ONLY 1
#include "qk_int_sv_f8_cuda_sm100.cu"

// HD=128, TS PV path, causal + lse + v_scale, per-thread quant, fp16 out
template __global__ void qk_int8_sv_f8_attn_kernel_sm100<
    128u, 128u, 128u, 128u, QuantGranularity::kPerThread, QuantGranularity::kPerThread, half,
    MaskMode::kCausal, /*return_lse=*/true, /*fuse_v_scale=*/true, /*PV_FROM_SMEM=*/false>(
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const __grid_constant__ CUtensorMap, float *__restrict__, float *__restrict__,
    float *__restrict__, half *, float *__restrict__, uint32_t, uint32_t, uint32_t,
    const uint32_t, const uint32_t, const uint32_t, float);

// HD=128, SS twin PV path, non-causal, per-warp quant, bf16 out
template __global__ void qk_int8_sv_f8_attn_kernel_sm100<
    128u, 128u, 128u, 128u, QuantGranularity::kPerWarp, QuantGranularity::kPerWarp, nv_bfloat16,
    MaskMode::kNone, /*return_lse=*/false, /*fuse_v_scale=*/false, /*PV_FROM_SMEM=*/true>(
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const __grid_constant__ CUtensorMap, float *__restrict__, float *__restrict__,
    float *__restrict__, nv_bfloat16 *, float *__restrict__, uint32_t, uint32_t, uint32_t,
    const uint32_t, const uint32_t, const uint32_t, float);

// HD=64, TS PV path, non-causal + v_scale, per-warp quant, fp16 out
template __global__ void qk_int8_sv_f8_attn_kernel_sm100<
    128u, 128u, 128u, 64u, QuantGranularity::kPerWarp, QuantGranularity::kPerWarp, half,
    MaskMode::kNone, /*return_lse=*/false, /*fuse_v_scale=*/true, /*PV_FROM_SMEM=*/false>(
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const __grid_constant__ CUtensorMap, float *__restrict__, float *__restrict__,
    float *__restrict__, half *, float *__restrict__, uint32_t, uint32_t, uint32_t,
    const uint32_t, const uint32_t, const uint32_t, float);

// HD=64, SS twin PV path, causal + lse, per-thread quant, bf16 out
template __global__ void qk_int8_sv_f8_attn_kernel_sm100<
    128u, 128u, 128u, 64u, QuantGranularity::kPerThread, QuantGranularity::kPerThread, nv_bfloat16,
    MaskMode::kCausal, /*return_lse=*/true, /*fuse_v_scale=*/false, /*PV_FROM_SMEM=*/true>(
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const __grid_constant__ CUtensorMap, float *__restrict__, float *__restrict__,
    float *__restrict__, nv_bfloat16 *, float *__restrict__, uint32_t, uint32_t, uint32_t,
    const uint32_t, const uint32_t, const uint32_t, float);
