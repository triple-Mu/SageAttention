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

// Device-only ptxas gate TU for qk_int_sv_f8_cuda_sm100_ws.cu (torch-free):
// explicitly instantiates the C1 warp-specialized kernel for HEAD_DIM
// {64,128} x mask/lse/gran/dtype combinations so every asm string, barrier
// protocol and TMEM access is machine-checked by ptxas at sm_100a/sm_110a.
// Only fuse_v_scale=true is instantiated: the launcher (and the
// SAGEATTN_SM100_WS switch) only reaches that variant.
//
// Not wired into any build or CI — compile it by hand from the repo root
// (nvcc >= 13.3, no GPU needed):
//
//   nvcc -std=c++17 -O3 --use_fast_math -cubin -Xptxas -v \
//        -gencode arch=compute_100a,code=sm_100a \
//        -I csrc/qattn -o /tmp/ws_probe.cubin \
//        bench/sm100_review/qk_int_sv_f8_cuda_sm100_ws_probe.cu
//
// Gate criteria (record in the M0 notes):
//   * no ptxas C7508 ("'setmaxnreg' ignored") — the kernel must keep
//     __launch_bounds__(512, 1) for the entry register count to be known;
//   * 0 spill stores/loads on every instantiation;
//   * SASS contains USETMAXREG TRY_ALLOC 0xc0 / DEALLOC 0x60 / DEALLOC 0x20
//     (= the 192/96/32 warpgroup budgets).
//
// The explicit instantiations below must track the kernel's namespace and
// parameter list in csrc/qattn/qk_int_sv_f8_cuda_sm100_ws.cu by hand.

#define SAGE_SM100_DEVICE_ONLY 1
#include "qk_int_sv_f8_cuda_sm100_ws.cu"

namespace sage {
namespace sm100 {

// HD=128, causal + lse, per-thread quant, fp16 out
template __global__ void qk_int8_sv_f8_attn_kernel_sm100_ws<
    128u, 128u, 512u, 128u, QuantGranularity::kPerThread, QuantGranularity::kPerThread, half,
    MaskMode::kCausal, /*return_lse=*/true, /*fuse_v_scale=*/true>(
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const __grid_constant__ CUtensorMap, const float *__restrict__, const float *__restrict__,
    const float *__restrict__, half *, float *__restrict__, const int64_t, const int64_t,
    uint32_t, const uint32_t, const uint32_t, const uint32_t, float);

// HD=128, non-causal, per-warp quant, bf16 out
template __global__ void qk_int8_sv_f8_attn_kernel_sm100_ws<
    128u, 128u, 512u, 128u, QuantGranularity::kPerWarp, QuantGranularity::kPerWarp, nv_bfloat16,
    MaskMode::kNone, /*return_lse=*/false, /*fuse_v_scale=*/true>(
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const __grid_constant__ CUtensorMap, const float *__restrict__, const float *__restrict__,
    const float *__restrict__, nv_bfloat16 *, float *__restrict__, const int64_t, const int64_t,
    uint32_t, const uint32_t, const uint32_t, const uint32_t, float);

// HD=64, non-causal, per-warp quant, fp16 out
template __global__ void qk_int8_sv_f8_attn_kernel_sm100_ws<
    128u, 128u, 512u, 64u, QuantGranularity::kPerWarp, QuantGranularity::kPerWarp, half,
    MaskMode::kNone, /*return_lse=*/false, /*fuse_v_scale=*/true>(
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const __grid_constant__ CUtensorMap, const float *__restrict__, const float *__restrict__,
    const float *__restrict__, half *, float *__restrict__, const int64_t, const int64_t,
    uint32_t, const uint32_t, const uint32_t, const uint32_t, float);

// HD=64, causal + lse, per-thread quant, bf16 out
template __global__ void qk_int8_sv_f8_attn_kernel_sm100_ws<
    128u, 128u, 512u, 64u, QuantGranularity::kPerThread, QuantGranularity::kPerThread, nv_bfloat16,
    MaskMode::kCausal, /*return_lse=*/true, /*fuse_v_scale=*/true>(
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const __grid_constant__ CUtensorMap, const float *__restrict__, const float *__restrict__,
    const float *__restrict__, nv_bfloat16 *, float *__restrict__, const int64_t, const int64_t,
    uint32_t, const uint32_t, const uint32_t, const uint32_t, float);

}  // namespace sm100
}  // namespace sage
