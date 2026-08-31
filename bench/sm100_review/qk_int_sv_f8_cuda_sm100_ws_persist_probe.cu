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

// Device-only ptxas gate TU for qk_int_sv_f8_cuda_sm100_ws_persist.cu
// (torch-free): same 4 instantiations and gate criteria as the ws probe
// (qk_int_sv_f8_cuda_sm100_ws_probe.cu), against the persistent kernel.
//
// Not wired into any build or CI — compile it by hand from the repo root
// (nvcc >= 13.3, no GPU needed):
//
//   nvcc -std=c++17 -O3 --use_fast_math -cubin -Xptxas -v \
//        -gencode arch=compute_100a,code=sm_100a \
//        -I csrc/qattn -o /tmp/ws_persist_probe.cubin \
//        bench/sm100_review/qk_int_sv_f8_cuda_sm100_ws_persist_probe.cu
//
// Gate criteria (C1_DESIGN.md section 13):
//   * no ptxas C7508 ("'setmaxnreg' ignored");
//   * 0 spill stores/loads, 0 stack on every instantiation;
//   * 128-register entry count;
//   * SASS contains USETMAXREG TRY_ALLOC 0xc0 / DEALLOC 0x58 / DEALLOC 0x28
//     (= the 192/88/40 warpgroup budgets).
//
// The explicit instantiations below must track the kernel's namespace and
// parameter list in csrc/qattn/qk_int_sv_f8_cuda_sm100_ws_persist.cu by hand.

#define SAGE_SM100_DEVICE_ONLY 1
#include "qk_int_sv_f8_cuda_sm100_ws_persist.cu"

namespace sage {
namespace sm100 {

// HD=128, causal + lse, per-thread quant, fp16 out
template __global__ void qk_int8_sv_f8_attn_kernel_sm100_ws_persist<
    128u, 128u, 512u, 128u, QuantGranularity::kPerThread, QuantGranularity::kPerThread, half,
    MaskMode::kCausal, /*return_lse=*/true, /*fuse_v_scale=*/true>(
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const float *__restrict__, const float *__restrict__, const float *__restrict__,
    float *__restrict__, const uint32_t, const uint32_t, const uint32_t, const uint32_t,
    const uint32_t, float);

// HD=128, non-causal, per-warp quant, bf16 out
template __global__ void qk_int8_sv_f8_attn_kernel_sm100_ws_persist<
    128u, 128u, 512u, 128u, QuantGranularity::kPerWarp, QuantGranularity::kPerWarp, nv_bfloat16,
    MaskMode::kNone, /*return_lse=*/false, /*fuse_v_scale=*/true>(
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const float *__restrict__, const float *__restrict__, const float *__restrict__,
    float *__restrict__, const uint32_t, const uint32_t, const uint32_t, const uint32_t,
    const uint32_t, float);

// HD=64, non-causal, per-warp quant, fp16 out
template __global__ void qk_int8_sv_f8_attn_kernel_sm100_ws_persist<
    128u, 128u, 512u, 64u, QuantGranularity::kPerWarp, QuantGranularity::kPerWarp, half,
    MaskMode::kNone, /*return_lse=*/false, /*fuse_v_scale=*/true>(
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const float *__restrict__, const float *__restrict__, const float *__restrict__,
    float *__restrict__, const uint32_t, const uint32_t, const uint32_t, const uint32_t,
    const uint32_t, float);

// HD=64, causal + lse, per-thread quant, bf16 out
template __global__ void qk_int8_sv_f8_attn_kernel_sm100_ws_persist<
    128u, 128u, 512u, 64u, QuantGranularity::kPerThread, QuantGranularity::kPerThread, nv_bfloat16,
    MaskMode::kCausal, /*return_lse=*/true, /*fuse_v_scale=*/true>(
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const float *__restrict__, const float *__restrict__, const float *__restrict__,
    float *__restrict__, const uint32_t, const uint32_t, const uint32_t, const uint32_t,
    const uint32_t, float);

}  // namespace sm100
}  // namespace sage
