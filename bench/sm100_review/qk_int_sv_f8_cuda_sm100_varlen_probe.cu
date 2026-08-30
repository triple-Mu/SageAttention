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

// Device-only ptxas gate TU for the SAGE_VARLEN build of
// qk_int_sv_f8_sm100_impl.cuh (torch-free): the same four instantiations as
// qk_int_sv_f8_cuda_sm100_probe.cu, compiled with the packed cu_seqlens
// parameter list, so the varlen-only code paths (SeqlenInfo, the signed trip
// count, the zero-trip fill, the runtime mask trigger, the dead-row override)
// are machine-checked by ptxas at sm_100a and sm_110a.
//
// Not wired into any build or CI beyond test/test_ptxas_gate.py — compile it
// by hand from the repo root (nvcc >= 13.3, no GPU needed) and check that
// ptxas -v reports register usage for all four instantiations:
//
//   nvcc -std=c++17 -O3 --use_fast_math -cubin -Xptxas -v \
//        -gencode arch=compute_100a,code=sm_100a \
//        -I csrc/qattn -o /tmp/probe_varlen.cubin \
//        bench/sm100_review/qk_int_sv_f8_cuda_sm100_varlen_probe.cu
//
//   (repeat with arch=compute_110a,code=sm_110a)
//
// The explicit instantiations below must track the kernel's namespace and
// parameter list in csrc/qattn/qk_int_sv_f8_sm100_impl.cuh by hand.

#define SAGE_SM100_DEVICE_ONLY 1
#define SAGE_VARLEN 1
#define SAGEATTN_ARCH_NS sm100_varlen
#include "qk_int_sv_f8_sm100_impl.cuh"

namespace sage {
namespace sm100_varlen {

// HD=128, TS PV path, causal + lse + v_scale, per-thread quant, fp16 out
template __global__ void qk_int8_sv_f8_attn_kernel_sm100<
    128u, 128u, 128u, 128u, QuantGranularity::kPerThread, QuantGranularity::kPerThread, half,
    MaskMode::kCausal, /*return_lse=*/true, /*fuse_v_scale=*/true, /*PV_FROM_SMEM=*/false>(
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const __grid_constant__ CUtensorMap, const float *__restrict__, const float *__restrict__,
    const float *__restrict__, half *, float *__restrict__, const int64_t, const int64_t,
    uint32_t, const int32_t *__restrict__, const int32_t *__restrict__, const uint32_t,
    const uint32_t, const uint32_t, const uint32_t, float);

// HD=128, SS twin PV path, non-causal, per-warp quant, bf16 out
template __global__ void qk_int8_sv_f8_attn_kernel_sm100<
    128u, 128u, 128u, 128u, QuantGranularity::kPerWarp, QuantGranularity::kPerWarp, nv_bfloat16,
    MaskMode::kNone, /*return_lse=*/false, /*fuse_v_scale=*/false, /*PV_FROM_SMEM=*/true>(
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const __grid_constant__ CUtensorMap, const float *__restrict__, const float *__restrict__,
    const float *__restrict__, nv_bfloat16 *, float *__restrict__, const int64_t, const int64_t,
    uint32_t, const int32_t *__restrict__, const int32_t *__restrict__, const uint32_t,
    const uint32_t, const uint32_t, const uint32_t, float);

// HD=64, TS PV path, non-causal + v_scale, per-warp quant, fp16 out
template __global__ void qk_int8_sv_f8_attn_kernel_sm100<
    128u, 128u, 128u, 64u, QuantGranularity::kPerWarp, QuantGranularity::kPerWarp, half,
    MaskMode::kNone, /*return_lse=*/false, /*fuse_v_scale=*/true, /*PV_FROM_SMEM=*/false>(
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const __grid_constant__ CUtensorMap, const float *__restrict__, const float *__restrict__,
    const float *__restrict__, half *, float *__restrict__, const int64_t, const int64_t,
    uint32_t, const int32_t *__restrict__, const int32_t *__restrict__, const uint32_t,
    const uint32_t, const uint32_t, const uint32_t, float);

// HD=64, SS twin PV path, causal + lse, per-thread quant, bf16 out
template __global__ void qk_int8_sv_f8_attn_kernel_sm100<
    128u, 128u, 128u, 64u, QuantGranularity::kPerThread, QuantGranularity::kPerThread, nv_bfloat16,
    MaskMode::kCausal, /*return_lse=*/true, /*fuse_v_scale=*/false, /*PV_FROM_SMEM=*/true>(
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
    const __grid_constant__ CUtensorMap, const float *__restrict__, const float *__restrict__,
    const float *__restrict__, nv_bfloat16 *, float *__restrict__, const int64_t, const int64_t,
    uint32_t, const int32_t *__restrict__, const int32_t *__restrict__, const uint32_t,
    const uint32_t, const uint32_t, const uint32_t, float);

}  // namespace sm100_varlen
}  // namespace sage
