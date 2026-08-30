// SPDX-License-Identifier: Apache-2.0
// E1/P3 prescreen: device-only instantiation TU for the production sm90
// kernel (fwd_cuda.cu:267 -> qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf:
// per-thread quant, hd128, fuse_v_scale) causal/non-causal, so cuobjdump can
// measure the softmax XU-chain instruction area without a full build.
//
// Build (needs torch headers for the impl header's <torch/types.h>):
//   nvcc -std=c++17 -O3 --use_fast_math -cubin -arch=sm_90a \
//        -diag-suppress=174 -U__CUDA_NO_HALF_OPERATORS__ \
//        -U__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
//        -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr \
//        -I csrc/qattn -I csrc -I $TORCH/include -I $TORCH/include/torch/csrc/api/include \
//        bench/microbench/area_probe_sm90.cu -o area_sm90a.cubin

#include "qk_int_sv_f8_sm90_impl.cuh"

namespace sage {
namespace sm90 {

#define SM90_DENSE_PARAMS                                                                      \
    const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,                  \
        const __grid_constant__ CUtensorMap, const float *__restrict__,                        \
        const float *__restrict__, const float *__restrict__, half *, float *__restrict__,    \
        const int64_t, const int64_t, uint32_t, const uint32_t, const uint32_t,                \
        const uint32_t, float

// production path: hd128, per-thread, fuse_v_scale, fp16 out, causal
template __global__ void qk_int8_sv_f8_attn_kernel<
    64u, 128u, 128u, 128u, QuantGranularity::kPerThread, QuantGranularity::kPerThread, half,
    MaskMode::kCausal, /*return_lse=*/false, /*fuse_v_scale=*/true>(SM90_DENSE_PARAMS);

// same, non-causal
template __global__ void qk_int8_sv_f8_attn_kernel<
    64u, 128u, 128u, 128u, QuantGranularity::kPerThread, QuantGranularity::kPerThread, half,
    MaskMode::kNone, /*return_lse=*/false, /*fuse_v_scale=*/true>(SM90_DENSE_PARAMS);

}  // namespace sm90
}  // namespace sage
