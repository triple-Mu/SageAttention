// SPDX-License-Identifier: Apache-2.0
// E1/P3 prescreen: device-only instantiation TU so cuobjdump can measure the
// softmax XU-chain instruction area of the production sm89 kernels without a
// full extension build. Instantiates the fwd_cuda.cu:249 default path (accum
// f16 + inst_buf + fuse_v_scale, per-thread quant, hd128) causal/non-causal,
// plus the accum-f32 flavour (fwd_cuda.cu:235) for sensitivity.
//
// Build (needs torch headers for the impl header's <torch/types.h>):
//   nvcc -std=c++17 -O3 --use_fast_math -cubin -arch=sm_89 \
//        -diag-suppress=174 -U__CUDA_NO_HALF_OPERATORS__ \
//        -U__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
//        -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr \
//        -I csrc/qattn -I csrc -I $TORCH/include -I $TORCH/include/torch/csrc/api/include \
//        bench/microbench/area_probe_sm89.cu -o area_sm89.cubin

#include "qk_int_sv_f8_cuda_sm89.cuh"

namespace sage {
namespace sm89 {

#define SM89_DENSE_PARAMS                                                                          \
    const int8_t *__restrict__, const int8_t *__restrict__, const int8_t *__restrict__,           \
        half *__restrict__, float *__restrict__, const float *__restrict__,                       \
        const float *__restrict__, const float *__restrict__, const float *__restrict__,          \
        const uint32_t, const uint32_t, const uint32_t, const int64_t, const uint32_t,            \
        const int64_t, const int64_t, const uint32_t, const int64_t, const int64_t,               \
        const int64_t, const uint32_t, const int64_t, const uint32_t, const int64_t, float

// default path: accum f16 + inst_buf + fuse_v_scale, per-thread, hd128, causal
template __global__ void qk_int_sv_f8_attn_kernel<
    128u, 64u, 32u, 64u, 128u, DataType::kInt8, QuantGranularity::kPerThread,
    QuantGranularity::kPerThread, float, /*use_inst_buf=*/true, half, ComputeUnit::kCudaCore,
    MaskMode::kCausal, /*return_lse=*/false, /*fuse_v_scale=*/true, /*fuse_v_mean=*/false,
    /*use_pv_fp16_accum=*/true>(SM89_DENSE_PARAMS);

// same, non-causal
template __global__ void qk_int_sv_f8_attn_kernel<
    128u, 64u, 32u, 64u, 128u, DataType::kInt8, QuantGranularity::kPerThread,
    QuantGranularity::kPerThread, float, /*use_inst_buf=*/true, half, ComputeUnit::kCudaCore,
    MaskMode::kNone, /*return_lse=*/false, /*fuse_v_scale=*/true, /*fuse_v_mean=*/false,
    /*use_pv_fp16_accum=*/true>(SM89_DENSE_PARAMS);

// accum f32 flavour (pv fp32+fp32), causal
template __global__ void qk_int_sv_f8_attn_kernel<
    128u, 64u, 32u, 64u, 128u, DataType::kInt8, QuantGranularity::kPerThread,
    QuantGranularity::kPerThread, float, /*use_inst_buf=*/true, half, ComputeUnit::kCudaCore,
    MaskMode::kCausal, /*return_lse=*/false, /*fuse_v_scale=*/true, /*fuse_v_mean=*/false,
    /*use_pv_fp16_accum=*/false>(SM89_DENSE_PARAMS);

}  // namespace sm89
}  // namespace sage
