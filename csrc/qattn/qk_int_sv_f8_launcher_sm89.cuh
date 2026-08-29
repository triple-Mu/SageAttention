/*
 * Copyright (c) 2024 by SageAttention team.
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

#pragma once
#include "launch_utils.cuh"
#include "qk_int_sv_f8_cuda_sm89.cuh"

// SAGEATTN_ARCH_NS (default sm89) is defined by qk_int_sv_f8_cuda_sm89.cuh;
// the launcher template must live in the same per-arch namespace as the
// kernel template it instantiates.
namespace sage {
namespace SAGEATTN_ARCH_NS {

// Single templated launcher body shared by every surviving sm89 / sm120
// qk_int8_sv_f8 variant (all of them fuse the V scale):
//   USE_INST_BUF  - inst_buf PV accumulation kernels
//   FUSE_V_MEAN   - fuse the +V_mean epilogue (value_mean_opt must be set)
//   USE_FP16_ACCUM - fp16 short-term PV accumulator (inst_buf variants only)
// The thin sm89_*.cu / sm120_*.cu TUs bind the public function names to
// instantiations of this template; each sm120 TU stays byte-identical to its
// sm89 sibling except for the attn_cuda_*.h include line (sed-regenerable).
template<bool USE_INST_BUF, bool FUSE_V_MEAN, bool USE_FP16_ACCUM>
torch::Tensor qk_int8_sv_f8_fuse_v_scale_attn_launcher_sm89(torch::Tensor        query,
                                                            torch::Tensor        key,
                                                            torch::Tensor        value,
                                                            torch::Tensor        output,
                                                            torch::Tensor        query_scale,
                                                            torch::Tensor        key_scale,
                                                            torch::Tensor        value_scale,
                                                            const torch::Tensor* value_mean_opt,
                                                            int                  tensor_layout,
                                                            int                  is_causal,
                                                            int                  qk_quant_gran,
                                                            float                sm_scale,
                                                            int                  return_lse)
{
    const c10::cuda::CUDAGuard device_guard(query.device());

    QKVLayout qkv = qkv_layout_parse<QKVFamily::kSVF8CudaCore>(
        query, key, value, output, query_scale, key_scale, &value_scale, value_mean_opt, tensor_layout, return_lse);
    SAGEATTN_QKV_LAYOUT_LOCALS_FP8(qkv);

    auto         out_dtype = output.scalar_type();
    cudaStream_t stream    = at::cuda::getCurrentCUDAStream();

    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
            DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
                DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
                    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(out_dtype, DTypeOut, {
                        constexpr int CTA_Q  = 128;
                        constexpr int CTA_K  = 64;
                        constexpr int WARP_Q = 32;
                        constexpr int WARP_K = 64;

                        TORCH_CHECK(value.size(0) == batch_size,
                                    "value.size(0) (",
                                    value.size(0),
                                    ") == batch_size (",
                                    batch_size,
                                    ") failed");
                        TORCH_CHECK(value.size(3) >= div_ceil(kv_len, CTA_K) * CTA_K,
                                    "value.size(3) (",
                                    value.size(3),
                                    ") >= div_ceil(kv_len, CTA_K) * CTA_K (",
                                    div_ceil(kv_len, CTA_K) * CTA_K,
                                    ") failed, kv_len=",
                                    kv_len,
                                    " CTA_K=",
                                    CTA_K);

                        constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

                        if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp)) {
                            CHECK_SHAPE(
                                query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q));
                            CHECK_SHAPE(
                                key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K));
                        }
                        else if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread)) {
                            CHECK_SHAPE(
                                query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q) * 8);
                            CHECK_SHAPE(
                                key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K) * 4);
                        }
                        else {
                            static_assert(QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp)
                                              || QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread),
                                          "Unsupported quantization granularity");
                        }

                        CHECK_SHAPE(value_scale, batch_size, num_kv_heads, head_dim);
                        if constexpr (FUSE_V_MEAN) {
                            const torch::Tensor& value_mean = *value_mean_opt;
                            CHECK_SHAPE(value_mean, batch_size, num_kv_heads, head_dim);
                        }

                        //                                     smem_Q                                     smem_K smem_V
                        //                                     smem_O
                        size_t smem_max = std::max(CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t)
                                                       + CTA_K * HEAD_DIM * sizeof(int8_t),
                                                   CTA_Q * HEAD_DIM * sizeof(half));

                        auto kernel_func = qk_int_sv_f8_attn_kernel<CTA_Q,
                                                                    CTA_K,
                                                                    WARP_Q,
                                                                    WARP_K,
                                                                    HEAD_DIM,
                                                                    DataType::kInt8,
                                                                    static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                                    static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                                    float,
                                                                    USE_INST_BUF,
                                                                    DTypeOut,
                                                                    ComputeUnit::kCudaCore,
                                                                    mask_mode,
                                                                    RETURN_LSE,
                                                                    true,
                                                                    FUSE_V_MEAN,
                                                                    USE_FP16_ACCUM>;

                        sage::set_max_dynamic_smem_once(kernel_func, smem_max, query.get_device());

                        dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
                        dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

                        float* value_mean_ptr = nullptr;
                        if constexpr (FUSE_V_MEAN) {
                            value_mean_ptr = reinterpret_cast<float*>(value_mean_opt->data_ptr());
                        }

                        kernel_func<<<grid, block, smem_max, stream>>>(
                            query.data_ptr<int8_t>(),
                            key.data_ptr<int8_t>(),
                            reinterpret_cast<int8_t*>(value.data_ptr()),
                            reinterpret_cast<DTypeOut*>(output.data_ptr()),
                            (RETURN_LSE) ? reinterpret_cast<float*>(lse.data_ptr()) : nullptr,
                            reinterpret_cast<float*>(query_scale.data_ptr()),
                            reinterpret_cast<float*>(key_scale.data_ptr()),
                            reinterpret_cast<float*>(value_scale.data_ptr()),
                            value_mean_ptr,
                            qo_len,
                            kv_len,
                            qo_per_kv_head,
                            stride_batch_q,
                            stride_seq_q,
                            stride_h_q,
                            stride_batch_k,
                            stride_seq_k,
                            stride_h_k,
                            stride_batch_v,
                            stride_h_v,
                            stride_d_v,
                            stride_batch_o,
                            stride_seq_o,
                            stride_h_o,
                            sm_scale);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    });
                });
            });
        });
    });

    return lse;
}

}  // namespace SAGEATTN_ARCH_NS
}  // namespace sage
