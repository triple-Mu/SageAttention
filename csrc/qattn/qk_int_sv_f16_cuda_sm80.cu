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

// Host launchers for the sm80 attention kernel; the kernel body and the
// sage::sm80 namespace rationale live in the impl header.
#include "launch_utils.cuh"
#include "qk_int_sv_f16_sm80_impl.cuh"

namespace sage {
namespace sm80 {

// Single templated launcher body shared by the four dense sm80 variants
// (same structure as qk_int_sv_f8_launcher_sm89.cuh):
//   AccumT       - PV accumulator dtype (float or half)
//   USE_INST_BUF - inst_buf PV accumulation kernel (fp32 accum; WARP_Q drops
//                  to 16 for head_dim 128 to fit the register budget)
//   FUSE_V_MEAN  - fuse the +V_mean epilogue (value_mean_opt must be set)
// The public functions below bind their historical names to instantiations
// of this template.
//
// tensor_layout 0 for [B, N, H, D], 1 for [B, H, N, D]
template<typename AccumT, bool USE_INST_BUF, bool FUSE_V_MEAN>
torch::Tensor qk_int8_sv_f16_attn_launcher_sm80(torch::Tensor        query,
                                                torch::Tensor        key,
                                                torch::Tensor        value,
                                                torch::Tensor        output,
                                                torch::Tensor        query_scale,
                                                torch::Tensor        key_scale,
                                                const torch::Tensor* value_mean_opt,
                                                int                  tensor_layout,
                                                int                  is_causal,
                                                int                  qk_quant_gran,
                                                float                sm_scale,
                                                int                  return_lse)
{
    const c10::cuda::CUDAGuard device_guard(query.device());

    QKVLayout qkv = qkv_layout_parse<QKVFamily::kSVF16>(query,
                                                        key,
                                                        value,
                                                        output,
                                                        query_scale,
                                                        key_scale,
                                                        /*value_scale_opt=*/nullptr,
                                                        value_mean_opt,
                                                        tensor_layout,
                                                        return_lse);
    SAGEATTN_QKV_LAYOUT_LOCALS_F16(qkv);

    auto         out_dtype = output.scalar_type();
    cudaStream_t stream    = at::cuda::getCurrentCUDAStream();

    if constexpr (FUSE_V_MEAN) {
        TORCH_CHECK(value_mean_opt->scalar_type() == out_dtype, "value_mean and output must have the same dtype");
    }

    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
            DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
                DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
                    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(out_dtype, DTypeOut, {
                        constexpr int CTA_Q  = 128;
                        constexpr int CTA_K  = 64;
                        constexpr int WARP_Q = (USE_INST_BUF && HEAD_DIM != 64) ? 16 : 32;
                        constexpr int WARP_K = 64;

                        constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

                        SAGEATTN_CHECK_QK_SCALE_SHAPES(div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q),
                                                       div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K));

                        if constexpr (FUSE_V_MEAN) {
                            const torch::Tensor& value_mean = *value_mean_opt;
                            CHECK_SHAPE(value_mean, batch_size, num_kv_heads, head_dim);
                        }

                        //                                     smem_Q                                     smem_K smem_V
                        //                                     smem_O
                        size_t smem_max = std::max(CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t)
                                                       + CTA_K * HEAD_DIM * sizeof(half),
                                                   CTA_Q * HEAD_DIM * sizeof(half));

                        auto kernel_func = qk_int_sv_f16_attn_kernel<CTA_Q,
                                                                     CTA_K,
                                                                     WARP_Q,
                                                                     WARP_K,
                                                                     HEAD_DIM,
                                                                     DataType::kInt8,
                                                                     static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                                     static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                                     AccumT,
                                                                     USE_INST_BUF,
                                                                     DTypeOut,
                                                                     ComputeUnit::kTensorCore,
                                                                     mask_mode,
                                                                     RETURN_LSE,
                                                                     FUSE_V_MEAN>;

                        sage::set_max_dynamic_smem_once(kernel_func, smem_max, query.get_device());

                        dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
                        dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

                        DTypeOut* value_mean_ptr = nullptr;
                        if constexpr (FUSE_V_MEAN) {
                            value_mean_ptr = reinterpret_cast<DTypeOut*>(value_mean_opt->data_ptr());
                        }

                        kernel_func<<<grid, block, smem_max, stream>>>(
                            query.data_ptr<int8_t>(),
                            key.data_ptr<int8_t>(),
                            reinterpret_cast<half*>(value.data_ptr()),
                            reinterpret_cast<DTypeOut*>(output.data_ptr()),
                            (RETURN_LSE) ? reinterpret_cast<float*>(lse.data_ptr()) : nullptr,
                            reinterpret_cast<float*>(query_scale.data_ptr()),
                            reinterpret_cast<float*>(key_scale.data_ptr()),
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
                            stride_seq_v,
                            stride_h_v,
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

torch::Tensor qk_int8_sv_f16_accum_f32_attn(torch::Tensor query,
                                            torch::Tensor key,
                                            torch::Tensor value,
                                            torch::Tensor output,
                                            torch::Tensor query_scale,
                                            torch::Tensor key_scale,
                                            int           tensor_layout,
                                            int           is_causal,
                                            int           qk_quant_gran,
                                            float         sm_scale,
                                            int           return_lse)
{
    return qk_int8_sv_f16_attn_launcher_sm80<float, false, false>(query,
                                                                  key,
                                                                  value,
                                                                  output,
                                                                  query_scale,
                                                                  key_scale,
                                                                  /*value_mean_opt=*/nullptr,
                                                                  tensor_layout,
                                                                  is_causal,
                                                                  qk_quant_gran,
                                                                  sm_scale,
                                                                  return_lse);
}

torch::Tensor qk_int8_sv_f16_accum_f16_attn(torch::Tensor query,
                                            torch::Tensor key,
                                            torch::Tensor value,
                                            torch::Tensor output,
                                            torch::Tensor query_scale,
                                            torch::Tensor key_scale,
                                            int           tensor_layout,
                                            int           is_causal,
                                            int           qk_quant_gran,
                                            float         sm_scale,
                                            int           return_lse)
{
    return qk_int8_sv_f16_attn_launcher_sm80<half, false, false>(query,
                                                                 key,
                                                                 value,
                                                                 output,
                                                                 query_scale,
                                                                 key_scale,
                                                                 /*value_mean_opt=*/nullptr,
                                                                 tensor_layout,
                                                                 is_causal,
                                                                 qk_quant_gran,
                                                                 sm_scale,
                                                                 return_lse);
}

torch::Tensor qk_int8_sv_f16_accum_f16_attn_inst_buf(torch::Tensor query,
                                                     torch::Tensor key,
                                                     torch::Tensor value,
                                                     torch::Tensor output,
                                                     torch::Tensor query_scale,
                                                     torch::Tensor key_scale,
                                                     int           tensor_layout,
                                                     int           is_causal,
                                                     int           qk_quant_gran,
                                                     float         sm_scale,
                                                     int           return_lse)
{
    return qk_int8_sv_f16_attn_launcher_sm80<float, true, false>(query,
                                                                 key,
                                                                 value,
                                                                 output,
                                                                 query_scale,
                                                                 key_scale,
                                                                 /*value_mean_opt=*/nullptr,
                                                                 tensor_layout,
                                                                 is_causal,
                                                                 qk_quant_gran,
                                                                 sm_scale,
                                                                 return_lse);
}

torch::Tensor qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(torch::Tensor query,
                                                        torch::Tensor key,
                                                        torch::Tensor value,
                                                        torch::Tensor output,
                                                        torch::Tensor query_scale,
                                                        torch::Tensor key_scale,
                                                        torch::Tensor value_mean,
                                                        int           tensor_layout,
                                                        int           is_causal,
                                                        int           qk_quant_gran,
                                                        float         sm_scale,
                                                        int           return_lse)
{
    return qk_int8_sv_f16_attn_launcher_sm80<half, false, true>(query,
                                                                key,
                                                                value,
                                                                output,
                                                                query_scale,
                                                                key_scale,
                                                                &value_mean,
                                                                tensor_layout,
                                                                is_causal,
                                                                qk_quant_gran,
                                                                sm_scale,
                                                                return_lse);
}
}  // namespace sm80
}  // namespace sage
