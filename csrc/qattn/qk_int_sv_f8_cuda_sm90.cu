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

// Host launchers for the sm90 attention kernel; the kernel body and the
// sage::sm90 namespace rationale live in the impl header.
#include "qk_int_sv_f8_sm90_impl.cuh"
#include "launch_utils.cuh"

namespace sage {
namespace sm90 {

torch::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(torch::Tensor query,
                                                                 torch::Tensor key,
                                                                 torch::Tensor value,
                                                                 torch::Tensor output,
                                                                 torch::Tensor query_scale,
                                                                 torch::Tensor key_scale,
                                                                 torch::Tensor value_scale,
                                                                 int           tensor_layout,
                                                                 int           is_causal,
                                                                 int           qk_quant_gran,
                                                                 float         sm_scale,
                                                                 int           return_lse)
{
    const c10::cuda::CUDAGuard device_guard(query.device());
    cudaStream_t               stream = at::cuda::getCurrentCUDAStream();

    QKVLayout qkv = qkv_layout_parse<QKVFamily::kSVF8TMA>(query,
                                                          key,
                                                          value,
                                                          output,
                                                          query_scale,
                                                          key_scale,
                                                          &value_scale,
                                                          /*value_mean_opt=*/nullptr,
                                                          tensor_layout,
                                                          return_lse);
    SAGEATTN_QKV_LAYOUT_LOCALS_FP8(qkv);

    auto out_dtype = output.scalar_type();

    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
            DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
                DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
                    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(out_dtype, DTypeOut, {
                        constexpr int CTA_Q       = 64;
                        constexpr int CTA_K       = 128;
                        constexpr int NUM_THREADS = 128;

                        constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

                        TORCH_CHECK(value.size(3) >= div_ceil(kv_len, CTA_K) * CTA_K,
                                    "value.size(3) >= div_ceil(kv_len, CTA_K) * CTA_K failed: value.size(3)=",
                                    value.size(3),
                                    ", kv_len=",
                                    kv_len,
                                    ", CTA_K=",
                                    CTA_K);

                        SAGEATTN_CHECK_QK_SCALE_SHAPES(div_ceil(qo_len, CTA_Q) * (NUM_THREADS / 32),
                                                       div_ceil(kv_len, CTA_K));

                        CHECK_SHAPE(value_scale, batch_size, num_kv_heads, head_dim);

                        QKVTensorMaps tma_maps = make_qkv_tensor_maps<CTA_Q, CTA_K, HEAD_DIM>(query, key, value, qkv);

                        auto*  kernel     = qk_int8_sv_f8_attn_kernel<CTA_Q,
                                                                 CTA_K,
                                                                 NUM_THREADS,
                                                                 HEAD_DIM,
                                                                 static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                                 static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                                 DTypeOut,
                                                                 mask_mode,
                                                                 RETURN_LSE,
                                                                 true>;
                        size_t smem_bytes = CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t)
                                            + CTA_K * HEAD_DIM * sizeof(int8_t);
                        sage::set_max_dynamic_smem_once(kernel, smem_bytes, query.get_device());

                        dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
                        kernel<<<grid, NUM_THREADS, smem_bytes, stream>>>(
                            tma_maps.q,
                            tma_maps.k,
                            tma_maps.v,
                            reinterpret_cast<float*>(query_scale.data_ptr()),
                            reinterpret_cast<float*>(key_scale.data_ptr()),
                            reinterpret_cast<float*>(value_scale.data_ptr()),
                            reinterpret_cast<DTypeOut*>(output.data_ptr()),
                            (RETURN_LSE) ? reinterpret_cast<float*>(lse.data_ptr()) : nullptr,
                            stride_batch_o,
                            stride_h_o,
                            stride_seq_o,
                            qo_len,
                            kv_len,
                            qo_per_kv_head,
                            sm_scale);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    });
                });
            });
        });
    });

    return lse;
}
}  // namespace sm90
}  // namespace sage
