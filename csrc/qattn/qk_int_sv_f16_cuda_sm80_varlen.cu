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

// The varlen half of the sm80 attention kernel: the same body as
// qk_int_sv_f16_cuda_sm80.cu, compiled from the same header with SAGE_VARLEN
// defined and into its own namespace (two different kernels cannot share a
// mangled name in the single _C extension). Everything varlen-specific is in
// the impl header behind that macro; what lives here is the launcher.
//
// One pv_accum_dtype is instantiated, the one plan.cpp resolves to for sm80
// by default (plain fp32 accumulation). The other two sm80 variants exist for
// dense callers only; adding them would double this file's compile time for
// configurations sageattn_varlen never asks for.

#define SAGE_VARLEN 1
#define SAGEATTN_ARCH_NS sm80_varlen

#include "qk_int_sv_f16_sm80_impl.cuh"

#include "attn_cuda_sm80_varlen.h"
#include "launch_utils.cuh"

namespace sage {
namespace sm80_varlen {

torch::Tensor qk_int8_sv_f16_accum_f32_varlen_attn(torch::Tensor query,
                                                   torch::Tensor key,
                                                   torch::Tensor value,
                                                   torch::Tensor output,
                                                   torch::Tensor query_scale,
                                                   torch::Tensor key_scale,
                                                   torch::Tensor cu_seqlens_q,
                                                   torch::Tensor cu_seqlens_k,
                                                   int64_t       max_seqlen_q_in,
                                                   int64_t       max_seqlen_k_in,
                                                   int           is_causal,
                                                   int           qk_quant_gran,
                                                   float         sm_scale,
                                                   int           return_lse)
{
    const c10::cuda::CUDAGuard device_guard(query.device());

    QKVVarlenLayout qkv = qkv_varlen_layout_parse<QKVFamily::kSVF16>(query,
                                                                     key,
                                                                     value,
                                                                     output,
                                                                     query_scale,
                                                                     key_scale,
                                                                     cu_seqlens_q,
                                                                     cu_seqlens_k,
                                                                     max_seqlen_q_in,
                                                                     max_seqlen_k_in,
                                                                     return_lse);
    SAGEATTN_QKV_VARLEN_LOCALS_F16(qkv);

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

                        constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

                        // The scale lengths are the block algebra of varlen.h:
                        // total/blk + batch blocks, each carrying the same
                        // per-block entries the dense layout has.
                        SAGEATTN_CHECK_QK_SCALE_SHAPES_VARLEN(blk_total(total_q, batch_size, CTA_Q) * (CTA_Q / WARP_Q),
                                                              blk_total(total_k, batch_size, CTA_K) * (CTA_K / WARP_K));

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
                                                                     float,
                                                                     false,
                                                                     DTypeOut,
                                                                     ComputeUnit::kTensorCore,
                                                                     mask_mode,
                                                                     RETURN_LSE,
                                                                     false>;

                        sage::set_max_dynamic_smem_once(kernel_func, smem_max, query.get_device());

                        // grid.x covers the longest sequence; the blocks the
                        // shorter ones do not need exit at the top of the kernel
                        dim3 grid(div_ceil(max_seqlen_q, CTA_Q), num_qo_heads, batch_size);
                        dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

                        kernel_func<<<grid, block, smem_max, stream>>>(
                            query.data_ptr<int8_t>(),
                            key.data_ptr<int8_t>(),
                            reinterpret_cast<half*>(value.data_ptr()),
                            reinterpret_cast<DTypeOut*>(output.data_ptr()),
                            (RETURN_LSE) ? reinterpret_cast<float*>(lse.data_ptr()) : nullptr,
                            reinterpret_cast<float*>(query_scale.data_ptr()),
                            reinterpret_cast<float*>(key_scale.data_ptr()),
                            nullptr,
                            cu_seqlens_q.data_ptr<int32_t>(),
                            cu_seqlens_k.data_ptr<int32_t>(),
                            q_scale_stride_h,
                            k_scale_stride_h,
                            static_cast<uint32_t>(total_q),
                            qo_per_kv_head,
                            0,  // stride_batch_q: the packed layout has no batch dimension
                            stride_seq_q,
                            stride_h_q,
                            0,  // stride_batch_k
                            stride_seq_k,
                            stride_h_k,
                            0,  // stride_batch_v
                            stride_seq_v,
                            stride_h_v,
                            0,  // stride_batch_o
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

}  // namespace sm80_varlen
}  // namespace sage
