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

// The varlen counterpart of qk_int_sv_f8_launcher_sm89.cuh. It is only ever
// included from a translation unit that has already defined SAGE_VARLEN, so
// the kernel template it instantiates is the packed one and the namespace it
// lands in is the arch's varlen namespace (SAGEATTN_ARCH_NS, which CMake sets
// to sm89_varlen / sm120_varlen for these TUs).

#ifndef SAGE_VARLEN
#error "include this only from a varlen translation unit (-DSAGE_VARLEN)"
#endif

#include "launch_utils.cuh"
#include "qk_int_sv_f8_cuda_sm89.cuh"

namespace sage {
namespace SAGEATTN_ARCH_NS {

// One templated launcher for the packed variants, mirroring the dense
// qk_int8_sv_f8_fuse_v_scale_attn_launcher_sm89. smooth_v (FUSE_V_MEAN) has no
// packed form - plan.cpp downgrades it - so the parameter is gone rather than
// defaulted, and the kernel static_asserts the same thing.
template<bool USE_INST_BUF, bool USE_FP16_ACCUM>
torch::Tensor qk_int8_sv_f8_fuse_v_scale_varlen_attn_launcher_sm89(torch::Tensor query,
                                                                   torch::Tensor key,
                                                                   torch::Tensor value,
                                                                   torch::Tensor output,
                                                                   torch::Tensor query_scale,
                                                                   torch::Tensor key_scale,
                                                                   torch::Tensor value_scale,
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

    QKVVarlenLayout qkv = qkv_varlen_layout_parse<QKVFamily::kSVF8CudaCore>(query,
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
    SAGEATTN_QKV_VARLEN_LOCALS_FP8(qkv);

    CHECK_CUDA(value_scale);
    CHECK_CONTIGUOUS(value_scale);
    CHECK_DTYPE(value_scale, torch::kFloat32);
    CHECK_DIMS(value_scale, 3);

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

                        // This is the pin behind the kernel's V addressing: it
                        // reads the sequence's slab at blk_offset * CTA_K, so
                        // the value must have been padded with exactly CTA_K.
                        TORCH_CHECK(padded_k == blk_total(total_k, batch_size, CTA_K) * CTA_K,
                                    "transposed value last dim (",
                                    padded_k,
                                    ") must be blk_total(total_k, batch_size, ",
                                    CTA_K,
                                    ") * ",
                                    CTA_K,
                                    " (",
                                    blk_total(total_k, batch_size, CTA_K) * CTA_K,
                                    "); prepare it with quant_v_fp8_varlen(pad_multiple=",
                                    CTA_K,
                                    ")");

                        // The scale lengths are the block algebra of varlen.h:
                        // total/blk + batch blocks, each carrying the same
                        // per-block entries the dense layout has.
                        SAGEATTN_CHECK_QK_SCALE_SHAPES_VARLEN(blk_total(total_q, batch_size, CTA_Q) * (CTA_Q / WARP_Q),
                                                              blk_total(total_k, batch_size, CTA_K) * (CTA_K / WARP_K));

                        CHECK_SHAPE(value_scale, batch_size, num_kv_heads, head_dim);

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
                                                                    false,
                                                                    USE_FP16_ACCUM>;

                        sage::set_max_dynamic_smem_once(kernel_func, smem_max, query.get_device());

                        // grid.x covers the longest sequence; the blocks the
                        // shorter ones do not need exit at the top of the kernel
                        dim3 grid(div_ceil(max_seqlen_q, CTA_Q), num_qo_heads, batch_size);
                        dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

                        kernel_func<<<grid, block, smem_max, stream>>>(
                            query.data_ptr<int8_t>(),
                            key.data_ptr<int8_t>(),
                            reinterpret_cast<int8_t*>(value.data_ptr()),
                            reinterpret_cast<DTypeOut*>(output.data_ptr()),
                            (RETURN_LSE) ? reinterpret_cast<float*>(lse.data_ptr()) : nullptr,
                            reinterpret_cast<float*>(query_scale.data_ptr()),
                            reinterpret_cast<float*>(key_scale.data_ptr()),
                            reinterpret_cast<float*>(value_scale.data_ptr()),
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
                            stride_h_v,
                            stride_d_v,
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

}  // namespace SAGEATTN_ARCH_NS
}  // namespace sage
