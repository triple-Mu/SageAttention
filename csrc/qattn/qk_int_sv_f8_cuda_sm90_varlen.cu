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

// The varlen half of the sm90 attention kernel: the same body as
// qk_int_sv_f8_cuda_sm90.cu, compiled from the same header with SAGE_VARLEN
// defined and into its own namespace (two different kernels cannot share a
// mangled name in the single _C extension). Everything varlen-specific is in
// the impl header behind that macro; what lives here is the launcher.
//
// One variant is instantiated, the one plan.cpp resolves to for sm90: fused V
// scale, fp32+fp32 accumulation. The unfused one exists for dense callers only.

#define SAGE_VARLEN 1
#define SAGEATTN_SM90_NS sm90_varlen

#include "qk_int_sv_f8_sm90_impl.cuh"

#include "attn_cuda_sm90_varlen.h"
#include "launch_utils.cuh"

namespace sage {
namespace sm90_varlen {

torch::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_varlen_attn_inst_buf(torch::Tensor query,
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
    cudaStream_t               stream = at::cuda::getCurrentCUDAStream();

    QKVVarlenLayout qkv = qkv_varlen_layout_parse<QKVFamily::kSVF8TMA>(query,
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
                        SAGEATTN_CHECK_QK_SCALE_SHAPES_VARLEN(blk_total(total_q, batch_size, CTA_Q)
                                                                  * (NUM_THREADS / 32),
                                                              blk_total(total_k, batch_size, CTA_K));

                        CHECK_SHAPE(value_scale, batch_size, num_kv_heads, head_dim);

                        // Rank-4 tensor maps over a rank-3 packed tensor: the
                        // batch extent is 1 and the kernel's batch coordinate
                        // is always 0, so the sequence offset can ride in the
                        // token coordinate and the map itself never sees a
                        // cu_seqlens value (which is what makes it safe to
                        // capture in a cudagraph). The batch stride is the
                        // tensor's own full extent - the driver rejects a
                        // stride that is not 16-byte aligned, so 0 is not an
                        // option even though nothing reads it.
                        CUtensorMap tma_map_Q =
                            create_tensor_map_4D<CTA_Q, HEAD_DIM>(reinterpret_cast<int8_t*>(query.data_ptr()),
                                                                  1,
                                                                  num_qo_heads,
                                                                  total_q,
                                                                  HEAD_DIM,
                                                                  static_cast<int64_t>(stride_seq_q) * total_q,
                                                                  stride_h_q,
                                                                  stride_seq_q);
                        CUtensorMap tma_map_K =
                            create_tensor_map_4D<CTA_K, HEAD_DIM>(reinterpret_cast<int8_t*>(key.data_ptr()),
                                                                  1,
                                                                  num_kv_heads,
                                                                  total_k,
                                                                  HEAD_DIM,
                                                                  static_cast<int64_t>(stride_seq_k) * total_k,
                                                                  stride_h_k,
                                                                  stride_seq_k);
                        CUtensorMap tma_map_V =
                            create_tensor_map_4D<HEAD_DIM, CTA_K>(reinterpret_cast<int8_t*>(value.data_ptr()),
                                                                  1,
                                                                  num_kv_heads,
                                                                  HEAD_DIM,
                                                                  padded_k,
                                                                  stride_h_v * num_kv_heads,
                                                                  stride_h_v,
                                                                  stride_d_v);

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

                        // grid.x covers the longest sequence; the blocks the
                        // shorter ones do not need exit at the top of the kernel
                        dim3 grid(div_ceil(max_seqlen_q, CTA_Q), num_qo_heads, batch_size);
                        kernel<<<grid, NUM_THREADS, smem_bytes, stream>>>(
                            tma_map_Q,
                            tma_map_K,
                            tma_map_V,
                            reinterpret_cast<float*>(query_scale.data_ptr()),
                            reinterpret_cast<float*>(key_scale.data_ptr()),
                            reinterpret_cast<float*>(value_scale.data_ptr()),
                            reinterpret_cast<DTypeOut*>(output.data_ptr()),
                            (RETURN_LSE) ? reinterpret_cast<float*>(lse.data_ptr()) : nullptr,
                            0,  // stride_batch_o: the packed layout has no batch dimension
                            stride_h_o,
                            stride_seq_o,
                            cu_seqlens_q.data_ptr<int32_t>(),
                            cu_seqlens_k.data_ptr<int32_t>(),
                            q_scale_stride_h,
                            k_scale_stride_h,
                            static_cast<uint32_t>(total_q),  // lse is [heads, total_q], contiguous
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

}  // namespace sm90_varlen
}  // namespace sage
