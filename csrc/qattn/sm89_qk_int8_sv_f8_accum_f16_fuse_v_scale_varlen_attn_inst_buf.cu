// The packed-layout half of the sm89 kernel family, compiled once into
// sage::sm89_varlen (SAGEATTN_ARCH_NS comes from CMake, as it does for the
// dense TUs). SAGE_VARLEN selects the packed body inside the shared header.
//
// One pv_accum_dtype is instantiated, the one plan.cpp resolves to for sm89 by
// default (fp32 accumulation with an fp16 instruction buffer). The other sm89
// variants exist for dense callers only; adding them would multiply this
// file's compile time for configurations sageattn_varlen never asks for.
#define SAGE_VARLEN 1

#include "qk_int_sv_f8_varlen_launcher_sm89.cuh"

namespace sage {
namespace SAGEATTN_ARCH_NS {

torch::Tensor qk_int8_sv_f8_accum_f16_fuse_v_scale_varlen_attn_inst_buf(torch::Tensor query,
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
    return qk_int8_sv_f8_fuse_v_scale_varlen_attn_launcher_sm89<true, true>(query,
                                                                            key,
                                                                            value,
                                                                            output,
                                                                            query_scale,
                                                                            key_scale,
                                                                            value_scale,
                                                                            cu_seqlens_q,
                                                                            cu_seqlens_k,
                                                                            max_seqlen_q_in,
                                                                            max_seqlen_k_in,
                                                                            is_causal,
                                                                            qk_quant_gran,
                                                                            sm_scale,
                                                                            return_lse);
}

}  // namespace SAGEATTN_ARCH_NS
}  // namespace sage
