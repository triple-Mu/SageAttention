#include "attn_cuda_sm89.h"
#include "qk_int_sv_f8_launcher_sm89.cuh"

torch::Tensor qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(torch::Tensor query,
                    torch::Tensor key,
                    torch::Tensor value,
                    torch::Tensor output,
                    torch::Tensor query_scale,
                    torch::Tensor key_scale,
                    torch::Tensor value_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse)
{
  return qk_int8_sv_f8_fuse_v_scale_attn_launcher_sm89<true, false, true>(
      query, key, value, output, query_scale, key_scale, value_scale, nullptr,
      tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse);
}
