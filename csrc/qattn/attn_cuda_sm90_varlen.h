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

#include <torch/types.h>

namespace sage {
namespace sm90_varlen {

// query/key/output are packed [total_tokens, heads, head_dim]; value is the
// block-padded V^T slab [kv_heads, head_dim, blk_total(total_k, batch, CTA_K) *
// CTA_K]; the scales are [heads, blocks] and the returned lse (when
// return_lse) is [heads, total_q]. max_seqlen_* size the grid only. See
// attn_cuda_sm90.h for the dense family.
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
                                                                        int           return_lse);

}  // namespace sm90_varlen
}  // namespace sage
