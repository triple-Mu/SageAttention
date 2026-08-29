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

// sm12x (consumer Blackwell) instantiations of the qk_int8 / sv_f8 kernels.
// The fp8 mma fp32 accumulator is exact on these parts, so plain fp32
// accumulation ("fp32") is the default path and the fp32+fp32 inst_buf
// variants are not built; the f16 inst_buf variant is kept as an opt-in
// speed mode until the block-scaled PV path lands.

namespace sage {
namespace sm120 {

torch::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(torch::Tensor query,
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
                                                        int           return_lse);

torch::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn(torch::Tensor query,
                                                                    torch::Tensor key,
                                                                    torch::Tensor value,
                                                                    torch::Tensor output,
                                                                    torch::Tensor query_scale,
                                                                    torch::Tensor key_scale,
                                                                    torch::Tensor value_scale,
                                                                    torch::Tensor value_mean,
                                                                    int           tensor_layout,
                                                                    int           is_causal,
                                                                    int           qk_quant_gran,
                                                                    float         sm_scale,
                                                                    int           return_lse);

torch::Tensor qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(torch::Tensor query,
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
                                                                 int           return_lse);

}  // namespace sm120
}  // namespace sage
