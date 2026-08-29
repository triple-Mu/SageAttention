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

// torch.ops.sageattention.fwd — the single attention entry point. Picks the
// kernel from the device's compute capability and the request via resolve()
// (plan.cpp, the one source of truth), validates the combination loudly
// (the old per-arch pipeline silently returned uninitialized memory for
// unsupported pv_accum_dtype), and dispatches onto the per-arch launchers.
//
// Functional shape: out is allocated here (not mutated in), lse is
// std::nullopt unless return_lse. The launchers keep their historical
// mutating signatures; this TU owns the translation.

#include <optional>
#include <tuple>

#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/types.h>

#include "arch.h"
#include "config.h"
#include "plan.h"
#include "sageattn_build_config.h"

#if SAGEATTN_BUILD_SM80
#include "../qattn/attn_cuda_sm80.h"
#endif
#if SAGEATTN_BUILD_SM89
#include "../qattn/attn_cuda_sm89.h"
#endif
#if SAGEATTN_BUILD_SM90
#include "../qattn/attn_cuda_sm90.h"
#endif
#if SAGEATTN_BUILD_SM100
#include "../qattn/attn_cuda_sm100.h"
#endif
#if SAGEATTN_BUILD_SM120
#include "../qattn/attn_cuda_sm120.h"
#endif

namespace sage {
namespace {

std::tuple<at::Tensor, std::optional<at::Tensor>> fwd_cuda(const at::Tensor&                query,
                                                           const at::Tensor&                key,
                                                           const at::Tensor&                value,
                                                           const at::Tensor&                query_scale,
                                                           const at::Tensor&                key_scale,
                                                           const std::optional<at::Tensor>& value_scale,
                                                           const std::optional<at::Tensor>& value_mean,
                                                           c10::string_view                 tensor_layout,
                                                           c10::string_view                 qk_quant_gran,
                                                           c10::string_view                 pv_accum_dtype,
                                                           c10::string_view                 v_layout,
                                                           bool                             is_causal,
                                                           double                           sm_scale,
                                                           bool                             return_lse,
                                                           at::ScalarType                   out_dtype)
{
    const c10::cuda::CUDAGuard device_guard(query.device());

    // Parse every string once; unknown values fail here.
    const TensorLayout layout = parse_tensor_layout(tensor_layout);
    const QuantGran    gran   = parse_quant_gran(qk_quant_gran);
    const PVAccum      pv     = parse_pv_accum(pv_accum_dtype);
    const VLayout      vl     = parse_v_layout(v_layout);

    TORCH_CHECK(query.dim() == 4, "query must be 4-D");
    const int head_dim = static_cast<int>(query.size(3));
    const CC  cc       = device_cc(query.device().index());
    Plan      plan     = resolve(
        cc, head_dim, std::nullopt, gran, pv, value_mean.has_value() ? std::optional<bool>(true) : std::nullopt);
    TORCH_CHECK(plan.error.empty(), "sageattention.fwd: ", plan.error);
    // fwd receives already-quantized inputs, so the caller must have fetched the
    // same plan; re-check the request against the resolved plan (a mismatch
    // means the tensors were prepared for a different kernel).
    TORCH_CHECK(plan.gran == gran,
                "qk_quant_gran \"",
                name(gran),
                "\" is not what the resolved plan selected (\"",
                name(plan.gran),
                "\")");
    TORCH_CHECK(plan.pv == pv,
                "pv_accum_dtype \"",
                name(pv),
                "\" is not supported for sm_",
                cc.major,
                cc.minor,
                " (resolved plan uses \"",
                name(plan.pv),
                "\")");
    // The most dangerous silent failure: mma_k16-permuted vs linear V layouts
    // share shape and dtype. Validate intent against the selected kernel.
    TORCH_CHECK(plan.v_layout == vl,
                "value was prepared with v_layout=\"",
                v_layout,
                "\" but the kernel selected for sm_",
                cc.major,
                cc.minor,
                " (",
                name(plan.backend),
                ") requires \"",
                name(plan.v_layout),
                "\"");
    TORCH_CHECK(plan.need_value_scale == value_scale.has_value(),
                plan.need_value_scale ? "this backend/config requires value_scale (fp8 PV path)" :
                                        "value_scale was passed but the selected kernel does not take it");
    TORCH_CHECK(plan.need_value_mean == value_mean.has_value(),
                plan.need_value_mean ? "smooth_v requires value_mean" :
                                       "value_mean was passed but the resolved plan has smooth_v off");
    // The fp8 path zero-pads V^T to v_pad_multiple and tiles KV by blk_k. The
    // varlen layout starts every sequence's V^T slab at a v_pad_multiple
    // boundary (varlen.h, pad_offset), so the two must be the same number or
    // the slabs stop landing on KV tile boundaries.
    TORCH_CHECK(!plan.pv_fp8 || plan.v_pad_multiple == plan.blk_k,
                "internal: v_pad_multiple ",
                plan.v_pad_multiple,
                " != blk_k ",
                plan.blk_k);

    at::Tensor out = at::empty(query.sizes(), query.options().dtype(out_dtype));

    const int   layout_int     = static_cast<int>(layout);
    const int   causal_int     = is_causal ? 1 : 0;
    const int   gran_int       = static_cast<int>(plan.gran);
    const float sm_scale_f32   = static_cast<float>(sm_scale);
    const int   return_lse_int = return_lse ? 1 : 0;

    at::Tensor lse;
    switch (plan.backend) {
#if SAGEATTN_BUILD_SM80
        case Backend::kSm80F16: {
            if (plan.smooth_v) {  // (fp16, smooth_v)
                lse = sm80::qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(query,
                                                                      key,
                                                                      value,
                                                                      out,
                                                                      query_scale,
                                                                      key_scale,
                                                                      *value_mean,
                                                                      layout_int,
                                                                      causal_int,
                                                                      gran_int,
                                                                      sm_scale_f32,
                                                                      return_lse_int);
            }
            else if (plan.pv == PVAccum::kFp32) {
                lse = sm80::qk_int8_sv_f16_accum_f32_attn(query,
                                                          key,
                                                          value,
                                                          out,
                                                          query_scale,
                                                          key_scale,
                                                          layout_int,
                                                          causal_int,
                                                          gran_int,
                                                          sm_scale_f32,
                                                          return_lse_int);
            }
            else if (plan.pv == PVAccum::kFp16) {
                lse = sm80::qk_int8_sv_f16_accum_f16_attn(query,
                                                          key,
                                                          value,
                                                          out,
                                                          query_scale,
                                                          key_scale,
                                                          layout_int,
                                                          causal_int,
                                                          gran_int,
                                                          sm_scale_f32,
                                                          return_lse_int);
            }
            else {  // kFp16Fp32
                lse = sm80::qk_int8_sv_f16_accum_f16_attn_inst_buf(query,
                                                                   key,
                                                                   value,
                                                                   out,
                                                                   query_scale,
                                                                   key_scale,
                                                                   layout_int,
                                                                   causal_int,
                                                                   gran_int,
                                                                   sm_scale_f32,
                                                                   return_lse_int);
            }
            break;
        }
#endif
#if SAGEATTN_BUILD_SM89
        case Backend::kSm89F8: {
            if (plan.smooth_v) {  // (fp32, smooth_v)
                lse = sm89::qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn(query,
                                                                                  key,
                                                                                  value,
                                                                                  out,
                                                                                  query_scale,
                                                                                  key_scale,
                                                                                  *value_scale,
                                                                                  *value_mean,
                                                                                  layout_int,
                                                                                  causal_int,
                                                                                  gran_int,
                                                                                  sm_scale_f32,
                                                                                  return_lse_int);
            }
            else if (plan.pv == PVAccum::kFp32) {
                lse = sm89::qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(query,
                                                                      key,
                                                                      value,
                                                                      out,
                                                                      query_scale,
                                                                      key_scale,
                                                                      *value_scale,
                                                                      layout_int,
                                                                      causal_int,
                                                                      gran_int,
                                                                      sm_scale_f32,
                                                                      return_lse_int);
            }
            else if (plan.pv == PVAccum::kFp32Fp32) {
                lse = sm89::qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(query,
                                                                               key,
                                                                               value,
                                                                               out,
                                                                               query_scale,
                                                                               key_scale,
                                                                               *value_scale,
                                                                               layout_int,
                                                                               causal_int,
                                                                               gran_int,
                                                                               sm_scale_f32,
                                                                               return_lse_int);
            }
            else {  // kFp32Fp16
                lse = sm89::qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(query,
                                                                               key,
                                                                               value,
                                                                               out,
                                                                               query_scale,
                                                                               key_scale,
                                                                               *value_scale,
                                                                               layout_int,
                                                                               causal_int,
                                                                               gran_int,
                                                                               sm_scale_f32,
                                                                               return_lse_int);
            }
            break;
        }
#endif
#if SAGEATTN_BUILD_SM90
        case Backend::kSm90F8: {  // (fp32+fp32) only
            lse = sm90::qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(query,
                                                                           key,
                                                                           value,
                                                                           out,
                                                                           query_scale,
                                                                           key_scale,
                                                                           *value_scale,
                                                                           layout_int,
                                                                           causal_int,
                                                                           gran_int,
                                                                           sm_scale_f32,
                                                                           return_lse_int);
            break;
        }
#endif
#if SAGEATTN_BUILD_SM100
        case Backend::kSm100F8: {  // (fp32) only
            lse = sm100::qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(query,
                                                                   key,
                                                                   value,
                                                                   out,
                                                                   query_scale,
                                                                   key_scale,
                                                                   *value_scale,
                                                                   layout_int,
                                                                   causal_int,
                                                                   gran_int,
                                                                   sm_scale_f32,
                                                                   return_lse_int);
            break;
        }
#endif
#if SAGEATTN_BUILD_SM120
        case Backend::kSm120F8: {
            if (plan.smooth_v) {  // (fp32, smooth_v)
                lse = sm120::qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn(query,
                                                                                   key,
                                                                                   value,
                                                                                   out,
                                                                                   query_scale,
                                                                                   key_scale,
                                                                                   *value_scale,
                                                                                   *value_mean,
                                                                                   layout_int,
                                                                                   causal_int,
                                                                                   gran_int,
                                                                                   sm_scale_f32,
                                                                                   return_lse_int);
            }
            else if (plan.pv == PVAccum::kFp32) {
                lse = sm120::qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(query,
                                                                       key,
                                                                       value,
                                                                       out,
                                                                       query_scale,
                                                                       key_scale,
                                                                       *value_scale,
                                                                       layout_int,
                                                                       causal_int,
                                                                       gran_int,
                                                                       sm_scale_f32,
                                                                       return_lse_int);
            }
            else {  // kFp32Fp16
                lse = sm120::qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(query,
                                                                                key,
                                                                                value,
                                                                                out,
                                                                                query_scale,
                                                                                key_scale,
                                                                                *value_scale,
                                                                                layout_int,
                                                                                causal_int,
                                                                                gran_int,
                                                                                sm_scale_f32,
                                                                                return_lse_int);
            }
            break;
        }
#endif
        default:
            TORCH_CHECK(false,
                        "unreachable: resolve() returned backend ",
                        name(plan.backend),
                        " which is not compiled into this build");
    }

    if (!return_lse) {
        return {out, std::nullopt};
    }
    return {out, lse};
}

}  // namespace
}  // namespace sage

TORCH_LIBRARY_FRAGMENT(sageattention, m)
{
    m.def("fwd(Tensor query, Tensor key, Tensor value, "
          "Tensor query_scale, Tensor key_scale, "
          "Tensor? value_scale=None, Tensor? value_mean=None, *, "
          "str tensor_layout=\"HND\", str qk_quant_gran=\"per_thread\", "
          "str pv_accum_dtype=\"fp32\", str v_layout=\"mma_k16\", "
          "bool is_causal=False, float sm_scale=1.0, bool return_lse=False, "
          "ScalarType out_dtype=float16) -> (Tensor out, Tensor? lse)");
}

TORCH_LIBRARY_IMPL(sageattention, CUDA, m)
{
    m.impl("fwd", TORCH_FN(sage::fwd_cuda));
}
