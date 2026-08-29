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

// torch.ops.sageattention.fwd_varlen — the packed [total_tokens, heads,
// head_dim] entry point. It is a separate op rather than a flag on fwd
// because the two disagree about rank, about the scale layout and about what
// "batch" means; fwd's dense contract (and the tests that pin its error
// strings) stays untouched.
//
// The decision table is the same resolve() (plan.cpp) with varlen=true, so a
// varlen call lands on the same backend, tile geometry and accumulator the
// dense call would have picked.

#include <optional>
#include <tuple>

#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/types.h>

#include "arch.h"
#include "config.h"
#include "plan.h"
#include "sageattn_build_config.h"
#include "varlen_check.h"

#if SAGEATTN_BUILD_VARLEN && SAGEATTN_BUILD_SM80
#include "../qattn/attn_cuda_sm80_varlen.h"
#endif

namespace sage {
namespace {

std::tuple<at::Tensor, std::optional<at::Tensor>> fwd_varlen_cuda(const at::Tensor&                query,
                                                                  const at::Tensor&                key,
                                                                  const at::Tensor&                value,
                                                                  const at::Tensor&                query_scale,
                                                                  const at::Tensor&                key_scale,
                                                                  const at::Tensor&                cu_seqlens_q,
                                                                  const at::Tensor&                cu_seqlens_k,
                                                                  const std::optional<at::Tensor>& value_scale,
                                                                  const std::optional<at::Tensor>& value_mean,
                                                                  c10::SymInt                      max_seqlen_q,
                                                                  c10::SymInt                      max_seqlen_k,
                                                                  c10::string_view                 qk_quant_gran,
                                                                  c10::string_view                 pv_accum_dtype,
                                                                  c10::string_view                 v_layout,
                                                                  bool                             is_causal,
                                                                  double                           sm_scale,
                                                                  bool                             return_lse,
                                                                  at::ScalarType                   out_dtype)
{
    const c10::cuda::CUDAGuard device_guard(query.device());

    TORCH_CHECK(SAGEATTN_BUILD_VARLEN != 0, "sageattention was built without varlen support (SAGE_BUILD_VARLEN=OFF)");

    const QuantGran gran = parse_quant_gran(qk_quant_gran);
    const PVAccum   pv   = parse_pv_accum(pv_accum_dtype);
    const VLayout   vl   = parse_v_layout(v_layout);

    TORCH_CHECK(query.dim() == 3, "query must be 3-D [total_tokens, heads, head_dim] in the varlen layout");
    const int head_dim = static_cast<int>(query.size(2));
    const CC  cc       = device_cc(query.device().index());
    Plan      plan     = resolve(cc,
                        head_dim,
                        std::nullopt,
                        gran,
                        pv,
                        value_mean.has_value() ? std::optional<bool>(true) : std::nullopt,
                        /*varlen=*/true);
    TORCH_CHECK(plan.error.empty(), "sageattention.fwd_varlen: ", plan.error);
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
    TORCH_CHECK(!value_mean.has_value(), "smooth_v has no varlen kernel; resolve() downgrades it");

    at::Tensor out = at::empty(query.sizes(), query.options().dtype(out_dtype));

    const int     causal_int     = is_causal ? 1 : 0;
    const int     gran_int       = static_cast<int>(plan.gran);
    const float   sm_scale_f32   = static_cast<float>(sm_scale);
    const int     return_lse_int = return_lse ? 1 : 0;
    const int64_t max_q          = max_seqlen_q.guard_int(__FILE__, __LINE__);
    const int64_t max_k          = max_seqlen_k.guard_int(__FILE__, __LINE__);

    at::Tensor lse;
    switch (plan.backend) {
#if SAGEATTN_BUILD_VARLEN && SAGEATTN_BUILD_SM80
        case Backend::kSm80F16: {
            // one instantiation, the default pv_accum_dtype for sm80
            TORCH_CHECK(plan.pv == PVAccum::kFp32,
                        "pv_accum_dtype \"",
                        name(plan.pv),
                        "\" has no varlen kernel yet; sm80 varlen implements \"fp32\"");
            lse = sm80_varlen::qk_int8_sv_f16_accum_f32_varlen_attn(query,
                                                                    key,
                                                                    value,
                                                                    out,
                                                                    query_scale,
                                                                    key_scale,
                                                                    cu_seqlens_q,
                                                                    cu_seqlens_k,
                                                                    max_q,
                                                                    max_k,
                                                                    causal_int,
                                                                    gran_int,
                                                                    sm_scale_f32,
                                                                    return_lse_int);
            break;
        }
#endif
        default:
            TORCH_CHECK(false, "varlen is not implemented for the ", name(plan.backend), " backend yet");
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
    // max_seqlen_* are SymInt: they size grid.x and appear in no output shape,
    // so dynamo keeps them symbolic instead of recompiling per batch shape.
    m.def("fwd_varlen(Tensor query, Tensor key, Tensor value, "
          "Tensor query_scale, Tensor key_scale, Tensor cu_seqlens_q, Tensor cu_seqlens_k, "
          "Tensor? value_scale=None, Tensor? value_mean=None, *, "
          "SymInt max_seqlen_q, SymInt max_seqlen_k, "
          "str qk_quant_gran=\"per_thread\", str pv_accum_dtype=\"fp32\", "
          "str v_layout=\"mma_k16\", bool is_causal=False, float sm_scale=1.0, "
          "bool return_lse=False, ScalarType out_dtype=float16) -> (Tensor out, Tensor? lse)");
}

TORCH_LIBRARY_IMPL(sageattention, CUDA, m)
{
    m.impl("fwd_varlen", TORCH_FN(sage::fwd_varlen_cuda));
}
