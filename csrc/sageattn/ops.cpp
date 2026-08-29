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

// TORCH_LIBRARY registration for torch.ops.sageattention.*.
//
// Layers:
//   - 8 low-level quantization ops (mutating, -> (), no fake needed): the
//     former pybind11 _fused surface, overloads merged via optional args.
//   - per-arch attention ops qattn_smXX_* (mutating output, return lse):
//     1:1 with the historical per-arch pybind functions; string/bool flags
//     replace the magic-int API. Guarded by SAGEATTN_BUILD_SMXX.
//   - host query ops: plan (dispatch decision table) and compiled_archs.
//
// This TU is host-only: it sees the launcher declarations (namespaced per
// arch in the attn_cuda_*.h headers) and the fused.h declarations; kernels
// live in their own TUs.

#include <optional>
#include <string>
#include <vector>

#include <torch/library.h>
#include <torch/types.h>

#include "../fused/fused.h"
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

// ---------------------------------------------------------------- helpers

inline int layout_flag(c10::string_view s)
{
    return static_cast<int>(parse_tensor_layout(s));
}
inline int gran_flag(c10::string_view s)
{
    QuantGran g = parse_quant_gran(s);
    TORCH_CHECK(g == QuantGran::kPerWarp || g == QuantGran::kPerThread,
                "qk_quant_gran must be \"per_warp\" or \"per_thread\" for attention ops");
    return static_cast<int>(g);
}

// ---------------------------------------------------------------- fused ops

void quant_per_block_int8(const at::Tensor&                input,
                          const std::optional<at::Tensor>& mean,
                          at::Tensor&                      output,
                          at::Tensor&                      scale,
                          std::optional<double>            sm_scale,
                          int64_t                          block_size,
                          c10::string_view                 tensor_layout)
{
    const int layout_int = layout_flag(tensor_layout);
    if (mean.has_value()) {
        TORCH_CHECK(!sm_scale.has_value(),
                    "quant_per_block_int8: sm_scale and mean are mutually exclusive "
                    "(the fused sub-mean path has no sm_scale)");
        quant_per_block_int8_fuse_sub_mean_cuda(input, *mean, output, scale, static_cast<int>(block_size), layout_int);
    }
    else if (sm_scale.has_value()) {
        quant_per_block_int8_cuda(
            input, output, scale, static_cast<float>(*sm_scale), static_cast<int>(block_size), layout_int);
    }
    else {
        quant_per_block_int8_cuda(input, output, scale, static_cast<int>(block_size), layout_int);
    }
}

void quant_per_warp_int8(const at::Tensor& input,
                         at::Tensor&       output,
                         at::Tensor&       scale,
                         int64_t           block_size,
                         int64_t           warp_block_size,
                         c10::string_view  tensor_layout)
{
    quant_per_warp_int8_cuda(input,
                             output,
                             scale,
                             static_cast<int>(block_size),
                             static_cast<int>(warp_block_size),
                             layout_flag(tensor_layout));
}

void quant_per_thread_int8_q(const at::Tensor& input,
                             at::Tensor&       output,
                             at::Tensor&       scale,
                             int64_t           block_size,
                             int64_t           warp_block_size,
                             c10::string_view  tensor_layout)
{
    quant_per_thread_int8_q_cuda(input,
                                 output,
                                 scale,
                                 static_cast<int>(block_size),
                                 static_cast<int>(warp_block_size),
                                 layout_flag(tensor_layout));
}

void quant_per_thread_int8_k(const at::Tensor&                input,
                             const std::optional<at::Tensor>& mean,
                             at::Tensor&                      output,
                             at::Tensor&                      scale,
                             int64_t                          block_size,
                             int64_t                          warp_block_size,
                             c10::string_view                 tensor_layout)
{
    const int layout_int = layout_flag(tensor_layout);
    if (mean.has_value()) {
        quant_per_thread_int8_k_fuse_sub_mean_cuda(
            input, *mean, output, scale, static_cast<int>(block_size), static_cast<int>(warp_block_size), layout_int);
    }
    else {
        quant_per_thread_int8_k_cuda(
            input, output, scale, static_cast<int>(block_size), static_cast<int>(warp_block_size), layout_int);
    }
}

void sub_mean(const at::Tensor& input, const at::Tensor& mean, at::Tensor& output, c10::string_view tensor_layout)
{
    sub_mean_cuda(input, mean, output, layout_flag(tensor_layout));
}

void transpose_pad_v(const at::Tensor& input, at::Tensor& output, c10::string_view tensor_layout, bool permute)
{
    if (permute) {
        transpose_pad_permute_cuda(input, output, layout_flag(tensor_layout));
    }
    else {
        transpose_pad_cuda(input, output, layout_flag(tensor_layout));
    }
}

void scale_fuse_quant(const at::Tensor& input,
                      at::Tensor&       output,
                      at::Tensor&       scale,
                      int64_t           num_tokens,
                      double            scale_max,
                      c10::string_view  tensor_layout)
{
    scale_fuse_quant_cuda(
        input, output, scale, static_cast<int>(num_tokens), static_cast<float>(scale_max), layout_flag(tensor_layout));
}

void mean_scale_fuse_quant(const at::Tensor& input,
                           at::Tensor&       output,
                           at::Tensor&       mean,
                           at::Tensor&       scale,
                           int64_t           num_tokens,
                           double            scale_max,
                           c10::string_view  tensor_layout)
{
    mean_scale_fuse_quant_cuda(input,
                               output,
                               mean,
                               scale,
                               static_cast<int>(num_tokens),
                               static_cast<float>(scale_max),
                               layout_flag(tensor_layout));
}

// ------------------------------------------------- per-arch attention shims
// The kernel launchers keep their historical int-flag signatures; these shims
// translate the string/bool op arguments once per call.

using BaseFn   = torch::Tensor (*)(torch::Tensor,
                                 torch::Tensor,
                                 torch::Tensor,
                                 torch::Tensor,
                                 torch::Tensor,
                                 torch::Tensor,
                                 int,
                                 int,
                                 int,
                                 float,
                                 int);
using ExtraFn  = torch::Tensor (*)(torch::Tensor,
                                  torch::Tensor,
                                  torch::Tensor,
                                  torch::Tensor,
                                  torch::Tensor,
                                  torch::Tensor,
                                  torch::Tensor,
                                  int,
                                  int,
                                  int,
                                  float,
                                  int);
using Extra2Fn = torch::Tensor (*)(torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   int,
                                   int,
                                   int,
                                   float,
                                   int);

template<BaseFn fn>
at::Tensor qattn_base(const at::Tensor& query,
                      const at::Tensor& key,
                      const at::Tensor& value,
                      at::Tensor&       output,
                      const at::Tensor& query_scale,
                      const at::Tensor& key_scale,
                      c10::string_view  tensor_layout,
                      bool              is_causal,
                      c10::string_view  qk_quant_gran,
                      double            sm_scale,
                      bool              return_lse)
{
    return fn(query,
              key,
              value,
              output,
              query_scale,
              key_scale,
              layout_flag(tensor_layout),
              is_causal ? 1 : 0,
              gran_flag(qk_quant_gran),
              static_cast<float>(sm_scale),
              return_lse ? 1 : 0);
}

template<ExtraFn fn>
at::Tensor qattn_extra(const at::Tensor& query,
                       const at::Tensor& key,
                       const at::Tensor& value,
                       at::Tensor&       output,
                       const at::Tensor& query_scale,
                       const at::Tensor& key_scale,
                       const at::Tensor& extra,
                       c10::string_view  tensor_layout,
                       bool              is_causal,
                       c10::string_view  qk_quant_gran,
                       double            sm_scale,
                       bool              return_lse)
{
    return fn(query,
              key,
              value,
              output,
              query_scale,
              key_scale,
              extra,
              layout_flag(tensor_layout),
              is_causal ? 1 : 0,
              gran_flag(qk_quant_gran),
              static_cast<float>(sm_scale),
              return_lse ? 1 : 0);
}

template<Extra2Fn fn>
at::Tensor qattn_extra2(const at::Tensor& query,
                        const at::Tensor& key,
                        const at::Tensor& value,
                        at::Tensor&       output,
                        const at::Tensor& query_scale,
                        const at::Tensor& key_scale,
                        const at::Tensor& value_scale,
                        const at::Tensor& value_mean,
                        c10::string_view  tensor_layout,
                        bool              is_causal,
                        c10::string_view  qk_quant_gran,
                        double            sm_scale,
                        bool              return_lse)
{
    return fn(query,
              key,
              value,
              output,
              query_scale,
              key_scale,
              value_scale,
              value_mean,
              layout_flag(tensor_layout),
              is_causal ? 1 : 0,
              gran_flag(qk_quant_gran),
              static_cast<float>(sm_scale),
              return_lse ? 1 : 0);
}

// ---------------------------------------------------------------- host ops

std::vector<int64_t> compiled_archs()
{
    std::vector<int64_t> archs;
#if SAGEATTN_BUILD_SM80
    archs.push_back(80);
#endif
#if SAGEATTN_BUILD_SM89
    archs.push_back(89);
#endif
#if SAGEATTN_BUILD_SM90
    archs.push_back(90);
#endif
#if SAGEATTN_BUILD_SM100
    archs.push_back(100);
#endif
#if SAGEATTN_BUILD_SM120
    archs.push_back(120);
#endif
    return archs;
}

// Full dispatch decision for one (cc, head_dim, requests) tuple. Pure host —
// primed into a Python dict at import time (returning str from inside a
// traced region breaks fullgraph, so Python never calls this in hot code).
std::tuple<std::string,
           std::string,
           std::string,
           bool,
           bool,
           bool,
           std::string,
           int64_t,
           double,
           bool,
           bool,
           int64_t,
           int64_t,
           int64_t,
           int64_t,
           std::string>
plan(int64_t                         cc_major,
     int64_t                         cc_minor,
     int64_t                         head_dim,
     std::optional<c10::string_view> backend,
     std::optional<c10::string_view> qk_quant_gran,
     std::optional<c10::string_view> pv_accum_dtype,
     std::optional<bool>             smooth_v)
{
    std::optional<Backend>   req_backend;
    std::optional<QuantGran> req_gran;
    std::optional<PVAccum>   req_pv;
    if (backend)
        req_backend = parse_backend(*backend);
    if (qk_quant_gran)
        req_gran = parse_quant_gran(*qk_quant_gran);
    if (pv_accum_dtype)
        req_pv = parse_pv_accum(*pv_accum_dtype);

    Plan p = resolve(CC{static_cast<int>(cc_major), static_cast<int>(cc_minor)},
                     static_cast<int>(head_dim),
                     req_backend,
                     req_gran,
                     req_pv,
                     smooth_v);
    return {name(p.backend),
            name(p.gran),
            name(p.pv),
            p.smooth_v,
            p.smooth_v_ignored,
            p.pv_fp8,
            name(p.v_layout),
            p.v_pad_multiple,
            p.v_scale_max,
            p.need_value_scale,
            p.need_value_mean,
            p.blk_q,
            p.warp_q,
            p.blk_k,
            p.warp_k,
            p.error};
}

}  // namespace
}  // namespace sage

// ---------------------------------------------------------------- registry

TORCH_LIBRARY(sageattention, m)
{
    // host queries (catch-all: no tensors involved)
    m.def("compiled_archs() -> int[]", TORCH_FN(sage::compiled_archs));
    m.def("plan(int cc_major, int cc_minor, int head_dim, str? backend=None, "
          "str? qk_quant_gran=None, str? pv_accum_dtype=None, bool? smooth_v=None) "
          "-> (str backend, str qk_quant_gran, str pv_accum_dtype, bool smooth_v, "
          "bool smooth_v_ignored, bool pv_fp8, str v_layout, int v_pad_multiple, "
          "float v_scale_max, bool need_value_scale, bool need_value_mean, "
          "int blk_q, int warp_q, int blk_k, int warp_k, str error)",
          TORCH_FN(sage::plan));

    // low-level quantization (mutating, no fake kernels needed)
    m.def("quant_per_block_int8(Tensor input, Tensor? mean, Tensor(a!) output, "
          "Tensor(b!) scale, float? sm_scale, int block_size, str tensor_layout) -> ()");
    m.def("quant_per_warp_int8(Tensor input, Tensor(a!) output, Tensor(b!) scale, "
          "int block_size, int warp_block_size, str tensor_layout) -> ()");
    m.def("quant_per_thread_int8_q(Tensor input, Tensor(a!) output, Tensor(b!) scale, "
          "int block_size, int warp_block_size, str tensor_layout) -> ()");
    m.def("quant_per_thread_int8_k(Tensor input, Tensor? mean, Tensor(a!) output, "
          "Tensor(b!) scale, int block_size, int warp_block_size, str tensor_layout) -> ()");
    m.def("sub_mean(Tensor input, Tensor mean, Tensor(a!) output, str tensor_layout) -> ()");
    m.def("transpose_pad_v(Tensor input, Tensor(a!) output, str tensor_layout, "
          "bool permute) -> ()");
    m.def("scale_fuse_quant(Tensor input, Tensor(a!) output, Tensor(b!) scale, "
          "int num_tokens, float scale_max, str tensor_layout) -> ()");
    m.def("mean_scale_fuse_quant(Tensor input, Tensor(a!) output, Tensor(b!) mean, "
          "Tensor(c!) scale, int num_tokens, float scale_max, str tensor_layout) -> ()");

    // per-arch attention ops (transitional 1:1 surface; the unified fwd op
    // dispatches onto the same launchers)
#define SAGE_QATTN_BASE_SCHEMA(op)                                                                                     \
#op "(Tensor query, Tensor key, Tensor value, Tensor(a!) output, "                                                 \
        "Tensor query_scale, Tensor key_scale, str tensor_layout, "                                                    \
        "bool is_causal, str qk_quant_gran, float sm_scale, bool return_lse)"                                          \
        " -> Tensor"
#define SAGE_QATTN_EXTRA_SCHEMA(op, extra)                                                                             \
#op "(Tensor query, Tensor key, Tensor value, Tensor(a!) output, "                                                 \
        "Tensor query_scale, Tensor key_scale, Tensor " extra ", str tensor_layout, "                                  \
        "bool is_causal, str qk_quant_gran, float sm_scale, bool return_lse)"                                          \
        " -> Tensor"
#define SAGE_QATTN_EXTRA2_SCHEMA(op)                                                                                   \
#op "(Tensor query, Tensor key, Tensor value, Tensor(a!) output, "                                                 \
        "Tensor query_scale, Tensor key_scale, Tensor value_scale, "                                                   \
        "Tensor value_mean, str tensor_layout, "                                                                       \
        "bool is_causal, str qk_quant_gran, float sm_scale, bool return_lse)"                                          \
        " -> Tensor"

#if SAGEATTN_BUILD_SM80
    m.def(SAGE_QATTN_BASE_SCHEMA(qattn_sm80_qk_int8_sv_f16_accum_f32_attn));
    m.def(SAGE_QATTN_BASE_SCHEMA(qattn_sm80_qk_int8_sv_f16_accum_f16_attn));
    m.def(SAGE_QATTN_BASE_SCHEMA(qattn_sm80_qk_int8_sv_f16_accum_f16_attn_inst_buf));
    m.def(SAGE_QATTN_EXTRA_SCHEMA(qattn_sm80_qk_int8_sv_f16_accum_f16_fuse_v_mean_attn, "value_mean"));
#endif
#if SAGEATTN_BUILD_SM89
    m.def(SAGE_QATTN_EXTRA_SCHEMA(qattn_sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn, "value_scale"));
    m.def(SAGE_QATTN_EXTRA2_SCHEMA(qattn_sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn));
    m.def(SAGE_QATTN_EXTRA_SCHEMA(qattn_sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf, "value_scale"));
    m.def(SAGE_QATTN_EXTRA_SCHEMA(qattn_sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf, "value_scale"));
#endif
#if SAGEATTN_BUILD_SM90
    m.def(SAGE_QATTN_BASE_SCHEMA(qattn_sm90_qk_int8_sv_f8_accum_f32_attn_inst_buf));
    m.def(SAGE_QATTN_EXTRA_SCHEMA(qattn_sm90_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf, "value_scale"));
#endif
#if SAGEATTN_BUILD_SM100
    m.def(SAGE_QATTN_BASE_SCHEMA(qattn_sm100_qk_int8_sv_f8_accum_f32_attn));
    m.def(SAGE_QATTN_EXTRA_SCHEMA(qattn_sm100_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn, "value_scale"));
#endif
#if SAGEATTN_BUILD_SM120
    m.def(SAGE_QATTN_EXTRA_SCHEMA(qattn_sm120_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn, "value_scale"));
    m.def(SAGE_QATTN_EXTRA2_SCHEMA(qattn_sm120_qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn));
    m.def(SAGE_QATTN_EXTRA_SCHEMA(qattn_sm120_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf, "value_scale"));
#endif
}

TORCH_LIBRARY_IMPL(sageattention, CUDA, m)
{
    m.impl("quant_per_block_int8", TORCH_FN(sage::quant_per_block_int8));
    m.impl("quant_per_warp_int8", TORCH_FN(sage::quant_per_warp_int8));
    m.impl("quant_per_thread_int8_q", TORCH_FN(sage::quant_per_thread_int8_q));
    m.impl("quant_per_thread_int8_k", TORCH_FN(sage::quant_per_thread_int8_k));
    m.impl("sub_mean", TORCH_FN(sage::sub_mean));
    m.impl("transpose_pad_v", TORCH_FN(sage::transpose_pad_v));
    m.impl("scale_fuse_quant", TORCH_FN(sage::scale_fuse_quant));
    m.impl("mean_scale_fuse_quant", TORCH_FN(sage::mean_scale_fuse_quant));

#if SAGEATTN_BUILD_SM80
    m.impl("qattn_sm80_qk_int8_sv_f16_accum_f32_attn",
           TORCH_FN(sage::qattn_base<sage::sm80::qk_int8_sv_f16_accum_f32_attn>));
    m.impl("qattn_sm80_qk_int8_sv_f16_accum_f16_attn",
           TORCH_FN(sage::qattn_base<sage::sm80::qk_int8_sv_f16_accum_f16_attn>));
    m.impl("qattn_sm80_qk_int8_sv_f16_accum_f16_attn_inst_buf",
           TORCH_FN(sage::qattn_base<sage::sm80::qk_int8_sv_f16_accum_f16_attn_inst_buf>));
    m.impl("qattn_sm80_qk_int8_sv_f16_accum_f16_fuse_v_mean_attn",
           TORCH_FN(sage::qattn_extra<sage::sm80::qk_int8_sv_f16_accum_f16_fuse_v_mean_attn>));
#endif
#if SAGEATTN_BUILD_SM89
    m.impl("qattn_sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn",
           TORCH_FN(sage::qattn_extra<sage::sm89::qk_int8_sv_f8_accum_f32_fuse_v_scale_attn>));
    m.impl("qattn_sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn",
           TORCH_FN(sage::qattn_extra2<sage::sm89::qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn>));
    m.impl("qattn_sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf",
           TORCH_FN(sage::qattn_extra<sage::sm89::qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf>));
    m.impl("qattn_sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf",
           TORCH_FN(sage::qattn_extra<sage::sm89::qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf>));
#endif
#if SAGEATTN_BUILD_SM90
    m.impl("qattn_sm90_qk_int8_sv_f8_accum_f32_attn_inst_buf",
           TORCH_FN(sage::qattn_base<sage::sm90::qk_int8_sv_f8_accum_f32_attn_inst_buf>));
    m.impl("qattn_sm90_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf",
           TORCH_FN(sage::qattn_extra<sage::sm90::qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf>));
#endif
#if SAGEATTN_BUILD_SM100
    m.impl("qattn_sm100_qk_int8_sv_f8_accum_f32_attn",
           TORCH_FN(sage::qattn_base<sage::sm100::qk_int8_sv_f8_accum_f32_attn>));
    m.impl("qattn_sm100_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn",
           TORCH_FN(sage::qattn_extra<sage::sm100::qk_int8_sv_f8_accum_f32_fuse_v_scale_attn>));
#endif
#if SAGEATTN_BUILD_SM120
    m.impl("qattn_sm120_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn",
           TORCH_FN(sage::qattn_extra<sage::sm120::qk_int8_sv_f8_accum_f32_fuse_v_scale_attn>));
    m.impl("qattn_sm120_qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn",
           TORCH_FN(sage::qattn_extra2<sage::sm120::qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn>));
    m.impl("qattn_sm120_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf",
           TORCH_FN(sage::qattn_extra<sage::sm120::qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf>));
#endif
}

// plan / compiled_archs need a CompositeExplicitAutograd (catch-all) impl —
// they take no tensors, so the CUDA dispatch key never matches.
TORCH_LIBRARY_IMPL(sageattention, CompositeExplicitAutograd, m)
{
    // registered inline in TORCH_LIBRARY above via m.def(schema, fn)
}
