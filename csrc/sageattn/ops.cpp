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
//   - host query ops: plan (dispatch decision table) and compiled_archs.
//
// This TU is host-only: it sees the fused.h declarations; kernels live in
// their own TUs. The attention ops (fwd / fwd_varlen) register in theirs.

#include <optional>
#include <string>
#include <vector>

#include <torch/library.h>
#include <torch/types.h>

#include "../fused/fused.h"
#include "config.h"
#include "plan.h"
#include "sageattn_build_config.h"
#include "varlen_check.h"

namespace sage {
namespace {

// ---------------------------------------------------------------- helpers

inline int layout_flag(c10::string_view s)
{
    return static_cast<int>(parse_tensor_layout(s));
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

// cu_seqlens switches the packed layout on: 3-D [total_tokens, heads,
// head_dim] in, [heads, head_dim, blk_total * pad_multiple] out. It is the one
// piece of the fp8 V pipeline that is pure fp16 arithmetic, so it is also the
// only piece whose packed addressing can be checked on a pre-sm_89 device -
// hence the argument here and not only inside quant_v_fp8_varlen.
void transpose_pad_v(const at::Tensor&                input,
                     at::Tensor&                      output,
                     c10::string_view                 tensor_layout,
                     bool                             permute,
                     const std::optional<at::Tensor>& cu_seqlens,
                     c10::SymInt                      max_seqlen,
                     int64_t                          pad_multiple)
{
    QuantVarlen varlen;
    if (cu_seqlens.has_value()) {
        check_cu_seqlens(*cu_seqlens, input, "cu_seqlens");
        varlen.cu_seqlens = cu_seqlens->data_ptr<int32_t>();
        varlen.batch_size = cu_seqlens->size(0) - 1;
        varlen.max_seqlen = max_seqlen.guard_int(__FILE__, __LINE__);
        varlen.pad_tokens = pad_multiple;
    }
    if (permute) {
        transpose_pad_permute_cuda(input, output, layout_flag(tensor_layout), varlen);
    }
    else {
        transpose_pad_cuda(input, output, layout_flag(tensor_layout), varlen);
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
     std::optional<bool>             smooth_v,
     bool                            varlen)
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
                     smooth_v,
                     varlen);
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
          "str? qk_quant_gran=None, str? pv_accum_dtype=None, bool? smooth_v=None, "
          "bool varlen=False) "
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
          "bool permute, Tensor? cu_seqlens=None, *, SymInt max_seqlen=0, "
          "int pad_multiple=64) -> ()");
    m.def("scale_fuse_quant(Tensor input, Tensor(a!) output, Tensor(b!) scale, "
          "int num_tokens, float scale_max, str tensor_layout) -> ()");
    m.def("mean_scale_fuse_quant(Tensor input, Tensor(a!) output, Tensor(b!) mean, "
          "Tensor(c!) scale, int num_tokens, float scale_max, str tensor_layout) -> ()");
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
}

// plan / compiled_archs need a CompositeExplicitAutograd (catch-all) impl —
// they take no tensors, so the CUDA dispatch key never matches.
TORCH_LIBRARY_IMPL(sageattention, CompositeExplicitAutograd, m)
{
    // registered inline in TORCH_LIBRARY above via m.def(schema, fn)
}
