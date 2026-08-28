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

// Functional quantization composites. These own the output allocation and the
// scale-shape arithmetic that used to be duplicated between quant.py and the
// launchers' CHECK_SHAPEs, and sink the V seq padding that used to be a full
// torch.cat copy in Python (sm90/sm100). One graph node each under
// torch.compile instead of 2-4 mutating ops plus empty()s.

#include <optional>
#include <tuple>

#include <ATen/ceil_div.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/types.h>

#include "../fused/fused.h"
#include "config.h"

namespace sage {
namespace {

// (seq_dim, nh_dim) for a rank-4 qkv tensor under the given layout.
inline std::pair<int64_t, int64_t> seq_nh_dims(TensorLayout layout)
{
    return layout == TensorLayout::kNHD ? std::make_pair<int64_t, int64_t>(1, 2) :
                                          std::make_pair<int64_t, int64_t>(2, 1);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> quant_qk_cuda(const at::Tensor&                query,
                                                                         const at::Tensor&                key,
                                                                         const std::optional<at::Tensor>& key_mean,
                                                                         c10::string_view                 tensor_layout,
                                                                         c10::string_view                 qk_quant_gran,
                                                                         int64_t                          blk_q,
                                                                         int64_t                          warp_q,
                                                                         int64_t                          blk_k,
                                                                         int64_t                          warp_k)
{
    const c10::cuda::CUDAGuard device_guard(query.device());
    const TensorLayout         layout = parse_tensor_layout(tensor_layout);
    const QuantGran            gran   = parse_quant_gran(qk_quant_gran);
    TORCH_CHECK(gran == QuantGran::kPerWarp || gran == QuantGran::kPerThread,
                "quant_qk supports per_warp / per_thread");
    TORCH_CHECK(query.dim() == 4 && key.dim() == 4, "query/key must be 4-D");

    auto [seq_dim, nh_dim] = seq_nh_dims(layout);
    const int64_t bsz      = query.size(0);
    const int64_t qo_len   = query.size(seq_dim);
    const int64_t kv_len   = key.size(seq_dim);
    const int64_t h_qo     = query.size(nh_dim);
    const int64_t h_kv     = key.size(nh_dim);

    int64_t q_scale_len, k_scale_len;
    if (gran == QuantGran::kPerWarp) {
        q_scale_len = at::ceil_div<int64_t>(qo_len, blk_q) * (blk_q / warp_q);
        k_scale_len = at::ceil_div<int64_t>(kv_len, blk_k);
    }
    else {
        q_scale_len = at::ceil_div<int64_t>(qo_len, blk_q) * (blk_q / warp_q) * 8;
        k_scale_len = at::ceil_div<int64_t>(kv_len, blk_k) * (blk_k / warp_k) * 4;
    }

    at::Tensor q_int8  = at::empty(query.sizes(), query.options().dtype(at::kChar));
    at::Tensor k_int8  = at::empty(key.sizes(), key.options().dtype(at::kChar));
    at::Tensor q_scale = at::empty({bsz, h_qo, q_scale_len}, query.options().dtype(at::kFloat));
    at::Tensor k_scale = at::empty({bsz, h_kv, k_scale_len}, key.options().dtype(at::kFloat));

    const int                 lf = static_cast<int>(layout);
    std::optional<at::Tensor> km;
    if (key_mean.has_value()) {
        // core passes k.mean(dim=seq_dim, keepdim=True); the kernels take it squeezed
        km = key_mean->dim() == 4 ? key_mean->squeeze(seq_dim) : *key_mean;
    }

    if (gran == QuantGran::kPerWarp) {
        quant_per_warp_int8_cuda(query, q_int8, q_scale, static_cast<int>(blk_q), static_cast<int>(warp_q), lf);
        if (km.has_value()) {
            quant_per_block_int8_fuse_sub_mean_cuda(key, *km, k_int8, k_scale, static_cast<int>(blk_k), lf);
        }
        else {
            quant_per_block_int8_cuda(key, k_int8, k_scale, static_cast<int>(blk_k), lf);
        }
    }
    else {
        quant_per_thread_int8_q_cuda(query, q_int8, q_scale, static_cast<int>(blk_q), static_cast<int>(warp_q), lf);
        if (km.has_value()) {
            quant_per_thread_int8_k_fuse_sub_mean_cuda(
                key, *km, k_int8, k_scale, static_cast<int>(blk_k), static_cast<int>(warp_k), lf);
        }
        else {
            quant_per_thread_int8_k_cuda(key, k_int8, k_scale, static_cast<int>(blk_k), static_cast<int>(warp_k), lf);
        }
    }
    return {q_int8, q_scale, k_int8, k_scale};
}

std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>> quant_v_fp8_cuda(const at::Tensor& value,
                                                                               c10::string_view  tensor_layout,
                                                                               c10::string_view  v_layout,
                                                                               double            scale_max,
                                                                               bool              smooth_v,
                                                                               int64_t           pad_multiple)
{
    const c10::cuda::CUDAGuard device_guard(value.device());
    const TensorLayout         layout = parse_tensor_layout(tensor_layout);
    const VLayout              vl     = parse_v_layout(v_layout);
    TORCH_CHECK(vl != VLayout::kSeq,
                "quant_v_fp8 produces transposed layouts "
                "(\"mma_k16\" / \"linear\"); the fp16 PV path takes V unquantized");
    TORCH_CHECK(pad_multiple == 64 || pad_multiple == 128, "pad_multiple must be 64 or 128, got ", pad_multiple);
    TORCH_CHECK(value.dim() == 4, "value must be 4-D");

    auto [seq_dim, nh_dim] = seq_nh_dims(layout);
    const int64_t bsz      = value.size(0);
    const int64_t kv_len   = value.size(seq_dim);
    const int64_t h_kv     = value.size(nh_dim);
    const int64_t head_dim = value.size(3);
    // Replaces the old Python-side torch.cat zero pad (a full V copy) for the
    // 128-aligned families: the transpose kernel zero-fills the padded tokens it
    // writes (pred_load kFillZero). It writes the 64-aligned prefix only, so for
    // pad_multiple=128 the [64-aligned, 128-aligned) tail of v_t stays
    // uninitialized -- nothing reads it, the quantization pass below stops at the
    // same 64-aligned bound and the fp8 tail comes from at::zeros.
    const int64_t padded_len = at::round_up<int64_t>(kv_len, pad_multiple);

    at::Tensor v_t;  // [B, H, D, padded] (HND) / [B, D, H, padded] (NHD)
    if (layout == TensorLayout::kHND) {
        v_t = at::empty({bsz, h_kv, head_dim, padded_len}, value.options());
    }
    else {
        v_t = at::empty({bsz, head_dim, h_kv, padded_len}, value.options());
    }
    const int lf = static_cast<int>(layout);
    if (vl == VLayout::kMmaK16) {
        transpose_pad_permute_cuda(value, v_t, lf);
    }
    else {
        transpose_pad_cuda(value, v_t, lf);
    }

    // zeros, not empty: the fp8 quant kernel writes tokens up to the 64-aligned
    // bound only; for pad_multiple=128 the [64-aligned, 128-aligned) tail must
    // still be valid fp8 (a NaN-encoded garbage byte would poison PV through
    // 0 * NaN). Zero bytes are exactly what quantizing the zero-padded input
    // produces, so this stays bit-identical to the old cat-then-quantize path.
    at::Tensor v_fp8   = at::zeros(v_t.sizes(), v_t.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor v_scale = at::empty({bsz, h_kv, head_dim}, value.options().dtype(at::kFloat));

    if (smooth_v) {
        at::Tensor vm = at::empty({bsz, h_kv, head_dim}, value.options().dtype(at::kFloat));
        mean_scale_fuse_quant_cuda(
            v_t, v_fp8, vm, v_scale, static_cast<int>(kv_len), static_cast<float>(scale_max), lf);
        return {v_fp8, v_scale, vm};
    }
    scale_fuse_quant_cuda(v_t, v_fp8, v_scale, static_cast<int>(kv_len), static_cast<float>(scale_max), lf);
    return {v_fp8, v_scale, std::nullopt};
}

std::tuple<at::Tensor, at::Tensor> sub_mean_v_cuda(const at::Tensor& value, c10::string_view tensor_layout)
{
    const c10::cuda::CUDAGuard device_guard(value.device());
    const TensorLayout         layout = parse_tensor_layout(tensor_layout);
    auto [seq_dim, nh_dim]            = seq_nh_dims(layout);
    (void)nh_dim;
    at::Tensor vm       = value.mean(seq_dim);
    at::Tensor smoothed = at::empty(value.sizes(), value.options().dtype(at::kHalf));
    sub_mean_cuda(value, vm, smoothed, static_cast<int>(layout));
    return {smoothed, vm};
}

}  // namespace
}  // namespace sage

TORCH_LIBRARY_FRAGMENT(sageattention, m)
{
    m.def("quant_qk(Tensor query, Tensor key, Tensor? key_mean=None, *, "
          "str tensor_layout=\"HND\", str qk_quant_gran=\"per_thread\", "
          "int blk_q=128, int warp_q=32, int blk_k=64, int warp_k=64) "
          "-> (Tensor q_int8, Tensor q_scale, Tensor k_int8, Tensor k_scale)");
    m.def("quant_v_fp8(Tensor value, *, str tensor_layout=\"HND\", "
          "str v_layout=\"mma_k16\", float scale_max=448.0, bool smooth_v=False, "
          "int pad_multiple=64) -> (Tensor v_fp8, Tensor v_scale, Tensor? v_mean)");
    m.def("sub_mean_v(Tensor value, *, str tensor_layout=\"HND\") "
          "-> (Tensor v_smoothed, Tensor v_mean)");
}

TORCH_LIBRARY_IMPL(sageattention, CUDA, m)
{
    m.impl("quant_qk", TORCH_FN(sage::quant_qk_cuda));
    m.impl("quant_v_fp8", TORCH_FN(sage::quant_v_fp8_cuda));
    m.impl("sub_mean_v", TORCH_FN(sage::sub_mean_v_cuda));
}
