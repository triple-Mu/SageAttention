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
#include <vector>

#include <ATen/ceil_div.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/types.h>

#include "../fused/fused.h"
#include "arch.h"
#include "config.h"
#include "sageattn_build_config.h"
#include "varlen.h"
#include "varlen_check.h"

namespace sage {
namespace {

// The fused kernel has no V^T to hold its first read in, so it reads V twice,
// and its gather moves 32-byte pieces of a token row rather than whole lines.
// That pays for itself only while the second pass still finds the first pass's
// bytes in L2, so the turnover point is a cache property and per-arch:
//  - sm_120 (V preparation alone, per-shape medians, b8 h16 d128): -41% at
//    512 tokens, -54% at 2048, -9% at 4096, then +10% at 5120 and +45% at
//    8192. Two of the kernel's 2048-token rounds is where it turns over.
//  - sm_89 (L20, 96 MB L2; bench/P4_VFUSE_L20.md): fused wins at every
//    b*h in {32,64,128,256} through 12288 padded tokens (0.80-0.85 of the
//    two-kernel path even at 12288) and turns over at 16384 (1.04-1.10);
//    12288 is the last grid point every arm still wins.
//  - other arches: not swept yet, they inherit the sm_120 value.
inline int64_t fused_v_quant_max_tokens(c10::DeviceIndex device)
{
    const CC cc = device_cc(device);
    if (cc == CC{8, 9}) {
        return 12288;
    }
    return 4096;
}

// Whether the two quant_v_fp8 composites below run the transpose and the fp8
// quantization as one kernel instead of two. Both produce the same bits (the
// fused kernel keeps the two-pass reduction order and permute), so every term
// here is about timing or about what the kernel can address.
inline bool use_fused_v_quant(int64_t head_dim, int64_t padded_tokens, c10::DeviceIndex device)
{
    return SAGEATTN_FUSED_V_QUANT != 0 && padded_tokens <= fused_v_quant_max_tokens(device)
           && transpose_quant_v_fp8_supported(head_dim);
}

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

    auto [seq_dim, nh_dim]     = seq_nh_dims(layout);
    const int64_t batch_size   = query.size(0);
    const int64_t qo_len       = query.size(seq_dim);
    const int64_t kv_len       = key.size(seq_dim);
    const int64_t num_qo_heads = query.size(nh_dim);
    const int64_t num_kv_heads = key.size(nh_dim);

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
    at::Tensor q_scale = at::empty({batch_size, num_qo_heads, q_scale_len}, query.options().dtype(at::kFloat));
    at::Tensor k_scale = at::empty({batch_size, num_kv_heads, k_scale_len}, key.options().dtype(at::kFloat));

    const int                 layout_int = static_cast<int>(layout);
    std::optional<at::Tensor> key_mean_squeezed;
    if (key_mean.has_value()) {
        // core passes k.mean(dim=seq_dim, keepdim=True); the kernels take it squeezed
        key_mean_squeezed = key_mean->dim() == 4 ? key_mean->squeeze(seq_dim) : *key_mean;
    }

    if (gran == QuantGran::kPerWarp) {
        quant_per_warp_int8_cuda(query, q_int8, q_scale, static_cast<int>(blk_q), static_cast<int>(warp_q), layout_int);
        if (key_mean_squeezed.has_value()) {
            quant_per_block_int8_fuse_sub_mean_cuda(
                key, *key_mean_squeezed, k_int8, k_scale, static_cast<int>(blk_k), layout_int);
        }
        else {
            quant_per_block_int8_cuda(key, k_int8, k_scale, static_cast<int>(blk_k), layout_int);
        }
    }
    else {
        quant_per_thread_int8_q_cuda(
            query, q_int8, q_scale, static_cast<int>(blk_q), static_cast<int>(warp_q), layout_int);
        if (key_mean_squeezed.has_value()) {
            quant_per_thread_int8_k_fuse_sub_mean_cuda(key,
                                                       *key_mean_squeezed,
                                                       k_int8,
                                                       k_scale,
                                                       static_cast<int>(blk_k),
                                                       static_cast<int>(warp_k),
                                                       layout_int);
        }
        else {
            quant_per_thread_int8_k_cuda(
                key, k_int8, k_scale, static_cast<int>(blk_k), static_cast<int>(warp_k), layout_int);
        }
    }
    return {q_int8, q_scale, k_int8, k_scale};
}

// quant_qk's packed [total_tokens, heads, head_dim] counterpart. The kernels
// are the same ones (a null cu_seqlens is what makes them dense), so the only
// thing that changes here is the addressing: no tensor_layout (packed 3-D has
// one meaning, and a layout difference degenerates to a stride), the scale
// tensors lose their batch dimension, and their length is the block algebra of
// varlen.h rather than ceil_div over a single sequence length.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
quant_qk_varlen_cuda(const at::Tensor&                query,
                     const at::Tensor&                key,
                     const at::Tensor&                cu_seqlens_q,
                     const at::Tensor&                cu_seqlens_k,
                     const std::optional<at::Tensor>& key_mean,
                     c10::SymInt                      max_seqlen_q,
                     c10::SymInt                      max_seqlen_k,
                     c10::string_view                 qk_quant_gran,
                     int64_t                          blk_q,
                     int64_t                          warp_q,
                     int64_t                          blk_k,
                     int64_t                          warp_k)
{
    const c10::cuda::CUDAGuard device_guard(query.device());
    // The quantization kernels themselves are always compiled (dense and
    // varlen share them); refusing here keeps SAGE_BUILD_VARLEN=OFF from
    // offering half a pipeline.
    TORCH_CHECK(SAGEATTN_BUILD_VARLEN != 0, "sageattention was built without varlen support (SAGE_BUILD_VARLEN=OFF)");
    const QuantGran gran = parse_quant_gran(qk_quant_gran);
    TORCH_CHECK(gran == QuantGran::kPerWarp || gran == QuantGran::kPerThread,
                "quant_qk_varlen supports per_warp / per_thread");
    TORCH_CHECK(query.dim() == 3 && key.dim() == 3, "packed query/key must be 3-D [total_tokens, heads, head_dim]");
    check_cu_seqlens(cu_seqlens_q, query, "cu_seqlens_q");
    check_cu_seqlens(cu_seqlens_k, key, "cu_seqlens_k");
    TORCH_CHECK(cu_seqlens_q.size(0) == cu_seqlens_k.size(0),
                "cu_seqlens_q and cu_seqlens_k must describe the same batch, got lengths ",
                cu_seqlens_q.size(0),
                " and ",
                cu_seqlens_k.size(0));

    const int64_t batch_size   = cu_seqlens_q.size(0) - 1;
    const int64_t total_q      = query.size(0);
    const int64_t total_k      = key.size(0);
    const int64_t num_qo_heads = query.size(1);
    const int64_t num_kv_heads = key.size(1);
    const int64_t max_q        = max_seqlen_q.guard_int(__FILE__, __LINE__);
    const int64_t max_k        = max_seqlen_k.guard_int(__FILE__, __LINE__);

    const int64_t q_blocks = sage::blk_total(total_q, batch_size, blk_q);
    const int64_t k_blocks = sage::blk_total(total_k, batch_size, blk_k);
    int64_t       q_scale_len, k_scale_len;
    if (gran == QuantGran::kPerWarp) {
        q_scale_len = q_blocks * (blk_q / warp_q);
        k_scale_len = k_blocks;
    }
    else {
        q_scale_len = q_blocks * (blk_q / warp_q) * 8;
        k_scale_len = k_blocks * (blk_k / warp_k) * 4;
    }

    at::Tensor q_int8 = at::empty(query.sizes(), query.options().dtype(at::kChar));
    at::Tensor k_int8 = at::empty(key.sizes(), key.options().dtype(at::kChar));
    // zeros, not empty: a sequence owns ceil(len / blk) + 1 blocks at most
    // (varlen.h, Property 1), so up to one scale block per sequence is never
    // written. The attention kernel does not read those, but leaving them
    // uninitialized makes the op's output run-to-run unstable, which is a
    // torch.compile correctness failure (opcheck's test_aot_dispatch_dynamic
    // compares eager against traced values). The tensors are [heads, blocks].
    at::Tensor q_scale = at::zeros({num_qo_heads, q_scale_len}, query.options().dtype(at::kFloat));
    at::Tensor k_scale = at::zeros({num_kv_heads, k_scale_len}, key.options().dtype(at::kFloat));

    const QuantVarlen varlen_q{cu_seqlens_q.data_ptr<int32_t>(), batch_size, max_q};
    const QuantVarlen varlen_k{cu_seqlens_k.data_ptr<int32_t>(), batch_size, max_k};
    // tensor_layout is unused on the varlen path (the strides carry the layout);
    // pass the HND flag so the argument still has a legal value.
    const int layout_int = static_cast<int>(TensorLayout::kHND);

    if (gran == QuantGran::kPerWarp) {
        quant_per_warp_int8_cuda(
            query, q_int8, q_scale, static_cast<int>(blk_q), static_cast<int>(warp_q), layout_int, varlen_q);
        if (key_mean.has_value()) {
            quant_per_block_int8_fuse_sub_mean_cuda(
                key, *key_mean, k_int8, k_scale, static_cast<int>(blk_k), layout_int, varlen_k);
        }
        else {
            quant_per_block_int8_cuda(key, k_int8, k_scale, static_cast<int>(blk_k), layout_int, varlen_k);
        }
    }
    else {
        quant_per_thread_int8_q_cuda(
            query, q_int8, q_scale, static_cast<int>(blk_q), static_cast<int>(warp_q), layout_int, varlen_q);
        if (key_mean.has_value()) {
            quant_per_thread_int8_k_fuse_sub_mean_cuda(key,
                                                       *key_mean,
                                                       k_int8,
                                                       k_scale,
                                                       static_cast<int>(blk_k),
                                                       static_cast<int>(warp_k),
                                                       layout_int,
                                                       varlen_k);
        }
        else {
            quant_per_thread_int8_k_cuda(
                key, k_int8, k_scale, static_cast<int>(blk_k), static_cast<int>(warp_k), layout_int, varlen_k);
        }
    }
    return {q_int8, q_scale, k_int8, k_scale};
}

// The key_mean the fuse_sub_mean branches above consume, computed per
// sequence: the varlen form of the dense k.mean(dim=seq_dim). One kernel
// (two for a multi-chunk max_seqlen) instead of the ATen composite that used
// to build it - repeat_interleave segment ids plus a float32 index_add_ over
// the whole packed tensor - whose atomics also made the result run-to-run
// non-deterministic. This one reduces in a fixed order.
at::Tensor segment_mean_varlen_cuda(const at::Tensor& input, const at::Tensor& cu_seqlens, c10::SymInt max_seqlen)
{
    const c10::cuda::CUDAGuard device_guard(input.device());
    TORCH_CHECK(SAGEATTN_BUILD_VARLEN != 0, "sageattention was built without varlen support (SAGE_BUILD_VARLEN=OFF)");
    TORCH_CHECK(input.dim() == 3, "packed input must be 3-D [total_tokens, heads, head_dim]");
    check_cu_seqlens(cu_seqlens, input, "cu_seqlens");

    const int64_t batch_size = cu_seqlens.size(0) - 1;
    const int64_t max_len    = max_seqlen.guard_int(__FILE__, __LINE__);

    // at::empty is enough: every (sequence, head, channel) is written, an
    // empty sequence's entries included (they get zeros from a clamped
    // divisor), so the op's output is run-to-run stable.
    at::Tensor mean = at::empty({batch_size, input.size(1), input.size(2)}, input.options());

    const QuantVarlen varlen{cu_seqlens.data_ptr<int32_t>(), batch_size, max_len};
    segment_mean_cuda(input, mean, varlen);
    return mean;
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

    auto [seq_dim, nh_dim]     = seq_nh_dims(layout);
    const int64_t batch_size   = value.size(0);
    const int64_t kv_len       = value.size(seq_dim);
    const int64_t num_kv_heads = value.size(nh_dim);
    const int64_t head_dim     = value.size(3);
    // Replaces the old Python-side torch.cat zero pad (a full V copy) for the
    // 128-aligned families: the transpose kernel zero-fills the padded tokens it
    // writes (pred_load kFillZero). It writes the 64-aligned prefix only, so for
    // pad_multiple=128 the [64-aligned, 128-aligned) tail of value_t stays
    // uninitialized -- nothing reads it, the quantization pass below stops at the
    // same 64-aligned bound and the fp8 tail comes from at::zeros.
    const int64_t padded_kv_len = at::round_up<int64_t>(kv_len, pad_multiple);

    // [B, H, D, padded] (HND) / [B, D, H, padded] (NHD)
    const std::vector<int64_t> vt_sizes = layout == TensorLayout::kHND ?
                                              std::vector<int64_t>{batch_size, num_kv_heads, head_dim, padded_kv_len} :
                                              std::vector<int64_t>{batch_size, head_dim, num_kv_heads, padded_kv_len};

    // The fp8 quant kernel writes tokens up to the 64-aligned bound only. When
    // the buffer runs past it (pad_multiple=128 and a kv_len whose 128-aligned
    // bound is larger) that tail must still be valid fp8 -- a NaN-encoded
    // garbage byte would poison PV through 0 * NaN -- so it is zero-filled
    // first; zero bytes are exactly what quantizing the zero-padded input
    // produces, which is what keeps this bit-identical to the old
    // cat-then-quantize path. When the two bounds coincide the kernel covers
    // every byte and the fill is pure bandwidth: that is every sm89/sm120 call
    // (pad_multiple=64) plus any already 128-aligned kv_len.
    const bool fill_covered_by_quant = padded_kv_len == at::round_up<int64_t>(kv_len, 64);
    const auto v_fp8_options         = value.options().dtype(at::kFloat8_e4m3fn);
    at::Tensor v_fp8 = fill_covered_by_quant ? at::empty(vt_sizes, v_fp8_options) : at::zeros(vt_sizes, v_fp8_options);
    at::Tensor v_scale = at::empty({batch_size, num_kv_heads, head_dim}, value.options().dtype(at::kFloat));
    at::Tensor value_mean;
    if (smooth_v) {
        value_mean = at::empty({batch_size, num_kv_heads, head_dim}, value.options().dtype(at::kFloat));
    }

    const int  layout_int = static_cast<int>(layout);
    const bool permute    = vl == VLayout::kMmaK16;

    if (use_fused_v_quant(head_dim, padded_kv_len, value.get_device())) {
        transpose_quant_v_fp8_cuda(
            value, v_fp8, value_mean, v_scale, static_cast<float>(scale_max), layout_int, permute);
    }
    else {
        at::Tensor value_t = at::empty(vt_sizes, value.options());
        if (permute) {
            transpose_pad_permute_cuda(value, value_t, layout_int);
        }
        else {
            transpose_pad_cuda(value, value_t, layout_int);
        }
        if (smooth_v) {
            mean_scale_fuse_quant_cuda(value_t,
                                       v_fp8,
                                       value_mean,
                                       v_scale,
                                       static_cast<int>(kv_len),
                                       static_cast<float>(scale_max),
                                       layout_int);
        }
        else {
            scale_fuse_quant_cuda(
                value_t, v_fp8, v_scale, static_cast<int>(kv_len), static_cast<float>(scale_max), layout_int);
        }
    }

    if (smooth_v) {
        return {v_fp8, v_scale, value_mean};
    }
    return {v_fp8, v_scale, std::nullopt};
}

// quant_v_fp8's packed [total_tokens, heads, head_dim] counterpart. The two
// kernels are the same ones (a null cu_seqlens is what makes them dense); the
// addressing is what changes. The transposed value loses its batch dimension
// and its padded axis becomes the block algebra of varlen.h: blk_total blocks
// of pad_multiple tokens, sequence b's slab starting at pad_offset. Every
// sequence's slab is zero from its length to its end, which is the premise the
// sm89 V load runs on - that load has no bound predicate, and its 16-token
// permute groups have to stay inside one sequence.
std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>> quant_v_fp8_varlen_cuda(const at::Tensor& value,
                                                                                      const at::Tensor& cu_seqlens_k,
                                                                                      c10::SymInt       max_seqlen_k,
                                                                                      c10::string_view  v_layout,
                                                                                      double            scale_max,
                                                                                      bool              smooth_v,
                                                                                      int64_t           pad_multiple)
{
    const c10::cuda::CUDAGuard device_guard(value.device());
    TORCH_CHECK(SAGEATTN_BUILD_VARLEN != 0, "sageattention was built without varlen support (SAGE_BUILD_VARLEN=OFF)");
    const VLayout vl = parse_v_layout(v_layout);
    TORCH_CHECK(vl != VLayout::kSeq,
                "quant_v_fp8_varlen produces transposed layouts "
                "(\"mma_k16\" / \"linear\"); the fp16 PV path takes V unquantized");
    TORCH_CHECK(pad_multiple == 64 || pad_multiple == 128, "pad_multiple must be 64 or 128, got ", pad_multiple);
    TORCH_CHECK(value.dim() == 3, "packed value must be 3-D [total_tokens, heads, head_dim]");
    check_cu_seqlens(cu_seqlens_k, value, "cu_seqlens_k");

    const int64_t batch_size   = cu_seqlens_k.size(0) - 1;
    const int64_t total_k      = value.size(0);
    const int64_t num_kv_heads = value.size(1);
    const int64_t head_dim     = value.size(2);
    const int64_t max_k        = max_seqlen_k.guard_int(__FILE__, __LINE__);
    const int64_t padded_total = sage::blk_total(total_k, batch_size, pad_multiple) * pad_multiple;

    const QuantVarlen varlen{cu_seqlens_k.data_ptr<int32_t>(), batch_size, max_k, pad_multiple};
    // tensor_layout is unused on the varlen path (the strides carry the
    // layout); pass the HND flag so the argument still has a legal value.
    const int  layout_int = static_cast<int>(TensorLayout::kHND);
    const bool permute    = vl == VLayout::kMmaK16;

    // zeros, not empty, for the same reason as the dense path plus one more:
    // the fp8 tail of every sequence's slab (from its 64-aligned length to the
    // end of the blocks varlen.h gave it) is never written, and the attention
    // kernel reads whole 64-token tiles. A NaN-encoded garbage byte there
    // would poison PV through 0 * NaN.
    at::Tensor v_fp8 = at::zeros({num_kv_heads, head_dim, padded_total}, value.options().dtype(at::kFloat8_e4m3fn));
    // zeros as well: an empty sequence produces no statistics at all, and an
    // uninitialized entry would make the op's output run-to-run unstable
    // (opcheck's test_aot_dispatch_dynamic compares eager against traced).
    at::Tensor v_scale = at::zeros({batch_size, num_kv_heads, head_dim}, value.options().dtype(at::kFloat));
    at::Tensor value_mean;
    if (smooth_v) {
        value_mean = at::zeros({batch_size, num_kv_heads, head_dim}, value.options().dtype(at::kFloat));
    }

    // The gate takes max_seqlen: every sequence runs the same kernel, so the
    // longest one decides which is cheaper for the batch.
    if (use_fused_v_quant(head_dim, at::round_up<int64_t>(max_k, pad_multiple), value.get_device())) {
        transpose_quant_v_fp8_cuda(
            value, v_fp8, value_mean, v_scale, static_cast<float>(scale_max), layout_int, permute, varlen);
    }
    else {
        // at::empty is enough: the transpose writes each sequence's slab up to
        // its 64-aligned length and the quantization pass reads exactly that
        // far, so the part a pad_multiple=128 slab leaves over is never touched
        // (the dense path makes the same trade).
        at::Tensor value_t = at::empty({num_kv_heads, head_dim, padded_total}, value.options());
        if (permute) {
            transpose_pad_permute_cuda(value, value_t, layout_int, varlen);
        }
        else {
            transpose_pad_cuda(value, value_t, layout_int, varlen);
        }
        if (smooth_v) {
            mean_scale_fuse_quant_cuda(value_t,
                                       v_fp8,
                                       value_mean,
                                       v_scale,
                                       /*num_tokens=*/0,
                                       static_cast<float>(scale_max),
                                       layout_int,
                                       varlen);
        }
        else {
            scale_fuse_quant_cuda(
                value_t, v_fp8, v_scale, /*num_tokens=*/0, static_cast<float>(scale_max), layout_int, varlen);
        }
    }

    if (smooth_v) {
        return {v_fp8, v_scale, value_mean};
    }
    return {v_fp8, v_scale, std::nullopt};
}

std::tuple<at::Tensor, at::Tensor> sub_mean_v_cuda(const at::Tensor& value, c10::string_view tensor_layout)
{
    const c10::cuda::CUDAGuard device_guard(value.device());
    const TensorLayout         layout = parse_tensor_layout(tensor_layout);
    auto [seq_dim, nh_dim]            = seq_nh_dims(layout);
    (void)nh_dim;
    at::Tensor value_mean     = value.mean(seq_dim);
    at::Tensor value_smoothed = at::empty(value.sizes(), value.options().dtype(at::kHalf));
    sub_mean_cuda(value, value_mean, value_smoothed, static_cast<int>(layout));
    return {value_smoothed, value_mean};
}

}  // namespace
}  // namespace sage

TORCH_LIBRARY_FRAGMENT(sageattention, m)
{
    m.def("quant_qk(Tensor query, Tensor key, Tensor? key_mean=None, *, "
          "str tensor_layout=\"HND\", str qk_quant_gran=\"per_thread\", "
          "int blk_q=128, int warp_q=32, int blk_k=64, int warp_k=64) "
          "-> (Tensor q_int8, Tensor q_scale, Tensor k_int8, Tensor k_scale)");
    // max_seqlen_* are SymInt on purpose: they only size grid.x and are absent
    // from every output shape, so dynamo keeps them symbolic instead of
    // specializing the graph on their value (one recompile per batch shape).
    m.def("quant_qk_varlen(Tensor query, Tensor key, Tensor cu_seqlens_q, "
          "Tensor cu_seqlens_k, Tensor? key_mean=None, *, SymInt max_seqlen_q, "
          "SymInt max_seqlen_k, str qk_quant_gran=\"per_thread\", "
          "int blk_q=128, int warp_q=32, int blk_k=64, int warp_k=64) "
          "-> (Tensor q_int8, Tensor q_scale, Tensor k_int8, Tensor k_scale)");
    // The mean is [batch_size, heads, head_dim]: batch_size is a static shape
    // (cu_seqlens.size(0) - 1), so the fake needs nothing from the contents,
    // and max_seqlen again only sizes the grid.
    m.def("segment_mean_varlen(Tensor input, Tensor cu_seqlens, *, SymInt max_seqlen) -> Tensor");
    m.def("quant_v_fp8(Tensor value, *, str tensor_layout=\"HND\", "
          "str v_layout=\"mma_k16\", float scale_max=448.0, bool smooth_v=False, "
          "int pad_multiple=64) -> (Tensor v_fp8, Tensor v_scale, Tensor? v_mean)");
    // v_fp8 is [heads, head_dim, blk_total(total_k, batch, pad) * pad]: the
    // padded axis is a static shape (varlen.h, Property 2), so a cudagraph
    // replay that rewrites cu_seqlens cannot resize it. v_scale / v_mean keep
    // their batch dimension - they are per (sequence, head, channel).
    m.def("quant_v_fp8_varlen(Tensor value, Tensor cu_seqlens_k, *, SymInt max_seqlen_k, "
          "str v_layout=\"mma_k16\", float scale_max=448.0, bool smooth_v=False, "
          "int pad_multiple=64) -> (Tensor v_fp8, Tensor v_scale, Tensor? v_mean)");
    m.def("sub_mean_v(Tensor value, *, str tensor_layout=\"HND\") "
          "-> (Tensor v_smoothed, Tensor v_mean)");
}

TORCH_LIBRARY_IMPL(sageattention, CUDA, m)
{
    m.impl("quant_qk", TORCH_FN(sage::quant_qk_cuda));
    m.impl("quant_qk_varlen", TORCH_FN(sage::quant_qk_varlen_cuda));
    m.impl("segment_mean_varlen", TORCH_FN(sage::segment_mean_varlen_cuda));
    m.impl("quant_v_fp8", TORCH_FN(sage::quant_v_fp8_cuda));
    m.impl("quant_v_fp8_varlen", TORCH_FN(sage::quant_v_fp8_varlen_cuda));
    m.impl("sub_mean_v", TORCH_FN(sage::sub_mean_v_cuda));
}
