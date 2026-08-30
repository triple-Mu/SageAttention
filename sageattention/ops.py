"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

FakeTensor kernels for the torch.ops.sageattention attention ops (needed for
torch.compile). The mutating quantization ops return () and need no fake.
"""

import torch

from ._layout import _seq_nh_dims


def _fwd_fake(
    query,
    key,
    value,
    query_scale,
    key_scale,
    value_scale=None,
    value_mean=None,
    *,
    tensor_layout="HND",
    qk_quant_gran="per_thread",
    pv_accum_dtype="fp32",
    v_layout="mma_k16",
    is_causal=False,
    sm_scale=1.0,
    return_lse=False,
    out_dtype=torch.float16,
    backend=None,
):
    out = torch.empty(query.shape, dtype=out_dtype, device=query.device)
    if not return_lse:
        return out, None
    seq_dim, nh_dim = _seq_nh_dims(tensor_layout)
    qo_len, num_qo_heads = query.size(seq_dim), query.size(nh_dim)
    lse = torch.empty(
        (query.size(0), num_qo_heads, qo_len), dtype=torch.float32, device=query.device
    )
    return out, lse


def _fwd_varlen_fake(
    query,
    key,
    value,
    query_scale,
    key_scale,
    cu_seqlens_q,
    cu_seqlens_k,
    value_scale=None,
    value_mean=None,
    *,
    max_seqlen_q,
    max_seqlen_k,
    qk_quant_gran="per_thread",
    pv_accum_dtype="fp32",
    v_layout="mma_k16",
    is_causal=False,
    sm_scale=1.0,
    return_lse=False,
    out_dtype=torch.float16,
):
    # packed [total_tokens, heads, head_dim]; the lse is head-major over the
    # same packed token axis. max_seqlen_* are never read: they size grid.x
    # only, which is what keeps them out of the traced shape environment.
    out = torch.empty(query.shape, dtype=out_dtype, device=query.device)
    if not return_lse:
        return out, None
    lse = torch.empty(
        (query.size(1), query.size(0)), dtype=torch.float32, device=query.device
    )
    return out, lse


def _ceil_div(a, b):
    return (a + b - 1) // b  # SymInt-friendly (no float round trip)


def _quant_qk_fake(
    query,
    key,
    key_mean=None,
    *,
    tensor_layout="HND",
    qk_quant_gran="per_thread",
    blk_q=128,
    warp_q=32,
    blk_k=64,
    warp_k=64,
):
    seq_dim, nh_dim = _seq_nh_dims(tensor_layout)
    qo_len, kv_len = query.size(seq_dim), key.size(seq_dim)
    if qk_quant_gran == "per_warp":
        qs = _ceil_div(qo_len, blk_q) * (blk_q // warp_q)
        ks = _ceil_div(kv_len, blk_k)
    else:
        qs = _ceil_div(qo_len, blk_q) * (blk_q // warp_q) * 8
        ks = _ceil_div(kv_len, blk_k) * (blk_k // warp_k) * 4
    dev = query.device
    return (
        torch.empty(query.shape, dtype=torch.int8, device=dev),
        torch.empty((query.size(0), query.size(nh_dim), qs), dtype=torch.float32, device=dev),
        torch.empty(key.shape, dtype=torch.int8, device=dev),
        torch.empty((key.size(0), key.size(nh_dim), ks), dtype=torch.float32, device=dev),
    )


def _blk_total(total_tokens, batch_size, cta_tokens):
    """csrc/sageattn/varlen.h blk_total: the packed block count of the whole
    batch. It reads (total_tokens, batch_size) only -- both static shapes --
    which is why no varlen fake kernel needs an unbacked SymInt."""
    return total_tokens // cta_tokens + batch_size


def _quant_qk_varlen_fake(
    query,
    key,
    cu_seqlens_q,
    cu_seqlens_k,
    key_mean=None,
    *,
    max_seqlen_q,
    max_seqlen_k,
    qk_quant_gran="per_thread",
    blk_q=128,
    warp_q=32,
    blk_k=64,
    warp_k=64,
):
    b = cu_seqlens_q.size(0) - 1
    q_blocks = _blk_total(query.size(0), b, blk_q)
    k_blocks = _blk_total(key.size(0), b, blk_k)
    if qk_quant_gran == "per_warp":
        qs = q_blocks * (blk_q // warp_q)
        ks = k_blocks
    else:
        qs = q_blocks * (blk_q // warp_q) * 8
        ks = k_blocks * (blk_k // warp_k) * 4
    dev = query.device
    return (
        torch.empty(query.shape, dtype=torch.int8, device=dev),
        torch.empty((query.size(1), qs), dtype=torch.float32, device=dev),
        torch.empty(key.shape, dtype=torch.int8, device=dev),
        torch.empty((key.size(1), ks), dtype=torch.float32, device=dev),
    )


def _segment_mean_varlen_fake(input, cu_seqlens, *, max_seqlen):
    # [batch_size, heads, head_dim]: batch_size is cu_seqlens.size(0) - 1, a
    # static shape. max_seqlen sizes the grid only, so it stays unread here.
    b = cu_seqlens.size(0) - 1
    return torch.empty((b, input.size(1), input.size(2)), dtype=input.dtype, device=input.device)


def _quant_v_fp8_fake(
    value,
    *,
    tensor_layout="HND",
    v_layout="mma_k16",
    scale_max=448.0,
    smooth_v=False,
    pad_multiple=64,
):
    seq_dim, nh_dim = _seq_nh_dims(tensor_layout)
    b, kv_len, h, d = (value.size(0), value.size(seq_dim), value.size(nh_dim), value.size(-1))
    padded = _ceil_div(kv_len, pad_multiple) * pad_multiple
    shape = (b, h, d, padded) if tensor_layout == "HND" else (b, d, h, padded)
    dev = value.device
    v_fp8 = torch.empty(shape, dtype=torch.float8_e4m3fn, device=dev)
    v_scale = torch.empty((b, h, d), dtype=torch.float32, device=dev)
    vm = torch.empty((b, h, d), dtype=torch.float32, device=dev) if smooth_v else None
    return v_fp8, v_scale, vm


def _quant_v_fp8_varlen_fake(
    value,
    cu_seqlens_k,
    *,
    max_seqlen_k,
    v_layout="mma_k16",
    scale_max=448.0,
    smooth_v=False,
    pad_multiple=64,
):
    # the padded V^T axis is blk_total blocks of pad_multiple tokens, which
    # reads (total_tokens, batch_size) only -- both static shapes
    b = cu_seqlens_k.size(0) - 1
    h, d = value.size(1), value.size(2)
    padded = _blk_total(value.size(0), b, pad_multiple) * pad_multiple
    dev = value.device
    v_fp8 = torch.empty((h, d, padded), dtype=torch.float8_e4m3fn, device=dev)
    v_scale = torch.empty((b, h, d), dtype=torch.float32, device=dev)
    vm = torch.empty((b, h, d), dtype=torch.float32, device=dev) if smooth_v else None
    return v_fp8, v_scale, vm


def _sub_mean_v_fake(value, *, tensor_layout="HND"):
    seq_dim, _ = _seq_nh_dims(tensor_layout)
    vm_shape = list(value.shape)
    del vm_shape[seq_dim]
    return (
        torch.empty(value.shape, dtype=torch.float16, device=value.device),
        torch.empty(vm_shape, dtype=value.dtype, device=value.device),
    )


def _register() -> None:
    torch.library.register_fake("sageattention::fwd")(_fwd_fake)
    torch.library.register_fake("sageattention::fwd_varlen")(_fwd_varlen_fake)
    torch.library.register_fake("sageattention::quant_qk")(_quant_qk_fake)
    torch.library.register_fake("sageattention::quant_qk_varlen")(_quant_qk_varlen_fake)
    torch.library.register_fake("sageattention::segment_mean_varlen")(_segment_mean_varlen_fake)
    torch.library.register_fake("sageattention::quant_v_fp8")(_quant_v_fp8_fake)
    torch.library.register_fake("sageattention::quant_v_fp8_varlen")(_quant_v_fp8_varlen_fake)
    torch.library.register_fake("sageattention::sub_mean_v")(_sub_mean_v_fake)


_register()
