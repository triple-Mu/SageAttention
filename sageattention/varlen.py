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

Python side of the flash-attention style varlen layout: q/k/v flattened to
``[total_tokens, heads, head_dim]`` with the sequence boundaries in a
``cu_seqlens`` prefix sum of shape ``[batch_size + 1]`` (int32, on the same
device). ``csrc/sageattn/varlen.h`` holds the matching kernel-side addressing.

The smooth_k reductions are pure ATen and stay inside a ``fullgraph=True``
region. The one thing that would break that is a shape derived from tensor
*contents*, which is why ``repeat_interleave`` is always given
``output_size``: the caller already knows ``total_tokens`` as a static shape,
so Dynamo never has to read ``cu_seqlens`` to size an output.
"""

import warnings
from typing import Literal, Optional, Tuple, Union, overload

import torch

from ._plan import get_plan
from .core import _LOG2E_V2, _pad_qkv_head_dim, _warned_configs

# Backends with a packed-layout attention kernel, mirroring the dispatch table
# in csrc/sageattn/fwd_varlen_cuda.cu. Checking here rather than letting the op
# raise keeps a device without one from paying for the quantization first.
_VARLEN_BACKENDS = frozenset({"sm80", "sm89", "sm90", "sm120"})


def _segment_ids(cu_seqlens: torch.Tensor, batch_size: int, total: int) -> torch.Tensor:
    """``[total]`` int64 tensor mapping each packed token to its sequence."""
    seg_len = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)
    return torch.repeat_interleave(
        torch.arange(batch_size, device=cu_seqlens.device), seg_len, output_size=total
    )


def _segment_mean(
    x: torch.Tensor, cu_seqlens: torch.Tensor, batch_size: int, total: int
) -> torch.Tensor:
    """Per-sequence mean of a packed ``[total, heads, head_dim]`` tensor.

    Returns ``[batch_size, heads, head_dim]`` in ``x``'s dtype. This is the
    varlen form of the dense ``k.mean(dim=seq_dim)`` smooth_k reduction, so it
    accumulates in float32 and casts once at the end. An empty sequence gets a
    zero mean (the divisor is clamped, the accumulator stays untouched).

    ``index_add_`` uses atomics on CUDA, so the result is run-to-run
    non-deterministic; the smooth_k path is compared with allclose, never with
    ``torch.equal``.
    """
    seg_id = _segment_ids(cu_seqlens, batch_size, total)
    acc = x.new_zeros((batch_size,) + x.shape[1:], dtype=torch.float32)
    acc.index_add_(0, seg_id, x.float())
    seg_len = (cu_seqlens[1:] - cu_seqlens[:-1]).clamp(min=1).to(torch.float32)
    return (acc / seg_len.view(-1, *([1] * (x.dim() - 1)))).to(x.dtype)


def _segment_lse_correction(
    q: torch.Tensor,
    key_mean: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    batch_size: int,
    total: int,
) -> torch.Tensor:
    """``q @ key_mean^T`` per token, as ``[heads, total]`` float32.

    smooth_k subtracts the per-sequence K mean before quantizing, so the lse
    the kernel returns is short by this term. ``key_mean`` is the
    ``[batch_size, kv_heads, head_dim]`` output of :func:`_segment_mean`;
    GQA is handled here exactly as the dense path handles it.

    The gather-then-multiply is the whole point: a per-token ``index_select``
    of the mean costs one q-sized tensor, whereas the dense
    ``matmul(q, km.transpose(-1, -2))`` shape would need a per-sequence
    ``[seq, seq]`` detour once the batch is packed.
    """
    q_per_kv_head = q.size(1) // key_mean.size(1)
    if q_per_kv_head > 1:
        key_mean = torch.repeat_interleave(key_mean, q_per_kv_head, dim=1)
    seg_id = _segment_ids(cu_seqlens_q, batch_size, total)
    return (q * key_mean.index_select(0, seg_id)).sum(-1, dtype=torch.float32).transpose(0, 1)


@overload
def sageattn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = ...,
    sm_scale: Optional[float] = ...,
    return_lse: Literal[False] = ...,
    *,
    qk_quant_gran: Optional[str] = ...,
    pv_accum_dtype: Optional[str] = ...,
    smooth_k: bool = ...,
    smooth_v: Optional[bool] = ...,
) -> torch.Tensor: ...


@overload
def sageattn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = ...,
    sm_scale: Optional[float] = ...,
    return_lse: Literal[True] = ...,
    *,
    qk_quant_gran: Optional[str] = ...,
    pv_accum_dtype: Optional[str] = ...,
    smooth_k: bool = ...,
    smooth_v: Optional[bool] = ...,
) -> Tuple[torch.Tensor, torch.Tensor]: ...


def sageattn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    *,
    qk_quant_gran: Optional[str] = None,
    pv_accum_dtype: Optional[str] = None,
    smooth_k: bool = True,
    smooth_v: Optional[bool] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Quantized attention over a packed batch of variable-length sequences.

    The flash-attention varlen layout: q/k/v carry every sequence's tokens end
    to end and ``cu_seqlens_*`` says where each one starts. This is a separate
    entry point from :func:`sageattn`, not a drop-in for
    ``F.scaled_dot_product_attention`` (``flash_attn_varlen_func`` is not one
    either).

    Parameters
    ----------
    q, k, v : ``[total_tokens, heads, head_dim]``, float16/bfloat16,
        head_dim <= 128, num_qo_heads divisible by num_kv_heads (GQA
        supported). q may have a different total length from k/v.
    cu_seqlens_q, cu_seqlens_k : ``[batch_size + 1]`` int32 prefix sums on the
        same device, starting at 0 and ending at the respective total_tokens.
        An empty sequence (two equal entries) is allowed.
    max_seqlen_q, max_seqlen_k : the longest sequence in each prefix sum. They
        size the grid; see the cudagraph note below.
    is_causal : causal mask with flash-attention's **bottom-right** alignment:
        row r of a sequence attends to keys up to ``r + (kv_len - qo_len)``.
        For a sequence with kv_len < qo_len the leading rows admit no key at
        all; they come back as zeros with an lse of ``-inf``.
    sm_scale : softmax scale; defaults to ``head_dim ** -0.5``.
    return_lse : also return the log-sum-exp (base e) per token, shaped
        ``[qo_heads, total_tokens]``. Note the head-major order: it is the
        kernel's own layout, and it differs from flash-attention's
        ``[total_tokens, heads]``.
    qk_quant_gran, pv_accum_dtype, smooth_k, smooth_v : as in
        :func:`sageattn`. smooth_k means the *per-sequence* K mean, not the
        batch's. smooth_v has no varlen kernel and is ignored (with a warning).

    Returns
    -------
    out : same shape and dtype as q, plus lse when return_lse=True.

    Notes
    -----
    The contents of ``cu_seqlens_*`` are never read on the host - that would
    be a device-to-host sync on every call - so:

    1. Under a cudagraph, only the *contents* may change between replays
       (update a resident buffer in place with ``.copy_()``). total_tokens,
       batch_size and the tensor addresses are baked into the capture.
    2. ``max_seqlen_*`` is baked into grid.x at capture time. A replay whose
       sequences are longer than the captured maximum silently drops the tails
       past it: capture with the model's upper bound.
    3. A prefix sum that disagrees with ``q.size(0)`` / ``k.size(0)`` reads out
       of bounds. Nothing on the host validates it.
    """
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        raise NotImplementedError(
            "SageAttention has no backward; call under torch.no_grad() or detach inputs"
        )

    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in (torch.float16, torch.bfloat16), "Input tensors must be float16 or bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."
    assert q.dim() == k.dim() == v.dim() == 3, (
        "q/k/v must be packed [total_tokens, heads, head_dim]; use sageattn() for a dense batch"
    )
    assert k.size(0) == v.size(0), "k and v must carry the same tokens"
    # checked before the pad: F.pad always returns a contiguous tensor
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, (
        "Last dim of qkv must be contiguous."
    )
    q, k, v, head_dim_og = _pad_qkv_head_dim(q, k, v)

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    cc = torch.cuda.get_device_capability(q.device.index)
    p = get_plan(cc, q.size(-1), qk_quant_gran, pv_accum_dtype, smooth_v, varlen=True)
    if p.smooth_v_ignored and not torch.compiler.is_compiling():
        key = (cc, p.pv_accum_dtype)
        if key not in _warned_configs:
            _warned_configs.add(key)
            warnings.warn("smooth_v has no varlen kernel and will be ignored.", stacklevel=2)
    if p.backend not in _VARLEN_BACKENDS:
        raise NotImplementedError(
            f"sageattn_varlen has packed-layout kernels for {sorted(_VARLEN_BACKENDS)}; "
            f"this device resolves to {p.backend}"
        )

    # batch_size and the totals are static shapes, so the traced graph never
    # has to look inside cu_seqlens
    batch_size = cu_seqlens_q.size(0) - 1
    total_q, total_k = q.size(0), k.size(0)

    km = None
    lse_correction = None
    if smooth_k:
        km = _segment_mean(k, cu_seqlens_k, batch_size, total_k)
        if return_lse:
            lse_correction = _segment_lse_correction(q, km, cu_seqlens_q, batch_size, total_q)

    q_int8, q_scale, k_int8, k_scale = torch.ops.sageattention.quant_qk_varlen(
        q,
        k,
        cu_seqlens_q,
        cu_seqlens_k,
        km,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        qk_quant_gran=p.qk_quant_gran,
        blk_q=p.blk_q,
        warp_q=p.warp_q,
        blk_k=p.blk_k,
        warp_k=p.warp_k,
    )

    # ---- V preparation ----
    # The fp8 V^T is padded per sequence rather than per batch entry: the
    # attention kernel reads whole CTA_K tiles out of a sequence's slab with no
    # bound predicate, so the padding has to sit inside the sequence.
    value_scale = None
    if p.pv_fp8:
        v_prep, value_scale, _ = torch.ops.sageattention.quant_v_fp8_varlen(
            v,
            cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
            v_layout=p.v_layout,
            scale_max=p.v_scale_max,
            smooth_v=False,  # resolve() downgrades it; there is no varlen v_mean kernel
            pad_multiple=p.v_pad_multiple,
        )
    else:
        v_prep = v.to(torch.float16)

    out, lse = torch.ops.sageattention.fwd_varlen(
        q_int8,
        k_int8,
        v_prep,
        q_scale,
        k_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        value_scale,
        None,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        qk_quant_gran=p.qk_quant_gran,
        pv_accum_dtype=p.pv_accum_dtype,
        v_layout=p.v_layout,
        is_causal=is_causal,
        sm_scale=sm_scale,
        return_lse=return_lse,
        out_dtype=dtype,
    )

    out = out[..., :head_dim_og]

    if return_lse:
        lse = lse / _LOG2E_V2
        if smooth_k:
            lse = lse + lse_correction * sm_scale
        return out, lse
    return out
