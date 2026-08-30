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
"""

import warnings
from typing import Literal, Optional, Set, Tuple, Union, overload

import torch

from ._layout import _seq_nh_dims
from ._plan import get_plan

# The v2.x truncated log2(e) literal, kept on purpose: the kernel-side constant
# (csrc/math.cuh) is a separate full-precision value, so aligning this one to
# full precision would change the returned lse bit pattern. Do not "fix".
_LOG2E_V2 = 1.44269504

_warned_configs: Set[Tuple[Tuple[int, int], str]] = set()


def _pad_qkv_head_dim(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pad head_dim up to 64 or 128 (kernel-supported sizes); returns the
    original head_dim for the final output slice."""
    head_dim_og = q.size(-1)
    if head_dim_og <= 64:
        pad = 64 - head_dim_og
    elif head_dim_og <= 128:
        pad = 128 - head_dim_og
    else:
        raise ValueError(f"Unsupported head_dim: {head_dim_og} (must be <= 128)")
    if pad:
        q = torch.nn.functional.pad(q, (0, pad))
        k = torch.nn.functional.pad(k, (0, pad))
        v = torch.nn.functional.pad(v, (0, pad))
    return q, k, v, head_dim_og


def _smooth_k_mean(
    q: torch.Tensor,
    k: torch.Tensor,
    tensor_layout: str,
    seq_dim: int,
    nh_dim: int,
    return_lse: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """K mean for smooth_k and, when the lse is requested, the q @ km^T
    correction term (both pure ATen, traceable)."""
    km = k.mean(dim=seq_dim, keepdim=True)
    lse_correction = None
    if return_lse:
        q_per_kv_heads = q.size(nh_dim) // k.size(nh_dim)
        km_b = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim) if q_per_kv_heads > 1 else km
        if tensor_layout == "NHD":
            lse_correction = (
                torch.matmul(q.transpose(1, 2), km_b.transpose(1, 2).transpose(2, 3))
                .squeeze(-1)
                .to(torch.float32)
            )
        else:
            lse_correction = torch.matmul(q, km_b.transpose(2, 3)).squeeze(-1).to(torch.float32)
    return km, lse_correction


@overload
def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = ...,
    is_causal: bool = ...,
    sm_scale: Optional[float] = ...,
    return_lse: Literal[False] = ...,
    *,
    qk_quant_gran: Optional[str] = ...,
    pv_accum_dtype: Optional[str] = ...,
    smooth_k: bool = ...,
    smooth_v: Optional[bool] = ...,
    attn_mask: Optional[torch.Tensor] = ...,
    dropout_p: float = ...,
    scale: Optional[float] = ...,
    enable_gqa: bool = ...,
) -> torch.Tensor: ...


@overload
def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = ...,
    is_causal: bool = ...,
    sm_scale: Optional[float] = ...,
    return_lse: Literal[True] = ...,
    *,
    qk_quant_gran: Optional[str] = ...,
    pv_accum_dtype: Optional[str] = ...,
    smooth_k: bool = ...,
    smooth_v: Optional[bool] = ...,
    attn_mask: Optional[torch.Tensor] = ...,
    dropout_p: float = ...,
    scale: Optional[float] = ...,
    enable_gqa: bool = ...,
) -> Tuple[torch.Tensor, torch.Tensor]: ...


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    *,
    # ---- advanced knobs; None = this device's default (torch.ops.sageattention.plan)
    qk_quant_gran: Optional[str] = None,  # "per_warp" | "per_thread"
    pv_accum_dtype: Optional[
        str
    ] = None,  # "fp32" | "fp16" | "fp16+fp32" | "fp32+fp32" | "fp32+fp16"
    smooth_k: bool = True,
    smooth_v: Optional[bool] = None,
    # ---- F.scaled_dot_product_attention drop-in compatibility (explicit, not
    #      silently swallowed like the old **kwargs did)
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Quantized attention (INT8 QK^T + FP8/FP16 PV); the kernel is selected
    automatically from the GPU's compute capability.

    Parameters
    ----------
    q, k, v : torch.Tensor
        ``[batch, heads, seq, head_dim]`` ("HND") or ``[batch, seq, heads,
        head_dim]`` ("NHD"), dtype float16/bfloat16, head_dim <= 128,
        num_qo_heads divisible by num_kv_heads (GQA supported).
    tensor_layout : "HND" (default) or "NHD".
    is_causal : apply a causal mask (requires qo_len == kv_len).
    sm_scale : softmax scale; defaults to ``head_dim ** -0.5``.
    return_lse : also return the log-sum-exp (base e) per row, shape
        ``[batch, qo_heads, qo_len]`` (ring attention etc.).
    qk_quant_gran, pv_accum_dtype, smooth_k, smooth_v :
        quantization granularity / PV accumulator / K,V mean-smoothing knobs;
        None keeps the per-device default (the original per-arch tuning).
    attn_mask, dropout_p, scale, enable_gqa :
        SDPA-compatibility arguments. ``scale`` maps to ``sm_scale``;
        ``attn_mask``/``dropout_p`` are unsupported and raise.

    Returns
    -------
    out : torch.Tensor (same shape/dtype as q), plus lse when return_lse=True.
    """
    # ---- SDPA drop-in compatibility ----
    if attn_mask is not None:
        raise NotImplementedError("sageattn does not support attn_mask (use is_causal)")
    if dropout_p != 0.0:
        raise NotImplementedError("sageattn does not support dropout")
    if scale is not None:
        sm_scale = scale
    del enable_gqa  # GQA is always supported; flag kept for signature parity

    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        raise NotImplementedError(
            "SageAttention has no backward; call under torch.no_grad() or detach inputs"
        )

    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in (torch.float16, torch.bfloat16), "Input tensors must be float16 or bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    # checked before the pad: F.pad always returns a contiguous tensor, so
    # after it the assert would be vacuously true
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, (
        "Last dim of qkv must be contiguous."
    )
    q, k, v, head_dim_og = _pad_qkv_head_dim(q, k, v)

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    cc = torch.cuda.get_device_capability(q.device.index)
    p = get_plan(cc, q.size(-1), qk_quant_gran, pv_accum_dtype, smooth_v)

    if p.smooth_v_ignored and not torch.compiler.is_compiling():
        key = (cc, p.pv_accum_dtype)
        if key not in _warned_configs:
            _warned_configs.add(key)
            warnings.warn(
                f"pv_accum_dtype is '{p.pv_accum_dtype}', smooth_v will be ignored.", stacklevel=2
            )

    seq_dim, nh_dim = _seq_nh_dims(tensor_layout)

    km = None
    lse_correction = None
    if smooth_k:
        km, lse_correction = _smooth_k_mean(q, k, tensor_layout, seq_dim, nh_dim, return_lse)

    # ---- QK int8 quantization (tile geometry comes from the plan) ----
    q_int8, q_scale, k_int8, k_scale = torch.ops.sageattention.quant_qk(
        q,
        k,
        km,
        tensor_layout=tensor_layout,
        qk_quant_gran=p.qk_quant_gran,
        blk_q=p.blk_q,
        warp_q=p.warp_q,
        blk_k=p.blk_k,
        warp_k=p.warp_k,
    )

    # ---- V preparation ----
    value_scale = None
    value_mean = None
    if p.pv_fp8:
        # pad_multiple sinks the old python-side torch.cat zero pad (a full V
        # copy on sm90/sm100) into the transpose kernel's zero fill
        v_prep, value_scale, value_mean = torch.ops.sageattention.quant_v_fp8(
            v,
            tensor_layout=tensor_layout,
            v_layout=p.v_layout,
            scale_max=p.v_scale_max,
            smooth_v=p.smooth_v,
            pad_multiple=p.v_pad_multiple,
        )
    elif p.smooth_v:
        v_prep, value_mean = torch.ops.sageattention.sub_mean_v(v, tensor_layout=tensor_layout)
    else:
        v_prep = v.to(torch.float16)

    out, lse = torch.ops.sageattention.fwd(
        q_int8,
        k_int8,
        v_prep,
        q_scale,
        k_scale,
        value_scale,
        value_mean,
        tensor_layout=tensor_layout,
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
