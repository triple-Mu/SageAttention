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

Shared quant-attention pipeline for the per-arch CUDA wrapper functions in
core.py. Each sageattn_qk_int8_pv_*_cuda* wrapper describes its architecture
with an ArchSpec and delegates to qattn_pipeline, which performs the common
sequence (asserts, layout/int flag conversion, head-dim padding, smooth_k
km/lse_correction, QK quantization, V preparation, kernel dispatch, output
slice, lse return math) with the per-arch differences driven by the spec.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from .quant import per_channel_fp8, sub_mean


def pad_qkv_head_dim(q, k, v):
    """Pad the head dimension of q/k/v up to 64 or 128. Returns the (possibly
    padded) tensors and the original head dim (for the final output slice)."""
    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    return q, k, v, head_dim_og


def smooth_k_mean_lse_correction(q, k, tensor_layout, seq_dim, nh_dim, smooth_k, return_lse):
    """Compute the K mean (km) for smooth_k and, when the lse is requested, the
    q @ km^T correction term. Returns (None, None) when smooth_k is off and
    (km, None) when the lse is not requested."""
    km = None
    lse_correction = None
    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        nqheads = q.size(nh_dim)
        nkheads = k.size(nh_dim)
        q_per_kv_heads = nqheads // nkheads
        if q_per_kv_heads > 1:
            # nheads_k => nheads_q
            km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim)
        else:
            km_broadcast = km
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km_broadcast.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km_broadcast.transpose(2, 3)).squeeze(-1).to(torch.float32)
    return km, lse_correction


def make_qk_quant(per_warp_fn, per_thread_fn, *, BLKQ, WARPQ, BLKK, WARPK):
    """Build an ArchSpec.quant callable with fixed block/warp parameters."""
    def quant(q, k, km, tensor_layout, qk_quant_gran, pv_accum_dtype):
        if qk_quant_gran == "per_warp":
            return per_warp_fn(q, k, km, tensor_layout=tensor_layout, BLKQ=BLKQ, WARPQ=WARPQ, BLKK=BLKK)
        elif qk_quant_gran == "per_thread":
            return per_thread_fn(q, k, km, tensor_layout=tensor_layout, BLKQ=BLKQ, WARPQ=WARPQ, BLKK=BLKK, WARPK=WARPK)
    return quant


@dataclass(frozen=True)
class ArchSpec:
    """Per-architecture description of the quant-attention pipeline.

    kernels maps (pv_accum_dtype, smooth_v) -> (kind, kernel) where kind is:
    - "v_fp16":        FP16 PV kernel; v is cast to fp16, no value scale.
    - "v_smooth_fp16": FP16 PV kernel; v is mean-smoothed, value mean passed.
    - "vscale":        FP8 PV kernel taking the per-channel value scale.
    - "vscale_vmean":  FP8 PV kernel taking the value scale and value mean.
    - "error":         raise NotImplementedError(kernel).
    A missing key runs no kernel at all (matching the original if/elif chains
    that silently fell through), unless unsupported_pv_accum_error is set.
    """

    # (q, k, km, tensor_layout, qk_quant_gran, pv_accum_dtype) -> (q_int8, q_scale, k_int8, k_scale)
    quant: Callable
    kernels: Dict[Tuple[str, bool], Tuple[str, Any]]
    # False: FP16 PV (no per-channel V quantization); True: FP8 PV.
    pv_fp8: bool
    # ((accum values, warn message or None for the generic template), ...);
    # smooth_v is reset to False when pv_accum_dtype matches.
    smooth_v_ignore: Tuple[Tuple[Tuple[str, ...], Optional[str]], ...] = ()
    # sm90/sm100: pad V's sequence dim to a multiple of 128 before quantization.
    v_pad_to_128: bool = False
    # True (sm89/sm90/sm12x): permute V's seq order within 16-token groups to
    # match the register A-fragment k-order of mma.sync/wgmma. False (sm100):
    # keep linear seq order — the tcgen05 kernel packs P in linear k-order.
    v_permute: bool = True
    # True (sm89/sm12x): scale_max 2.25 for 'fp32+fp16', else 448.0. False: 448.0.
    v_scale_max_by_accum: bool = False
    # sm80: raise ValueError for an unrecognized pv_accum_dtype.
    unsupported_pv_accum_error: bool = False
    # Optional per-arch pv_accum_dtype asserts, at their original positions.
    assert_pv_accum_after_gran: Optional[Callable[[str], None]] = None
    assert_pv_accum_after_dtype: Optional[Callable[[str], None]] = None


def qattn_pipeline(q, k, v, spec, *, tensor_layout, is_causal, qk_quant_gran,
                   sm_scale, pv_accum_dtype, smooth_k, smooth_v, return_lse):
    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran in ["per_warp", "per_thread"], "qk_quant_gran must be either 'per_warp' or 'per_thread'."
    if spec.assert_pv_accum_after_gran is not None:
        spec.assert_pv_accum_after_gran(pv_accum_dtype)
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."
    if spec.assert_pv_accum_after_dtype is not None:
        spec.assert_pv_accum_after_dtype(pv_accum_dtype)

    # FIXME(DefTruth): make sage attention work compatible with distributed
    # env, for example, xDiT which launch by torchrun. Without this workaround,
    # sage attention will run into illegal memory access error after first
    # inference step in distributed env for multi gpus inference. This small
    # workaround also make sage attention work compatible with torch.compile
    # through non-fullgraph compile mode.
    torch.cuda.set_device(v.device)

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0

    q, k, v, head_dim_og = pad_qkv_head_dim(q, k, v)

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if _tensor_layout == 0 else 2
    nh_dim = 2 if _tensor_layout == 0 else 1

    km, lse_correction = smooth_k_mean_lse_correction(q, k, tensor_layout, seq_dim, nh_dim, smooth_k, return_lse)

    q_int8, q_scale, k_int8, k_scale = spec.quant(q, k, km, tensor_layout, qk_quant_gran, pv_accum_dtype)

    o = torch.empty(q.size(), dtype=dtype, device=q.device)

    for accum_values, message in spec.smooth_v_ignore:
        if pv_accum_dtype in accum_values and smooth_v:
            warnings.warn(message if message is not None else f"pv_accum_dtype is {pv_accum_dtype}, smooth_v will be ignored.")
            smooth_v = False

    if spec.pv_fp8:
        if spec.v_pad_to_128:
            # pad v to multiple of 128
            # TODO: modify per_channel_fp8 kernel to handle this
            kv_len = k.size(seq_dim)
            v_pad_len = 128 - (kv_len % 128) if kv_len % 128 != 0 else 0
            if v_pad_len > 0:
                if tensor_layout == "HND":
                    v = torch.cat([v, torch.zeros(v.size(0), v.size(1), v_pad_len, v.size(3), dtype=v.dtype, device=v.device)], dim=2)
                else:
                    v = torch.cat([v, torch.zeros(v.size(0), v_pad_len, v.size(2), v.size(3), dtype=v.dtype, device=v.device)], dim=1)

        quant_v_scale_max = 448.0
        if spec.v_scale_max_by_accum and pv_accum_dtype == 'fp32+fp16':
            quant_v_scale_max = 2.25

        v_fp8, v_scale, vm = per_channel_fp8(v, tensor_layout=tensor_layout, scale_max=quant_v_scale_max, smooth_v=smooth_v, permute=spec.v_permute)

    entry = spec.kernels.get((pv_accum_dtype, smooth_v))
    if entry is None:
        if spec.unsupported_pv_accum_error:
            raise ValueError(f"Unsupported pv_accum_dtype: {pv_accum_dtype}")
    else:
        kind, kernel = entry
        if kind == "error":
            raise NotImplementedError(kernel)
        elif kind == "v_fp16":
            v = v.to(torch.float16)
            lse = kernel(q_int8, k_int8, v, o, q_scale, k_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        elif kind == "v_smooth_fp16":
            smoothed_v, vm = sub_mean(v, tensor_layout=tensor_layout)
            lse = kernel(q_int8, k_int8, smoothed_v, o, q_scale, k_scale, vm, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        elif kind == "vscale":
            lse = kernel(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        elif kind == "vscale_vmean":
            lse = kernel(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, vm, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o
