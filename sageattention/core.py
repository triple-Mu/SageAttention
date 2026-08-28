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

import torch

try:
    from . import sm80_compile
    SM80_ENABLED = True
except:
    SM80_ENABLED = False

try:
    from . import sm89_compile
    SM89_ENABLED = True
except:
    SM89_ENABLED = False

try:
    from . import sm90_compile
    SM90_ENABLED = True
except:
    SM90_ENABLED = False

try:
    from . import sm100_compile
    SM100_ENABLED = True
except:
    SM100_ENABLED = False

try:
    from . import sm120_compile
    SM120_ENABLED = True
except:
    SM120_ENABLED = False

from .quant import per_block_int8 as per_block_int8_cuda
from .quant import per_warp_int8 as per_warp_int8_cuda
from .quant import per_thread_int8_cuda
from .quant import sub_mean
from .quant import per_channel_fp8

from ._pipeline import (
    ArchSpec,
    make_qk_quant,
    pad_qkv_head_dim,
    qattn_pipeline,
    smooth_k_mean_lse_correction,
)

from typing import Any, List, Literal, Optional, Tuple, Union
import os


def get_cuda_arch_versions():
    cuda_archs = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cuda_archs.append(f"sm{major}{minor}")
    return cuda_archs


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    **kwargs: Any,
):
    """
    Automatically selects the appropriate implementation of the SageAttention kernel based on the GPU compute capability.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    """

    arch = get_cuda_arch_versions()[q.device.index]
    if arch == "sm80":
        return sageattn_qk_int8_pv_fp16_cuda(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype="fp32")
    elif arch == "sm86":
        # The Triton backend was removed; sm86 runs the generic sm80 CUDA kernels
        # (the sm80 extension always carries a native sm_86 cubin).
        return sageattn_qk_int8_pv_fp16_cuda(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype="fp32")
    elif arch == "sm89":
        return sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype="fp32+fp16")
    elif arch == "sm90":
        return sageattn_qk_int8_pv_fp8_cuda_sm90(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype="fp32+fp32")
    elif int(arch[2:]) in (120, 121):
        # sm12x (consumer Blackwell): exact fp32 accumulator for fp8 mma -> plain fp32
        # accumulation, per-thread quantization via CUDA (no Triton dependency).
        if SM120_ENABLED:
            return sageattn_qk_int8_pv_fp8_cuda_sm120(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype="fp32")
        # Fallback when the wheel was built without the sm120 extension: sm89 kernels
        # (their fatbin carries sm_12x cubins) with the Ada-workaround accumulator mode.
        return sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran="per_warp", sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype="fp32+fp16")
    elif arch in ("sm100", "sm110") and SM100_ENABLED \
            and os.getenv("SAGEATTN_SM100_TCGEN05", "0").upper() in {"1", "TRUE", "YES"}:
        # tcgen05 (5th-gen tensor core) kernels: int8 QK + fp8 PV with a true
        # fp32 accumulator. Opt-in via SAGEATTN_SM100_TCGEN05=1 until
        # hardware-validated; dispatch keys on exact (major, minor) — sm103
        # and family variants stay on the sm89 fallback below (GB300 dropped
        # INT8 tensor cores, so the kind::i8 kernel cannot exist there).
        return sageattn_qk_int8_pv_fp8_cuda_sm100(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype="fp32")
    elif int(arch[2:]) >= 100:
        # Other Blackwell-class archs (sm100/sm103/sm110/...) run the sm89 fp8 kernels: they have an accurate fp32 accumulator for fp8 mma and the triton kernel is currently not usable on them.
        return sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran="per_warp", sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype="fp32+fp16")
    else:
        raise ValueError(f"Unsupported CUDA architecture: {arch}")


def sageattn_qk_int8_pv_fp16_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    quantization_backend: str = "triton",
    is_causal: bool =False,
    attn_mask: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Removed: the Triton backend was dropped in favor of the CUDA kernels.

    Use :func:`sageattn_qk_int8_pv_fp16_cuda` (or the auto-dispatching
    :func:`sageattn`) instead.
    """
    raise RuntimeError(
        "sageattn_qk_int8_pv_fp16_triton has been removed together with the Triton backend; "
        "use sageattn_qk_int8_pv_fp16_cuda (or sageattn) instead.")


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
    smooth_k: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """Removed: varlen was implemented on the dropped Triton backend.

    A CUDA varlen implementation is planned; until then this raises.
    """
    raise NotImplementedError(
        "sageattn_varlen was Triton-only and has been removed with the Triton backend; "
        "a CUDA varlen implementation is planned.")


def _sm80_qk_quant(q, k, km, tensor_layout, qk_quant_gran, pv_accum_dtype):
    if qk_quant_gran == "per_warp":
        return per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=(16 if (q.size(-1) == 128 and pv_accum_dtype == "fp16+fp32") else 32), BLKK=64)
    elif qk_quant_gran == "per_thread":
        return per_thread_int8_cuda(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=(16 if (q.size(-1) == 128 and pv_accum_dtype == "fp16+fp32") else 32), BLKK=64, WARPK=64)


if SM80_ENABLED:
    _SM80_SPEC = ArchSpec(
        quant=_sm80_qk_quant,
        pv_fp8=False,
        kernels={
            ("fp32", False): ("v_fp16", sm80_compile.qk_int8_sv_f16_accum_f32_attn),
            ("fp16", True): ("v_smooth_fp16", sm80_compile.qk_int8_sv_f16_accum_f16_fuse_v_mean_attn),
            ("fp16", False): ("v_fp16", sm80_compile.qk_int8_sv_f16_accum_f16_attn),
            ("fp16+fp32", False): ("v_fp16", sm80_compile.qk_int8_sv_f16_accum_f16_attn_inst_buf),
        },
        smooth_v_ignore=((("fp32", "fp16+fp32"), None),),
        unsupported_pv_accum_error=True,
    )

if SM89_ENABLED:
    _SM89_SPEC = ArchSpec(
        quant=make_qk_quant(per_warp_int8_cuda, per_thread_int8_cuda, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64),
        pv_fp8=True,
        kernels={
            ("fp32", True): ("vscale_vmean", sm89_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn),
            ("fp32", False): ("vscale", sm89_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn),
            ("fp32+fp32", False): ("vscale", sm89_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf),
            ("fp32+fp16", False): ("vscale", sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf),
        },
        smooth_v_ignore=(
            (("fp32+fp32",), "pv_accum_dtype is 'fp32+fp32', smooth_v will be ignored."),
            (("fp32+fp16",), "pv_accum_dtype is 'fp32+fp16', smooth_v will be ignored."),
        ),
        v_scale_max_by_accum=True,
    )

if SM90_ENABLED:
    _SM90_SPEC = ArchSpec(
        quant=make_qk_quant(per_warp_int8_cuda, per_thread_int8_cuda, BLKQ=64, WARPQ=16, BLKK=128, WARPK=128),
        pv_fp8=True,
        kernels={
            ("fp32", False): ("error", "Please use pv_accum_dtype='fp32+fp32' for sm90."),
            ("fp32+fp32", False): ("vscale", sm90_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf),
        },
        v_pad_to_128=True,
    )

if SM100_ENABLED:
    def _sm100_assert_pv_accum(pv_accum_dtype):
        assert pv_accum_dtype == "fp32", "The tcgen05 kernels only support pv_accum_dtype='fp32'."

    _SM100_SPEC = ArchSpec(
        # tcgen05 kernel tile: CTA_Q = 128 (4 warp-scale groups of 32 rows), CTA_K = 128.
        quant=make_qk_quant(per_warp_int8_cuda, per_thread_int8_cuda, BLKQ=128, WARPQ=32, BLKK=128, WARPK=128),
        pv_fp8=True,
        kernels={
            # full 448 e4m3 range: the tcgen05 fp32 accumulator is exact, no headroom clamp needed.
            ("fp32", False): ("vscale", sm100_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn),
        },
        v_pad_to_128=True,
        # linear kv order (no mma-fragment permutation): the tcgen05 kernel's
        # P operand is packed in linear k-order, so V must be linear too.
        v_permute=False,
        assert_pv_accum_after_gran=_sm100_assert_pv_accum,
    )

if SM120_ENABLED:
    def _sm120_assert_pv_accum(pv_accum_dtype):
        assert pv_accum_dtype in ["fp32", "fp32+fp16"], \
            "pv_accum_dtype must be 'fp32' or 'fp32+fp16' on sm12x ('fp32+fp32' is pointless here: the plain fp32 accumulator is already exact)."

    _SM120_SPEC = ArchSpec(
        quant=make_qk_quant(per_warp_int8_cuda, per_thread_int8_cuda, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64),
        pv_fp8=True,
        kernels={
            ("fp32", True): ("vscale_vmean", sm120_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn),
            ("fp32", False): ("vscale", sm120_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn),
            ("fp32+fp16", False): ("vscale", sm120_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf),
        },
        smooth_v_ignore=((("fp32+fp16",), "pv_accum_dtype is 'fp32+fp16', smooth_v will be ignored."),),
        v_scale_max_by_accum=True,
        assert_pv_accum_after_dtype=_sm120_assert_pv_accum,
    )


def sageattn_qk_int8_pv_fp16_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with INT8 quantization for Q and K, FP16 PV with FP16/FP32 accumulation, implemented using CUDA.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    qk_quant_gran : str
        The granularity of quantization for Q and K, either "per_warp" or "per_thread".
        Default: "per_thread".

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    pv_accum_dtype : str
        The dtype of the accumulation of the product of the value tensor and the attention weights, either "fp16", "fp16+fp32" or "fp32".
        - "fp16": PV accumulation is done in fully in FP16. This is the fastest option but may lead to numerical instability. `smooth_v` option will increase the accuracy in cases when the value tensor has a large bias (like in CogVideoX-2b).
        - "fp32": PV accumulation is done in FP32. This is the most accurate option but may be slower than "fp16" due to CUDA core overhead.
        - "fp16+fp32": PV accumulation is done in FP16, but added to a FP32 buffer every few iterations. This offers a balance between speed and accuracy.
        Default: "fp32".

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    smooth_v : bool
        Whether to smooth the value tensor by subtracting the mean along the sequence dimension.
        smooth_v will be ignored if pv_accum_dtype is "fp32" or "fp16+fp32".
        Default: False.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """

    assert SM80_ENABLED, "SM80 kernel is not available. make sure you GPUs with compute capability 8.0 or higher."
    return qattn_pipeline(
        q, k, v, _SM80_SPEC,
        tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran=qk_quant_gran,
        sm_scale=sm_scale, pv_accum_dtype=pv_accum_dtype, smooth_k=smooth_k,
        smooth_v=smooth_v, return_lse=return_lse,
    )


def sageattn_qk_int8_pv_fp8_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp16",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with INT8 quantization for Q and K, FP8 PV with FP32 accumulation, implemented using CUDA.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    qk_quant_gran : str
        The granularity of quantization for Q and K, either "per_warp" or "per_thread".
        Default: "per_thread".

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    pv_accum_dtype : str
        The dtype of the accumulation of the product of the value tensor and the attention weights, either "fp32" or "fp32+fp32".
        - "fp32": PV accumulation is done in fully in FP32. However, due to the hardware issue, there are only 22 valid bits in the FP32 accumulator.
        - "fp32+fp32": PV accumulation is done in FP32 (actually FP22), but added to a FP32 buffer every few iterations. This offers a balance between speed and accuracy.
        Default: "fp32+fp32".

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    smooth_v : bool
        Whether to smooth the value tensor by subtracting the mean along the sequence dimension.
        smooth_v will be ignored if pv_accum_dtype is "fp32+fp32".
        Default: False.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

            torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """

    assert SM89_ENABLED, "SM89 kernel is not available. Make sure you GPUs with compute capability 8.9."
    return qattn_pipeline(
        q, k, v, _SM89_SPEC,
        tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran=qk_quant_gran,
        sm_scale=sm_scale, pv_accum_dtype=pv_accum_dtype, smooth_k=smooth_k,
        smooth_v=smooth_v, return_lse=return_lse,
    )


def sageattn_qk_int8_pv_fp8_cuda_sm120(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with INT8 quantization for Q and K, FP8 PV, specialized for
    sm12x (consumer Blackwell). Unlike Ada, the fp8 mma fp32 accumulator is
    exact on these parts, so the default path accumulates plainly in FP32
    (no inst_buf two-level machinery) with the full 448 V quantization range,
    and per-thread quantization runs through CUDA kernels (no Triton
    dependency).

    Parameters mirror :func:`sageattn_qk_int8_pv_fp8_cuda`, except:

    pv_accum_dtype : str
        - "fp32" (default): plain FP32 PV accumulation (exact on sm12x); allows `smooth_v`.
        - "fp32+fp16": opt-in speed mode (f8f8f16 mma at double rate, V range clamped to 2.25).
    """

    assert SM120_ENABLED, "SM120 kernel is not available. Make sure you have GPUs with compute capability 12.x and the package was built for them."
    return qattn_pipeline(
        q, k, v, _SM120_SPEC,
        tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran=qk_quant_gran,
        sm_scale=sm_scale, pv_accum_dtype=pv_accum_dtype, smooth_k=smooth_k,
        smooth_v=smooth_v, return_lse=return_lse,
    )


def sageattn_qk_int8_pv_fp8_cuda_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp32",
    smooth_k: bool = True,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with INT8 quantization for Q and K, FP8 PV with FP32 accumulation, implemented using CUDA.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    qk_quant_gran : str
        The granularity of quantization for Q and K, either "per_warp" or "per_thread".
        Default: "per_thread".

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    pv_accum_dtype : str
        The dtype of the accumulation of the product of the value tensor and the attention weights, either "fp32" or "fp32+fp32".
        - "fp32": PV accumulation is done in fully in FP32. However, due to the hardware issue, there are only 22 valid bits in the FP32 accumulator.
        - "fp32+fp32": PV accumulation is done in FP32 (actually FP22), but added to a FP32 buffer every few iterations. This offers a balance between speed and accuracy.
        Default: "fp32+fp32".

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

            torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """

    assert SM90_ENABLED, "SM90 kernel is not available. Make sure you GPUs with compute capability 9.0."
    return qattn_pipeline(
        q, k, v, _SM90_SPEC,
        tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran=qk_quant_gran,
        sm_scale=sm_scale, pv_accum_dtype=pv_accum_dtype, smooth_k=smooth_k,
        smooth_v=False, return_lse=return_lse,
    )


def sageattn_qk_int8_pv_fp8_cuda_sm100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_warp",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32",
    smooth_k: bool = True,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with INT8 quantization for Q and K, FP8 PV with FP32 accumulation,
    implemented with tcgen05 (5th-gen tensor core) instructions for sm100/sm110.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    qk_quant_gran : str
        The granularity of quantization for Q and K, either "per_warp" or "per_thread".
        Default: "per_warp".

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    pv_accum_dtype : str
        Must be "fp32": the tcgen05 fp8 mma accumulator is a true FP32 (unlike
        sm90's FP22), so the inst_buf/fp32+fp32 machinery does not exist here.

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """

    assert SM100_ENABLED, "SM100 tcgen05 kernel is not available. Make sure you have GPUs with compute capability 10.0 or 11.0 and the package was built for them."
    return qattn_pipeline(
        q, k, v, _SM100_SPEC,
        tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran=qk_quant_gran,
        sm_scale=sm_scale, pv_accum_dtype=pv_accum_dtype, smooth_k=smooth_k,
        smooth_v=False, return_lse=return_lse,
    )
