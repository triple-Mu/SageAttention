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

FlashAttention-3 baselines for the benchmarks and examples, wrapped in the
sageattn calling convention (the examples install them as
F.scaled_dot_product_attention).
"""

from typing import Optional

import torch

try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3

    FA3_ENABLED = True
except ImportError:
    FA3_ENABLED = False


@torch.compiler.disable
def fa3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    *,
    # SDPA drop-in compatibility (the examples pass attn_mask/dropout_p);
    # accepted and ignored, exactly as the old **kwargs swallowed them
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
):
    del attn_mask, dropout_p, scale, enable_gqa
    dtype = q.dtype
    assert FA3_ENABLED, "FA3 not available"
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], (
        "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    )
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    if tensor_layout == "HND":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

    o = flash_attn_func_v3(q, k, v, causal=is_causal, softmax_scale=sm_scale)[0]

    if tensor_layout == "HND":
        o = o.transpose(1, 2)

    return o


@torch.compiler.disable
def fa3_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    *,
    # SDPA drop-in compatibility; accepted and ignored, as in fa3 above
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
):
    del attn_mask, dropout_p, scale, enable_gqa
    dtype = q.dtype
    assert FA3_ENABLED, "FA3 not available"
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], (
        "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    )
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    if tensor_layout == "HND":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

    q_scale = (q.abs().max().to(torch.float32) / 448.0).unsqueeze(-1)
    k_scale = (k.abs().max().to(torch.float32) / 448.0).unsqueeze(-1)
    v_scale = (v.abs().max().to(torch.float32) / 448.0).unsqueeze(-1)

    # scale == 0 (all-zero tensor) makes x / scale = 0 / 0 = NaN, and a
    # subnormal scale overflows the division to inf; fp8_e4m3fn encodes both
    # as 0x7f. Mirror MeanScaleKernel in csrc/fused/fused.cu: zero the
    # reciprocal when it is not finite -- the descale handed to FA3 is (near)
    # zero, so the tensor dequantizes to the correct 0.
    q_recip = (1.0 / q_scale).nan_to_num(nan=0.0, posinf=0.0)
    k_recip = (1.0 / k_scale).nan_to_num(nan=0.0, posinf=0.0)
    v_recip = (1.0 / v_scale).nan_to_num(nan=0.0, posinf=0.0)

    q_f8 = (q * q_recip).to(torch.float8_e4m3fn)
    k_f8 = (k * k_recip).to(torch.float8_e4m3fn)
    v_f8 = (v * v_recip).to(torch.float8_e4m3fn)

    o = flash_attn_func_v3(
        q_f8,
        k_f8,
        v_f8,
        descale_q=q_scale,
        descale_k=k_scale,
        descale_v=v_scale,
        causal=is_causal,
        softmax_scale=sm_scale,
    )[0].to(dtype)

    if tensor_layout == "HND":
        o = o.transpose(1, 2)

    return o
