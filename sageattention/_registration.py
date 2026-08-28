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

torch.library registration factory for the per-arch _qattn extension modules.

Every smXX_compile module used to spell out one `torch.library.custom_op`
wrapper plus a `register_fake` shape function per kernel entry point, all
byte-identical except for the extension module, the op namespace and the
argument shape. The four argument shapes that actually occur are:

- "base":         (q, k, v, o, q_scale, k_scale, <flags>)
- "vmean":        (q, k, v, o, q_scale, k_scale, value_mean, <flags>)
- "vscale":       (q, k, v, o, q_scale, k_scale, value_scale, <flags>)
- "vscale_vmean": (q, k, v, o, q_scale, k_scale, value_scale, value_mean, <flags>)

where <flags> = (tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse).

`register_qattn_ops` registers a list of (name, shape[, docstring]) entries for
one extension module and returns the custom-op objects keyed by name. The
makers carry full type annotations because `torch.library.custom_op` inspects
them to build the op schema.
"""

import torch


def _fake_base(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    tensor_layout: int,
    is_causal: int,
    qk_quant_gran: int,
    sm_scale: float,
    return_lse: int,
) -> torch.Tensor:
    batch_size = query.size(0)

    if tensor_layout == 0:
        num_qo_heads = query.size(2)
        qo_len = query.size(1)
    else:
        num_qo_heads = query.size(1)
        qo_len = query.size(2)

    if return_lse:
        lse = torch.empty((batch_size, num_qo_heads, qo_len), dtype=torch.float32, device=query.device)
    else:
        lse = torch.empty((0))
    return lse


def _fake_vmean(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value_mean: torch.Tensor,
    tensor_layout: int,
    is_causal: int,
    qk_quant_gran: int,
    sm_scale: float,
    return_lse: int,
) -> torch.Tensor:
    return _fake_base(
        query, key, value, output, query_scale, key_scale, tensor_layout,
        is_causal, qk_quant_gran, sm_scale, return_lse
    )


def _fake_vscale(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    tensor_layout: int,
    is_causal: int,
    qk_quant_gran: int,
    sm_scale: float,
    return_lse: int,
) -> torch.Tensor:
    return _fake_base(
        query, key, value, output, query_scale, key_scale, tensor_layout,
        is_causal, qk_quant_gran, sm_scale, return_lse
    )


def _fake_vscale_vmean(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    tensor_layout: int,
    is_causal: int,
    qk_quant_gran: int,
    sm_scale: float,
    return_lse: int,
) -> torch.Tensor:
    return _fake_base(
        query, key, value, output, query_scale, key_scale, tensor_layout,
        is_causal, qk_quant_gran, sm_scale, return_lse
    )


def _make_base(ext, namespace, name, doc):
    def op_impl(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        query_scale: torch.Tensor,
        key_scale: torch.Tensor,
        tensor_layout: int,
        is_causal: int,
        qk_quant_gran: int,
        sm_scale: float,
        return_lse: int,
    ) -> torch.Tensor:
        return getattr(ext, name)(
            query, key, value, output, query_scale, key_scale, tensor_layout,
            is_causal, qk_quant_gran, sm_scale, return_lse
        )
    return _register(op_impl, _fake_base, namespace, name, doc)


def _make_vmean(ext, namespace, name, doc):
    def op_impl(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        query_scale: torch.Tensor,
        key_scale: torch.Tensor,
        value_mean: torch.Tensor,
        tensor_layout: int,
        is_causal: int,
        qk_quant_gran: int,
        sm_scale: float,
        return_lse: int,
    ) -> torch.Tensor:
        return getattr(ext, name)(
            query, key, value, output, query_scale, key_scale, value_mean,
            tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse
        )
    return _register(op_impl, _fake_vmean, namespace, name, doc)


def _make_vscale(ext, namespace, name, doc):
    def op_impl(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        query_scale: torch.Tensor,
        key_scale: torch.Tensor,
        value_scale: torch.Tensor,
        tensor_layout: int,
        is_causal: int,
        qk_quant_gran: int,
        sm_scale: float,
        return_lse: int,
    ) -> torch.Tensor:
        return getattr(ext, name)(
            query, key, value, output, query_scale, key_scale, value_scale,
            tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse
        )
    return _register(op_impl, _fake_vscale, namespace, name, doc)


def _make_vscale_vmean(ext, namespace, name, doc):
    def op_impl(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        query_scale: torch.Tensor,
        key_scale: torch.Tensor,
        value_scale: torch.Tensor,
        value_mean: torch.Tensor,
        tensor_layout: int,
        is_causal: int,
        qk_quant_gran: int,
        sm_scale: float,
        return_lse: int,
    ) -> torch.Tensor:
        return getattr(ext, name)(
            query, key, value, output, query_scale, key_scale, value_scale,
            value_mean, tensor_layout, is_causal, qk_quant_gran, sm_scale,
            return_lse
        )
    return _register(op_impl, _fake_vscale_vmean, namespace, name, doc)


def _register(op_impl, fake_impl, namespace, name, doc):
    op_impl.__name__ = name
    if doc is not None:
        op_impl.__doc__ = doc
    qualname = f"{namespace}::{name}"
    op = torch.library.custom_op(qualname, mutates_args=("output",), device_types="cuda")(op_impl)
    torch.library.register_fake(qualname)(fake_impl)
    return op


_MAKERS = {
    "base": _make_base,
    "vmean": _make_vmean,
    "vscale": _make_vscale,
    "vscale_vmean": _make_vscale_vmean,
}


def register_qattn_ops(ext, namespace, ops):
    """Register CUDA custom ops (+ fake impls) for one _qattn extension module.

    Parameters
    ----------
    ext : module
        The compiled _qattn_smXX extension module.
    namespace : str
        The torch.library namespace, e.g. "sageattention_sm90".
    ops : list
        (name, shape) or (name, shape, docstring) tuples, with shape one of
        "base", "vmean", "vscale", "vscale_vmean".

    Returns
    -------
    dict
        name -> registered custom op object.
    """
    registered = {}
    for entry in ops:
        name, shape = entry[0], entry[1]
        doc = entry[2] if len(entry) > 2 else None
        registered[name] = _MAKERS[shape](ext, namespace, name, doc)
    return registered
