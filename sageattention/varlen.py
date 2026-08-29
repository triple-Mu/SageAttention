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

Only the smooth_k reductions live here so far; the public ``sageattn_varlen``
entry point lands with the kernels.

Both helpers are pure ATen and stay inside a ``fullgraph=True`` region. The one
thing that would break that is a shape derived from tensor *contents*, which is
why ``repeat_interleave`` is always given ``output_size``: the caller already
knows ``total_tokens`` as a static shape, so Dynamo never has to read
``cu_seqlens`` to size an output.
"""

import torch


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
