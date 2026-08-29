"""Stale shared memory must not leak into the output through the V tile.

Rows of the V tile past kv_len used to be loaded with cp.async fill mode
kNoFill, so shared memory kept whatever the previous CTA left there. The
out-of-bound mask drives those columns' probabilities to exactly 0, but
0 * NaN is NaN, so a single stale NaN half poisoned the whole output row.
The trigger needs two things: a partial last KV tile, and NaN bit patterns
in the allocator's recycled blocks -- which is what _dirty_allocator stages.
"""

import pytest
import torch

from conftest import cos_sim, rel_l1, requires_backend, sdpa_ref
from sageattention import sageattn

DEV = "cuda"
pytestmark = requires_backend("sm80")

COS_MIN = 0.99
REL_L1_MAX = 0.06


def _dirty_allocator():
    """Fill the allocator's cache with fp16 NaN bit patterns."""
    n = 64 * 1024 * 1024 // 2  # 64 MiB
    junk = torch.full((n,), float("nan"), device=DEV, dtype=torch.float16)
    del junk
    torch.cuda.empty_cache()
    # 0x7f00..0x7fff are all fp16 NaNs, distinct from torch.full's 0x7e00
    junk = torch.randint(0x7F00, 0x7FFF, (n,), device=DEV, dtype=torch.int16).view(torch.float16)
    del junk


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("kv_len", [1, 33, 577])
def test_partial_kv_tile_with_dirty_smem(kv_len, head_dim, is_causal):
    _dirty_allocator()
    # sageattn's causal path wants qo_len == kv_len; the partial tile is on
    # the KV side, so shrinking qo does not weaken the test.
    qo_len = kv_len if is_causal else 300
    g = torch.Generator(device=DEV).manual_seed(0)
    q = torch.randn((1, 4, qo_len, head_dim), device=DEV, dtype=torch.float16, generator=g)
    k = torch.randn((1, 4, kv_len, head_dim), device=DEV, dtype=torch.float16, generator=g)
    v = torch.randn((1, 4, kv_len, head_dim), device=DEV, dtype=torch.float16, generator=g)

    with torch.no_grad():
        out = sageattn(q, k, v, is_causal=is_causal, smooth_k=False)

    assert not torch.isnan(out).any(), (
        f"NaNs in output: {torch.isnan(out).sum().item()}/{out.numel()}"
    )
    ref = sdpa_ref(q, k, v, is_causal=is_causal)
    cs = cos_sim(out, ref)
    l1 = rel_l1(out, ref)
    assert cs > COS_MIN and l1 < REL_L1_MAX, f"cos_sim={cs:.5f} rel_l1={l1:.5f}"
