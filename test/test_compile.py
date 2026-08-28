"""torch.compile integration.

sageattn() must trace end-to-end without a graph break (the plan lookup is a
constant-folded dict, all ops carry fake kernels), stay numerically identical
to eager, survive CUDA graphs, and produce a single dynamic graph across
varying sequence lengths.
"""

import pytest
import torch

from conftest import requires_backend
from sageattention import sageattn

DEV = "cuda"
pytestmark = requires_backend("sm80")

B, H, QO, HEAD_DIM = 2, 4, 256, 64


def _qkv(kv_len=QO, qo_len=QO, head_dim=HEAD_DIM, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn((B, H, qo_len, head_dim), device=DEV, dtype=torch.float16, generator=g)
    k = torch.randn((B, H, kv_len, head_dim), device=DEV, dtype=torch.float16, generator=g)
    v = torch.randn((B, H, kv_len, head_dim), device=DEV, dtype=torch.float16, generator=g)
    return q, k, v


def _reset():
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()


@pytest.mark.parametrize("smooth_k", [True, False])
@pytest.mark.parametrize("return_lse", [False, True])
def test_no_graph_breaks(smooth_k, return_lse):
    _reset()
    q, k, v = _qkv()

    def fn(q, k, v):
        return sageattn(q, k, v, is_causal=True, return_lse=return_lse, smooth_k=smooth_k)

    with torch.no_grad():
        explanation = torch._dynamo.explain(fn)(q, k, v)
    assert explanation.graph_break_count == 0, explanation.break_reasons


@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("is_causal", [False, True])
def test_compiled_matches_eager(layout, is_causal):
    _reset()
    q, k, v = _qkv()
    if layout == "NHD":
        q, k, v = (t.transpose(1, 2).contiguous() for t in (q, k, v))

    compiled = torch.compile(sageattn, fullgraph=True)
    with torch.no_grad():
        want = sageattn(q, k, v, tensor_layout=layout, is_causal=is_causal)
        got = compiled(q, k, v, tensor_layout=layout, is_causal=is_causal)
    assert torch.equal(got, want)


def _assert_lse_equal(got_lse, want_lse, smooth_k):
    """The kernel lse is bit-identical; the smooth_k correction
    (lse/log2e + q@km^T * sm_scale) is plain ATen, and inductor contracts it
    into an FMA, so that path is only equal to within one fp32 ulp."""
    if smooth_k:
        torch.testing.assert_close(got_lse, want_lse, rtol=1e-6, atol=1e-6)
    else:
        assert torch.equal(got_lse, want_lse)


@pytest.mark.parametrize("smooth_k", [True, False])
def test_compiled_matches_eager_return_lse(smooth_k):
    _reset()
    q, k, v = _qkv()
    compiled = torch.compile(sageattn, fullgraph=True)
    with torch.no_grad():
        want_out, want_lse = sageattn(q, k, v, is_causal=True, return_lse=True, smooth_k=smooth_k)
        got_out, got_lse = compiled(q, k, v, is_causal=True, return_lse=True, smooth_k=smooth_k)
    assert torch.equal(got_out, want_out)
    _assert_lse_equal(got_lse, want_lse, smooth_k)


@pytest.mark.parametrize("head_dim", [64, 128])
def test_reduce_overhead_cudagraphs(head_dim):
    """mode="reduce-overhead" runs under CUDA graphs, which recycle the output
    buffer — clone before comparing."""
    _reset()
    q, k, v = _qkv(head_dim=head_dim)
    compiled = torch.compile(sageattn, mode="reduce-overhead", fullgraph=True)
    with torch.no_grad():
        want = sageattn(q, k, v, is_causal=True)
        for _ in range(4):
            got = compiled(q, k, v, is_causal=True).clone()
            assert torch.equal(got, want)


def test_dynamic_shapes_single_graph():
    _reset()
    compiled = torch.compile(sageattn, dynamic=True, fullgraph=True)
    with torch.no_grad():
        for i, kv_len in enumerate((512, 640, 777)):
            q, k, v = _qkv(kv_len=kv_len, seed=i)
            want = sageattn(q, k, v)
            got = compiled(q, k, v)
            assert torch.equal(got, want), kv_len
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == 1, dict(
        torch._dynamo.utils.counters["stats"]
    )


def test_dynamic_shapes_single_graph_return_lse():
    _reset()
    compiled = torch.compile(sageattn, dynamic=True, fullgraph=True)
    with torch.no_grad():
        for i, kv_len in enumerate((512, 640, 777)):
            q, k, v = _qkv(kv_len=kv_len, seed=i)
            want_out, want_lse = sageattn(q, k, v, return_lse=True)
            got_out, got_lse = compiled(q, k, v, return_lse=True)
            assert torch.equal(got_out, want_out), kv_len
            _assert_lse_equal(got_lse, want_lse, smooth_k=True)
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == 1, dict(
        torch._dynamo.utils.counters["stats"]
    )


def test_head_dim_padding_compiles():
    """head_dim 72 pads to 128 inside the traced region."""
    _reset()
    q, k, v = _qkv(head_dim=72)
    compiled = torch.compile(sageattn, fullgraph=True)
    with torch.no_grad():
        want = sageattn(q, k, v)
        got = compiled(q, k, v)
    assert got.shape == q.shape
    assert torch.equal(got, want)
