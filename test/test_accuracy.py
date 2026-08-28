"""Numerical accuracy of sageattn against an fp32 SDPA math reference.

Quantized attention is lossy by construction, so the gates are the ones the
SageAttention papers report: cosine similarity and relative L1 error.
"""

import pytest
import torch

from conftest import cos_sim, ref_scores, rel_l1, requires_backend, sdpa_ref
from sageattention import sageattn

DEV = "cuda"
pytestmark = requires_backend("sm80")

COS_MIN = 0.99
REL_L1_MAX = 0.06

# The first causal rows softmax over 1-2 keys, so their lse *is* a single
# quantized QK score: |lse| ~ 0.2 there while a full row sits around 5-6.
# rtol alone would gate those rows at int8 noise level, hence the atol floor
# (measured worst case over the sweep below: 0.035).
LSE_ATOL = 5e-2


def _qkv(
    b=2,
    h_qo=4,
    h_kv=None,
    qo_len=512,
    kv_len=None,
    head_dim=64,
    layout="HND",
    dtype=torch.float16,
    seed=0,
):
    h_kv = h_qo if h_kv is None else h_kv
    kv_len = qo_len if kv_len is None else kv_len
    g = torch.Generator(device=DEV).manual_seed(seed)

    def make(h, n):
        shape = (b, h, n, head_dim) if layout == "HND" else (b, n, h, head_dim)
        return torch.randn(shape, device=DEV, dtype=dtype, generator=g)

    return make(h_qo, qo_len), make(h_kv, kv_len), make(h_kv, kv_len)


def _check(out, ref, tag=""):
    cs = cos_sim(out, ref)
    l1 = rel_l1(out, ref)
    assert cs > COS_MIN, f"{tag} cos_sim={cs:.5f} rel_l1={l1:.5f}"
    assert l1 < REL_L1_MAX, f"{tag} cos_sim={cs:.5f} rel_l1={l1:.5f}"


# ------------------------------------------------------------ main sweep


@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("gran", ["per_warp", "per_thread"])
@pytest.mark.parametrize("pv", ["fp32", "fp16", "fp16+fp32"])
@pytest.mark.parametrize("smooth_k", [True, False])
def test_accuracy_sm80(layout, is_causal, head_dim, gran, pv, smooth_k):
    q, k, v = _qkv(head_dim=head_dim, layout=layout)
    with torch.no_grad():
        out = sageattn(
            q,
            k,
            v,
            tensor_layout=layout,
            is_causal=is_causal,
            qk_quant_gran=gran,
            pv_accum_dtype=pv,
            smooth_k=smooth_k,
        )
    ref = sdpa_ref(q, k, v, tensor_layout=layout, is_causal=is_causal)
    _check(out, ref, f"{layout}/{head_dim}/{gran}/{pv}/smooth_k={smooth_k}")


@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("smooth_k", [True, False])
def test_accuracy_smooth_v(layout, is_causal, smooth_k):
    """smooth_v is only fused on sm80 for the pure-fp16 accumulator."""
    q, k, v = _qkv(layout=layout)
    with torch.no_grad():
        out = sageattn(
            q,
            k,
            v,
            tensor_layout=layout,
            is_causal=is_causal,
            pv_accum_dtype="fp16",
            smooth_v=True,
            smooth_k=smooth_k,
        )
    ref = sdpa_ref(q, k, v, tensor_layout=layout, is_causal=is_causal)
    _check(out, ref, f"smooth_v/{layout}/causal={is_causal}")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_dtypes(dtype):
    q, k, v = _qkv(dtype=dtype)
    with torch.no_grad():
        out = sageattn(q, k, v, is_causal=True)
    ref = sdpa_ref(q, k, v, is_causal=True)
    _check(out, ref, str(dtype))


# ------------------------------------------------------------ shape edges


@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("is_causal", [False, True])
def test_accuracy_gqa(layout, is_causal):
    q, k, v = _qkv(h_qo=8, h_kv=2, layout=layout)
    with torch.no_grad():
        out = sageattn(q, k, v, tensor_layout=layout, is_causal=is_causal)
    ref = sdpa_ref(q, k, v, tensor_layout=layout, is_causal=is_causal)
    _check(out, ref, f"gqa/{layout}")


@pytest.mark.parametrize("qo_len,kv_len", [(256, 1024), (1024, 256), (300, 777)])
def test_accuracy_unequal_seq_len(qo_len, kv_len):
    """qo_len != kv_len is only defined for the non-causal path."""
    q, k, v = _qkv(qo_len=qo_len, kv_len=kv_len)
    with torch.no_grad():
        out = sageattn(q, k, v, is_causal=False)
    ref = sdpa_ref(q, k, v, is_causal=False)
    _check(out, ref, f"{qo_len}x{kv_len}")


@pytest.mark.parametrize("head_dim", [40, 72, 96])
def test_accuracy_padded_head_dim(head_dim):
    q, k, v = _qkv(head_dim=head_dim)
    with torch.no_grad():
        out = sageattn(q, k, v, is_causal=True)
    ref = sdpa_ref(q, k, v, is_causal=True)
    assert out.shape == q.shape
    _check(out, ref, f"head_dim={head_dim}")


def test_accuracy_custom_sm_scale():
    q, k, v = _qkv()
    with torch.no_grad():
        out = sageattn(q, k, v, sm_scale=0.05)
    ref = sdpa_ref(q, k, v, sm_scale=0.05)
    _check(out, ref, "sm_scale=0.05")


# -------------------------------------------------------------------- lse


@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("smooth_k", [True, False])
def test_lse_matches_logsumexp(layout, is_causal, smooth_k):
    """sageattn returns the natural-log lse; the reference is logsumexp over
    the fp32 pre-softmax scores."""
    q, k, v = _qkv(qo_len=256, layout=layout)
    with torch.no_grad():
        out, lse = sageattn(
            q, k, v, tensor_layout=layout, is_causal=is_causal, return_lse=True, smooth_k=smooth_k
        )
    scores = ref_scores(q, k, tensor_layout=layout, is_causal=is_causal)
    ref_lse = torch.logsumexp(scores, dim=-1)
    assert lse.shape == ref_lse.shape
    torch.testing.assert_close(lse, ref_lse, rtol=2e-2, atol=LSE_ATOL)


def test_lse_gqa():
    q, k, v = _qkv(h_qo=8, h_kv=2, qo_len=256)
    with torch.no_grad():
        _, lse = sageattn(q, k, v, is_causal=True, return_lse=True)
    ref_lse = torch.logsumexp(ref_scores(q, k, is_causal=True), dim=-1)
    torch.testing.assert_close(lse, ref_lse, rtol=2e-2, atol=LSE_ATOL)
