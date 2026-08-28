"""Public API contract of sageattention.sageattn.

The rewritten entry point replaced the old ``**kwargs`` catch-all with an
explicit signature, so unknown keywords must now fail loudly and the
SDPA-compatibility arguments must either work or raise NotImplementedError.
"""

import warnings

import pytest
import torch

from conftest import requires_backend
from sageattention import sageattn

DEV = "cuda"
pytestmark = requires_backend("sm80")


def _qkv(b=2, h=4, seq=256, head_dim=64, layout="HND", dtype=torch.float16, requires_grad=False):
    shape = (b, h, seq, head_dim) if layout == "HND" else (b, seq, h, head_dim)
    return [
        torch.randn(shape, device=DEV, dtype=dtype, requires_grad=requires_grad) for _ in range(3)
    ]


# ------------------------------------------------------------- signature


def test_typo_kwarg_raises_type_error():
    q, k, v = _qkv()
    with torch.no_grad():
        with pytest.raises(TypeError):
            sageattn(q, k, v, tensor_layotu="HND")


def test_unknown_kwarg_is_not_swallowed():
    q, k, v = _qkv()
    with torch.no_grad():
        with pytest.raises(TypeError):
            sageattn(q, k, v, backend="triton")


def test_advanced_knobs_are_keyword_only():
    q, k, v = _qkv()
    with torch.no_grad():
        with pytest.raises(TypeError):
            # 8th positional would be qk_quant_gran if it were positional
            sageattn(q, k, v, "HND", False, None, False, "per_warp")


# --------------------------------------------------- SDPA compatibility


def test_attn_mask_raises():
    q, k, v = _qkv()
    mask = torch.zeros((256, 256), device=DEV, dtype=torch.bool)
    with torch.no_grad():
        with pytest.raises(NotImplementedError):
            sageattn(q, k, v, attn_mask=mask)


def test_dropout_raises():
    q, k, v = _qkv()
    with torch.no_grad():
        with pytest.raises(NotImplementedError):
            sageattn(q, k, v, dropout_p=0.1)


def test_scale_maps_to_sm_scale():
    q, k, v = _qkv()
    with torch.no_grad():
        a = sageattn(q, k, v, scale=0.1)
        b = sageattn(q, k, v, sm_scale=0.1)
    assert torch.equal(a, b)


def test_scale_actually_changes_output():
    q, k, v = _qkv()
    with torch.no_grad():
        a = sageattn(q, k, v, scale=0.1)
        default = sageattn(q, k, v)
    assert not torch.equal(a, default)


def test_enable_gqa_flag_is_accepted():
    q = torch.randn((2, 8, 256, 64), device=DEV, dtype=torch.float16)
    k = torch.randn((2, 2, 256, 64), device=DEV, dtype=torch.float16)
    v = torch.randn((2, 2, 256, 64), device=DEV, dtype=torch.float16)
    with torch.no_grad():
        out = sageattn(q, k, v, enable_gqa=True)
    assert out.shape == q.shape


# ------------------------------------------------------------- autograd


def test_requires_grad_raises():
    q, k, v = _qkv(requires_grad=True)
    with pytest.raises(NotImplementedError):
        sageattn(q, k, v)


def test_requires_grad_ok_under_no_grad():
    q, k, v = _qkv(requires_grad=True)
    with torch.no_grad():
        out = sageattn(q, k, v)
    assert out.shape == q.shape and not out.requires_grad


# ---------------------------------------------------------------- shapes


@pytest.mark.parametrize("head_dim", [32, 72, 100, 128])
def test_head_dim_padding_preserves_shape(head_dim):
    q, k, v = _qkv(head_dim=head_dim)
    with torch.no_grad():
        out = sageattn(q, k, v)
    assert out.shape == q.shape
    assert out.size(-1) == head_dim


def test_head_dim_over_128_raises():
    q, k, v = _qkv(head_dim=160)
    with torch.no_grad():
        with pytest.raises(ValueError):
            sageattn(q, k, v)


@pytest.mark.parametrize("layout", ["HND", "NHD"])
def test_return_lse_shapes(layout):
    b, h, seq = 2, 4, 256
    q, k, v = _qkv(b=b, h=h, seq=seq, layout=layout)
    with torch.no_grad():
        out, lse = sageattn(q, k, v, tensor_layout=layout, return_lse=True)
    assert out.shape == q.shape
    assert lse.shape == (b, h, seq)
    assert lse.dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_output_dtype_follows_input(dtype):
    q, k, v = _qkv(dtype=dtype)
    with torch.no_grad():
        out = sageattn(q, k, v)
    assert out.dtype == dtype


# ------------------------------------------------------ argument policing


def test_lowercase_tensor_layout_raises():
    q, k, v = _qkv()
    with torch.no_grad():
        with pytest.raises((ValueError, RuntimeError)):
            sageattn(q, k, v, tensor_layout="hnd")


def test_unsupported_pv_accum_dtype_raises():
    """fp32+fp16 is an sm89/sm120 accumulator; sm80 must reject it."""
    q, k, v = _qkv()
    with torch.no_grad():
        with pytest.raises(ValueError):
            sageattn(q, k, v, pv_accum_dtype="fp32+fp16")


def test_bogus_qk_quant_gran_raises():
    q, k, v = _qkv()
    with torch.no_grad():
        with pytest.raises((ValueError, RuntimeError)):
            sageattn(q, k, v, qk_quant_gran="bogus")


def test_cpu_input_raises():
    q, k, v = (t.cpu() for t in _qkv())
    with torch.no_grad():
        with pytest.raises(AssertionError):
            sageattn(q, k, v)


def test_float32_input_raises():
    q, k, v = _qkv(dtype=torch.float32)
    with torch.no_grad():
        with pytest.raises(AssertionError):
            sageattn(q, k, v)


def test_smooth_v_downgrade_warns_once():
    """sm80 + fp32 has no fused v_mean kernel: smooth_v is dropped with a
    warning rather than silently honoured -- and _warned_configs keeps it to
    one warning per (cc, pv_accum_dtype), not one per call."""
    from sageattention import core

    core._warned_configs.clear()
    q, k, v = _qkv()
    try:
        with torch.no_grad():
            with pytest.warns(UserWarning, match="smooth_v will be ignored"):
                sageattn(q, k, v, pv_accum_dtype="fp32", smooth_v=True)

            with warnings.catch_warnings(record=True) as caught:
                # pytest installs its own filters; "always" makes the absence
                # of a warning below mean the code did not raise one
                warnings.simplefilter("always")
                sageattn(q, k, v, pv_accum_dtype="fp32", smooth_v=True)
        assert [str(w.message) for w in caught] == []
    finally:
        core._warned_configs.clear()
