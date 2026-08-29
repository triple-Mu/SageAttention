"""torch.ops.sageattention.plan / compiled_archs.

`sageattention._plan.PLAN` is an import-time snapshot of the C++ decision
table; it must stay bit-identical to what plan() returns on demand, otherwise
sageattn() would quantize for a different kernel than fwd() dispatches to.
"""

import pytest
import torch

import sageattention  # noqa: F401
from conftest import CC, COMPILED_ARCHS
from sageattention._plan import PLAN, Plan, get_plan

PLAN_OP = torch.ops.sageattention.plan


def test_compiled_archs_non_empty():
    archs = torch.ops.sageattention.compiled_archs()
    assert isinstance(archs, list) and archs, "build carries no kernel families"
    assert all(a in (80, 89, 90, 100, 120) for a in archs), archs
    assert archs == sorted(archs)


def test_plan_namedtuple_matches_op_schema():
    """`Plan(*plan_op(...))` unpacks positionally, so a field reordered or
    inserted on the C++ side would silently shift every value. The names in
    the schema's return list must match Plan._fields one for one."""
    schema_names = [r.name for r in PLAN_OP.default._schema.returns]
    assert list(Plan._fields) == schema_names


def test_plan_table_parity():
    """Every primed entry re-resolves to exactly the same 16-tuple."""
    assert PLAN, "the import-time plan table is empty"
    mismatches = []
    for (cc, head_dim, gran, pv, sv, varlen), cached in PLAN.items():
        fresh = Plan(*PLAN_OP(cc[0], cc[1], head_dim, None, gran, pv, sv, varlen))
        if tuple(fresh) != tuple(cached):
            mismatches.append((cc, head_dim, gran, pv, sv, varlen, tuple(cached), tuple(fresh)))
    assert not mismatches, mismatches[:5]


def test_plan_fields_are_typed():
    p = Plan(*PLAN_OP(8, 0, 128, None, None, None, None))
    assert p.backend == "sm80"
    assert p.qk_quant_gran in ("per_warp", "per_thread")
    assert p.pv_accum_dtype in ("fp32", "fp16", "fp16+fp32", "fp32+fp32", "fp32+fp16")
    assert isinstance(p.smooth_v, bool) and isinstance(p.pv_fp8, bool)
    assert p.v_layout in ("seq", "mma_k16", "linear")
    assert (p.blk_q, p.blk_k) == (128, 64)
    assert p.error == ""


@pytest.mark.skipif(CC is None, reason="needs CUDA")
def test_local_cc_resolves():
    p = Plan(*PLAN_OP(CC[0], CC[1], 128, None, None, None, None))
    if CC >= (8, 0):
        assert not p.error, p.error
        assert int(p.backend[2:]) in COMPILED_ARCHS
    else:
        assert p.error


@pytest.mark.skipif(
    CC is None or CC >= (8, 9) or CC < (8, 0),
    reason="only sm80-class devices resolve to the sm80 backend",
)
def test_local_cc_is_sm80_backend():
    p = Plan(*PLAN_OP(CC[0], CC[1], 128, None, None, None, None))
    assert p.backend == "sm80"
    assert p.pv_accum_dtype == "fp32"
    assert p.qk_quant_gran == "per_thread"
    assert p.pv_fp8 is False and p.v_layout == "seq"
    assert p.need_value_scale is False


@pytest.mark.parametrize(
    "backend,pv",
    [
        ("sm80", "fp32+fp16"),
        ("sm80", "fp32+fp32"),
        ("sm89", "fp16"),
        ("sm89", "fp16+fp32"),
        ("sm90", "fp32"),
        ("sm100", "fp16"),
        ("sm120", "fp32+fp32"),
    ],
)
def test_illegal_pv_accum_has_error(backend, pv):
    """Illegal (backend, pv_accum_dtype) pairs report an error string instead
    of silently returning garbage (the pre-refactor failure mode)."""
    p = Plan(*PLAN_OP(8, 0, 128, backend, None, pv, None))
    assert p.error, (backend, pv)
    if int(backend[2:]) in COMPILED_ARCHS:
        assert "pv_accum_dtype" in p.error


def test_illegal_head_dim_has_error():
    p = Plan(*PLAN_OP(8, 0, 72, None, None, None, None))
    assert "head_dim" in p.error


def test_pre_ampere_has_error():
    p = Plan(*PLAN_OP(7, 5, 128, None, None, None, None))
    assert "compute capability" in p.error


def test_bad_string_raises():
    with pytest.raises(RuntimeError):
        PLAN_OP(8, 0, 128, "sm42", None, None, None)
    with pytest.raises(RuntimeError):
        PLAN_OP(8, 0, 128, None, "bogus", None, None)
    with pytest.raises(RuntimeError):
        PLAN_OP(8, 0, 128, None, None, "int4", None)


def test_smooth_v_downgrade_is_flagged():
    """sm80 fuses v_mean only into the pure-fp16 accumulator; every other pv
    silently drops smooth_v and must say so via smooth_v_ignored."""
    fp16 = Plan(*PLAN_OP(8, 0, 128, "sm80", None, "fp16", True))
    assert fp16.smooth_v is True and fp16.smooth_v_ignored is False
    assert fp16.need_value_mean is True

    fp32 = Plan(*PLAN_OP(8, 0, 128, "sm80", None, "fp32", True))
    assert fp32.smooth_v is False and fp32.smooth_v_ignored is True
    assert fp32.need_value_mean is False


def test_sm80_warp_q_geometry():
    """warp_q shrinks to 16 only for (head_dim=128, fp16+fp32) — the scale
    tensor shape sageattn allocates depends on it."""
    assert Plan(*PLAN_OP(8, 0, 128, "sm80", None, "fp16+fp32", None)).warp_q == 16
    assert Plan(*PLAN_OP(8, 0, 64, "sm80", None, "fp16+fp32", None)).warp_q == 32
    assert Plan(*PLAN_OP(8, 0, 128, "sm80", None, "fp32", None)).warp_q == 32


def test_get_plan_raises_on_error():
    with pytest.raises(ValueError):
        get_plan((8, 0), 128, None, "fp32+fp16", None)
    with pytest.raises(ValueError):
        get_plan((7, 0), 128, None, None, None)


def test_get_plan_uncached_capability():
    """A cc outside the primed table falls through to the live op once, then
    is memoized so later calls stay a pure dict lookup (traceable)."""
    key = ((8, 5), 128, None, None, None, False)
    PLAN.pop(key, None)
    p = get_plan(*key)
    assert p.backend == "sm80" and not p.error
    assert PLAN.get(key) == p
