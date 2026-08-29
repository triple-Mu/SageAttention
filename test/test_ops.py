"""torch.library.opcheck coverage for every torch.ops.sageattention.* operator.

opcheck validates the schema (including the `Tensor(a!)` mutation
annotations), the autograd registration, and the fake/meta kernels against the
real CUDA implementation, which is what torch.compile relies on.
"""

import pytest
import torch
from torch.library import opcheck

from conftest import (
    ALL_BACKENDS,
    CC,
    backend_available,
    requires_backend,
    requires_cuda,
    requires_fp8_backend,
    requires_fp8_cast,
)
from sageattention import sageattn
from sageattention._plan import Plan

DEV = "cuda"
OPS = torch.ops.sageattention

# opcheck's "test_schema" util (SchemaCheckMode) clones/compares inputs with
# ops like mul, which torch 2.13 does not implement for Float8_e4m3fn on CUDA;
# fp8-involving cases run every util except that one.
NO_SCHEMA_FP8 = ("test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic")

# sm80 quantization tile geometry (see plan.cpp fill_tiles)
BLKQ, BLKK, WARPK = 128, 64, 64


def _warp_q(head_dim, pv_accum_dtype):
    return 16 if (head_dim == 128 and pv_accum_dtype == "fp16+fp32") else 32


def _qk_shapes(layout, b, h_qo, h_kv, qo_len, kv_len, head_dim):
    if layout == "HND":
        return (b, h_qo, qo_len, head_dim), (b, h_kv, kv_len, head_dim)
    return (b, qo_len, h_qo, head_dim), (b, kv_len, h_kv, head_dim)


def _cdiv(a, b):
    return (a + b - 1) // b


# --------------------------------------------------------------- fwd op


def _fwd_args(
    layout,
    head_dim,
    pv,
    gran,
    b=2,
    h=4,
    qo_len=256,
    kv_len=256,
    smooth_v=False,
    dtype=torch.float16,
):
    qshape, kshape = _qk_shapes(layout, b, h, h, qo_len, kv_len, head_dim)
    q = torch.randint(-100, 100, qshape, device=DEV, dtype=torch.int8)
    k = torch.randint(-100, 100, kshape, device=DEV, dtype=torch.int8)
    v = torch.randn(kshape, device=DEV, dtype=torch.float16)

    warp_q = _warp_q(head_dim, pv)
    q_len = _cdiv(qo_len, BLKQ) * (BLKQ // warp_q)
    k_len = _cdiv(kv_len, BLKK)
    if gran == "per_thread":
        q_len *= 8
        k_len *= (BLKK // WARPK) * 4
    q_scale = torch.rand((b, h, q_len), device=DEV, dtype=torch.float32) * 0.01 + 0.001
    k_scale = torch.rand((b, h, k_len), device=DEV, dtype=torch.float32) * 0.01 + 0.001

    value_mean = None
    if smooth_v:
        # the sm80 fused-v_mean kernel requires value_mean.dtype == out dtype
        value_mean = torch.randn((b, h, head_dim), device=DEV, dtype=dtype)
    return (q, k, v, q_scale, k_scale, None, value_mean)


@requires_backend("sm80")
@pytest.mark.parametrize("pv", ["fp32", "fp16", "fp16+fp32"])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("return_lse", [False, True])
def test_opcheck_fwd_sm80(pv, layout, head_dim, return_lse):
    gran = "per_warp"
    args = _fwd_args(layout, head_dim, pv, gran)
    kwargs = dict(
        tensor_layout=layout,
        qk_quant_gran=gran,
        pv_accum_dtype=pv,
        v_layout="seq",
        is_causal=False,
        sm_scale=head_dim**-0.5,
        return_lse=return_lse,
        out_dtype=torch.float16,
    )
    opcheck(OPS.fwd.default, args, kwargs)


@requires_backend("sm80")
@pytest.mark.parametrize("gran", ["per_warp", "per_thread"])
@pytest.mark.parametrize("is_causal", [False, True])
def test_opcheck_fwd_sm80_gran_causal(gran, is_causal):
    args = _fwd_args("HND", 128, "fp32", gran)
    kwargs = dict(
        tensor_layout="HND",
        qk_quant_gran=gran,
        pv_accum_dtype="fp32",
        v_layout="seq",
        is_causal=is_causal,
        sm_scale=0.08838834764831845,
        return_lse=True,
        out_dtype=torch.float16,
    )
    opcheck(OPS.fwd.default, args, kwargs)


@requires_backend("sm80")
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_opcheck_fwd_sm80_smooth_v(out_dtype):
    """pv=fp16 is the only sm80 config with a fused v_mean kernel."""
    args = _fwd_args("HND", 64, "fp16", "per_warp", smooth_v=True, dtype=out_dtype)
    kwargs = dict(
        tensor_layout="HND",
        qk_quant_gran="per_warp",
        pv_accum_dtype="fp16",
        v_layout="seq",
        is_causal=False,
        sm_scale=0.125,
        return_lse=False,
        out_dtype=out_dtype,
    )
    opcheck(OPS.fwd.default, args, kwargs)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
def test_opcheck_fwd_plan_driven(backend, head_dim, layout):
    """Same test for every arch family: derive the whole input geometry from
    the plan, so it stays correct wherever it happens to run. Exactly one
    backend is live on a given GPU; the rest skip."""
    if not backend_available(backend):
        pytest.skip(f"{backend} is not the backend this device resolves to")

    p = Plan(*torch.ops.sageattention.plan(CC[0], CC[1], head_dim, None, None, None, None))
    assert not p.error, p.error

    b, h, seq = 2, 4, 256
    qshape, kshape = _qk_shapes(layout, b, h, h, seq, seq, head_dim)
    q = torch.randint(-100, 100, qshape, device=DEV, dtype=torch.int8)
    k = torch.randint(-100, 100, kshape, device=DEV, dtype=torch.int8)

    q_len = _cdiv(seq, p.blk_q) * (p.blk_q // p.warp_q)
    k_len = _cdiv(seq, p.blk_k)
    if p.qk_quant_gran == "per_thread":
        q_len *= 8
        k_len *= (p.blk_k // p.warp_k) * 4
    q_scale = torch.rand((b, h, q_len), device=DEV, dtype=torch.float32) * 0.01 + 0.001
    k_scale = torch.rand((b, h, k_len), device=DEV, dtype=torch.float32) * 0.01 + 0.001

    v_fp16 = torch.randn(kshape, device=DEV, dtype=torch.float16)
    if p.pv_fp8:
        v, value_scale, value_mean = torch.ops.sageattention.quant_v_fp8(
            v_fp16,
            tensor_layout=layout,
            v_layout=p.v_layout,
            scale_max=p.v_scale_max,
            smooth_v=p.smooth_v,
            pad_multiple=p.v_pad_multiple,
        )
    else:
        v, value_scale, value_mean = v_fp16, None, None

    opcheck(
        OPS.fwd.default,
        (q, k, v, q_scale, k_scale, value_scale, value_mean),
        dict(
            tensor_layout=layout,
            qk_quant_gran=p.qk_quant_gran,
            pv_accum_dtype=p.pv_accum_dtype,
            v_layout=p.v_layout,
            is_causal=True,
            sm_scale=head_dim**-0.5,
            return_lse=True,
            out_dtype=torch.float16,
        ),
        test_utils=NO_SCHEMA_FP8 if p.pv_fp8 else "ALL",
    )


# ------------------------------------------------------ per-arch qattn ops


def _qattn_args(head_dim=64, pv="fp32", layout="HND", b=2, h=4, seq=256, gran="per_warp"):
    q, k, v, q_scale, k_scale, _, _ = _fwd_args(
        layout, head_dim, pv, gran, b=b, h=h, qo_len=seq, kv_len=seq
    )
    qshape, _ = _qk_shapes(layout, b, h, h, seq, seq, head_dim)
    out = torch.empty(qshape, device=DEV, dtype=torch.float16)
    return q, k, v, out, q_scale, k_scale


@requires_backend("sm80")
@pytest.mark.parametrize(
    "op_name,pv",
    [
        ("qattn_sm80_qk_int8_sv_f16_accum_f32_attn", "fp32"),
        ("qattn_sm80_qk_int8_sv_f16_accum_f16_attn", "fp16"),
        ("qattn_sm80_qk_int8_sv_f16_accum_f16_attn_inst_buf", "fp16+fp32"),
    ],
)
@pytest.mark.parametrize("return_lse", [False, True])
def test_opcheck_qattn_sm80_base(op_name, pv, return_lse):
    q, k, v, out, q_scale, k_scale = _qattn_args(head_dim=64, pv=pv)
    op = getattr(OPS, op_name).default
    opcheck(op, (q, k, v, out, q_scale, k_scale, "HND", False, "per_warp", 0.125, return_lse))


@requires_backend("sm80")
@pytest.mark.parametrize("layout", ["HND", "NHD"])
def test_opcheck_qattn_sm80_fuse_v_mean(layout):
    b, h, seq, head_dim = 2, 4, 256, 64
    q, k, v, out, q_scale, k_scale = _qattn_args(
        head_dim=head_dim, pv="fp16", layout=layout, b=b, h=h, seq=seq
    )
    v_mean = torch.randn((b, h, head_dim), device=DEV, dtype=torch.float16)
    opcheck(
        OPS.qattn_sm80_qk_int8_sv_f16_accum_f16_fuse_v_mean_attn.default,
        (q, k, v, out, q_scale, k_scale, v_mean, layout, False, "per_warp", 0.125, True),
    )


@requires_backend("sm80")
def test_opcheck_qattn_sm80_per_thread():
    q, k, v, out, q_scale, k_scale = _qattn_args(head_dim=128, pv="fp32", gran="per_thread")
    opcheck(
        OPS.qattn_sm80_qk_int8_sv_f16_accum_f32_attn.default,
        (q, k, v, out, q_scale, k_scale, "HND", True, "per_thread", 0.08838834764831845, True),
    )


# ------------------------------------------------------- quantization ops

B, H, SEQ, HEAD_DIM = 2, 4, 256, 64
LAYOUTS = ["HND", "NHD"]


def _v_shape(layout, b=B, h=H, seq=SEQ, head_dim=HEAD_DIM):
    return (b, h, seq, head_dim) if layout == "HND" else (b, seq, h, head_dim)


@requires_cuda
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("with_mean", [False, True])
def test_opcheck_quant_per_block_int8(layout, with_mean):
    shape = _v_shape(layout)
    inp = torch.randn(shape, device=DEV, dtype=torch.float16)
    out = torch.empty(shape, device=DEV, dtype=torch.int8)
    scale = torch.empty((B, H, _cdiv(SEQ, BLKK)), device=DEV, dtype=torch.float32)
    mean = torch.randn((B, H, HEAD_DIM), device=DEV, dtype=torch.float16) if with_mean else None
    # sm_scale and mean are mutually exclusive in the op
    sm_scale = None if with_mean else 0.125
    opcheck(OPS.quant_per_block_int8.default, (inp, mean, out, scale, sm_scale, BLKK, layout))


@requires_cuda
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("warp_q", [16, 32])
def test_opcheck_quant_per_warp_int8(layout, warp_q):
    shape = _v_shape(layout)
    inp = torch.randn(shape, device=DEV, dtype=torch.float16)
    out = torch.empty(shape, device=DEV, dtype=torch.int8)
    scale = torch.empty(
        (B, H, _cdiv(SEQ, BLKQ) * (BLKQ // warp_q)), device=DEV, dtype=torch.float32
    )
    opcheck(OPS.quant_per_warp_int8.default, (inp, out, scale, BLKQ, warp_q, layout))


@requires_cuda
@pytest.mark.parametrize("layout", LAYOUTS)
def test_opcheck_quant_per_thread_int8_q(layout):
    shape = _v_shape(layout)
    inp = torch.randn(shape, device=DEV, dtype=torch.float16)
    out = torch.empty(shape, device=DEV, dtype=torch.int8)
    scale = torch.empty(
        (B, H, _cdiv(SEQ, BLKQ) * (BLKQ // 32) * 8), device=DEV, dtype=torch.float32
    )
    opcheck(OPS.quant_per_thread_int8_q.default, (inp, out, scale, BLKQ, 32, layout))


@requires_cuda
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("with_mean", [False, True])
def test_opcheck_quant_per_thread_int8_k(layout, with_mean):
    shape = _v_shape(layout)
    inp = torch.randn(shape, device=DEV, dtype=torch.float16)
    out = torch.empty(shape, device=DEV, dtype=torch.int8)
    scale = torch.empty(
        (B, H, _cdiv(SEQ, BLKK) * (BLKK // WARPK) * 4), device=DEV, dtype=torch.float32
    )
    mean = torch.randn((B, H, HEAD_DIM), device=DEV, dtype=torch.float16) if with_mean else None
    opcheck(OPS.quant_per_thread_int8_k.default, (inp, mean, out, scale, BLKK, WARPK, layout))


@requires_cuda
@pytest.mark.parametrize("layout", LAYOUTS)
def test_opcheck_sub_mean(layout):
    shape = _v_shape(layout)
    v = torch.randn(shape, device=DEV, dtype=torch.float16)
    vm = v.mean(dim=1 if layout == "NHD" else 2)
    out = torch.empty(shape, device=DEV, dtype=torch.float16)
    opcheck(OPS.sub_mean.default, (v, vm, out, layout))


def _transposed_v(layout, kv_len=SEQ, pad_multiple=64, head_dim=HEAD_DIM):
    padded = _cdiv(kv_len, pad_multiple) * pad_multiple
    if layout == "HND":
        return (B, H, head_dim, padded)
    return (B, head_dim, H, padded)


@requires_cuda
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("permute", [False, True])
def test_opcheck_transpose_pad_v(layout, permute):
    v = torch.randn(_v_shape(layout), device=DEV, dtype=torch.float16)
    out = torch.empty(_transposed_v(layout), device=DEV, dtype=torch.float16)
    opcheck(OPS.transpose_pad_v.default, (v, out, layout, permute))


# kv_len % 128 in (0, 64] is where the 128-aligned output is strictly longer
# than the 64-aligned range the kernel writes; 1000 also crosses a CTA boundary.
PAD128_KV = [64, 320, 1000]


@requires_cuda
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("permute", [False, True])
@pytest.mark.parametrize("kv_len", PAD128_KV)
def test_transpose_pad_v_output_longer_than_64_aligned(layout, permute, kv_len):
    """quant_v_fp8(pad_multiple=128) hands this op a 128-aligned output. The
    kernel writes the 64-aligned prefix and must leave the rest alone rather
    than reject the shape."""
    v = torch.randn(_v_shape(layout, seq=kv_len), device=DEV, dtype=torch.float16)
    out = torch.full(
        _transposed_v(layout, kv_len, 128), float("nan"), device=DEV, dtype=torch.float16
    )
    OPS.transpose_pad_v(v, out, layout, permute)

    written = _cdiv(kv_len, 64) * 64
    assert not out[..., :written].isnan().any()
    assert out[..., written:].isnan().all(), "wrote past the 64-aligned bound"
    # mma_k16 permutes inside 16-token groups, so the zero-fill bound rounds up
    bound = _cdiv(kv_len, 16) * 16 if permute else kv_len
    assert (out[..., bound:written] == 0).all()

    ref = torch.empty(_transposed_v(layout, kv_len), device=DEV, dtype=torch.float16)
    OPS.transpose_pad_v(v, ref, layout, permute)
    assert torch.equal(out[..., :written], ref)


@requires_cuda
@pytest.mark.parametrize("layout", LAYOUTS)
def test_transpose_pad_v_rejects_bad_output_seq(layout):
    v = torch.randn(_v_shape(layout, seq=320), device=DEV, dtype=torch.float16)
    for last in (256, 384 + 32):  # shorter than 64-aligned / not 64-aligned
        shape = list(_transposed_v(layout, 320, 128))
        shape[3] = last
        out = torch.empty(shape, device=DEV, dtype=torch.float16)
        with pytest.raises(RuntimeError, match="multiple of"):
            OPS.transpose_pad_v(v, out, layout, True)


@requires_fp8_cast
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("v_layout", ["mma_k16", "linear"])
@pytest.mark.parametrize("kv_len", PAD128_KV)
def test_quant_v_fp8_pad_multiple_128(layout, v_layout, kv_len):
    """pad_multiple=128 (sm90/sm100) must equal python-side zero padding
    followed by the 64-aligned quantization, and leave a zero fp8 tail."""
    seq_dim = 1 if layout == "NHD" else 2
    v = torch.randn(_v_shape(layout, seq=kv_len), device=DEV, dtype=torch.float16)

    padded = _cdiv(kv_len, 128) * 128
    pad_shape = list(v.shape)
    pad_shape[seq_dim] = padded - kv_len
    v_ref_src = torch.cat([v, torch.zeros(pad_shape, dtype=v.dtype, device=DEV)], dim=seq_dim)

    kw = dict(tensor_layout=layout, v_layout=v_layout, scale_max=448.0, smooth_v=False)
    ref = OPS.quant_v_fp8(v_ref_src, pad_multiple=64, **kw)
    got = OPS.quant_v_fp8(v, pad_multiple=128, **kw)

    assert got[0].size(3) == padded
    assert torch.equal(ref[0], got[0])
    assert torch.equal(ref[1], got[1])
    bound = _cdiv(kv_len, 16) * 16 if v_layout == "mma_k16" else kv_len
    assert not got[0][..., bound:].view(torch.uint8).any()


# On and off every boundary the fused kernel has to reproduce: the ceil16
# statistics bound, the ceil64 quantize bound, and the 2048-token round the
# reduction leaves are laid out in.
FUSED_V_KV = [16, 100, 320, 1000, 2048, 2049, 4096]


@requires_fp8_cast
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("v_layout", ["mma_k16", "linear"])
@pytest.mark.parametrize("smooth_v", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_quant_v_fp8_matches_two_kernel_path(layout, v_layout, smooth_v, dtype):
    """quant_v_fp8 runs the transpose and the fp8 quantization as one kernel.
    It has to stay bit-identical to transpose_pad_v + (mean_)scale_fuse_quant,
    which is the pair the golden dumps were taken with and which
    SAGE_FUSED_V_QUANT=OFF falls back to."""
    kw = dict(tensor_layout=layout, v_layout=v_layout, scale_max=448.0, smooth_v=smooth_v)
    for head_dim in (64, 128):
        for kv_len in FUSED_V_KV:
            shape = _v_shape(layout, seq=kv_len, head_dim=head_dim)
            v = torch.randn(shape, device=DEV, dtype=dtype)
            got = OPS.quant_v_fp8(v, pad_multiple=64, **kw)

            v_t = torch.empty(
                _transposed_v(layout, kv_len, 64, head_dim), device=DEV, dtype=dtype
            )
            OPS.transpose_pad_v(v, v_t, layout, v_layout == "mma_k16")
            v_fp8 = torch.zeros(v_t.shape, device=DEV, dtype=torch.float8_e4m3fn)
            v_scale = torch.empty((B, H, head_dim), device=DEV, dtype=torch.float32)
            v_mean = torch.empty((B, H, head_dim), device=DEV, dtype=torch.float32)
            if smooth_v:
                OPS.mean_scale_fuse_quant(v_t, v_fp8, v_mean, v_scale, kv_len, 448.0, layout)
            else:
                OPS.scale_fuse_quant(v_t, v_fp8, v_scale, kv_len, 448.0, layout)

            tag = f"head_dim={head_dim} kv_len={kv_len}"
            assert torch.equal(got[0].view(torch.uint8), v_fp8.view(torch.uint8)), tag
            assert torch.equal(got[1], v_scale), tag
            if smooth_v:
                assert torch.equal(got[2], v_mean), tag


@requires_fp8_cast
@pytest.mark.parametrize("layout", LAYOUTS)
def test_opcheck_scale_fuse_quant(layout):
    shape = _transposed_v(layout)
    inp = torch.randn(shape, device=DEV, dtype=torch.float16)
    out = torch.empty(shape, device=DEV, dtype=torch.float8_e4m3fn)
    scale = torch.empty((B, H, HEAD_DIM), device=DEV, dtype=torch.float32)
    opcheck(
        OPS.scale_fuse_quant.default,
        (inp, out, scale, SEQ, 448.0, layout),
        test_utils=NO_SCHEMA_FP8,
    )


@requires_fp8_cast
@pytest.mark.parametrize("layout", LAYOUTS)
def test_opcheck_mean_scale_fuse_quant(layout):
    shape = _transposed_v(layout)
    inp = torch.randn(shape, device=DEV, dtype=torch.float16)
    out = torch.empty(shape, device=DEV, dtype=torch.float8_e4m3fn)
    mean = torch.empty((B, H, HEAD_DIM), device=DEV, dtype=torch.float32)
    scale = torch.empty((B, H, HEAD_DIM), device=DEV, dtype=torch.float32)
    opcheck(
        OPS.mean_scale_fuse_quant.default,
        (inp, out, mean, scale, SEQ, 2.25, layout),
        test_utils=NO_SCHEMA_FP8,
    )


# ---------------------------------- functional quant ops used by sageattn()


@requires_cuda
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("gran", ["per_warp", "per_thread"])
@pytest.mark.parametrize("with_mean", [False, True])
def test_opcheck_quant_qk(layout, gran, with_mean):
    shape = _v_shape(layout)
    q = torch.randn(shape, device=DEV, dtype=torch.float16)
    k = torch.randn(shape, device=DEV, dtype=torch.float16)
    # sageattn passes the keepdim mean straight through
    km = k.mean(dim=1 if layout == "NHD" else 2, keepdim=True) if with_mean else None
    opcheck(
        OPS.quant_qk.default,
        (q, k, km),
        dict(
            tensor_layout=layout,
            qk_quant_gran=gran,
            blk_q=BLKQ,
            warp_q=32,
            blk_k=BLKK,
            warp_k=WARPK,
        ),
    )


@requires_cuda
@pytest.mark.parametrize("layout", LAYOUTS)
def test_opcheck_sub_mean_v(layout):
    v = torch.randn(_v_shape(layout), device=DEV, dtype=torch.float16)
    opcheck(OPS.sub_mean_v.default, (v,), dict(tensor_layout=layout))


@requires_fp8_cast
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("v_layout", ["mma_k16", "linear"])
@pytest.mark.parametrize("smooth_v", [False, True])
def test_opcheck_quant_v_fp8(layout, v_layout, smooth_v):
    v = torch.randn(_v_shape(layout), device=DEV, dtype=torch.float16)
    opcheck(
        OPS.quant_v_fp8.default,
        (v,),
        dict(
            tensor_layout=layout,
            v_layout=v_layout,
            scale_max=448.0,
            smooth_v=smooth_v,
            pad_multiple=64,
        ),
    )


# ------------------------------------- fp8 V quantization, zero-amax channels

# arbitrary channel indices, < every supported head_dim; LIVE_CH stays random
ZERO_CH, CONST_CH, LIVE_CH = 3, 5, 0


@requires_fp8_cast
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("smooth_v", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_quant_v_fp8_zero_amax_channel(layout, smooth_v, head_dim):
    """A channel with amax == 0 used to quantize as 0 * (scale_max / 0) = NaN,
    i.e. a whole fp8 row of 0x7f that poisons its output channel through the
    PV mma. It must quantize to zero values with a zero scale instead. amax is
    zero for an all-zero channel and, under smooth_v, for a constant channel
    (max == min == mean). seq must stay 256: the constant-channel mean is only
    exact when the token count is 16-aligned and (3.0 * n) / n survives the
    fast-math reciprocal division, both of which hold for n = 256."""
    v = torch.randn(_v_shape(layout, seq=256, head_dim=head_dim), device=DEV, dtype=torch.float16)
    v[..., ZERO_CH] = 0.0
    v[..., CONST_CH] = 3.0

    v_fp8, v_scale, v_mean = OPS.quant_v_fp8(
        v,
        tensor_layout=layout,
        v_layout="mma_k16",
        scale_max=448.0,
        smooth_v=smooth_v,
        pad_multiple=64,
    )

    assert not v_fp8.float().isnan().any()
    d_dim = 2 if layout == "HND" else 1  # v_fp8 is transposed: [B,H,D,seq] / [B,D,H,seq]
    # compare values, not bytes: -0.0 (byte 0x80) is a legal quantization of 0
    assert (v_fp8.select(d_dim, ZERO_CH).float() == 0).all()
    assert (v_scale[..., ZERO_CH] == 0).all()
    # a healthy channel must not get zeroed along with the degenerate ones
    assert (v_scale[..., LIVE_CH] > 0).all()
    assert v_fp8.select(d_dim, LIVE_CH).view(torch.uint8).any()
    if smooth_v:
        assert (v_fp8.select(d_dim, CONST_CH).float() == 0).all()
        assert (v_scale[..., CONST_CH] == 0).all()
        assert (v_mean[..., CONST_CH] == 3.0).all()


@requires_fp8_cast
@pytest.mark.parametrize("layout", LAYOUTS)
def test_quant_v_fp8_subnormal_amax_channel(layout):
    """bf16 subnormals are fp32 subnormals, so scale_max / amax reaches inf
    without amax being zero (ftz flushes amax to zero, or the division itself
    overflows); the channel's zero elements then quantized to NaN exactly like
    the amax == 0 case. fp16 cannot trigger this: its smallest subnormal is a
    normal fp32 value."""
    v = torch.randn(_v_shape(layout), device=DEV, dtype=torch.bfloat16)
    v[..., ZERO_CH] = 0.0
    v[0, 0, 0, ZERO_CH] = 1e-40  # rounds to the smallest bf16 subnormal

    v_fp8, v_scale, _ = OPS.quant_v_fp8(
        v,
        tensor_layout=layout,
        v_layout="mma_k16",
        scale_max=448.0,
        smooth_v=False,
        pad_multiple=64,
    )

    assert not v_fp8.float().isnan().any()
    d_dim = 2 if layout == "HND" else 1
    assert (v_fp8.select(d_dim, ZERO_CH).float() == 0).all()
    # amax / scale_max: either flushed to zero or an fp32 subnormal
    assert (v_scale[..., ZERO_CH] < 1e-30).all()


@requires_fp8_backend
@pytest.mark.parametrize("smooth_v", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_sageattn_zero_v_channel_fp8(smooth_v, head_dim):
    """End-to-end on the fp8 PV backends: an all-zero V channel must come out
    as exactly zero, not NaN. smooth_v only reaches the fused v_mean kernel on
    the backends whose plan supports it (sm120); elsewhere it downgrades and
    duplicates the False case."""
    q, k, v = [
        torch.randn((2, 4, 256, head_dim), device=DEV, dtype=torch.float16) for _ in range(3)
    ]
    v[..., ZERO_CH] = 0.0
    with torch.no_grad():
        out = sageattn(q, k, v, is_causal=False, smooth_v=smooth_v)
    assert not out.isnan().any()
    assert (out[..., ZERO_CH] == 0).all()
    assert out[..., LIVE_CH].any()  # the rest of the output must survive


# ------------------------------------------------------------- fake coverage


def test_qattn_ops_list_covers_dispatcher():
    """`ops._QATTN_OPS` is hand-written; a qattn kernel added on the C++ side
    but not listed there would silently ship without a fake kernel and break
    torch.compile. Cross-check it against what the dispatcher actually has
    (the reverse does not hold: the list spans every arch, this build only
    carries some of them)."""
    from sageattention.ops import _QATTN_OPS

    # private API, but the only way to enumerate registrations by key
    registered = torch._C._dispatch_get_registrations_for_dispatch_key("CUDA")
    prefix = "sageattention::qattn_"
    dispatched = {name[len("sageattention::") :] for name in registered if name.startswith(prefix)}
    assert dispatched, "build carries no qattn CUDA registrations"
    assert dispatched <= set(_QATTN_OPS), sorted(dispatched - set(_QATTN_OPS))
