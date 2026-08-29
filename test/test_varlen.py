"""The varlen kernels: packed [total_tokens, heads, head_dim] inputs with the
sequence boundaries in a cu_seqlens prefix sum.

The gate that actually pins the kernels is here: on an equal-length batch the
packed path must be *bit-identical* to the dense one, output for output. The
dense and varlen kernels share their body (one quantization kernel with a
cu_seqlens argument, one attention kernel body compiled into two translation
units), so anything that drifts shows up as a torch.equal failure rather than
as a slow accuracy leak.

test_varlen_utils.py covers the block algebra and the segment reductions that
these kernels are addressed with.
"""

import pytest
import torch
from torch.library import opcheck

from conftest import (
    cos_sim,
    rel_l1,
    requires_backend,
    requires_cuda,
    requires_fp8_cast,
    requires_varlen,
    requires_varlen_backend,
)
from sageattention import sageattn, sageattn_varlen

DEV = "cuda"
OPS = torch.ops.sageattention
pytestmark = requires_varlen

# sm80 quantization tile geometry (plan.cpp fill_tiles)
BLKQ, WARPQ, BLKK, WARPK = 128, 32, 64, 64

# the sageattn accuracy gates (test_accuracy.py)
COS_MIN = 0.99
REL_L1_MAX = 0.06

# the kernels return the lse in base 2; core.py divides by this same literal
LOG2E = 1.44269504


def blk_offset(cu, b, cta_tokens):
    """csrc/sageattn/varlen.h: sequence b's first CTA-sized block."""
    return cu[b] // cta_tokens + b


def cdiv(a, b):
    return (a + b - 1) // b


def cu_of(lens):
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    return cu


def pack(dense, lens, layout="HND"):
    """[B, H, N, D] (or [B, N, H, D]) -> [sum(lens), H, D], sequence b keeping
    its first lens[b] tokens."""
    if layout == "HND":
        dense = dense.transpose(1, 2)  # [B, N, H, D]
    return torch.cat([dense[b, : lens[b]] for b in range(len(lens))], dim=0).contiguous()


def unpack_lse(lse, lens):
    """[heads, total] -> [batch, heads, n] for an equal-length batch."""
    h, total = lse.shape
    return lse.view(h, len(lens), lens[0]).transpose(0, 1).contiguous()


def rand_qkv(b, h_qo, h_kv, n, head_dim, layout="HND", seed=0, dtype=torch.float16):
    g = torch.Generator(device=DEV).manual_seed(seed)

    def one(h):
        shape = (b, h, n, head_dim) if layout == "HND" else (b, n, h, head_dim)
        return torch.randn(shape, device=DEV, dtype=dtype, generator=g)

    return one(h_qo), one(h_kv), one(h_kv)


# ------------------------------------------------------- quant_qk_varlen
# Equal-length batch: every packed output must equal the dense one, exactly.

QUANT_SHAPES = [
    # (batch, qo heads, kv heads, seq len)
    (1, 2, 2, 128),  # one sequence, block-aligned
    (3, 4, 2, 256),  # GQA
    (2, 2, 2, 100),  # not a multiple of blk_q or blk_k
    (4, 1, 1, 64),  # shorter than blk_q: the CTA is mostly padding
]


@requires_backend("sm80")
@pytest.mark.parametrize("gran", ["per_warp", "per_thread"])
@pytest.mark.parametrize("smooth_k", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("b,h_qo,h_kv,n", QUANT_SHAPES, ids=lambda p: str(p))
def test_quant_qk_varlen_equals_dense(gran, smooth_k, head_dim, b, h_qo, h_kv, n):
    q, k, _ = rand_qkv(b, h_qo, h_kv, n, head_dim, seed=hash((b, n, head_dim)) % 2**31)
    lens = [n] * b
    cu = torch.tensor(cu_of(lens), device=DEV, dtype=torch.int32)
    km = k.mean(dim=2) if smooth_k else None

    geom = dict(blk_q=BLKQ, warp_q=WARPQ, blk_k=BLKK, warp_k=WARPK)
    dense = OPS.quant_qk(q, k, km, tensor_layout="HND", qk_quant_gran=gran, **geom)
    got = OPS.quant_qk_varlen(
        pack(q, lens),
        pack(k, lens),
        cu,
        cu,
        km,
        max_seqlen_q=n,
        max_seqlen_k=n,
        qk_quant_gran=gran,
        **geom,
    )

    assert torch.equal(got[0], pack(dense[0], lens)), "q_int8"
    assert torch.equal(got[2], pack(dense[2], lens)), "k_int8"

    # the scales are block-indexed, so sequence b lives at its block base
    for name, dense_scale, packed_scale, blk, per_blk in (
        ("q_scale", dense[1], got[1], BLKQ, (BLKQ // WARPQ) * (8 if gran == "per_thread" else 1)),
        ("k_scale", dense[3], got[3], BLKK, (BLKK // WARPK) * (4 if gran == "per_thread" else 1)),
    ):
        width = cdiv(n, blk) * per_blk
        for seq in range(b):
            base = blk_offset(cu_of(lens), seq, blk) * per_blk
            assert torch.equal(packed_scale[:, base : base + width], dense_scale[seq, :, :width]), (
                f"{name} sequence {seq}"
            )


@requires_backend("sm80")
@pytest.mark.parametrize("gran", ["per_warp", "per_thread"])
def test_quant_qk_varlen_ragged_matches_per_sequence_dense(gran):
    """Ragged batch: each sequence quantized on its own (batch of 1) must come
    out exactly as its slice of the packed run. This is the property the
    equal-length gate above cannot see."""
    lens = [37, 128, 1, 300]
    head_dim, heads = 128, 2
    cu_list = cu_of(lens)
    total = cu_list[-1]
    g = torch.Generator(device=DEV).manual_seed(11)
    packed = torch.randn((total, heads, head_dim), device=DEV, dtype=torch.float16, generator=g)
    cu = torch.tensor(cu_list, device=DEV, dtype=torch.int32)

    geom = dict(blk_q=BLKQ, warp_q=WARPQ, blk_k=BLKK, warp_k=WARPK)
    got = OPS.quant_qk_varlen(
        packed,
        packed,
        cu,
        cu,
        None,
        max_seqlen_q=max(lens),
        max_seqlen_k=max(lens),
        qk_quant_gran=gran,
        **geom,
    )

    for seq, n in enumerate(lens):
        if n == 0:
            continue
        one = packed[cu_list[seq] : cu_list[seq] + n].transpose(0, 1).unsqueeze(0).contiguous()
        ref = OPS.quant_qk(one, one, None, tensor_layout="HND", qk_quant_gran=gran, **geom)
        assert torch.equal(got[0][cu_list[seq] : cu_list[seq] + n], ref[0][0].transpose(0, 1)), (
            f"q_int8 sequence {seq}"
        )
        for idx, blk, per_blk in (
            (1, BLKQ, (BLKQ // WARPQ) * (8 if gran == "per_thread" else 1)),
            (3, BLKK, (BLKK // WARPK) * (4 if gran == "per_thread" else 1)),
        ):
            width = cdiv(n, blk) * per_blk
            base = blk_offset(cu_list, seq, blk) * per_blk
            assert torch.equal(got[idx][:, base : base + width], ref[idx][0, :, :width]), (
                f"scale {idx} sequence {seq}"
            )


@requires_backend("sm80")
@pytest.mark.parametrize("gran", ["per_warp", "per_thread"])
@pytest.mark.parametrize("smooth_k", [False, True])
def test_opcheck_quant_qk_varlen(gran, smooth_k):
    """The fake kernel is what torch.compile traces; test_aot_dispatch_dynamic
    is the util that would catch a shape formula that disagrees with C++."""
    lens = [37, 128, 300]
    cu_list = cu_of(lens)
    cu = torch.tensor(cu_list, device=DEV, dtype=torch.int32)
    g = torch.Generator(device=DEV).manual_seed(5)
    packed = torch.randn((cu_list[-1], 2, 128), device=DEV, dtype=torch.float16, generator=g)
    km = None
    if smooth_k:
        km = torch.randn((len(lens), 2, 128), device=DEV, dtype=torch.float16, generator=g)

    opcheck(
        OPS.quant_qk_varlen.default,
        (packed, packed, cu, cu, km),
        dict(
            max_seqlen_q=max(lens),
            max_seqlen_k=max(lens),
            qk_quant_gran=gran,
            blk_q=BLKQ,
            warp_q=WARPQ,
            blk_k=BLKK,
            warp_k=WARPK,
        ),
    )


@requires_backend("sm80")
def test_quant_qk_varlen_empty_sequence():
    """An empty sequence owns one (unused) scale block and no tokens; its
    neighbours must be unaffected."""
    lens = [64, 0, 64]
    cu_list = cu_of(lens)
    cu = torch.tensor(cu_list, device=DEV, dtype=torch.int32)
    g = torch.Generator(device=DEV).manual_seed(3)
    packed = torch.randn((sum(lens), 2, 64), device=DEV, dtype=torch.float16, generator=g)

    got = OPS.quant_qk_varlen(
        packed,
        packed,
        cu,
        cu,
        None,
        max_seqlen_q=64,
        max_seqlen_k=64,
        qk_quant_gran="per_thread",
        blk_q=BLKQ,
        warp_q=WARPQ,
        blk_k=BLKK,
        warp_k=WARPK,
    )
    for seq in (0, 2):
        one = packed[cu_list[seq] : cu_list[seq + 1]].transpose(0, 1).unsqueeze(0).contiguous()
        ref = OPS.quant_qk(
            one,
            one,
            None,
            tensor_layout="HND",
            qk_quant_gran="per_thread",
            blk_q=BLKQ,
            warp_q=WARPQ,
            blk_k=BLKK,
            warp_k=WARPK,
        )
        base = blk_offset(cu_list, seq, BLKK) * 4
        assert torch.equal(got[3][:, base : base + 4], ref[3][0, :, :4])


# --------------------------------------------------------- quant_v_fp8_varlen
# The fp8 V path keeps a second coordinate system: the transposed value is
# zero-padded per sequence, because the sm89 V load has no bound predicate.

# fp8 tensors have no comparison support in opcheck's schema/autograd utils;
# the same three the dense fp8 opchecks run.
NO_SCHEMA_FP8 = ("test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic")

V_PAD = 64  # sm89 / sm120 padded V^T block (plan.cpp v_pad_of)


def pad_offset(cu, b, pad):
    """csrc/sageattn/varlen.h: sequence b's first padded token slot."""
    return blk_offset(cu, b, pad) * pad


def blk_total(total, batch, pad):
    """csrc/sageattn/varlen.h: the padded axis length of the whole batch."""
    return total // pad + batch


def rand_packed(lens, heads=2, head_dim=64, seed=17):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn((sum(lens), heads, head_dim), device=DEV, dtype=torch.float16, generator=g)


def one_sequence(packed, cu_list, b):
    """Sequence b as a dense batch of one, [1, heads, n, head_dim]."""
    return packed[cu_list[b] : cu_list[b + 1]].transpose(0, 1).unsqueeze(0).contiguous()


@requires_cuda
@pytest.mark.parametrize("permute", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("pad", [64, 128])
@pytest.mark.parametrize("lens", [[128], [37, 128, 1, 300], [64, 0, 64]], ids=str)
def test_transpose_pad_v_varlen_matches_per_sequence_dense(permute, head_dim, pad, lens):
    """The transpose is the only fp16-in/fp16-out stage of the fp8 V pipeline,
    so it is the one whose packed addressing can be checked on a pre-sm_89
    device. Each sequence's slab must equal the dense transpose of that
    sequence, zero fill included, and nothing past the written prefix may move
    (the buffer starts as NaN). pad=128 is the sm90/sm100 alignment: the kernel
    still writes only the 64-aligned prefix, so a slab's tail stays untouched
    and the fp8 allocation is what zeroes it."""
    cu_list = cu_of(lens)
    total, batch, heads = cu_list[-1], len(lens), 2
    packed = rand_packed(lens, heads, head_dim)
    cu = torch.tensor(cu_list, device=DEV, dtype=torch.int32)

    padded_total = blk_total(total, batch, pad) * pad
    out = torch.full((heads, head_dim, padded_total), float("nan"), device=DEV, dtype=torch.float16)
    OPS.transpose_pad_v(packed, out, "HND", permute, cu, max_seqlen=max(lens), pad_multiple=pad)

    for b, n in enumerate(lens):
        base = pad_offset(cu_list, b, pad)
        written = cdiv(n, 64) * 64
        # a sequence never writes past the blocks varlen.h gave it
        assert out[:, :, base + written : pad_offset(cu_list, b + 1, pad)].isnan().all(), (
            f"sequence {b} spilled"
        )
        if n == 0:
            continue

        ref = torch.empty((1, heads, head_dim, written), device=DEV, dtype=torch.float16)
        OPS.transpose_pad_v(one_sequence(packed, cu_list, b), ref, "HND", permute)
        slab = out[:, :, base : base + written]
        assert torch.equal(slab, ref[0]), f"sequence {b}"
        # mma_k16 permutes inside 16-token groups, so its zero fill starts there
        bound = cdiv(n, 16) * 16 if permute else n
        assert (slab[:, :, bound:] == 0).all(), f"sequence {b} tail is not zero"


@requires_fp8_cast
@pytest.mark.parametrize("v_layout", ["mma_k16", "linear"])
@pytest.mark.parametrize("smooth_v", [False, True])
def test_quant_v_fp8_varlen_matches_per_sequence_dense(v_layout, smooth_v):
    lens = [37, 128, 1, 300]
    cu_list = cu_of(lens)
    packed = rand_packed(lens, seed=23)
    cu = torch.tensor(cu_list, device=DEV, dtype=torch.int32)

    kw = dict(v_layout=v_layout, scale_max=448.0, smooth_v=smooth_v, pad_multiple=V_PAD)
    got = OPS.quant_v_fp8_varlen(packed, cu, max_seqlen_k=max(lens), **kw)

    for b, n in enumerate(lens):
        ref = OPS.quant_v_fp8(one_sequence(packed, cu_list, b), tensor_layout="HND", **kw)
        base = pad_offset(cu_list, b, V_PAD)
        written = cdiv(n, V_PAD) * V_PAD
        slab = got[0][:, :, base : base + written].view(torch.uint8)
        assert torch.equal(slab, ref[0][0].view(torch.uint8)), f"v_fp8 sequence {b}"
        assert torch.equal(got[1][b], ref[1][0]), f"v_scale sequence {b}"
        if smooth_v:
            assert torch.equal(got[2][b], ref[2][0]), f"v_mean sequence {b}"


@requires_fp8_cast
@pytest.mark.parametrize("smooth_v", [False, True])
@pytest.mark.parametrize("pad", [64, 128])
def test_quant_v_fp8_varlen_tail_is_zero(smooth_v, pad):
    """The white-box invariant the sm89 V load runs on: it reads whole CTA_K
    tiles straight out of the slab with no predicate, so every byte from a
    sequence's length to the end of its blocks has to be a zero fp8 - not
    stale, and above all not a NaN encoding (0 * NaN would poison PV). At
    pad=128 (sm90/sm100) part of that tail is never written at all, and the
    zeroed allocation is what has to cover it."""
    lens = [37, 128, 0, 1, 300]
    cu_list = cu_of(lens)
    packed = rand_packed(lens, seed=29)
    cu = torch.tensor(cu_list, device=DEV, dtype=torch.int32)

    v_fp8 = OPS.quant_v_fp8_varlen(
        packed, cu, max_seqlen_k=max(lens), smooth_v=smooth_v, pad_multiple=pad
    )[0].view(torch.uint8)

    for b, n in enumerate(lens):
        base, end = pad_offset(cu_list, b, pad), pad_offset(cu_list, b + 1, pad)
        assert (v_fp8[:, :, base + n : end] == 0).all(), f"sequence {b}"


@requires_fp8_cast
@pytest.mark.parametrize("v_layout", ["mma_k16", "linear"])
@pytest.mark.parametrize("smooth_v", [False, True])
def test_opcheck_quant_v_fp8_varlen(v_layout, smooth_v):
    lens = [37, 128, 300]
    cu = torch.tensor(cu_of(lens), device=DEV, dtype=torch.int32)
    opcheck(
        OPS.quant_v_fp8_varlen.default,
        (rand_packed(lens, seed=31), cu),
        dict(
            max_seqlen_k=max(lens),
            v_layout=v_layout,
            scale_max=448.0,
            smooth_v=smooth_v,
            pad_multiple=V_PAD,
        ),
        test_utils=NO_SCHEMA_FP8,
    )


# ------------------------------------------------------------- fwd_varlen

GEOM = dict(blk_q=BLKQ, warp_q=WARPQ, blk_k=BLKK, warp_k=WARPK)


def quant_varlen(q, k, cu_q, cu_k, gran, key_mean=None):
    return OPS.quant_qk_varlen(
        q,
        k,
        cu_q,
        cu_k,
        key_mean,
        max_seqlen_q=seg_max(cu_q),
        max_seqlen_k=seg_max(cu_k),
        qk_quant_gran=gran,
        **GEOM,
    )


def seg_max(cu):
    cu = cu.tolist() if torch.is_tensor(cu) else cu
    return max(cu[i + 1] - cu[i] for i in range(len(cu) - 1))


def fwd_varlen(q_int8, k_int8, v, q_scale, k_scale, cu_q, cu_k, causal, gran, lse=False):
    return OPS.fwd_varlen(
        q_int8,
        k_int8,
        v.to(torch.float16),
        q_scale,
        k_scale,
        cu_q,
        cu_k,
        None,
        None,
        max_seqlen_q=seg_max(cu_q),
        max_seqlen_k=seg_max(cu_k),
        qk_quant_gran=gran,
        pv_accum_dtype="fp32",
        v_layout="seq",
        is_causal=causal,
        sm_scale=q_int8.size(-1) ** -0.5,
        return_lse=lse,
        out_dtype=torch.float16,
    )


def fwd_dense(q_int8, k_int8, v, q_scale, k_scale, causal, gran, lse=False):
    return OPS.fwd(
        q_int8,
        k_int8,
        v.to(torch.float16),
        q_scale,
        k_scale,
        None,
        None,
        tensor_layout="HND",
        qk_quant_gran=gran,
        pv_accum_dtype="fp32",
        v_layout="seq",
        is_causal=causal,
        sm_scale=q_int8.size(-1) ** -0.5,
        return_lse=lse,
        out_dtype=torch.float16,
    )


FWD_SHAPES = [
    # (batch, qo heads, kv heads, seq len)
    (1, 2, 2, 128),
    (2, 4, 2, 256),  # GQA
    (3, 2, 2, 100),  # not a multiple of CTA_Q or CTA_K
    (2, 1, 1, 577),
]


@requires_backend("sm80")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("gran", ["per_warp", "per_thread"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("b,h_qo,h_kv,n", FWD_SHAPES, ids=lambda p: str(p))
def test_fwd_varlen_equals_dense(causal, gran, head_dim, b, h_qo, h_kv, n):
    """An equal-length batch has delta == 0, where bottom-right causal is the
    dense top-left one and the varlen tile structure degenerates to the dense
    one. Both outputs must therefore be bit-identical."""
    q, k, v = rand_qkv(b, h_qo, h_kv, n, head_dim, seed=(b * 977 + n) % 2**31)
    lens = [n] * b
    cu = torch.tensor(cu_of(lens), device=DEV, dtype=torch.int32)

    dq, dqs, dk, dks = OPS.quant_qk(q, k, None, tensor_layout="HND", qk_quant_gran=gran, **GEOM)
    out_d, lse_d = fwd_dense(dq, dk, v, dqs, dks, causal, gran, lse=True)

    pq, pqs, pk, pks = quant_varlen(pack(q, lens), pack(k, lens), cu, cu, gran)
    out_v, lse_v = fwd_varlen(pq, pk, pack(v, lens), pqs, pks, cu, cu, causal, gran, lse=True)

    assert torch.equal(out_v, pack(out_d, lens)), "out"
    assert torch.equal(unpack_lse(lse_v, lens), lse_d), "lse"


@requires_backend("sm80")
@pytest.mark.parametrize("gran", ["per_warp", "per_thread"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_fwd_varlen_ragged_equals_per_sequence_dense(gran, head_dim):
    """Ragged, non-causal: each sequence must come out exactly as a batch-of-one
    dense run of itself, including sequences whose q and kv lengths differ."""
    q_lens = [37, 128, 300, 1]
    k_lens = [200, 128, 64, 999]
    cu_q = torch.tensor(cu_of(q_lens), device=DEV, dtype=torch.int32)
    cu_k = torch.tensor(cu_of(k_lens), device=DEV, dtype=torch.int32)
    g = torch.Generator(device=DEV).manual_seed(17)
    heads = 2
    q = torch.randn((sum(q_lens), heads, head_dim), device=DEV, dtype=torch.float16, generator=g)
    k = torch.randn((sum(k_lens), heads, head_dim), device=DEV, dtype=torch.float16, generator=g)
    v = torch.randn((sum(k_lens), heads, head_dim), device=DEV, dtype=torch.float16, generator=g)

    pq, pqs, pk, pks = quant_varlen(q, k, cu_q, cu_k, gran)
    out_v, lse_v = fwd_varlen(pq, pk, v, pqs, pks, cu_q, cu_k, False, gran, lse=True)

    cq, ck = cu_of(q_lens), cu_of(k_lens)
    for b in range(len(q_lens)):
        one_q = q[cq[b] : cq[b + 1]].transpose(0, 1).unsqueeze(0).contiguous()
        one_k = k[ck[b] : ck[b + 1]].transpose(0, 1).unsqueeze(0).contiguous()
        one_v = v[ck[b] : ck[b + 1]].transpose(0, 1).unsqueeze(0).contiguous()
        dq, dqs, dk, dks = OPS.quant_qk(
            one_q, one_k, None, tensor_layout="HND", qk_quant_gran=gran, **GEOM
        )
        out_d, lse_d = fwd_dense(dq, dk, one_v, dqs, dks, False, gran, lse=True)
        got = out_v[cq[b] : cq[b + 1]].transpose(0, 1).unsqueeze(0)
        assert torch.equal(got, out_d), f"out sequence {b}"
        assert torch.equal(lse_v[:, cq[b] : cq[b + 1]].unsqueeze(0), lse_d), f"lse sequence {b}"


# --------------------------------------------- bottom-right causal (delta != 0)


def sdpa_varlen_ref(q, k, v, cu_q, cu_k, is_causal, sm_scale):
    """Per-sequence fp32 attention over packed inputs, with flash-attention's
    bottom-right causal alignment: row r of a sequence attends to keys up to
    r + (kv_len - qo_len). A row that admits no key gets a zero output and an
    lse of -inf, which is what the kernel is required to write."""
    out = torch.zeros_like(q, dtype=torch.float32)
    lse = torch.full((q.size(1), q.size(0)), float("-inf"), device=q.device, dtype=torch.float32)
    for b in range(len(cu_q) - 1):
        qs, qe, ks, ke = cu_q[b], cu_q[b + 1], cu_k[b], cu_k[b + 1]
        if qe == qs or ke == ks:
            continue
        qb = q[qs:qe].transpose(0, 1).float()  # [heads, n_q, head_dim]
        kb = k[ks:ke].transpose(0, 1).float()
        vb = v[ks:ke].transpose(0, 1).float()
        if qb.size(0) != kb.size(0):
            rep = qb.size(0) // kb.size(0)
            kb, vb = kb.repeat_interleave(rep, 0), vb.repeat_interleave(rep, 0)
        s = torch.matmul(qb, kb.transpose(-1, -2)) * sm_scale
        if is_causal:
            n_q, n_k = s.size(-2), s.size(-1)
            keep = torch.ones(n_q, n_k, dtype=torch.bool, device=s.device).tril(diagonal=n_k - n_q)
            s = s.masked_fill(~keep, float("-inf"))
        p = torch.nan_to_num(torch.softmax(s, dim=-1))  # all-masked rows -> 0
        out[qs:qe] = torch.matmul(p, vb).transpose(0, 1)
        lse[:, qs:qe] = torch.logsumexp(s, dim=-1)
    return out, lse


def sage_varlen_raw(q, k, v, cu_q, cu_k, causal, gran, lse=False):
    """quant_qk_varlen + fwd_varlen, i.e. the kernel pipeline without the
    Python-side smooth_k."""
    pq, pqs, pk, pks = quant_varlen(q, k, cu_q, cu_k, gran)
    return fwd_varlen(pq, pk, v, pqs, pks, cu_q, cu_k, causal, gran, lse=lse)


# (q_lens, k_lens) -- every one has at least one sequence whose delta is not a
# multiple of CTA_K, which is where the diagonal band crosses a third KV tile
CAUSAL_RAGGED = [
    ([100, 256, 37], [163, 256, 300]),  # delta = 63 / 0 / 263
    ([300, 1], [300, 1]),  # delta = 0, ragged lengths
    ([1000], [1]),  # delta = -999: most rows see no key
    ([128, 64], [1000, 65]),  # delta = 872 / 1
]


@requires_backend("sm80")
@pytest.mark.parametrize("gran", ["per_warp", "per_thread"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("q_lens,k_lens", CAUSAL_RAGGED, ids=lambda p: "x".join(map(str, p)))
def test_fwd_varlen_bottom_right_causal(gran, head_dim, q_lens, k_lens):
    """delta = kv_len - qo_len is what the masked-tile arithmetic is derived
    from; a tile left unmasked (or a trip count underflowed in unsigned) shows
    up here as a broken row, not as noise."""
    cu_q, cu_k = cu_of(q_lens), cu_of(k_lens)
    g = torch.Generator(device=DEV).manual_seed(23)
    heads = 2
    q = torch.randn((cu_q[-1], heads, head_dim), device=DEV, dtype=torch.float16, generator=g)
    k = torch.randn((cu_k[-1], heads, head_dim), device=DEV, dtype=torch.float16, generator=g)
    v = torch.randn((cu_k[-1], heads, head_dim), device=DEV, dtype=torch.float16, generator=g)
    tq = torch.tensor(cu_q, device=DEV, dtype=torch.int32)
    tk = torch.tensor(cu_k, device=DEV, dtype=torch.int32)

    out, lse = sage_varlen_raw(q, k, v, tq, tk, True, gran, lse=True)
    lse = lse / LOG2E  # the op returns log2; sageattn_varlen does this divide
    ref, ref_lse = sdpa_varlen_ref(q, k, v, cu_q, cu_k, True, head_dim**-0.5)

    cs, l1 = cos_sim(out, ref), rel_l1(out, ref)
    assert cs > COS_MIN and l1 < REL_L1_MAX, f"cos_sim={cs:.5f} rel_l1={l1:.5f}"

    # rows with no admissible key: exactly zero, lse exactly -inf
    dead = torch.isinf(ref_lse) & (ref_lse < 0)
    if dead.any():
        assert torch.equal(lse[dead], ref_lse[dead]), "an all-masked row must carry -inf"
        assert not out.transpose(0, 1)[dead].any(), "an all-masked row must be zero"
    live = ~dead
    torch.testing.assert_close(lse[live], ref_lse[live], rtol=2e-2, atol=5e-2)


@requires_backend("sm80")
@pytest.mark.parametrize("causal", [False, True])
def test_fwd_varlen_empty_kv_sequence(causal):
    """A sequence with no keys at all: O is zero and lse is -inf, and its
    neighbours are untouched."""
    q_lens, k_lens = [64, 128, 32], [64, 0, 32]
    cu_q, cu_k = cu_of(q_lens), cu_of(k_lens)
    g = torch.Generator(device=DEV).manual_seed(29)
    q = torch.randn((cu_q[-1], 2, 64), device=DEV, dtype=torch.float16, generator=g)
    k = torch.randn((max(cu_k[-1], 1), 2, 64), device=DEV, dtype=torch.float16, generator=g)
    v = torch.randn((max(cu_k[-1], 1), 2, 64), device=DEV, dtype=torch.float16, generator=g)
    tq = torch.tensor(cu_q, device=DEV, dtype=torch.int32)
    tk = torch.tensor(cu_k, device=DEV, dtype=torch.int32)

    out, lse = sage_varlen_raw(
        q, k[: cu_k[-1]], v[: cu_k[-1]], tq, tk, causal, "per_thread", lse=True
    )
    ref, _ = sdpa_varlen_ref(q, k, v, cu_q, cu_k, causal, 64**-0.5)

    empty = slice(cu_q[1], cu_q[2])
    assert not out[empty].any(), "the empty sequence must produce zeros"
    assert torch.isinf(lse[:, empty]).all() and (lse[:, empty] < 0).all()
    for other in (slice(0, cu_q[1]), slice(cu_q[2], cu_q[3])):
        cs = cos_sim(out[other], ref[other])
        assert cs > COS_MIN, f"neighbour cos_sim={cs:.5f}"


@requires_backend("sm80")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
def test_opcheck_fwd_varlen(causal, return_lse):
    q_lens, k_lens = [37, 128, 300], [200, 128, 64]
    cu_q = torch.tensor(cu_of(q_lens), device=DEV, dtype=torch.int32)
    cu_k = torch.tensor(cu_of(k_lens), device=DEV, dtype=torch.int32)
    g = torch.Generator(device=DEV).manual_seed(31)
    q = torch.randn((sum(q_lens), 2, 128), device=DEV, dtype=torch.float16, generator=g)
    k = torch.randn((sum(k_lens), 2, 128), device=DEV, dtype=torch.float16, generator=g)
    v = torch.randn((sum(k_lens), 2, 128), device=DEV, dtype=torch.float16, generator=g)
    pq, pqs, pk, pks = quant_varlen(q, k, cu_q, cu_k, "per_thread")

    opcheck(
        OPS.fwd_varlen.default,
        (pq, pk, v, pqs, pks, cu_q, cu_k, None, None),
        dict(
            max_seqlen_q=max(q_lens),
            max_seqlen_k=max(k_lens),
            qk_quant_gran="per_thread",
            pv_accum_dtype="fp32",
            v_layout="seq",
            is_causal=causal,
            sm_scale=128**-0.5,
            return_lse=return_lse,
            out_dtype=torch.float16,
        ),
    )


# ------------------------------------------------------- sageattn_varlen API
# From here down the tests go through the public entry point and name no tile
# geometry, accumulator or V layout, so they are gated on "some packed-layout
# backend is runnable" rather than on sm80: an sm89 / sm120 device runs this
# whole section against its own fp8 kernels.


def dense_of(packed, lens):
    """Packed -> [batch, heads, n, head_dim] for an equal-length batch."""
    n = lens[0]
    return packed.view(len(lens), n, packed.size(1), packed.size(2)).transpose(1, 2).contiguous()


@requires_varlen_backend
@pytest.mark.parametrize("smooth_k", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_varlen_vs_dense_end_to_end(smooth_k, causal, head_dim):
    """The equal-length batch again, this time through the public entry points.
    Without smooth_k the two are bit-identical; with it the per-sequence mean
    goes through index_add_ atomics, which are not run-to-run deterministic."""
    b, heads, n = 3, 4, 256
    lens = [n] * b
    cu = torch.tensor(cu_of(lens), device=DEV, dtype=torch.int32)
    g = torch.Generator(device=DEV).manual_seed(41)
    packed = [
        torch.randn((b * n, heads, head_dim), device=DEV, dtype=torch.float16, generator=g)
        for _ in range(3)
    ]
    q, k, v = packed

    with torch.no_grad():
        out_v, lse_v = sageattn_varlen(
            q, k, v, cu, cu, n, n, is_causal=causal, return_lse=True, smooth_k=smooth_k
        )
        out_d, lse_d = sageattn(
            *(dense_of(t, lens) for t in packed),
            is_causal=causal,
            return_lse=True,
            smooth_k=smooth_k,
        )

    got_dense = dense_of(out_v, lens)
    if smooth_k:
        torch.testing.assert_close(got_dense, out_d, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(unpack_lse(lse_v, lens), lse_d, rtol=1e-3, atol=1e-3)
    else:
        assert torch.equal(got_dense, out_d), "out"
        assert torch.equal(unpack_lse(lse_v, lens), lse_d), "lse"


VARLEN_SEGMENTATIONS = [
    ([256], [256]),  # single sequence
    ([100, 256, 37], [100, 256, 37]),  # ragged, delta = 0
    ([128, 64, 300], [163, 700, 300]),  # ragged, delta = 35 / 636 / 0
    ([64, 0, 128], [64, 0, 128]),  # an empty sequence in the middle
    ([300], [1]),  # kv much shorter than qo
]


@requires_varlen_backend
@pytest.mark.parametrize("smooth_k", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("q_lens,k_lens", VARLEN_SEGMENTATIONS, ids=lambda p: "x".join(map(str, p)))
def test_sageattn_varlen_matches_segmented_sdpa(smooth_k, causal, q_lens, k_lens):
    """Accuracy against an fp32 per-sequence reference, on the sageattn gates."""
    head_dim, heads = 64, 4
    cu_q, cu_k = cu_of(q_lens), cu_of(k_lens)
    g = torch.Generator(device=DEV).manual_seed(43)
    q = torch.randn((cu_q[-1], heads, head_dim), device=DEV, dtype=torch.float16, generator=g)
    k = torch.randn((cu_k[-1], heads, head_dim), device=DEV, dtype=torch.float16, generator=g)
    v = torch.randn((cu_k[-1], heads, head_dim), device=DEV, dtype=torch.float16, generator=g)
    tq = torch.tensor(cu_q, device=DEV, dtype=torch.int32)
    tk = torch.tensor(cu_k, device=DEV, dtype=torch.int32)

    with torch.no_grad():
        out, lse = sageattn_varlen(
            q,
            k,
            v,
            tq,
            tk,
            max(q_lens),
            max(k_lens),
            is_causal=causal,
            return_lse=True,
            smooth_k=smooth_k,
        )
    ref, ref_lse = sdpa_varlen_ref(q, k, v, cu_q, cu_k, causal, head_dim**-0.5)

    live = ~(torch.isinf(ref_lse) & (ref_lse < 0))
    if live.any():
        cs, l1 = cos_sim(out, ref), rel_l1(out, ref)
        assert cs > COS_MIN and l1 < REL_L1_MAX, f"cos_sim={cs:.5f} rel_l1={l1:.5f}"
        torch.testing.assert_close(lse[live], ref_lse[live], rtol=2e-2, atol=5e-2)
    assert torch.equal(lse[~live], ref_lse[~live]), "dead rows must carry -inf"


@requires_varlen_backend
def test_sageattn_varlen_gqa_and_head_dim_pad():
    """GQA plus a head_dim that is not 64 or 128: the pad is the dense one and
    the output is sliced back."""
    q_lens = k_lens = [96, 33]
    cu = torch.tensor(cu_of(q_lens), device=DEV, dtype=torch.int32)
    g = torch.Generator(device=DEV).manual_seed(47)
    q = torch.randn((sum(q_lens), 8, 96), device=DEV, dtype=torch.float16, generator=g)
    k = torch.randn((sum(k_lens), 2, 96), device=DEV, dtype=torch.float16, generator=g)
    v = torch.randn((sum(k_lens), 2, 96), device=DEV, dtype=torch.float16, generator=g)
    with torch.no_grad():
        out = sageattn_varlen(q, k, v, cu, cu, max(q_lens), max(k_lens), is_causal=True)
    assert out.shape == q.shape and out.dtype == q.dtype
    ref, _ = sdpa_varlen_ref(q, k, v, cu_of(q_lens), cu_of(k_lens), True, 96**-0.5)
    cs = cos_sim(out, ref)
    assert cs > COS_MIN, f"cos_sim={cs:.5f}"


@requires_varlen_backend
@pytest.mark.parametrize("causal", [False, True])
def test_sageattn_varlen_empty_kv_through_the_api(causal):
    """The empty-KV shortcut, reached through the public entry point so that it
    covers whichever backend the device resolves to (the kernel-level version
    above is pinned to sm80). On the fp8 backends this is also the check that
    the sequence owning no key still has a valid, zeroed V^T slab."""
    q_lens, k_lens = [64, 128, 32], [64, 0, 32]
    cu_q, cu_k = cu_of(q_lens), cu_of(k_lens)
    g = torch.Generator(device=DEV).manual_seed(53)
    q = torch.randn((cu_q[-1], 2, 64), device=DEV, dtype=torch.float16, generator=g)
    k = torch.randn((cu_k[-1], 2, 64), device=DEV, dtype=torch.float16, generator=g)
    v = torch.randn((cu_k[-1], 2, 64), device=DEV, dtype=torch.float16, generator=g)
    tq = torch.tensor(cu_q, device=DEV, dtype=torch.int32)
    tk = torch.tensor(cu_k, device=DEV, dtype=torch.int32)

    with torch.no_grad():
        out, lse = sageattn_varlen(
            q, k, v, tq, tk, max(q_lens), max(k_lens), is_causal=causal, return_lse=True
        )

    empty = slice(cu_q[1], cu_q[2])
    assert not out[empty].any(), "the empty sequence must produce zeros"
    assert torch.isinf(lse[:, empty]).all() and (lse[:, empty] < 0).all()
    assert out[: cu_q[1]].any() and out[cu_q[2] :].any(), "the neighbours must be untouched"


@requires_varlen_backend
def test_sageattn_varlen_rejects_dense_input():
    q = torch.randn((2, 4, 64, 64), device=DEV, dtype=torch.float16)
    cu = torch.tensor([0, 64, 128], device=DEV, dtype=torch.int32)
    with pytest.raises(AssertionError, match="packed"):
        sageattn_varlen(q, q, q, cu, cu, 64, 64)


# ---------------------------------------------------------- compile / cudagraph


@requires_varlen_backend
@pytest.mark.parametrize("smooth_k", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
def test_varlen_no_graph_breaks(smooth_k, return_lse):
    """Nothing in sageattn_varlen may read cu_seqlens on the host: a shape
    derived from tensor contents is a graph break (and a device sync)."""
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    lens = [128, 384]
    cu = torch.tensor(cu_of(lens), device=DEV, dtype=torch.int32)
    g = torch.Generator(device=DEV).manual_seed(53)
    q, k, v = (
        torch.randn((sum(lens), 2, 64), device=DEV, dtype=torch.float16, generator=g)
        for _ in range(3)
    )

    def fn(q, k, v, cu):
        return sageattn_varlen(
            q, k, v, cu, cu, 384, 384, is_causal=True, return_lse=return_lse, smooth_k=smooth_k
        )

    with torch.no_grad():
        explanation = torch._dynamo.explain(fn)(q, k, v, cu)
    assert explanation.graph_break_count == 0, explanation.break_reasons


@requires_varlen_backend
def test_varlen_compiled_matches_eager():
    torch._dynamo.reset()
    lens = [100, 256, 37]
    cu = torch.tensor(cu_of(lens), device=DEV, dtype=torch.int32)
    g = torch.Generator(device=DEV).manual_seed(59)
    q, k, v = (
        torch.randn((sum(lens), 4, 64), device=DEV, dtype=torch.float16, generator=g)
        for _ in range(3)
    )
    compiled = torch.compile(sageattn_varlen, fullgraph=True)
    with torch.no_grad():
        want = sageattn_varlen(q, k, v, cu, cu, 256, 256, is_causal=True, smooth_k=False)
        got = compiled(q, k, v, cu, cu, 256, 256, is_causal=True, smooth_k=False)
    assert torch.equal(got, want)


@requires_varlen_backend
def test_cudagraph_replay_with_a_new_segmentation():
    """The whole point of the closed-form block algebra: no tensor shape depends
    on the contents of cu_seqlens, so a replay may re-split the same tokens.
    max_seqlen is baked into grid.x at capture, so it is captured at the upper
    bound and every replayed sequence stays under it."""
    total, heads, head_dim, max_seqlen = 512, 2, 64, 512
    g = torch.Generator(device=DEV).manual_seed(61)
    q, k, v = (
        torch.randn((total, heads, head_dim), device=DEV, dtype=torch.float16, generator=g)
        for _ in range(3)
    )
    cu = torch.tensor(cu_of([256, 256]), device=DEV, dtype=torch.int32)

    def run():
        return sageattn_varlen(
            q, k, v, cu, cu, max_seqlen, max_seqlen, is_causal=True, smooth_k=False
        )

    with torch.no_grad():
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = run()

        for lens in ([256, 256], [100, 412], [512, 0], [1, 511]):
            cu.copy_(torch.tensor(cu_of(lens), device=DEV, dtype=torch.int32))
            graph.replay()
            torch.cuda.synchronize()
            want = run()
            assert torch.equal(static_out, want), f"replay with {lens}"


@requires_varlen_backend
def test_smooth_v_is_ignored_with_a_warning():
    """resolve() drops smooth_v for varlen; the entry point says so once and
    returns the same answer it would have without the request."""
    from sageattention.core import _warned_configs

    lens = [128, 64]
    cu = torch.tensor(cu_of(lens), device=DEV, dtype=torch.int32)
    g = torch.Generator(device=DEV).manual_seed(67)
    q, k, v = (
        torch.randn((sum(lens), 2, 64), device=DEV, dtype=torch.float16, generator=g)
        for _ in range(3)
    )
    _warned_configs.clear()
    with torch.no_grad():
        with pytest.warns(UserWarning, match="smooth_v"):
            got = sageattn_varlen(q, k, v, cu, cu, 128, 128, smooth_v=True, smooth_k=False)
        want = sageattn_varlen(q, k, v, cu, cu, 128, 128, smooth_v=False, smooth_k=False)
    assert torch.equal(got, want)


@requires_varlen_backend
def test_requires_grad_raises():
    lens = [64]
    cu = torch.tensor(cu_of(lens), device=DEV, dtype=torch.int32)
    q = torch.randn((64, 2, 64), device=DEV, dtype=torch.float16, requires_grad=True)
    with pytest.raises(NotImplementedError, match="backward"):
        sageattn_varlen(q, q, q, cu, cu, 64, 64)


@requires_varlen_backend
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_output_dtype_follows_input(dtype):
    """bfloat16 is a separate kernel instantiation (DTypeOut), so it gets its
    own equality check against the per-sequence dense path."""
    lens = [128, 64]
    cu_list = cu_of(lens)
    cu = torch.tensor(cu_list, device=DEV, dtype=torch.int32)
    g = torch.Generator(device=DEV).manual_seed(71)
    q, k, v = (
        torch.randn((sum(lens), 2, 64), device=DEV, dtype=dtype, generator=g) for _ in range(3)
    )
    with torch.no_grad():
        got = sageattn_varlen(q, k, v, cu, cu, 128, 128, is_causal=True, smooth_k=False)
    assert got.dtype == dtype

    outs = []
    for b in range(len(lens)):
        seg = slice(cu_list[b], cu_list[b + 1])
        one = (t[seg].transpose(0, 1).unsqueeze(0).contiguous() for t in (q, k, v))
        with torch.no_grad():
            outs.append(sageattn(*one, is_causal=True, smooth_k=False)[0].transpose(0, 1))
    assert torch.equal(got, torch.cat(outs, 0))
