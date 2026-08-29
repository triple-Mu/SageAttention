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

from conftest import requires_backend

DEV = "cuda"
OPS = torch.ops.sageattention

# sm80 quantization tile geometry (plan.cpp fill_tiles)
BLKQ, WARPQ, BLKK, WARPK = 128, 32, 64, 64


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
