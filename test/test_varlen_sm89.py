"""The sm89 varlen kernel: the packed layout over the fp8 CUDA-core path.

test_varlen.py holds the sm80 half and the shared contract, test_varlen_sm90.py
the Hopper one; this file is the same set of gates on the Ada instantiation.
The packed TU only exists at sm89's default pv_accum_dtype ("fp32+fp16",
plan.cpp), whose fp16 PV accumulator is kept in range by quantizing V with
scale_max = 2.25 instead of the full fp8 448. The dense kernel sharing the
body is sm89::qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf, so on an
equal-length batch (delta == 0, the varlen tile structure is the dense one)
the two must agree bit for bit.

The ``sageattn_varlen`` end-to-end tests live in test_varlen.py, which runs
them on whichever packed backend the device resolves to.
"""

import pytest
import torch
from torch.library import opcheck

from conftest import cos_sim, rel_l1, requires_backend, requires_varlen

DEV = "cuda"
OPS = torch.ops.sageattention
pytestmark = [requires_varlen, requires_backend("sm89")]

# sm89 quantization tile geometry (plan.cpp fill_tiles)
BLKQ, WARPQ, BLKK, WARPK = 128, 32, 64, 64
GEOM = dict(blk_q=BLKQ, warp_q=WARPQ, blk_k=BLKK, warp_k=WARPK)
V_PAD = 64  # plan.cpp v_pad_of(sm89); the launcher pins it equal to CTA_K
SCALE_MAX = 2.25  # plan.cpp v_scale_max for fp32+fp16: headroom for the fp16 accumulator
PV = "fp32+fp16"  # the only pv_accum_dtype the sm89 packed TU instantiates

# opcheck's "test_schema" util clones/compares inputs with ops torch does not
# implement for Float8_e4m3fn on CUDA; test_ops.py skips it the same way.
NO_SCHEMA_FP8 = ("test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic")

# the sageattn accuracy gates (test_accuracy.py)
COS_MIN = 0.99
REL_L1_MAX = 0.06


def cu_of(lens):
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    return cu


def seg_max(cu):
    cu = cu.tolist() if torch.is_tensor(cu) else cu
    return max(cu[i + 1] - cu[i] for i in range(len(cu) - 1))


def pack(dense, lens):
    """[B, H, N, D] -> [sum(lens), H, D], sequence b keeping its first tokens."""
    dense = dense.transpose(1, 2)  # [B, N, H, D]
    return torch.cat([dense[b, : lens[b]] for b in range(len(lens))], dim=0).contiguous()


def unpack_lse(lse, lens):
    """[heads, total] -> [batch, heads, n] for an equal-length batch."""
    h, _ = lse.shape
    return lse.view(h, len(lens), lens[0]).transpose(0, 1).contiguous()


def rand_qkv(b, h_qo, h_kv, n, head_dim, seed=0, dtype=torch.float16):
    g = torch.Generator(device=DEV).manual_seed(seed)

    def one(h):
        return torch.randn((b, h, n, head_dim), device=DEV, dtype=dtype, generator=g)

    return one(h_qo), one(h_kv), one(h_kv)


def quant_v_dense(v_hnd):
    """[B, H, N, D] -> ([B, H, D, round_up(N, 64)] fp8, [B, H, D] scale)."""
    v_fp8, v_scale, _ = OPS.quant_v_fp8(
        v_hnd,
        tensor_layout="HND",
        v_layout="mma_k16",
        scale_max=SCALE_MAX,
        smooth_v=False,
        pad_multiple=V_PAD,
    )
    return v_fp8, v_scale


def pack_v_fp8(v_packed, cu_k, max_seqlen_k):
    """The varlen V^T slab and its scales, as the sm89 kernel wants them:
    ``[kv_heads, head_dim, blk_total(total_k, batch, 64) * 64]`` fp8 and
    ``[batch, kv_heads, head_dim]`` float32. pad_multiple is CTA_K, which is
    what the launcher pins the padded extent against."""
    v_fp8, v_scale, _ = OPS.quant_v_fp8_varlen(
        v_packed,
        cu_k,
        max_seqlen_k=max_seqlen_k,
        v_layout="mma_k16",
        scale_max=SCALE_MAX,
        smooth_v=False,
        pad_multiple=V_PAD,
    )
    return v_fp8, v_scale


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


def fwd_varlen(q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale, cu_q, cu_k, causal, gran,
               lse=False):
    return OPS.fwd_varlen(
        q_int8,
        k_int8,
        v_fp8,
        q_scale,
        k_scale,
        cu_q,
        cu_k,
        v_scale,
        None,
        max_seqlen_q=seg_max(cu_q),
        max_seqlen_k=seg_max(cu_k),
        qk_quant_gran=gran,
        pv_accum_dtype=PV,
        v_layout="mma_k16",
        is_causal=causal,
        sm_scale=q_int8.size(-1) ** -0.5,
        return_lse=lse,
        out_dtype=torch.float16,
    )


def fwd_dense(q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale, causal, gran, lse=False):
    return OPS.fwd(
        q_int8,
        k_int8,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        None,
        tensor_layout="HND",
        qk_quant_gran=gran,
        pv_accum_dtype=PV,
        v_layout="mma_k16",
        is_causal=causal,
        sm_scale=q_int8.size(-1) ** -0.5,
        return_lse=lse,
        out_dtype=torch.float16,
    )


FWD_SHAPES = [
    # (batch, qo heads, kv heads, seq len)
    (1, 2, 2, 128),
    (2, 4, 2, 256),  # GQA
    (3, 2, 2, 100),  # shorter than CTA_Q, not a multiple of CTA_K
    (2, 1, 1, 577),  # a multiple of neither CTA_Q nor CTA_K
]


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
    dv, dvs = quant_v_dense(v)
    out_d, lse_d = fwd_dense(dq, dk, dv, dqs, dks, dvs, causal, gran, lse=True)

    pq, pqs, pk, pks = quant_varlen(pack(q, lens), pack(k, lens), cu, cu, gran)
    pv_fp8, pvs = pack_v_fp8(pack(v, lens), cu, n)
    out_v, lse_v = fwd_varlen(pq, pk, pv_fp8, pqs, pks, pvs, cu, cu, causal, gran, lse=True)

    assert torch.equal(out_v, pack(out_d, lens)), "out"
    assert torch.equal(unpack_lse(lse_v, lens), lse_d), "lse"


@pytest.mark.parametrize("gran", ["per_warp", "per_thread"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_fwd_varlen_ragged_equals_per_sequence_dense(gran, head_dim):
    """Ragged, non-causal: each sequence must come out exactly as a batch-of-one
    dense run of itself, including a single-token sequence and sequences whose
    q and kv lengths differ. This is the test that catches a K tile reading
    across a sequence boundary."""
    q_lens = [37, 128, 300, 1]
    k_lens = [200, 128, 64, 999]
    cq, ck = cu_of(q_lens), cu_of(k_lens)
    cu_q = torch.tensor(cq, device=DEV, dtype=torch.int32)
    cu_k = torch.tensor(ck, device=DEV, dtype=torch.int32)
    g = torch.Generator(device=DEV).manual_seed(17)
    heads = 2
    q = torch.randn((sum(q_lens), heads, head_dim), device=DEV, dtype=torch.float16, generator=g)
    k = torch.randn((sum(k_lens), heads, head_dim), device=DEV, dtype=torch.float16, generator=g)
    v = torch.randn((sum(k_lens), heads, head_dim), device=DEV, dtype=torch.float16, generator=g)

    pq, pqs, pk, pks = quant_varlen(q, k, cu_q, cu_k, gran)
    pv_fp8, pvs = pack_v_fp8(v, cu_k, max(k_lens))
    out_v, lse_v = fwd_varlen(pq, pk, pv_fp8, pqs, pks, pvs, cu_q, cu_k, False, gran, lse=True)

    for b in range(len(q_lens)):
        one_q = q[cq[b] : cq[b + 1]].transpose(0, 1).unsqueeze(0).contiguous()
        one_k = k[ck[b] : ck[b + 1]].transpose(0, 1).unsqueeze(0).contiguous()
        one_v = v[ck[b] : ck[b + 1]].transpose(0, 1).unsqueeze(0).contiguous()
        dq, dqs, dk, dks = OPS.quant_qk(
            one_q, one_k, None, tensor_layout="HND", qk_quant_gran=gran, **GEOM
        )
        dv, dvs = quant_v_dense(one_v)
        out_d, lse_d = fwd_dense(dq, dk, dv, dqs, dks, dvs, False, gran, lse=True)
        got = out_v[cq[b] : cq[b + 1]].transpose(0, 1).unsqueeze(0)
        assert torch.equal(got, out_d), f"out sequence {b}"
        assert torch.equal(lse_v[:, cq[b] : cq[b + 1]].unsqueeze(0), lse_d), f"lse sequence {b}"


# --------------------------------------------- bottom-right causal (delta != 0)


def sdpa_varlen_ref(q, k, v, cu_q, cu_k, is_causal, sm_scale):
    """Per-sequence fp32 attention over packed inputs, with flash-attention's
    bottom-right causal alignment. A row that admits no key gets a zero output
    and an lse of -inf, which is what the kernel is required to write."""
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
    """quant_qk_varlen + quant_v_fp8_varlen + fwd_varlen: the kernel pipeline
    without the Python-side smooth_k."""
    pq, pqs, pk, pks = quant_varlen(q, k, cu_q, cu_k, gran)
    pv_fp8, pvs = pack_v_fp8(v, cu_k, seg_max(cu_k))
    return fwd_varlen(pq, pk, pv_fp8, pqs, pks, pvs, cu_q, cu_k, causal, gran, lse=lse)


# (q_lens, k_lens) -- each has a sequence whose delta is not a multiple of
# CTA_K, which is where the diagonal band crosses a second KV tile
CAUSAL_RAGGED = [
    ([128, 64], [128, 64]),  # delta == 0, the dense structure
    ([100, 200], [163, 200]),  # delta = 63: not a CTA_K multiple
    ([300], [428]),  # delta = 128: exactly two KV tiles
    ([1000], [1]),  # kv far shorter than qo, most rows admit no key
    ([64, 130], [300, 131]),  # mixed, both off every alignment
]


@pytest.mark.parametrize("gran", ["per_warp", "per_thread"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("q_lens,k_lens", CAUSAL_RAGGED, ids=lambda p: "x".join(map(str, p)))
def test_fwd_varlen_bottom_right_causal(gran, head_dim, q_lens, k_lens):
    cu_q = torch.tensor(cu_of(q_lens), device=DEV, dtype=torch.int32)
    cu_k = torch.tensor(cu_of(k_lens), device=DEV, dtype=torch.int32)
    g = torch.Generator(device=DEV).manual_seed(7)
    heads = 2
    q = torch.randn((sum(q_lens), heads, head_dim), device=DEV, dtype=torch.float16, generator=g)
    k = torch.randn((sum(k_lens), heads, head_dim), device=DEV, dtype=torch.float16, generator=g)
    v = torch.randn((sum(k_lens), heads, head_dim), device=DEV, dtype=torch.float16, generator=g)

    out, lse = sage_varlen_raw(q, k, v, cu_q, cu_k, True, gran, lse=True)
    ref, ref_lse = sdpa_varlen_ref(
        q, k, v, cu_of(q_lens), cu_of(k_lens), True, head_dim**-0.5
    )

    live = ref_lse.isfinite()
    assert torch.equal(lse.isfinite(), live), "the -inf rows must be exactly the empty ones"
    assert cos_sim(out[live.any(0)], ref[live.any(0)]) > COS_MIN
    assert rel_l1(out[live.any(0)], ref[live.any(0)]) < REL_L1_MAX
    dead = ~live.any(0)
    if dead.any():
        assert torch.equal(out[dead], torch.zeros_like(out[dead])), "empty rows must be zero"


@pytest.mark.parametrize("causal", [False, True])
def test_fwd_varlen_empty_kv_sequence(causal):
    """A sequence with no keys at all: zero output, -inf lse, and the other
    sequences unaffected."""
    q_lens = [64, 96, 32]
    k_lens = [128, 0, 64]
    cu_q = torch.tensor(cu_of(q_lens), device=DEV, dtype=torch.int32)
    cu_k = torch.tensor(cu_of(k_lens), device=DEV, dtype=torch.int32)
    g = torch.Generator(device=DEV).manual_seed(3)
    heads, head_dim = 2, 128
    q = torch.randn((sum(q_lens), heads, head_dim), device=DEV, dtype=torch.float16, generator=g)
    k = torch.randn((sum(k_lens), heads, head_dim), device=DEV, dtype=torch.float16, generator=g)
    v = torch.randn((sum(k_lens), heads, head_dim), device=DEV, dtype=torch.float16, generator=g)

    out, lse = sage_varlen_raw(q, k, v, cu_q, cu_k, causal, "per_thread", lse=True)
    cq = cu_of(q_lens)
    empty = slice(cq[1], cq[2])
    assert torch.equal(out[empty], torch.zeros_like(out[empty]))
    assert torch.equal(lse[:, empty], torch.full_like(lse[:, empty], float("-inf")))
    assert lse[:, cq[0] : cq[1]].isfinite().all()
    assert lse[:, cq[2] : cq[3]].isfinite().all()


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
def test_opcheck_fwd_varlen(causal, return_lse):
    q_lens, k_lens = [64, 100], [128, 100]
    cu_q = torch.tensor(cu_of(q_lens), device=DEV, dtype=torch.int32)
    cu_k = torch.tensor(cu_of(k_lens), device=DEV, dtype=torch.int32)
    g = torch.Generator(device=DEV).manual_seed(5)
    heads, head_dim = 2, 64
    q = torch.randn((sum(q_lens), heads, head_dim), device=DEV, dtype=torch.float16, generator=g)
    k = torch.randn((sum(k_lens), heads, head_dim), device=DEV, dtype=torch.float16, generator=g)
    v = torch.randn((sum(k_lens), heads, head_dim), device=DEV, dtype=torch.float16, generator=g)

    pq, pqs, pk, pks = quant_varlen(q, k, cu_q, cu_k, "per_thread")
    pv_fp8, pvs = pack_v_fp8(v, cu_k, max(k_lens))
    opcheck(
        OPS.fwd_varlen.default,
        (pq, pk, pv_fp8, pqs, pks, cu_q, cu_k, pvs, None),
        {
            "max_seqlen_q": seg_max(cu_q),
            "max_seqlen_k": seg_max(cu_k),
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": PV,
            "v_layout": "mma_k16",
            "is_causal": causal,
            "sm_scale": head_dim**-0.5,
            "return_lse": return_lse,
            "out_dtype": torch.float16,
        },
        test_utils=NO_SCHEMA_FP8,
    )
