"""varlen groundwork: the block algebra, the smooth_k segment reductions and
the varlen dimension of the plan table.

No varlen kernel exists yet. What is pinned here is everything the kernels will
be written against: the closed-form offsets in ``csrc/sageattn/varlen.h`` (both
of their provable properties, plus a direct comparison against the compiled C++
so the Python mirror below cannot drift), the smooth_k reductions in
``sageattention/varlen.py`` (the mean is the ``segment_mean_varlen`` op, the
lse correction is pure ATen), and the plan decisions varlen changes.
"""

import random
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from conftest import CC, requires_cuda, requires_varlen
from sageattention._plan import PLAN, Plan, get_plan
from sageattention.varlen import _segment_lse_correction, _segment_mean

REPO = Path(__file__).resolve().parent.parent
CSRC = REPO / "csrc"
PLAN_OP = torch.ops.sageattention.plan

# ---------------------------------------------------------------- block algebra
# Python mirror of csrc/sageattn/varlen.h. The C++ parity test below compiles
# the real header, so these three lines are checked, not assumed.


def blk_offset(cu_seqlens, batch_idx, cta_tokens):
    return cu_seqlens[batch_idx] // cta_tokens + batch_idx


def pad_offset(cu_seqlens, batch_idx, cta_tokens):
    return blk_offset(cu_seqlens, batch_idx, cta_tokens) * cta_tokens


def blk_total(total_tokens, batch_size, cta_tokens):
    return total_tokens // cta_tokens + batch_size


def _random_cu(rng, batch_size, max_len=600):
    """A cu_seqlens prefix sum; every fourth sequence may come out empty."""
    lens = [
        rng.randrange(0, max_len) if i % 4 == 0 else rng.randrange(1, max_len)
        for i in range(batch_size)
    ]
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    return cu


CTA_TOKENS = (64, 128)


@pytest.mark.parametrize("cta_tokens", CTA_TOKENS)
def test_every_sequence_owns_its_blocks(cta_tokens):
    """Property 1: consecutive block bases leave room for ceil(len / CTA)
    blocks, so one sequence's scale entries never overlap the next one's."""
    rng = random.Random(1234 + cta_tokens)
    for _ in range(200):
        batch_size = rng.randrange(1, 9)
        cu = _random_cu(rng, batch_size)
        for b in range(batch_size):
            room = blk_offset(cu, b + 1, cta_tokens) - blk_offset(cu, b, cta_tokens)
            need = -(-(cu[b + 1] - cu[b]) // cta_tokens)
            assert room >= need, (cu, b, cta_tokens, room, need)
            assert room <= need + 1, "at most one padding block per sequence"


@pytest.mark.parametrize("cta_tokens", CTA_TOKENS)
def test_block_total_is_a_static_shape(cta_tokens):
    """Property 2: the block count depends on (total_tokens, batch_size) only.
    That is what lets a cudagraph replay rewrite cu_seqlens in place, and what
    keeps the fake kernels free of unbacked SymInts."""
    rng = random.Random(99 + cta_tokens)
    for _ in range(200):
        batch_size = rng.randrange(1, 9)
        cu = _random_cu(rng, batch_size)
        total = cu[-1]
        assert blk_offset(cu, batch_size, cta_tokens) == blk_total(total, batch_size, cta_tokens)

        # a different split of the same (total, batch_size) gives the same total
        other = sorted(rng.randrange(0, total + 1) for _ in range(batch_size - 1))
        other = [0] + other + [total]
        assert blk_offset(other, batch_size, cta_tokens) == blk_total(total, batch_size, cta_tokens)


@pytest.mark.parametrize("cta_tokens", CTA_TOKENS)
def test_padded_slabs_hold_their_sequence(cta_tokens):
    """The fp8 V^T buffer gives sequence b the token range
    [pad_offset(b), pad_offset(b + 1)). The permuted V loads have no predicate,
    so that range has to start on a block boundary and be long enough for the
    whole sequence; the leftover tokens are the zero fill."""
    rng = random.Random(5 + cta_tokens)
    for _ in range(200):
        batch_size = rng.randrange(1, 9)
        cu = _random_cu(rng, batch_size)
        for b in range(batch_size):
            start = pad_offset(cu, b, cta_tokens)
            assert start % cta_tokens == 0
            assert pad_offset(cu, b + 1, cta_tokens) - start >= cu[b + 1] - cu[b]


# ------------------------------------------------------------- C++ header parity

NVCC = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
CXX = shutil.which("g++")

_PARITY_MAIN = """
#include <cstdio>
#include "sageattn/varlen.h"

int main()
{
    const int32_t cu[7] = {0, 100, 100, 333, 334, 1000, 1001};
    const int32_t ctas[2] = {64, 128};
    for (int32_t b = 0; b < 6; ++b) {
        for (int i = 0; i < 2; ++i) {
            const int32_t cta = ctas[i];
            printf("%d %d %d %d %d\\n", b, cta, sage::seq_len(cu, b),
                   sage::blk_offset(cu, b, cta), sage::pad_offset(cu, b, cta));
        }
    }
    printf("T %lld %lld\\n", (long long)sage::blk_total(1001, 6, 64),
           (long long)sage::blk_total(1001, 6, 128));
    return 0;
}
"""


@pytest.mark.skipif(CXX is None, reason="needs a host C++ compiler")
def test_cpp_offsets_match_the_python_mirror(tmp_path):
    """varlen.h is host-compilable and agrees with the mirror above, value for
    value. If either side is edited alone this fails."""
    src = tmp_path / "parity.cpp"
    src.write_text(_PARITY_MAIN)
    exe = tmp_path / "parity"
    subprocess.run([CXX, "-std=c++17", "-I", str(CSRC), str(src), "-o", str(exe)], check=True)
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout

    cu = [0, 100, 100, 333, 334, 1000, 1001]
    for line in out.splitlines():
        f = line.split()
        if f[0] == "T":
            assert [int(f[1]), int(f[2])] == [blk_total(1001, 6, 64), blk_total(1001, 6, 128)]
            continue
        b, cta, seq_len, blk, pad = map(int, f)
        assert seq_len == cu[b + 1] - cu[b], line
        assert blk == blk_offset(cu, b, cta), line
        assert pad == pad_offset(cu, b, cta), line


_SEQLEN_INFO_TU = """
#include "sageattn/seqlen_info.cuh"

__global__ void probe(const int32_t* cu_q, const int32_t* cu_k, int32_t* out)
{
    sage::SeqlenInfo<true, 128, 64> varlen(cu_q, cu_k, static_cast<int32_t>(blockIdx.z));
    sage::SeqlenInfo<false, 128, 64> dense(1024, 1024);
    out[0] = varlen.offset_q + varlen.blk_k_base + varlen.delta + dense.seqlen_q;
}
"""


@pytest.mark.skipif(not Path(NVCC).exists(), reason="needs nvcc")
def test_seqlen_info_compiles_for_device(tmp_path):
    """SeqlenInfo has no call site until the varlen kernels land, so nothing
    would otherwise instantiate it (or fire its static_asserts) during P0."""
    src = tmp_path / "seqlen_info_probe.cu"
    src.write_text(_SEQLEN_INFO_TU)
    subprocess.run(
        [NVCC, "-std=c++17", "-c", "-I", str(CSRC), str(src), "-o", str(tmp_path / "probe.o")],
        check=True,
    )


# ------------------------------------------------------------ segment reductions

DEV = "cuda" if CC is not None else "cpu"


def _packed(lens, heads=4, head_dim=64, dtype=torch.float16, seed=0):
    total = sum(lens)
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn((total, heads, head_dim), device=DEV, dtype=dtype, generator=g)
    cu = torch.tensor([0] + list(lens), device=DEV, dtype=torch.int32).cumsum(0).to(torch.int32)
    return x, cu, total


SEGMENTATIONS = [
    [128],
    [1, 255],
    [100, 0, 233],  # an empty sequence in the middle
    [0, 64],  # an empty sequence first
    [37, 128, 1, 300],
    [1500, 0, 700],  # max_seqlen > one op-internal chunk: the two-kernel path
    [2500],  # three partial-sum chunks folded by the second kernel
]


def _per_sequence_means(x, lens):
    means, start = [], 0
    for n in lens:
        means.append(
            x[start : start + n].float().mean(0)
            if n
            else torch.zeros((x.size(1), x.size(2)), device=x.device, dtype=torch.float32)
        )
        start += n
    return means


@requires_cuda
@requires_varlen
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("lens", SEGMENTATIONS, ids=lambda p: "x".join(map(str, p)))
def test_segment_mean_matches_per_sequence_mean(lens, head_dim, dtype):
    x, cu, _ = _packed(lens, head_dim=head_dim, dtype=dtype)
    got = _segment_mean(x, cu, max(lens))
    assert got.shape == (len(lens), x.size(1), x.size(2))
    assert got.dtype == x.dtype

    # one output-dtype ulp around 1.0: the reference is float32, so casting the
    # mean once costs up to eps/2 relative on its own
    tol = 1e-3 if dtype == torch.float16 else 8e-3
    for b, want in enumerate(_per_sequence_means(x, lens)):
        torch.testing.assert_close(got[b].float(), want, rtol=tol, atol=tol)


@requires_cuda
@requires_varlen
def test_segment_mean_takes_max_seqlen_as_an_upper_bound():
    """max_seqlen only opens the grid, so a caller may pass its model-level
    upper bound (the cudagraph capture pattern) instead of the exact maximum."""
    lens = [37, 128, 1, 300]
    x, cu, _ = _packed(lens)
    exact = _segment_mean(x, cu, max(lens))
    padded = _segment_mean(x, cu, 4096)
    assert torch.equal(exact, padded)


@requires_cuda
@requires_varlen
def test_segment_mean_is_deterministic():
    """The op reduces in a fixed order; the atomic index_add_ composite it
    replaced was run-to-run non-deterministic and could only be allclose'd."""
    x, cu, _ = _packed([1500, 0, 700])
    assert torch.equal(_segment_mean(x, cu, 1500), _segment_mean(x, cu, 1500))


@requires_cuda
@requires_varlen
def test_segment_mean_takes_a_strided_input():
    """k sliced out of a fused qkv projection is the common non-contiguous
    caller: only the last dim is contiguous, which is all the op requires."""
    lens = [100, 0, 233]
    qkv, cu, _ = _packed(lens, heads=3 * 4)
    k = qkv.view(qkv.size(0), 3, 4, qkv.size(2))[:, 1]
    assert not k.is_contiguous()
    got = _segment_mean(k, cu, max(lens))
    for b, want in enumerate(_per_sequence_means(k.contiguous(), lens)):
        torch.testing.assert_close(got[b].float(), want, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("q_per_kv_head", [1, 2])
def test_segment_lse_correction_matches_per_sequence_matmul(q_per_kv_head):
    lens = [37, 0, 128, 300]
    kv_heads = 2
    x, cu, total = _packed(lens, heads=kv_heads * q_per_kv_head, seed=7)
    key_mean = torch.randn(
        (len(lens), kv_heads, x.size(2)),
        device=DEV,
        dtype=x.dtype,
        generator=torch.Generator(device=DEV).manual_seed(3),
    )
    got = _segment_lse_correction(x, key_mean, cu, len(lens), total)
    assert got.shape == (x.size(1), total)

    km = key_mean.repeat_interleave(q_per_kv_head, dim=1) if q_per_kv_head > 1 else key_mean
    start = 0
    for b, n in enumerate(lens):
        want = (x[start : start + n] * km[b]).sum(-1, dtype=torch.float32).transpose(0, 1)
        torch.testing.assert_close(got[:, start : start + n], want)
        start += n


@requires_cuda
@requires_varlen
def test_segment_mean_has_no_graph_break():
    """The reduction sits inside the traced region of sageattn_varlen, so a
    data-dependent shape here would cost the whole graph. The op's fake reads
    batch_size off cu_seqlens' *shape* only, which is what prevents it."""
    torch._dynamo.reset()
    x, cu, _ = _packed([37, 128, 1, 300])

    def fn(x, cu):
        return _segment_mean(x, cu, 300)

    with torch.no_grad():
        explanation = torch._dynamo.explain(fn)(x, cu)
    assert explanation.graph_break_count == 0, explanation.break_reasons


# ------------------------------------------------------------------- plan table


def test_plan_table_carries_the_varlen_dimension():
    assert any(key[-1] for key in PLAN), "no varlen entry was primed"
    assert any(not key[-1] for key in PLAN), "no dense entry was primed"


def test_varlen_downgrades_smooth_v():
    """smooth_v has no varlen kernel; the request is dropped the same way an
    unsupported (backend, pv_accum_dtype) pair drops it."""
    dense = Plan(*PLAN_OP(8, 0, 128, "sm80", None, "fp16", True, False))
    assert dense.smooth_v is True and dense.smooth_v_ignored is False

    varlen = Plan(*PLAN_OP(8, 0, 128, "sm80", None, "fp16", True, True))
    assert varlen.smooth_v is False and varlen.smooth_v_ignored is True
    assert varlen.need_value_mean is False


def test_varlen_rejects_sm100():
    p = Plan(*PLAN_OP(10, 0, 128, "sm100", None, None, None, True))
    assert "varlen" in p.error or "not in this build" in p.error


def test_varlen_leaves_the_dense_decision_alone():
    """Apart from smooth_v and sm100, varlen must resolve to exactly the dense
    decision, error strings included: the varlen TU compiles the same kernel
    body from the same *_impl.cuh."""
    for (cc, head_dim, gran, pv, sv, varlen), plan in list(PLAN.items()):
        if varlen or sv:
            continue
        other = PLAN.get((cc, head_dim, gran, pv, sv, True))
        if other is None:  # a key another test resolved on demand
            continue
        if "sm100" in (plan.backend, other.backend):
            continue  # varlen rejects sm100; test_varlen_rejects_sm100 covers it
        assert tuple(plan) == tuple(other), (cc, head_dim, gran, pv, sv)


def test_v_pad_multiple_equals_blk_k_on_the_fp8_path():
    """The fp8 V^T slabs are padded to v_pad_multiple and the attention kernel
    tiles KV by blk_k; varlen starts each sequence's slab at a v_pad_multiple
    boundary, so the two numbers must agree. fwd_cuda.cu asserts the same."""
    bad = [k for k, p in PLAN.items() if not p.error and p.pv_fp8 and p.v_pad_multiple != p.blk_k]
    assert not bad, bad[:5]


def test_get_plan_varlen_keyword():
    dense = get_plan((8, 0), 128, None, None, None)
    varlen = get_plan((8, 0), 128, None, None, None, varlen=True)
    assert tuple(dense) == tuple(varlen)


def test_get_plan_uncached_varlen_capability():
    key = ((8, 5), 128, None, None, None, True)
    PLAN.pop(key, None)
    p = get_plan(*key)
    assert p.backend == "sm80" and not p.error
    assert PLAN.get(key) == p
