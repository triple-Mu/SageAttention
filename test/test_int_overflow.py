"""Guard tests for the layered int-width contract.

Tier 1: strided views with as few tokens as possible push a single stride
past the bound while keeping the backing storage small; the host-side
TORCH_CHECKs must fire before any kernel launch. The end-to-end >4GiB
correctness cases (batch isolation at seq~700K) need a large-memory GPU and
live on the batch-C hardware checklist.
"""

import pytest
import torch

import sageattention  # noqa: F401  (registers torch.ops.sageattention.*)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")


def _strided(shape, strides, dtype=torch.float16):
    """A real (small) allocation viewed with an oversized stride."""
    numel = 1 + sum((s - 1) * st for s, st in zip(shape, strides))
    base = torch.zeros(numel, dtype=dtype, device="cuda")
    return torch.as_strided(base, shape, strides)


def test_quant_seq_stride_over_2p24_rejected():
    # stride_seq_input is loop-carried (uint32 in-kernel): must be < 2^24.
    # NHD [b, n, h, d] with n=2 keeps the storage at ~64 MB.
    big = 1 << 25
    x = _strided((1, 2, 4, 64), (2 * big, big, 64, 1))
    out = torch.empty((1, 2, 4, 64), dtype=torch.int8, device="cuda")
    scale = torch.empty((1, 4, 1), dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="2\\^24"):
        torch.ops.sageattention.quant_per_warp_int8(x, out, scale, 128, 32, "NHD")


def test_attn_v_seq_stride_over_2p24_rejected():
    # sm80 requires fully contiguous q/k (their seq stride can never overflow)
    # but v is only lastdim-contiguous — its stride_seq_v carries the bound.
    q = torch.randint(-1, 2, (1, 1, 64, 64), dtype=torch.int8, device="cuda")
    k = q.clone()
    big = 1 << 25
    v = _strided((1, 1, 64, 64), (64 * big, 64 * big, big, 1))
    o = torch.empty((1, 1, 64, 64), dtype=torch.float16, device="cuda")
    qs = torch.rand(1, 1, 16, dtype=torch.float32, device="cuda")
    ks = torch.rand(1, 1, 4, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="2\\^24"):
        torch.ops.sageattention.qattn_sm80_qk_int8_sv_f16_accum_f32_attn(
            q, k, v, o, qs, ks, "HND", False, "per_thread", 0.125, False
        )


def test_batch_stride_over_int32_accepted():
    # batch strides are 64-bit now: a stride_bz just past 2^31 must pass the
    # layout checks (it used to wrap through int narrowing). n=2 tokens keeps
    # the actual kernel work trivial; values are zeros so the result is exact.
    # mem_get_info is queried here, not in a decorator: at collection time it
    # would initialize CUDA in every pytest process.
    if torch.cuda.mem_get_info()[0] < 7 * 1024**3:
        pytest.skip("needs ~5 GB free VRAM")
    big_bz = (1 << 31) + 64  # elements, > int32 max
    x = _strided((2, 2, 4, 64), (big_bz, 64 * 4, 64, 1))  # NHD, ~4.3 GB fp16
    out = torch.empty((2, 2, 4, 64), dtype=torch.int8, device="cuda")
    scale = torch.empty((2, 4, 4), dtype=torch.float32, device="cuda")  # ceil(2/128)*(128/32)
    torch.ops.sageattention.quant_per_warp_int8(x, out, scale, 128, 32, "NHD")
    torch.cuda.synchronize()
    assert out.abs().max().item() == 0  # zeros in -> zeros out


def test_large_seq_batch_isolation():
    # The old uint32 base-offset math wrapped when the per-batch stride
    # h*n*hd exceeded 2^32 elements, bleeding batch 1 into batch 0. Trigger
    # the same stride with a large h and a small n (runtime scales with
    # h*n^2*hd, so shrinking n keeps this at seconds instead of hours):
    # 4096 * 8448 * 128 = 4.43e9 > 2^32.
    if torch.cuda.mem_get_info()[0] < 100 * 1024**3:
        pytest.skip("needs ~100 GB free VRAM (H100/H200/B200)")
    from sageattention import sageattn

    b, h, n, hd = 2, 4096, 8448, 128
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn((b, h, n, hd), device="cuda", dtype=torch.float16, generator=g)
    k = torch.randn((b, h, n, hd), device="cuda", dtype=torch.float16, generator=g)
    v = torch.randn((b, h, n, hd), device="cuda", dtype=torch.float16, generator=g)
    assert h * n * hd > 2**32

    # smooth_k=False: k.mean() reduction trees may differ between the batched
    # and per-batch calls; the kernels under test are batch-independent.
    out2 = sageattn(q, k, v, is_causal=True, smooth_k=False)
    for i in range(b):
        ref = sageattn(q[i : i + 1], k[i : i + 1], v[i : i + 1], is_causal=True, smooth_k=False)
        assert torch.equal(out2[i : i + 1], ref), f"batch {i} not isolated"
