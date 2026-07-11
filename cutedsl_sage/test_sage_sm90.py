# CuteDSL SageAttention sm90 正确性测试。
#   量化单测：任意 GPU/CPU 可跑；kernel 测试（Task 4 追加）需 H200。
#   远程执行：./hyper01.sh test
import math

import pytest
import torch
import torch.nn.functional as F

try:
    from .core import quant_q_int8_per_warp, quant_k_int8_per_block, quant_v_fp8_per_channel
except ImportError:
    from core import quant_q_int8_per_warp, quant_k_int8_per_block, quant_v_fp8_per_channel

LOG2E = math.log2(math.e)


# ============ 量化函数单测 ============

def test_q_scale_shape_order_and_tail():
    """分段常数张量验证段序：第 g 个 16 行段填 g+1 → scale[..., g] == (g+1)/127。"""
    b, s, n, d = 2, 337, 3, 64                       # ceil(337/64)=6 块 → 24 段，有效段 22
    g = torch.arange(s) // 16
    q = (g + 1).to(torch.float16)[None, :, None, None].expand(b, s, n, d).contiguous()
    q_int8, q_scale = quant_q_int8_per_warp(q)
    assert q_int8.shape == (b, s, n, d) and q_int8.dtype == torch.int8 and q_int8.is_contiguous()
    assert q_scale.shape == (b, n, 24) and q_scale.dtype == torch.float32 and q_scale.is_contiguous()
    n_valid = (s + 15) // 16                         # 22，末段仅 1 有效行（amax 只看有效行）
    expect = (torch.arange(n_valid).float() + 1) / 127.0
    assert torch.allclose(q_scale[..., :n_valid], expect.expand(b, n, -1))
    assert (q_int8.float() == 127).all()             # 段内常数 → 全 127
    tail = q_scale[..., n_valid:]                    # 全 padding 段
    assert (tail > 0).all() and not torch.isnan(tail).any()


def test_q_roundtrip_error():
    torch.manual_seed(0)
    q = torch.randn(1, 200, 2, 128, dtype=torch.float16)   # 非 64 倍数
    q_int8, q_scale = quant_q_int8_per_warp(q)
    scale_row = q_scale.repeat_interleave(16, dim=-1)[..., :200].permute(0, 2, 1)[..., None]
    deq = q_int8.float() * scale_row
    # round-to-nearest 半步长上界；3e-5 相对 slack 覆盖 fp32 中间舍入（127·2^-23 两次）
    assert ((deq - q.float()).abs() <= scale_row * (0.5 + 3e-5)).all()


def test_k_smooth_tail_and_roundtrip():
    torch.manual_seed(0)
    b, s, n, d = 1, 337, 2, 64                       # ceil(337/128)=3 块，尾块 81 有效行
    k = torch.randn(b, s, n, d, dtype=torch.float16) * 4
    km = k.mean(dim=1, keepdim=True)
    k_int8, k_scale = quant_k_int8_per_block(k, km)
    assert k_int8.shape == (b, s, n, d) and k_scale.shape == (b, n, 3)
    kf = k.float() - km.float()
    expect_tail = kf[:, 256:].abs().amax(dim=(1, 3)) / 127.0   # 尾块 amax 只含有效行
    assert torch.allclose(k_scale[..., 2], expect_tail)
    scale_row = k_scale.repeat_interleave(128, dim=-1)[..., :s].permute(0, 2, 1)[..., None]
    assert ((k_int8.float() * scale_row - kf).abs() <= scale_row * (0.5 + 3e-5)).all()


def test_v_pad_zero_scale_and_roundtrip():
    torch.manual_seed(0)
    b, s, n, d = 1, 337, 2, 128
    v = torch.randn(b, s, n, d, dtype=torch.bfloat16)
    v_fp8, v_scale = quant_v_fp8_per_channel(v)
    assert v_fp8.shape == (b, n, d, 384) and v_fp8.dtype == torch.float8_e4m3fn
    assert v_fp8.is_contiguous() and v_scale.shape == (b, n, d)
    assert (v_fp8.float()[..., s:] == 0).all()       # pad 区精确零
    vt = v.permute(0, 2, 3, 1).float()
    assert torch.allclose(v_scale, vt.abs().amax(-1).clamp_min(1e-7) / 448.0)
    deq = v_fp8.float()[..., :s] * v_scale[..., None]
    # e4m3 正规数相对误差 ≤ 2^-4，次正规绝对步长 ≤ v_scale·2^-9
    bound = vt.abs() * 2.0 ** -4 + v_scale[..., None] * 2.0 ** -9
    assert ((deq - vt).abs() <= bound + 1e-7).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
