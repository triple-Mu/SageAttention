# CuteDSL SageAttention sm90 正确性测试。
#   量化单测：任意 GPU/CPU 可跑；kernel 测试（Task 4 追加）需 H200。
#   远程执行：./hyper01.sh test
import math

import pytest
import torch

try:
    from .core import LOG2_448, quant_q_int8_per_warp, quant_k_int8_per_block, quant_v_fp8_per_channel
except ImportError:
    from core import LOG2_448, quant_q_int8_per_warp, quant_k_int8_per_block, quant_v_fp8_per_channel

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


# ---- 阈值：2026-07-12 H200 全矩阵实测标定（kernel @ 7d027e9），规则 max_observed×3
#      取整到一位有效数字；若 ×3 比原阈值松则保持原值（收紧原则）----
SIM_DIFF_MAX = 1e-5        # 一级：1-cossim（kernel vs 量化模拟，残差仅 FP22 累加 + exp2/rcp 舍入）
                           #   实测 max 3.24e-6 ×3=9.7e-6 → 1e-5（原 1e-4）
SIM_MAXREL_MAX = 2e-2      # 一级：max|diff|/max|ref|（s≥4096 放宽为 4e-2，FP22 误差随 s 增长）
                           #   实测 max 7.21e-3 (s<4096) ×3=2.16e-2 → 2e-2 不变；s≥4096 实测 max 8.4e-3
E2E_DIFF_MAX = 2e-3        # 二级：1-cossim（vs fp32 SDPA，量化损失应在论文水平）
                           #   实测 max 8.55e-4 ×3=2.6e-3 取整反而更松 → 保持 2e-3（余量 2.3×）
E2E_L1_MAX = 7e-2          # 二级：相对 L1。实测 max 4.12e-2 ×3=1.2e-1 更松 → 保持 7e-2（余量 1.7×）


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """1 - 余弦相似度（bench/utils.py 同款）。"""
    x, y = x.double(), y.double()
    return (1 - 2 * (x * y).sum() / (x * x + y * y).sum()).item()


def rel_l1(x: torch.Tensor, ref: torch.Tensor) -> float:
    return ((x - ref).abs().sum() / ref.abs().sum()).item()


# ============ 参考实现 ============

def ref_sdpa(q, k, v, is_causal, sm_scale):
    """fp32 SDPA 参考（支持 GQA）。输入 NHD，返回 [b, s_q, n_q, d] fp32。"""
    n_q, n_kv = q.shape[2], k.shape[2]
    qf = q.permute(0, 2, 1, 3).float()
    kf = k.permute(0, 2, 1, 3).float()
    vf = v.permute(0, 2, 1, 3).float()
    if n_q != n_kv:
        r = n_q // n_kv
        kf = kf.repeat_interleave(r, dim=1)
        vf = vf.repeat_interleave(r, dim=1)
    s = torch.matmul(qf, kf.transpose(-1, -2)) * sm_scale
    if is_causal:
        mask = torch.ones(q.shape[1], k.shape[1], dtype=torch.bool, device=q.device).tril()
        s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, vf).permute(0, 2, 1, 3)


def ref_quant_sim(q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale, is_causal, sm_scale):
    """fp32 精确模拟量化 attention，与 kernel 逐块同构：
    per-128 列块 online softmax，P448 = exp2((S−m)·c + log2 448) 直接在 448 常量域生成
    （与 kernel 的 exp2 常量偏移同构，P1.2），转 e4m3 后累加；
    row_sum448 用量化前 f32 P448，O = acc·v_scale/row_sum448（448 代数相消）。
    残差仅剩 kernel 的 FP22 累加与 exp2/rcp 舍入。
    注意：P→e4m3 发生在 running-max 域，与全局 softmax 后量化不可交换（实测差 3e-4）。
    返回 [b, s_q, n_q, d] fp32。
    """
    b, s_q, n_q, d = q_int8.shape
    s_k, n_kv = k_int8.shape[1], k_int8.shape[2]
    g = n_q // n_kv
    dev = q_int8.device

    qf = q_int8.permute(0, 2, 1, 3).float()
    kf = k_int8.permute(0, 2, 1, 3).float()
    vf = v_fp8.float()[..., :s_k]                    # pad 列在 kernel 中被 mask 压零，等价于截断
    if g != 1:
        kf = kf.repeat_interleave(g, dim=1)
        vf = vf.repeat_interleave(g, dim=1)
        k_scale = k_scale.repeat_interleave(g, dim=1)
        v_scale = v_scale.repeat_interleave(g, dim=1)

    qs_row = q_scale.repeat_interleave(16, dim=-1)[..., :s_q]     # 行 r → scale[r//16]
    ks_col = k_scale.repeat_interleave(128, dim=-1)[..., :s_k]    # 列 j → scale[j//128]

    S = torch.matmul(qf, kf.transpose(-1, -2))       # |S|<2^24，fp32 精确
    S = S * qs_row[..., :, None] * ks_col[..., None, :]
    if is_causal:
        mask = torch.ones(s_q, s_k, dtype=torch.bool, device=dev).tril()
        S = S.masked_fill(~mask, float("-inf"))

    CTA_K = 128
    c = sm_scale * LOG2E
    m_run = torch.full((b, n_q, s_q, 1), -5e6, device=dev)   # 有限大负数避免 -inf 参与运算出 NaN
    d_run = torch.zeros_like(m_run)
    acc = torch.zeros(b, n_q, s_q, d, device=dev)
    for it in range((s_k + CTA_K - 1) // CTA_K):
        sl = slice(it * CTA_K, min((it + 1) * CTA_K, s_k))
        sb = S[..., sl]
        m_new = torch.maximum(m_run, sb.amax(-1, keepdim=True))
        resc = torch.exp2((m_run - m_new) * c)                 # m 补偿项不带 448 偏移
        pb448 = torch.exp2((sb - m_new) * c + LOG2_448)        # 448 常量域，与 kernel 同构
        d_run = d_run * resc + pb448.sum(-1, keepdim=True)     # row_sum448 用量化前 f32 P448
        pb448_q = pb448.to(torch.float8_e4m3fn).float()
        acc = acc * resc + torch.matmul(pb448_q, vf[..., sl].transpose(-1, -2))
        m_run = m_new
    o = acc * v_scale[:, :, None, :] / d_run
    return o.permute(0, 2, 1, 3)


# ============ kernel 编译冒烟（需 H200 + cutlass DSL）============

def _sm90_available():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


@pytest.mark.skipif(not _sm90_available(), reason="需要 sm90 GPU")
def test_compile_smoke():
    try:
        from .core import SageAttnSm90
    except ImportError:
        from core import SageAttnSm90
    kern = SageAttnSm90.from_args(128, False, torch.float16, 1)
    assert kern.compiled_kernel is not None
    assert SageAttnSm90.from_args(128, False, torch.float16, 1) is kern   # 缓存命中


def _sage_call():
    try:
        from .core import sageattn_qk_int8_pv_fp8_hopper
    except ImportError:
        from core import sageattn_qk_int8_pv_fp8_hopper
    return sageattn_qk_int8_pv_fp8_hopper


def _run_two_level(q, k, v, is_causal, smooth_k=True, check_l2=True):
    """一级：kernel vs 量化模拟逐元素紧阈值；二级：端到端 vs fp32 SDPA 松阈值。
    check_l2=False 供对抗性输入使用（量化模型固有损失超阈值，二级比对无意义）。"""
    dtype = q.dtype
    d = q.shape[-1]
    s_max_dim = max(q.shape[1], k.shape[1])
    sm_scale = d ** -0.5
    o = _sage_call()(q, k, v, is_causal=is_causal, sm_scale=sm_scale, smooth_k=smooth_k)
    assert o.shape == q.shape and o.dtype == dtype
    of = o.float()

    # km 表达式与 core.py 端到端内部完全一致 → 量化输入 bit-identical
    km = k.mean(dim=1, keepdim=True) if smooth_k else None
    q_i8, q_sc = quant_q_int8_per_warp(q)
    k_i8, k_sc = quant_k_int8_per_block(k, km)
    v_f8, v_sc = quant_v_fp8_per_channel(v)
    o_sim = ref_quant_sim(q_i8, q_sc, k_i8, k_sc, v_f8, v_sc, is_causal, sm_scale)
    o_sim = o_sim.to(dtype).float()                  # 两侧同 cast，剔除输出精度影响
    d1 = calc_diff(of, o_sim)
    maxrel = ((of - o_sim).abs().max() / o_sim.abs().max().clamp_min(1e-6)).item()
    maxrel_lim = SIM_MAXREL_MAX if s_max_dim < 4096 else 2 * SIM_MAXREL_MAX
    assert d1 < SIM_DIFF_MAX, f"level1 1-cossim={d1:.3e}"
    assert maxrel < maxrel_lim, f"level1 maxrel={maxrel:.3e}"

    if not check_l2:
        print(f"  L1: 1-cos={d1:.2e} maxrel={maxrel:.2e} | L2: skipped")
        return
    o_ref = ref_sdpa(q, k, v, is_causal, sm_scale)
    d2, l2 = calc_diff(of, o_ref), rel_l1(of, o_ref)
    assert d2 < E2E_DIFF_MAX, f"level2 1-cossim={d2:.3e}"
    assert l2 < E2E_L1_MAX, f"level2 rel_l1={l2:.3e}"
    print(f"  L1: 1-cos={d1:.2e} maxrel={maxrel:.2e} | L2: 1-cos={d2:.2e} rel_l1={l2:.2e}")


def _mk_qkv(b, n_q, n_kv, s_q, s_k, d, dtype, seed=42):
    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(b, s_q, n_q, d, dtype=dtype, device=dev)
    # K 加通道偏置让 smooth_k 起实际作用
    k = torch.randn(b, s_k, n_kv, d, dtype=dtype, device=dev) \
        + torch.randn(1, 1, n_kv, d, dtype=dtype, device=dev) * 2
    v = torch.randn(b, s_k, n_kv, d, dtype=dtype, device=dev)
    return q, k, v


# (b, n_q, n_kv, s_q, s_k, d)；大组合标 slow（迭代时 -m "not slow" 跳过）
_SHAPES = (
    [(b, n, n, s, s, d) for b in (1, 2) for n in (8, 24)
     for s in (1024, 4096, 337) for d in (64, 128)]
    + [(1, 8, 8, s, s, d) for s in (64, 100, 128) for d in (64, 128)]   # 单 KV 块边界
    + [(1, 4, 4, 337, 1024, 128)]                                       # s_q != s_k（仅 non-causal）
    + [(2, 8, 4, 1024, 1024, 128), (2, 8, 4, 1024, 1024, 64)]           # GQA
)


def _params():
    out = []
    for c in _SHAPES:
        b, n_q, n_kv, s_q, s_k, d = c
        marks = [pytest.mark.slow] if b * n_q * max(s_q, s_k) >= 2 * 24 * 4096 else []
        out.append(pytest.param(*c, marks=marks,
                                id=f"b{b}nq{n_q}nkv{n_kv}sq{s_q}sk{s_k}d{d}"))
    return out


@pytest.mark.skipif(not _sm90_available(), reason="需要 sm90 GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("is_causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("b,n_q,n_kv,s_q,s_k,d", _params())
def test_kernel_two_level(b, n_q, n_kv, s_q, s_k, d, is_causal, dtype):
    if is_causal and s_q != s_k:
        pytest.skip("causal 要求 s_q == s_k")
    q, k, v = _mk_qkv(b, n_q, n_kv, s_q, s_k, d, dtype)
    _run_two_level(q, k, v, is_causal)


@pytest.mark.skipif(not _sm90_available(), reason="需要 sm90 GPU")
@pytest.mark.parametrize("is_causal", [False, True], ids=["full", "causal"])
def test_kernel_scale_indexing(is_causal):
    """异质量级 stress：相邻 warp 段 / K 块 / V channel 的 amax 差 4 倍，
    kernel 侧任一 scale 索引错位都会被一级比对放大为数量级误差，稳定检出。
    仅一级比对：该输入 max|logit|≈3.7e4，int8 噪声过 exp 后二级固有损失
    1-cossim=3.8e-2/8.7e-2（full/causal，sim-vs-sdpa 实测，与 kernel 无关）。"""
    b, n, s, d = 1, 4, 512, 128
    q, k, v = _mk_qkv(b, n, n, s, s, d, torch.float16, seed=0)
    dev = q.device
    q = q * (4.0 ** (torch.arange(s, device=dev) // 16 % 4))[None, :, None, None].half()
    k = k * (4.0 ** (torch.arange(s, device=dev) // 128 % 4))[None, :, None, None].half()
    v = v * (4.0 ** (torch.arange(d, device=dev) % 4))[None, None, None, :].half()
    _run_two_level(q, k, v, is_causal, check_l2=False)


@pytest.mark.skipif(not _sm90_available(), reason="需要 sm90 GPU")
def test_kernel_no_smooth_k():
    q, k, v = _mk_qkv(1, 8, 8, 1024, 1024, 128, torch.float16)
    _run_two_level(q, k, v, is_causal=False, smooth_k=False)


@pytest.mark.slow
@pytest.mark.skipif(not _sm90_available(), reason="需要 sm90 GPU")
@pytest.mark.parametrize("s", [16384, 32768])
def test_fp22_long_seq(s):
    """长序列误差趋势观察（两级累加后块内仍有 FP22 局部累加）：只按二级（松）阈值断言，
    一级指标打印出来供人工评估。
    n=2 控制 ref_quant_sim 显存（s=32768 时 S 矩阵 2×32768²×4B ≈ 8.6GB/份）。"""
    q, k, v = _mk_qkv(1, 2, 2, s, s, 128, torch.float16)
    sm_scale = 128 ** -0.5
    o = _sage_call()(q, k, v, is_causal=False, sm_scale=sm_scale).float()
    km = k.mean(dim=1, keepdim=True)
    q_i8, q_sc = quant_q_int8_per_warp(q)
    k_i8, k_sc = quant_k_int8_per_block(k, km)
    v_f8, v_sc = quant_v_fp8_per_channel(v)
    o_sim = ref_quant_sim(q_i8, q_sc, k_i8, k_sc, v_f8, v_sc, False, sm_scale).float()
    o_ref = ref_sdpa(q, k, v, False, sm_scale)
    print(f"\n[s={s}] L1: 1-cos={calc_diff(o, o_sim):.2e} "
          f"maxrel={((o - o_sim).abs().max() / o_sim.abs().max()).item():.2e} | "
          f"L2: 1-cos={calc_diff(o, o_ref):.2e} rel_l1={rel_l1(o, o_ref):.2e}")
    assert calc_diff(o, o_ref) < E2E_DIFF_MAX


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
