# Triton 版量化 kernel：与 core.py 的 torch 参考实现逐 bit 一致（bit-identical）。
# 数值契约（不可变），关键是复刻 torch CUDA 标量除法的真实指令序列（H200 实测 bit 级确认）：
#   - torch `tensor / c`   = tensor * rn(1/c)      （TensorIterator cpu-scalar 分母的倒数优化）
#   - torch `c / tensor`   = rn(1/tensor) * c      （__rtruediv__ → reciprocal().mul(c)）
#     —— 两者都不是 IEEE 除法（与 div.rn 有 1ulp 差异），故这里用同款 rcp_rn/常量倒数复刻
#   - fp16/bf16 → fp32 转换、abs、max 归约、fp32 乘法均为精确/顺序无关运算，两侧天然一致
#   - round 用 libdevice.rint（round-half-to-even），对齐 torch.round（nearbyint / cvt.rni）
#   - fp32→e4m3 用 rtne 下转（PTX cvt.rn.satfinite.e4m3x2），|x|≤448·(1+ε) 时与
#     torch 的 c10 软件转换逐 bit 一致（satfinite 与 c10 仅在 x≥464 时不同，此处不可达）
#   - 运算顺序严格复刻 torch 版：amax→clamp_min(1e-7)→scale=amax/127；量化乘 (127/amax)
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_AMAX_EPS = 1e-7  # 与 core._AMAX_EPS 一致（core 依赖本模块，常量就地定义避免循环 import）


@triton.jit
def _quant_q_int8_per_warp_kernel(Q, Out, Scale, s,
                                  stride_qb, stride_qs, stride_qn,
                                  stride_ob, stride_os, stride_on,
                                  stride_sb, stride_sn,
                                  D: tl.constexpr):
    # 一个 program 处理一个 16 行段 ×(单头, d)：amax over (16, d)
    seg = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    rows = seg * 16 + tl.arange(0, 16)
    cols = tl.arange(0, D)
    row_ok = rows < s
    x = tl.load(Q + b * stride_qb + h * stride_qn
                + rows[:, None] * stride_qs + cols[None, :],
                mask=row_ok[:, None], other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x)), 1e-7)
    tl.store(Scale + b * stride_sb + h * stride_sn + seg,
             amax * (1.0 / 127.0))                  # = torch `amax / 127.0`
    r = libdevice.rcp_rn(amax) * 127.0              # = torch `127.0 / amax`
    y = tl.minimum(tl.maximum(libdevice.rint(x * r), -128.0), 127.0).to(tl.int8)
    tl.store(Out + b * stride_ob + h * stride_on
             + rows[:, None] * stride_os + cols[None, :],
             y, mask=row_ok[:, None])


@triton.jit
def _quant_k_int8_per_block_kernel(K, KM, Out, Scale, s,
                                   stride_kb, stride_ks, stride_kn,
                                   stride_mb, stride_mn,
                                   stride_ob, stride_os, stride_on,
                                   stride_sb, stride_sn,
                                   HAS_KM: tl.constexpr, D: tl.constexpr):
    # 一个 program 处理一个 128 行块 ×(单头, d)，量化前融合减 km（smooth_k）
    blk = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    rows = blk * 128 + tl.arange(0, 128)
    cols = tl.arange(0, D)
    row_ok = rows < s
    x = tl.load(K + b * stride_kb + h * stride_kn
                + rows[:, None] * stride_ks + cols[None, :],
                mask=row_ok[:, None], other=0.0).to(tl.float32)
    if HAS_KM:
        km = tl.load(KM + b * stride_mb + h * stride_mn + cols).to(tl.float32)
        # torch 版先减 km 再补零：padding 行保持 0，不污染 amax
        x = tl.where(row_ok[:, None], x - km[None, :], 0.0)
    amax = tl.maximum(tl.max(tl.abs(x)), 1e-7)
    tl.store(Scale + b * stride_sb + h * stride_sn + blk,
             amax * (1.0 / 127.0))                  # = torch `amax / 127.0`
    r = libdevice.rcp_rn(amax) * 127.0              # = torch `127.0 / amax`
    y = tl.minimum(tl.maximum(libdevice.rint(x * r), -128.0), 127.0).to(tl.int8)
    tl.store(Out + b * stride_ob + h * stride_on
             + rows[:, None] * stride_os + cols[None, :],
             y, mask=row_ok[:, None])


@triton.jit
def _quant_v_fp8_amax_kernel(V, Amax, VScale, s,
                             stride_vb, stride_vs, stride_vn,
                             stride_ab, stride_an,
                             BS: tl.constexpr, BD: tl.constexpr):
    # pass 1：per-channel amax over s（仅有效行）；同时写出 v_scale = amax/448
    dblk = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    cols = dblk * BD + tl.arange(0, BD)
    base = V + b * stride_vb + h * stride_vn + cols[None, :]
    acc = tl.zeros((BD,), tl.float32)
    for s0 in range(0, s, BS):
        rows = s0 + tl.arange(0, BS)
        x = tl.load(base + rows[:, None] * stride_vs,
                    mask=rows[:, None] < s, other=0.0).to(tl.float32)
        acc = tl.maximum(acc, tl.max(tl.abs(x), axis=0))
    amax = tl.maximum(acc, 1e-7)
    off = b * stride_ab + h * stride_an + cols
    tl.store(Amax + off, amax)                          # 量化 pass 复用同一 amax
    tl.store(VScale + off, amax * (1.0 / 448.0))        # = torch `amax / 448.0`


@triton.jit
def _quant_v_fp8_transpose_kernel(V, Amax, Out, s,
                                  stride_vb, stride_vs, stride_vn,
                                  stride_ab, stride_an,
                                  stride_ob, stride_on, stride_od,
                                  D_BLKS: tl.constexpr,
                                  BS: tl.constexpr, BD: tl.constexpr):
    # pass 2：v·(448/amax) → e4m3，转置 materialize 为 [b,n,d,s_pad]；
    # s_pad 为 BS 整数倍，pad 行由 masked load 得 0 → e4m3 精确 0
    pid0 = tl.program_id(0)
    sblk = pid0 // D_BLKS
    dblk = pid0 % D_BLKS
    h = tl.program_id(1)
    b = tl.program_id(2)
    rows = sblk * BS + tl.arange(0, BS)
    cols = dblk * BD + tl.arange(0, BD)
    x = tl.load(V + b * stride_vb + h * stride_vn
                + rows[:, None] * stride_vs + cols[None, :],
                mask=rows[:, None] < s, other=0.0).to(tl.float32)
    amax = tl.load(Amax + b * stride_ab + h * stride_an + cols)
    r = libdevice.rcp_rn(amax) * 448.0              # = torch `448.0 / amax`
    y = (x * r[None, :]).to(tl.float8e4nv, fp_downcast_rounding="rtne")
    tl.store(Out + b * stride_ob + h * stride_on
             + cols[:, None] * stride_od + rows[None, :],
             tl.trans(y))


def quant_q_int8_per_warp(q: torch.Tensor):
    """triton 版，语义/输出与 core._quant_q_int8_per_warp_torch bit-identical。"""
    b, s, n, d = q.shape
    nseg = (s + 63) // 64 * 4
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    q_scale = torch.empty((b, n, nseg), dtype=torch.float32, device=q.device)
    _quant_q_int8_per_warp_kernel[(nseg, n, b)](
        q, q_int8, q_scale, s,
        q.stride(0), q.stride(1), q.stride(2),
        q_int8.stride(0), q_int8.stride(1), q_int8.stride(2),
        q_scale.stride(0), q_scale.stride(1),
        D=d)
    return q_int8, q_scale


def quant_k_int8_per_block(k: torch.Tensor, km=None):
    """triton 版，语义/输出与 core._quant_k_int8_per_block_torch bit-identical。"""
    b, s, n, d = k.shape
    nblk = (s + 127) // 128
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    k_scale = torch.empty((b, n, nblk), dtype=torch.float32, device=k.device)
    has_km = km is not None
    _quant_k_int8_per_block_kernel[(nblk, n, b)](
        k, km if has_km else k, k_int8, k_scale, s,
        k.stride(0), k.stride(1), k.stride(2),
        km.stride(0) if has_km else 0, km.stride(2) if has_km else 0,
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2),
        k_scale.stride(0), k_scale.stride(1),
        HAS_KM=has_km, D=d,
        num_warps=8)
    return k_int8, k_scale


def quant_v_fp8_per_channel(v: torch.Tensor):
    """triton 版，语义/输出与 core._quant_v_fp8_per_channel_torch bit-identical。"""
    b, s, n, d = v.shape
    s_pad = (s + 127) // 128 * 128
    BS, BD = 128, 64                # H200 扫参结果：quant pass (128,64,w4)、amax pass (512,64,w8)
    v_scale = torch.empty((b, n, d), dtype=torch.float32, device=v.device)
    amax = torch.empty((b, n, d), dtype=torch.float32, device=v.device)
    v_fp8 = torch.empty((b, n, d, s_pad), dtype=torch.float8_e4m3fn, device=v.device)
    _quant_v_fp8_amax_kernel[(d // BD, n, b)](
        v, amax, v_scale, s,
        v.stride(0), v.stride(1), v.stride(2),
        amax.stride(0), amax.stride(1),
        BS=512, BD=BD, num_warps=8)
    _quant_v_fp8_transpose_kernel[(s_pad // BS * (d // BD), n, b)](
        v, amax, v_fp8, s,
        v.stride(0), v.stride(1), v.stride(2),
        amax.stride(0), amax.stride(1),
        v_fp8.stride(0), v_fp8.stride(1), v_fp8.stride(2),
        D_BLKS=d // BD, BS=BS, BD=BD)
    return v_fp8, v_scale
