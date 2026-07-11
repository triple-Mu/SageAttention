# CuteDSL SageAttention (Hopper)：torch 量化 + CuTe-DSL kernel + 端到端 API。
# kernel 部分见文件下半部（Task 3 追加）。
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# ===== torch 量化（正确性优先，后续可换 triton/融合 kernel）=====

_AMAX_EPS = 1e-7    # 防除零下限，与 CUDA 版一致（csrc/fused/fused.cu:147）


def quant_q_int8_per_warp(q: torch.Tensor):
    """Q per-warp int8 量化：seq 按 64 分块、块内 4 段各 16 行，amax over (16 行, d)。

    q: [b, s, n, d] fp16/bf16 -> (q_int8 [b,s,n,d] int8, q_scale [b, n, ceil(s/64)*4] fp32)
    scale 段序 = seq 顺序的 16 行段（行 r 对应索引 r//16），与 CUDA quant_per_warp_int8_cuda 一致。
    尾块补零行不抬高 amax；全 padding 段 scale 钳到 _AMAX_EPS/127（非 0/NaN）。
    """
    b, s, n, d = q.shape
    nblk = (s + 63) // 64
    s_pad = nblk * 64
    qf = q.float()
    if s_pad != s:
        qf = F.pad(qf, (0, 0, 0, 0, 0, s_pad - s))
    qv = qf.view(b, nblk * 4, 16, n, d)
    amax = qv.abs().amax(dim=(2, 4)).clamp_min(_AMAX_EPS)              # [b, nblk*4, n]
    q_scale = (amax / 127.0).permute(0, 2, 1).contiguous()             # [b, n, nblk*4]
    # torch.round 为 round-half-to-even，与 CUDA float_to_int8_rn (cvt.rni) 一致
    q_int8 = torch.round(qv * (127.0 / amax)[:, :, None, :, None]) \
        .clamp_(-128, 127).to(torch.int8)
    return q_int8.view(b, s_pad, n, d)[:, :s].contiguous(), q_scale


def quant_k_int8_per_block(k: torch.Tensor, km: Optional[torch.Tensor] = None):
    """K per-block int8 量化（BLKK=128），量化前减 km（smooth_k）。

    k: [b, s, n_kv, d]，km: [b, 1, n_kv, d] 或 None
    -> (k_int8 [b,s,n_kv,d] int8, k_scale [b, n_kv, ceil(s/128)] fp32)
    先减 km 再补零，保证 padding 行不污染尾块 amax。
    """
    b, s, n, d = k.shape
    nblk = (s + 127) // 128
    s_pad = nblk * 128
    kf = k.float()
    if km is not None:
        kf = kf - km.float()
    if s_pad != s:
        kf = F.pad(kf, (0, 0, 0, 0, 0, s_pad - s))
    kv = kf.view(b, nblk, 128, n, d)
    amax = kv.abs().amax(dim=(2, 4)).clamp_min(_AMAX_EPS)              # [b, nblk, n]
    k_scale = (amax / 127.0).permute(0, 2, 1).contiguous()             # [b, n, nblk]
    k_int8 = torch.round(kv * (127.0 / amax)[:, :, None, :, None]) \
        .clamp_(-128, 127).to(torch.int8)
    return k_int8.view(b, s_pad, n, d)[:, :s].contiguous(), k_scale


def quant_v_fp8_per_channel(v: torch.Tensor):
    """V per-channel e4m3 量化并 materialize 为 [b, n_kv, d, s_pad] contiguous。

    v: [b, s, n_kv, d]；s_pad = ceil(s/128)*128，尾部零补（e4m3 精确 0）。
    v_scale = amax/448（amax 仅统计有效 seq），v_fp8 = v·448/amax。无 token permute。
    """
    b, s, n, d = v.shape
    s_pad = (s + 127) // 128 * 128
    vt = v.permute(0, 2, 3, 1).float()                                 # [b, n, d, s]
    amax = vt.abs().amax(dim=-1).clamp_min(_AMAX_EPS)                  # [b, n, d]
    v_scale = (amax / 448.0).contiguous()
    v_scaled = vt * (448.0 / amax)[..., None]                          # 幅值 ≤448，e4m3 不溢出
    if s_pad != s:
        v_scaled = F.pad(v_scaled, (0, s_pad - s))
    v_fp8 = v_scaled.to(torch.float8_e4m3fn).contiguous()
    return v_fp8, v_scale
