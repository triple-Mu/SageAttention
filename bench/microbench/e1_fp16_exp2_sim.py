# SPDX-License-Identifier: Apache-2.0
# E1 prescreen, numerics side: does computing the softmax exp2 in fp16
# (ex2.approx.f16x2) instead of fp32 move the end-to-end attention output,
# given that P is quantized to e4m3 (3 mantissa bits) right after?
#
# Models the sage fp8 pipeline per (batch, head):
#   S(int32) -> float -> *qk_scale -> fmaf(sm_scale, -(row_max - 8.807))
#     -> exp2               <- fp32 (baseline) vs fp16 (E1)
#     -> e4m3 quantize -> P @ V_e4m3 (fp32 accum) -> * v_scale / denom
# row_max and the denominator stay fp32 in both variants (per task spec).
#
# Rounding fidelity note: torch casts/exp2 stand in for MUFU.EX2 / F2FP /
# cvt.satfinite (repo memory: don't replicate kernel rounding bit-by-bit).
#
# Run: /home/ubuntu/miniconda3/envs/torch/bin/python e1_fp16_exp2_sim.py

import torch

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
E4M3 = torch.float8_e4m3fn
S_FP8_OFFSET = 8.807


def quant_rowwise_int8(x):
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
    q = torch.round(x / scale).clamp(-127, 127)
    return q, scale


def pipeline(q, k, v, exp2_fp16):
    """One (seq_q, seq_k, hd) head. Returns O [seq_q, hd] float32."""
    qi, qs = quant_rowwise_int8(q)
    ki, ks = quant_rowwise_int8(k)
    # exact int32 S via float64 matmul (products/sums < 2^53)
    s_i32 = (qi.double() @ ki.double().t()).float()
    s_deq = s_i32 * (qs * ks.t())  # fp32 dequant, [seq_q, seq_k]

    sm_scale = (1.0 / q.shape[-1] ** 0.5) * 1.4426950408889634  # log2e folded
    # fmaf(m, sm_scale, -offset) in the kernel; mul+add is sub-ulp equivalent
    row_max = s_deq.amax(-1, keepdim=True) * sm_scale - S_FP8_OFFSET
    arg = s_deq * sm_scale - row_max  # fp32 FFMA chain, in (-inf, 8.807]

    if exp2_fp16:
        p = torch.exp2(arg.to(torch.float16)).float()  # F2FP pack + MUFU.EX2.F16
    else:
        p = torch.exp2(arg)  # MUFU.EX2 f32
    denom = p.sum(-1, keepdim=True)  # fp32, pre-quantization P as in the kernels

    p8 = p.to(E4M3).float()
    vs = v.abs().amax(dim=0, keepdim=True).clamp_min(1e-12) / 448.0  # per-channel
    v8 = (v / vs).to(E4M3).float()
    return (p8 @ v8) * vs / denom


def reference(q, k, v):
    """float64 softmax of the same dequantized logits, same quantized V."""
    qi, qs = quant_rowwise_int8(q)
    ki, ks = quant_rowwise_int8(k)
    s_deq = ((qi.double() @ ki.double().t()) * (qs * ks.t()).double())
    sm_scale = 1.0 / q.shape[-1] ** 0.5
    p = torch.softmax(s_deq * sm_scale, dim=-1)
    vs = v.abs().amax(dim=0, keepdim=True).clamp_min(1e-12) / 448.0
    v8 = (v / vs).to(E4M3).double()
    return (p @ v8 * vs.double()).float()


def metrics(a, b):
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    rel_l1 = ((a - b).abs().sum() / b.abs().sum().clamp_min(1e-20)).item()
    row_cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return cos, rel_l1, row_cos


def make_cases():
    cases = {}
    for hd in (64, 128):
        for n in (1024, 4096):
            q = torch.randn(n, hd, device=DEV)
            k = torch.randn(n, hd, device=DEV)
            v = torch.randn(n, hd, device=DEV)
            cases[f"randn_n{n}_hd{hd}"] = (q, k, v)
    # large dynamic range: a few dominant keys -> most P deep in the tail,
    # exp2 output crossing the fp16 subnormal region before e4m3 flushes it
    hd = 128
    n = 2048
    q = torch.randn(n, hd, device=DEV)
    k = torch.randn(n, hd, device=DEV) * 0.05
    k[:8] = q[:8] * 6.0  # logit gap ~ tens after sm_scale
    v = torch.randn(n, hd, device=DEV)
    cases["spiky_rowmax"] = (q, k, v)
    # near-uniform logits: every P lands at the e4m3 top bin (448)
    q = torch.randn(n, hd, device=DEV) * 0.01
    k = torch.randn(n, hd, device=DEV) * 0.01
    cases["uniform_logits"] = (q, k, v)
    # heavy-tailed activations (outlier channels, like real K distributions)
    q = torch.randn(n, hd, device=DEV) * torch.linspace(0.1, 8, hd, device=DEV)
    k = torch.randn(n, hd, device=DEV) * torch.linspace(8, 0.1, hd, device=DEV)
    cases["outlier_channels"] = (q, k, v)
    return cases


def main():
    print(f"device={DEV}, torch={torch.__version__}")
    # elementwise double-rounding probe: e4m3(exp2_fp16(x)) vs e4m3(exp2_fp32(x))
    x = torch.linspace(-30.0, S_FP8_OFFSET, 2_000_001, device=DEV)
    p32 = torch.exp2(x).to(E4M3).float()
    p16 = torch.exp2(x.to(torch.float16)).float().to(E4M3).float()
    diff = (p32 != p16)
    rel = ((p16 - p32).abs() / p32.abs().clamp_min(1e-20))[diff]
    rel_med = rel.median().item() if rel.numel() else 0.0
    rel_max = rel.max().item() if rel.numel() else 0.0
    print(
        f"[elementwise] e4m3 mismatch after fp16 exp2: {diff.float().mean().item():.3e} "
        f"of args in [-30, 8.807]; among mismatches rel err "
        f"median={rel_med:.3e} max={rel_max:.3e}"
    )

    hdr = (
        f"{'case':<22} {'cos(base,ref)':>14} {'cos(e1,ref)':>14} "
        f"{'cos(e1,base)':>14} {'relL1(e1,base)':>15} {'worst row cos':>14}"
    )
    print(hdr)
    print("-" * len(hdr))
    worst = 1.0
    for name, (q, k, v) in make_cases().items():
        ref = reference(q, k, v)
        base = pipeline(q, k, v, exp2_fp16=False)
        e1 = pipeline(q, k, v, exp2_fp16=True)
        cb, _, _ = metrics(base, ref)
        ce, _, _ = metrics(e1, ref)
        cd, rd, row_cos = metrics(e1, base)
        wr = row_cos.min().item()
        worst = min(worst, wr)
        print(f"{name:<22} {cb:>14.8f} {ce:>14.8f} {cd:>14.8f} {rd:>15.3e} {wr:>14.8f}")
    print(f"\nworst per-row cos(e1, base) across all cases: {worst:.8f}")


if __name__ == "__main__":
    main()
