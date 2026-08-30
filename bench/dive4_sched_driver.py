#!/usr/bin/env python
"""Dive 4 prescreen driver: CTA early-exit / tail-wave waste of sageattn_varlen.

The varlen grid is opened to max_seqlen_q (see
csrc/qattn/qk_int_sv_f8_varlen_launcher_sm89.cuh), so in a skewed batch the
query blocks past a sequence's end launch and return immediately, and the last
wave of real CTAs drains unevenly. This driver measures how much of the
attention kernel's duration that scheduling costs, two ways:

* ``--mode matrix``: for one shape x causal flag, walk the bench_varlen.py
  skew profiles (equal / .5-1x / .25-1x / .1-1x). Each skewed profile is paired
  with an equal-length control of the *same total tokens* ("eqtok"). Per config
  it reports the attention kernel's mean duration from a kineto pass plus an
  unprofiled wall-clock loop, and emits one JSON line. The skew-vs-eqtok delta
  is the scheduling-waste upper bound; the JSON carries the kernel's exact
  KV-tile work counts so the FLOP-mismatch of that pairing (a skewed batch does
  more tile work than an equal batch of the same tokens) can be divided out.

* ``--mode loop``: free-running loop of a single config for external
  profilers (ncu / nsys). Prints the config JSON, runs warmup + iters with no
  profiler hooks of its own.

Sequence lengths replicate bench_varlen.py exactly (same RNG, seed 0, longest
sequence pinned to seq_len so max_seqlen and therefore the grid is identical
across profiles of one shape).
"""

import argparse
import json
import math
import random
import time

import torch

from sageattention import sageattn_varlen

CTA_Q, CTA_K = 128, 64

PROFILES = [("equal", 1.0), ("ragged .5-1x", 0.5), ("ragged .25-1x", 0.25), ("ragged .1-1x", 0.1)]


def parse_shape(spec: str):
    fields = {tok[0]: int(tok[1:]) for tok in spec.split()}
    return fields["b"], fields["h"], fields["n"], fields["d"]


def seqlens(batch: int, seq_len: int, low: float, seed: int = 0) -> list:
    """bench_varlen.py's generator, bit for bit: longest pinned to seq_len."""
    rng = random.Random(seed)
    lens = [seq_len] + [max(1, round(seq_len * rng.uniform(low, 1.0))) for _ in range(batch - 1)]
    rng.shuffle(lens)
    return lens


def eqtok_lens(lens: list) -> list:
    """Equal-length control with the same total tokens (remainder spread)."""
    total, batch = sum(lens), len(lens)
    base, rem = divmod(total, batch)
    return [base + 1] * rem + [base] * (batch - rem)


def kv_tiles(lens: list, causal: bool) -> int:
    """Total CTA_K-tile iterations the kernel executes (its num_iterations
    summed over every non-empty CTA of one head; multiply by heads for the
    grid total). Mirrors qk_int_sv_f8_cuda_sm89.cuh line 488."""
    tiles = 0
    for n in lens:
        nq = math.ceil(n / CTA_Q)
        if causal:
            tiles += sum(math.ceil(min(n, (qi + 1) * CTA_Q) / CTA_K) for qi in range(nq))
        else:
            tiles += nq * math.ceil(n / CTA_K)
    return tiles


def flops(lens: list, heads: int, head_dim: int, causal: bool) -> float:
    f = sum(4 * heads * head_dim * n * n for n in lens)
    return f / 2 if causal else f


def build_case(lens, heads, head_dim, device, dtype):
    total = sum(lens)
    cu = torch.tensor([0] + lens, device=device, dtype=torch.int32).cumsum(0).to(torch.int32)
    gen = torch.Generator(device=device).manual_seed(0)
    q = torch.randn((total, heads, head_dim), device=device, dtype=dtype, generator=gen)
    k = torch.randn((total, heads, head_dim), device=device, dtype=dtype, generator=gen)
    v = torch.randn((total, heads, head_dim), device=device, dtype=dtype, generator=gen)
    return q, k, v, cu


def run_config(lens, heads, head_dim, causal, warmup, iters, kineto):
    """Returns (e2e ms/call, {kernel key: mean device µs/call}, attn µs/call)."""
    q, k, v, cu = build_case(lens, heads, head_dim, "cuda", torch.float16)
    max_len = max(lens)

    def call():
        sageattn_varlen(q, k, v, cu, cu, max_len, max_len, is_causal=causal)

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        call()
    torch.cuda.synchronize()
    e2e_ms = (time.perf_counter() - t0) * 1e3 / iters

    if not kineto:
        return e2e_ms, {}, None

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            call()
        torch.cuda.synchronize()

    kernels, attn_us = {}, None
    for row in prof.key_averages():
        if row.device_time_total <= 0:
            continue
        us_per_call = row.device_time_total / iters
        kernels[row.key[:100]] = round(us_per_call, 2)
        if "attn_kernel" in row.key:
            assert attn_us is None, "more than one attention kernel in the trace"
            assert row.count == iters, f"attn kernel count {row.count} != iters {iters}"
            attn_us = us_per_call
    assert attn_us is not None, "attention kernel not found in kineto trace"
    return e2e_ms, kernels, attn_us


def emit(shape_label, batch, heads, head_dim, causal, profile, kind, lens, rec):
    grid_x = math.ceil(max(lens) / CTA_Q)
    out = {
        "shape": shape_label,
        "causal": int(causal),
        "profile": profile,
        "kind": kind,
        "lens": lens,
        "tokens": sum(lens),
        "max_seqlen": max(lens),
        "grid": [grid_x, heads, batch],
        "ctas_launched": grid_x * heads * batch,
        "ctas_useful": sum(math.ceil(n / CTA_Q) for n in lens) * heads,
        "kv_tiles": kv_tiles(lens, causal) * heads,
        "flops": flops(lens, heads, head_dim, causal),
    }
    out.update(rec)
    print(json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", required=True, help="e.g. 'b8 h16 n4096 d128'")
    ap.add_argument("--causal", type=int, choices=[0, 1], required=True)
    ap.add_argument("--mode", choices=["matrix", "loop"], default="matrix")
    ap.add_argument("--low", type=float, help="loop mode: profile's low fraction (1.0 = equal)")
    ap.add_argument("--lens", help="loop mode: comma-separated lengths, overrides --low")
    ap.add_argument("--kind", choices=["skew", "eqtok"], default="skew", help="loop mode")
    ap.add_argument("--kineto", action="store_true", help="loop mode: also report attn_us")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    batch, heads, seq_len, head_dim = parse_shape(args.shape)
    causal = bool(args.causal)

    if args.mode == "loop":
        if args.lens:
            lens, profile = [int(x) for x in args.lens.split(",")], "custom"
            assert len(lens) == batch and max(lens) <= seq_len
        else:
            lens, profile = seqlens(batch, seq_len, args.low), args.low
        if args.kind == "eqtok":
            lens = eqtok_lens(lens)
        e2e_ms, kernels, attn_us = run_config(
            lens, heads, head_dim, causal, args.warmup, args.iters, args.kineto
        )
        rec = {"e2e_ms": round(e2e_ms, 4)}
        if args.kineto:
            rec.update({"attn_us": round(attn_us, 2), "kernels": kernels})
        emit(args.shape, batch, heads, head_dim, causal, profile, args.kind, lens, rec)
        print(f"loop done: {e2e_ms:.3f} ms/call over {args.iters} iters", flush=True)
        return

    for profile, low in PROFILES:
        lens = seqlens(batch, seq_len, low)
        cases = [("skew", lens)] if low >= 1.0 else [("skew", lens), ("eqtok", eqtok_lens(lens))]
        for kind, case_lens in cases:
            e2e_ms, kernels, attn_us = run_config(
                case_lens, heads, head_dim, causal, args.warmup, args.iters, True
            )
            rec = {"e2e_ms": round(e2e_ms, 4), "attn_us": round(attn_us, 2), "kernels": kernels}
            emit(args.shape, batch, heads, head_dim, causal, profile, kind, case_lens, rec)


if __name__ == "__main__":
    main()
