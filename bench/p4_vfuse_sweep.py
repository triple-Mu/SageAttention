#!/usr/bin/env python
"""P4 fused-V-quant threshold sweep: fused (TransposeQuantFp8Kernel) vs
separate (TransposePadPermuteKernel + MeanScaleKernel) across (b*h, kv_len).

Arch-parameterized successor of microbench/vfuse_l20/p4_vfuse.py: the V
preprocessing arguments (v_layout / pad_multiple / scale_max) are taken from
torch.ops.sageattention.plan for the local device, so the same script runs on
sm89 (L20), sm100 (B200) and sm120 unchanged. Override them with --v-params
on devices whose plan resolves to the fp16 path (sm80/sm86).

Per (b, kv) it measures:
  * composite  torch.ops.sageattention.quant_v_fp8 -- whichever path the
    installed build dispatches. The fused path only runs at or below the
    per-arch threshold (fused_v_quant_max_tokens in csrc/sageattn/
    quant_cuda.cu); for fused numbers above it, rebuild that one TU with
    every `return <N>;` in fused_v_quant_max_tokens raised to
    `int64_t{1} << 60` and rerun with --tag fuseall. The threshold is a host
    constant, the kernels do not change.
  * separate   the forced two-kernel path through the low-level ops
    transpose_pad_v + (mean_)scale_fuse_quant, mirroring quant_v_fp8's else
    branch (same allocations, same arguments); build-independent.

Two timing views per case: CUDA-event wall time (median over repeats) and
torch.profiler per-kernel device microseconds (the primary number; free of
allocator/launch noise). The profiler pass also records which kernels the
composite launched, which is the dispatch self-check.

--bitcheck additionally compares composite vs separate outputs (v_fp8 and
scale tensors) bit for bit. Rows where the composite dispatched the separate
path are marked trivial=True (both sides ran the same kernels).
"""

import argparse
import json
import statistics
import sys
import time

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

import sageattention  # noqa: F401  (loads _C, registers torch.ops.sageattention)

OPS = torch.ops.sageattention
KERNELS = ("TransposeQuantFp8Kernel", "TransposePadPermuteKernel", "MeanScaleKernel")
WARMUP = 5
REPEATS = 5


def iters_for(kv):
    if kv <= 4096:
        return 200
    if kv <= 8192:
        return 100
    if kv <= 16384:
        return 50
    return 30


def v_params_from_plan(head_dim):
    cc = torch.cuda.get_device_capability()
    p = OPS.plan(cc[0], cc[1], head_dim)
    # (backend, gran, pv, smooth_v, smooth_v_ignored, pv_fp8, v_layout,
    #  v_pad_multiple, v_scale_max, need_value_scale, need_value_mean,
    #  blk_q, warp_q, blk_k, warp_k, error)
    backend, v_layout, pad, scale_max, err = p[0], p[6], p[7], p[8], p[15]
    if err:
        raise SystemExit(f"plan error: {err}")
    if pad == 0:
        raise SystemExit(
            f"plan for cc {cc} resolves to {backend} (fp16 V path, no quant_v_fp8); "
            "pass --v-params LAYOUT,PAD,SCALE_MAX to force one, e.g. mma_k16,64,448"
        )
    return backend, v_layout, pad, scale_max


def make_composite(v, layout, v_layout, scale_max, pad):
    def fn():
        return OPS.quant_v_fp8(
            v,
            tensor_layout=layout,
            v_layout=v_layout,
            scale_max=scale_max,
            smooth_v=False,
            pad_multiple=pad,
        )

    return fn


def make_separate(v, kv, b, h, d, layout, v_layout, scale_max, pad):
    padded = (kv + pad - 1) // pad * pad
    vt_sizes = (b, h, d, padded) if layout == "HND" else (b, d, h, padded)
    permute = v_layout == "mma_k16"
    # Mirror the composite's else branch: the quantize pass writes tokens up
    # to the 64-aligned bound only, so the fp8 buffer needs a zero fill
    # exactly when the pad_multiple=128 tail runs past it.
    covered = padded == (kv + 63) // 64 * 64
    fp8_alloc = torch.empty if covered else torch.zeros

    def fn():
        vt = torch.empty(vt_sizes, device=v.device, dtype=v.dtype)
        OPS.transpose_pad_v(v, vt, layout, permute)
        fp8 = fp8_alloc(vt_sizes, device=v.device, dtype=torch.float8_e4m3fn)
        scale = torch.empty((b, h, d), device=v.device, dtype=torch.float32)
        OPS.scale_fuse_quant(vt, fp8, scale, kv, scale_max, layout)
        return fp8, scale

    return fn


def event_time_us(fn, iters):
    """Median over REPEATS of (wall ms across iters)/iters, in us."""
    outs = []
    for _ in range(REPEATS):
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        outs.append(start.elapsed_time(end) / iters * 1000.0)
    return statistics.median(outs), outs


def kernel_time_us(fn, iters):
    """Per-kernel device us/call from one profiled pass; returns
    ({kernel_substring: us_per_call}, [raw kernel names])."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    per = {}
    names = []
    for evt in prof.key_averages():
        if evt.device_type != DeviceType.CUDA or not evt.count:
            continue
        dev_us = float(evt.self_device_time_total)
        if dev_us <= 0:
            continue
        names.append(evt.key)
        for pat in KERNELS:
            if pat in evt.key:
                per[pat] = per.get(pat, 0.0) + dev_us / iters
    return per, sorted(set(names))


def bit_compare(comp_fn, sep_fn):
    """Bitwise compare composite vs separate (v_fp8 bytes + scale words)."""
    c_fp8, c_scale = comp_fn()[:2]
    s_fp8, s_scale = sep_fn()
    fp8_eq = torch.equal(c_fp8.view(torch.uint8), s_fp8.view(torch.uint8))
    scale_eq = torch.equal(c_scale, s_scale)
    out = {"fp8_equal": fp8_eq, "scale_equal": scale_eq}
    if not fp8_eq:
        d = (c_fp8.float() - s_fp8.float()).abs()
        nbytes = (c_fp8.view(torch.uint8) != s_fp8.view(torch.uint8)).sum().item()
        out.update(fp8_max_abs=d.max().item(), fp8_diff_bytes=int(nbytes))
    if not scale_eq:
        d = (c_scale - s_scale).abs()
        r = d / c_scale.abs().clamp_min(1e-30)
        out.update(scale_max_abs=d.max().item(), scale_max_rel=r.max().item())
    del c_fp8, c_scale, s_fp8, s_scale
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="label for this install (e.g. prod / fuseall)")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--bh", default="32,64,128,256", help="comma list of b*h products")
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--kv", default="4096,6144,8192,12288,16384,24576", help="comma list of kv lengths")
    ap.add_argument("--layout", default="HND", choices=("HND", "NHD"))
    ap.add_argument("--v-params", default=None, help="override plan: LAYOUT,PAD,SCALE_MAX (e.g. mma_k16,64,448)")
    ap.add_argument("--bitcheck", action="store_true", help="also compare fused vs separate outputs bitwise")
    args = ap.parse_args()

    torch.cuda.set_device(0)
    h, d = args.heads, args.head_dim
    if args.v_params:
        vl, pad, sm = args.v_params.split(",")
        backend, v_layout, pad, scale_max = "override", vl, int(pad), float(sm)
    else:
        backend, v_layout, pad, scale_max = v_params_from_plan(d)

    bh_list = [int(x) for x in args.bh.split(",")]
    kv_list = [int(x) for x in args.kv.split(",")]
    bad = [x for x in bh_list if x % args.heads]
    if bad:
        raise SystemExit(f"--bh values {bad} not divisible by --heads {args.heads}")

    prop = torch.cuda.get_device_properties(0)
    meta = {
        "tag": args.tag,
        "gpu": prop.name,
        "cc": (prop.major, prop.minor),
        "l2_bytes": prop.L2_cache_size,
        "sm_count": prop.multi_processor_count,
        "torch": torch.__version__,
        "compiled_archs": sorted(OPS.compiled_archs()),
        "backend": backend,
        "v_layout": v_layout,
        "pad_multiple": pad,
        "scale_max": scale_max,
        "heads": h,
        "head_dim": d,
        "layout": args.layout,
        "argv": sys.argv,
        "time_utc": time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime()),
    }
    print(json.dumps(meta, indent=1))

    rows = []
    for bh in bh_list:
        b = bh // h
        for kv in kv_list:
            g = torch.Generator(device="cuda").manual_seed(bh * 1000003 + kv)
            shape = (b, h, kv, d) if args.layout == "HND" else (b, kv, h, d)
            v = torch.randn(shape, device="cuda", dtype=torch.float16, generator=g)
            iters = iters_for(kv)

            comp = make_composite(v, args.layout, v_layout, scale_max, pad)
            sep = make_separate(v, kv, b, h, d, args.layout, v_layout, scale_max, pad)

            for path, fn in (("composite", comp), ("separate", sep)):
                wall_us, wall_all = event_time_us(fn, iters)
                per_kernel, names = kernel_time_us(fn, iters)
                row = {
                    "bh": bh,
                    "b": b,
                    "kv": kv,
                    "path": path,
                    "iters": iters,
                    "wall_us": round(wall_us, 3),
                    "wall_us_all": [round(x, 3) for x in wall_all],
                    "kernel_us": {k: round(t, 3) for k, t in per_kernel.items()},
                    "kernel_us_sum": round(sum(per_kernel.values()), 3),
                    "kernels_seen": names,
                }
                if path == "composite":
                    fused = "TransposeQuantFp8Kernel" in per_kernel
                    row["fused_dispatch"] = fused
                    if args.bitcheck:
                        bc = bit_compare(comp, sep)
                        bc["trivial"] = not fused
                        row["bitcheck"] = bc
                rows.append(row)
                extra = ""
                if path == "composite":
                    extra = f" fused={row['fused_dispatch']}"
                    if args.bitcheck:
                        extra += f" bits_eq={row['bitcheck']['fp8_equal'] and row['bitcheck']['scale_equal']}"
                print(
                    f"bh={bh:4d} b={b:2d} kv={kv:6d} {path:9s} wall={wall_us:9.2f}us "
                    f"kernels={row['kernel_us_sum']:9.2f}us{extra}"
                )
            del v
            torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
