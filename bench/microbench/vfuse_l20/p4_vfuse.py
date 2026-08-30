#!/usr/bin/env python
"""V preprocessing microbench: fused (TransposeQuantFp8Kernel) vs separate
(TransposePadPermuteKernel + MeanScaleKernel), the two paths quant_v_fp8
switches between at kFusedVQuantMaxTokens.

Per kv_len it times:
  * composite  torch.ops.sageattention.quant_v_fp8 with the sm89 default
    arguments (mma_k16, scale_max 2.25, smooth_v False, pad 64) -- whichever
    path the installed build dispatches (the fused one only under the
    threshold; run once against a threshold-raised build to get the fused
    numbers above it);
  * separate   the forced two-kernel sequence through the low-level ops
    transpose_pad_v + scale_fuse_quant, mirroring quant_v_fp8's else-branch
    (same allocations, same arguments).

Two timing views per case: CUDA-event wall time around the calls, and
torch.profiler per-kernel device microseconds (the primary number; free of
allocator/launch noise). The profiler pass also records which kernels the
composite actually launched, which is the dispatch check.

Shapes: b4 h32 d128 HND, kv_len {1024,2048,3072,4096,6144,8192,12288,16384}.
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

B, H, D = 4, 32, 128
LENS = (1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384)
LAYOUT = "HND"
SCALE_MAX = 2.25  # sm89 default plan (pv fp32+fp16)
PAD = 64
WARMUP = 5
REPEATS = 5

KERNELS = ("TransposeQuantFp8Kernel", "TransposePadPermuteKernel", "MeanScaleKernel")


def iters_for(kv):
    if kv <= 4096:
        return 200
    if kv <= 8192:
        return 100
    return 50


def make_composite(v):
    def fn():
        return torch.ops.sageattention.quant_v_fp8(
            v,
            tensor_layout=LAYOUT,
            v_layout="mma_k16",
            scale_max=SCALE_MAX,
            smooth_v=False,
            pad_multiple=PAD,
        )

    return fn


def make_separate(v, kv):
    padded = (kv + PAD - 1) // PAD * PAD
    vt_sizes = (B, H, D, padded)  # HND

    def fn():
        vt = torch.empty(vt_sizes, device=v.device, dtype=v.dtype)
        torch.ops.sageattention.transpose_pad_v(v, vt, LAYOUT, True)
        fp8 = torch.empty(vt_sizes, device=v.device, dtype=torch.float8_e4m3fn)
        scale = torch.empty((B, H, D), device=v.device, dtype=torch.float32)
        torch.ops.sageattention.scale_fuse_quant(vt, fp8, scale, kv, SCALE_MAX, LAYOUT)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="label for this install (e.g. fix / fuseall)")
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()

    torch.cuda.set_device(0)
    prop = torch.cuda.get_device_properties(0)
    meta = {
        "tag": args.tag,
        "gpu": prop.name,
        "cc": (prop.major, prop.minor),
        "l2_bytes": prop.L2_cache_size,
        "sm_count": prop.multi_processor_count,
        "torch": torch.__version__,
        "compiled_archs": sorted(torch.ops.sageattention.compiled_archs()),
        "shape": {"b": B, "h": H, "d": D, "layout": LAYOUT},
        "scale_max": SCALE_MAX,
        "argv": sys.argv,
        "time_utc": time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime()),
    }
    print(json.dumps(meta, indent=1))

    rows = []
    for kv in LENS:
        g = torch.Generator(device="cuda").manual_seed(kv)
        v = torch.randn((B, H, kv, D), device="cuda", dtype=torch.float16, generator=g)
        iters = iters_for(kv)

        for path, fn in (("composite", make_composite(v)), ("separate", make_separate(v, kv))):
            wall_us, wall_all = event_time_us(fn, iters)
            per_kernel, names = kernel_time_us(fn, iters)
            ksum = sum(per_kernel.values())
            row = {
                "kv": kv,
                "path": path,
                "iters": iters,
                "wall_us": round(wall_us, 3),
                "wall_us_all": [round(x, 3) for x in wall_all],
                "kernel_us": {k: round(v_, 3) for k, v_ in per_kernel.items()},
                "kernel_us_sum": round(ksum, 3),
                "kernels_seen": names,
            }
            rows.append(row)
            print(
                f"kv={kv:6d} {path:9s} wall={wall_us:9.2f}us kernels={ksum:9.2f}us "
                f"{sorted(per_kernel)}"
            )
        del v
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
