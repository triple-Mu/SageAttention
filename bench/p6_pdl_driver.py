#!/usr/bin/env python
"""P6 PDL prescreen driver: steady-state ``sageattn`` loop for gap profiling.

One shape per process. Warmup runs outside the capture range; the measured
loop sits between ``torch.cuda.profiler.start()/stop()`` so
``nsys profile --capture-range=cudaProfilerApi`` records exactly the
steady-state iterations. The measured loop is free-running (no per-iteration
synchronize): launches are queued ahead, so inter-kernel gaps on the GPU
timeline reflect launch latency / dependency serialization -- the part PDL
could hide -- rather than CPU stalls we insert ourselves.

Typical use (inside the target container)::

    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
        -t cuda -s none --cpuctxsw=none -o out/seq1024 \
        python bench/p6_pdl_driver.py --seq 1024
    nsys stats --report cuda_gpu_trace --format csv --output out/seq1024 \
        out/seq1024.nsys-rep
    python bench/p6_pdl_gaps.py out/seq1024_cuda_gpu_trace.csv --iters 100

Modes:
  --time-only       no profiler hooks; wall-clock ms/call reference.
  --kineto PATH     torch.profiler cross-check; writes a chrome trace that
                    bench/p6_pdl_gaps.py can also read.
"""

import argparse
import time

import torch
from sageattention import sageattn


def run_loop(q, k, v, iters, causal):
    for _ in range(iters):
        sageattn(q, k, v, tensor_layout="HND", is_causal=causal)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--time-only", action="store_true")
    ap.add_argument("--kineto", type=str, default=None, help="chrome trace out path")
    args = ap.parse_args()

    torch.manual_seed(0)
    shape = (args.batch, args.heads, args.seq, args.head_dim)  # HND
    q = torch.randn(shape, dtype=torch.float16, device="cuda")
    k = torch.randn(shape, dtype=torch.float16, device="cuda")
    v = torch.randn(shape, dtype=torch.float16, device="cuda")

    run_loop(q, k, v, args.warmup, args.causal)
    torch.cuda.synchronize()

    if args.kineto:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            run_loop(q, k, v, args.iters, args.causal)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
        prof.export_chrome_trace(args.kineto)
    else:
        if not args.time_only:
            torch.cuda.profiler.start()
        t0 = time.perf_counter()
        run_loop(q, k, v, args.iters, args.causal)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if not args.time_only:
            torch.cuda.profiler.stop()

    ms = (t1 - t0) * 1e3 / args.iters
    print(
        f"b{args.batch} h{args.heads} d{args.head_dim} seq{args.seq} "
        f"causal={int(args.causal)} iters={args.iters}: {ms:.4f} ms/call"
    )


if __name__ == "__main__":
    main()
