"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

sageattn_varlen against the padded batch it replaces.

The control group is `sageattn` on the same tokens padded up to max_seqlen,
which is what a caller without a packed layout pays today. Both sides are
timed through the public API, so quantization and the smooth_k reduction are
inside the measurement -- the varlen quantization also skips the padding, and
splitting that out would flatter the packed path.

Every profile of a shape shares one dense number: the longest sequence is
pinned to seq_len, so the padded batch is the same [batch, seq_len, ...] work
no matter how ragged the rest is. The ratio column is then exactly what the
packed layout buys.

    python bench/bench_varlen.py
    python bench/bench_varlen.py --csv varlen.csv --iters 100

This script only uses `sageattention`'s public API (unlike the other scripts
in this directory, which still import the removed per-arch pybind modules).
"""

import argparse
import csv
import random
import statistics

import torch

from sageattention import sageattn, sageattn_varlen

# (label, batch, heads, seq_len, head_dim)
SHAPES = [
    ("b8 h16 n1024 d128", 8, 16, 1024, 128),
    ("b8 h16 n4096 d128", 8, 16, 4096, 128),
    ("b4 h16 n8192 d128", 4, 16, 8192, 128),
    ("b16 h8 n2048 d64", 16, 8, 2048, 64),
    ("b8 h32 n4096 d64", 8, 32, 4096, 64),
]

# (label, shortest sequence as a fraction of seq_len)
PROFILES = [("equal", 1.0), ("ragged .25-1x", 0.25), ("ragged .1-1x", 0.1)]


def parse_shape(spec: str):
    """'b8 h16 n4096 d128' or '8,16,4096,128' -> (label, batch, heads, seq, dim)."""
    if "," in spec:
        batch, heads, seq, dim = (int(x) for x in spec.split(","))
        return (f"b{batch} h{heads} n{seq} d{dim}", batch, heads, seq, dim)
    fields = {tok[0]: int(tok[1:]) for tok in spec.split()}
    return (spec, fields["b"], fields["h"], fields["n"], fields["d"])


def parse_profile(spec: str):
    """'0.25' -> ('ragged .25-1x', 0.25); '1' -> ('equal', 1.0)."""
    low = float(spec)
    return ("equal" if low >= 1.0 else f"ragged {spec.lstrip('0')}-1x", low)


def seqlens(batch: int, seq_len: int, low: float, seed: int = 0) -> list:
    """Sequence lengths spread over [low * seq_len, seq_len], longest pinned.

    Pinning one sequence to seq_len keeps max_seqlen (and therefore the padded
    control group) identical across profiles of the same shape.
    """
    rng = random.Random(seed)
    lens = [seq_len] + [max(1, round(seq_len * rng.uniform(low, 1.0))) for _ in range(batch - 1)]
    rng.shuffle(lens)
    return lens


def timeit(fns, warmup: int, iters: int) -> list:
    """Median ms of each callable, sampled round-robin.

    One iteration runs every callable once, so a clock or thermal drift over
    the run lands on all of them instead of on whichever went last.
    """
    for _ in range(warmup):
        for fn in fns:
            fn()
    torch.cuda.synchronize()

    samples = [[] for _ in fns]
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        for i, fn in enumerate(fns):
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples[i].append(start.elapsed_time(end))
    return [statistics.median(s) for s in samples]


def run_case(shape, lens, causal, device, dtype, warmup, iters):
    batch, heads, seq_len, head_dim = shape
    total = sum(lens)
    cu = torch.tensor([0] + lens, device=device, dtype=torch.int32).cumsum(0).to(torch.int32)

    gen = torch.Generator(device=device).manual_seed(0)
    shape = (batch, seq_len, heads, head_dim)
    qd = torch.randn(shape, device=device, dtype=dtype, generator=gen)
    kd = torch.randn(shape, device=device, dtype=dtype, generator=gen)
    vd = torch.randn(shape, device=device, dtype=dtype, generator=gen)

    # the packed tensors are the dense ones with the padding cut away
    qp = torch.cat([qd[b, : lens[b]] for b in range(batch)])
    kp = torch.cat([kd[b, : lens[b]] for b in range(batch)])
    vp = torch.cat([vd[b, : lens[b]] for b in range(batch)])

    def varlen():
        return sageattn_varlen(qp, kp, vp, cu, cu, seq_len, seq_len, is_causal=causal)

    def dense():
        return sageattn(qd, kd, vd, tensor_layout="NHD", is_causal=causal)

    t_varlen, t_dense = timeit([varlen, dense], warmup, iters)

    # useful FLOPs: 4 * heads * head_dim * n^2 per sequence, halved when causal
    def tflops(ns, ms):
        f = sum(4 * heads * head_dim * n * n for n in ns)
        return (f / 2 if causal else f) / ms / 1e9

    return {
        "tokens": total,
        "padded_tokens": batch * seq_len,
        "varlen_ms": t_varlen,
        "dense_ms": t_dense,
        "speedup": t_dense / t_varlen,
        "varlen_tflops": tflops(lens, t_varlen),
        "dense_tflops": tflops([seq_len] * batch, t_dense),
    }


def main():
    ap = argparse.ArgumentParser(description="sageattn_varlen vs the padded dense batch")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", help="fail unless the device resolves to this backend")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--causal", default="both", choices=["0", "1", "both"])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--csv", help="also write the rows to this file")
    ap.add_argument(
        "--shape",
        action="append",
        help="override the shape list, repeatable: 'b8 h16 n4096 d128' or '8,16,4096,128'",
    )
    ap.add_argument(
        "--profile",
        action="append",
        help="override the raggedness list, repeatable: the shortest sequence as a "
        "fraction of seq_len (1 = equal lengths)",
    )
    args = ap.parse_args()

    shapes = [parse_shape(s) for s in args.shape] if args.shape else SHAPES
    profiles = [parse_profile(p) for p in args.profile] if args.profile else PROFILES

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    cc = torch.cuda.get_device_capability(device.index)
    # the same table sageattn_varlen dispatches through
    backend = torch.ops.sageattention.plan(cc[0], cc[1], 128, None, None, None, None, True)[0]
    if args.backend and args.backend != backend:
        raise SystemExit(f"device resolves to backend {backend!r}, not {args.backend!r}")

    print(
        f"{torch.cuda.get_device_name(device)}  sm_{cc[0]}{cc[1]}  backend={backend}  {args.dtype}"
    )
    print(f"warmup={args.warmup} iters={args.iters} (median, round-robin sampling)\n")

    causals = [False, True] if args.causal == "both" else [args.causal == "1"]
    header = f"{'shape':<20}{'profile':<15}{'tokens':>16}{'varlen ms':>11}{'dense ms':>11}"
    rows = []
    for causal in causals:
        print(f"-- causal={int(causal)}")
        print(header + f"{'speedup':>9}{'varlen TF':>11}{'dense TF':>10}")
        for label, *shape in shapes:
            for profile, low in profiles:
                lens = seqlens(shape[0], shape[2], low)
                r = run_case(shape, lens, causal, device, dtype, args.warmup, args.iters)
                tokens = f"{r['tokens']}/{r['padded_tokens']}"
                print(
                    f"{label:<20}{profile:<15}{tokens:>16}"
                    f"{r['varlen_ms']:>11.3f}{r['dense_ms']:>11.3f}"
                    f"{r['speedup']:>9.2f}{r['varlen_tflops']:>11.1f}{r['dense_tflops']:>10.1f}"
                )
                rows.append(dict(shape=label, profile=profile, causal=int(causal), **r))
        print()

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
