#!/usr/bin/env python
"""P6 PDL prescreen analyzer: inter-kernel gap share of the GPU e2e window.

Input is a GPU timeline from bench/p6_pdl_driver.py, either

* an nsys CSV: ``nsys stats --report cuda_gpu_trace --format csv``, or
* a kineto chrome trace (``--kineto`` mode of the driver).

Method: keep kernels + memsets (drop memcpy rows), sort by start, take the
kernel name of the last row as the per-call terminator (the attention kernel
is the last launch of every ``sageattn`` call), split the trace into calls,
drop the first and last call (head/tail), then over the remaining window::

    gap_total = window - union(busy intervals)
    intra-call gap = gaps not following an attention kernel
    inter-call gap = gaps following an attention kernel (call boundary: python
                     overhead when the launch queue drains, otherwise the
                     scheduling latency of the next call's first GPU op)

Prints per-kernel totals, per-transition gap means, and one JSON summary line.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict


def short_name(name):
    if name.startswith("[CUDA"):
        return name.strip("[]").replace("CUDA ", "")
    s = name
    if s.startswith("void "):
        s = s[5:]
    s = s.replace("<unnamed>::", "")
    cut = len(s)
    for ch in "<(":
        i = s.find(ch)
        if i != -1:
            cut = min(cut, i)
    s = s[:cut].strip()
    return s.split("::")[-1] or name[:40]


def load_nsys_csv(path):
    ops = []
    with open(path, newline="") as f:
        # nsys may prepend log lines before the header
        rows = [r for r in csv.reader(f) if r]
    hdr_i = next(i for i, r in enumerate(rows) if any(c.startswith("Start") for c in r))
    hdr = rows[hdr_i]
    col = {}
    for want in ("Start", "Duration", "Strm", "Name"):
        col[want] = next(i for i, c in enumerate(hdr) if c.startswith(want))
    for r in rows[hdr_i + 1 :]:
        try:
            start = int(r[col["Start"]].replace(",", ""))
            dur = int(r[col["Duration"]].replace(",", ""))
        except ValueError:
            continue
        ops.append((start, dur, r[col["Strm"]], r[col["Name"]]))
    return ops


def load_kineto_json(path):
    with open(path) as f:
        ev = json.load(f)["traceEvents"]
    ops = []
    for e in ev:
        cat = e.get("cat", "")
        if cat not in ("kernel", "gpu_memset", "gpu_memcpy"):
            continue
        name = e["name"] if cat == "kernel" else f"[CUDA {cat[4:]}]"
        # kineto ts/dur are microseconds
        ops.append((int(e["ts"] * 1000), int(e["dur"] * 1000), str(e.get("args", {}).get("stream", "?")), name))
    return ops


def union_busy(intervals):
    busy = 0
    cur_s, cur_e = intervals[0]
    overlap = 0
    for s, e in intervals[1:]:
        if s <= cur_e:
            overlap += min(e, cur_e) - s
            cur_e = max(cur_e, e)
        else:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
    busy += cur_e - cur_s
    return busy, overlap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", help=".csv (nsys cuda_gpu_trace) or .json (kineto)")
    ap.add_argument("--iters", type=int, default=None, help="expected call count (sanity check)")
    ap.add_argument("--label", type=str, default="")
    args = ap.parse_args()

    if args.trace.endswith(".json"):
        ops = load_kineto_json(args.trace)
    else:
        ops = load_nsys_csv(args.trace)
    ops.sort(key=lambda t: t[0])

    memcpy = [o for o in ops if "memcpy" in o[3]]
    ops = [o for o in ops if "memcpy" not in o[3]]
    if not ops:
        sys.exit("no GPU ops in trace")
    streams = sorted({o[2] for o in ops})

    att_name = next(o[3] for o in reversed(ops) if not o[3].startswith("[CUDA"))
    # split into calls, each ending right after an attention kernel
    calls, cur = [], []
    for o in ops:
        cur.append(o)
        if o[3] == att_name:
            calls.append(cur)
            cur = []
    n_calls = len(calls)
    if n_calls < 4:
        sys.exit(f"only {n_calls} calls found; need >= 4")
    body = [o for c in calls[1:-1] for o in c]  # drop head/tail call

    w_start, w_end = body[0][0], max(o[0] + o[1] for o in body)
    window = w_end - w_start
    busy, overlap = union_busy([(o[0], o[0] + o[1]) for o in body])
    gap_total = window - busy

    # gap attribution by transition (prev kernel -> next kernel)
    trans = defaultdict(lambda: [0, 0])  # (prev, nxt) -> [count, total_ns]
    inter_call_gap = 0
    prev_end, prev_name = body[0][0] + body[0][1], body[0][3]
    for s, d, _, name in body[1:]:
        g = s - prev_end
        if g > 0:
            trans[(short_name(prev_name), short_name(name))][0] += 1
            trans[(short_name(prev_name), short_name(name))][1] += g
            if prev_name == att_name:
                inter_call_gap += g
        prev_end, prev_name = max(prev_end, s + d), name
    intra_call_gap = gap_total - inter_call_gap

    per_kernel = defaultdict(lambda: [0, 0])
    for s, d, _, name in body:
        per_kernel[short_name(name)][0] += 1
        per_kernel[short_name(name)][1] += d

    lbl = args.label or args.trace
    print(f"== {lbl} ==")
    print(
        f"calls kept {len(calls) - 2}/{n_calls} (expected {args.iters}), "
        f"streams {streams}, memcpy rows dropped {len(memcpy)}, overlap {overlap / 1e3:.1f} us"
    )
    print(
        f"window {window / 1e6:.3f} ms  busy {busy / 1e6:.3f} ms  "
        f"gap {gap_total / 1e6:.3f} ms ({100 * gap_total / window:.2f}%)  "
        f"intra-call {intra_call_gap / 1e6:.3f} ms ({100 * intra_call_gap / window:.2f}%)  "
        f"inter-call {inter_call_gap / 1e6:.3f} ms ({100 * inter_call_gap / window:.2f}%)"
    )
    print("-- per-kernel (over kept window) --")
    for name, (cnt, tot) in sorted(per_kernel.items(), key=lambda kv: -kv[1][1]):
        print(f"{name:44s} n={cnt:5d} total={tot / 1e6:9.3f} ms mean={tot / cnt / 1e3:8.2f} us ({100 * tot / window:5.2f}%)")
    print("-- gaps by transition --")
    for (a, b), (cnt, tot) in sorted(trans.items(), key=lambda kv: -kv[1][1]):
        print(f"{a} -> {b}: n={cnt} total={tot / 1e3:.1f} us mean={tot / cnt / 1e3:.2f} us")
    print(
        "JSON "
        + json.dumps(
            {
                "label": lbl,
                "calls": len(calls) - 2,
                "window_ms": window / 1e6,
                "gap_pct": 100 * gap_total / window,
                "intra_call_gap_pct": 100 * intra_call_gap / window,
                "inter_call_gap_pct": 100 * inter_call_gap / window,
                "att_kernel": short_name(att_name),
            }
        )
    )


if __name__ == "__main__":
    main()
