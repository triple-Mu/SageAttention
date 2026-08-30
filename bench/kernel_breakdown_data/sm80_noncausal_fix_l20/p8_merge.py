#!/usr/bin/env python
"""Merge the four P8 kernel_breakdown JSONs (base/fix x r1/r2) into per-shape
attention-kernel ratios and the per-seq_len aggregate table.

ratio = base / fix computed per round (paired), > 1 means fix is faster.
Aggregates are ratios of per-group sums (the SM89 report's convention).

Usage: p8_merge.py <dir with {base,fix}_r{1,2}/kb_*.json[.gz]> <out.csv>
"""
import csv
import glob
import gzip
import json
import math
import sys
from pathlib import Path


def load_one(d):
    files = sorted(glob.glob(str(Path(d) / "kb_*.json*")))
    assert len(files) == 1, (d, files)
    f = files[0]
    op = gzip.open if f.endswith(".gz") else open
    with op(f, "rt") as fh:
        return json.load(fh)


def attn_map(blob):
    out = {}
    for rec in blob["records"]:
        if rec["status"] != "ok":
            continue
        out[rec["shape_id"]] = (rec["roles"].get("attention", float("nan")), rec)
    return out


def main():
    root, out_csv = sys.argv[1], sys.argv[2]
    sides = {}
    for var in ("base", "fix"):
        for r in (1, 2):
            sides[(var, r)] = attn_map(load_one(Path(root) / f"{var}_r{r}"))

    shape_ids = sorted(sides[("fix", 1)], key=lambda s: (sides[("fix", 1)][s][1]["s"], s))
    rows = []
    for sid in shape_ids:
        rec = sides[("fix", 1)][sid][1]
        vals = {f"{var}_r{r}": sides[(var, r)][sid][0] for var in ("base", "fix") for r in (1, 2)}
        r1 = vals["base_r1"] / vals["fix_r1"]
        r2 = vals["base_r2"] / vals["fix_r2"]
        rows.append(
            dict(
                shape_id=sid,
                group=rec["group"],
                s=rec["s"],
                causal=rec["causal"],
                **{k: round(v, 2) for k, v in vals.items()},
                ratio_r1=round(r1, 4),
                ratio_r2=round(r2, 4),
                ratio_gm=round(math.sqrt(r1 * r2), 4),
            )
        )

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def agg(sel, label):
        if not sel:
            return
        per_round = []
        for r in (1, 2):
            base = sum(x[f"base_r{r}"] for x in sel)
            fix = sum(x[f"fix_r{r}"] for x in sel)
            per_round.append(base / fix)
        gm = math.sqrt(per_round[0] * per_round[1])
        print(
            f"{label:>22s} n={len(sel):2d} r1={per_round[0]:.4f} r2={per_round[1]:.4f} "
            f"mean={gm:.4f}"
        )

    nc = [x for x in rows if not x["causal"]]
    for s in sorted({x["s"] for x in nc}):
        agg([x for x in nc if x["s"] == s], f"s={s}")
    agg(nc, "non-causal total")
    agg([x for x in rows if x["causal"]], "causal total")
    worst = min(nc, key=lambda x: x["ratio_gm"])
    best = max(nc, key=lambda x: x["ratio_gm"])
    print(f"non-causal per-shape ratio_gm: min {worst['ratio_gm']} ({worst['shape_id']}), "
          f"max {best['ratio_gm']} ({best['shape_id']})")
    spread = [abs(x["ratio_r1"] - x["ratio_r2"]) for x in rows]
    print(f"round spread |r1-r2|: median {sorted(spread)[len(spread)//2]:.4f} max {max(spread):.4f}")
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
