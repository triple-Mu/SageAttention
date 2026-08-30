#!/usr/bin/env python
"""Merge the eight A100 P8 kernel_breakdown JSONs (base/fix x r1..r4) into
per-shape attention-kernel ratios and per-seq_len aggregates.

ratio = base / fix per round pair, > 1 means fix is faster. Rounds 1-2 ran
base-first, rounds 3-4 fix-first, so the 4-pair geometric mean is balanced
against the monotonic thermal drift this A100 shows. The causal group is the
SASS-byte-identical control; its ratio is the residual drift of that pairing
order, and 'ctrl-corrected' divides the non-causal aggregate by it.

Usage: p8_merge8.py <dir with {base,fix}_r{1..4}/kb_*.json[.gz]> <out.csv>
"""
import csv
import glob
import gzip
import json
import math
import sys
from pathlib import Path

ROUNDS = (1, 2, 3, 4)


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


def gm(vals):
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def main():
    root, out_csv = sys.argv[1], sys.argv[2]
    sides = {}
    for var in ("base", "fix"):
        for r in ROUNDS:
            sides[(var, r)] = attn_map(load_one(Path(root) / f"{var}_r{r}"))

    shape_ids = sorted(sides[("fix", 1)], key=lambda s: (sides[("fix", 1)][s][1]["s"], s))
    rows = []
    for sid in shape_ids:
        rec = sides[("fix", 1)][sid][1]
        vals = {f"{var}_r{r}": sides[(var, r)][sid][0] for var in ("base", "fix") for r in ROUNDS}
        ratios = {f"ratio_r{r}": vals[f"base_r{r}"] / vals[f"fix_r{r}"] for r in ROUNDS}
        rows.append(
            dict(
                shape_id=sid,
                group=rec["group"],
                s=rec["s"],
                causal=rec["causal"],
                **{k: round(v, 2) for k, v in vals.items()},
                **{k: round(v, 4) for k, v in ratios.items()},
                ratio_gm=round(gm(list(ratios.values())), 4),
            )
        )

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def agg_ratios(sel):
        per_round = []
        for r in ROUNDS:
            base = sum(x[f"base_r{r}"] for x in sel)
            fix = sum(x[f"fix_r{r}"] for x in sel)
            per_round.append(base / fix)
        return per_round

    def show(sel, label):
        if not sel:
            return None
        pr = agg_ratios(sel)
        print(
            f"{label:>22s} n={len(sel):2d} "
            + " ".join(f"r{r}={v:.4f}" for r, v in zip(ROUNDS, pr))
            + f"  gm12={gm(pr[:2]):.4f} gm34={gm(pr[2:]):.4f} gm_all={gm(pr):.4f}"
        )
        return pr

    nc = [x for x in rows if not x["causal"]]
    ca = [x for x in rows if x["causal"]]
    for s in sorted({x["s"] for x in nc}):
        show([x for x in nc if x["s"] == s], f"s={s}")
    pr_nc = show(nc, "non-causal total")
    pr_ca = show(ca, "causal control")
    corr = [n / c for n, c in zip(pr_nc, pr_ca)]
    print(
        f"{'ctrl-corrected nc':>22s} n={len(nc):2d} "
        + " ".join(f"r{r}={v:.4f}" for r, v in zip(ROUNDS, corr))
        + f"  gm12={gm(corr[:2]):.4f} gm34={gm(corr[2:]):.4f} gm_all={gm(corr):.4f}"
    )
    worst = min(nc, key=lambda x: x["ratio_gm"])
    best = max(nc, key=lambda x: x["ratio_gm"])
    print(f"non-causal per-shape ratio_gm: min {worst['ratio_gm']} ({worst['shape_id']}), "
          f"max {best['ratio_gm']} ({best['shape_id']})")
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
