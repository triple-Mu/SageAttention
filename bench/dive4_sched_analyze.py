#!/usr/bin/env python
"""Reduce dive4_sched_driver.py --mode matrix output to the waste table.

Reads the JSONL (one record per config, skew records paired with their eqtok
control), prints one row per (shape, causal, profile):

  raw%   = 1 - attn_eqtok / attn_skew
           the task's prescribed upper bound: same total tokens, equal lengths.
           Inflated by the FLOP mismatch (a skewed batch does more KV-tile work
           than the equal batch of the same tokens, Jensen), so also:
  norm%  = 1 - attn_eqtok * (tiles_skew / tiles_eqtok) / attn_skew
           the eqtok control re-scaled to the skewed profile's exact KV-tile
           count, i.e. the time an equally-efficient perfectly-packed kernel
           would need for the skewed work. This is the scheduling waste.
  empty% = launched CTAs that early-exit (grid opened to max_seqlen).

    python bench/dive4_sched_analyze.py matrix.jsonl
"""

import json
import sys


def main():
    recs = []
    for line in open(sys.argv[1]):
        line = line.strip()
        if line.startswith("{"):
            recs.append(json.loads(line))

    by_key = {}
    for r in recs:
        by_key[(r["shape"], r["causal"], r["profile"], r["kind"])] = r

    header = (
        f"{'shape':<20}{'causal':>7}{'profile':<16}{'attn µs':>9}{'eqtok µs':>10}"
        f"{'raw%':>7}{'norm%':>7}{'empty%':>8}{'tiles k':>9}{'e2e µs':>8}"
    )
    print(header)
    last = None
    for (shape, causal, profile, kind), r in sorted(
        by_key.items(), key=lambda kv: (kv[0][1], kv[0][0], -kv[1]["tokens"])
    ):
        if kind != "skew":
            continue
        if last is not None and last != (shape, causal):
            print()
        last = (shape, causal)
        empty = 100.0 * (1 - r["ctas_useful"] / r["ctas_launched"])
        row = (
            f"{shape:<20}{causal:>7}{profile:<16}{r['attn_us']:>9.1f}"
        )
        if profile == "equal":
            print(row + f"{'--':>10}{'--':>7}{'--':>7}{empty:>8.1f}{r['kv_tiles']/1e3:>9.1f}{r['e2e_ms']*1e3:>8.0f}")
            continue
        eq = by_key[(shape, causal, profile, "eqtok")]
        raw = 100.0 * (1 - eq["attn_us"] / r["attn_us"])
        ideal = eq["attn_us"] * r["kv_tiles"] / eq["kv_tiles"]
        norm = 100.0 * (1 - ideal / r["attn_us"])
        print(
            row
            + f"{eq['attn_us']:>10.1f}{raw:>7.1f}{norm:>7.1f}{empty:>8.1f}"
            + f"{r['kv_tiles']/1e3:>9.1f}{r['e2e_ms']*1e3:>8.0f}"
        )


if __name__ == "__main__":
    main()
