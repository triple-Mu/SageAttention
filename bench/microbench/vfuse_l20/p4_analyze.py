#!/usr/bin/env python
"""Turn p4_fix.json + p4_fuseall.json into the report table.

fused    = composite rows from the threshold-raised build (fuseall)
separate = forced two-kernel rows from the product build (fix)
checks   = product-build composite dispatch per kv_len, and fused<=4096
           cross-build agreement.

Usage: p4_analyze.py <p4_fix.json> <p4_fuseall.json>
"""
import json
import sys


def load(p):
    with open(p) as f:
        return json.load(f)


def rows_by(blob, path):
    return {r["kv"]: r for r in blob["rows"] if r["path"] == path}


def main():
    fix = load(sys.argv[1])
    fuseall = load(sys.argv[2])
    print("meta fix:", fix["meta"]["gpu"], "L2 MB:", fix["meta"]["l2_bytes"] / 2**20,
          "SMs:", fix["meta"]["sm_count"])

    sep = rows_by(fix, "separate")
    comp_fix = rows_by(fix, "composite")
    comp_all = rows_by(fuseall, "composite")
    sep_all = rows_by(fuseall, "separate")

    print("\n| kv_len | fused us | separate us | fused/separate | fused wall | sep wall |")
    print("|---|---|---|---|---|---|")
    for kv in sorted(sep):
        f_us = comp_all[kv]["kernel_us_sum"]
        s_us = sep[kv]["kernel_us_sum"]
        print(f"| {kv} | {f_us:.1f} | {s_us:.1f} | {f_us / s_us:.3f} | "
              f"{comp_all[kv]['wall_us']:.1f} | {sep[kv]['wall_us']:.1f} |")

    print("\nper-kernel split (separate, product build):")
    for kv in sorted(sep):
        ks = sep[kv]["kernel_us"]
        print(f"  kv={kv:6d} " + "  ".join(f"{k}={v:.1f}" for k, v in sorted(ks.items())))

    print("\ndispatch check (product build composite):")
    for kv in sorted(comp_fix):
        seen = [k for k in ("TransposeQuantFp8Kernel", "TransposePadPermuteKernel",
                            "MeanScaleKernel") if k in comp_fix[kv]["kernel_us"]]
        print(f"  kv={kv:6d} -> {seen}")

    print("\ncross-build agreement:")
    print("  composite fused<=4096, fix vs fuseall (same code path):")
    for kv in sorted(comp_fix):
        if "TransposeQuantFp8Kernel" in comp_fix[kv]["kernel_us"]:
            a, b = comp_fix[kv]["kernel_us_sum"], comp_all[kv]["kernel_us_sum"]
            print(f"    kv={kv:6d} fix={a:.1f} fuseall={b:.1f} delta={(b - a) / a * 100:+.1f}%")
    print("  separate path, fix vs fuseall (identical kernels):")
    for kv in sorted(sep):
        a, b = sep[kv]["kernel_us_sum"], sep_all[kv]["kernel_us_sum"]
        print(f"    kv={kv:6d} fix={a:.1f} fuseall={b:.1f} delta={(b - a) / a * 100:+.1f}%")


if __name__ == "__main__":
    main()
