#!/usr/bin/env python3
"""Normalize a CUDA arch list for the SageAttention build.

Reads the arch list from --archs (may be empty), falling back to the
TORCH_CUDA_ARCH_LIST environment variable, then to local GPU detection
(nvidia-smi first, torch second). Prints one line: a ';'-separated list of
normalized entries like "8.0", "8.6", "8.9+PTX". Exits non-zero with a
message on stderr if nothing usable is found or a token is unparsable.

Token grammar (kept byte-compatible with the historical setup.py parser):
"8.0", "80", "sm_90", "compute_120", "8.9+PTX"; an 'a'/'f' suffix is dropped
with a warning (per-source arch suffixes are chosen by the build itself).
Entries below 8.0 are a hard error when given explicitly, and skipped with a
warning when they come from GPU detection.
"""
import argparse
import os
import re
import subprocess
import sys


def parse_arch_token(token):
    it = token.strip().lower().replace("sm_", "").replace("compute_", "")
    if not it:
        return None
    ptx = it.endswith("+ptx")
    if ptx:
        it = it[:-4]
    if it.endswith(("a", "f")):
        print(f"warning: ignoring {it[-1]!r} suffix on arch entry {token.strip()!r}: "
              "per-source arch suffixes are chosen automatically", file=sys.stderr)
        it = it[:-1]
    if it.isdigit() and len(it) >= 2:
        it = f"{it[:-1]}.{it[-1]}"
    match = re.fullmatch(r"(\d+)\.(\d+)", it)
    if match is None:
        raise ValueError(
            f"Unparsable arch entry {token.strip()!r} "
            "(expected forms like '8.0', '80', 'sm_90', '12.0+PTX')")
    return (int(match.group(1)), int(match.group(2))), ptx


def detect_local_ccs():
    """(cc, ptx=False) pairs from local GPUs; sub-8.0 skipped with a warning."""
    ccs = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            line = line.strip()
            if line:
                major, minor = line.split(".")
                ccs.append((int(major), int(minor)))
    except (OSError, subprocess.CalledProcessError, ValueError):
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    ccs.append(torch.cuda.get_device_capability(i))
        except ImportError:
            pass
    usable = []
    for cc in ccs:
        if cc < (8, 0):
            print(f"warning: skipping GPU with compute capability {cc[0]}.{cc[1]} "
                  "(SageAttention requires 8.0+)", file=sys.stderr)
        else:
            usable.append(cc)
    return usable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="")
    args = ap.parse_args()

    spec = args.archs.strip() or os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    requested = {}
    if spec:
        for item in spec.replace(",", ";").split(";"):
            parsed = parse_arch_token(item)
            if parsed is None:
                continue
            cc, ptx = parsed
            if cc < (8, 0):
                print(f"error: compute capability {cc[0]}.{cc[1]} is not supported: "
                      "SageAttention requires 8.0 or higher.", file=sys.stderr)
                return 1
            requested[cc] = requested.get(cc, False) or ptx
    else:
        for cc in detect_local_ccs():
            requested.setdefault(cc, False)

    if not requested:
        print("error: no target compute capabilities. Set TORCH_CUDA_ARCH_LIST "
              "or build on a machine with supported GPUs.", file=sys.stderr)
        return 1

    entries = [f"{cc[0]}.{cc[1]}" + ("+PTX" if ptx else "")
               for cc, ptx in sorted(requested.items())]
    print(";".join(entries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
