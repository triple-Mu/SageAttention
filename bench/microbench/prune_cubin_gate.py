#!/usr/bin/env python3
"""SASS gate for SAGE_PRUNE_GENCODE: prove pruning only drops whole cubins.

Usage:
    cuobjdump -xelf all <off>/_C.abi3.so   (into one dir per build)
    cuobjdump -xelf all <on>/_C.abi3.so
    python prune_cubin_gate.py <off_cubin_dir> <on_cubin_dir>

Pairs the extracted cubins by (source TU, sm arch, product kernel symbol set)
and requires every surviving pair to be sha256-identical, or -- when the TU's
compilation identity legitimately changed -- identical as nvdisasm text after
scrubbing the two name-only build-identity tokens:

  1. nvcc's ``_INTERNAL_<modid>_<len>_<tu>_cu_<hash>`` local-symbol prefix;
     the hash covers the preprocessed input and the gencode list, so removing
     a --generate-code entry renames every local symbol of the TU's other,
     otherwise bit-identical cubins.
  2. CUB/CCCL's arch-list inline namespace ``_V_<ver>_SM_<list>`` (minted per
     compiled arch set precisely so mixed-arch TUs cannot ODR-collide).

Exit 0 when no pair has a real SASS difference and the ON set introduces no
new cubin; the pruned-cubin list is printed for the report.
"""
import re
import subprocess
import sys

CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"
NVDISASM = "/usr/local/cuda/bin/nvdisasm"

TU_RE = re.compile(r"_INTERNAL_[0-9a-f]{8}_\d+_([A-Za-z0-9_]+?)_cu_[0-9a-f]{8}")
SCRUB_RES = [
    (re.compile(r"_INTERNAL_[0-9a-f]{8}_(\d+_[A-Za-z0-9_]+?_cu)_[0-9a-f]{8}"), r"_INTERNAL_x_\1_x"),
    (re.compile(r"\d+_V_\d+_SM(?:_\d+)+"), r"_V_x_SM_x"),
]


def sha256(path):
    out = subprocess.run(["sha256sum", path], capture_output=True, text=True, check=True)
    return out.stdout.split()[0]


def identity(path):
    """(tu, arch, product kernel names) -- stable across the prune."""
    out = subprocess.run([CUOBJDUMP, "-symbols", path], capture_output=True, text=True).stdout
    m = TU_RE.search(out)
    tu = m.group(1) if m else "<unknown>"
    # product kernels only: cub's EmptyKernel name embeds the arch list and
    # would defeat the pairing; sage kernels also carry the per-object-library
    # namespace (sm89 vs sm120) that tells the twin compilations apart.
    funcs = frozenset(
        line.split()[-1]
        for line in out.splitlines()
        if line.startswith("STT_FUNC") and "STB_GLOBAL" in line and "_ZN4sage" in line
    )
    arch = re.search(r"sm_[0-9]+a?", path).group(0)
    return tu, arch, funcs


def load(cubin_dir):
    rows = {}
    ls = subprocess.run(["ls", cubin_dir], capture_output=True, text=True, check=True)
    for fname in ls.stdout.split():
        if not fname.endswith(".cubin"):
            continue
        path = f"{cubin_dir}/{fname}"
        key = identity(path)
        assert key not in rows, f"cannot tell {fname} and {rows[key][0]} apart"
        rows[key] = (fname, sha256(path), path)
    return rows


def norm_sass(path):
    out = subprocess.run([NVDISASM, "-c", path], capture_output=True, text=True).stdout
    for rx, repl in SCRUB_RES:
        out = rx.sub(repl, out)
    return out


def main():
    off, on = load(sys.argv[1]), load(sys.argv[2])
    byte_same, norm_same, real_diff, pruned = [], [], [], []
    for key in sorted(off, key=lambda k: (k[0], k[1])):
        if key not in on:
            pruned.append(key)
            continue
        _, so, po = off[key]
        _, sn, pn = on[key]
        if so == sn:
            byte_same.append(key)
        elif norm_sass(po) == norm_sass(pn):
            norm_same.append(key)
        else:
            real_diff.append(key)
    on_only = sorted((k[0], k[1]) for k in on if k not in off)

    print(f"pairs byte-identical:             {len(byte_same)}")
    print(f"pairs identical after hash-scrub: {len(norm_same)}")
    for tu, arch, _ in norm_same:
        print(f"    {arch:8s} {tu}")
    print(f"pairs with REAL SASS diff:        {len(real_diff)}")
    for tu, arch, _ in real_diff:
        print(f"    !! {arch:8s} {tu}")
    print(f"pruned (in OFF only):             {len(pruned)}")
    for tu, arch, _ in sorted(pruned, key=lambda k: (k[0], k[1])):
        print(f"    {arch:8s} {tu}")
    print(f"ON-only (must be none):           {on_only or 'none'}")
    return 1 if real_diff or on_only else 0


if __name__ == "__main__":
    sys.exit(main())
