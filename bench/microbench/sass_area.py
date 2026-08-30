# SPDX-License-Identifier: Apache-2.0
# E1/P3 prescreen: static instruction-area stats over cuobjdump SASS.
#
# Per kernel:
#   - total SASS instructions (NOP excluded)
#   - MUFU.EX2 (f32) / MUFU.EX2.F16 counts
#   - producer-opcode histogram for MUFU.EX2 inputs (identifies the FFMA that
#     feeds each exp2; def-use is a nearest-preceding-writer scan)
#   - E1 gate metric: (MUFU.EX2 + feeder FFMA + FMAX) / total
#   - P3 gate metric (--pairs): share of f32 ALU instructions (FFMA/FADD/FMUL/
#     FMAX) that sit in an adjacent same-opcode independent pair, i.e. what a
#     packed f32x2 rewrite could merge in the best static case
#
# Usage: python sass_area.py [--pairs] cubin_or_sass.txt ...
#        (a .cubin argument is piped through `cuobjdump -sass`)

import argparse
import re
import subprocess
import sys
from collections import Counter

INSTR_RE = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(@!?U?P\d+\s+)?(.+?)\s*;")
FUNC_RE = re.compile(r"^\s*Function : (\S+)")

# sm_100a PTX has add/mul/fma.rn.f32x2 (-> FADD2/FMUL2/FFMA2 SASS) but no
# packed max/min (ptxas 13.3 rejects max.f32x2), so FMNMX(3) is not packable.
PACKABLE = {"FFMA", "FADD", "FMUL"}


def parse_sass(text):
    """[(kernel_name, [instr_str, ...])]; instr_str like 'FFMA R4, R2, R3, R4'."""
    kernels = []
    cur = None
    for line in text.splitlines():
        m = FUNC_RE.match(line)
        if m:
            cur = (m.group(1), [])
            kernels.append(cur)
            continue
        m = INSTR_RE.search(line)
        if m and cur is not None:
            cur[1].append(m.group(2).strip())
    return kernels


def opcode(instr):
    return instr.split(None, 1)[0]


def dest_reg(instr):
    """First operand if it looks like a plain register (good enough for ALU)."""
    parts = instr.split(None, 1)
    if len(parts) < 2:
        return None
    op = parts[0]
    if op.split(".")[0] in ("ST", "STG", "STS", "STL", "RED", "BRA", "EXIT", "BAR", "ATOM",
                            "ATOMG", "CCTL", "MEMBAR", "DEPBAR", "FENCE", "RET", "BSSY",
                            "BSYNC", "WARPSYNC", "ERRBAR"):
        return None
    first = parts[1].split(",")[0].strip()
    m = re.fullmatch(r"(R\d+)", first)
    return m.group(1) if m else None


def src_regs(instr):
    parts = instr.split(None, 1)
    if len(parts) < 2:
        return []
    ops = [o.strip() for o in parts[1].split(",")]
    regs = []
    for o in ops[1:]:  # skip dest
        m = re.search(r"R\d+", o)
        if m:
            regs.append(m.group(0))
    return regs


def analyze(name, instrs, want_pairs):
    body = [i for i in instrs if opcode(i) != "NOP"]
    ops = Counter(opcode(i).split(".")[0] for i in body)
    full = Counter(opcode(i) for i in body)
    total = len(body)

    mufu_ex2_f32 = sum(c for o, c in full.items() if o.startswith("MUFU.EX2") and ".F16" not in o)
    mufu_ex2_f16 = sum(c for o, c in full.items() if o.startswith("MUFU.EX2.F16"))
    mufu_other = ops.get("MUFU", 0) - mufu_ex2_f32 - mufu_ex2_f16

    # nearest-preceding-writer scan for each MUFU.EX2 input
    writer_of = {}
    feeders = Counter()
    for i, instr in enumerate(body):
        op = opcode(instr)
        if op.startswith("MUFU.EX2"):
            for s in src_regs(instr):
                w = writer_of.get(s)
                feeders[opcode(w).split(".")[0] if w else "<param/none>"] += 1
        d = dest_reg(instr)
        if d:
            writer_of[d] = instr

    fmax = ops.get("FMNMX", 0) + ops.get("FMNMX3", 0)
    ffma_feed = feeders.get("FFMA", 0)
    e1_area = mufu_ex2_f32 + mufu_ex2_f16 + ffma_feed + fmax

    print(f"\n== {name}")
    print(f"  total instrs (no NOP): {total}")
    print(f"  MUFU.EX2 f32: {mufu_ex2_f32}   MUFU.EX2.F16: {mufu_ex2_f16}   "
          f"other MUFU: {mufu_other}")
    print(f"  MUFU.EX2 input producers: {dict(feeders)}")
    print(f"  FMNMX(+3) (fmax): {fmax}   FFMA: {ops.get('FFMA', 0)}   "
          f"FMUL: {ops.get('FMUL', 0)}   FADD: {ops.get('FADD', 0)}")
    print(f"  E1 XU-chain area = (EX2 {mufu_ex2_f32 + mufu_ex2_f16} + feederFFMA {ffma_feed} "
          f"+ FMAX {fmax}) / {total} = {100.0 * e1_area / total:.2f}%")

    if want_pairs:
        paired = Counter()
        i = 0
        while i + 1 < len(body):
            a, b = body[i], body[i + 1]
            oa, ob = opcode(a).split(".")[0], opcode(b).split(".")[0]
            if oa == ob and oa in PACKABLE:
                da, db = dest_reg(a), dest_reg(b)
                if da and db and da != db and da not in src_regs(b):
                    paired[oa] += 2
                    i += 2
                    continue
            i += 1
        n_paired = sum(paired.values())
        f32_total = sum(ops.get(o, 0) for o in PACKABLE)
        print(f"  P3 packable: {n_paired} of {f32_total} f32 ALU instrs in adjacent independent "
              f"same-op pairs = {100.0 * n_paired / total:.2f}% of kernel "
              f"(saves {100.0 * n_paired / 2 / total:.2f}% if merged); by op {dict(paired)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", action="store_true", help="also compute P3 f32x2 pairing stats")
    ap.add_argument("inputs", nargs="+")
    args = ap.parse_args()
    for path in args.inputs:
        if path.endswith(".cubin"):
            text = subprocess.run(["cuobjdump", "-sass", path], check=True,
                                  capture_output=True, text=True).stdout
        else:
            text = open(path).read()
        print(f"### {path}")
        for name, instrs in parse_sass(text):
            analyze(name, instrs, args.pairs)


if __name__ == "__main__":
    sys.exit(main())
