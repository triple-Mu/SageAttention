# SPDX-License-Identifier: Apache-2.0
# Track D static audit over a built _C.abi3.so (P2 baseline):
#   1. dense attention kernels: param layout carries no cu_seqlens-style
#      pointer (KPARAM_INFO table; the mangled name has no PKi either), and
#      the LDG census shows no extra pointer-chasing loads vs the varlen twin
#      -> SeqlenInfo<false> is fully folded.
#   2. quant/fused kernels: size of the `cu_seqlens != nullptr` sentinel block
#      (the varlen-only code a dense launch jumps over), auto-detected as the
#      null test of an 8-byte param + the @!P BRA guarding it.
#
# Usage: python so_param_audit.py /path/to/_C.abi3.so   (read-only)

import re
import subprocess
import sys
from collections import Counter

INSTR_RE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\d+\s+)?(.+?)\s*;")
FUNC_RE = re.compile(r"^\s*Function : (\S+)")
PARAM_BASE = 0x160  # sm8x kernel param base in c[0x0]


def run(*cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def parse_sass(text):
    """{mangled: [(addr, pred, instr)]}"""
    out = {}
    cur = None
    for line in text.splitlines():
        m = FUNC_RE.match(line)
        if m:
            cur = []
            out[m.group(1)] = cur
            continue
        m = INSTR_RE.search(line)
        if m and cur is not None:
            cur.append((int(m.group(1), 16), (m.group(2) or "").strip(), m.group(3).strip()))
    return out


def parse_params(elf_text):
    """{mangled: [(ordinal, offset, size)]} from EIATTR_KPARAM_INFO."""
    out = {}
    cur = None
    for line in elf_text.splitlines():
        m = re.fullmatch(r"\.nv\.info\.(\S+)", line.strip())
        if m:
            cur = out.setdefault(m.group(1), [])
            continue
        m = re.search(r"Ordinal\s*:\s*(0x[0-9a-f]+)\s*Offset\s*:\s*(0x[0-9a-f]+)\s*"
                      r"Size\s*:\s*(0x[0-9a-f]+)", line)
        if m and cur is not None:
            cur.append(tuple(int(g, 16) for g in m.groups()))
    return out


def fmt_params(params):
    return " ".join(f"{o}:+{off:#x}/{sz}B" for o, off, sz in sorted(params))


def audit_attn_pair(name_args, dense, varlen, dparams, vparams):
    print(f"\n== attn pair <...{name_args[-60:]}>")
    print(f"  dense : {len(dense)} instrs, {len(dparams)} params: {fmt_params(dparams)}")
    print(f"  varlen: {len(varlen)} instrs, {len(vparams)} params: {fmt_params(vparams)}")
    dsz = sum(p[2] for p in dparams)
    vsz = sum(p[2] for p in vparams)
    print(f"  param bytes dense={dsz} varlen={vsz} (delta {vsz - dsz}; expected +20 = "
          "2x8B cu_seqlens ptrs + 3x4B strides - 2x4B qo/kv_len)")
    for tag, body in (("dense", dense), ("varlen", varlen)):
        ldg = Counter(i.split(None, 1)[0] for _, _, i in body if i.startswith("LDG"))
        print(f"  {tag} LDG census: {dict(ldg)}")


def audit_sentinel(sym, body, params):
    """Find a null test of an 8-byte param and measure the block it guards."""
    slots = {}
    for _, off, sz in params:
        if sz == 8:
            slots[PARAM_BASE + off] = off
            slots[PARAM_BASE + off + 4] = off
    test = None
    for addr, _, instr in body:
        if instr.startswith("ISETP.NE") and " RZ, c[0x0][" in instr:
            m = re.search(r"c\[0x0\]\[(0x[0-9a-f]+)\]", instr)
            if m and int(m.group(1), 16) in slots:
                test = (addr, slots[int(m.group(1), 16)])
                break
    total = len(body)
    if test is None:
        print(f"    total {total:4d}  no pointer null-test found")
        return
    bra = next(((a, i) for a, p, i in body
                if a > test[0] and p.startswith("@!") and i.startswith("BRA")), None)
    if bra is None:
        print(f"    total {total:4d}  null-test of param@+{test[1]:#x} predicated, no BRA")
        return
    target = int(bra[1].split()[-1], 16)
    block = [x for x in body if bra[0] < x[0] < target]
    print(f"    total {total:4d}  varlen-only block {len(block):3d} instrs "
          f"({100.0 * len(block) / total:.1f}%)  [param@+{test[1]:#x}, "
          f"skip {bra[0]:#x}->{target:#x}; dense path pays 2 ISETP + 1 BRA]")


def main():
    so = sys.argv[1]
    sass = parse_sass(run("cuobjdump", "-sass", so))
    params = parse_params(run("cuobjdump", "-elf", so))

    # ---- 1. dense vs varlen attention pairs (3 samples) ----
    args_re = re.compile(r"attn_kernelI(.*)EEv")
    dense_by_args, varlen_by_args = {}, {}
    for sym in sass:
        m = args_re.search(sym)
        if not m:
            continue
        (varlen_by_args if "varlen" in sym else dense_by_args)[m.group(1)] = sym
    picks = sorted(a for a in varlen_by_args if a in dense_by_args)
    for a in picks[:3]:
        d, v = dense_by_args[a], varlen_by_args[a]
        audit_attn_pair(a, sass[d], sass[v], params[d], params[v])

    # ---- 2. quant/fused kernels: sentinel block sizes (P2 baseline) ----
    fams = ["QuantPerThreadQInt8Kernel", "QuantPerThreadKInt8Kernel", "QuantInt8Kernel",
            "TransposeQuantFp8Kernel", "SubMeanKernel", "TransposePadPermuteKernel",
            "MeanScaleKernel"]
    print("\n== quant/fused cu_seqlens sentinel blocks (P2 static baseline)")
    for fam in fams:
        syms = sorted(s for s in sass if fam in s and "__half" in s) \
            or sorted(s for s in sass if fam in s)
        print(f" {fam}:")
        for sym in syms[:2]:
            audit_sentinel(sym, sass[sym], params.get(sym, []))


if __name__ == "__main__":
    main()
