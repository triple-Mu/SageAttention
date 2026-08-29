#!/usr/bin/env bash
#
# Per-kernel SASS / ptxas-resource diff between two checkouts of this repo.
#
# The refactoring gate: a change that only moves code must leave every device
# kernel byte-identical. Incremental builds do not answer that question (the
# sm90 objects come out different when ninja reuses a stale build tree), so
# both sides are always configured and built from scratch here.
#
# usage:
#   tools/sass_diff.sh [options] <tree_a> <tree_b>
#
#   <tree_a> <tree_b>   two source checkouts, e.g. made with
#                       `git worktree add /tmp/wt_base <ref>`
#
# options:
#   -o DIR   working directory for the two build trees and the dumps
#            (default: a fresh mktemp -d)
#   -a LIST  compute capabilities, cmake list syntax
#            (default: 8.6;8.9;9.0;10.0;12.0 -- every kernel family)
#   -j N     ninja jobs (default 4)
#   -t N     nvcc --threads (default 4).  j*t is the real concurrency; over
#            ~16 the compile OOMs on a 32 GB box.
#
# env: PYTHON / CMAKE / NINJA override the tool paths.
#
# exit 0 iff, for every object, every kernel has the same instruction stream,
# the same opcode histogram and the same ptxas resource line.  Mangled symbol
# names are stripped before comparing (a rename is not a codegen change), so
# kernels are paired by order of appearance inside each (object, arch) section.
set -euo pipefail

ARCHS="8.6;8.9;9.0;10.0;12.0"
JOBS=4
NVCC_THREADS=4
OUT=""
while getopts "o:a:j:t:" opt; do
  case "$opt" in
    o) OUT="$OPTARG" ;;
    a) ARCHS="$OPTARG" ;;
    j) JOBS="$OPTARG" ;;
    t) NVCC_THREADS="$OPTARG" ;;
    *) sed -n '2,30p' "$0" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
if [ $# -ne 2 ]; then sed -n '2,30p' "$0" >&2; exit 2; fi

TREE_A=$(realpath "$1")
TREE_B=$(realpath "$2")
OUT=${OUT:-$(mktemp -d)}
mkdir -p "$OUT"
PYTHON=${PYTHON:-$(command -v python3 || command -v python)}
CMAKE=${CMAKE:-$(command -v cmake)}
NINJA=${NINJA:-$(command -v ninja)}

build() {  # build <tree> <name>
  local tree=$1 name=$2 dir="$OUT/build_$2"
  rm -rf "$dir"
  echo "[sass_diff] configuring $name ($tree)"
  "$CMAKE" -S "$tree" -B "$dir" -G Ninja \
    -DCMAKE_MAKE_PROGRAM="$NINJA" \
    -DPython_EXECUTABLE="$PYTHON" \
    -DSAGE_CUDA_ARCHS="$ARCHS" \
    -DSAGE_NVCC_THREADS="$NVCC_THREADS" > "$OUT/cmake_$name.log" 2>&1
  echo "[sass_diff] building $name"
  "$CMAKE" --build "$dir" --target _C -j"$JOBS" > "$OUT/build_$name.log" 2>&1
}

build "$TREE_A" a
build "$TREE_B" b

"$PYTHON" - "$OUT/build_a" "$OUT/build_b" <<'PY'
import collections, re, subprocess, sys
from pathlib import Path

INSN = re.compile(r'^\s+/\*[0-9a-f]+\*/\s+(.+?);')
FUNC = re.compile(r'^\s*Function : (\S+)')
ARCH = re.compile(r'^\s*code for (sm_\d+)')
RES = re.compile(r'^\s*(REG:.*)$')


def dump(obj):
    """{(arch, index): [insn, ...]} for one .cu.o, symbols stripped."""
    txt = subprocess.run(['cuobjdump', '-sass', str(obj)], capture_output=True, text=True).stdout
    out, arch, cur, counter = {}, None, None, collections.Counter()
    for line in txt.splitlines():
        a = ARCH.match(line)
        if a:
            arch = a.group(1)
            continue
        if FUNC.match(line):
            cur = (arch, counter[arch])
            counter[arch] += 1
            out[cur] = []
            continue
        i = INSN.match(line)
        if i and cur is not None:
            out[cur].append(re.sub(r'\s+', ' ', i.group(1)).strip())
    return out


def resources(obj):
    """[REG/STACK/SHARED/... line, ...] in emission order, symbols stripped."""
    txt = subprocess.run(['cuobjdump', '-res-usage', str(obj)], capture_output=True, text=True).stdout
    return [m.group(1).strip() for m in map(RES.match, txt.splitlines()) if m]


def objects(build):
    return {str(p.relative_to(build / 'CMakeFiles')): p
            for p in sorted((build / 'CMakeFiles').rglob('*.cu.o'))}


a, b = Path(sys.argv[1]), Path(sys.argv[2])
oa, ob = objects(a), objects(b)
if oa.keys() != ob.keys():
    print('object sets differ:', sorted(oa.keys() ^ ob.keys()))
    sys.exit(1)

n_kernel = n_insn = 0
bad_stream, bad_hist, bad_res = [], [], []
for name in oa:
    da, db = dump(oa[name]), dump(ob[name])
    if da.keys() != db.keys():
        bad_stream.append((name, 'kernel count/arch mismatch', sorted(da), sorted(db)))
        continue
    for key in da:
        n_kernel += 1
        n_insn += len(da[key])
        if da[key] != db[key]:
            bad_stream.append((name, key, len(da[key]), len(db[key])))
        if (collections.Counter(w.split()[0] for w in da[key])
                != collections.Counter(w.split()[0] for w in db[key])):
            bad_hist.append((name, key))
    ra, rb = resources(oa[name]), resources(ob[name])
    if ra != rb:
        bad_res.append((name, [x for x in zip(ra, rb) if x[0] != x[1]][:3]))

print(f'objects           : {len(oa)}')
print(f'kernels compared  : {n_kernel}')
print(f'instructions      : {n_insn}')
print(f'stream mismatches : {len(bad_stream)}')
print(f'opcode-hist diffs : {len(bad_hist)}')
print(f'ptxas res diffs   : {len(bad_res)}')
for row in bad_stream[:10]:
    print('  STREAM', row)
for row in bad_hist[:10]:
    print('  HIST  ', row)
for row in bad_res[:10]:
    print('  RES   ', row)
sys.exit(1 if (bad_stream or bad_hist or bad_res) else 0)
PY
echo "[sass_diff] byte-identical; build trees kept in $OUT"
