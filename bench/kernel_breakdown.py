#!/usr/bin/env python
"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

Per-kernel breakdown of ``sageattn`` at its *default* settings, measured the
same way on a baseline (pre-refactor, per-arch pybind) install and on a current
(``torch.ops.sageattention.*``) install, then merged into one table.

Like ``tools/compare_reference.py`` this file is self-contained: it imports
nothing from the repository, so it can be copied into a baseline environment
that has no current checkout.

Do NOT run it as ``python -m bench.kernel_breakdown``: ``-m`` puts the current
directory at the head of ``sys.path``, so from a SageAttention checkout
``import sageattention`` would pick up the *source tree* instead of the
installed package under test. Always run it as a plain script path, from a
directory that is not a checkout.

Reproduction, end to end
------------------------
1. Baseline environment (commit 0a5d2e4, the per-arch pybind era)::

       git worktree add --detach /path/baseline 0a5d2e4
       cd /path/baseline
       TORCH_CUDA_ARCH_LIST=<cc> MAX_JOBS=4 NVCC_APPEND_FLAGS=--threads=4 \
           python setup.py build_ext --inplace
       cd /tmp && PYTHONPATH=/path/baseline python <this file> \
           --out /shared/kb --side baseline

2. New environment (this branch, CMake build)::

       cmake -S . -B build/cmake -G Ninja -DPython_EXECUTABLE=$(which python) \
             -DSAGE_CUDA_ARCHS=<cc> -DSAGE_NVCC_THREADS=4
       cmake --build build/cmake -j 4
       cp build/cmake/lib/sageattention/_C.abi3.so sageattention/
       cd /tmp && python <this file> --out /shared/kb --side new

   sm100 needs ``SAGEATTN_SM100_TCGEN05=1`` in the environment on *both* sides
   for the tcgen05 kernels to be dispatched at all.

3. Either environment, merging both sides::

       python <this file> --report /shared/kb \
           --csv /shared/kb/breakdown.csv --md /shared/kb/breakdown.md

``--subset`` (first and last shape of every group, 8 shapes) is the smoke run.
``--dump-names`` runs one shape per group and prints the raw-kernel-name to
role inventory without writing anything: use it first on a new arch to check
that ROLE_PATTERNS still covers the kernels this GPU dispatches, and patch the
table in place if it does not.

What is measured
----------------
``sageattn(q, k, v)`` with library defaults (HND, fp16, head_dim 128, smooth_k
on, everything else resolved per device), wrapped in ``torch.profiler`` with
CUDA activity: warmup 3, then N timed iterations (N = 10 / 5 / 3 for
seq_len <= 8192 / <= 32768 / above). ``key_averages()`` is aggregated by kernel
name and divided by N, so every number is *microseconds of device time per
sageattn call*. Memset/memcpy rows are kept -- they are real device work.

Every record stores the full raw kernel-name table. The role mapping below is a
derived view; when it is wrong, the raw table is still the ground truth.

Traceability
------------
Each output JSON carries: side, arch, compiled archs, GPU name + UUID, driver,
torch/CUDA versions, the installed package path and version, the source commit
of the tree that built it (+dirty flag), the exact argv, host name, UTC and
local timestamps, the ROLE_PATTERNS in force, and per shape the untouched
kernel-name -> (us_per_call, call count) table. ``--report`` names the source
JSON of every column it prints.
"""

import argparse
import csv
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

SCHEMA_VERSION = 1

HEAD_DIM = 128
TENSOR_LAYOUT = "HND"
DTYPE = torch.float16
WARMUP = 3


# ------------------------------------------------------------- shape matrix
#
# AIGC-shaped workloads, head_dim 128 throughout. The head counts are the two
# model anchors (56 and 40) plus the head counts an Ulysses split leaves on one
# rank. 50 shapes per side.

SHAPE_GROUPS = (
    # (group, head counts, seq lens, batch sizes, is_causal)
    ("core", (56, 40, 24, 10, 5, 3), (4096, 32768, 65536), (1, 2), False),
    ("longctx", (56, 40, 24), (16384, 131072), (1,), False),
    ("serving", (40, 24), (1024, 8192), (8,), False),
    ("causal", (56, 40), (4096, 32768), (1,), True),
)


def shape_id(rec):
    return f"b{rec['b']}h{rec['h']}s{rec['s']}" + ("c" if rec["causal"] else "")


def build_shapes(subset=False, spec=None):
    """The built-in matrix, one group at a time. ``subset`` keeps the first and
    last shape of each group (the two extremes); ``spec`` replaces the matrix
    entirely."""
    if spec:
        return parse_shapes(spec)
    shapes = []
    for group, heads, seqs, batches, causal in SHAPE_GROUPS:
        block = [
            {"group": group, "b": b, "h": h, "s": s, "d": HEAD_DIM, "causal": causal}
            for h in heads
            for s in seqs
            for b in batches
        ]
        shapes += [block[0], block[-1]] if subset else block
    return shapes


def parse_shapes(spec):
    """``--shapes`` grammar: comma-separated ``BxHxS`` items, optional trailing
    ``c`` for causal. Example: ``1x40x4096,2x56x32768c``."""
    out = []
    for item in spec.replace(" ", "").split(","):
        if not item:
            continue
        causal = item.endswith("c")
        body = item[:-1] if causal else item
        parts = body.split("x")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise SystemExit(f"unparsable --shapes item {item!r} (expected BxHxS or BxHxSc)")
        b, h, s = (int(p) for p in parts)
        out.append({"group": "custom", "b": b, "h": h, "s": s, "d": HEAD_DIM, "causal": causal})
    if not out:
        raise SystemExit("--shapes parsed to nothing")
    return out


def iters_for(seq_len):
    if seq_len <= 8192:
        return 10
    if seq_len <= 32768:
        return 5
    return 3


# ------------------------------------------------------------- role mapping
#
# Raw demangled kernel name -> role. First match wins, so the specific patterns
# come before the generic elementwise catch-all. Both sides are covered by the
# same table: the refactor moved the attention kernels into `sage::smXX::` and
# the fused kernels into an anonymous namespace, but kept the names, so the
# patterns are deliberately unanchored.
#
# Verified against real sm_86 profiles of both sides (2026-08-30); the fp8-only
# names were read out of the built object files with `nm -C`. Run --dump-names
# on a new arch before trusting it.
#
#   attention  qk_int_sv_f16_attn_kernel        sm80  (both sides)
#              qk_int_sv_f8_attn_kernel         sm89 / sm120
#              qk_int8_sv_f8_attn_kernel        sm90
#              qk_int8_sv_f8_attn_kernel_sm100  sm100
#   quant_q/k  QuantInt8Kernel                  per_warp Q and per_block K; the
#                                               5th template argument is
#                                               sub_mean and is what tells the
#                                               two apart (K is the smoothed one)
#              QuantPerThread{Q,K}Int8Kernel    per_thread
#   transpose_pad         TransposePadPermuteKernel   baseline V^T pass
#   mean_scale_fp8        MeanScaleKernel             baseline V fp8 pass, and
#                                                     the new side's long-seq
#                                                     fallback
#   transpose_quant_fused TransposeQuantFp8Kernel     new side, the two above
#                                                     fused into one
#   v_pad_cat             CatArrayBatchedCopy*        baseline python-side zero
#                                                     padding of V (sm90/sm100)
#   fill_memset           Memset / FillFunctor        zeroed scale and fp8
#                                                     buffers
#   k_mean                reduce_kernel<..MeanOps..>  smooth_k's k.mean(), ATen
#   segment_mean          SegmentMean*Kernel          varlen smooth_k only
#   sub_mean_v            SubMeanKernel               smooth_v only (off by
#                                                     default on every arch)

ROLE_PATTERNS = (
    ("attention", r"qk_int8?_sv_f(?:16|8)_attn_kernel"),
    ("segment_mean", r"SegmentMean(?:Finish)?Kernel"),
    ("quant_k", r"QuantPerThreadKInt8Kernel"),
    ("quant_q", r"QuantPerThreadQInt8Kernel"),
    # QuantInt8Kernel<head_dim, tokens, packs, has_sm_scale, sub_mean, T>
    ("quant_k", r"QuantInt8Kernel<[^>]*,\s*(?:true|false),\s*true,"),
    ("quant_q", r"QuantInt8Kernel<"),
    ("transpose_quant_fused", r"TransposeQuantFp8Kernel"),
    ("transpose_pad", r"TransposePadPermuteKernel"),
    ("mean_scale_fp8", r"MeanScaleKernel"),
    ("sub_mean_v", r"SubMeanKernel"),
    ("k_mean", r"reduce_kernel<.*MeanOps"),
    ("v_pad_cat", r"CatArrayBatchedCopy"),
    ("fill_memset", r"^Memset|FillFunctor"),
    ("other_elementwise", r"elementwise_kernel|^Memcpy"),
)

# Report order; "other" is the fallback bucket and is always last.
ROLES = (
    "k_mean",
    "segment_mean",
    "quant_q",
    "quant_k",
    "transpose_pad",
    "mean_scale_fp8",
    "transpose_quant_fused",
    "sub_mean_v",
    "v_pad_cat",
    "fill_memset",
    "attention",
    "other_elementwise",
    "other",
)

_COMPILED_ROLES = tuple((role, re.compile(pat)) for role, pat in ROLE_PATTERNS)


def role_of(name):
    for role, rx in _COMPILED_ROLES:
        if rx.search(name):
            return role
    return "other"


# ------------------------------------------------------------------ backends
#
# Both classes expose the same informal interface (they are deliberately
# unrelated types -- one talks to pybind modules, the other to torch.ops):
#
#   kind            -> short string for the record
#   compiled_archs() -> [int]
#   call(q, k, v, causal) -> runs sageattn with library defaults


class LegacyBackend:
    """Pre-refactor install: pybind modules ``sageattention._fused`` and
    ``sageattention._qattn_smXX``, dispatched by ``core.sageattn``."""

    side = "baseline"
    kind = "legacy-pybind"

    def __init__(self):
        import sageattention

        self.sageattn = sageattention.sageattn
        self.archs = []
        for arch in (80, 89, 90, 100, 120):
            try:
                __import__(f"sageattention._qattn_sm{arch}", fromlist=["_"])
            except ImportError:
                continue
            self.archs.append(arch)

    def compiled_archs(self):
        return self.archs

    def call(self, q, k, v, causal):
        return self.sageattn(q, k, v, tensor_layout=TENSOR_LAYOUT, is_causal=causal)


class NewBackend:
    """Post-refactor install: a single ``_C`` extension registering
    ``torch.ops.sageattention.*``."""

    side = "new"
    kind = "torch-ops"

    def __init__(self):
        import sageattention  # noqa: F401  (loads _C, registers the ops)

        self.sageattn = sageattention.sageattn

    def compiled_archs(self):
        return sorted(torch.ops.sageattention.compiled_archs())

    def call(self, q, k, v, causal):
        return self.sageattn(q, k, v, tensor_layout=TENSOR_LAYOUT, is_causal=causal)


def detect_backend(forced=None):
    """Same probe as tools/compare_reference.py: torch.ops first, pybind after."""
    if forced == "baseline":
        return LegacyBackend()
    if forced == "new":
        return NewBackend()
    try:
        import sageattention  # noqa: F401

        torch.ops.sageattention.compiled_archs()
        return NewBackend()
    except Exception:  # noqa: BLE001  (any import/attribute failure means "not the new build")
        pass
    try:
        import sageattention._fused  # noqa: F401

        return LegacyBackend()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "cannot classify the installed sageattention: neither "
            "torch.ops.sageattention.compiled_archs (new build) nor "
            f"sageattention._fused (baseline build) is available ({exc!r}). "
            "Use --side to force one."
        )


def sm100_tcgen05_enabled():
    return os.environ.get("SAGEATTN_SM100_TCGEN05", "0").upper() in {"1", "TRUE", "YES"}


def resolve_arch(cc, compiled):
    """Which kernel family this device dispatches to; mirrors plan.cpp (and the
    pre-refactor core.py if/elif chain, which agrees with it)."""
    compiled = set(compiled)
    if cc < (8, 0):
        return None
    if cc[0] == 8 and cc[1] < 9:
        arch = 80
    elif cc == (8, 9):
        arch = 89 if 89 in compiled else 80
    elif cc[0] == 9:
        arch = 90 if (90 in compiled and cc[1] == 0) else 80
    elif cc[0] == 12:
        arch = 120 if 120 in compiled else (89 if 89 in compiled else 80)
    elif cc[0] >= 10:
        if cc in {(10, 0), (11, 0)} and 100 in compiled and sm100_tcgen05_enabled():
            arch = 100
        elif 89 in compiled:
            arch = 89
        else:
            arch = 80
    else:
        return None
    return arch if arch in compiled else None


# --------------------------------------------------------------- provenance


def run_cmd(argv, timeout=30):
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def source_commit(pkg_file):
    """Commit of the tree the installed package was built from, plus whether it
    had uncommitted changes. Walks up from the package directory; an installed
    wheel has no git dir and reports "unknown"."""
    d = Path(pkg_file).resolve().parent
    head = run_cmd(["git", "-C", str(d), "rev-parse", "HEAD"])
    if head is None:
        return {"commit": "unknown", "dirty": None, "dir": str(d)}
    status = run_cmd(["git", "-C", str(d), "status", "--porcelain"])
    top = run_cmd(["git", "-C", str(d), "rev-parse", "--show-toplevel"])
    return {
        "commit": head.strip(),
        "dirty": None if status is None else bool(status.strip()),
        "dir": top.strip() if top else str(d),
    }


def gpu_uuid(device):
    return str(torch.cuda.get_device_properties(device).uuid)


def check_exclusive(device, allow_shared):
    """Refuse to measure on a GPU another process is computing on. Returns the
    record of what was found (also stored in the meta block)."""
    uuid = gpu_uuid(device)
    if allow_shared:
        return {"status": "skipped", "reason": "--allow-shared", "foreign": []}
    raw = run_cmd(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    if raw is None:
        raise SystemExit(
            "nvidia-smi is unavailable, so exclusive use of the GPU cannot be "
            "verified. Timings on a shared GPU are meaningless; pass "
            "--allow-shared to measure anyway."
        )
    foreign = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        pid, mem, uu = parts[0], parts[1], parts[2]
        if uu.removeprefix("GPU-") != uuid:
            continue
        if pid.isdigit() and int(pid) == os.getpid():
            continue
        foreign.append({"pid": pid, "used_mib": mem})
    if foreign:
        listed = ", ".join(f"pid {f['pid']} ({f['used_mib']} MiB)" for f in foreign)
        raise SystemExit(
            f"GPU {device} (GPU-{uuid}) already has compute processes: {listed}. "
            "Timings would be contended; free the GPU or pass --allow-shared."
        )
    return {"status": "exclusive", "reason": None, "foreign": []}


def build_meta(be, args, device, exclusive):
    import sageattention

    cc = torch.cuda.get_device_capability(device)
    props = torch.cuda.get_device_properties(device)
    compiled = be.compiled_archs()
    driver = run_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return {
        "schema_version": SCHEMA_VERSION,
        "side": args.side if args.side != "auto" else be.side,
        "backend_kind": be.kind,
        "arch": resolve_arch(cc, compiled),
        "compiled_archs": compiled,
        "cc": list(cc),
        "gpu_name": props.name,
        "gpu_uuid": gpu_uuid(device),
        "gpu_total_memory_bytes": props.total_memory,
        "gpu_index": device,
        "driver_version": driver.strip() if driver else "unknown",
        "exclusive_check": exclusive,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "sageattention_file": str(sageattention.__file__),
        "sageattention_version": str(getattr(sageattention, "__version__", "unknown")),
        "source": source_commit(sageattention.__file__),
        "sm100_tcgen05": os.environ.get("SAGEATTN_SM100_TCGEN05", "0"),
        "argv": sys.argv,
        "cmdline": " ".join([sys.executable] + sys.argv),
        "cwd": os.getcwd(),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "hostname": platform.node(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "head_dim": HEAD_DIM,
        "tensor_layout": TENSOR_LAYOUT,
        "dtype": str(DTYPE),
        "warmup": WARMUP,
        "subset": bool(args.subset),
        "shapes_spec": args.shapes,
        "role_patterns": [list(p) for p in ROLE_PATTERNS],
    }


# -------------------------------------------------------------- measurement


def profile_call(call, iters):
    """Per-kernel device time of one ``call()``, in microseconds. The profiler
    is CPU+CUDA because kineto needs the CPU side to correlate the launches;
    only the CUDA rows are read back."""
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            call()
        torch.cuda.synchronize()
    kernels = {}
    for e in prof.key_averages():
        if e.device_type != DeviceType.CUDA:
            continue
        if not e.count:
            continue
        entry = kernels.setdefault(e.key, {"us_total": 0.0, "count": 0})
        entry["us_total"] += float(e.self_device_time_total)
        entry["count"] += int(e.count)
    for entry in kernels.values():
        entry["us_per_call"] = entry["us_total"] / iters
        entry["calls_per_iter"] = entry["count"] / iters
    return kernels


def is_oom(exc):
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def measure_shape(be, spec, device):
    """One shape -> one record. Allocation and execution failures are recorded
    and stepped over, they never abort the sweep."""
    iters = iters_for(spec["s"])
    rec = dict(spec)
    rec["shape_id"] = shape_id(spec)
    rec["iters"] = iters
    rec["warmup"] = WARMUP
    q = k = v = None
    try:
        shape = (spec["b"], spec["h"], spec["s"], spec["d"])
        q = torch.randn(shape, dtype=DTYPE, device=device)
        k = torch.randn(shape, dtype=DTYPE, device=device)
        v = torch.randn(shape, dtype=DTYPE, device=device)
        kernels = profile_call(lambda: be.call(q, k, v, spec["causal"]), iters)
    except Exception as exc:  # noqa: BLE001  (an OOM here must not kill the sweep)
        rec["status"] = "OOM" if is_oom(exc) else "error"
        rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
        rec["kernels"] = {}
        rec["roles"] = {}
        rec["total_us"] = 0.0
        print(f"  {rec['shape_id']:>16}  {rec['status']}: {rec['error'].splitlines()[0]}")
    else:
        roles = {}
        for name, entry in kernels.items():
            roles[role_of(name)] = roles.get(role_of(name), 0.0) + entry["us_per_call"]
        rec["status"] = "ok"
        rec["error"] = None
        rec["kernels"] = kernels
        rec["roles"] = roles
        rec["total_us"] = sum(roles.values())
        print(
            f"  {rec['shape_id']:>16}  iters={iters:2d}  total={rec['total_us']:10.1f} us  "
            f"attn={roles.get('attention', 0.0):10.1f} us  kernels={len(kernels)}"
        )
    finally:
        q = k = v = None
        torch.cuda.empty_cache()
    return rec


def inventory(records):
    """kernel name -> role over every record, and the names that fell through."""
    name_role = {}
    for rec in records:
        for name in rec.get("kernels", {}):
            name_role[name] = role_of(name)
    unmatched = sorted(n for n, r in name_role.items() if r == "other")
    return dict(sorted(name_role.items())), unmatched


# ------------------------------------------------------------------ reports


def load_runs(report_dir):
    runs = []
    for path in sorted(Path(report_dir).glob("*.json")):
        try:
            blob = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            print(f"WARNING: skipping {path.name}: {exc}", file=sys.stderr)
            continue
        if "meta" not in blob or "records" not in blob:
            print(f"WARNING: skipping {path.name}: not a kernel_breakdown dump", file=sys.stderr)
            continue
        if blob.get("schema_version") != SCHEMA_VERSION:
            print(
                f"WARNING: {path.name} has schema_version "
                f"{blob.get('schema_version')!r}, expected {SCHEMA_VERSION}",
                file=sys.stderr,
            )
        blob["file"] = path.name
        runs.append(blob)
    if not runs:
        raise SystemExit(f"no kernel_breakdown JSON found in {report_dir}")
    return runs


def collect(runs):
    """(shape_id -> {"spec", "baseline", "new"}) with the source file of each
    side, so every printed number can be traced back to its dump."""
    table = {}
    for blob in runs:
        side = blob["meta"]["side"]
        for rec in blob["records"]:
            slot = table.setdefault(rec["shape_id"], {"spec": rec, "sides": {}})
            if side in slot["sides"]:
                print(
                    f"WARNING: {rec['shape_id']} appears twice for side {side}; "
                    f"{blob['file']} wins over {slot['sides'][side]['file']}",
                    file=sys.stderr,
                )
            slot["sides"][side] = {"rec": rec, "file": blob["file"]}
    order = sorted(
        table,
        key=lambda sid: (
            table[sid]["spec"].get("group", ""),
            table[sid]["spec"]["s"],
            table[sid]["spec"]["h"],
            table[sid]["spec"]["b"],
        ),
    )
    return table, order


def side_roles(slot, side):
    """(role -> us, status) for one side of one shape; a missing side is zeros."""
    entry = slot["sides"].get(side)
    if entry is None:
        return {}, "missing"
    return entry["rec"].get("roles", {}), entry["rec"].get("status", "ok")


def cell(base_us, new_us, base_status, new_status):
    if base_status == "OOM" or new_status == "OOM":
        return "OOM"
    ratio = (base_us / new_us) if new_us > 0 else 0.0
    return f"{base_us:.1f}/{new_us:.1f}/{ratio:.3f}"


def write_csv(path, table, order):
    """Rows = roles, columns = shapes, cell = baseline us / new us / ratio
    (ratio = baseline / new, so > 1 means the new build is faster). A side with
    no record contributes 0."""
    used_roles = [
        r
        for r in ROLES
        if any(
            side_roles(table[sid], side)[0].get(r, 0.0) > 0
            for sid in order
            for side in ("baseline", "new")
        )
    ]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["field"] + list(order))
        for field in ("group", "b", "h", "s", "d"):
            w.writerow([field] + [table[sid]["spec"][field] for sid in order])
        w.writerow(["causal"] + [int(table[sid]["spec"]["causal"]) for sid in order])
        w.writerow(["iters"] + [table[sid]["spec"].get("iters", "") for sid in order])
        for side in ("baseline", "new"):
            w.writerow([f"{side}_status"] + [side_roles(table[sid], side)[1] for sid in order])
            w.writerow(
                [f"{side}_json"]
                + [
                    table[sid]["sides"][side]["file"] if side in table[sid]["sides"] else ""
                    for sid in order
                ]
            )
        w.writerow(["role (baseline us / new us / ratio)"] + [""] * len(order))
        for role in used_roles + ["TOTAL"]:
            row = [role]
            for sid in order:
                b_roles, b_status = side_roles(table[sid], "baseline")
                n_roles, n_status = side_roles(table[sid], "new")
                if role == "TOTAL":
                    b_us, n_us = sum(b_roles.values()), sum(n_roles.values())
                else:
                    b_us, n_us = b_roles.get(role, 0.0), n_roles.get(role, 0.0)
                row.append(cell(b_us, n_us, b_status, n_status))
            w.writerow(row)
    return used_roles


def md_table(rows):
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    out = []
    for n, row in enumerate(rows):
        out.append("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |")
        if n == 0:
            out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    return out


def write_md(path, runs, table, order, used_roles):
    lines = ["# Kernel breakdown: baseline vs new", ""]

    lines += ["## Runs", ""]
    rows = [["file", "side", "arch", "gpu", "commit", "dirty", "torch", "cuda", "timestamp"]]
    for blob in runs:
        m = blob["meta"]
        rows.append(
            [
                blob["file"],
                m["side"],
                m.get("arch"),
                m.get("gpu_name", "?"),
                m.get("source", {}).get("commit", "?")[:12],
                m.get("source", {}).get("dirty"),
                m.get("torch_version", "?"),
                m.get("cuda_version", "?"),
                m.get("timestamp_utc", "?"),
            ]
        )
    lines += md_table(rows) + [""]

    lines += [
        "## Per-role microseconds per sageattn call, aggregated by seq_len",
        "",
        "`ratio` = baseline / new (> 1 = the new build is faster). Shapes where "
        "either side hit OOM are excluded from the aggregate and listed below.",
        "",
    ]
    by_seq = {}
    for sid in order:
        spec = table[sid]["spec"]
        b_roles, b_status = side_roles(table[sid], "baseline")
        n_roles, n_status = side_roles(table[sid], "new")
        if b_status != "ok" or n_status != "ok":
            continue
        bucket = by_seq.setdefault(spec["s"], {"n": 0, "b": {}, "nw": {}})
        bucket["n"] += 1
        for role in used_roles:
            bucket["b"][role] = bucket["b"].get(role, 0.0) + b_roles.get(role, 0.0)
            bucket["nw"][role] = bucket["nw"].get(role, 0.0) + n_roles.get(role, 0.0)
    for seq in sorted(by_seq):
        bucket = by_seq[seq]
        lines += [f"### seq_len = {seq} ({bucket['n']} shapes)", ""]
        rows = [["role", "baseline us", "new us", "ratio"]]
        for role in used_roles + ["TOTAL"]:
            if role == "TOTAL":
                b_us, n_us = sum(bucket["b"].values()), sum(bucket["nw"].values())
            else:
                b_us, n_us = bucket["b"].get(role, 0.0), bucket["nw"].get(role, 0.0)
            if b_us == 0.0 and n_us == 0.0:
                continue
            ratio = f"{b_us / n_us:.3f}" if n_us > 0 else "-"
            rows.append([role, f"{b_us:.1f}", f"{n_us:.1f}", ratio])
        lines += md_table(rows) + [""]

    lines += ["## Per-shape totals", ""]
    rows = [["shape", "group", "b", "h", "s", "causal", "baseline us", "new us", "ratio", "note"]]
    for sid in order:
        spec = table[sid]["spec"]
        b_roles, b_status = side_roles(table[sid], "baseline")
        n_roles, n_status = side_roles(table[sid], "new")
        b_us, n_us = sum(b_roles.values()), sum(n_roles.values())
        ratio = f"{b_us / n_us:.3f}" if (b_status == "ok" and n_status == "ok" and n_us) else "-"
        note = "" if (b_status == "ok" and n_status == "ok") else f"{b_status}/{n_status}"
        rows.append(
            [
                sid,
                spec.get("group", ""),
                spec["b"],
                spec["h"],
                spec["s"],
                int(spec["causal"]),
                f"{b_us:.1f}",
                f"{n_us:.1f}",
                ratio,
                note,
            ]
        )
    lines += md_table(rows) + [""]

    lines += [
        "## Role mapping actually applied",
        "",
        "Every raw kernel name each side emitted, verbatim, under the role it was "
        "folded into. This is the audit trail for the aggregates above: if a role "
        "looks wrong, the names here say why.",
        "",
    ]
    for blob in runs:
        m = blob["meta"]
        lines += [f"### {m['side']} ({blob['file']}, arch sm{m.get('arch')})", ""]
        by_role = {}
        for name, role in blob.get("kernel_role_map", {}).items():
            by_role.setdefault(role, []).append(name)
        if not by_role:
            lines += ["(no kernels recorded)", ""]
            continue
        for role in [r for r in ROLES if r in by_role] + sorted(set(by_role) - set(ROLES)):
            names = sorted(by_role[role])
            lines += [f"- **{role}** ({len(names)} kernel(s))"]
            lines += [f"  - `{n}`" for n in names]
        lines += [""]

    unmatched = sorted({n for blob in runs for n in blob.get("unmatched_kernels", [])})
    lines += ["## Unmatched kernels", ""]
    lines += ([f"- `{n}`" for n in unmatched] if unmatched else ["None."]) + [""]

    Path(path).write_text("\n".join(lines))


# -------------------------------------------------------------------- driver


def main():
    ap = argparse.ArgumentParser(
        description="per-kernel breakdown of sageattn, baseline vs new",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Reproduction, end to end"
        + __doc__.split("Reproduction, end to end")[1].split("What is measured")[0],
    )
    ap.add_argument("--out", default="kernel-breakdown", help="directory for the JSON dump")
    ap.add_argument(
        "--report",
        metavar="DIR",
        help="merge every JSON in DIR into a CSV + markdown report and exit",
    )
    ap.add_argument("--csv", help="report CSV path (default: <report dir>/breakdown.csv)")
    ap.add_argument("--md", help="report markdown path (default: <report dir>/breakdown.md)")
    ap.add_argument(
        "--side",
        choices=("auto", "baseline", "new"),
        default="auto",
        help="label (and force) the install under test instead of probing for it",
    )
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument(
        "--subset", action="store_true", help="first and last shape of each group (smoke run)"
    )
    ap.add_argument("--shapes", help="replace the matrix: BxHxS[c] items, comma separated")
    ap.add_argument(
        "--dump-names",
        action="store_true",
        help="run one shape per group, print the kernel-name -> role inventory, write nothing",
    )
    ap.add_argument(
        "--allow-shared",
        action="store_true",
        help="measure even though another process is computing on the GPU",
    )
    args = ap.parse_args()

    if args.report:
        runs = load_runs(args.report)
        table, order = collect(runs)
        csv_path = args.csv or str(Path(args.report) / "breakdown.csv")
        md_path = args.md or str(Path(args.report) / "breakdown.md")
        used_roles = write_csv(csv_path, table, order)
        write_md(md_path, runs, table, order, used_roles)
        sides = sorted({blob["meta"]["side"] for blob in runs})
        print(f"{len(runs)} run(s), sides {sides}, {len(order)} shape(s)")
        print(f"csv: {Path(csv_path).resolve()}")
        print(f"md:  {Path(md_path).resolve()}")
        unmatched = sorted({n for blob in runs for n in blob.get("unmatched_kernels", [])})
        for name in unmatched:
            print(f"WARNING: unmatched kernel folded into 'other': {name}", file=sys.stderr)
        return 0

    if sys.path and os.path.isfile(os.path.join(sys.path[0], "sageattention", "__init__.py")):
        print(
            f"WARNING: {sys.path[0]} contains a sageattention source tree and is on "
            "sys.path; the import below may shadow the installed package. Run this "
            "file by path (not with -m) from outside the checkout.",
            file=sys.stderr,
        )

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    torch.cuda.set_device(args.device)
    exclusive = check_exclusive(args.device, args.allow_shared)
    be = detect_backend(None if args.side == "auto" else args.side)
    meta = build_meta(be, args, args.device, exclusive)
    if meta["arch"] is None:
        raise SystemExit(
            f"this build has no kernel family for sm_{meta['cc'][0]}{meta['cc'][1]} "
            f"(compiled: {meta['compiled_archs']})"
        )
    print(json.dumps(meta, indent=1, sort_keys=True, default=str))

    if args.dump_names:
        seen = set()
        shapes = []
        for spec in build_shapes(subset=True, spec=args.shapes):
            if spec["group"] not in seen:
                seen.add(spec["group"])
                shapes.append(spec)
    else:
        shapes = build_shapes(subset=args.subset, spec=args.shapes)
    print(f"[run] side={meta['side']} arch=sm{meta['arch']} shapes={len(shapes)}")

    records = [measure_shape(be, spec, args.device) for spec in shapes]
    name_role, unmatched = inventory(records)

    if args.dump_names:
        print("\n[names] raw kernel name -> role")
        for name, role in sorted(name_role.items(), key=lambda kv: (kv[1], kv[0])):
            print(f"  {role:22s} {name}")
        print(f"\n[names] {len(name_role)} kernel(s), {len(unmatched)} unmatched")
        for name in unmatched:
            print(f"UNMATCHED: {name}", file=sys.stderr)
        return 1 if unmatched else 0

    for name in unmatched:
        print(f"WARNING: unmatched kernel folded into 'other': {name}", file=sys.stderr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = out_dir / f"kb_{meta['side']}_sm{meta['arch']}_{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "meta": meta,
                "records": records,
                "kernel_role_map": name_role,
                "unmatched_kernels": unmatched,
            },
            indent=1,
            sort_keys=True,
            default=str,
        )
    )
    n_ok = sum(1 for r in records if r["status"] == "ok")
    n_oom = sum(1 for r in records if r["status"] == "OOM")
    n_err = sum(1 for r in records if r["status"] == "error")
    print(f"shapes: ok={n_ok} oom={n_oom} error={n_err}")
    print(f"written to {path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
