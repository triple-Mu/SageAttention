# Copyright (c) 2025 by SageAttention team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""sm100 WS-auto crossover refinement sweep (measurement only, no source change).

Fills the sampling gaps left by the r3/r4 sweeps that set the auto heuristic
cuts (d128: non-causal >= 16384, causal >= 32768, see
csrc/qattn/qk_int_sv_f8_cuda_sm100.cu sm100_ws_auto_pick):
  - d128 b{1,4} h32 non-causal seq {6144..14336}   (gap 4096..16384)
  - d128 b{1,4} h32 causal     seq {20480..28672}  (gap 16384..32768)
  - d64  b{1,4} h32 s32768 causal {0,1}            (does d64 still lose long?)

Same protocol as scripts/cdsl_bench_fwd.py on the bench host: low-level fwd op
(no quant pre/post), CUDA-event timeit with an 800 ms budget, exclusive-card
check. SAGEATTN_SM100_WS is read once per process, so ws-vs-old comparison is
one process per side, order alternated across rounds by the orchestrator;
per-shape median across rounds.

Usage (bench host, PYTHONPATH at the tree under test):
    SAGEATTN_SM100_TCGEN05=1 SAGEATTN_SM100_WS={0,1} \
        python ws_auto_sweep.py --tag {old,ws}-rN --out NAME.json
"""
import argparse
import json
import os
import subprocess

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--budget-ms", type=float, default=800.0)
ap.add_argument("--min-iters", type=int, default=3)
ap.add_argument("--max-iters", type=int, default=30)
a = ap.parse_args()

HEADS = 32
CTA_Q = CTA_K = 128
WARP_Q = 32
GRID = [(128, b, s, 0) for b in (1, 4) for s in (6144, 8192, 10240, 12288, 14336)]
GRID += [(128, b, s, 1) for b in (1, 4) for s in (20480, 24576, 28672)]
GRID += [(64, b, 32768, c) for c in (0, 1) for b in (1, 4)]

MY_PID = str(os.getpid())


def compute_apps():
    """Other processes holding the card, per BENCH_PROTOCOL's exclusivity rule."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception as exc:
        return [f"query-failed: {exc}"]
    return [ln.strip() for ln in out.splitlines() if ln.strip() and ln.split(",")[0].strip() != MY_PID]


def timeit(fn):
    torch.cuda.synchronize()
    flush = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); fn(); e.record(); torch.cuda.synchronize()
    one = s.elapsed_time(e)
    iters = max(a.min_iters, min(a.max_iters, int(a.budget_ms / max(one, 1e-3))))
    flush.zero_()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    del flush
    return s.elapsed_time(e) / iters


def make_kernel_case(hd, b, seq, causal):
    import sageattention  # noqa: F401  (loads _C, registers torch.ops.sageattention.*)
    q = torch.randint(-95, 95, (b, HEADS, seq, hd), dtype=torch.int8, device="cuda")
    k = torch.randint(-95, 95, (b, HEADS, seq, hd), dtype=torch.int8, device="cuda")
    q_scale = torch.randn(b, HEADS, seq // CTA_Q * (CTA_Q // WARP_Q), dtype=torch.float, device="cuda")
    k_scale = torch.randn(b, HEADS, seq // CTA_K, dtype=torch.float, device="cuda")
    v = torch.randn(b, HEADS, hd, seq, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
    v_scale = torch.rand(b, HEADS, hd, dtype=torch.float, device="cuda") + 0.5
    sm_scale = 1 / (hd ** 0.5)
    keep = [q, k, v, q_scale, k_scale, v_scale]
    fwd = torch.ops.sageattention.fwd

    def fn():
        fwd(q, k, v, q_scale, k_scale, v_scale, None,
            tensor_layout="HND", qk_quant_gran="per_warp", pv_accum_dtype="fp32",
            v_layout="linear", is_causal=bool(causal), sm_scale=sm_scale,
            return_lse=False, out_dtype=torch.float16)
    return fn, keep


res = {}
for (hd, b, seq, causal) in GRID:
    key = f"hd{hd}/causal={causal}/b{b}h{HEADS}s{seq}"
    before = compute_apps()
    try:
        fn, keep = make_kernel_case(hd, b, seq, causal)
        ms = timeit(fn)
        after = compute_apps()
        flops = 4 * HEADS * b * hd * seq * seq / (2 if causal else 1)
        rec = {"ms": ms, "tflops": flops / (ms * 1e-3) * 1e-12,
               "exclusive": not before and not after}
        if before or after:
            rec["intruders"] = {"before": before, "after": after}
        res[key] = rec
        print(f"{a.tag} {key}: {ms:.4f} ms {rec['tflops']:.1f} TFLOPS excl={rec['exclusive']}", flush=True)
        del fn, keep
    except Exception as exc:
        res[key] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"{a.tag} {key}: ERROR {exc}", flush=True)
    torch.cuda.empty_cache()

json.dump({"tag": a.tag, "results": res}, open(a.out, "w"), indent=1)
print("WROTE", a.out)
