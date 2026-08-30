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

"""sm100 hang-stress driver for the C1 ws kernel (and the 128-thread kernel).

Sweeps seq 2k..128k x causal, then hammers one fixed shape, all through the
functional fwd op (the transitional per-arch ops are retired). Prints before
every launch and synchronizes after it, so on a hang the last line names the
launch that never came back. Select the kernel with SAGEATTN_SM100_TCGEN05 /
SAGEATTN_SM100_WS in the environment.

Usage (from a tree on PYTHONPATH):
    SAGEATTN_SM100_TCGEN05=1 SAGEATTN_SM100_WS=1 \
        python bench/sm100_review/ws_stress.py
"""

import argparse
import time

import torch

import sageattention  # noqa: F401  (loads _C, registers torch.ops.sageattention.*)

ap = argparse.ArgumentParser()
ap.add_argument("--hd", type=int, default=128)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--heads", type=int, default=32)
ap.add_argument("--sweep-reps", type=int, default=10)
ap.add_argument("--fixed-seq", type=int, default=32768)
ap.add_argument("--fixed-reps", type=int, default=8000)
a = ap.parse_args()

CTA = 128


def mk(seq, hd, b, h):
    q = torch.randint(-95, 95, (b, h, seq, hd), dtype=torch.int8, device="cuda")
    k = torch.randint(-95, 95, (b, h, seq, hd), dtype=torch.int8, device="cuda")
    qs = torch.rand(b, h, seq // CTA * (CTA // 32), dtype=torch.float, device="cuda") + 0.5
    ks = torch.rand(b, h, seq // CTA, dtype=torch.float, device="cuda") + 0.5
    v = torch.randn(b, h, hd, seq, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
    vs = torch.rand(b, h, hd, dtype=torch.float, device="cuda") + 0.5
    return q, k, v, qs, ks, vs


def run(args, hd, causal):
    q, k, v, qs, ks, vs = args
    torch.ops.sageattention.fwd(q, k, v, qs, ks, vs, None,
                                tensor_layout="HND", qk_quant_gran="per_warp",
                                pv_accum_dtype="fp32", v_layout="linear",
                                is_causal=causal, sm_scale=1 / (hd ** 0.5),
                                return_lse=False, out_dtype=torch.float16)
    torch.cuda.synchronize()


t0 = time.time()
for seq in (2048, 4096, 8192, 16384, 32768, 65536, 131072):
    args = mk(seq, a.hd, a.batch, a.heads)
    for causal in (False, True):
        for i in range(a.sweep_reps):
            print(f"sweep seq={seq} causal={causal} rep={i} t={time.time()-t0:.1f}s", flush=True)
            run(args, a.hd, causal)
    del args
    torch.cuda.empty_cache()
print("SWEEP_DONE", flush=True)

args = mk(a.fixed_seq, a.hd, a.batch, a.heads)
for causal in (False, True):
    for i in range(a.fixed_reps):
        if i % 200 == 0:
            print(f"fixed seq={a.fixed_seq} causal={causal} rep={i} t={time.time()-t0:.1f}s", flush=True)
        run(args, a.hd, causal)
print("STRESS_ALL_DONE", flush=True)
