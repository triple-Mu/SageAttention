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

"""One-shape rebuild sanity for the ws_auto_sweep run: ws output must be
bit-equal to old (the ws kernel's bit-exact contract, cheap stand-in for the
full golden gate on a measurement-only session).

Run twice with the same seed: first --save under SAGEATTN_SM100_WS=0, then
--ref under SAGEATTN_SM100_WS=1; the second invocation compares and prints
SANITY_PASS / SANITY_FAIL.
"""
import argparse

import torch

import sageattention  # noqa: F401  (loads _C, registers torch.ops.sageattention.*)

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--save", metavar="PT")
g.add_argument("--ref", metavar="PT")
a = ap.parse_args()

HD, B, HEADS, SEQ, CTA, WARP_Q = 128, 2, 32, 8192, 128, 32
torch.manual_seed(0)
q = torch.randint(-95, 95, (B, HEADS, SEQ, HD), dtype=torch.int8, device="cuda")
k = torch.randint(-95, 95, (B, HEADS, SEQ, HD), dtype=torch.int8, device="cuda")
q_scale = torch.rand(B, HEADS, SEQ // CTA * (CTA // WARP_Q), dtype=torch.float, device="cuda") + 0.5
k_scale = torch.rand(B, HEADS, SEQ // CTA, dtype=torch.float, device="cuda") + 0.5
v = torch.randn(B, HEADS, HD, SEQ, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
v_scale = torch.rand(B, HEADS, HD, dtype=torch.float, device="cuda") + 0.5

outs = []
for causal in (False, True):
    o = torch.ops.sageattention.fwd(q, k, v, q_scale, k_scale, v_scale, None,
                                    tensor_layout="HND", qk_quant_gran="per_warp",
                                    pv_accum_dtype="fp32", v_layout="linear",
                                    is_causal=causal, sm_scale=1 / (HD ** 0.5),
                                    return_lse=False, out_dtype=torch.float16)
    outs.append(o[0].cpu())
torch.cuda.synchronize()

if a.save:
    torch.save(outs, a.save)
    print("SANITY_SAVED", a.save)
else:
    ref = torch.load(a.ref)
    bad = sum((r.view(torch.int16) != o.view(torch.int16)).sum().item() for r, o in zip(ref, outs))
    print("SANITY_PASS" if bad == 0 else f"SANITY_FAIL mismatched_halfwords={bad}")
