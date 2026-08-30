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

"""Single-shape ncu driver for the sm100 attention kernels via the fwd op.

Tensor setup itself launches kernels, so filter by name instead of counting
launches, e.g.:
    ncu -k "regex:qk_int8_sv_f8_attn_kernel_sm100" --launch-skip 2 \
        --launch-count 1 python bench/sm100_review/ws_prof.py --seq 16384
"""

import argparse

import torch

import sageattention  # noqa: F401

ap = argparse.ArgumentParser()
ap.add_argument("--seq", type=int, default=16384)
ap.add_argument("--causal", type=int, default=0)
ap.add_argument("--reps", type=int, default=8)
ap.add_argument("--hd", type=int, default=128)
a = ap.parse_args()

CTA, b, h = 128, 4, 32
q = torch.randint(-95, 95, (b, h, a.seq, a.hd), dtype=torch.int8, device="cuda")
k = torch.randint(-95, 95, (b, h, a.seq, a.hd), dtype=torch.int8, device="cuda")
qs = torch.rand(b, h, a.seq // CTA * 4, dtype=torch.float, device="cuda") + 0.5
ks = torch.rand(b, h, a.seq // CTA, dtype=torch.float, device="cuda") + 0.5
v = torch.randn(b, h, a.hd, a.seq, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
vs = torch.rand(b, h, a.hd, dtype=torch.float, device="cuda") + 0.5
for _ in range(a.reps):
    torch.ops.sageattention.fwd(q, k, v, qs, ks, vs, None,
                                tensor_layout="HND", qk_quant_gran="per_warp",
                                pv_accum_dtype="fp32", v_layout="linear",
                                is_causal=bool(a.causal), sm_scale=1 / (a.hd ** 0.5),
                                return_lse=False, out_dtype=torch.float16)
    torch.cuda.synchronize()
print("PROF_DONE")
