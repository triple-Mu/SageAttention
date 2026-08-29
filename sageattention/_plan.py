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

Import-time snapshot of the C++ dispatch table.

torch.ops.sageattention.plan is the single source of truth (resolve() in
plan.cpp), but calling an op that returns strings inside a torch.compile
traced region breaks fullgraph. So the full decision table is primed into a
plain Python dict here at import time; sageattn() then only does a dict
lookup, which Dynamo constant-folds.

Consequences (documented): SAGEATTN_SM100_TCGEN05 is read once per process,
and the table reflects the kernel families compiled into this build.
"""

from typing import Dict, NamedTuple, Optional, Tuple

import torch


class Plan(NamedTuple):
    backend: str
    qk_quant_gran: str
    pv_accum_dtype: str
    smooth_v: bool
    smooth_v_ignored: bool
    pv_fp8: bool
    v_layout: str
    v_pad_multiple: int
    v_scale_max: float
    need_value_scale: bool
    need_value_mean: bool
    blk_q: int
    warp_q: int
    blk_k: int
    warp_k: int
    error: str


# Real + plausible-future capabilities; unknown ones fall back to a slow path.
_CCS = [(8, 0), (8, 6), (8, 7), (8, 8), (8, 9), (9, 0), (10, 0), (10, 3), (11, 0), (12, 0), (12, 1)]
_HDS = (64, 128)
_GRANS = (None, "per_warp", "per_thread")
_PVS = (None, "fp16", "fp32", "fp16+fp32", "fp32+fp32", "fp32+fp16")
_SVS = (None, False, True)
_VARLENS = (False, True)

# (compute capability, head_dim, qk_quant_gran, pv_accum_dtype, smooth_v, varlen)
PlanKey = Tuple[Tuple[int, int], int, Optional[str], Optional[str], Optional[bool], bool]

PLAN: Dict[PlanKey, Plan] = {}


def _prime() -> None:
    plan_op = torch.ops.sageattention.plan
    for cc in _CCS:
        for hd in _HDS:
            for gran in _GRANS:
                for pv in _PVS:
                    for sv in _SVS:
                        for varlen in _VARLENS:
                            PLAN[(cc, hd, gran, pv, sv, varlen)] = Plan(
                                *plan_op(cc[0], cc[1], hd, None, gran, pv, sv, varlen)
                            )


_prime()


def get_plan(
    cc: Tuple[int, int],
    head_dim: int,
    qk_quant_gran: Optional[str],
    pv_accum_dtype: Optional[str],
    smooth_v: Optional[bool],
    varlen: bool = False,
) -> Plan:
    """Look the plan up in the primed table.

    A key that was never primed (hardware newer than `_CCS`) falls back to the
    C++ op once and is memoized, so every later call is a pure dict lookup.
    That single fallback call returns strings and therefore cannot be traced:
    an unprimed cc must be run once in eager mode before torch.compile
    (fullgraph=True) sees it.
    """
    key: PlanKey = (cc, head_dim, qk_quant_gran, pv_accum_dtype, smooth_v, varlen)
    p = PLAN.get(key)
    if p is None:  # capability not in the primed table (new hardware)
        p = Plan(
            *torch.ops.sageattention.plan(
                cc[0], cc[1], head_dim, None, qk_quant_gran, pv_accum_dtype, smooth_v, varlen
            )
        )
        if not p.error:
            PLAN[key] = p
    if p.error:
        raise ValueError(p.error)
    return p
