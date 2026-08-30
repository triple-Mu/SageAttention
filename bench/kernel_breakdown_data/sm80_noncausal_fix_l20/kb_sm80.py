#!/usr/bin/env python
"""Run bench/kernel_breakdown.py with the sm80 backend forced on a non-sm80
device (here: L20, cc 8.9, whose default is the sm89 fp8 backend).

Two injections, both process-local, no source edits:
  * sageattention.core.get_plan -> resolve with backend="sm80", so sageattn
    prepares sm80-style tensors (int8 QK + fp16 V, per_thread / fp32 accum).
  * torch.ops.sageattention.fwd -> the same op with backend="sm80" appended,
    since core.sageattn does not forward a backend argument.

Everything else (shape matrix, profiling, JSON schema) is kernel_breakdown's.

Usage: python kb_sm80.py <path-to-kernel_breakdown.py> [kernel_breakdown args]
"""

import runpy
import sys

import torch
import sageattention  # noqa: F401  (loads _C, registers the ops)
import sageattention.core as core
from sageattention._plan import Plan

_plan_op = torch.ops.sageattention.plan


def _get_plan_sm80(cc, head_dim, gran, pv, sv, varlen=False):
    p = Plan(*_plan_op(cc[0], cc[1], head_dim, "sm80", gran, pv, sv, varlen))
    if p.error:
        raise ValueError(p.error)
    return p


core.get_plan = _get_plan_sm80

_ns = torch.ops.sageattention
_real_fwd = _ns.fwd  # materialize the packet before shadowing the attribute


def _fwd_sm80(*args, **kw):
    kw.setdefault("backend", "sm80")
    return _real_fwd(*args, **kw)


_ns.fwd = _fwd_sm80

if len(sys.argv) < 2:
    raise SystemExit("usage: kb_sm80.py <kernel_breakdown.py> [args...]")

kb = sys.argv[1]
sys.argv = [kb] + sys.argv[2:]
runpy.run_path(kb, run_name="__main__")
