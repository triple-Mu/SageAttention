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

Bit-exact comparison harness between a *baseline* SageAttention install (the
pybind ``sageattention._fused`` / ``sageattention._qattn_smXX`` era) and the
current ``torch.ops.sageattention.*`` build.

This file is intentionally self-contained: it imports nothing from the
repository, so it can be copied into any environment that has some version of
``sageattention`` installed.

Two-process workflow
--------------------
1. In the baseline environment::

       python compare_reference.py --dump --golden-dir /shared/golden

2. In the new environment (same GPU model, ideally the same machine)::

       python compare_reference.py --check --golden-dir /shared/golden

   Exit code 0 means every reproducible case matched bit for bit.

Do NOT run this as ``python -m tools.compare_reference``: ``-m`` puts the
current directory at the head of ``sys.path``, so from a SageAttention
checkout ``import sageattention`` would pick up the *source tree* instead of
the installed package under test. Always run it as a plain script path.

Reproducibility contract
------------------------
Every input tensor is generated on the CPU from a per-case fixed seed and then
moved to the GPU, so the inputs are reproducible across machines *as long as
both environments agree on torch's CPU RNG*. torch makes no cross-major-version
compatibility promise there, so each case also stores the hash of its own
inputs. On ``--check`` an input hash mismatch is reported as ``ENV-MISMATCH``
(the case says nothing about the kernels) rather than ``DIFF``.

Sections (``--section``, repeatable)
------------------------------------
quant  low-level quantization ops (the former ``_fused`` surface)
attn   per-arch ``qattn_smXX_*`` kernels on synthetic pre-quantized inputs
e2e    the public python wrapper end to end
equiv  new-build-only internal consistency checks; needs no baseline

Golden layout (``--golden-dir``, default ``./sage-golden``, also settable via
``SAGE_GOLDEN_DIR``)::

    hashes.json    {"meta": {...}, "cases": {key: {"in", "out", "stable"}}}
    tensors.pt     optional (--dump-tensors), used by --check to point at the
                   first differing element

The 2026-08 pre-refactor golden dumps are *not* readable by this tool: they
have no input hashes and no meta block, so ``--import-legacy-golden`` is not
offered. Re-dump from the baseline environment instead.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

SCHEMA_VERSION = 1

RAISED = {"status": "raised"}

_LAYOUT_FLAG = {"NHD": 0, "HND": 1}
_GRAN_FLAG = {"per_warp": 2, "per_thread": 3}
_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


# --------------------------------------------------------------- arch tables

# arch -> [(kernel name, extra tensor arguments)]; the extras are appended after
# key_scale in the historical argument order.
ARCH_ATTN_OPS = {
    80: [
        ("qk_int8_sv_f16_accum_f32_attn", ()),
        ("qk_int8_sv_f16_accum_f16_attn", ()),
        ("qk_int8_sv_f16_accum_f16_attn_inst_buf", ()),
        ("qk_int8_sv_f16_accum_f16_fuse_v_mean_attn", ("value_mean",)),
    ],
    89: [
        ("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn", ("value_scale",)),
        ("qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn", ("value_scale", "value_mean")),
        ("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf", ("value_scale",)),
        ("qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf", ("value_scale",)),
    ],
    90: [
        ("qk_int8_sv_f8_accum_f32_attn_inst_buf", ()),
        ("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf", ("value_scale",)),
    ],
    100: [
        ("qk_int8_sv_f8_accum_f32_attn", ()),
        ("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn", ("value_scale",)),
    ],
    120: [
        ("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn", ("value_scale",)),
        ("qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn", ("value_scale", "value_mean")),
        ("qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf", ("value_scale",)),
    ],
}

# (blk_q, warp_q, blk_k, warp_k) per arch; must match fill_tiles() in plan.cpp.
ARCH_TILES = {
    80: (128, 32, 64, 64),
    89: (128, 32, 64, 64),
    90: (64, 16, 128, 128),
    100: (128, 32, 128, 128),
    120: (128, 32, 64, 64),
}

# (v_layout, v pad multiple); sm80 takes V unquantized in sequence order.
ARCH_V = {
    80: ("seq", 0),
    89: ("mma_k16", 64),
    90: ("mma_k16", 128),
    100: ("linear", 128),
    120: ("mma_k16", 64),
}

# Which compute capabilities each per-arch kernel family can actually run on
# (the cubin/fatbin coverage encoded in the historical setup.py EXT_SERVES).
ARCH_RUNS_ON = {
    80: lambda cc: cc >= (8, 0),
    89: lambda cc: cc == (8, 9) or cc[0] >= 10,
    90: lambda cc: cc == (9, 0),
    100: lambda cc: cc in {(10, 0), (11, 0)},
    120: lambda cc: cc[0] == 12,
}

# Legacy per-arch python wrappers (core.py at the pre-refactor commit).
LEGACY_E2E_FN = {
    80: "sageattn_qk_int8_pv_fp16_cuda",
    89: "sageattn_qk_int8_pv_fp8_cuda",
    90: "sageattn_qk_int8_pv_fp8_cuda_sm90",
    100: "sageattn_qk_int8_pv_fp8_cuda_sm100",
    120: "sageattn_qk_int8_pv_fp8_cuda_sm120",
}

# pv_accum_dtype values each backend accepts, and the (arch, pv) pairs that
# have a fused smooth_v kernel (everything else silently downgrades smooth_v).
E2E_PV = {
    80: ("fp32", "fp16", "fp16+fp32"),
    89: ("fp32", "fp32+fp32", "fp32+fp16"),
    90: ("fp32+fp32",),
    100: ("fp32",),
    120: ("fp32", "fp32+fp16"),
}
E2E_DEFAULT_PV = {80: "fp32", 89: "fp32+fp16", 90: "fp32+fp32", 100: "fp32", 120: "fp32"}
SMOOTH_V_FUSED = {(80, "fp16"), (89, "fp32"), (120, "fp32")}

# (arch, pv_accum_dtype, smooth_v) -> kernel that torch.ops.sageattention.fwd
# dispatches to; mirrors the switch in fwd_cuda.cu. The two base (no
# value_scale) sm90/sm100 kernels are unreachable through fwd.
FWD_TO_QATTN = {
    (80, "fp32", False): "qk_int8_sv_f16_accum_f32_attn",
    (80, "fp16", False): "qk_int8_sv_f16_accum_f16_attn",
    (80, "fp16+fp32", False): "qk_int8_sv_f16_accum_f16_attn_inst_buf",
    (80, "fp16", True): "qk_int8_sv_f16_accum_f16_fuse_v_mean_attn",
    (89, "fp32", False): "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn",
    (89, "fp32", True): "qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn",
    (89, "fp32+fp32", False): "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf",
    (89, "fp32+fp16", False): "qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf",
    (90, "fp32+fp32", False): "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf",
    (100, "fp32", False): "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn",
    (120, "fp32", False): "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn",
    (120, "fp32", True): "qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn",
    (120, "fp32+fp16", False): "qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf",
}


def sm100_tcgen05_enabled():
    return os.environ.get("SAGEATTN_SM100_TCGEN05", "0").upper() in {"1", "TRUE", "YES"}


def resolve_backend(cc, compiled):
    """Python mirror of resolve()'s backend selection (plan.cpp). Returns the
    arch int, or None when this capability has no runnable kernel family."""
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


# ------------------------------------------------------------------ backends
#
# Both backend classes expose the same informal interface (they are deliberately
# unrelated types — the legacy one talks to pybind modules, the new one to
# torch.ops):
#
#   compiled_archs()                                     -> [int]
#   attn(arch, name, q, k, v, out, q_scale, k_scale,
#        extras, tensor_layout, is_causal, gran,
#        sm_scale, return_lse)                           -> lse | None
#   quant_per_block_int8(x, mean, out, scale, sm_scale, blk, layout)
#   quant_per_warp_int8(x, out, scale, blk, wblk, layout)
#   quant_per_thread_int8_q(x, out, scale, blk, wblk, layout)
#   quant_per_thread_int8_k(x, mean, out, scale, blk, wblk, layout)
#   sub_mean(x, mean, out, layout)
#   transpose_pad_v(x, out, layout, permute)
#   scale_fuse_quant(x, out, scale, n, scale_max, layout)
#   mean_scale_fuse_quant(x, out, mean, scale, n, scale_max, layout)
#   e2e(arch, q, k, v, **kwargs)                         -> out | (out, lse)
#
# Only the new backend implements the equiv section (fwd / quant_qk /
# quant_v_fp8 / plan have no legacy counterpart).


class LegacyBackend:
    """Pre-refactor install: pybind modules sageattention._fused and
    sageattention._qattn_smXX."""

    name = "legacy"
    has_equiv = False

    def __init__(self):
        import sageattention._fused as fused

        self.fused = fused
        self.qattn = {}
        for arch in ARCH_ATTN_OPS:
            try:
                self.qattn[arch] = __import__(f"sageattention._qattn_sm{arch}", fromlist=["_"])
            except ImportError:
                pass

    def compiled_archs(self):
        return sorted(self.qattn)

    def attn(
        self,
        arch,
        name,
        q,
        k,
        v,
        out,
        q_scale,
        k_scale,
        extras,
        tensor_layout,
        is_causal,
        gran,
        sm_scale,
        return_lse,
    ):
        fn = getattr(self.qattn[arch], name)
        return fn(
            q,
            k,
            v,
            out,
            q_scale,
            k_scale,
            *extras,
            _LAYOUT_FLAG[tensor_layout],
            int(is_causal),
            _GRAN_FLAG[gran],
            sm_scale,
            int(return_lse),
        )

    def quant_per_block_int8(self, x, mean, out, scale, sm_scale, blk, layout):
        lf = _LAYOUT_FLAG[layout]
        if mean is not None:
            self.fused.quant_per_block_int8_fuse_sub_mean_cuda(x, mean, out, scale, blk, lf)
        elif sm_scale is not None:
            self.fused.quant_per_block_int8_cuda(x, out, scale, sm_scale, blk, lf)
        else:
            self.fused.quant_per_block_int8_cuda(x, out, scale, blk, lf)

    def quant_per_warp_int8(self, x, out, scale, blk, wblk, layout):
        self.fused.quant_per_warp_int8_cuda(x, out, scale, blk, wblk, _LAYOUT_FLAG[layout])

    def quant_per_thread_int8_q(self, x, out, scale, blk, wblk, layout):
        self.fused.quant_per_thread_int8_q_cuda(x, out, scale, blk, wblk, _LAYOUT_FLAG[layout])

    def quant_per_thread_int8_k(self, x, mean, out, scale, blk, wblk, layout):
        lf = _LAYOUT_FLAG[layout]
        if mean is not None:
            self.fused.quant_per_thread_int8_k_fuse_sub_mean_cuda(
                x, mean, out, scale, blk, wblk, lf
            )
        else:
            self.fused.quant_per_thread_int8_k_cuda(x, out, scale, blk, wblk, lf)

    def sub_mean(self, x, mean, out, layout):
        self.fused.sub_mean_cuda(x, mean, out, _LAYOUT_FLAG[layout])

    def transpose_pad_v(self, x, out, layout, permute):
        lf = _LAYOUT_FLAG[layout]
        if permute:
            self.fused.transpose_pad_permute_cuda(x, out, lf)
        else:
            self.fused.transpose_pad_cuda(x, out, lf)

    def scale_fuse_quant(self, x, out, scale, n, scale_max, layout):
        self.fused.scale_fuse_quant_cuda(x, out, scale, n, scale_max, _LAYOUT_FLAG[layout])

    def mean_scale_fuse_quant(self, x, out, mean, scale, n, scale_max, layout):
        self.fused.mean_scale_fuse_quant_cuda(
            x, out, mean, scale, n, scale_max, _LAYOUT_FLAG[layout]
        )

    def e2e(self, arch, q, k, v, **kw):
        from sageattention import core

        fn = getattr(core, LEGACY_E2E_FN[arch])
        return fn(q, k, v, **kw)


class NewBackend:
    """Post-refactor install: a single _C extension registering
    torch.ops.sageattention.*."""

    name = "new"
    has_equiv = True

    def __init__(self):
        import sageattention  # noqa: F401  (loads _C, registers the ops)

        self.ops = torch.ops.sageattention
        self.sageattn = sageattention.sageattn

    def compiled_archs(self):
        return sorted(self.ops.compiled_archs())

    def attn(
        self,
        arch,
        name,
        q,
        k,
        v,
        out,
        q_scale,
        k_scale,
        extras,
        tensor_layout,
        is_causal,
        gran,
        sm_scale,
        return_lse,
    ):
        fn = getattr(self.ops, f"qattn_sm{arch}_{name}")
        lse = fn(
            q,
            k,
            v,
            out,
            q_scale,
            k_scale,
            *extras,
            tensor_layout,
            bool(is_causal),
            gran,
            sm_scale,
            bool(return_lse),
        )
        # the C++ side returns an empty CPU placeholder when return_lse is off
        return lse if (return_lse and lse is not None and lse.numel()) else None

    def quant_per_block_int8(self, x, mean, out, scale, sm_scale, blk, layout):
        self.ops.quant_per_block_int8(x, mean, out, scale, sm_scale, blk, layout)

    def quant_per_warp_int8(self, x, out, scale, blk, wblk, layout):
        self.ops.quant_per_warp_int8(x, out, scale, blk, wblk, layout)

    def quant_per_thread_int8_q(self, x, out, scale, blk, wblk, layout):
        self.ops.quant_per_thread_int8_q(x, out, scale, blk, wblk, layout)

    def quant_per_thread_int8_k(self, x, mean, out, scale, blk, wblk, layout):
        self.ops.quant_per_thread_int8_k(x, mean, out, scale, blk, wblk, layout)

    def sub_mean(self, x, mean, out, layout):
        self.ops.sub_mean(x, mean, out, layout)

    def transpose_pad_v(self, x, out, layout, permute):
        self.ops.transpose_pad_v(x, out, layout, permute)

    def scale_fuse_quant(self, x, out, scale, n, scale_max, layout):
        self.ops.scale_fuse_quant(x, out, scale, n, scale_max, layout)

    def mean_scale_fuse_quant(self, x, out, mean, scale, n, scale_max, layout):
        self.ops.mean_scale_fuse_quant(x, out, mean, scale, n, scale_max, layout)

    def e2e(self, arch, q, k, v, **kw):
        # the unified entry point picks `arch` itself from the device's cc
        return self.sageattn(q, k, v, **kw)


def detect_backend(forced=None):
    if forced == "legacy":
        return LegacyBackend()
    if forced == "new":
        return NewBackend()
    try:
        import sageattention  # noqa: F401

        torch.ops.sageattention.compiled_archs()
        return NewBackend()
    except Exception:
        pass
    try:
        import sageattention._fused  # noqa: F401

        return LegacyBackend()
    except Exception as exc:
        raise SystemExit(
            "cannot classify the installed sageattention: neither "
            "torch.ops.sageattention.compiled_archs (new build) nor "
            f"sageattention._fused (legacy build) is available ({exc!r}). "
            "Use --backend to force one."
        )


# --------------------------------------------------------------------- utils


def case_seed(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def gen(shape, key, *, dtype=None, kind="randn", device="cuda"):
    """Deterministic input tensor: generated on the CPU from a hash of `key`,
    then moved to the device (so the bytes do not depend on the GPU)."""
    g = torch.Generator(device="cpu").manual_seed(case_seed(key))
    if kind == "randn":
        return torch.randn(shape, generator=g, dtype=torch.float32).to(dtype).to(device)
    if kind == "int8":
        return torch.randint(-127, 128, shape, generator=g, dtype=torch.int8).to(device)
    if kind == "scale":
        return (torch.rand(shape, generator=g, dtype=torch.float32) * 0.05 + 1e-4).to(device)
    raise ValueError(kind)


def thash(t):
    if t is None:
        return "none"
    if isinstance(t, str):
        return t
    return hashlib.sha256(
        t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()[:24]


def ceil_div(a, b):
    return (a + b - 1) // b


def qkv_shape(b, h, seq, d, layout):
    return (b, seq, h, d) if layout == "NHD" else (b, h, seq, d)


def vt_shape(b, h, d, padded, layout):
    """Shape of the transposed value buffer the fp8 path works on."""
    return (b, h, d, padded) if layout == "HND" else (b, d, h, padded)


def seq_nh_dims(layout):
    return (1, 2) if layout == "NHD" else (2, 1)


def scale_lens(gran, qo_len, kv_len, blk_q, warp_q, blk_k, warp_k):
    if gran == "per_warp":
        return (ceil_div(qo_len, blk_q) * (blk_q // warp_q), ceil_div(kv_len, blk_k))
    return (
        ceil_div(qo_len, blk_q) * (blk_q // warp_q) * 8,
        ceil_div(kv_len, blk_k) * (blk_k // warp_k) * 4,
    )


# ------------------------------------------------------------- quant section

QUANT_SHAPES = [(2, 4, 63), (2, 4, 64), (1, 8, 1000), (2, 4, 1024)]
QUANT_FNS = [
    "pb_q",
    "pb_k",
    "pb_k_sm",
    "pw",
    "pt_q",
    "pt_k",
    "pt_k_sm",
    "sub_mean",
    "tpp",
    "tp",
    "sq",
    "msq",
]


def quant_cases(env, subset):
    # sq/msq emit fp8 through cvt.rn.satfinite.e4m3x2, which needs sm_89+; the
    # launchers reject lower capabilities outright (the real pipeline never
    # calls them there either).
    fp8_ok = env["cc"] >= (8, 9)
    for fn in QUANT_FNS:
        if fn in ("sq", "msq") and not fp8_ok:
            continue
        for hd in (64, 128):
            for layout in ("NHD", "HND"):
                for dt in ("float16", "bfloat16"):
                    for b, h, seq in QUANT_SHAPES:
                        if dt == "bfloat16" and (layout == "NHD" or seq not in (64, 1000)):
                            continue
                        if subset and seq > 128:
                            continue
                        yield dict(fn=fn, hd=hd, layout=layout, dt=dt, b=b, h=h, seq=seq)


def quant_build(be, c):
    dt = _DTYPES[c["dt"]]
    key = json.dumps(c, sort_keys=True)
    b, h, seq, hd, layout = c["b"], c["h"], c["seq"], c["hd"], c["layout"]
    inputs = {}
    if c["fn"] in ("sq", "msq"):
        # the fp8 quantizers consume the *transposed* buffer; the padded tail
        # must be zero (that is what transpose_pad produces) or garbage leaks
        # into the per-channel amax
        padded = ceil_div(seq, 64) * 64
        x = gen(vt_shape(b, h, hd, padded, layout), key + "/xt", dtype=dt)
        if padded != seq:
            x[..., seq:] = 0
        inputs["x"] = x
    else:
        inputs["x"] = gen(qkv_shape(b, h, seq, hd, layout), key + "/x", dtype=dt)
    if c["fn"] in ("pb_k_sm", "pt_k_sm"):
        inputs["mean"] = gen((b, h, hd), key + "/mean", dtype=dt)
    if c["fn"] == "sub_mean":
        inputs["mean"] = inputs["x"].mean(dim=seq_nh_dims(layout)[0])
    return inputs


def quant_execute(be, c, inputs):
    fn, hd, layout = c["fn"], c["hd"], c["layout"]
    b, h, seq = c["b"], c["h"], c["seq"]
    dt = _DTYPES[c["dt"]]
    x = inputs["x"]
    dev = x.device

    if fn in ("pb_q", "pb_k", "pb_k_sm"):
        blk = 128 if fn == "pb_q" else 64
        xi = torch.empty(x.shape, dtype=torch.int8, device=dev)
        sc = torch.empty((b, h, ceil_div(seq, blk)), dtype=torch.float32, device=dev)
        be.quant_per_block_int8(
            x,
            inputs.get("mean"),
            xi,
            sc,
            hd**-0.5 * 1.44269504 if fn == "pb_q" else None,
            blk,
            layout,
        )
        out = {"int8": xi, "scale": sc}
    elif fn == "pw":
        blk, wblk = 128, 32
        xi = torch.empty(x.shape, dtype=torch.int8, device=dev)
        sc = torch.empty(
            (b, h, ceil_div(seq, blk) * (blk // wblk)), dtype=torch.float32, device=dev
        )
        be.quant_per_warp_int8(x, xi, sc, blk, wblk, layout)
        out = {"int8": xi, "scale": sc}
    elif fn == "pt_q":
        blk, wblk = 128, 32
        xi = torch.empty(x.shape, dtype=torch.int8, device=dev)
        sc = torch.empty(
            (b, h, ceil_div(seq, blk) * (blk // wblk) * 8), dtype=torch.float32, device=dev
        )
        be.quant_per_thread_int8_q(x, xi, sc, blk, wblk, layout)
        out = {"int8": xi, "scale": sc}
    elif fn in ("pt_k", "pt_k_sm"):
        blk, wblk = 64, 64
        xi = torch.empty(x.shape, dtype=torch.int8, device=dev)
        sc = torch.empty(
            (b, h, ceil_div(seq, blk) * (blk // wblk) * 4), dtype=torch.float32, device=dev
        )
        be.quant_per_thread_int8_k(x, inputs.get("mean"), xi, sc, blk, wblk, layout)
        out = {"int8": xi, "scale": sc}
    elif fn == "sub_mean":
        sm = torch.empty(x.shape, dtype=torch.float16, device=dev)
        be.sub_mean(x, inputs["mean"], sm, layout)
        out = {"smoothed": sm}
    elif fn in ("tpp", "tp"):
        y = torch.empty(vt_shape(b, h, hd, ceil_div(seq, 64) * 64, layout), dtype=dt, device=dev)
        be.transpose_pad_v(x, y, layout, fn == "tpp")
        out = {"t": y}
    else:  # sq / msq
        y = torch.zeros(x.shape, dtype=torch.float8_e4m3fn, device=dev)
        sc = torch.empty((b, h, hd), dtype=torch.float32, device=dev)
        if fn == "sq":
            be.scale_fuse_quant(x, y, sc, seq, 448.0, layout)
            out = {"fp8": y, "scale": sc}
        else:
            mean = torch.empty((b, h, hd), dtype=torch.float32, device=dev)
            be.mean_scale_fuse_quant(x, y, mean, sc, seq, 448.0, layout)
            out = {"fp8": y, "scale": sc, "mean": mean}
    torch.cuda.synchronize()
    return out


# -------------------------------------------------------------- attn section

# (b, h_qo, h_kv, qo_len, kv_len)
ATTN_SHAPES = [
    (1, 2, 2, 63, 63),
    (2, 8, 2, 128, 128),  # GQA
    (1, 4, 4, 1000, 1000),
    (2, 8, 8, 2048, 2048),
    (1, 2, 2, 77, 199),  # cross attention
    (1, 2, 2, 33, 33),  # kv_len < CTA_K
]


def attn_cases(env, subset):
    for arch in env["runnable_archs"]:
        for name, extras in ARCH_ATTN_OPS[arch]:
            for hd in (64, 128):
                for layout in ("NHD", "HND"):
                    for causal in (0, 1):
                        for gran in ("per_warp", "per_thread"):
                            for lse in (0, 1):
                                for out_dt in ("float16", "bfloat16"):
                                    for b, hq, hk, qo, kv in ATTN_SHAPES:
                                        if causal and qo != kv:
                                            continue
                                        # bf16 output only on a slice of the grid
                                        if out_dt == "bfloat16" and (
                                            layout == "NHD" or gran == "per_warp" or lse == 1
                                        ):
                                            continue
                                        if subset and qo > 256:
                                            continue
                                        yield dict(
                                            arch=arch,
                                            name=name,
                                            hd=hd,
                                            layout=layout,
                                            causal=causal,
                                            gran=gran,
                                            lse=lse,
                                            out=out_dt,
                                            b=b,
                                            hq=hq,
                                            hk=hk,
                                            qo=qo,
                                            kv=kv,
                                        )


def attn_warp_q(arch, name, hd):
    warp_q = ARCH_TILES[arch][1]
    if arch == 80 and hd == 128 and name.endswith("inst_buf"):
        return 16  # the sm80 fp16+fp32 inst_buf kernel halves WARP_Q at hd=128
    return warp_q


def attn_build(be, c):
    arch, name, hd, layout = c["arch"], c["name"], c["hd"], c["layout"]
    b, hq, hk, qo, kv = c["b"], c["hq"], c["hk"], c["qo"], c["kv"]
    out_dtype = _DTYPES[c["out"]]
    key = json.dumps(c, sort_keys=True)
    extras = dict(ARCH_ATTN_OPS[arch])[name]
    v_layout, pad_mult = ARCH_V[arch]

    blk_q, _, blk_k, warp_k = ARCH_TILES[arch]
    warp_q = attn_warp_q(arch, name, hd)
    qs_len, ks_len = scale_lens(c["gran"], qo, kv, blk_q, warp_q, blk_k, warp_k)

    inputs = {
        "q": gen(qkv_shape(b, hq, qo, hd, layout), key + "/q", kind="int8"),
        "k": gen(qkv_shape(b, hk, kv, hd, layout), key + "/k", kind="int8"),
        "q_scale": gen((b, hq, qs_len), key + "/qs", kind="scale"),
        "k_scale": gen((b, hk, ks_len), key + "/ks", kind="scale"),
    }

    if v_layout == "seq":  # sm80: fp16 V in the original token order
        if "value_mean" in extras:
            v_src = gen(qkv_shape(b, hk, kv, hd, layout), key + "/v", dtype=out_dtype)
            vm = v_src.mean(dim=seq_nh_dims(layout)[0])
            v = torch.empty(v_src.shape, dtype=torch.float16, device=v_src.device)
            be.sub_mean(v_src, vm, v, layout)
            inputs["v"] = v
            inputs["value_mean"] = vm
        else:
            inputs["v"] = gen(qkv_shape(b, hk, kv, hd, layout), key + "/v", dtype=torch.float16)
        return inputs

    # fp8 archs: build V through the (separately compared) quantization path,
    # exactly the way the pre-refactor pipeline did — zero-pad the sequence in
    # python first, then transpose+quantize. Doing it this way keeps both
    # backends on the same code path and keeps the transpose kernel's own
    # 64-token alignment rule satisfied for the 128-aligned families.
    padded = ceil_div(kv, pad_mult) * pad_mult
    v_src = gen(qkv_shape(b, hk, kv, hd, layout), key + "/v", dtype=torch.float16)
    if padded != kv:
        pad_shape = qkv_shape(b, hk, padded - kv, hd, layout)
        v_src = torch.cat(
            [v_src, torch.zeros(pad_shape, dtype=v_src.dtype, device=v_src.device)],
            dim=seq_nh_dims(layout)[0],
        )
    v_t = torch.empty(vt_shape(b, hk, hd, padded, layout), dtype=torch.float16, device=v_src.device)
    be.transpose_pad_v(v_src, v_t, layout, v_layout == "mma_k16")
    # 2.25 is the sm89/sm120 headroom clamp for the f8f8f16 (fp32+fp16) kernels
    scale_max = 2.25 if (arch in (89, 120) and "accum_f16" in name) else 448.0
    v_fp8 = torch.zeros(v_t.shape, dtype=torch.float8_e4m3fn, device=v_t.device)
    v_scale = torch.empty((b, hk, hd), dtype=torch.float32, device=v_t.device)
    if "value_mean" in extras:
        vm = torch.empty((b, hk, hd), dtype=torch.float32, device=v_t.device)
        be.mean_scale_fuse_quant(v_t, v_fp8, vm, v_scale, padded, scale_max, layout)
        inputs["value_mean"] = vm
    else:
        be.scale_fuse_quant(v_t, v_fp8, v_scale, padded, scale_max, layout)
    inputs["v"] = v_fp8
    if "value_scale" in extras:
        inputs["value_scale"] = v_scale
    torch.cuda.synchronize()
    return inputs


def attn_execute(be, c, inputs):
    arch, name = c["arch"], c["name"]
    extras = dict(ARCH_ATTN_OPS[arch])[name]
    out_dtype = _DTYPES[c["out"]]
    q = inputs["q"]
    out = torch.empty(q.shape, dtype=out_dtype, device=q.device)
    lse = be.attn(
        arch,
        name,
        q,
        inputs["k"],
        inputs["v"],
        out,
        inputs["q_scale"],
        inputs["k_scale"],
        [inputs[e] for e in extras],
        c["layout"],
        c["causal"],
        c["gran"],
        c["hd"] ** -0.5,
        c["lse"],
    )
    torch.cuda.synchronize()
    return {"o": out, "lse": lse if c["lse"] else None}


# --------------------------------------------------------------- e2e section


def e2e_cases(env, subset):
    arch = env["e2e_arch"]
    if arch is None:
        return
    for hd in (64, 96, 128):
        for layout in ("HND", "NHD"):
            for causal in (False, True):
                for lse in (False, True):
                    for smooth_k in (True, False):
                        for gran in ("per_warp", "per_thread"):
                            for pv in E2E_PV[arch]:
                                svs = (False, True) if (arch, pv) in SMOOTH_V_FUSED else (False,)
                                for smooth_v in svs:
                                    if not smooth_k and (layout == "NHD" or lse):
                                        continue
                                    if pv != E2E_DEFAULT_PV[arch] and hd == 96:
                                        continue
                                    if subset and (hd == 96 or gran == "per_warp"):
                                        continue
                                    yield dict(
                                        arch=arch,
                                        hd=hd,
                                        layout=layout,
                                        causal=causal,
                                        lse=lse,
                                        smooth_k=smooth_k,
                                        gran=gran,
                                        pv=pv,
                                        smooth_v=smooth_v,
                                    )


def e2e_build(be, c):
    key = json.dumps(c, sort_keys=True)
    b, h, seq = 2, 4, 512
    shape = qkv_shape(b, h, seq, c["hd"], c["layout"])
    return {n: gen(shape, f"{key}/{n}", dtype=torch.float16) for n in ("q", "k", "v")}


def e2e_execute(be, c, inputs):
    r = be.e2e(
        c["arch"],
        inputs["q"],
        inputs["k"],
        inputs["v"],
        tensor_layout=c["layout"],
        is_causal=c["causal"],
        return_lse=c["lse"],
        qk_quant_gran=c["gran"],
        pv_accum_dtype=c["pv"],
        smooth_k=c["smooth_k"],
        smooth_v=c["smooth_v"],
    )
    torch.cuda.synchronize()
    if c["lse"]:
        return {"o": r[0], "lse": r[1]}
    return {"o": r}


# ------------------------------------------------------------- equiv section
#
# New-build-only self-consistency checks. They need no baseline: each case
# compares two paths inside the same process and reports "yes"/"no".

EQUIV_ATTN_SHAPES = [(1, 2, 2, 128, 128), (2, 4, 2, 577, 577), (1, 2, 2, 77, 199)]


def equiv_cases(env, subset):
    arch = env["e2e_arch"]
    if arch is not None:
        for hd in (64, 128):
            for layout in ("HND", "NHD"):
                for causal in (False, True):
                    for lse in (False, True):
                        for gran in ("per_warp", "per_thread"):
                            for pv in E2E_PV[arch]:
                                svs = (False, True) if (arch, pv) in SMOOTH_V_FUSED else (False,)
                                for smooth_v in svs:
                                    for b, hq, hk, qo, kv in EQUIV_ATTN_SHAPES:
                                        if causal and qo != kv:
                                            continue
                                        if subset and qo > 256:
                                            continue
                                        yield dict(
                                            kind="fwd_vs_qattn",
                                            arch=arch,
                                            hd=hd,
                                            layout=layout,
                                            causal=causal,
                                            lse=lse,
                                            gran=gran,
                                            pv=pv,
                                            smooth_v=smooth_v,
                                            b=b,
                                            hq=hq,
                                            hk=hk,
                                            qo=qo,
                                            kv=kv,
                                        )
    # quant_v_fp8(pad_multiple=128) must be bit-identical to python-side
    # zero padding followed by the 64-aligned quantization, which is what the
    # pre-refactor sm90/sm100 path did. Needs the fp8 converter (sm_89+).
    if env["cc"] >= (8, 9):
        for layout in ("HND", "NHD"):
            for v_layout in ("mma_k16", "linear"):
                for hd in (64, 128):
                    for kv in (64, 320, 1000):
                        yield dict(kind="v_pad128", layout=layout, v_layout=v_layout, hd=hd, kv=kv)
    yield dict(kind="plan_backend")


def equiv_build(be, c):
    if c["kind"] == "fwd_vs_qattn":
        key = json.dumps(c, sort_keys=True)
        b, hq, hk, qo, kv, hd, layout = (
            c["b"],
            c["hq"],
            c["hk"],
            c["qo"],
            c["kv"],
            c["hd"],
            c["layout"],
        )
        return {
            "q": gen(qkv_shape(b, hq, qo, hd, layout), key + "/q", dtype=torch.float16),
            "k": gen(qkv_shape(b, hk, kv, hd, layout), key + "/k", dtype=torch.float16),
            "v": gen(qkv_shape(b, hk, kv, hd, layout), key + "/v", dtype=torch.float16),
        }
    if c["kind"] == "v_pad128":
        key = json.dumps(c, sort_keys=True)
        shape = qkv_shape(1, 2, c["kv"], c["hd"], c["layout"])
        return {"v": gen(shape, key + "/v", dtype=torch.float16)}
    return {}


def _plan_of(cc, hd, gran, pv, smooth_v):
    fields = [
        "backend",
        "qk_quant_gran",
        "pv_accum_dtype",
        "smooth_v",
        "smooth_v_ignored",
        "pv_fp8",
        "v_layout",
        "v_pad_multiple",
        "v_scale_max",
        "need_value_scale",
        "need_value_mean",
        "blk_q",
        "warp_q",
        "blk_k",
        "warp_k",
        "error",
    ]
    vals = torch.ops.sageattention.plan(cc[0], cc[1], hd, None, gran, pv, smooth_v)
    return dict(zip(fields, vals))


def equiv_execute(be, c, inputs):
    ops = torch.ops.sageattention
    if c["kind"] == "plan_backend":
        bad = []
        compiled = set(be.compiled_archs())
        for cc in [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (11, 0), (12, 0), (12, 1)]:
            p = _plan_of(cc, 128, None, None, None)
            mine = resolve_backend(cc, compiled)
            got = None if p["error"] else int(p["backend"][2:])
            if mine != got:
                bad.append(f"sm_{cc[0]}{cc[1]}: mirror={mine} plan={got}")
        return {"equal": "yes" if not bad else "no: " + "; ".join(bad)}

    if c["kind"] == "v_pad128":
        v, kv, layout, v_layout = inputs["v"], c["kv"], c["layout"], c["v_layout"]
        seq_dim = seq_nh_dims(layout)[0]
        padded = ceil_div(kv, 128) * 128
        v_ref_src = v
        if padded != kv:
            pad_shape = list(v.shape)
            pad_shape[seq_dim] = padded - kv
            v_ref_src = torch.cat(
                [v, torch.zeros(pad_shape, dtype=v.dtype, device=v.device)], dim=seq_dim
            )
        ref = ops.quant_v_fp8(
            v_ref_src,
            tensor_layout=layout,
            v_layout=v_layout,
            scale_max=448.0,
            smooth_v=False,
            pad_multiple=64,
        )
        got = ops.quant_v_fp8(
            v,
            tensor_layout=layout,
            v_layout=v_layout,
            scale_max=448.0,
            smooth_v=False,
            pad_multiple=128,
        )
        torch.cuda.synchronize()
        if not (torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])):
            return {"equal": "no: quant_v_fp8(pad_multiple=128) != cat-pad + 64-align"}
        # white-box: every token past the real sequence must be exactly zero
        # (mma_k16 permutes inside 16-token groups, so round the bound up)
        bound = ceil_div(kv, 16) * 16 if v_layout == "mma_k16" else kv
        tail = got[0][..., bound:]
        if tail.numel() and tail.view(torch.uint8).any():
            return {"equal": "no: padded fp8 tail is not zero"}
        return {"equal": "yes"}

    # fwd vs the per-arch op it dispatches to
    arch, hd, layout = c["arch"], c["hd"], c["layout"]
    p = _plan_of(
        torch.cuda.get_device_capability(inputs["q"].device.index),
        hd,
        c["gran"],
        c["pv"],
        c["smooth_v"],
    )
    if p["error"]:
        return {"equal": "skip: " + p["error"]}
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    q_int8, q_scale, k_int8, k_scale = ops.quant_qk(
        q,
        k,
        None,
        tensor_layout=layout,
        qk_quant_gran=p["qk_quant_gran"],
        blk_q=p["blk_q"],
        warp_q=p["warp_q"],
        blk_k=p["blk_k"],
        warp_k=p["warp_k"],
    )
    value_scale = value_mean = None
    if p["pv_fp8"]:
        v_prep, value_scale, value_mean = ops.quant_v_fp8(
            v,
            tensor_layout=layout,
            v_layout=p["v_layout"],
            scale_max=p["v_scale_max"],
            smooth_v=p["smooth_v"],
            pad_multiple=p["v_pad_multiple"],
        )
    elif p["smooth_v"]:
        v_prep, value_mean = ops.sub_mean_v(v, tensor_layout=layout)
    else:
        v_prep = v.to(torch.float16)

    out_a, lse_a = ops.fwd(
        q_int8,
        k_int8,
        v_prep,
        q_scale,
        k_scale,
        value_scale,
        value_mean,
        tensor_layout=layout,
        qk_quant_gran=p["qk_quant_gran"],
        pv_accum_dtype=p["pv_accum_dtype"],
        v_layout=p["v_layout"],
        is_causal=c["causal"],
        sm_scale=hd**-0.5,
        return_lse=c["lse"],
        out_dtype=torch.float16,
    )

    name = FWD_TO_QATTN[(arch, p["pv_accum_dtype"], p["smooth_v"])]
    extras = dict(ARCH_ATTN_OPS[arch])[name]
    avail = {"value_scale": value_scale, "value_mean": value_mean}
    out_b = torch.empty(q_int8.shape, dtype=torch.float16, device=q.device)
    lse_b = be.attn(
        arch,
        name,
        q_int8,
        k_int8,
        v_prep,
        out_b,
        q_scale,
        k_scale,
        [avail[e] for e in extras],
        layout,
        c["causal"],
        p["qk_quant_gran"],
        hd**-0.5,
        c["lse"],
    )
    torch.cuda.synchronize()
    if not torch.equal(out_a, out_b):
        return {"equal": f"no: fwd != {name} (out)"}
    if c["lse"] and not torch.equal(lse_a, lse_b):
        return {"equal": f"no: fwd != {name} (lse)"}
    return {"equal": "yes"}


SECTIONS = {
    "quant": (quant_cases, quant_build, quant_execute),
    "attn": (attn_cases, attn_build, attn_execute),
    "e2e": (e2e_cases, e2e_build, e2e_execute),
    "equiv": (equiv_cases, equiv_build, equiv_execute),
}


# -------------------------------------------------------------------- driver


def build_env(be):
    cc = torch.cuda.get_device_capability(torch.cuda.current_device())
    compiled = be.compiled_archs()
    return {
        "cc": cc,
        "compiled_archs": compiled,
        # a kernel family is testable only if it is both compiled in and has
        # cubins for this GPU
        "runnable_archs": [a for a in compiled if ARCH_RUNS_ON[a](cc)],
        "e2e_arch": resolve_backend(cc, set(compiled)),
    }


def git_commit():
    for d in (Path(__file__).resolve().parent, Path.cwd()):
        try:
            out = subprocess.run(
                ["git", "-C", str(d), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:  # noqa: BLE001  (git missing / not a repo)
            pass
    return "unknown"


def build_meta(be, env, args):
    import sageattention

    return {
        "schema_version": SCHEMA_VERSION,
        "backend": be.name,
        "sageattention_file": str(sageattention.__file__),
        "sageattention_version": str(getattr(sageattention, "__version__", "unknown")),
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "cc": list(env["cc"]),
        "compiled_archs": env["compiled_archs"],
        "runnable_archs": env["runnable_archs"],
        "e2e_arch": env["e2e_arch"],
        "git_commit": git_commit(),
        "sections": args.section,
        "subset": bool(args.subset),
        "repeat": args.repeat,
    }


# meta keys whose disagreement invalidates a comparison outright vs. keys that
# are merely worth a warning
META_FATAL = ("schema_version",)
META_WARN = (
    "cc",
    "device_name",
    "torch_version",
    "runnable_archs",
    "e2e_arch",
    "sections",
    "subset",
)


def run_all(be, env, args, keep_tensors):
    results = {}
    tensors = {}
    for section in args.section:
        if section == "equiv" and not be.has_equiv:
            print(f"[skip] section equiv: not available on the {be.name} backend")
            continue
        cases, build, execute = SECTIONS[section]
        n = 0
        for c in cases(env, args.subset):
            key = section + "/" + json.dumps(c, sort_keys=True)
            n += 1
            try:
                inputs = build(be, c)
            except Exception:  # noqa: BLE001
                # exception types are deliberately not recorded: the refactor
                # turned a batch of asserts into ValueError and the type churn
                # would drown the real diffs
                results[key] = {"in": {}, "out": dict(RAISED), "stable": True}
                continue
            in_h = {k: thash(v) for k, v in inputs.items()}
            runs = []
            for _ in range(args.repeat):
                try:
                    runs.append(execute(be, c, inputs))
                except Exception:  # noqa: BLE001
                    runs.append(None)
            hashes = [
                dict(RAISED) if r is None else {k: thash(v) for k, v in r.items()} for r in runs
            ]
            results[key] = {
                "in": in_h,
                "out": hashes[0],
                "stable": all(h == hashes[0] for h in hashes),
            }
            if keep_tensors and runs[0] is not None:
                small = {
                    k: v
                    for k, v in runs[0].items()
                    if torch.is_tensor(v) and v.numel() <= (1 << 18)
                }
                if small:
                    tensors[key] = {k: v.detach().cpu() for k, v in small.items()}
        print(f"[run] {section}: {n} cases")
    return results, tensors


def locate_diff(golden_t, current_t):
    """First differing element of two same-shape tensors, as text."""
    if golden_t.shape != current_t.shape or golden_t.dtype != current_t.dtype:
        return (
            f"shape/dtype differ: golden {tuple(golden_t.shape)}/{golden_t.dtype} "
            f"vs current {tuple(current_t.shape)}/{current_t.dtype}"
        )
    a = golden_t.contiguous().view(torch.uint8).flatten()
    b = current_t.contiguous().cpu().view(torch.uint8).flatten()
    neq = (a != b).nonzero()
    if not neq.numel():
        return "byte-identical (hash mismatch is a harness bug)"
    byte_idx = int(neq[0].item())
    elem = byte_idx // golden_t.element_size()
    idx = []
    rest = elem
    for dim in reversed(golden_t.shape):
        idx.append(rest % dim)
        rest //= dim
    idx = tuple(reversed(idx))
    try:
        gv = golden_t.flatten()[elem].float().item()
        cv = current_t.cpu().flatten()[elem].float().item()
        vals = f"golden={gv!r} current={cv!r}"
    except Exception:  # noqa: BLE001  (dtype without a float cast)
        vals = f"golden_byte={int(a[byte_idx])} current_byte={int(b[byte_idx])}"
    return f"first diff at {idx} ({elem} of {golden_t.numel()}): {vals}"


def do_check(be, env, args, golden_dir, results, tensors):
    hpath = golden_dir / "hashes.json"
    if not hpath.exists():
        raise SystemExit(
            f"no golden dump at {hpath}; run --dump in the baseline "
            "environment first (or point --golden-dir at it)"
        )
    blob = json.loads(hpath.read_text())
    if "meta" not in blob or "cases" not in blob:
        raise SystemExit(
            f"{golden_dir}/hashes.json has no meta/cases block: it predates this "
            f"tool (schema_version {SCHEMA_VERSION}). Re-dump it from the "
            "baseline environment."
        )
    gmeta, golden = blob["meta"], blob["cases"]
    meta = build_meta(be, env, args)
    for k in META_FATAL:
        if gmeta.get(k) != meta.get(k):
            raise SystemExit(
                f"golden {k}={gmeta.get(k)!r} but this tool is {meta.get(k)!r}; "
                "the whole dump is void, re-dump it from the baseline environment"
            )
    for k in META_WARN:
        if gmeta.get(k) != meta.get(k):
            print(f"WARNING: meta.{k}: golden={gmeta.get(k)!r} now={meta.get(k)!r}")
    print(
        f"golden: backend={gmeta.get('backend')} "
        f"sageattention={gmeta.get('sageattention_version')} "
        f"@{gmeta.get('git_commit')} torch={gmeta.get('torch_version')}"
    )

    gtensors = {}
    tpath = golden_dir / "tensors.pt"
    if tpath.exists():
        gtensors = torch.load(tpath, map_location="cpu", weights_only=True)

    wl = [w for w in args.expect_diff.split(",") if w]
    n_ok = n_diff = n_wl = n_skip = n_env = n_missing = 0
    for key, g in sorted(golden.items()):
        section = key.split("/", 1)[0]
        # a section that was not run here (--section, or equiv on a legacy
        # install) is skipped rather than reported as a thousand MISSINGs
        if section not in args.section or (section == "equiv" and not be.has_equiv):
            n_skip += 1
            continue
        if not g["stable"]:
            n_skip += 1
            continue
        r = results.get(key)
        if r is None:
            print(f"MISSING: {key}")
            n_missing += 1
            continue
        if r["in"] != g["in"]:
            print(f"ENV-MISMATCH: {key}\n  golden_in={g['in']}\n  now_in    ={r['in']}")
            n_env += 1
            continue
        if r["out"] == g["out"]:
            n_ok += 1
        elif any(w in key for w in wl):
            n_wl += 1
        else:
            print(f"DIFF: {key}\n  golden={g['out']}\n  now   ={r['out']}")
            for name, gt in gtensors.get(key, {}).items():
                ct = tensors.get(key, {}).get(name)
                if ct is not None and thash(gt) != thash(ct):
                    print(f"  {name}: {locate_diff(gt, ct)}")
            n_diff += 1
    extra = [k for k in results if k not in golden]
    print(
        f"ok={n_ok} diff={n_diff} env_mismatch={n_env} missing={n_missing} "
        f"whitelisted={n_wl} skipped={n_skip} extra={len(extra)}"
    )
    if extra:
        print(f"note: {len(extra)} case(s) exist here but not in the golden (e.g. {extra[0]})")
    return 1 if (n_diff or n_env or n_missing) else 0


def report_equiv(results):
    bad = [
        (k, r)
        for k, r in sorted(results.items())
        if k.startswith("equiv/") and r["out"] != {"equal": "yes"}
    ]
    total = sum(1 for k in results if k.startswith("equiv/"))
    if not total:
        return 0
    for k, r in bad:
        print(f"EQUIV-FAIL: {k}\n  {r['out']}")
    print(f"equiv: {total - len(bad)}/{total} passed")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(
        description="bit-exact comparison against a baseline sageattention install"
    )
    ap.add_argument("--dump", action="store_true", help="write the golden dump")
    ap.add_argument("--check", action="store_true", help="compare against the golden dump")
    ap.add_argument("--golden-dir", default=os.environ.get("SAGE_GOLDEN_DIR", "sage-golden"))
    ap.add_argument(
        "--dump-tensors",
        action="store_true",
        help="also store the small output tensors (lets --check point at "
        "the first differing element; ~150 MB on a full run)",
    )
    ap.add_argument(
        "--backend", choices=("new", "legacy"), help="force the backend instead of probing for it"
    )
    ap.add_argument(
        "--section",
        action="append",
        choices=sorted(SECTIONS),
        help="section to run (repeatable; default: all)",
    )
    ap.add_argument(
        "--repeat", type=int, default=2, help="runs per case for the stability check (default 2)"
    )
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--subset", action="store_true", help="fast subset of every section")
    ap.add_argument(
        "--expect-diff",
        default="",
        help="comma-separated substrings of case keys whose diffs are expected",
    )
    args = ap.parse_args()
    if not args.section:
        args.section = sorted(SECTIONS)
    if args.repeat < 1:
        ap.error("--repeat must be >= 1")
    if not (args.dump or args.check):
        ap.print_help()
        return 2

    if sys.path and os.path.isfile(os.path.join(sys.path[0], "sageattention", "__init__.py")):
        print(
            f"WARNING: {sys.path[0]} contains a sageattention source tree and is on "
            "sys.path; the import below may shadow the installed package. Run this "
            "file by path (not with -m) from outside the checkout."
        )

    torch.cuda.set_device(args.device)
    be = detect_backend(args.backend)
    env = build_env(be)
    meta = build_meta(be, env, args)
    print(json.dumps(meta, indent=1, sort_keys=True))

    golden_dir = Path(args.golden_dir)
    if args.check and not (golden_dir / "hashes.json").exists():
        raise SystemExit(
            f"no golden dump at {golden_dir / 'hashes.json'}; run --dump "
            "in the baseline environment first (or point --golden-dir at it)"
        )
    keep_tensors = args.dump_tensors or (args.check and (golden_dir / "tensors.pt").exists())
    results, tensors = run_all(be, env, args, keep_tensors)
    n_unstable = sum(1 for r in results.values() if not r["stable"])
    n_raised = sum(1 for r in results.values() if r["out"] == RAISED)
    print(f"cases: {len(results)}, unstable: {n_unstable}, raised: {n_raised}")

    rc = report_equiv(results)

    if args.dump:
        golden_dir.mkdir(parents=True, exist_ok=True)
        (golden_dir / "hashes.json").write_text(
            json.dumps({"meta": meta, "cases": results}, indent=1, sort_keys=True)
        )
        if args.dump_tensors:
            torch.save(tensors, golden_dir / "tensors.pt")
        for key, r in sorted(results.items()):
            if not r["stable"]:
                print(f"UNSTABLE: {key}")
        print(f"golden written to {golden_dir.resolve()}")
        return rc

    return max(rc, do_check(be, env, args, golden_dir, results, tensors))


if __name__ == "__main__":
    sys.exit(main())
