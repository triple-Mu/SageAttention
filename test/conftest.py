"""Shared fixtures/helpers for the SageAttention test suite.

The build only carries the kernel families listed by
``torch.ops.sageattention.compiled_archs()``; every runtime test is gated on
``requires_backend(...)`` so a single-arch build (or a machine whose GPU
resolves to a different backend) skips instead of failing.
"""

import pytest
import torch

import sageattention  # noqa: F401  (registers torch.ops.sageattention.*)
from sageattention._plan import Plan

CUDA_AVAILABLE = torch.cuda.is_available()
CC = torch.cuda.get_device_capability() if CUDA_AVAILABLE else None
COMPILED_ARCHS = tuple(torch.ops.sageattention.compiled_archs())

ALL_BACKENDS = ("sm80", "sm89", "sm90", "sm100", "sm120")


def resolved_backend(head_dim=128):
    """Backend this device's compute capability resolves to, or None."""
    if CC is None:
        return None
    p = Plan(*torch.ops.sageattention.plan(CC[0], CC[1], head_dim, None, None, None, None))
    return None if p.error else p.backend


RESOLVED_BACKEND = resolved_backend()


def backend_available(name):
    """True when `name` is both compiled in and what this GPU resolves to."""
    if not CUDA_AVAILABLE:
        return False
    if int(name[2:]) not in COMPILED_ARCHS:
        return False
    return RESOLVED_BACKEND == name


def requires_backend(name):
    return pytest.mark.skipif(
        not backend_available(name),
        reason=f"backend {name} not runnable here "
        f"(compiled_archs={COMPILED_ARCHS}, cc={CC}, resolved={RESOLVED_BACKEND})",
    )


requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="needs CUDA")

# Backends with a packed-layout kernel. The sageattn_varlen API tests are
# written against the public entry point and say nothing about tile geometry or
# accumulator, so they run on whichever of these the device resolves to; the
# kernel-level tests stay pinned to one backend with requires_backend.
VARLEN_BACKENDS = ("sm80", "sm89", "sm90", "sm120")  # sm100 gated off pending the async-hang fix
requires_varlen_backend = pytest.mark.skipif(
    not any(backend_available(b) for b in VARLEN_BACKENDS),
    reason=f"no packed-layout backend runnable here "
    f"(compiled_archs={COMPILED_ARCHS}, cc={CC}, resolved={RESOLVED_BACKEND})",
)

# -DSAGE_BUILD_VARLEN=OFF drops the packed-layout kernel TUs; the ops stay
# registered and raise, so their tests skip rather than fail.
BUILD_VARLEN = bool(getattr(sageattention._C, "build_varlen", 0))
requires_varlen = pytest.mark.skipif(
    not BUILD_VARLEN, reason="built with SAGE_BUILD_VARLEN=OFF"
)

# The fp8 quantization kernels emit `cvt.rn.satfinite.e4m3x2.f32`, which only
# exists on sm_89+; below that the device code falls into RUNTIME_ASSERT ->
# __brkpt(), which kills the CUDA context rather than raising. The same trap
# is reached on an sm_89+ GPU running a build with no sm_89+ code in it (an
# sm_86 cubin loads fine on sm_89), so gate on the build as well.
requires_fp8_cast = pytest.mark.skipif(
    CC is None or CC < (8, 9) or not any(a >= 89 for a in COMPILED_ARCHS),
    reason=f"fp8 cast needs an sm_89+ device and build, "
    f"device=sm_{CC}, compiled_archs={COMPILED_ARCHS}"
    if CC
    else "needs CUDA",
)

# End-to-end fp8 PV coverage additionally needs this GPU to resolve to an fp8
# backend (every backend except sm80); a mismatched build resolves to sm80 or
# to nothing and would silently test the fp16 path instead.
requires_fp8_backend = pytest.mark.skipif(
    RESOLVED_BACKEND in (None, "sm80"),
    reason=f"needs an fp8 PV backend, resolved={RESOLVED_BACKEND}",
)


def sdpa_ref(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None):
    """fp32 SDPA math reference. Inputs/outputs follow `tensor_layout`;
    the math is always done in [batch, heads, seq, dim] float32."""
    if sm_scale is None:
        sm_scale = q.size(-1) ** -0.5
    qf, kf, vf = q.float(), k.float(), v.float()
    if tensor_layout == "NHD":
        qf, kf, vf = (t.transpose(1, 2) for t in (qf, kf, vf))

    h_qo, h_kv = qf.size(1), kf.size(1)
    if h_qo != h_kv:
        rep = h_qo // h_kv
        kf = kf.repeat_interleave(rep, dim=1)
        vf = vf.repeat_interleave(rep, dim=1)

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = torch.nn.functional.scaled_dot_product_attention(
            qf, kf, vf, is_causal=is_causal, scale=sm_scale
        )

    if tensor_layout == "NHD":
        out = out.transpose(1, 2)
    return out.to(q.dtype)


def ref_scores(q, k, tensor_layout="HND", is_causal=False, sm_scale=None):
    """fp32 QK^T * sm_scale with the causal mask applied, in [b, h_qo, qo, kv]."""
    if sm_scale is None:
        sm_scale = q.size(-1) ** -0.5
    qf, kf = q.float(), k.float()
    if tensor_layout == "NHD":
        qf, kf = qf.transpose(1, 2), kf.transpose(1, 2)
    h_qo, h_kv = qf.size(1), kf.size(1)
    if h_qo != h_kv:
        kf = kf.repeat_interleave(h_qo // h_kv, dim=1)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * sm_scale
    if is_causal:
        qo_len, kv_len = scores.size(-2), scores.size(-1)
        mask = torch.ones(qo_len, kv_len, dtype=torch.bool, device=scores.device).tril(
            diagonal=kv_len - qo_len
        )
        scores = scores.masked_fill(~mask, float("-inf"))
    return scores


def cos_sim(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def rel_l1(a, b):
    """(out - ref).abs().sum() / ref.abs().sum()"""
    return ((a.float() - b.float()).abs().sum() / b.float().abs().sum()).item()
