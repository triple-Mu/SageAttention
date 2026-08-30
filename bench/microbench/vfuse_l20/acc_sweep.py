#!/usr/bin/env python
"""Accuracy numbers collector: the same cases as
test/test_accuracy.py::test_accuracy_pv_sweep and ::test_accuracy_smooth_v,
but printing cos_sim / rel_l1 per case instead of asserting, so the measured
values can go into a report (HARDWARE_CHECKLIST: the sm89 numbers were never
collected, only pass/fail).

Run from the tree's test/ directory (imports conftest for the exact same
reference and metric definitions).
"""

import csv
import itertools
import sys

import torch

from conftest import RESOLVED_BACKEND, cos_sim, rel_l1, sdpa_ref
from sageattention import sageattn

PV_COMBOS = {
    "sm80": ("fp32", "fp16", "fp16+fp32"),
    "sm89": ("fp32", "fp32+fp32", "fp32+fp16"),
    "sm90": ("fp32+fp32",),
    "sm100": ("fp32",),
    "sm120": ("fp32", "fp32+fp16"),
}
SMOOTH_V_PV = {"sm80": "fp16", "sm89": "fp32", "sm120": "fp32"}

DEV = "cuda"


def _qkv(b=2, h_qo=4, h_kv=None, qo_len=512, kv_len=None, head_dim=64, layout="HND",
         dtype=torch.float16, seed=0):
    h_kv = h_qo if h_kv is None else h_kv
    kv_len = qo_len if kv_len is None else kv_len
    g = torch.Generator(device=DEV).manual_seed(seed)

    def make(h, n):
        shape = (b, h, n, head_dim) if layout == "HND" else (b, n, h, head_dim)
        return torch.randn(shape, device=DEV, dtype=dtype, generator=g)

    return make(h_qo, qo_len), make(h_kv, kv_len), make(h_kv, kv_len)


def main():
    out_csv = sys.argv[1] if len(sys.argv) > 1 else "acc_sweep.csv"
    backend = RESOLVED_BACKEND
    print(f"resolved backend: {backend}")
    pvs = PV_COMBOS[backend]
    rows = []

    # ---- main sweep (mirrors test_accuracy_pv_sweep) ----
    for layout, causal, hd, gran, pv, sk in itertools.product(
        ("HND", "NHD"), (False, True), (64, 128), ("per_warp", "per_thread"), pvs, (True, False)
    ):
        q, k, v = _qkv(head_dim=hd, layout=layout)
        with torch.no_grad():
            out = sageattn(q, k, v, tensor_layout=layout, is_causal=causal,
                           qk_quant_gran=gran, pv_accum_dtype=pv, smooth_k=sk)
        ref = sdpa_ref(q, k, v, tensor_layout=layout, is_causal=causal)
        rows.append({
            "case": "pv_sweep", "layout": layout, "causal": causal, "head_dim": hd,
            "gran": gran, "pv": pv, "smooth_k": sk, "smooth_v": "",
            "cos_sim": f"{cos_sim(out, ref):.6f}", "rel_l1": f"{rel_l1(out, ref):.6f}",
        })

    # ---- smooth_v sweep (mirrors test_accuracy_smooth_v) ----
    sv_pv = SMOOTH_V_PV.get(backend)
    if sv_pv:
        for layout, causal, sk in itertools.product(("HND", "NHD"), (False, True), (True, False)):
            q, k, v = _qkv(layout=layout)
            with torch.no_grad():
                out = sageattn(q, k, v, tensor_layout=layout, is_causal=causal,
                               pv_accum_dtype=sv_pv, smooth_v=True, smooth_k=sk)
            ref = sdpa_ref(q, k, v, tensor_layout=layout, is_causal=causal)
            rows.append({
                "case": "smooth_v", "layout": layout, "causal": causal, "head_dim": 64,
                "gran": "", "pv": sv_pv, "smooth_k": sk, "smooth_v": True,
                "cos_sim": f"{cos_sim(out, ref):.6f}", "rel_l1": f"{rel_l1(out, ref):.6f}",
            })

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # worst case per pv (the checklist's ask)
    print(f"{len(rows)} cases -> {out_csv}")
    for pv in pvs:
        sel = [r for r in rows if r["pv"] == pv and r["case"] == "pv_sweep"]
        wc = min(float(r["cos_sim"]) for r in sel)
        wl = max(float(r["rel_l1"]) for r in sel)
        print(f"pv={pv:10s} n={len(sel):3d} worst cos_sim={wc:.6f} worst rel_l1={wl:.6f}")
    if sv_pv:
        sel = [r for r in rows if r["case"] == "smooth_v"]
        wc = min(float(r["cos_sim"]) for r in sel)
        wl = max(float(r["rel_l1"]) for r in sel)
        print(f"smooth_v(pv={sv_pv}) n={len(sel)} worst cos_sim={wc:.6f} worst rel_l1={wl:.6f}")


if __name__ == "__main__":
    main()
