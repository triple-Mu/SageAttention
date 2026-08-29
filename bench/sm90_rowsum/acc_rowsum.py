"""sm90 accuracy probe: sageattn vs an fp32 SDPA math reference, plus the lse it
returns. One process per tree; the seeds are fixed so the two runs see identical
inputs and the lse tensors can be differenced directly.
"""

import argparse
import json

import torch

AP = argparse.ArgumentParser()
AP.add_argument("--tag", required=True)
AP.add_argument("--out", required=True)
AP.add_argument("--lse-out", required=True)
args = AP.parse_args()

from sageattention import sageattn

DEV = "cuda"
CASES = [
    # (batch, heads_q, heads_kv, seq, head_dim, causal)
    (2, 8, 8, 1024, 64, False),
    (2, 8, 8, 1024, 64, True),
    (2, 8, 8, 4096, 64, False),
    (2, 8, 8, 4096, 64, True),
    (1, 8, 2, 2048, 64, False),
    (2, 8, 8, 1024, 128, False),
    (2, 8, 8, 1024, 128, True),
    (2, 8, 8, 4096, 128, False),
    (2, 8, 8, 4096, 128, True),
    (1, 8, 2, 2048, 128, False),
]


def sdpa_ref(q, k, v, is_causal, sm_scale):
    qf, kf, vf = q.float(), k.float(), v.float()
    h_qo, h_kv = qf.size(1), kf.size(1)
    if h_qo != h_kv:
        rep = h_qo // h_kv
        kf = kf.repeat_interleave(rep, dim=1)
        vf = vf.repeat_interleave(rep, dim=1)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        return torch.nn.functional.scaled_dot_product_attention(
            qf, kf, vf, is_causal=is_causal, scale=sm_scale
        )


def cos_sim(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def rel_l1(a, b):
    return ((a.float() - b.float()).abs().sum() / b.float().abs().sum()).item()


rows = []
lse_store = {}
for b, hq, hk, s, hd, causal in CASES:
    torch.manual_seed(20260830 + s + hd)
    q = torch.randn(b, hq, s, hd, dtype=torch.float16, device=DEV)
    k = torch.randn(b, hk, s, hd, dtype=torch.float16, device=DEV)
    v = torch.randn(b, hk, s, hd, dtype=torch.float16, device=DEV)
    sm_scale = hd**-0.5

    out, lse = sageattn(q, k, v, tensor_layout="HND", is_causal=causal, return_lse=True)
    ref = sdpa_ref(q, k, v, causal, sm_scale)

    key = f"b{b}_hq{hq}_hk{hk}_s{s}_d{hd}_c{int(causal)}"
    rows.append(
        {
            "tag": args.tag,
            "case": key,
            "hd": hd,
            "causal": int(causal),
            "cos_sim": cos_sim(out, ref),
            "rel_l1": rel_l1(out, ref),
        }
    )
    lse_store[key] = lse.detach().float().cpu()

with open(args.out, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
torch.save(lse_store, args.lse_out)
print(f"{args.tag}: {len(rows)} cases")
