"""sm90 row-sum A/B: kernel-only timing of qk_int8_sv_f8_..._fuse_v_scale_attn_inst_buf.

Harness for the experiment written up in test/HARDWARE_CHECKLIST.md section 6
("sm90 row sum on the tensor core"); the captures it produced are in ./data.

One process per tree, PYTHONPATH picking the tree, one JSON line per point.
Run it four times so each version is measured in both orders - A, B, then B, A -
and only believe a difference the two rounds agree on:

    for r in r1 r2; do
      [ $r = r1 ] && order="base new" || order="new base"
      for t in $order; do
        CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/path/to/$t python bench_rowsum.py \
          --gpu 5 --tag ${t}_${r} --out ${t}_${r}.jsonl --reps 20
      done
    done

Exclusivity: nvidia-smi memory.used is sampled before and after every point and
must stay within a slack of the value taken right after our own allocations;
the card must start near-empty for that to mean anything.
"""

import argparse
import json
import subprocess
import sys
import time

import torch

AP = argparse.ArgumentParser()
AP.add_argument("--gpu", type=int, required=True, help="physical index for the nvidia-smi probe")
AP.add_argument("--tag", required=True)
AP.add_argument("--out", required=True)
AP.add_argument("--reps", type=int, default=20)
AP.add_argument("--slack-mib", type=int, default=512)
AP.add_argument("--head-dims", default="64,128")
args = AP.parse_args()

import sageattention  # noqa: F401  (registers torch.ops.sageattention)


# stand-in for the retired qattn_sm90_..._fuse_v_scale_attn_inst_buf op:
# fwd with backend="sm90" launches the identical kernel. `o` stays in the
# signature but is unused — fwd allocates its own output, and the bench only
# times, never reads it.
def OP(q, k, v, o, q_scale, k_scale, v_scale, tensor_layout, is_causal, gran, sm_scale, lse):
    torch.ops.sageattention.fwd(
        q,
        k,
        v,
        q_scale,
        k_scale,
        v_scale,
        tensor_layout=tensor_layout,
        qk_quant_gran=gran,
        pv_accum_dtype="fp32+fp32",
        v_layout="mma_k16",
        is_causal=is_causal,
        sm_scale=sm_scale,
        return_lse=lse,
        backend="sm90",
    )

# (seq, batch, heads): batch/heads shrink with seq so no single point runs long.
SHAPES = [
    (1024, 4, 32),
    (2048, 4, 32),
    (4096, 4, 32),
    (8192, 2, 32),
    (16384, 1, 32),
    (32768, 1, 16),
    (65536, 1, 8),
]
HEAD_DIMS = tuple(int(x) for x in args.head_dims.split(","))
GRANS = ("per_warp", "per_thread")
CAUSALS = (False, True)


def gpu_mem_mib():
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            str(args.gpu),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def make_inputs(seq, batch, heads, hd, gran):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(1234 + seq + hd)
    q = torch.randint(-95, 95, (batch, heads, seq, hd), dtype=torch.int8, device=dev, generator=g)
    k = torch.randint(-95, 95, (batch, heads, seq, hd), dtype=torch.int8, device=dev, generator=g)
    v = (
        torch.randn(batch, heads, hd, seq, dtype=torch.float16, device=dev, generator=g)
        .to(torch.float8_e4m3fn)
    )
    o = torch.empty(batch, heads, seq, hd, dtype=torch.float16, device=dev)
    v_scale = torch.rand(batch, heads, hd, dtype=torch.float32, device=dev, generator=g) + 0.5
    nq = (seq + 63) // 64
    nk = (seq + 127) // 128
    if gran == "per_warp":
        qs = torch.rand(batch, heads, nq * 4, dtype=torch.float32, device=dev, generator=g) + 0.5
        ks = torch.rand(batch, heads, nk, dtype=torch.float32, device=dev, generator=g) + 0.5
    else:
        qs = torch.rand(batch, heads, nq * 4 * 8, dtype=torch.float32, device=dev, generator=g) + 0.5
        ks = torch.rand(batch, heads, nk * 4, dtype=torch.float32, device=dev, generator=g) + 0.5
    return q, k, v, o, qs, ks, v_scale


def time_point(q, k, v, o, qs, ks, vs, gran, causal, sm_scale, reps):
    for _ in range(5):
        OP(q, k, v, o, qs, ks, vs, "HND", causal, gran, sm_scale, False)
    torch.cuda.synchronize()
    best = float("inf")
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(reps):
        start.record()
        OP(q, k, v, o, qs, ks, vs, "HND", causal, gran, sm_scale, False)
        end.record()
        end.synchronize()
        best = min(best, start.elapsed_time(end))
    return best * 1e3  # us


results = []
dropped = []
with open(args.out, "w") as fh:
    for hd in HEAD_DIMS:
        for seq, batch, heads in SHAPES:
            for gran in GRANS:
                bufs = make_inputs(seq, batch, heads, hd, gran)
                torch.cuda.synchronize()
                base_mem = gpu_mem_mib()
                for causal in CAUSALS:
                    m0 = gpu_mem_mib()
                    t = time_point(*bufs, gran, causal, hd**-0.5, args.reps)
                    m1 = gpu_mem_mib()
                    rec = {
                        "tag": args.tag,
                        "hd": hd,
                        "seq": seq,
                        "batch": batch,
                        "heads": heads,
                        "gran": gran,
                        "causal": int(causal),
                        "us": t,
                        "mem_before": m0,
                        "mem_after": m1,
                        "mem_base": base_mem,
                    }
                    ok = (
                        abs(m0 - base_mem) <= args.slack_mib
                        and abs(m1 - base_mem) <= args.slack_mib
                    )
                    rec["exclusive"] = ok
                    if not ok:
                        dropped.append(rec)
                    results.append(rec)
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                del bufs
                torch.cuda.empty_cache()

print(f"{args.tag}: {len(results)} points, {len(dropped)} non-exclusive", file=sys.stderr)
