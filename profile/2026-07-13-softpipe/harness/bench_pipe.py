# 软件流水三模式 A/B bench（kernel-only，预量化输入，口径同 bench_compare.py）。
# 同进程同卡交错：每形状按 round-robin 跑 ROUNDS 轮 do_bench，取各模式中值的中值，
# 消除时段漂移。模式规格 "mode" 或 "mode:minctasm"（如 full:2）。
#   用法：python bench_pipe.py [--full] [modes...]   默认 serial qk_hide full
import os
import statistics
import sys

import torch
from triton.testing import do_bench

import sageattention._qattn_sm90 as qattn_sm90

from core import SageAttnSm90

BATCH, HEADS = 4, 32
ROUNDS = 3


def bench_ms(fn):
    return do_bench(fn, warmup=25, rep=100, return_mode="median")


def tflops(s, d, causal, ms):
    f = 4 * BATCH * HEADS * s * s * d / (2 if causal else 1)
    return f / (ms * 1e-3) / 1e12


def make_inputs(d, seq):
    dev = "cuda"
    q_i8 = torch.randint(-95, 95, (BATCH, seq, HEADS, d), dtype=torch.int8, device=dev)
    k_i8 = torch.randint(-95, 95, (BATCH, seq, HEADS, d), dtype=torch.int8, device=dev)
    q_scale = torch.randn(BATCH, HEADS, seq // 64 * 4, dtype=torch.float32, device=dev).abs()
    k_scale = torch.randn(BATCH, HEADS, seq // 128, dtype=torch.float32, device=dev).abs()
    v_scale = torch.randn(BATCH, HEADS, d, dtype=torch.float32, device=dev).abs()
    v_dsl = torch.randn(BATCH, HEADS, d, seq, dtype=torch.float16, device=dev).to(torch.float8_e4m3fn)
    v_cuda = torch.randn(BATCH, d, HEADS, seq, dtype=torch.float16, device=dev).to(torch.float8_e4m3fn)
    o = torch.empty(BATCH, seq, HEADS, d, dtype=torch.float16, device=dev)
    return q_i8, k_i8, q_scale, k_scale, v_scale, v_dsl, v_cuda, o


def main():
    args = [a for a in sys.argv[1:]]
    full_matrix = "--full" in args
    specs = [a for a in args if not a.startswith("--")] or ["serial", "qk_hide", "full"]

    dev_name = torch.cuda.get_device_name()
    print(f"GPU: {dev_name} | torch {torch.__version__} | b={BATCH} h={HEADS} "
          f"rounds={ROUNDS} modes={specs}")
    seqs = (1024, 4096, 16384, 32768) if full_matrix else (4096, 16384)
    causals = (False, True) if full_matrix else (False,)

    hdr = "| seq | d | causal | CUDA " + "".join(f"| {m} " for m in specs) + \
          "".join(f"| {m}/serial " for m in specs if m != "serial") + "| best/CUDA |"
    print("\n" + hdr)
    print("|" + "---|" * (hdr.count("|") - 1))

    for d in (128, 64):
        for causal in causals:
            for seq in seqs:
                q_i8, k_i8, q_sc, k_sc, v_sc, v_dsl, v_cuda, o = make_inputs(d, seq)
                sm_scale = d ** -0.5
                ck = qattn_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf
                _c = 1 if causal else 0
                fns = {"cuda": lambda: ck(q_i8, k_i8, v_cuda, o, q_sc, k_sc, v_sc,
                                          0, _c, 2, sm_scale, 0)}
                for spec in specs:
                    mode, _, mct = spec.partition(":")
                    os.environ["SAGE_PIPE_MODE"] = mode
                    os.environ["SAGE_MINCTASM"] = mct or "0"
                    kern = SageAttnSm90.from_args(d, causal, torch.float16, 1)
                    fns[spec] = (lambda kern=kern: kern.run(
                        q_i8, k_i8, v_dsl, q_sc, k_sc, v_sc, o, sm_scale))
                res = {k: [] for k in fns}
                for _ in range(ROUNDS):                      # 交错轮换
                    for name, fn in fns.items():
                        fn(); torch.cuda.synchronize()
                        res[name].append(bench_ms(fn))
                ms = {k: statistics.median(v) for k, v in res.items()}
                row = f"| {seq} | {d} | {causal} | {tflops(seq, d, causal, ms['cuda']):.1f} "
                row += "".join(f"| {tflops(seq, d, causal, ms[m]):.1f} " for m in specs)
                row += "".join(f"| {ms['serial'] / ms[m]:.3f} "
                               for m in specs if m != "serial")
                best = min(ms[m] for m in specs)
                row += f"| {ms['cuda'] / best:.3f} |"
                print(row, flush=True)
                del q_i8, k_i8, q_sc, k_sc, v_sc, v_dsl, v_cuda, o
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
