# ncu profile harness：CuTeDSL vs CUDA sm90 SageAttention kernel（kernel-only，预量化输入）。
# 口径与 cutedsl_sage/bench_compare.py 一致。
#
# 环境变量：
#   PROF_KERNEL = cuda | dsl        选择要跑的 kernel（ncu 按进程抓，分开跑）
#   PROF_D      = 128 | 64          head_dim（默认 128）
#   PROF_LIST   = 1                 用 torch.profiler 打印 kernel 名后退出（用于确定 ncu regex）
#
# 固定主形状：b=4, h=32, s=4096, non-causal。warmup 3 次 + 精确 5 次调用，
# ncu 用 -s 3 -c 1 跳过 warmup 抓稳态第 4 次 launch。
import os
import sys
from pathlib import Path

import torch

B, H, S = 4, 32, 4096
WARMUP, ITERS = 3, 5

REPO_ROOT = Path(__file__).resolve().parents[1]  # cutedsl_sage 同步副本：与 core.py 同目录


def build_inputs(d):
    dev = "cuda"
    torch.manual_seed(0)
    q_i8 = torch.randint(-95, 95, (B, S, H, d), dtype=torch.int8, device=dev)
    k_i8 = torch.randint(-95, 95, (B, S, H, d), dtype=torch.int8, device=dev)
    q_scale = torch.randn(B, H, S // 64 * 4, dtype=torch.float32, device=dev).abs()
    k_scale = torch.randn(B, H, S // 128, dtype=torch.float32, device=dev).abs()
    v_scale = torch.randn(B, H, d, dtype=torch.float32, device=dev).abs()
    return q_i8, k_i8, q_scale, k_scale, v_scale


def make_fn(which, d):
    sm_scale = d ** -0.5
    q_i8, k_i8, q_scale, k_scale, v_scale = build_inputs(d)
    if which == "cuda":
        import sageattention._qattn_sm90 as qattn_sm90
        kernel = qattn_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf
        # NHD 时 V 为 [b, d, h, s]
        v = torch.randn(B, d, H, S, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
        o = torch.empty(B, S, H, d, dtype=torch.float16, device="cuda")
        return lambda: kernel(q_i8, k_i8, v, o, q_scale, k_scale, v_scale,
                              0, 0, 2, sm_scale, 0)  # NHD, non-causal, per_warp
    elif which == "dsl":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from core import SageAttnSm90
        # CuTe-DSL 的 V 为 [b, h, d, s]
        v = torch.randn(B, H, d, S, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
        o = torch.empty(B, S, H, d, dtype=torch.float16, device="cuda")
        kern = SageAttnSm90.from_args(d, False, torch.float16, 1)
        return lambda: kern.run(q_i8, k_i8, v, q_scale, k_scale, v_scale, o, sm_scale)
    raise ValueError(f"PROF_KERNEL={which}")


def main():
    which = os.environ["PROF_KERNEL"]
    d = int(os.environ.get("PROF_D", "128"))
    fn = make_fn(which, d)

    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    if os.environ.get("PROF_LIST") == "1":
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        for evt in prof.key_averages():
            if evt.device_type.name == "CUDA":
                print(f"KERNEL: {evt.key}")
        return

    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    print(f"done: {which} d={d} b={B} h={H} s={S} warmup={WARMUP} iters={ITERS}")


if __name__ == "__main__":
    main()
