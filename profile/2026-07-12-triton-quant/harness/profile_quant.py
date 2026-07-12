# ncu profile harness：triton 量化 kernel（cutedsl_sage/quant_triton.py）vs CUDA fused
# 量化 kernel（sageattention._fused，sm90 e2e 实际派发参数）。口径均为 NHD fp16 输入。
#
# 环境变量：
#   PROF_KERNEL = triton | cuda     选择被试（ncu 按进程抓，分开跑）
#   PROF_LIST   = 1                 用 torch.profiler 打印 kernel 名后退出（确定 ncu regex）
#
# 固定主形状：b=4, h=32, s=4096, d=128。km 预计算（两侧共同的 torch k.mean，不在被试内）。
# 每次迭代 kernel 数：triton 4 个（q / k / v_amax / v_transpose），
# CUDA 4 个（per-warp q / per-block-fuse-sub-mean k / transpose_pad_permute / scale_fuse_quant）。
# warmup 3 次 + 5 次调用，ncu 用 -s 12 -c 4 跳过 warmup 抓稳态第 4 轮。
import os
import sys
from pathlib import Path

import torch

B, H, S, D = 4, 32, 4096, 128
WARMUP, ITERS = 3, 5

REPO_ROOT = Path(__file__).resolve().parents[3]


def make_fn(which):
    dev = "cuda"
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, dtype=torch.float16, device=dev)
    k = torch.randn(B, S, H, D, dtype=torch.float16, device=dev)
    v = torch.randn(B, S, H, D, dtype=torch.float16, device=dev)
    km = k.mean(dim=1, keepdim=True)   # 预计算，torch reduce 不在被试内

    if which == "triton":
        sys.path.insert(0, str(REPO_ROOT / "cutedsl_sage"))
        import quant_triton

        def fn():
            quant_triton.quant_q_int8_per_warp(q)
            quant_triton.quant_k_int8_per_block(k, km)
            quant_triton.quant_v_fp8_per_channel(v)
        return fn

    if which == "cuda":
        from sageattention.quant import per_warp_int8, per_channel_fp8
        # sm90 e2e 派发参数（sageattention/core.py:967）：BLKQ=64, WARPQ=16, BLKK=128
        km_s = km  # per_warp_int8 内部自行 squeeze

        def fn():
            per_warp_int8(q, k, km_s, tensor_layout="NHD", BLKQ=64, WARPQ=16, BLKK=128)
            per_channel_fp8(v, tensor_layout="NHD", smooth_v=False)  # s 为 128 倍数，无需 pad
        return fn

    raise ValueError(f"PROF_KERNEL={which}")


def main():
    which = os.environ["PROF_KERNEL"]
    fn = make_fn(which)

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
                print(f"KERNEL: {evt.key}  us={evt.self_device_time_total:.1f}")
        return

    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    print(f"done: {which} b={B} h={H} s={S} d={D} warmup={WARMUP} iters={ITERS}")


if __name__ == "__main__":
    main()
