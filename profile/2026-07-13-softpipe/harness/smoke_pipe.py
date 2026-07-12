# 软件流水三模式 bit-identical 冒烟：serial/qk_hide/full 逐位一致（设计保证：
# softmax/pack/merge 操作数与执行序不变，仅 wgmma commit/wait 编排不同）。
# 覆盖：单块(trip_count=1)/尾块/causal/GQA/长序列。远程：./hyper01.sh run 'python smoke_pipe.py'
import os

import torch

import core
from core import SageAttnSm90

SHAPES = [
    # (b, n_q, n_kv, s_q, s_k, d, causal)
    (1, 8, 8, 128, 128, 64, False),      # 单 KV 块（trip_count=1）
    (1, 8, 8, 128, 128, 128, False),
    (1, 8, 8, 100, 100, 128, False),     # 单块 + 尾 mask
    (1, 8, 8, 1000, 1000, 128, False),   # 多块 + 尾块
    (1, 8, 8, 1024, 1024, 64, True),     # causal
    (1, 8, 8, 337, 337, 128, True),      # causal + 尾块
    (2, 8, 4, 1024, 1024, 128, False),   # GQA
    (2, 8, 8, 4096, 4096, 128, False),   # 长序列
    (2, 8, 8, 4096, 4096, 64, True),
]
MODES = ["serial", "qk_hide", "full"]


def main():
    dev = "cuda"
    torch.manual_seed(0)
    for b, n_q, n_kv, s_q, s_k, d, causal in SHAPES:
        q = torch.randn(b, s_q, n_q, d, dtype=torch.float16, device=dev)
        k = torch.randn(b, s_k, n_kv, d, dtype=torch.float16, device=dev)
        v = torch.randn(b, s_k, n_kv, d, dtype=torch.float16, device=dev)
        km = k.mean(dim=1, keepdim=True)
        q_i8, q_sc = core.quant_q_int8_per_warp(q)
        k_i8, k_sc = core.quant_k_int8_per_block(k, km)
        v_f8, v_sc = core.quant_v_fp8_per_channel(v)
        sm_scale = d ** -0.5
        outs = {}
        for mode in MODES:
            os.environ["SAGE_PIPE_MODE"] = mode
            kern = SageAttnSm90.from_args(d, causal, torch.float16, n_q // n_kv)
            o = torch.empty(q.shape, dtype=q.dtype, device=dev)
            kern.run(q_i8, k_i8, v_f8, q_sc, k_sc, v_sc, o, sm_scale)
            torch.cuda.synchronize()
            outs[mode] = o
        for mode in MODES[1:]:
            same = torch.equal(outs["serial"].view(torch.int16),
                               outs[mode].view(torch.int16))
            tag = "OK " if same else "FAIL"
            print(f"[{tag}] {mode:8s} b{b} nq{n_q} nkv{n_kv} sq{s_q} sk{s_k} "
                  f"d{d} causal={causal}")
            assert same, f"{mode} not bit-identical"
    print("ALL BIT-IDENTICAL")


if __name__ == "__main__":
    main()
