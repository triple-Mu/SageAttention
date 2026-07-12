# H200 性能对比：CUDA C++ sm90 kernel vs CuTe-DSL kernel vs torch SDPA。
# kernel-only：预量化输入，不含量化开销（与 bench/ 目录惯例一致）；
# 端到端：含 torch 量化（CuTeDSL 侧量化为 torch 临时实现，非本期优化目标）。
#
# 口径说明：
# - CUDA 侧用 qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf（fp32+fp32 两级累加 + fuse v_scale），
#   与 Python API 实际使用的变体一致；CuTeDSL 为单级 FP32（FP22）累加 + fuse v_scale。
# - 计时统一 triton.testing.do_bench(warmup=25, rep=100) 取中值。
import sys

import torch
import torch.nn.functional as F
from triton.testing import do_bench

import sageattention._qattn_sm90 as qattn_sm90
from sageattention.core import sageattn_qk_int8_pv_fp8_cuda_sm90

from core import SageAttnSm90, sageattn_qk_int8_pv_fp8_hopper

BATCH, HEADS = 4, 32


def bench_ms(fn):
    return do_bench(fn, warmup=25, rep=100, return_mode="median")


def tflops(b, h, s, d, causal, ms):
    f = 4 * b * h * s * s * d / (2 if causal else 1)
    return f / (ms * 1e-3) / 1e12


def bench_kernel_only(d, seq, causal):
    """返回 (ms_cuda, ms_dsl, ms_sdpa)。seq 需为 128 的倍数（免 V padding）。"""
    b, h = BATCH, HEADS
    dev = "cuda"
    sm_scale = d ** -0.5

    # 预量化输入（随机数值，只测速度）；q/k int8 NHD [b,s,h,d]，scale 布局两 kernel 相同
    q_i8 = torch.randint(-95, 95, (b, seq, h, d), dtype=torch.int8, device=dev)
    k_i8 = torch.randint(-95, 95, (b, seq, h, d), dtype=torch.int8, device=dev)
    q_scale = torch.randn(b, h, seq // 64 * 4, dtype=torch.float32, device=dev).abs()
    k_scale = torch.randn(b, h, seq // 128, dtype=torch.float32, device=dev).abs()
    v_scale = torch.randn(b, h, d, dtype=torch.float32, device=dev).abs()

    # 1) CUDA sm90：NHD 时 V 为 [b, d, h, s]（csrc/qattn/qk_int_sv_f8_cuda_sm90.cu:816）
    v_cuda = torch.randn(b, d, h, seq, dtype=torch.float16, device=dev).to(torch.float8_e4m3fn)
    o_cuda = torch.empty(b, seq, h, d, dtype=torch.float16, device=dev)
    kernel = qattn_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf
    _causal = 1 if causal else 0
    fn_cuda = lambda: kernel(q_i8, k_i8, v_cuda, o_cuda, q_scale, k_scale, v_scale,
                             0, _causal, 2, sm_scale, 0)   # NHD, per_warp
    fn_cuda()
    torch.cuda.synchronize()
    ms_cuda = bench_ms(fn_cuda)
    del v_cuda, o_cuda

    # 2) CuTe-DSL：V 为 [b, h, d, s_pad]，s_pad=ceil(s/128)*128（此处 seq 恰为倍数）
    v_dsl = torch.randn(b, h, d, seq, dtype=torch.float16, device=dev).to(torch.float8_e4m3fn)
    o_dsl = torch.empty(b, seq, h, d, dtype=torch.float16, device=dev)
    kern = SageAttnSm90.from_args(d, causal, torch.float16, 1)   # 首次调用触发编译
    fn_dsl = lambda: kern.run(q_i8, k_i8, v_dsl, q_scale, k_scale, v_scale, o_dsl, sm_scale)
    fn_dsl()
    torch.cuda.synchronize()
    ms_dsl = bench_ms(fn_dsl)
    del v_dsl, o_dsl, q_i8, k_i8, q_scale, k_scale, v_scale

    # 3) torch SDPA 参照（fp16，[b,h,s,d]）
    qf = torch.randn(b, h, seq, d, dtype=torch.float16, device=dev)
    kf = torch.randn(b, h, seq, d, dtype=torch.float16, device=dev)
    vf = torch.randn(b, h, seq, d, dtype=torch.float16, device=dev)
    fn_sdpa = lambda: F.scaled_dot_product_attention(qf, kf, vf, is_causal=causal)
    fn_sdpa()
    torch.cuda.synchronize()
    ms_sdpa = bench_ms(fn_sdpa)
    del qf, kf, vf

    torch.cuda.empty_cache()
    return ms_cuda, ms_dsl, ms_sdpa


def bench_e2e(seq, causal, d=128):
    """端到端（含量化）：返回 (ms_cuda, ms_dsl)。输入 fp16 NHD [b,s,n,d]。"""
    b, h = BATCH, HEADS
    dev = "cuda"
    q = torch.randn(b, seq, h, d, dtype=torch.float16, device=dev)
    k = torch.randn(b, seq, h, d, dtype=torch.float16, device=dev)
    v = torch.randn(b, seq, h, d, dtype=torch.float16, device=dev)

    fn_cuda = lambda: sageattn_qk_int8_pv_fp8_cuda_sm90(
        q, k, v, tensor_layout="NHD", is_causal=causal, qk_quant_gran="per_warp")
    fn_cuda()
    torch.cuda.synchronize()
    ms_cuda = bench_ms(fn_cuda)

    fn_dsl = lambda: sageattn_qk_int8_pv_fp8_hopper(q, k, v, is_causal=causal)
    fn_dsl()
    torch.cuda.synchronize()
    ms_dsl = bench_ms(fn_dsl)

    del q, k, v
    torch.cuda.empty_cache()
    return ms_cuda, ms_dsl


def main():
    # --quick：只跑主 4 点 kernel-only（s∈{4096,16384} × d∈{64,128}，non-causal），迭代用
    quick = "--quick" in sys.argv
    torch.manual_seed(0)
    dev_name = torch.cuda.get_device_name()
    print(f"GPU: {dev_name} | torch {torch.__version__} | batch={BATCH} heads={HEADS}")
    print(f"kernel-only 口径：预量化输入，per_warp；CUDA=fp32+fp32 两级累加，CuTeDSL=单级 FP32(FP22) 累加，均 fuse v_scale\n")

    print("## Kernel-only TFLOPS\n")
    print("| seq | d | causal | CUDA TFLOPS | CuTeDSL TFLOPS | SDPA TFLOPS | CuTeDSL/CUDA |")
    print("|---:|---:|:---|---:|---:|---:|---:|")
    for d in (64, 128):
        for causal in ((False,) if quick else (False, True)):
            for seq in ((4096, 16384) if quick else (1024, 2048, 4096, 8192, 16384, 32768)):
                ms_cuda, ms_dsl, ms_sdpa = bench_kernel_only(d, seq, causal)
                tf = lambda ms: tflops(BATCH, HEADS, seq, d, causal, ms)
                print(f"| {seq} | {d} | {causal} | {tf(ms_cuda):.1f} | {tf(ms_dsl):.1f} "
                      f"| {tf(ms_sdpa):.1f} | {ms_cuda / ms_dsl:.3f} |", flush=True)
    if quick:
        return

    print("\n## 端到端（含量化，d=128；CuTeDSL 量化为 torch 临时实现，非本期优化目标）\n")
    print("| seq | causal | CUDA e2e ms | CuTeDSL e2e ms |")
    print("|---:|:---|---:|---:|")
    for causal in (False, True):
        for seq in (1024, 4096, 16384):
            ms_cuda, ms_dsl = bench_e2e(seq, causal)
            print(f"| {seq} | {causal} | {ms_cuda:.3f} | {ms_dsl:.3f} |", flush=True)


if __name__ == "__main__":
    main()
