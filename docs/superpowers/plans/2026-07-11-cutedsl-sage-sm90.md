# CuteDSL SageAttention (Hopper) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `cutedsl_sage/` 用 CuTe-DSL 实现 Hopper SageAttention forward（QK int8 per-warp + PV fp8 per-channel + 单级 FP32 累加），端到端 API + torch 量化，在 hyper01 (H200) 上通过双参考正确性验证。

**Architecture:** kernel 以 cutlass 官方 `examples/python/CuTeDSL/hopper/fmha.py` 为底本：1 load warpgroup（producer）+ 1 math warpgroup（consumer），CTA_Q=64/CTA_K=128，`PipelineTmaAsync` 管 K/V 环形 smem，QK 用 `MmaI8Op`(s8s8s32) SS、PV 用 `MmaF8Op`(e4m3→f32) RS；host 侧走 `cute.compile(--enable-tvm-ffi)` + fake tensor 编译缓存，产物直接吃 torch.Tensor。本计划中的代码已经过两轮多 agent 起草 + API/数值对抗验证（全部 cute API 已对照本地 CuTeDSL 源码逐项核实），但**从未在 Hopper 上编译运行过**——执行时以远程迭代循环为准，计划代码是高置信起点而非保证正确的终点。

**Tech Stack:** CuTe-DSL (nvidia-cutlass-dsl 4.6.0) / apache-tvm-ffi / PyTorch 2.11 / pytest；远程 H200（hyper01 容器 sglang-diffusion-qwenimage）。

## Global Constraints

（逐条来自 spec `docs/superpowers/specs/2026-07-11-cutedsl-sage-sm90-design.md`，任务隐含遵守）

- 输入 layout 固定 NHD `[b, s, n, d]`，`d ∈ {64, 128}`（其余 assert），最后一维 contiguous
- 量化粒度：Q per-warp（BLKQ=64 / WARPQ=16），K per-block（BLKK=128），V per-channel（scale_max=448）
- kernel tile：CTA_Q=64 / CTA_K=128；grid = `(ceil(s_q/64), b, n_q)`（x=seq 块, y=batch, z=head）
- PV 累加：**单级 FP32**（FP22），禁止引入两级 fp32+fp32
- **P→e4m3 契约**：显式 `P * 448.0` 后转 e4m3（禁止 CUDA 版的 exp2 偏移 2^8.807 技巧）；row_sum 累加**量化前**的 f32 P；epilogue `O = acc · v_scale/448 / row_sum`
- causal 要求 `s_q == s_k`；GQA 要求 `n_q % n_kv == 0`；不支持 `return_lse`
- 远程测试路径：`ssh hyper01` → 容器 `sglang-diffusion-qwenimage` → `source /data/.torch/bin/activate`；挂载映射 宿主机 `/data02/triplemu` ↔ 容器 `/data`（已用 docker inspect 确认）
- `nvidia-cutlass-dsl==4.6.0`（若 API drift，按 Task 1 的 API 检查降级到 API 齐全的版本；本地权威源码参考：`/home/ubuntu/workspace/github/llm/flash-attention/csrc/cutlass/`，checkout 2026-01-05）
- 遵守用户代码规范：注释中文少而精、不加无谓错误处理、最少复杂度
- 分支 `cutedsl-sage-sm90`；每个任务完成即 commit，消息末尾 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

## 远程迭代循环（Task 3-5 通用方法论）

1. 本地改代码 → `./cutedsl_sage/hyper01.sh test -k <关键字>`（自动 rsync + 容器内 pytest）
2. **编译错误**：报错栈会指向 core.py 行 → 注释里有 `fmha.py:NNN` 来源行号，对照底本；若疑似 4.6.0 API drift，在容器内 grep 安装源码：`./cutedsl_sage/hyper01.sh run 'grep -rn "<symbol>" $(python -c "import cutlass,os;print(os.path.dirname(cutlass.__file__))")'`
3. **数值错误**：先缩形状（b=1, n=1, s=128, d=128, non-causal 单 KV 块）；一级比对（vs ref_quant_sim）失配时按「哪个 warp/哪个列块错」定位到 scale 索引或 mask；必要时在 kernel 里 `cute.printf` 单点值
4. 每修一个问题立即重跑该用例，绿了再扩大形状集

---

### Task 1: 远程环境准备 + 工具脚本 + DSL 冒烟

**Files:**
- Create: `cutedsl_sage/hyper01.sh`
- Create: `cutedsl_sage/smoke_cutedsl.py`

**Interfaces:**
- Consumes: 无（首个任务）
- Produces: `./hyper01.sh {setup|sync|run '<cmd>'|test [pytest args]}`——后续所有任务的远程执行入口；远程容器内可用的 cutlass DSL 环境

- [ ] **Step 1: 写远程辅助脚本 `cutedsl_sage/hyper01.sh`**

```bash
#!/usr/bin/env bash
# hyper01 (H200) 远程验证：rsync cutedsl_sage/ 到容器挂载路径 + docker exec 执行。
#   ./hyper01.sh setup        容器内安装 nvidia-cutlass-dsl / apache-tvm-ffi / pytest
#   ./hyper01.sh sync         仅同步代码
#   ./hyper01.sh run '<cmd>'  同步后在容器内执行（venv 已激活、cwd 为 cutedsl_sage）
#   ./hyper01.sh test [args]  同步后跑 pytest（默认 test_sage_sm90.py）
set -euo pipefail

REMOTE=hyper01
CONTAINER=sglang-diffusion-qwenimage
VENV=/data/.torch/bin/activate
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10)

# 挂载映射已确认（docker inspect）：宿主机 /data02/triplemu ↔ 容器 /data
HOST_DIR=/data02/triplemu/workspace/SageAttention/cutedsl_sage
CTR_DIR=/data/workspace/SageAttention/cutedsl_sage

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

do_sync() {
  ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p '$HOST_DIR'"
  rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
    --exclude __pycache__ --exclude .pytest_cache \
    "$LOCAL_DIR/" "$REMOTE:$HOST_DIR/"
}

in_container() {
  local cmd="source $VENV && cd $CTR_DIR && $1"
  ssh "${SSH_OPTS[@]}" "$REMOTE" "docker exec -i $CONTAINER bash -c $(printf '%q' "$cmd")"
}

case "${1:-}" in
  setup) in_container "pip install 'nvidia-cutlass-dsl==4.6.0' apache-tvm-ffi pytest" ;;
  sync)  do_sync ;;
  run)   shift; do_sync; in_container "$*" ;;
  # printf '%q ' 逐参数转义，保留 -m 'not slow' 这类带空格参数的边界
  test)  shift; do_sync; in_container "python -m pytest -v -x -s $(printf '%q ' "${@:-test_sage_sm90.py}")" ;;
  *)     echo "用法: $0 {setup|sync|run '<cmd>'|test [pytest args]}" >&2; exit 1 ;;
esac
```

- [ ] **Step 2: 写 DSL 冒烟脚本 `cutedsl_sage/smoke_cutedsl.py`**

验证两件事：① `--enable-tvm-ffi` 编译产物直接吃 torch.Tensor 并落当前 stream；② kernel 代码（Task 3）依赖的关键 API 在已安装版本中全部存在。

```python
"""CuTe-DSL 环境冒烟。在 H200 容器内运行：python smoke_cutedsl.py"""
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute


class AddOne:
    def __init__(self):
        self.compiled = None

    @cute.jit
    def __call__(self, x: cute.Tensor, stream: cuda.CUstream):
        self.kernel(x).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx < cute.size(x.shape):
            x[tidx] = x[tidx] + 1.0


def check_apis():
    import cutlass.utils.hopper_helpers as sm90_utils
    import cutlass.pipeline as pipeline
    from cutlass.cute.nvgpu import cpasync, warpgroup
    checks = {
        "cute.runtime.make_fake_compact_tensor": hasattr(cute.runtime, "make_fake_compact_tensor"),
        "cute.runtime.make_fake_stream": hasattr(cute.runtime, "make_fake_stream"),
        "cute.sym_int": hasattr(cute, "sym_int"),
        "sm90_utils.make_trivial_tiled_mma": hasattr(sm90_utils, "make_trivial_tiled_mma"),
        "sm90_utils.make_smem_layout_a/b/epi": all(
            hasattr(sm90_utils, f"make_smem_layout_{s}") for s in ("a", "b", "epi")),
        "sm90_utils.compute_tile_shape_or_override": hasattr(sm90_utils, "compute_tile_shape_or_override"),
        "sm90_utils.sm90_get_smem_store_op": hasattr(sm90_utils, "sm90_get_smem_store_op"),
        "pipeline.PipelineTmaAsync": hasattr(pipeline, "PipelineTmaAsync"),
        "pipeline.PipelineTmaStore": hasattr(pipeline, "PipelineTmaStore"),
        "pipeline.pipeline_init_arrive/wait": hasattr(pipeline, "pipeline_init_arrive")
            and hasattr(pipeline, "pipeline_init_wait"),
        "pipeline.arrive_and_wait": hasattr(pipeline, "arrive_and_wait"),
        "pipeline.CooperativeGroup/Agent": hasattr(pipeline, "CooperativeGroup") and hasattr(pipeline, "Agent"),
        "cpasync.CopyBulkTensorTileG2SOp": hasattr(cpasync, "CopyBulkTensorTileG2SOp"),
        "cpasync.CopyBulkTensorTileS2GOp": hasattr(cpasync, "CopyBulkTensorTileS2GOp"),
        "cpasync.make_tiled_tma_atom": hasattr(cpasync, "make_tiled_tma_atom"),
        "cpasync.tma_partition": hasattr(cpasync, "tma_partition"),
        "cpasync.prefetch_descriptor": hasattr(cpasync, "prefetch_descriptor"),
        "warpgroup.MmaI8Op": hasattr(warpgroup, "MmaI8Op"),
        "warpgroup.MmaF8Op": hasattr(warpgroup, "MmaF8Op"),
        "warpgroup.OperandSource/OperandMajorMode": hasattr(warpgroup, "OperandSource")
            and hasattr(warpgroup, "OperandMajorMode"),
        "cute.make_rmem_tensor_like": hasattr(cute, "make_rmem_tensor_like"),
        "cute.recast_ptr/recast_tensor": hasattr(cute, "recast_ptr") and hasattr(cute, "recast_tensor"),
        "cute.arch.warp_reduction_max/sum": hasattr(cute.arch, "warp_reduction_max")
            and hasattr(cute.arch, "warp_reduction_sum"),
        "cute.arch.shuffle_sync_op/prmt": hasattr(cute.arch, "shuffle_sync_op")
            and hasattr(cute.arch, "prmt"),
        "cute.arch.warpgroup_reg_alloc/dealloc": hasattr(cute.arch, "warpgroup_reg_alloc")
            and hasattr(cute.arch, "warpgroup_reg_dealloc"),
        "cute.arch.rcp_approx": hasattr(cute.arch, "rcp_approx"),
        "cute.math.exp2": hasattr(cute.math, "exp2"),
        "cute.nvgpu.warpgroup.fence/commit/wait": hasattr(cute.nvgpu.warpgroup, "fence")
            and hasattr(cute.nvgpu.warpgroup, "commit_group") and hasattr(cute.nvgpu.warpgroup, "wait_group"),
        "cute.nvgpu.warp.StMatrix8x8x16bOp": hasattr(cute.nvgpu.warp, "StMatrix8x8x16bOp"),
    }
    print(f"cutlass version: {cutlass.__version__}")
    missing = [k for k, ok in checks.items() if not ok]
    for k, ok in checks.items():
        print(f"  [{'OK' if ok else 'MISSING'}] {k}")
    return missing


def main():
    missing = check_apis()
    op = AddOne()
    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (128,), stride_order=(0,), assumed_align=4)
    op.compiled = cute.compile(
        op, x_fake,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi")
    x = torch.zeros(128, dtype=torch.float32, device="cuda")
    op.compiled(x)
    torch.cuda.synchronize()
    assert (x == 1).all()
    print("tvm-ffi smoke: OK")
    assert not missing, f"missing APIs: {missing}"


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 安装远程环境**

```bash
chmod +x cutedsl_sage/hyper01.sh
./cutedsl_sage/hyper01.sh setup
```
Expected: pip 成功安装 `nvidia-cutlass-dsl==4.6.0`、`apache-tvm-ffi`、`pytest`（tvm-ffi 若是 cutlass-dsl 的依赖会自动带上，重复安装无害）。

- [ ] **Step 4: 跑冒烟**

```bash
./cutedsl_sage/hyper01.sh run 'python smoke_cutedsl.py'
```
Expected: 打印 `cutlass version: 4.6.0`、全部 API `[OK]`、`tvm-ffi smoke: OK`。
若有 `[MISSING]`：依次尝试 `pip install 'nvidia-cutlass-dsl==4.5.3'`（再 4.4.2 / 4.3.5）直到 API 齐全，并把最终版本号回写到本文件 Global Constraints 与 hyper01.sh setup 行。
若 `AddOne` 编译或运行失败：这是环境级阻塞，先解决再进 Task 2（对照 `flash-attention/flash_attn/cute/interface.py:980` 的用法差异）。

- [ ] **Step 5: Commit**

```bash
git add cutedsl_sage/hyper01.sh cutedsl_sage/smoke_cutedsl.py
git commit -m "$(cat <<'EOF'
Add hyper01 remote helper and CuTe-DSL smoke test

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: 基类 SyntaxError 修复 + torch 量化函数 + 量化单测

**Files:**
- Modify: `cutedsl_sage/cutedsl_sage.py:34-45`（修复 SyntaxError，已用 ast.parse 确认 line 44 报 "positional argument follows keyword argument unpacking"）
- Create: `cutedsl_sage/core.py`（本任务只写量化部分；kernel 部分 Task 3 追加）
- Create: `cutedsl_sage/test_sage_sm90.py`（本任务只写量化单测；参考实现与 kernel 测试 Task 4 追加）

**Interfaces:**
- Consumes: Task 1 的 `hyper01.sh`
- Produces（Task 3/4 依赖的精确签名）:
  - `quant_q_int8_per_warp(q: Tensor[b,s,n,d] fp16/bf16) -> (q_int8: Tensor[b,s,n,d] int8 contiguous, q_scale: Tensor[b,n,ceil(s/64)*4] fp32 contiguous)`
  - `quant_k_int8_per_block(k: Tensor[b,s,n_kv,d], km: Tensor[b,1,n_kv,d]|None) -> (k_int8, k_scale: Tensor[b,n_kv,ceil(s/128)] fp32)`
  - `quant_v_fp8_per_channel(v: Tensor[b,s,n_kv,d]) -> (v_fp8: Tensor[b,n_kv,d,s_pad] float8_e4m3fn contiguous, v_scale: Tensor[b,n_kv,d] fp32)`，`s_pad = ceil(s/128)*128`

- [ ] **Step 1: 修复 `cutedsl_sage/cutedsl_sage.py` 的 cute.compile 调用**

把 `from_args` 内的编译调用（38-44 行）替换为（位置参数移到 `**kwargs` 之前）：

```python
        # 修复：位置参数（fake stream）必须在 **kwargs 解包之前
        kernel.compiled_kernel = cute.compile(
            kernel,
            *args,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
            **kwargs,
        )
```

其余保持骨架原样（`key = ...` 占位不动——SageAttnSm90 覆写 from_args，不走基类实现）。

- [ ] **Step 2: 验证语法修复**

```bash
python3 -c "import ast; ast.parse(open('cutedsl_sage/cutedsl_sage.py').read()); print('syntax OK')"
```
Expected: `syntax OK`（修复前该命令报 SyntaxError line 44）。

- [ ] **Step 3: 写量化单测（先写测试，`cutedsl_sage/test_sage_sm90.py`）**

```python
# CuteDSL SageAttention sm90 正确性测试。
#   量化单测：任意 GPU/CPU 可跑；kernel 测试（Task 4 追加）需 H200。
#   远程执行：./hyper01.sh test
import math

import pytest
import torch
import torch.nn.functional as F

try:
    from .core import quant_q_int8_per_warp, quant_k_int8_per_block, quant_v_fp8_per_channel
except ImportError:
    from core import quant_q_int8_per_warp, quant_k_int8_per_block, quant_v_fp8_per_channel

LOG2E = math.log2(math.e)


# ============ 量化函数单测 ============

def test_q_scale_shape_order_and_tail():
    """分段常数张量验证段序：第 g 个 16 行段填 g+1 → scale[..., g] == (g+1)/127。"""
    b, s, n, d = 2, 337, 3, 64                       # ceil(337/64)=6 块 → 24 段，有效段 22
    g = torch.arange(s) // 16
    q = (g + 1).to(torch.float16)[None, :, None, None].expand(b, s, n, d).contiguous()
    q_int8, q_scale = quant_q_int8_per_warp(q)
    assert q_int8.shape == (b, s, n, d) and q_int8.dtype == torch.int8 and q_int8.is_contiguous()
    assert q_scale.shape == (b, n, 24) and q_scale.dtype == torch.float32 and q_scale.is_contiguous()
    n_valid = (s + 15) // 16                         # 22，末段仅 1 有效行（amax 只看有效行）
    expect = (torch.arange(n_valid).float() + 1) / 127.0
    assert torch.allclose(q_scale[..., :n_valid], expect.expand(b, n, -1))
    assert (q_int8.float() == 127).all()             # 段内常数 → 全 127
    tail = q_scale[..., n_valid:]                    # 全 padding 段
    assert (tail > 0).all() and not torch.isnan(tail).any()


def test_q_roundtrip_error():
    torch.manual_seed(0)
    q = torch.randn(1, 200, 2, 128, dtype=torch.float16)   # 非 64 倍数
    q_int8, q_scale = quant_q_int8_per_warp(q)
    scale_row = q_scale.repeat_interleave(16, dim=-1)[..., :200].permute(0, 2, 1)[..., None]
    deq = q_int8.float() * scale_row
    # round-to-nearest 半步长上界；3e-5 相对 slack 覆盖 fp32 中间舍入（127·2^-23 两次）
    assert ((deq - q.float()).abs() <= scale_row * (0.5 + 3e-5)).all()


def test_k_smooth_tail_and_roundtrip():
    torch.manual_seed(0)
    b, s, n, d = 1, 337, 2, 64                       # ceil(337/128)=3 块，尾块 81 有效行
    k = torch.randn(b, s, n, d, dtype=torch.float16) * 4
    km = k.mean(dim=1, keepdim=True)
    k_int8, k_scale = quant_k_int8_per_block(k, km)
    assert k_int8.shape == (b, s, n, d) and k_scale.shape == (b, n, 3)
    kf = k.float() - km.float()
    expect_tail = kf[:, 256:].abs().amax(dim=(1, 3)) / 127.0   # 尾块 amax 只含有效行
    assert torch.allclose(k_scale[..., 2], expect_tail)
    scale_row = k_scale.repeat_interleave(128, dim=-1)[..., :s].permute(0, 2, 1)[..., None]
    assert ((k_int8.float() * scale_row - kf).abs() <= scale_row * (0.5 + 3e-5)).all()


def test_v_pad_zero_scale_and_roundtrip():
    torch.manual_seed(0)
    b, s, n, d = 1, 337, 2, 128
    v = torch.randn(b, s, n, d, dtype=torch.bfloat16)
    v_fp8, v_scale = quant_v_fp8_per_channel(v)
    assert v_fp8.shape == (b, n, d, 384) and v_fp8.dtype == torch.float8_e4m3fn
    assert v_fp8.is_contiguous() and v_scale.shape == (b, n, d)
    assert (v_fp8.float()[..., s:] == 0).all()       # pad 区精确零
    vt = v.permute(0, 2, 3, 1).float()
    assert torch.allclose(v_scale, vt.abs().amax(-1).clamp_min(1e-7) / 448.0)
    deq = v_fp8.float()[..., :s] * v_scale[..., None]
    # e4m3 正规数相对误差 ≤ 2^-4，次正规绝对步长 ≤ v_scale·2^-9
    bound = vt.abs() * 2.0 ** -4 + v_scale[..., None] * 2.0 ** -9
    assert ((deq - vt).abs() <= bound + 1e-7).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
```

- [ ] **Step 4: 跑测试确认失败**

```bash
./cutedsl_sage/hyper01.sh test
```
Expected: FAIL/ERROR，`ModuleNotFoundError`/`ImportError: cannot import name 'quant_q_int8_per_warp'`（core.py 还不存在）。

- [ ] **Step 5: 写量化实现（创建 `cutedsl_sage/core.py`）**

```python
# CuteDSL SageAttention (Hopper)：torch 量化 + CuTe-DSL kernel + 端到端 API。
# kernel 部分见文件下半部（Task 3 追加）。
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# ===== torch 量化（正确性优先，后续可换 triton/融合 kernel）=====

_AMAX_EPS = 1e-7    # 防除零下限，与 CUDA 版一致（csrc/fused/fused.cu:147）


def quant_q_int8_per_warp(q: torch.Tensor):
    """Q per-warp int8 量化：seq 按 64 分块、块内 4 段各 16 行，amax over (16 行, d)。

    q: [b, s, n, d] fp16/bf16 -> (q_int8 [b,s,n,d] int8, q_scale [b, n, ceil(s/64)*4] fp32)
    scale 段序 = seq 顺序的 16 行段（行 r 对应索引 r//16），与 CUDA quant_per_warp_int8_cuda 一致。
    尾块补零行不抬高 amax；全 padding 段 scale 钳到 _AMAX_EPS/127（非 0/NaN）。
    """
    b, s, n, d = q.shape
    nblk = (s + 63) // 64
    s_pad = nblk * 64
    qf = q.float()
    if s_pad != s:
        qf = F.pad(qf, (0, 0, 0, 0, 0, s_pad - s))
    qv = qf.view(b, nblk * 4, 16, n, d)
    amax = qv.abs().amax(dim=(2, 4)).clamp_min(_AMAX_EPS)              # [b, nblk*4, n]
    q_scale = (amax / 127.0).permute(0, 2, 1).contiguous()             # [b, n, nblk*4]
    # torch.round 为 round-half-to-even，与 CUDA float_to_int8_rn (cvt.rni) 一致
    q_int8 = torch.round(qv * (127.0 / amax)[:, :, None, :, None]) \
        .clamp_(-128, 127).to(torch.int8)
    return q_int8.view(b, s_pad, n, d)[:, :s].contiguous(), q_scale


def quant_k_int8_per_block(k: torch.Tensor, km: Optional[torch.Tensor] = None):
    """K per-block int8 量化（BLKK=128），量化前减 km（smooth_k）。

    k: [b, s, n_kv, d]，km: [b, 1, n_kv, d] 或 None
    -> (k_int8 [b,s,n_kv,d] int8, k_scale [b, n_kv, ceil(s/128)] fp32)
    先减 km 再补零，保证 padding 行不污染尾块 amax。
    """
    b, s, n, d = k.shape
    nblk = (s + 127) // 128
    s_pad = nblk * 128
    kf = k.float()
    if km is not None:
        kf = kf - km.float()
    if s_pad != s:
        kf = F.pad(kf, (0, 0, 0, 0, 0, s_pad - s))
    kv = kf.view(b, nblk, 128, n, d)
    amax = kv.abs().amax(dim=(2, 4)).clamp_min(_AMAX_EPS)              # [b, nblk, n]
    k_scale = (amax / 127.0).permute(0, 2, 1).contiguous()             # [b, n, nblk]
    k_int8 = torch.round(kv * (127.0 / amax)[:, :, None, :, None]) \
        .clamp_(-128, 127).to(torch.int8)
    return k_int8.view(b, s_pad, n, d)[:, :s].contiguous(), k_scale


def quant_v_fp8_per_channel(v: torch.Tensor):
    """V per-channel e4m3 量化并 materialize 为 [b, n_kv, d, s_pad] contiguous。

    v: [b, s, n_kv, d]；s_pad = ceil(s/128)*128，尾部零补（e4m3 精确 0）。
    v_scale = amax/448（amax 仅统计有效 seq），v_fp8 = v·448/amax。无 token permute。
    """
    b, s, n, d = v.shape
    s_pad = (s + 127) // 128 * 128
    vt = v.permute(0, 2, 3, 1).float()                                 # [b, n, d, s]
    amax = vt.abs().amax(dim=-1).clamp_min(_AMAX_EPS)                  # [b, n, d]
    v_scale = (amax / 448.0).contiguous()
    v_scaled = vt * (448.0 / amax)[..., None]                          # 幅值 ≤448，e4m3 不溢出
    if s_pad != s:
        v_scaled = F.pad(v_scaled, (0, s_pad - s))
    v_fp8 = v_scaled.to(torch.float8_e4m3fn).contiguous()
    return v_fp8, v_scale
```

- [ ] **Step 6: 跑测试确认通过**

```bash
./cutedsl_sage/hyper01.sh test
```
Expected: 4 个量化单测全部 PASS。

- [ ] **Step 7: Commit**

```bash
git add cutedsl_sage/cutedsl_sage.py cutedsl_sage/core.py cutedsl_sage/test_sage_sm90.py
git commit -m "$(cat <<'EOF'
Fix skeleton SyntaxError; add torch per-warp/per-block/per-channel quant with unit tests

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: CuTe-DSL kernel 主体 + 编译冒烟

**Files:**
- Modify: `cutedsl_sage/core.py`（在量化函数之后追加 kernel 部分与端到端入口）
- Modify: `cutedsl_sage/test_sage_sm90.py`（追加编译冒烟测试）

**Interfaces:**
- Consumes: Task 2 的三个 quant 函数（签名见 Task 2）
- Produces:
  - `class SageAttnSm90(CutedslKernel)`：`from_args(head_dim: int, is_causal: bool, out_dtype: torch.dtype, gqa_ratio: int) -> SageAttnSm90`（编译缓存）；`run(q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale, o, sm_scale)` 就地写 `o`
  - `sageattn_qk_int8_pv_fp8_hopper(q, k, v, is_causal=False, sm_scale=None, smooth_k=True) -> Tensor`（端到端，Task 4/5 测试对象）

**说明：** 以下代码是「多 agent 起草 + 两名独立审查员对照本地 CuTeDSL 源码/CUDA 基准逐项验证」后的版本，验证发现的 4 处修复已应用（删除未使用的 rank-4 切片、run 层 causal 守卫、mainloop 后 q_handle.release()、两处 UNCERTAIN 经源码确认后清除）。注释中 `fmha.py:NNN` 指本地底本 `/home/ubuntu/workspace/github/llm/flash-attention/csrc/cutlass/examples/python/CuTeDSL/hopper/fmha.py` 行号，编译报错时按行号对照。

- [ ] **Step 1: 在 `cutedsl_sage/core.py` 末尾追加 kernel 部分**

```python
# ===== CuTe-DSL kernel（Hopper sm90a）=====
# 移植底本：cutlass examples/python/CuTeDSL/hopper/fmha.py（注释 fmha.py:NNN 为其行号）

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

try:
    from .cutedsl_sage import CutedslKernel
except ImportError:
    from cutedsl_sage import CutedslKernel

LOG2_E = math.log2(math.e)

_TORCH_TO_CUTLASS = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}


class SageAttnSm90(CutedslKernel):
    # 独立缓存，key = (head_dim, is_causal, out_dtype, gqa_ratio)
    compile_cache: dict = {}

    CTA_Q = 64
    CTA_K = 128

    def __init__(self, head_dim: int, is_causal: bool, o_dtype, gqa_ratio: int):
        super().__init__()
        assert head_dim in (64, 128)
        self.head_dim = head_dim
        self.is_causal = is_causal
        self.gqa_ratio = gqa_ratio

        self.q_dtype = cutlass.Int8
        self.k_dtype = cutlass.Int8
        self.v_dtype = cutlass.Float8E4M3FN
        self.o_dtype = o_dtype
        self.qk_acc_dtype = cutlass.Int32      # s8s8s32，|S|<2^24 转 f32 无损
        self.pv_acc_dtype = cutlass.Float32    # 单级 FP32（FP22）累加

        # fmha.py:160-170
        self.qk_mma_tiler = (self.CTA_Q, self.CTA_K, head_dim)
        self.pv_mma_tiler = (self.CTA_Q, head_dim, self.CTA_K)
        self.cta_tiler = self.qk_mma_tiler
        self.atom_layout_mnk = (1, 1, 1)
        self.cluster_shape_mnk = (1, 1, 1)

        # K/V 共用一个环形 smem 缓冲（fmha.py:601），每次 KV 迭代占 2 个 stage（K 一个 V 一个），
        # 一个 stage = 128*d 字节（K 块与 V 块字节数恒等）
        self.q_stage = 1
        self.kv_stage = 6 if head_dim == 128 else 10
        self.epi_stage = 2

        self.num_threads_per_warp_group = 128
        self.num_warps_per_warp_group = 4
        self.threads_per_cta = 256            # WG0 = load, WG1 = math
        self.load_warp_group_id = 0
        self.math_warp_group_id = 1
        self.num_regs_load = 24
        self.num_regs_mma = 240
        self.buffer_align_bytes = 1024

    # ---------------- host：编译缓存 ----------------
    @classmethod
    def from_args(cls, head_dim: int, is_causal: bool, out_dtype: torch.dtype,
                  gqa_ratio: int) -> "SageAttnSm90":
        key = (head_dim, is_causal, out_dtype, gqa_ratio)
        cached = cls.compile_cache.get(key)
        if cached is not None:
            return cached

        kernel = cls(head_dim, is_causal, _TORCH_TO_CUTLASS[out_dtype], gqa_ratio)

        # b/s/n 动态、d/dtype 静态；紧凑行主序 fake tensor（runtime.py:604），
        # stride_order 值 0 = 最小 stride（runtime.py:642 示例），(3,2,1,0) 即 torch contiguous
        sym = cute.sym_int
        b, s_q, s_k = sym(), sym(), sym()
        s_pad = sym(divisibility=128)
        n_q = sym()
        n_kv = n_q if gqa_ratio == 1 else sym()
        n_wb_q, n_blk_k = sym(), sym()

        def fake4(dtype, shape):
            return cute.runtime.make_fake_compact_tensor(
                dtype, shape, stride_order=(3, 2, 1, 0), assumed_align=16)

        def fake3(dtype, shape):
            return cute.runtime.make_fake_compact_tensor(
                dtype, shape, stride_order=(2, 1, 0), assumed_align=4)

        q_fake = fake4(cutlass.Int8, (b, s_q, n_q, head_dim))
        k_fake = fake4(cutlass.Int8, (b, s_k, n_kv, head_dim))
        # fp8 走 uint8 view workaround（interface.py:988），__call__ 内 recast 回 e4m3
        v_fake = fake4(cutlass.Uint8, (b, n_kv, head_dim, s_pad))
        o_fake = fake4(kernel.o_dtype, (b, s_q, n_q, head_dim))
        qs_fake = fake3(cutlass.Float32, (b, n_q, n_wb_q))     # ceil(s_q/64)*4
        ks_fake = fake3(cutlass.Float32, (b, n_kv, n_blk_k))   # ceil(s_k/128)
        vs_fake = fake3(cutlass.Float32, (b, n_kv, head_dim))

        kernel.compiled_kernel = cute.compile(          # interface.py:980 同款
            kernel,
            q_fake, k_fake, v_fake, qs_fake, ks_fake, vs_fake, o_fake,
            0.0,                                        # scale_softmax_log2（运行时标量）
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),  # runtime.py:712
            options="--enable-tvm-ffi",
        )
        cls.compile_cache[key] = kernel
        return kernel

    def run(self, q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale, o, sm_scale):
        if self.compiled_kernel is None:
            raise RuntimeError("SageAttnSm90 must be created via from_args before running")
        if self.is_causal:
            # causal mask 为 top-left 对齐且 KV 上界按 q 位置截断，s_q != s_k 时结果静默错误
            assert q_int8.shape[1] == k_int8.shape[1], "causal requires s_q == s_k"
        # TVM-FFI 编译产物直接吃 torch.Tensor，stream 取调用方当前 torch stream
        self.compiled_kernel(
            q_int8, k_int8, v_fp8.view(torch.uint8),    # interface.py:988 fp8 workaround
            q_scale, k_scale, v_scale, o,
            sm_scale * LOG2_E,
        )

    # ---------------- host jit：TiledMma / TMA / smem / launch（fmha.py:204-475）----------------
    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,          # (b, s_q, n_q, d)  int8
        k: cute.Tensor,          # (b, s_k, n_kv, d) int8
        v: cute.Tensor,          # (b, n_kv, d, s_pad) uint8(=e4m3)
        q_scale: cute.Tensor,    # (b, n_q, ceil(s_q/64)*4) f32
        k_scale: cute.Tensor,    # (b, n_kv, ceil(s_k/128)) f32
        v_scale: cute.Tensor,    # (b, n_kv, d) f32
        o: cute.Tensor,          # (b, s_q, n_q, d) fp16/bf16
        scale_softmax_log2: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        # uint8 → e4m3（recast_ptr 支持 dtype 重解释，core.py:3080）
        v = cute.make_tensor(
            cute.recast_ptr(v.iterator, dtype=self.v_dtype), v.layout
        )

        # 维度重排（对齐 fmha.py:226-253 的 (s, d, h, b) 约定）
        def _permute(t, order):
            return cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    tuple(t.shape[i] for i in order),
                    stride=tuple(t.stride[i] for i in order),
                ),
            )

        mQ = _permute(q, (1, 3, 2, 0))   # (s_q, d, n_q, b)
        mK = _permute(k, (1, 3, 2, 0))   # (s_k, d, n_kv, b)
        mV = _permute(v, (2, 3, 1, 0))   # (d, s_pad, n_kv, b)，s 连续 → B K-major
        mO = _permute(o, (1, 3, 2, 0))   # (s_q, d, n_q, b)

        self.q_layout = utils.LayoutEnum.from_tensor(mQ)    # fmha.py:290
        self.k_layout = utils.LayoutEnum.from_tensor(mK)
        self.v_layout = utils.LayoutEnum.from_tensor(mV)
        self.o_layout = utils.LayoutEnum.from_tensor(mO)

        self.epi_tile = sm90_utils.compute_tile_shape_or_override(   # fmha.py:286
            self.cta_tiler, self.o_dtype
        )

        # QK: s8s8s32 SS（MmaI8Op, mma.py:408；fmha.py:300）
        qk_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.q_dtype, self.k_dtype,
            self.q_layout.sm90_mma_major_mode(),
            self.k_layout.sm90_mma_major_mode(),
            self.qk_acc_dtype, self.atom_layout_mnk, self.qk_mma_tiler[:2],
        )
        # PV: e4m3 RS，P 在 RMEM（MmaF8Op, mma.py:322；fmha.py:310）
        pv_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.v_dtype, self.v_dtype,
            warpgroup.OperandMajorMode.K,
            self.v_layout.sm90_mma_major_mode(),
            self.pv_acc_dtype, self.atom_layout_mnk, self.pv_mma_tiler[:2],
            warpgroup.OperandSource.RMEM,
        )

        # smem 布局（fmha.py:327-357；d=128→SW128 / d=64→SW64，V s 连续→SW128）
        q_smem_layout_staged = sm90_utils.make_smem_layout_a(
            self.q_layout, self.qk_mma_tiler, self.q_dtype, self.q_stage)
        k_smem_layout_staged = sm90_utils.make_smem_layout_b(
            self.k_layout, self.qk_mma_tiler, self.k_dtype, self.kv_stage)
        v_smem_layout_staged = sm90_utils.make_smem_layout_b(
            self.v_layout, self.pv_mma_tiler, self.v_dtype, self.kv_stage)
        o_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.epi_stage,
            cute.append(cute.append(self.epi_tile, self.epi_stage), 1),
            smem_order=(1, 0, 2, 3) if self.o_layout.is_m_major_c() else (0, 1, 2, 3),
        )

        # TMA atoms（fmha.py:359-401）
        tma_atom_q, tma_tensor_q = self._make_tma_atoms_and_tensors(
            mQ, q_smem_layout_staged,
            (self.qk_mma_tiler[0], self.qk_mma_tiler[2]), 1)
        tma_atom_k, tma_tensor_k = self._make_tma_atoms_and_tensors(
            mK, k_smem_layout_staged,
            (self.qk_mma_tiler[1], self.qk_mma_tiler[2]), 1)
        tma_atom_v, tma_tensor_v = self._make_tma_atoms_and_tensors(
            mV, v_smem_layout_staged,
            (self.pv_mma_tiler[1], self.pv_mma_tiler[2]), 1)

        o_smem_layout = cute.slice_(o_smem_layout_staged, (None, None, 0, 0))
        tma_atom_o, tma_tensor_o = cpasync.make_tiled_tma_atom(   # fmha.py:396
            cpasync.CopyBulkTensorTileS2GOp(), mO, o_smem_layout, self.epi_tile)

        q_smem_layout = cute.slice_(q_smem_layout_staged, (None, None, 0))
        # K/V 块字节数相等（128*d），共用 tx_count（fmha.py:403-406）
        self.tma_copy_q_bytes = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        self.tma_copy_kv_bytes = cute.size_in_bytes(
            self.k_dtype, cute.slice_(k_smem_layout_staged, (None, None, 0)))

        @cute.struct
        class SharedStorage:                              # fmha.py:414
            load_q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            load_kv_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.kv_stage * 2]
            sVScale: cute.struct.MemRange[cutlass.Float32, self.head_dim]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, cute.cosize(o_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(k_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # grid = (ceil(s_q/64), b, n_q)
        grid = (cute.ceil_div(mQ.shape[0], self.CTA_Q), mQ.shape[3], mQ.shape[2])

        self.kernel(                                       # fmha.py:446-475
            qk_tiled_mma, pv_tiled_mma,
            tma_atom_q, tma_tensor_q,
            tma_atom_k, tma_tensor_k,
            tma_atom_v, tma_tensor_v,
            tma_atom_o, tma_tensor_o,
            q_scale, k_scale, v_scale,
            scale_softmax_log2,
            q_smem_layout_staged, k_smem_layout_staged,
            v_smem_layout_staged, o_smem_layout_staged,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    # ---------------- device kernel ----------------
    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom, mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom, mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom, mV_dkl: cute.Tensor,
        tma_atom_o: cute.CopyAtom, mO_qdl: cute.Tensor,
        mQScale: cute.Tensor,
        mKScale: cute.Tensor,
        mVScale: cute.Tensor,
        scale_softmax_log2: cutlass.Float32,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx_q, b_idx, head_idx = cute.arch.block_idx()
        kv_head = head_idx // self.gqa_ratio            # GQA

        smem = utils.SmemAllocator()                    # fmha.py:570
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = self.make_and_init_load_q_pipeline(
            storage.load_q_mbar_ptr.data_ptr())
        load_kv_producer, load_kv_consumer = self.make_and_init_load_kv_pipeline(
            storage.load_kv_mbar_ptr.data_ptr())
        tma_store_pipeline = self.make_and_init_tma_store_pipeline()

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group)

        # smem 张量；V 与 K 共用同一缓冲、按 stage 交替（fmha.py:593-602）
        sQ = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner)
        sK = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner)
        sV = cute.make_tensor(
            cute.recast_ptr(sK.iterator, v_smem_layout_staged.inner, self.v_dtype),
            v_smem_layout_staged.outer)
        sO = storage.sO.get_tensor(
            o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner)
        sVScale = storage.sVScale.get_tensor(cute.make_layout(self.head_dim))

        seqlen_q = mQ_qdl.shape[0]
        seqlen_k = mK_kdl.shape[0]

        # KV 迭代次数；causal 上界 ceil((bx+1)*64/128)，且仅最后 1 块需要 mask（64<128 保证）
        k_blk_total = cute.ceil_div(seqlen_k, self.CTA_K)
        if const_expr(self.is_causal):
            trip_count = min(
                cute.ceil_div((bidx_q + 1) * self.CTA_Q, self.CTA_K), k_blk_total)
        else:
            trip_count = k_blk_total

        # TMA partition（fmha.py:615-647）
        gQ_qdl = cute.flat_divide(mQ_qdl, cute.select(self.qk_mma_tiler, mode=[0, 2]))
        qk_thr_mma = qk_tiled_mma.get_slice(tidx)
        tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
        tQsQ, tQgQ_qdl = cpasync.tma_partition(
            tma_atom_q, 0, cute.make_layout(1),
            cute.group_modes(sQ, 0, 2), cute.group_modes(tSgQ_qdl, 0, 3))

        gK_kdl = cute.flat_divide(mK_kdl, cute.select(self.qk_mma_tiler, mode=[1, 2]))
        tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
        tKsK, tKgK_kdl = cpasync.tma_partition(
            tma_atom_k, 0, cute.make_layout(1),
            cute.group_modes(sK, 0, 2), cute.group_modes(tSgK_kdl, 0, 3))

        gV_dkl = cute.flat_divide(mV_dkl, cute.select(self.pv_mma_tiler, mode=[1, 2]))
        pv_thr_mma = pv_tiled_mma.get_slice(tidx)
        tOsV_g = pv_thr_mma.partition_B(gV_dkl)
        tVsV, tVgV_dkl = cpasync.tma_partition(
            tma_atom_v, 0, cute.make_layout(1),
            cute.group_modes(sV, 0, 2), cute.group_modes(tOsV_g, 0, 3))

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk)   # fmha.py:654

        if warp_idx == 0:                                # fmha.py:657
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)
            cpasync.prefetch_descriptor(tma_atom_o)

        # ---------------- producer：load warpgroup ----------------
        if warp_group_idx == self.load_warp_group_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_load)
            if warp_idx == 0:
                # Q 单 stage 一次装载
                q_handle = load_q_producer.acquire_and_advance()
                tQgQ = tQgQ_qdl[(None, None, 0, head_idx, b_idx)]
                cute.copy(tma_atom_q, tQgQ[(None, bidx_q)],
                          tQsQ[(None, q_handle.index)],
                          tma_bar_ptr=q_handle.barrier)       # fmha.py:705-711

                tKgK = tKgK_kdl[(None, None, 0, kv_head, b_idx)]
                tVgV = tVgV_dkl[(None, 0, None, kv_head, b_idx)]
                kv_iter = Int32(0)
                while kv_iter < trip_count:                   # K/V 交替入队
                    k_handle = load_kv_producer.acquire_and_advance()
                    cute.copy(tma_atom_k, tKgK[(None, kv_iter)],
                              tKsK[(None, k_handle.index)],
                              tma_bar_ptr=k_handle.barrier)
                    v_handle = load_kv_producer.acquire_and_advance()
                    cute.copy(tma_atom_v, tVgV[(None, kv_iter)],
                              tVsV[(None, v_handle.index)],
                              tma_bar_ptr=v_handle.barrier)
                    kv_iter += 1

        # ---------------- consumer：math warpgroup ----------------
        if warp_group_idx == self.math_warp_group_id:
            cute.arch.warpgroup_reg_alloc(self.num_regs_mma)

            # per-warp q_scale：idx = ((b*n_q+h)*ceil(s_q/64) + bx)*4 + warp（sm90.cu:177）；
            # acc fragment 行 = 16*warp_local + lane/4 (+8) 恰在该 warp 的 16 行段内，每 warp 一个标量
            warp_local = warp_idx - self.math_warp_group_id * self.num_warps_per_warp_group
            q_scale_val = mQScale[(b_idx, head_idx, bidx_q * 4 + warp_local)]

            # v_scale/448 预折算进 smem，epilogue 按列取用
            t_local = tidx - self.num_threads_per_warp_group
            if t_local < self.head_dim:
                sVScale[t_local] = mVScale[(b_idx, kv_head, t_local)] * (1.0 / 448.0)
            pipeline.arrive_and_wait(                          # named barrier（fmha.py:1171）
                barrier_id=self.math_warp_group_id,
                num_threads=self.num_threads_per_warp_group)

            # mainloop 前置（fmha.py:813-845）
            tSsQ = qk_thr_mma.partition_A(sQ)
            tSsK = qk_thr_mma.partition_B(sK)
            tSrQ = qk_thr_mma.make_fragment_A(tSsQ)
            tSrK = qk_thr_mma.make_fragment_B(tSsK)
            tOsV = pv_thr_mma.partition_B(sV)
            tOrV = pv_thr_mma.make_fragment_B(tOsV)

            q_handle = load_q_consumer.wait()                  # fmha.py:826

            # 全局 (q_idx, k_idx) 坐标（fmha.py:796/829 identity-tensor 技术）
            cP = cute.make_identity_tensor((seqlen_q, seqlen_k))
            gPcP = cute.local_tile(cP, self.qk_mma_tiler[:2], (None, None))
            ptScP = qk_thr_mma.partition_C(gPcP)

            pv_acc_shape = pv_thr_mma.partition_shape_C(
                (self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
            acc_pv = pv_thr_mma.make_fragment_C(pv_acc_shape)
            acc_pv.fill(0.0)
            qk_acc_shape = qk_thr_mma.partition_shape_C(
                (self.qk_mma_tiler[0], self.qk_mma_tiler[1]))

            s_max_layout = cute.make_layout(
                cute.size(self.layout_acc_mn(pv_tiled_mma, acc_pv.layout), mode=[0]))
            s_max = cute.make_rmem_tensor_like(s_max_layout, Float32)
            a_sum = cute.make_rmem_tensor_like(s_max, Float32)
            s_max.fill(-Float32.inf)     # 统一初值 → 无 is_first 特判：
            a_sum.fill(0.0)              # 首次 rescale 因子 exp2(-inf)=0，acc_pv 已清零

            # 前 trip_count-1 块不 mask，最后 1 块做 residual(+causal) mask
            kv_iter = Int32(0)
            load_kv_consumer, kv_iter, s_max, a_sum = self.compute(
                False, trip_count - 1, kv_iter,
                qk_thr_mma, qk_tiled_mma, pv_tiled_mma, load_kv_consumer,
                q_handle, tSrQ, tSrK, tOrV, acc_pv, s_max, a_sum,
                ptScP, bidx_q, qk_acc_shape,
                q_scale_val, mKScale, b_idx, kv_head,
                scale_softmax_log2, seqlen_k)
            load_kv_consumer, kv_iter, s_max, a_sum = self.compute(
                True, Int32(1), kv_iter,
                qk_thr_mma, qk_tiled_mma, pv_tiled_mma, load_kv_consumer,
                q_handle, tSrQ, tSrK, tOrV, acc_pv, s_max, a_sum,
                ptScP, bidx_q, qk_acc_shape,
                q_scale_val, mKScale, b_idx, kv_head,
                scale_softmax_log2, seqlen_k)

            cute.nvgpu.warpgroup.wait_group(0)
            q_handle.release()                                 # fmha.py:1037

            # ---- tail：行和归约 + O = acc · v_scale/448 · 1/row_sum（fmha.py:1503-1561 改造）----
            reduction_target_pv = self.reduction_target_n(pv_tiled_mma)
            for r in cutlass.range_constexpr(cute.rank(reduction_target_pv)):
                for i in cutlass.range_constexpr(cute.size(a_sum)):
                    a_sum[i] = cute.arch.warp_reduction_sum(
                        a_sum[i], threads_in_group=reduction_target_pv.shape[r])

            acc_mn = cute.make_tensor(
                acc_pv.iterator, self.layout_acc_mn(pv_tiled_mma, acc_pv.layout))
            cO = cute.make_identity_tensor(
                (self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
            tOcO = pv_thr_mma.partition_C(cO)
            tOcO_mn = cute.make_tensor(
                tOcO.iterator, self.layout_acc_mn(pv_tiled_mma, tOcO.layout))
            for i in cutlass.range_constexpr(cute.size(acc_mn, mode=[0])):
                inv_sum = cute.arch.rcp_approx(a_sum[i])       # fmha.py:1548
                if a_sum[i] == 0.0 or a_sum[i] != a_sum[i]:
                    inv_sum = 1.0
                for j in cutlass.range_constexpr(cute.size(acc_mn, mode=[1])):
                    col = tOcO_mn[(i, j)][1]                   # per-channel v_scale 列索引
                    acc_mn[(i, j)] = acc_mn[(i, j)] * inv_sum * sVScale[col]

            # ---- epilogue：r2s(StMatrix) + TMA store（fmha.py:1092-1190，单 math WG）----
            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                self.o_layout, elem_ty_d=self.o_dtype, elem_ty_acc=self.pv_acc_dtype)
            copy_atom_O = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(self.o_layout.is_m_major_c(), 4),
                self.o_dtype)
            tiled_copy_O_Atom = cute.make_tiled_copy_C_atom(copy_atom_O, pv_tiled_mma)
            tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_O_Atom)
            thr_copy_r2s = tiled_copy_r2s.get_slice(
                tidx % self.num_threads_per_warp_group)
            tRS_sD = thr_copy_r2s.partition_D(sO)
            tRS_rAcc = tiled_copy_r2s.retile(acc_pv)

            rD_shape = cute.shape(thr_copy_r2s.partition_S(sO))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor_like(tRS_rD_layout, self.pv_acc_dtype)
            size_tRS_rD = cute.size(tRS_rD)

            gD = cute.local_tile(mO_qdl, self.pv_mma_tiler[:2],
                                 (bidx_q, 0, head_idx, b_idx))   # fmha.py:1130
            sepi_for_tma_partition = cute.group_modes(sO, 0, 2)
            tcgc_for_tma_partition = cute.zipped_divide(gD, self.epi_tile)
            bSG_sD, bSG_gD = cpasync.tma_partition(
                tma_atom_o, 0, cute.make_layout(1),
                sepi_for_tma_partition, tcgc_for_tma_partition)
            epi_tile_num = cute.size(tcgc_for_tma_partition, mode=[1])

            for epi_idx in cutlass.range_constexpr(epi_tile_num):
                for epi_v in cutlass.range_constexpr(size_tRS_rD):
                    tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]
                tRS_rD_out = cute.make_rmem_tensor_like(tRS_rD_layout, self.o_dtype)
                acc_vec = tRS_rD.load()
                tRS_rD_out.store(acc_vec.to(self.o_dtype))
                epi_buffer = epi_idx % self.epi_stage
                cute.copy(tiled_copy_r2s, tRS_rD_out,
                          tRS_sD[(None, None, None, epi_buffer, 0)])
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta)
                pipeline.arrive_and_wait(
                    barrier_id=self.math_warp_group_id,
                    num_threads=self.num_threads_per_warp_group)
                if warp_idx == 4:            # math WG 首 warp 发 TMA store
                    cute.copy(tma_atom_o, bSG_sD[(None, epi_buffer, 0)],
                              bSG_gD[(None, epi_idx)])
                    tma_store_pipeline.producer_commit()
                    tma_store_pipeline.producer_acquire()
                pipeline.arrive_and_wait(
                    barrier_id=self.math_warp_group_id,
                    num_threads=self.num_threads_per_warp_group)
        return

    # ---------------- mainloop 迭代（fmha.py:1199-1299 改造：dequant + 运行时 scale）----------------
    @cute.jit
    def compute(
        self,
        masked: cutlass.Constexpr,
        k_tile_count: Int32,
        kv_iter: Int32,
        qk_thr_mma: cute.ThrMma,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        load_kv_consumer: pipeline.PipelineConsumer,
        q_handle,
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        tOrV: cute.Tensor,
        acc_pv: cute.Tensor,
        s_max: cute.Tensor,
        a_sum: cute.Tensor,
        ptScP: cute.Tensor,
        bidx_q: Int32,
        qk_acc_shape: cute.Shape,
        q_scale_val: Float32,
        mKScale: cute.Tensor,
        b_idx: Int32,
        kv_head: Int32,
        scale_softmax_log2: Float32,
        seqlen_k: Int32,
    ) -> Tuple[pipeline.PipelineConsumer, Int32, cute.Tensor, cute.Tensor]:
        while k_tile_count > 0:
            k_tile_count -= 1
            tScP = cute.slice_(ptScP, (None, None, None, bidx_q, kv_iter))  # fmha.py:1233
            # per-block k_scale：idx = (b*n_kv + kv_head)*ceil(s_k/128) + kv_iter（sm90.cu:188）
            k_scale_val = mKScale[(b_idx, kv_head, kv_iter)]
            kv_iter += 1

            acc_qk = qk_thr_mma.make_fragment_C(qk_acc_shape)   # Int32 acc
            k_handle = load_kv_consumer.wait_and_advance()

            cute.nvgpu.warpgroup.fence()                        # fmha.py:1242
            self.gemm_zero_acc(
                qk_tiled_mma,
                tSrQ[(None, None, None, q_handle.index)],
                tSrK[(None, None, None, k_handle.index)],
                acc_qk)
            cute.nvgpu.warpgroup.commit_group()
            tok = load_kv_consumer.try_wait()                   # fmha.py:1253
            cute.nvgpu.warpgroup.wait_group(0)

            # dequant：S_f32 = f32(S_i32) · q_scale(per-warp) · k_scale(per-block)
            # TensorSSA.to 支持 int→float（tensor.py:1772 sitofp），Int32→Float32 无损
            acc_s = cute.make_rmem_tensor_like(acc_qk, Float32)
            acc_s.store(acc_qk.load().to(Float32))
            dequant = q_scale_val * k_scale_val
            for i in cutlass.range_constexpr(cute.size(acc_s)):
                acc_s[i] = acc_s[i] * dequant

            if const_expr(masked):
                # residual mask（k 尾块）+ causal 对角 mask，均在 dequant 后 S_f32 上做
                for i in cutlass.range_constexpr(cute.size(acc_s)):
                    q_pos = tScP[i][0]
                    k_pos = tScP[i][1]
                    if k_pos >= seqlen_k:
                        acc_s[i] = -Float32.inf
                    if const_expr(self.is_causal):
                        if k_pos > q_pos:
                            acc_s[i] = -Float32.inf

            s_max, a_sum = self.softmax_step(
                acc_s, qk_tiled_mma, s_max, a_sum, acc_pv, pv_tiled_mma,
                scale_softmax_log2)

            # P∈[0,1] 显式 ×448 后转 e4m3（契约；epilogue 用 v_scale/448 除回）
            for i in cutlass.range_constexpr(cute.size(acc_s)):
                acc_s[i] = acc_s[i] * 448.0
            acc_p_op = self.make_acc_into_op(
                acc_s, pv_tiled_mma.tv_layout_A, self.v_dtype)  # fmha.py:1275
            v_handle = load_kv_consumer.wait_and_advance(tok)

            cute.nvgpu.warpgroup.fence()                        # fmha.py:1282
            pv_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            cute.gemm(pv_tiled_mma, acc_pv, acc_p_op,
                      tOrV[(None, None, None, v_handle.index)], acc_pv)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

            k_handle.release()
            v_handle.release()
        return load_kv_consumer, kv_iter, s_max, a_sum

    # ---------------- online softmax（fmha.py:1301-1397 改造：m 统一初值 -inf）----------------
    @cute.jit
    def softmax_step(
        self,
        acc_s: cute.Tensor,
        qk_tiled_mma: cute.TiledMma,
        s_max: cute.Tensor,
        a_sum: cute.Tensor,
        acc_pv: cute.Tensor,
        pv_tiled_mma: cute.TiledMma,
        scale_softmax_log2: Float32,
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        acc_s_mn = cute.make_tensor(
            acc_s.iterator, self.layout_acc_mn(qk_tiled_mma, acc_s.layout))
        acc_pv_mn = cute.make_tensor(
            acc_pv.iterator, self.layout_acc_mn(pv_tiled_mma, acc_pv.layout))
        reduction_target_qk = self.reduction_target_n(qk_tiled_mma)
        red_rank = cute.rank(reduction_target_qk)
        s_max_prev = cute.make_rmem_tensor_like(s_max, Float32)

        for i in cutlass.range_constexpr(cute.size(acc_s_mn, mode=[0])):
            s_max_prev[i] = s_max[i]
            for j in cutlass.range_constexpr(cute.size(acc_s_mn, mode=[1])):
                s_max[i] = cutlass.max(s_max[i], acc_s_mn[(i, j)])
            for r in cutlass.range_constexpr(red_rank):        # quad reduction
                s_max[i] = cute.arch.warp_reduction_max(
                    s_max[i], threads_in_group=reduction_target_qk.shape[r])

            local_max = s_max[i]
            if s_max[i] == -Float32.inf:      # 整行被 mask 时避免 -inf 参与运算
                local_max = 0.0
            scale_max = scale_softmax_log2 * local_max
            for j in cutlass.range_constexpr(cute.size(acc_s_mn, mode=[1])):
                acc_s_mn[(i, j)] = cute.math.exp2(
                    scale_softmax_log2 * acc_s_mn[(i, j)] - scale_max, fastmath=True)

            # 首迭代 s_max_prev=-inf → scale_pv=0，旧 acc（已清零）与旧 sum 均被清除
            scale_pv = cute.math.exp2(
                (s_max_prev[i] - local_max) * scale_softmax_log2, fastmath=True)
            a_sum[i] = a_sum[i] * scale_pv + acc_s_mn[(i, None)].load().reduce(
                cute.ReductionOp.ADD, Float32.zero, 0)         # fmha.py:1393
            for j in cutlass.range_constexpr(cute.size(acc_pv_mn, mode=[1])):
                acc_pv_mn[(i, j)] *= scale_pv

        return s_max, a_sum

    # ---------------- 以下工具方法从 fmha.py 照抄 ----------------
    @cute.jit
    def reduction_target_n(self, tiled_mma):                   # fmha.py:1399
        separated = self.layout_separate(
            tiled_mma.shape_mnk[0],
            cute.make_layout(tiled_mma.tv_layout_C.shape[0]),
            tiled_mma.tv_layout_C.stride[0],
        )
        return separated[1]

    @staticmethod
    def convert_c_layout_to_a_layout(c, a):                    # fmha.py:1408
        return cute.make_layout(
            (a, c.shape[1], (c.shape[2], cute.size(c, mode=[0]) // cute.size(a))),
            stride=(
                c.stride[0],
                c.stride[1],
                (c.stride[2], cute.size(a, mode=[2]) * c.stride[0][2]),
            ),
        )

    @cute.jit
    def make_acc_into_op(self, acc, operand_layout_tv, Element):  # fmha.py:1419-1500
        operand = cute.make_rmem_tensor_like(
            self.convert_c_layout_to_a_layout(acc.layout, operand_layout_tv.shape[1]),
            Element,
        )
        operand_as_acc = cute.make_tensor(operand.iterator, acc.layout)
        acc_vec = acc.load()
        operand_as_acc.store(acc_vec.to(Element))

        if cutlass.const_expr(Element.width == 8):
            # 8-bit A fragment 的 warp 内重排（shuffle_sync + prmt），逐字照抄底本
            tidx, _, _ = cute.arch.thread_idx()
            tid = tidx % 4
            values_u32 = cute.recast_tensor(operand, cutlass.Uint32)
            for n in cutlass.range_constexpr(cute.size(values_u32, mode=[1])):
                for k in cutlass.range_constexpr(cute.size(values_u32, mode=[2])):
                    for ii in cutlass.range_constexpr(0, 8, 4):
                        values_tmp_0 = values_u32[ii // 2 + 0, n, k]
                        values_tmp_1 = values_u32[ii // 2 + 1, n, k]

                        v_to_send = 1
                        if tid == 1 or tid == 2:
                            v_to_send = 0
                        t_to_recv_from = (0x3021 >> (tid * 4)) & 0xF
                        values_tmp_a = values_tmp_1
                        if v_to_send == 0:
                            values_tmp_a = values_tmp_0
                        values_tmp_a = cute.arch.shuffle_sync_op(
                            values_tmp_a, t_to_recv_from, 0xFFFFFFFF, 7199)

                        v_to_send = 1 - v_to_send
                        t_to_recv_from = (0x2130 >> (tid * 4)) & 0xF
                        values_tmp_b = values_tmp_1
                        if v_to_send == 0:
                            values_tmp_b = values_tmp_0
                        values_tmp_b = cute.arch.shuffle_sync_op(
                            values_tmp_b, t_to_recv_from, 0xFFFFFFFF, 7199)

                        order = 0x5410
                        if v_to_send == 0:
                            order = 0x1054
                        values_u32[ii // 2 + 0, n, k] = cute.arch.prmt(
                            values_tmp_a, values_tmp_b, order)
                        order = 0x7632
                        if v_to_send == 0:
                            order = 0x3276
                        values_u32[ii // 2 + 1, n, k] = cute.arch.prmt(
                            values_tmp_a, values_tmp_b, order)
        return operand

    @staticmethod
    def layout_separate(thr, src, ref):                        # fmha.py:1563
        lt = cute.make_layout(())
        ge = cute.make_layout(())
        for k, v in enumerate(ref):
            if cutlass.const_expr(v < thr):
                lt = cute.append(lt, src[k])
            else:
                ge = cute.append(ge, src[k])
        r = None
        if cutlass.const_expr(cute.rank(lt) == 1):
            r = cute.append(lt, ge)
        else:
            r = cute.append(cute.append(cute.make_layout(()), lt), ge)
        return r

    @staticmethod
    @cute.jit
    def gemm_zero_acc(tiled_mma, A, B, C):                     # fmha.py:1581
        rA = cute.rank(A)
        rB = cute.rank(B)
        rC = cute.rank(C)
        if cutlass.const_expr(rA == 2 and rB == 2 and rC == 1):
            for k_block_idx in range(cute.size(A, mode=[1]), unroll_full=True):
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, k_block_idx != 0)
                cute.gemm(tiled_mma, C, A[None, k_block_idx], B[None, k_block_idx], C)
        elif cutlass.const_expr(rA == 3 and rB == 3 and rC == 3):
            for k_block_idx in range(cute.size(A, mode=[2]), unroll_full=True):
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, k_block_idx != 0)
                cute.gemm(tiled_mma, C, A[None, None, k_block_idx],
                          B[None, None, k_block_idx], C)
        else:
            assert 0

    @cute.jit
    def layout_acc_mn(self, tiled_mma, acc):                   # fmha.py:1610
        separated = self.layout_separate(
            tiled_mma.shape_mnk[0], acc[0], tiled_mma.tv_layout_C.stride[1])
        V_M = separated[0]
        V_N = separated[1]
        V_M1 = None
        V_N1 = None
        if cutlass.const_expr(cute.rank(V_M) == 1):
            V_M1 = cute.append(V_M, acc[1])
        else:
            V_M1 = cute.append(cute.append(cute.make_layout(()), V_M), acc[1])
        if cutlass.const_expr(cute.rank(V_N) == 1):
            V_N1 = cute.append(V_N, acc[2])
        else:
            V_N1 = cute.append(cute.append(cute.make_layout(()), V_N), acc[2])
        r = None
        if cutlass.const_expr(cute.rank(V_M1) == 1):
            r = cute.append(V_M1, V_N1)
        else:
            r = cute.append(cute.append(cute.make_layout(()), V_M1), V_N1)
        return r

    # ---------------- pipeline 构造（fmha.py:1636-1680；单 math WG）----------------
    def make_and_init_load_q_pipeline(self, load_q_mbar_ptr):
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_warps_per_warp_group)
        return pipeline.PipelineTmaAsync.create(
            barrier_storage=load_q_mbar_ptr,
            num_stages=self.q_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.tma_copy_q_bytes,
            defer_sync=True,
        ).make_participants()

    def make_and_init_load_kv_pipeline(self, load_kv_mbar_ptr):
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_warps_per_warp_group)
        return pipeline.PipelineTmaAsync.create(
            barrier_storage=load_kv_mbar_ptr,
            num_stages=self.kv_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.tma_copy_kv_bytes,
            defer_sync=True,
        ).make_participants()

    def make_and_init_tma_store_pipeline(self):                # fmha.py:1672
        tma_store_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        return pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=tma_store_producer_group,
        )

    @staticmethod
    def _make_tma_atoms_and_tensors(tensor, smem_layout_staged, smem_tile,
                                    mcast_dim):                # fmha.py:1696-1731
        op = (
            cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op, tensor, smem_layout, smem_tile, num_multicast=mcast_dim)
        return tma_atom, tma_tensor


# ===== 端到端入口 =====
def sageattn_qk_int8_pv_fp8_hopper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
) -> torch.Tensor:
    """SageAttention forward（Hopper，QK int8 per-warp + PV fp8 per-channel）。NHD 布局。"""
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    assert q.dtype in (torch.float16, torch.bfloat16) and q.dtype == k.dtype == v.dtype
    b, s_q, n_q, head_dim = q.shape
    _, s_k, n_kv, _ = k.shape
    assert head_dim in (64, 128)
    assert k.shape[-1] == head_dim and v.shape[-1] == head_dim
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    assert n_q % n_kv == 0
    if is_causal:
        assert s_q == s_k, "causal requires s_q == s_k"
    if sm_scale is None:
        sm_scale = head_dim ** -0.5

    km = k.mean(dim=1, keepdim=True) if smooth_k else None
    q_int8, q_scale = quant_q_int8_per_warp(q)
    k_int8, k_scale = quant_k_int8_per_block(k, km)
    v_fp8, v_scale = quant_v_fp8_per_channel(v)

    o = torch.empty(q.shape, dtype=q.dtype, device=q.device)

    kernel = SageAttnSm90.from_args(head_dim, is_causal, q.dtype, n_q // n_kv)
    kernel.run(q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale, o, sm_scale)
    return o
```

- [ ] **Step 2: 追加编译冒烟测试到 `cutedsl_sage/test_sage_sm90.py`**

在文件末尾（`if __name__` 块之前）追加：

```python
# ============ kernel 编译冒烟（需 H200 + cutlass DSL）============

def _sm90_available():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


@pytest.mark.skipif(not _sm90_available(), reason="需要 sm90 GPU")
def test_compile_smoke():
    try:
        from .core import SageAttnSm90
    except ImportError:
        from core import SageAttnSm90
    kern = SageAttnSm90.from_args(128, False, torch.float16, 1)
    assert kern.compiled_kernel is not None
    assert SageAttnSm90.from_args(128, False, torch.float16, 1) is kern   # 缓存命中
```

- [ ] **Step 3: 语法检查后远程编译冒烟**

```bash
python3 -m py_compile cutedsl_sage/core.py && echo "syntax OK"
./cutedsl_sage/hyper01.sh test -k 'compile_smoke'
```
Expected: `test_compile_smoke PASS`（首次编译分钟级）。
**编译错误是本任务的预期情况**——按「远程迭代循环」§2 逐个修：报错行 → 注释 `fmha.py:NNN` → 对照底本/容器内安装源码。已知高风险点（起草时的风险清单）：
- `make_smem_layout_epi` 的 `smem_order` 参数形态、`compute_tile_shape_or_override` 返回 tile 与 epi 循环次数的匹配
- `.launch(min_blocks_per_mp=1)` 参数若不存在则删掉
- `pipeline.arrive_and_wait(barrier_id=..., num_threads=...)` 关键字名以安装版本为准
- `cute.slice_(o_smem_layout_staged, (None, None, 0, 0))` 的 rank 与 `make_smem_layout_epi` 返回 rank 匹配
- fake tensor 的 `sym_int(divisibility=128)` 关键字形态

- [ ] **Step 4: Commit**

```bash
git add cutedsl_sage/core.py cutedsl_sage/test_sage_sm90.py
git commit -m "$(cat <<'EOF'
Add CuTe-DSL sm90 SageAttention kernel (int8 QK + fp8 PV, warp-specialized) with e2e API

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: 双参考实现 + kernel 正确性测试 + 远程迭代

**Files:**
- Modify: `cutedsl_sage/test_sage_sm90.py`（追加参考实现与 kernel 正确性测试）

**Interfaces:**
- Consumes: `sageattn_qk_int8_pv_fp8_hopper`、三个 quant 函数（签名见 Task 2/3）
- Produces: `ref_sdpa`、`ref_quant_sim`（逐块 online 版本）、`calc_diff`、`rel_l1`、测试矩阵——Task 5 验收复用

**关键背景（不可回退的决定）：** `ref_quant_sim` 必须与 kernel **逐块同构**（per-128 列块 online softmax、P 在 running-max 域 ×448 转 e4m3）。独立审查已实测证伪「全局 softmax 后统一量化」的写法：即使 kernel 完美，全局域量化 vs 逐块域量化的模型差就有 1-cossim≈3e-4 / maxrel≈2e-2，一级紧阈值必挂。

- [ ] **Step 1: 追加参考实现到 `cutedsl_sage/test_sage_sm90.py`**

在量化单测之后、`_sm90_available` 之前插入：

```python
# ---- 阈值：初值，Task 5 在 H200 实测标定后按 max_observed×3 固化 ----
SIM_DIFF_MAX = 1e-4        # 一级：1-cossim（kernel vs 量化模拟，残差仅 FP22 累加 + exp2/rcp 舍入）
SIM_MAXREL_MAX = 2e-2      # 一级：max|diff|/max|ref|（s≥4096 放宽为 4e-2，FP22 误差随 s 增长）
E2E_DIFF_MAX = 2e-3        # 二级：1-cossim（vs fp32 SDPA，量化损失应在论文水平）
E2E_L1_MAX = 7e-2          # 二级：相对 L1


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """1 - 余弦相似度（bench/utils.py 同款）。"""
    x, y = x.double(), y.double()
    return (1 - 2 * (x * y).sum() / (x * x + y * y).sum()).item()


def rel_l1(x: torch.Tensor, ref: torch.Tensor) -> float:
    return ((x - ref).abs().sum() / ref.abs().sum()).item()


# ============ 参考实现 ============

def ref_sdpa(q, k, v, is_causal, sm_scale):
    """fp32 SDPA 参考（支持 GQA）。输入 NHD，返回 [b, s_q, n_q, d] fp32。"""
    n_q, n_kv = q.shape[2], k.shape[2]
    qf = q.permute(0, 2, 1, 3).float()
    kf = k.permute(0, 2, 1, 3).float()
    vf = v.permute(0, 2, 1, 3).float()
    if n_q != n_kv:
        r = n_q // n_kv
        kf = kf.repeat_interleave(r, dim=1)
        vf = vf.repeat_interleave(r, dim=1)
    s = torch.matmul(qf, kf.transpose(-1, -2)) * sm_scale
    if is_causal:
        mask = torch.ones(q.shape[1], k.shape[1], dtype=torch.bool, device=q.device).tril()
        s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, vf).permute(0, 2, 1, 3)


def ref_quant_sim(q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale, is_causal, sm_scale):
    """fp32 精确模拟量化 attention，与 kernel 逐块同构：
    per-128 列块 online softmax，P 在 running-max 域 ×448 转 e4m3 后累加，
    row_sum 用量化前 f32 P。残差仅剩 kernel 的 FP22 累加与 exp2/rcp 舍入。
    注意：P→e4m3 发生在 running-max 域，与全局 softmax 后量化不可交换（实测差 3e-4）。
    返回 [b, s_q, n_q, d] fp32。
    """
    b, s_q, n_q, d = q_int8.shape
    s_k, n_kv = k_int8.shape[1], k_int8.shape[2]
    g = n_q // n_kv
    dev = q_int8.device

    qf = q_int8.permute(0, 2, 1, 3).float()
    kf = k_int8.permute(0, 2, 1, 3).float()
    vf = v_fp8.float()[..., :s_k]                    # pad 列在 kernel 中被 mask 压零，等价于截断
    if g != 1:
        kf = kf.repeat_interleave(g, dim=1)
        vf = vf.repeat_interleave(g, dim=1)
        k_scale = k_scale.repeat_interleave(g, dim=1)
        v_scale = v_scale.repeat_interleave(g, dim=1)

    qs_row = q_scale.repeat_interleave(16, dim=-1)[..., :s_q]     # 行 r → scale[r//16]
    ks_col = k_scale.repeat_interleave(128, dim=-1)[..., :s_k]    # 列 j → scale[j//128]

    S = torch.matmul(qf, kf.transpose(-1, -2))       # |S|<2^24，fp32 精确
    S = S * qs_row[..., :, None] * ks_col[..., None, :]
    if is_causal:
        mask = torch.ones(s_q, s_k, dtype=torch.bool, device=dev).tril()
        S = S.masked_fill(~mask, float("-inf"))

    CTA_K = 128
    c = sm_scale * LOG2E
    m_run = torch.full((b, n_q, s_q, 1), -5e6, device=dev)   # 有限大负数避免 -inf 参与运算出 NaN
    d_run = torch.zeros_like(m_run)
    acc = torch.zeros(b, n_q, s_q, d, device=dev)
    for it in range((s_k + CTA_K - 1) // CTA_K):
        sl = slice(it * CTA_K, min((it + 1) * CTA_K, s_k))
        sb = S[..., sl]
        m_new = torch.maximum(m_run, sb.amax(-1, keepdim=True))
        resc = torch.exp2((m_run - m_new) * c)
        pb = torch.exp2((sb - m_new) * c)
        d_run = d_run * resc + pb.sum(-1, keepdim=True)       # row_sum 用量化前 P
        pb448 = (pb * 448.0).to(torch.float8_e4m3fn).float()
        acc = acc * resc + torch.matmul(pb448, vf[..., sl].transpose(-1, -2))
        m_run = m_new
    o = acc * (v_scale / 448.0)[:, :, None, :] / d_run
    return o.permute(0, 2, 1, 3)
```

- [ ] **Step 2: 追加 kernel 正确性测试**

在 `test_compile_smoke` 之后追加：

```python
def _sage_call():
    try:
        from .core import sageattn_qk_int8_pv_fp8_hopper
    except ImportError:
        from core import sageattn_qk_int8_pv_fp8_hopper
    return sageattn_qk_int8_pv_fp8_hopper


def _run_two_level(q, k, v, is_causal, smooth_k=True):
    """一级：kernel vs 量化模拟逐元素紧阈值；二级：端到端 vs fp32 SDPA 松阈值。"""
    dtype = q.dtype
    d = q.shape[-1]
    s_max_dim = max(q.shape[1], k.shape[1])
    sm_scale = d ** -0.5
    o = _sage_call()(q, k, v, is_causal=is_causal, sm_scale=sm_scale, smooth_k=smooth_k)
    assert o.shape == q.shape and o.dtype == dtype
    of = o.float()

    # km 表达式与 core.py 端到端内部完全一致 → 量化输入 bit-identical
    km = k.mean(dim=1, keepdim=True) if smooth_k else None
    q_i8, q_sc = quant_q_int8_per_warp(q)
    k_i8, k_sc = quant_k_int8_per_block(k, km)
    v_f8, v_sc = quant_v_fp8_per_channel(v)
    o_sim = ref_quant_sim(q_i8, q_sc, k_i8, k_sc, v_f8, v_sc, is_causal, sm_scale)
    o_sim = o_sim.to(dtype).float()                  # 两侧同 cast，剔除输出精度影响
    d1 = calc_diff(of, o_sim)
    maxrel = ((of - o_sim).abs().max() / o_sim.abs().max().clamp_min(1e-6)).item()
    maxrel_lim = SIM_MAXREL_MAX if s_max_dim < 4096 else 2 * SIM_MAXREL_MAX
    assert d1 < SIM_DIFF_MAX, f"level1 1-cossim={d1:.3e}"
    assert maxrel < maxrel_lim, f"level1 maxrel={maxrel:.3e}"

    o_ref = ref_sdpa(q, k, v, is_causal, sm_scale)
    d2, l2 = calc_diff(of, o_ref), rel_l1(of, o_ref)
    assert d2 < E2E_DIFF_MAX, f"level2 1-cossim={d2:.3e}"
    assert l2 < E2E_L1_MAX, f"level2 rel_l1={l2:.3e}"
    print(f"  L1: 1-cos={d1:.2e} maxrel={maxrel:.2e} | L2: 1-cos={d2:.2e} rel_l1={l2:.2e}")


def _mk_qkv(b, n_q, n_kv, s_q, s_k, d, dtype, seed=42):
    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(b, s_q, n_q, d, dtype=dtype, device=dev)
    # K 加通道偏置让 smooth_k 起实际作用
    k = torch.randn(b, s_k, n_kv, d, dtype=dtype, device=dev) \
        + torch.randn(1, 1, n_kv, d, dtype=dtype, device=dev) * 2
    v = torch.randn(b, s_k, n_kv, d, dtype=dtype, device=dev)
    return q, k, v


# (b, n_q, n_kv, s_q, s_k, d)；大组合标 slow（迭代时 -m "not slow" 跳过）
_SHAPES = (
    [(b, n, n, s, s, d) for b in (1, 2) for n in (8, 24)
     for s in (1024, 4096, 337) for d in (64, 128)]
    + [(1, 8, 8, s, s, d) for s in (64, 100, 128) for d in (64, 128)]   # 单 KV 块边界
    + [(1, 4, 4, 337, 1024, 128)]                                       # s_q != s_k（仅 non-causal）
    + [(2, 8, 4, 1024, 1024, 128), (2, 8, 4, 1024, 1024, 64)]           # GQA
)


def _params():
    out = []
    for c in _SHAPES:
        b, n_q, n_kv, s_q, s_k, d = c
        marks = [pytest.mark.slow] if b * n_q * max(s_q, s_k) >= 2 * 24 * 4096 else []
        out.append(pytest.param(*c, marks=marks,
                                id=f"b{b}nq{n_q}nkv{n_kv}sq{s_q}sk{s_k}d{d}"))
    return out


@pytest.mark.skipif(not _sm90_available(), reason="需要 sm90 GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("is_causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("b,n_q,n_kv,s_q,s_k,d", _params())
def test_kernel_two_level(b, n_q, n_kv, s_q, s_k, d, is_causal, dtype):
    if is_causal and s_q != s_k:
        pytest.skip("causal 要求 s_q == s_k")
    q, k, v = _mk_qkv(b, n_q, n_kv, s_q, s_k, d, dtype)
    _run_two_level(q, k, v, is_causal)


@pytest.mark.skipif(not _sm90_available(), reason="需要 sm90 GPU")
@pytest.mark.parametrize("is_causal", [False, True], ids=["full", "causal"])
def test_kernel_scale_indexing(is_causal):
    """异质量级 stress：相邻 warp 段 / K 块 / V channel 的 amax 差 4 倍，
    kernel 侧任一 scale 索引错位都会被一级比对放大为数量级误差，稳定检出。"""
    b, n, s, d = 1, 4, 512, 128
    q, k, v = _mk_qkv(b, n, n, s, s, d, torch.float16, seed=0)
    dev = q.device
    q = q * (4.0 ** (torch.arange(s, device=dev) // 16 % 4))[None, :, None, None].half()
    k = k * (4.0 ** (torch.arange(s, device=dev) // 128 % 4))[None, :, None, None].half()
    v = v * (4.0 ** (torch.arange(d, device=dev) % 4))[None, None, None, :].half()
    _run_two_level(q, k, v, is_causal)


@pytest.mark.skipif(not _sm90_available(), reason="需要 sm90 GPU")
def test_kernel_no_smooth_k():
    q, k, v = _mk_qkv(1, 8, 8, 1024, 1024, 128, torch.float16)
    _run_two_level(q, k, v, is_causal=False, smooth_k=False)
```

- [ ] **Step 3: 先跑最小用例验证失败模式可读**

```bash
./cutedsl_sage/hyper01.sh test -k 'b1nq8nkv8sq128sk128d128 and full and fp16'
```
Expected: 首轮很可能 FAIL（数值失配）。这是本任务的核心迭代循环——按「远程迭代循环」§3 定位：
- 整体全错/输出全零 → 先查 epilogue 写回与 O tensor 布局（`_permute` 顺序、TMA store box）
- 每 16 行一段错 → q_scale 索引或 warp_local 计算
- 每 128 列一块错 → k_scale 索引或 KV pipeline 串位（K/V handle 次序）
- 输出行间正确但整体偏移 → row_sum / 448 补偿链
- 仅尾块形状（s=337/100）错 → residual mask 或 TMA OOB
- 仅 causal 错 → trip_count 上界或对角 mask

- [ ] **Step 4: 逐级扩大用例集直到非 slow 全绿**

```bash
./cutedsl_sage/hyper01.sh test -m 'not slow' -k 'not scale_indexing'   # 主矩阵
./cutedsl_sage/hyper01.sh test -k 'scale_indexing or no_smooth_k'      # stress + no-smooth
```
Expected: 全部 PASS。若一级阈值（SIM_DIFF_MAX/SIM_MAXREL_MAX）对正确实现仍误报（FP22 实测超初值），先确认失配无结构性 pattern（随机分布、幅度与 s 成比例）再放宽常量，并记录实测值。

- [ ] **Step 5: Commit**

```bash
git add cutedsl_sage/test_sage_sm90.py
git commit -m "$(cat <<'EOF'
Add dual-reference correctness tests (block-wise online quant sim + SDPA)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: 全矩阵验收 + 阈值标定 + 长序列 FP22 观察

**Files:**
- Modify: `cutedsl_sage/test_sage_sm90.py`（追加长序列观察用例；固化标定后的阈值常量）
- Modify: `docs/superpowers/specs/2026-07-11-cutedsl-sage-sm90-design.md`（回写实测阈值）

**Interfaces:**
- Consumes: Task 4 的全部测试与阈值常量
- Produces: 固化的验收基线（阈值常量 + 全矩阵绿）

- [ ] **Step 1: 追加长序列观察用例**

在 `test_kernel_no_smooth_k` 之后追加：

```python
@pytest.mark.slow
@pytest.mark.skipif(not _sm90_available(), reason="需要 sm90 GPU")
@pytest.mark.parametrize("s", [16384, 32768])
def test_fp22_long_seq(s):
    """单级 FP32(FP22) 累加的长序列误差趋势观察：只按二级（松）阈值断言，
    一级指标打印出来供人工评估是否需要两级累加。
    n=2 控制 ref_quant_sim 显存（s=32768 时 S 矩阵 2×32768²×4B ≈ 8.6GB/份）。"""
    q, k, v = _mk_qkv(1, 2, 2, s, s, 128, torch.float16)
    sm_scale = 128 ** -0.5
    o = _sage_call()(q, k, v, is_causal=False, sm_scale=sm_scale).float()
    km = k.mean(dim=1, keepdim=True)
    q_i8, q_sc = quant_q_int8_per_warp(q)
    k_i8, k_sc = quant_k_int8_per_block(k, km)
    v_f8, v_sc = quant_v_fp8_per_channel(v)
    o_sim = ref_quant_sim(q_i8, q_sc, k_i8, k_sc, v_f8, v_sc, False, sm_scale).float()
    o_ref = ref_sdpa(q, k, v, False, sm_scale)
    print(f"\n[s={s}] L1: 1-cos={calc_diff(o, o_sim):.2e} "
          f"maxrel={((o - o_sim).abs().max() / o_sim.abs().max()).item():.2e} | "
          f"L2: 1-cos={calc_diff(o, o_ref):.2e} rel_l1={rel_l1(o, o_ref):.2e}")
    assert calc_diff(o, o_ref) < E2E_DIFF_MAX
```

- [ ] **Step 2: 跑全量验收（含 slow）**

```bash
./cutedsl_sage/hyper01.sh test
```
Expected: 全部 PASS（含 slow 组合与长序列观察）。记录每类用例的一级/二级指标输出。

- [ ] **Step 3: 阈值标定固化**

按 Step 2 的实测输出，把 `SIM_DIFF_MAX`/`SIM_MAXREL_MAX`/`E2E_DIFF_MAX`/`E2E_L1_MAX` 改为各指标 `max_observed × 3`（取整到一位有效数字），并在常量注释里记录标定日期与 commit。把长序列（16K/32K）的一级指标实测值追加到 spec 的「风险与取舍」一节（FP22 是否需要两级累加的决策依据）。

- [ ] **Step 4: 复跑确认阈值不误报**

```bash
./cutedsl_sage/hyper01.sh test
```
Expected: 全部 PASS。

- [ ] **Step 5: 最终 Commit**

```bash
git add cutedsl_sage/test_sage_sm90.py docs/superpowers/specs/2026-07-11-cutedsl-sage-sm90-design.md
git commit -m "$(cat <<'EOF'
Calibrate correctness thresholds on H200; add long-seq FP22 observation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## 起草阶段已核实的关键公式（执行时勿再推导，直接对照）

- **per-warp q_scale**（CUDA 基准 sm90.cu:177）：`idx = ((b*n_q + h_q)*ceil(s_q/64) + bx)*4 + warp_local`；GQA 下按 n_q（Q 头）索引
- **per-block k_scale**（sm90.cu:188）：`idx = (b*n_kv + h_q//gqa_ratio)*ceil(s_k/128) + kv_iter`
- **补偿链闭合**：`O = diag(1/row_sum) · [exp2((S·qs·ks − m)·sm_scale·log2e) · 448 → e4m3 → WGMMA(P, V_fp8)] · v_scale/448`，其中 `v_scale = amax_v/448`；row_sum 累加量化前 f32 P
- **WGMMA m64nN acc fragment**：warp w 覆盖行 `16w..16w+15`；行 = `16*warp_local + lane/4 (+8)`，列 = `16*i + 2*(lane%4) + 8*(j/4) + (j%2)`——per-warp 标量天然行对齐
- **CUDA 版差异点（勿照抄）**：CUDA 版 row_sum 初始化为 1.0（分母 +1 平滑）、P 转 e4m3 用 exp2 偏移 2^8.807 且不乘 448、用 fp32+fp32 两级累加——本实现契约均不同，以本计划的契约为准
