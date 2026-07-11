# CuteDSL SageAttention (Hopper) 设计文档

日期：2026-07-11
状态：已批准（用户确认：端到端 + torch 量化；单级 fp32 累加；head_dim 64/128；SDPA + 量化模拟双参考；kernel 结构方案 A）

## 目标

在 `cutedsl_sage/` 下用 CuTe-DSL (Python) 实现 Hopper (sm90) 的 SageAttention forward kernel：

- QK^T 用 INT8 WGMMA（s8s8s32），PV 用 FP8 E4M3 WGMMA（**单级 FP32 accumulator**，即硬件 FP22 精度；对应现有 CUDA 版 `pv_accum_dtype="fp32"` 那个未实现的变体）
- 输入 layout 固定 NHD `[b, s, n, d]`，`d ∈ {64, 128}`（其余 assert 拒绝）
- 量化粒度 per-warp（Q：BLKQ=64 / WARPQ=16；K：per-block BLKK=128，与现有 sm90 CUDA 路径一致）
- 支持 causal 与 non-causal；支持 GQA（`n_q % n_kv == 0`）；首版不支持 `return_lse`
- 正确性在 hyper01（H200，容器 `sglang-diffusion-qwenimage`，venv `/data/.torch`）上验证

非目标（本期）：性能调优、两级 fp32+fp32 累加、triton/融合量化 kernel、varlen、sm100。

## 背景：现有 CUDA sm90 实现要点

（分析对象 `csrc/qattn/qk_int_sv_f8_cuda_sm90.cu`、`sageattention/core.py:829`、`sageattention/quant.py`）

- 单 warpgroup（128 线程）全 consumer、TMA 单缓冲 ping-pong，FA2 风格；CTA_Q=64 / CTA_K=128；grid = `(ceil(s_q/64), n_q, b)`
- QK：`wgmma m64n128k32.s32.s8.s8`（SS）；PV：`wgmma m64nNk32.f32.e4m3.e4m3`（A=P 在寄存器）；Python 层只允许 `fp32+fp32` 两级累加
- per-warp scale：`q_scale [b, n_q, ceil(s_q/64)*4]`（每 warp 16 行），`k_scale [b, n_kv, ceil(s_k/128)]`；`v_scale [b, n_kv, d]` per-channel，scale_max=448
- V 要求转置布局（NHD 下 `[b, d, n_kv, s_pad]`）、seq 零补到 128 倍数、且 16-token 组内 permute `[0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15]`——该 permute 仅为补偿手写 `RS_32_to_8` 寄存器打包顺序，**CuteDSL 版不需要**（标准 smem swizzle + fmha.py 的 fragment 重排替代）
- smooth_k（K 减 seq 均值）融合在量化 kernel；softmax 恒等性保证输出无需修正
- sm_scale 不融进 Q 量化，在 attention kernel 内乘（exp2 域）

## 组件与文件

```
cutedsl_sage/
├── cutedsl_sage.py    # 基类骨架（已有）：补齐 from_args 的 cache key 提取与 fake tensor 构造
├── core.py            # 新增：CuteDSL kernel + torch 量化 + 端到端 API
└── test_sage_sm90.py  # 新增：正确性测试脚本（远程 H200 上运行）
```

### core.py 三层结构

1. **`SageAttnSm90(CutedslKernel)`**
   - `@cute.jit __call__`：host 侧——from_dlpack 张量 layout 处理、TiledMma / TMA atom / smem layout / `@cute.struct SharedStorage` 构造、grid 计算、`kernel(...).launch(...)`
   - `@cute.kernel kernel`：device 侧主体（见 Kernel 设计）
2. **torch 量化函数**（正确性优先，后续可换 triton/融合 kernel）
   - `quant_q_int8_per_warp(q, BLKQ=64, WARPQ=16)`：seq 维按 64 分块、块内 4 个 16 行段，amax over `(16, d)`；`q_scale [b, n_q, ceil(s/64)*4]` fp32；`q_int8 = round(q · 127/amax)`
   - `quant_k_int8_per_block(k, km, BLKK=128)`：`k − km` 后 seq 维 128 分块 amax over `(128, d)`；`k_scale [b, n_kv, ceil(s/128)]`
   - `quant_v_fp8_per_channel(v)`：seq 零补到 128 倍数 → 转置 materialize 为 `[b, n_kv, d, s_pad]` contiguous → per-channel amax over seq → `v_scale = amax/448`，`v_fp8 = (v · 448/amax).to(float8_e4m3fn)`。**无 16-token permute**
3. **端到端入口** `sageattn_qk_int8_pv_fp8_hopper(q, k, v, is_causal=False, sm_scale=None, smooth_k=True)`
   - 输入 fp16/bf16 NHD `[b, s, n, d]`；断言 `d ∈ {64,128}`、最后一维 contiguous、causal 时 `s_q == s_k`、`n_q % n_kv == 0`
   - `sm_scale` 默认 `d**-0.5`；`smooth_k=True` 时 `km = k.mean(dim=1, keepdim=True)`
   - 输出 `o` 同输入 dtype/shape

## Kernel 设计（方案 A：cutlass `examples/python/CuTeDSL/hopper/fmha.py` 为底本）

参考底本：`flash-attention/csrc/cutlass/examples/python/CuTeDSL/hopper/fmha.py`（原生 fp8 支持）；host 集成参照 `flash_attn/cute/interface.py` 的 TVM-FFI + compile_cache 模式。

### 线程组织与 pipeline

- 256 线程 = 2 warpgroup：WG math（consumer，`warpgroup_reg_alloc(240)`）+ WG load（producer，`warpgroup_reg_dealloc(24)`）
- `PipelineTmaAsync` 管 K/V stage：d=128 时 kv_stage=3 起步（smem ≈ Q 8KB + 3×(16KB K + 16KB V) = 104KB ≤ 228KB），d=64 可加深
- tile = (CTA_Q=64, CTA_K=128)；grid = `(ceil(s_q/64), b, n_q)`，对齐骨架 `s_idx, b_idx, head_block_idx = cute.arch.block_idx()` 约定；GQA：`kv_head = head // (n_q // n_kv)`

### 数据通路

- TMA G2S：Q `[64, d]`、K `[128, d]`（K-major，d=128 → SW128 / d=64 → SW64）、V `[d, 128]`（s 连续，SW128）；smem layout 用 `cutlass.utils.hopper_helpers.make_smem_layout_a/b`
- QK^T：`make_trivial_tiled_mma(Int8, Int8, K-major, K-major, Int32, (1,1,1), (64, 128))`（`MmaI8Op`，SS），d 维 reduction 拆 `d/32` 次 k32；`|S_i32| ≤ 127²·128 ≈ 2.1e6 < 2²⁴`，int32→f32 无损
- **dequant（核心改造点）**：fmha.py 的 scale 是编译期常量，这里改为运行时——`S_f32 = f32(S_i32) · q_scale[q_scale_base + warp_idx] · k_scale[j]`，其中 `q_scale_base = ((b_idx·n_q + head)·s_blocks + bx)·4` 按张量实际 layout 计算；q_scale/k_scale 经 gmem 标量读（每 warp / 每迭代各一次，可放 smem 或直接 ldg）
- mask：dequant 后在 S_f32 上置 `-inf`。复用 fmha.py 的 identity-tensor 坐标 + 三段 trip 结构：
  - non-causal：仅 kv 尾块 residual mask（`k_idx ≥ kv_len`）
  - causal：KV 循环上界 `ceil((bx+1)·64/128)`；因 CTA_Q=64 < CTA_K=128，对角 mask 只落在最后 1 个 KV 块
- online softmax（exp2 域，fmha.py `softmax_step` 改造）：`m = rowmax(S_f32)`（quad reduction）；`P = exp2((S − m)·sm_scale·log2e)`；rescale 旧 acc 乘 `exp2((m_prev − m)·sm_scale·log2e)`；`row_sum` 累进
- P → e4m3：P∈[0,1] 乘 448 转 e4m3，复用 fmha.py `make_acc_into_op` 的 8-bit A-fragment 重排（`shuffle_sync` + `prmt`）
- PV：`make_trivial_tiled_mma(Float8E4M3FN, Float8E4M3FN, K-major, K-major, Float32, (1,1,1), (64, d), OperandSource.RMEM)`（`MmaF8Op`，RS），seq 维 reduction 拆 4 次 k32；**单级 F32 累加**（FP22）
- epilogue：`O = acc · v_scale[列]/448 / row_sum` → cast fp16/bf16 → smem → TMA store；qo 尾块行谓词。`v_scale` per-channel 对应 acc 的 N 列，gmem 读 d 个标量
- 边界：Q/K TMA OOB 零填；V 物理 pad 行经 mask 后 P=0 无贡献；kv 尾块统一由 residual mask 压零

## Host 集成（对接 cutedsl_sage.py 骨架）

- `from_args` cache key = `(head_dim, is_causal, out_dtype, gqa_ratio)`（静态）；`b/s_q/s_k/n` 与 stride 用 fake tensor + `mark_layout_dynamic` 符号化，同 key 不同 seqlen 复用编译产物
- fp8 torch tensor 走 `view(torch.uint8)` + `element_type = Float8E4M3FN`（flash_attn interface.py 同款 workaround）
- `cute.compile(..., options="--enable-tvm-ffi")` + `make_fake_stream(use_tvm_ffi_env_stream=True)`：编译产物直接接受 torch.Tensor、自动使用调用方当前 torch stream
- `run(q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale, o, sm_scale, ...)`；`kv_len` 等动态标量随张量 shape 传入

## 测试计划（hyper01 / H200）

1. **环境**：`ssh hyper01` → `docker exec sglang-diffusion-qwenimage` → `source /data/.torch/bin/activate`；`pip install nvidia-cutlass-dsl apache-tvm-ffi`（容器现缺这两个包；torch 2.11.0+cu128、triton 3.6.0 已有）；代码 rsync 到 hyper01 的容器挂载路径（`/data` 下，具体映射实施时确认）
2. **双参考验证**（`test_sage_sm90.py`）：
   - 参考①：torch SDPA（fp32 计算）——端到端 cossim / 相对 L1，衡量量化损失在 SageAttention 论文水平
   - 参考②：torch 量化模拟——完全相同的 Q int8 per-warp、K int8 per-block（含 smooth_k）、V e4m3 per-channel、P×448→e4m3 量化，在 fp32 里精确模拟 attention，与 kernel 输出逐元素比。隔离 kernel bug 与量化损失；剩余容差仅来自 FP22 累加与 exp2 舍入，阈值实测标定（预期 cossim ≳ 0.9999）
3. **形状矩阵**：b∈{1,2} × n∈{8,24} × s∈{1024, 4096, 337} × d∈{64,128} × causal∈{False,True} × dtype∈{fp16,bf16}，外加一组 GQA（n_q=2·n_kv）
4. 长序列（16K+）单独观察 FP22 累加误差趋势
5. 验收：全部组合两级参考过阈值

## 风险与取舍

- `nvidia-cutlass-dsl` 需 ≥4.2（`--enable-tvm-ffi`）；DSL API 迭代快，以实际安装版本自带 examples 校对 fmha.py 用法
- 单级 FP32 累加（FP22）超长序列精度衰减——已知取舍，两级累加留作后续迭代
- torch 量化实现的端到端性能不是本期目标；kernel 本体性能优化（2 math WG ping-pong 等）留作后续
- 本地开发机为 sm86，无法本地运行，编译/数值问题只能远程迭代
