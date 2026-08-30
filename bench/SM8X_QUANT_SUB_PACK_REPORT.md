# sm8x quant_k 寄存器修复：sub_mean 减法 pack 化

2026-08-30。修复 `QuantPerThreadKInt8Kernel<hd,64,true,T>`（dense/varlen 的 smooth_k quant_k，sm80/86/89/120 全部走 warp_k=64 档）在 sm8x 上 110–118 registers 的寄存器膨胀。commit `2b0f5f0`，只改 `csrc/fused/quant_per_thread.cu`。

## 结论

膨胀根因不是 kQuantCache 注释怀疑的全展开或 cache 数组本身，而是 sub_mean 的逐元素 half 减法：sm8x ptxas 把每个差值 splat 进独立寄存器再进 cache，cache 行从 packed 2 registers 变成 4 registers。改成 per-pair `__hsub2`（每 lane 与标量减法 bit 相同）后寄存器回到注释预测的 ~72：

| 实例（registers） | 修复前 | 修复后 | CTA/SM |
|---|---|---|---|
| sm86/89 hd128 sub | 110 | 72 | 4 → 7 |
| sm86/89 hd64 sub | 72 | 56 | 7 → 9 |
| sm80 hd128 sub | 118 | 80 | 4 → 6 |
| sm80 hd64 sub | 72 | 56 | — |

sm90/100/120 用 `__CUDA_ARCH__ < 900` 门控排除，SASS 逐字节不变（归一化 anon-namespace 哈希后 diff，三 arch 全部 identical）。排除原因：sm90/100 这些实例在 32-register pin 下另有 codegen；sm120 fp16 本来就 packed（54 regs），强行 pack 反而把它的 bf16 档从 80 推到 96。

实机（sm86）结果分两半：

- **packed 修复性能持平（0.99–1.01×）**：这个 kernel 在 3080 Ti Laptop 上 DRAM 带宽饱和，occupancy 33% → 58% 不改变吞吐。修复的实际收益是 SASS 少 48 条指令（HADD2 192 → 160）加寄存器余量，不是本机提速。
- **register cache 本身仍值 ~1.6×**：去 cache 的 reload 形态（47 regs，满 occupancy）反而掉到 0.61–0.64×，与 kQuantCacheBytes 注释当年在同款卡上的测量一致。cache 该保留，occupancy 不是这个 kernel 的约束。

## 测试条件

| 字段 | 值 |
|---|---|
| 机器 | RTX 3080 Ti Laptop（sm_86），本机 |
| GPU 独占 | 是；bench 前 `nvidia-smi` 确认 15 MiB / 4% util，无其他进程 |
| 形状 | (b,h,s) ∈ {(8,40,1024), (1,56,4096), (8,24,8192), (1,40,16384), (2,24,32768), (1,40,65536)}，hd ∈ {128, 64} |
| dtype | fp16（bf16 只做了静态寄存器对比，未上机） |
| 调用 | `torch.ops.sageattention.quant_qk`，per_thread，blk 128/32/64/64（sm89 tiling，即目标实例） |
| 计时 | torch profiler per-kernel self time，warmup 5，iters 20/10/5 按 seq 分档 |
| 重复 | 7 轮，三方 round-robin 轮转（抗热漂移），取中位数；轮间最大 spread 1.7%，quant_q 同 kernel 对照组三方 1.00× |
| 源码 | 基线 = `773cfcb`（干净树），packed 修复 = `2b0f5f0`（干净树）；reload 探针 = 基线 + 强制 kCache=false + packed 减法（脏树，仅探索性对照） |
| 工具链 | nvcc /usr/local/cuda（与 ptxas 基线同套），torch 2.13.0+cu132，sm8.6-only 构建 |

## 数据

quant_k 每次调用 µs（7 轮中位数；倍率 = 基线/该组，>1 为该组快）：

| hd | 形状 | 基线（逐元素减法，110 regs） | packed 减法（72 regs） | 去 cache reload（47 regs） | packed 倍率 | reload 倍率 |
|---|---|---|---|---|---|---|
| 128 | b8h40s1024 | 288.2 | 286.6 | 462.6 | 1.01× | 0.62× |
| 128 | b1h56s4096 | 185.9 | 186.7 | 302.3 | 1.00× | 0.62× |
| 128 | b8h24s8192 | 1250.6 | 1260.3 | 2041.4 | 0.99× | 0.61× |
| 128 | b1h40s16384 | 523.2 | 526.9 | 851.5 | 0.99× | 0.61× |
| 128 | b2h24s32768 | 1249.5 | 1260.2 | 2042.6 | 0.99× | 0.61× |
| 128 | b1h40s65536 | 2081.6 | 2097.4 | 3403.7 | 0.99× | 0.61× |
| 64 | 六个形状 | 103.3–1045.7 | 103.0–1047.5 | 161.4–1694.0 | 1.00× | 0.62–0.64× |

## 归因

- 修复前 sm89 SASS：16 个 row load 全部在 amax pass 上半段发射（LDG 19 条 = 16 行 + mean + cu_seqlens），减法编成 `HADD2 Rd, Rx.H0_H0, -Ry.H0_H0` 的 splat 形态，每元素占一寄存器；load 目的寄存器与 splat 后的 cache 寄存器两套并存，64 + 32 registers 起步，对上 110。修复后 HADD2 192 → 160（64 条 splat 减并成 32 条 packed），load 寄存器就地成为 cache，对上 72。
- 全展开不是根因但不可动：cached 路径 `#pragma unroll 1` 探针下 cache 数组因动态下标退化成 128 B local memory（stack frame 128 B）。cache 存活的前提就是全展开。
- launch_bounds 方向（静态探索，未上机）：压在逐元素减法上全是重 spill（cap 64 时 132st/184ld）；压在 packed 上虽然干净（cap 64 → 63 regs / cap 56 → 56 regs），但会让 hd64 的 live 实例 `K<64,64,sub>` spill 16st/16ld。鉴于 sm86 实测 occupancy 不敏感，不 arm。

## 没做到的

- sm89 实机（4090/L40S）未测：本任务的原始疑问点，clab 无 sm89 档位。sm86 是最近的 proxy（寄存器数与 sm89 逐实例一致），但结论外推见限制条款。
- sm80 实机未测（118 → 80 只有静态数字）。
- bf16 未上机（静态：sm86/89 与 fp16 同数字）。
- 端到端 sageattn 未测，只测了 per-kernel；本机 quant_k 持平，端到端也应持平。

## 限制条款

- 「packed 修复性能持平」是 3080 Ti Laptop 的结论。sm89 desktop（4090 SM 多、带宽/SM 比例不同）occupancy 敏感度可能更高，届时 packed 的收益下界是 0、上界为正——修复方向不因此反转，SASS 严格变少且寄存器严格变少。
- 「去 cache 掉 1.6×」在 sm89 的 72 MB L2 上劣化幅度可能收窄（reload 第二遍更可能 L2 命中），但要翻正需要 L2 完全吸收双倍读且 occupancy 收益兑现，方向未验证。若在 sm89 上重测 cache 取舍，用本报告的 reload 探针（treeC 配方）。

## Gate

- `python tools/compare_reference.py --check`：packed 修复与 reload 探针两个构建均 ok=1493 diff=0（extra=48 为 varlen equiv 预期）。
- pytest：552 passed / 154 skipped（与 `773cfcb` 口径一致）。
- sm90/100/120 SASS 逐字节不变。

## 可复现命令

```
# 寄存器矩阵（单 TU，全 arch）
bash <scratch>/qreg/nvcc_tu.sh csrc/fused/quant_per_thread.cu 89 out/
# bench（一进程一方，PYTHONPATH 选树；7 轮 round-robin）
PYTHONPATH=<tree> python <scratch>/qbench/kbench.py --out bench.jsonl --side <名字> --round <n>
```

## 产物路径

本机 scratchpad `/tmp/claude-1000/-home-ubuntu-workspace-github-llm-SageAttention--claude-worktrees-gracious-mayer-c96121/a24db3c6-111a-488c-8d1b-b48b941d013a/scratchpad/`：

- `qreg/base/`、`qreg/fix/`：修复前后六 arch 的 ptxas/SASS/cubin；`qreg/v_*`：全部变体（packed / nocache / rawcache / bounds8 / bounds9 / …）
- `qbench/bench.jsonl`：504 条原始记录；`qbench/kbench.py`：bench 脚本；`qbench/treeB`、`treeC`：对照构建树
