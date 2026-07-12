# NCU Profile：exp2 FMA emulation（判负回退，H200 / sm90）

**TL;DR**（主形状 b=4, h=32, s=4096, non-causal）：把 softmax_step 逐元素 P448
`cute.math.exp2`（MUFU.EX2）换成上游 flash-attention 的 FMA emulation
（add.rm 位技巧 + 5 阶 Horner + shl/add.s32），XU 59.8%→1.2%、issue_active
0.608→0.784（SFU 排队确实压 issue），但 sm90 无 f32x2 packed，逐元素成本
2→11.4 条（实测 +9.4：FMA +8.0、ALU +1.8），指令 +108.6% » issue +29% →
d64 duration 1307.7→2114.5μs（+61.7%）、1/8 占比也 +6.0%。bench 扫参
（RES/8 ∈ {0,1/8,1/4,1/2,1}）单调劣化，无档位 ≥1% 收益 → **回退**。
详细叙事：`.superpowers/sdd/exp2-emul-report.md`。

## Setup

- 机器/容器/ncu/harness 与 2026-07-13-occupancy 轮一致（hyper01 H200 132 SM，
  GPU7，ncu `--set full --section PmSampling --section PmSampling_WarpStates`
  外加 `--metrics smsp__inst_executed_pipe_{xu,fma,alu}.sum`，`-k regex:SageAttnSm90
  -s 3 -c 1`）；harness 为 `harness/profile_kernels.py`（cutedsl_sage 同步副本改
  REPO_ROOT 一行）。
- 被试：基线 6ee29dd + `exp2-emulation.patch`（本目录），`SAGE_EX2_EMU_RES` 环境
  变量控制模拟占比（RES/8；0=纯 MUFU 基线，codegen 与 HEAD 一致）。
- 精度：5 阶 fpminimax（rel err ~1.6e-7）通过一级闸（136 non-slow 全绿含全模拟）；
  上游默认 3 阶（~8.8e-5）/4 阶（~3.0e-6）maxrel 2.3e-2/2.2e-2 超 2e-2 红线。

## Headline

d64（emu_d64_res0 / res1 / res8）：

| metric | res0 | res1 (1/8) | res8 (全) |
|---|---:|---:|---:|
| duration | 1307.7 μs | 1386.6 (+6.0%) | 2114.5 (+61.7%) |
| 指令总数 | 563.4M | 642.1M (+14.0%) | 1175.4M (+108.6%) |
| XU(MUFU) insts / util | 69.27M / 59.8% | 60.88M / 49.5% | 2.16M / 1.2% |
| FMA insts / util | 197.2M / 21.3% | 264.5M / 26.9% | 719.5M / 48.0% |
| ALU insts / util | 204.3M / 44.1% | 219.8M / 44.7% | 352.0M / 47.0% |
| issue_active | 0.608 | 0.653 | 0.784 |
| eligible warps/smsp | 1.284 | 1.479 | 1.832 |
| stalled_math_pipe_throttle | 0.421 | 0.407 | 0.282 |
| local ld/st（spill） | 0 / 0 | 0 / 0 | 2.29M / 1.31M |

d128（res0→res8）：duration 1664.6→2507.4 μs（+50.6%）；指令 614.0M→1219.0M；
XU 47.0%→0.97%；FMA 19.4%→42.5%；issue 0.521→0.686。

交叉验证：duration ∝ 指令/issue —— res1 1.140/1.074=1.061（实测 1.060）、
res8 2.086/1.290=1.617（实测 1.617）。XU 残余 2.16M = scale_pv 2.10M + rcp
（scale_pv 按纪律保留 MUFU）。基线 XU 69.27M ≈ 纯 MUFU.EX2（逐元素 67.11M +
scale_pv 2.10M）——sm90 的 I2F/F2FP 不在 XU pipe（在 ALU），I2F bias-trick 消融
（IADD 替代 sitofp）XU/duration 均不动，该候选路线一并排除。

## 结论

1. SFU 争抢假设证伪：基线 XU 60% 非饱和（CUDA 62.7% 更高且更快）；清空 XU 换来的
   issue 提升在 sm90 标量指令账下永远入不敷出（收支平衡需每元素净增 ≤~4 条，
   f32x2 packed 才能做到，sm100 专属——上游 +3~9% 不可移植）。
2. d64 残余 ~4%（issue 0.608 vs 0.637）与 SFU 无关；占用收官判定维持。
3. 实现回退，patch/报告/harness 存档本目录；核对文件：
   `analysis/compare_emu_d64_res0_vs_emu_d64_res1_vs_emu_d64_res8.txt`、
   `analysis/compare_emu_d128_res0_vs_emu_d128_res8.txt`、全量 JSON ×5、
   `reports/*.ncu-rep` ×5。
