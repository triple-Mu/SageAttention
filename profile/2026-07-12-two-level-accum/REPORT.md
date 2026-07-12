# NCU Profile：两级 PV 累加（inst_buf 式）改动归因（H200 / sm90）

**TL;DR**（主形状 b=4, h=32, s=4096, d=128, non-causal）：两级累加相对单级基线（c554098）
**指令净变化 ≈ 0（FMUL −67.1M 与 FFMA +67.1M 精确对消，总数仅 +0.46%）、duration −0.52%
（1.9991 vs 2.0095 ms）、occupancy 与 tensor pipe 完全不变（2 CTA/SM、19.1%）**；代价是每
math 线程一次性 ~16B 的 local 存取（139,264 LDL + 131,072 STL，折算每 kv 迭代 0.5B，非逐迭代
spill）。DSL 的 FFMA 数（138.4M）现与 CUDA 完全一致；与 CUDA 的剩余差距 1.186× 全部由剩余
标量指令面积（760.4M vs 613.6M = 1.239×）构成，发射率反而已反超（0.541 vs 0.519/cyc）。

## 1. Setup（可复现）

- 机器/容器/GPU/ncu 与上轮 `profile/2026-07-12-cutedsl-vs-cuda-sm90/` 完全一致（H200 132 SM，
  GPU 2，ncu 2025.1.1）；harness 同款复用（`harness/profile_kernels.py`，预量化输入、
  warmup 3 + 5 次调用、`-s 3 -c 1` 抓稳态第 4 次 launch）。
- 三个被试：
  - `full_dsl2_*` / `source_dsl2_d128`：**两级累加**（f07d5a3，regs 224 / 2 CTA/SM / kv_stage 4）
  - `base_full_dsl_*` / `base_source_dsl_d128`：**单级基线**（c554098，同配置，仅少两级结构）——
    本轮临时 checkout 该版本 core.py 现场采集，用于把两级改动从 P1/P2a 中隔离出来
  - `prev_full_cuda_d128` 等 `prev_*`：上轮报告的 CUDA/旧 DSL 报告**逐字节副本**（md5 一致，
    git 去重零成本），保证本 run 目录自包含
- 采集命令（full 两轮 + source 一轮，与上轮相同）：
  ```
  ncu --set full --section PmSampling --section PmSampling_WarpStates --devices 0 \
      -k regex:SageAttnSm90 -s 3 -c 1 -o reports/<tag> python harness/profile_kernels.py
  ncu --set source --section SourceCounters --devices 0 -k regex:SageAttnSm90 -s 3 -c 1 ...
  ```

## 2. Headline：两级 vs 单级基线 vs CUDA（d128 主形状）

| metric | 两级 dsl2 | 单级 base | CUDA | 两级 vs base |
|---|---:|---:|---:|---:|
| duration | 1.9991 ms | 2.0095 ms | 1.6850 ms | **−0.52%** |
| warp 指令总数 | 760.43M | 756.91M | 613.58M | **+0.46%** |
| 发射率 issue_active/cyc | 0.541 | 0.538 | 0.519 | +0.6% |
| eligible warps/smsp | 0.762 | 0.769 | 0.964 | ≈ |
| achieved occupancy | 15.26% | 15.26% | 18.49% | 不变 |
| CTA/SM（regs/smem limit） | 2 / 2 | 2 / 2 | 3 / 3 | 不变 |
| launch regs/thread；smem/CTA | 128；82.9KB | 128；82.9KB | 167；40KB | 不变 |
| tensor pipe active (of active) | 19.11% | 19.07% | 22.68% | 不变 |
| GMMA 指令 | IGMMA 4.19M + QGMMA 4.19M | 同 | 同 | 不变 |
| local ld / st（条） | **139,264 / 131,072** | 0 / 0 | 0 / 0 | 新增（见 §4） |
| DRAM read 占峰值 | 2.05% | 2.04% | 2.43% | 均远离 DRAM bound |

交叉验证：vs CUDA duration 比 1.186 ≈ 指令比 1.239 × 发射率比 (0.519/0.541)=0.959（乘积
1.189）——上轮的分解框架依旧闭合；改动后短板只剩指令面积一项，发射效率已优于 CUDA。

## 3. 指令数分解（source set，SASS opcode 全 kernel 聚合）

基准单位 67.1M = 每元素每迭代 1 条（8192 CTA × 32 kv-iter × 4 warp × 64 元素/线程）。

| opcode | 单级 base | 两级 dsl2 | Δ | 归因 |
|---|---:|---:|---:|---|
| FMUL | 141.56M | 74.45M | **−67.11M** | softmax_step 内逐列 `acc_pv *= scale_pv` 循环消除 |
| FFMA | 71.30M | 138.41M | **+67.11M** | 块后 `acc_pv = acc_pv·o_scale + temp` 合并（rescale 融进 FFMA） |
| UIADD3 | 14.69M | 16.60M | +1.91M | 第二累加器/合并循环的地址与控制 |
| LDL+STL | 0 | 0.27M | +0.27M | 一次性溢出（§4） |
| UMOV/IMAD/VIADD 等杂项 | — | — | +1.2M | 同上控制开销 |
| **合计** | 764.03M | 767.52M | **+3.50M (+0.46%)** | FP 指令严格零和，只付 0.46% 控制开销 |
| （对照 CUDA） | | FFMA 138.4M / FMUL 8.4M | | **DSL 的 FFMA 数已与 CUDA 逐条对齐** |

不变项（两版完全相同）：FMNMX 71.3M、FADD 69.3M、MUFU.EX2 69.2M、I2FP 67.1M、SEL 53.4M、
F2FP 33.6M、PRMT 33.6M、SHFL.IDX 16.8M、IGMMA/QGMMA 各 4.19M——两级改动没有碰任何其他环节。

**剩余 vs CUDA 的 +146.8M 指令面积**（下一步优化的目标清单）：SEL 53.4M（行 max 残留的
NaN 语义选择）、PRMT 33.6M（fp8 pack 重排）、SHFL.IDX 16.8M（quad reduce）、
UMOV 20.7M + UIADD3 16.6M（DSL 地址生成），与上轮报告 §4 的诊断一致。

## 4. 两级累加的开销去向（spill 与 stall）

- **local 存取为一次性、与寄存器预算无关**：139,264 LDL + 131,072 STL（= 557,056 + 524,288
  sectors，~17MB/launch）。折算每 math 线程 ~16B（4 个 f32）存取一次、每 kv 迭代仅 0.5B——
  不是主循环逐迭代 spill。配置矩阵（§6）显示 224/232/240 regs 下 local 流量**同值**，
  说明是 DSL 代码生成的固定栈用量（pcsamp 中 STL 出现在 prologue 段 PC，inst_exec=每 warp
  1 次），而非边际寄存器压力。
- **stall 分布仅微移**（per issue-active ratio，base→两级）：long_scoreboard 0.901→0.843、
  mio_throttle 0.148→0.122、barrier 0.698→0.737、gmma 0.054→0.065。pcsamp 热点结构不变：
  DEPBAR（wait_group）15.81%→16.82%、prologue BRA long_scoreboard ~14.8%、MUFU.EX2 ~10.5%。
- **收益来源**：merge FFMA 挪到 PV wgmma 完成之后，softmax_step 在 QK-wait 与 PV-issue 之间的
  关键路径少了 64 次 FMUL（只写 2 个标量 o_scale），PV wgmma 提早发射；净效 duration
  d128 −0.52%、d64 −1.87%（d64 无 spill：local ld/st = 0，temp 仅 32×f32）。
- 小项：shared st bank conflict 194,624→264,010（+69K；总量 <0.1% 周期，可忽略；其中 194K
  在基线就存在，来自 P1/P2a 期的 epilogue 交错，非本改动引入）。

## 5. d64 副形状（full_dsl2_d64 vs base_full_dsl_d64）

duration 1.6775 vs 1.7094 ms（**−1.87%**）；指令 710.58M vs 707.60M（+0.42%）；发射率
0.591→0.604；achieved occupancy 15.04%（不变）；**local ld/st = 0**（无溢出）。

## 6. 配置矩阵（bench_compare --quick，TFLOPS；spill 由 ncu local 流量判定）

| 配置 | d64 s4096 | d64 s16384 | d128 s4096 | d128 s16384 | local（d128） |
|---|---:|---:|---:|---:|---|
| 单级基线 224/2CTA | 426.8 | 470.9 | 727.1 | 777.7 | 0 |
| **A 两级 224/2CTA（选定）** | 434.5 | 478.1 | 729.2 | 781.8 | 一次性 ~17MB |
| B 两级 240/1CTA | 310.5 | 341.8 | 500.5 | 545.1 | 同 A |
| C 两级 232/2CTA | 435.5 | 474.6 | 718.0 | 776.6 | 同 A |

B 失去 2 CTA/SM 掩盖（-31%）；C 相对 A 无收益（65536 regs 满配无余量，d128 反降 1.5%）；
**A 选定**——perf 不降反微升，且免去精度/性能取舍。

## 7. 结论

1. 两级累加在 DSL 上的成本被压到测量噪声之下（d128 −0.52%、d64 −1.87%，均为**负成本**），
   FP 指令严格零和、occupancy/tensor pipe/访存完全不变——收益（一级 1-cossim 长序列
   1.37e-5 → 4.09e-8，见 sdd 报告）纯赚。
2. 与 CUDA 的差距解释已简化为单变量：**指令面积 1.239×**（发射率已反超）。下一步（P2b 或
   标量指令继续瘦身：SEL/PRMT/SHFL.IDX/地址生成）按上轮报告 §6 优先级推进。

## 8. Artifacts

`harness/profile_kernels.py`（复用上轮）；`reports/`：full/source × {dsl2, base} d128 +
full × {dsl2, base} d64 + `prev_*`（上轮 CUDA/DSL 副本，md5 与原件一致）；`analysis/`：
`compare_full_dsl2_d128_vs_base_full_dsl_d128_vs_prev_full_cuda_d128.txt`、
`compare_full_dsl2_d64_vs_base_full_dsl_d64.txt`、`sass_opcode_{source_dsl2,base_source_dsl,prev_source_dsl}_d128.txt`、
全量 metric JSON。远端副本：hyper01 `/data02/triplemu/workspace/SageAttention/profile/2026-07-12-two-level-accum/`。
Caveat：ncu replay duration 与 do_bench 中值不同源但结论一致；无 lineinfo，SASS 级归因
（与上轮方法相同）；base 为 c554098 现场重采，与 dsl2 同机同 GPU 背靠背采集。
