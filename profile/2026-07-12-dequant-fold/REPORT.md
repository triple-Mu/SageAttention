# NCU Profile：dequant 折叠 + mask FSEL 消减（指令面积收官，H200 / sm90）

**TL;DR**（主形状 b=4, h=32, s=4096, non-causal）：把逐元素 ×dequant 的 64 次/块 FMUL
折进 exp2 的 FFMA 系数（rowmax 移到 raw 域）、行 max 串行链改树形归约、mask 尾块
if 置写改 `select_` 单条 FSEL 后，**d128 指令 692.4M → 597.1M（−13.8%），
首次低于 CUDA（0.973×）**；duration d128 1822.5 → 1812.8 μs（vs CUDA **1.076×**，
上轮 1.082×），d64 1589.4 → 1480.2 μs（−6.9%，vs CUDA 1.261×→**1.174×**）。
bench 口径 d64 +7.3%/+3.3%（0.785→0.842 / 0.824→0.851），d128 s4096 0.93→0.96。
**指令面积维度就此收官**：剩余差距（d128 1.076×）全在发射率/占用
（issue 0.464 vs 0.519、eligible 0.649 vs 0.964），继续提速须走占用/发射结构
（3 CTA/SM、第二 math WG），不再是指令问题。§6 给最终清单与结论。

## 1. Setup（可复现）

- 机器/容器/ncu 与前几轮一致（hyper01 H200 132 SM，容器 `sglang-diffusion-qwenimage`，
  ncu 2025.1.1）；harness 逐字节复用 `harness/profile_kernels.py`（预量化输入、
  warmup 3 + 5 次调用、`-s 3 -c 1` 抓稳态第 4 次 launch）。全程 GPU7（空闲，0 MiB）。
- 被试（同一改动的三个递进阶段，正好构成消融）：
  - `full_dsl4_*` / `source_dsl4_d128`：仅 dequant 折叠（中间态，d128 曾回退，见 §3）
  - `full_dsl4t_*` / `source_dsl4t_d128`：+ rowmax 树形归约
  - `full_dsl4sel_*` / `source_dsl4sel_d128`：+ mask `select_`（**最终合入版**）
  - `prev_full_dsl3b_*` / `prev_source_dsl3b_d128`：offline-permute 版基线
    （HEAD d39f158，上轮报告逐字节副本）
  - `prev_full_cuda_*` / `prev_source_cuda_d128`：CUDA 基线（首轮报告逐字节副本）
- 采集命令与历轮相同（`--set full --section PmSampling*` / `--set source --section
  SourceCounters`）；分析脚本 `analysis/` 四件套复用。

## 2. Headline（full set）

d128 主形状：

| metric | **dsl4sel（最终）** | dsl4t | dsl4 | dsl3b（基线） | CUDA |
|---|---:|---:|---:|---:|---:|
| duration | **1812.8 μs** | 1849.3 | 1884.0 | 1822.5 | 1685.0 |
| warp 指令总数 | **597.11M** | 628.44M | 628.43M | 692.38M | 613.58M |
| vs CUDA duration / 指令比 | **1.076× / 0.973×** | 1.098/1.024 | 1.118/1.024 | 1.082/1.128 | — |
| 发射率 issue_active/cyc | 0.464 | 0.479 | 0.467 | 0.538 | 0.519 |
| eligible warps/smsp | 0.649 | 0.679 | 0.658 | 0.750 | 0.964 |
| launch regs/thread；CTA/SM | 128；2 | 同 | 同 | 同 | 167；3 |
| local ld/st（条） | 139,264/131,072 | 同 | 同 | 同 | 0 |

d64 副形状：duration **1480.2**（dsl3b 1589.4，−6.9%；CUDA 1260.8，vs **1.174×**）；
指令 **548.16M**（642.53M → −94.4M；CUDA 564.04M，**0.972×**）；发射率 0.522 vs
0.570 vs 0.637。

交叉验证（分解闭合）：d128 dur 比 1.076 ≈ 指令比 0.973 × 发射率比 (0.519/0.464)=1.119
→ 1.089（残差 1.2%）；d64 1.174 ≈ 0.972 × (0.637/0.522)=1.220 → 1.186（残差 1.0%）。
**指令面积首次低于 CUDA，剩余差距完全由发射率解释。**

## 3. 消融：为什么需要三步而不是一步（d128 的回退与修复）

仅做 dequant 折叠（dsl4）时 d128 **回退**：指令 −64.0M 但 duration +3.4%
（bench 同会话 A/B 交错 ×2 复现：797/838 → 773/818 TFLOPS）。stall 归因
（source set，pcsamp by opcode，总样本 156,881 → 164,062）：

| stall@opcode | dsl3b | dsl4 | dsl4t | **dsl4sel** |
|---|---:|---:|---:|---:|
| FMNMX | 7,970 | **15,464** | 11,374 | 10,630 |
| WARPGROUP.DEPBAR | 25,620 | 30,338 | 27,578 | 30,873 |
| MUFU.EX2 | 22,075 | 24,228 | 25,273 | 26,908 |
| FMUL | 6,033 | 1,189 | 1,402 | ~1.2K |
| 总样本 | 156,881 | 164,062 | 160,039 | **157,009** |

被删的 64 条独立 FMUL 一直在为行 max 串行链（31 级 FMNMX 依赖）作发射填充；
删掉后链裸暴露（FMNMX stall ×1.94）。修复 = 树形归约（fmax 精确可结合 → 逐位一致、
零指令增量，关键路径 31→5 级），FMNMX stall 回落，bench 回到持平；再叠加 mask
select_（§4）后 duration 净赢。教训入档：**在低占用 kernel 里删"廉价填充指令"
必须同时缩短它掩护的依赖链**。

## 4. mask 尾块 FSEL：`if 置写` → `select_`（998 条 → 68 条）

`dump_fsel2.py` 对账（source set，exec=32,768 的 mask 尾块 PC 数）：
dsl4t 仍为 **998 条**（DSL 把逐元素 `if k_pos >= seqlen_k: acc_s[i]=-inf` 展开成
ISETP + 链式 FSEL，含大量冗余）；改成 `acc_s[i] = select_(keep, acc_s[i], -inf)`
后 **68 条**（≈ 每元素 1 条 + 常量装载），FSEL 总量 34.73M → 4.26M（−30.5M，
残余 = 2.2M mask + 2.1M local_max 守卫）。d128 duration −2.0%（1849.3→1812.8）；
d64 bench +5.6%（464.6→490.5 TFLOPS，s4096）——d64 指令占比更高，收益 1:1 兑现。
causal 语义并入同一 keep 谓词（`keep & (k_pos <= q_pos)`），146 测试全绿。

## 5. 指令数分解 vs CUDA（source set，d128；总量 604.10M vs 613.58M = 0.985×¹）

| 功能 | dsl4sel | CUDA | 差 | 备注 |
|---|---:|---:|---:|---|
| dequant I2FP | 67.1M | 67.1M | 0 | |
| dequant ×scale FMUL | **0** | 0 | **0** | 本轮消除（原 67.1M）；折进 exp2 FFMA 系数 c′=c·dequant |
| FMUL（rescale+m_cand+c′） | 10.5M | 8.4M | +2.1M | m_cand 2.1M + c′ 1.0M 为折叠新增（每行/每块 1 次） |
| FFMA（exp2 参数+两级 merge） | 138.4M | 138.4M | 0 | 系数 c→c′ 零增量 |
| 行 max FMNMX | 71.3M | 73.4M | −2.1M | 树形归约同量、只改依赖深度 |
| exp2 MUFU.EX2 | 69.2M | 69.2M | 0 | |
| 行 sum FADD | 69.3M | 67.2M | +2.1M | |
| fp8 pack F2FP | 33.6M | 33.6M | 0 | |
| fp8 pack PRMT | 16.8M | 0 | **+16.8M** | 剩余最大单项；需 F2FP.MERGE_C 目标寄存器复用（DSL codegen） |
| mask FSEL | 4.3M | 2.1M | +2.2M | 本轮 −30.5M（998→68 条 PC） |
| quad reduce SHFL.BFLY | 4.3M | 4.3M | 0 | |
| 地址/uniform/控制杂项 | ~110.9M | ~141.5M | **−30.6M** | DSL 反而更省（CUDA 的 ULOP3/USHF/PLOP3 更多） |
| **合计** | **604.1M** | **613.6M** | **−9.5M（0.985×）** | ¹full 口径 597.1M（0.973×） |

## 6. 收官结论：指令面积已闭卷，剩余差距清单

1. **指令面积 vs CUDA：d128 0.973×、d64 0.972×——已低于 CUDA，此维度收官。**
   名义上还能抠的只剩 pack PRMT 16.8M（需 DSL 生成 F2FP.MERGE_C 高半字直写，
   codegen 层面，非 kernel 代码可控）与 FADD/FMUL/FSEL 合计 ~6M 零头；
   即便全部归零也只 −3.8%，且已被地址/uniform 项的 −30.6M 优势覆盖。
2. **剩余 duration 差距（d128 1.076×、d64 1.174×）100% 在发射率/占用**：
   issue 0.464 vs 0.519（d128）、eligible 0.649 vs 0.964、occupancy 15.3% vs 18.5%
   （2 vs 3 CTA/SM，128 vs 167 regs）。残余 stall 大项全是延迟性：
   DEPBAR 30.9K（wgmma 等待，结构性）、MUFU.EX2 26.9K（CUDA 同款 31.7K，EX2 吞吐上限）、
   BRA/long_scoreboard 25.3K（TMA 等待）。
3. **「还能不能再快」：能，但不在指令面积**。下一档是占用/发射结构：
   3 CTA/SM（压 smem/kv_stage 或寄存器再分配）或第二 math WG（P2b 路线，
   此前 ping-pong 实验的教训是须避开共享 KV pipeline 同相共振）。d64 空间更大
   （eligible 0.735 vs CUDA 1.477，CUDA 4 CTA/SM）。两级累加的 acc_pv_temp
   寄存器结构性翻倍仍是 3 CTA/SM 的主要阻碍。
4. bench 口径最终（GPU7，同会话 A/B ×2 均值）：d64 s4096/16384 **0.842/0.851**
   （+7.3%/+3.3%），d128 s4096 **0.960**（+2.8%），d128 s16384 0.932（−1.1%，
   会话噪声带 829–846 内；ncu 锁频口径 −0.5% 为准）。全矩阵 24 点：
   d128 non-causal 0.90–1.01、causal 0.93–1.06；d64 non-causal 0.84–0.85、
   causal 0.87–0.89。
5. 数值：改动前后 L1 1-cos max 2.99e-7 → 3.00e-7（阈 1e-5），maxrel max
   5.65e-3 → 8.13e-3（阈 2e-2，余量 2.5×；max 统计量对 P448 ±1ulp → e4m3
   舍入翻转敏感，聚合指标不动 → 纯舍入无结构性 pattern）；125 例中 71 例逐位不变；
   FP22 长序列 4.1e-8（s=16K/32K）与改动前同量级。树形归约与 select_ 对
   fold-only 版**逐位零差**（实测 L1 全量 diff 为空）。

## 7. Artifacts

本目录：`harness/profile_kernels.py`；`reports/`（dsl4/dsl4t/dsl4sel 的 full
d128+d64 与 source d128²，prev dsl3b/CUDA 逐字节副本 ×4）；`analysis/`
（并排对比 ×4、SASS opcode 聚合 ×4、全量 metric JSON、脚本四件套）。
²dsl4t_d64 full 为消融中间态一并存档。
远端副本：hyper01 `/data02/triplemu/workspace/SageAttention/profile/2026-07-12-dequant-fold/`
（容器内 `/data/workspace/...`）。实现细节与验证记录：
`.superpowers/sdd/dequant-fold-report.md`。
