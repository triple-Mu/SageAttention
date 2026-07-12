# FA3 式 GEMM-softmax 软件流水（softpipe）：判负回退，serial 基线维持

**TL;DR**：给 CuTe-DSL SageAttention sm90 kernel 实施三模式软件流水
（`serial`/`qk_hide`/`full`，见 §2 编排），两道闸全绿且三模式输出与 serial
**逐位一致**，但 bench 全线判负：默认占用下 qk_hide 0.62×/full 0.46×（d128，
spill 实锤：188M/336M local sectors）；放宽占用消掉 spill 后仍 0.91~0.93×。
**同占用对照证明重叠机制本身有效**（d128 2CTA 下 qk_hide vs serial
**+4.8~7.2%**，实验 A 翻正确认），但腾出 +64 regs 在飞 acc 的唯一途径是降占用
（d128 3→2 CTA -13%、d64 4→3 CTA -11%），代价恒大于收益。**根因是寄存器墙**：
serial 峰值 167/126 regs 已贴满 3/4 CTA 预算，kernel 无任何寄存器空隙容纳
「wgmma 横跨标量段在飞」。实现 revert，patch/harness/ncu 数据存档本目录。

## 1. 前任设计情报核实（三点全确认）

1. `wait_group(N)`（cute.nvgpu.warpgroup.helpers.py:114 →
   `nvvm.wgmma_wait_group_sync_aligned`）为 PTX FIFO 语义：等到 ≤N 组 pending，
   组按 commit 序退休，N 是编译期立即数；**不能选择性等特定组**；空 commit 组
   立即完成、不占 pending 计数（不能垫位）。
2. `PipelineConsumer/Producer/ImmutableResourceHandle` 实现
   `__extract_mlir_values__/__new_from_mlir_values__`（pipeline/sm90.py:989/1234/1461）
   → handle 可跨动态 while 携带/作 @cute.jit 参数（实测编译运行均可）。
3. KV ring：consumer cursor 严格 K₀,V₀,K₁,V₁… 推进；wait K_{t+1} 须先 acquire
   V_t（handle 持住晚 release）；kv_stage=3/5 下持 2 stage 仍留预取空间，无死锁。

## 2. 编排与在飞组计数表（FIFO 约束的解法）

FIFO 推论：想「等旧组、保新组在飞」只有一种编排——把要保留的组 commit 得更晚，
然后 `wait_group(1)`。据此两种流水（记号：pending 旧→新；W(N)=wait_group(N)）：

**qk_hide**（commit 序「PV_t 先、QK_{t+1} 后」→ merge_t ∥ QK_{t+1}）：

| 流水点 | pending | wait |
|---|---|---|
| 迭代顶 | [QK_t] | W(0) |
| convert→softmax_t→pack→PV_t commit | [PV_t] | — |
| QK_{t+1} commit 后 | [PV_t, QK_{t+1}] | W(1)：退 PV_t、**QK_{t+1} 在飞** |
| merge_t | [QK_{t+1}] | ∥ ✓ |

**full**（QK_{t+1} 提前发 + merge skew 一拍 → softmax_t ∥ {QK_{t+1}, PV_{t-1}}）：

| 流水点 | pending | wait |
|---|---|---|
| 迭代顶 | [QK_t, PV_{t-1}] | W(1)：退 QK_t、**PV_{t-1} 在飞** |
| convert；acquire V_t/K_{t+1}；QK_{t+1} commit | [PV_{t-1}, QK_{t+1}] | — |
| softmax_t | 同上 | ∥∥ ✓✓ |
| merge_{t-1} 前 | 同上 | W(1)：退 PV_{t-1}、**QK_{t+1} 在飞** |
| merge_{t-1}(o_scale_prev)→pack→PV_t commit | [QK_{t+1}, PV_t] | 跨迭代 |

非均匀点消解：① QK_0 在 prologue 就地 W(0)（一次性裸奔，使首迭代 W(1) no-op
语义正确）；② merge_{-1} 恒等化（acc_pv_temp.fill(0)+o_scale_prev.fill(1)，
acc_pv 全 +0 时 FFMA 逐位不变）→ 免动态 guard；③ 尾块以 masked=True 特化为
排水迭代（W(0)）。acc_qk 单缓冲即可（每迭代先 convert 腾空再发 QK_{t+1}），
无需 +64 regs 双缓冲——但在飞期寄存器 pinned 的代价照付（见 §4）。

## 3. 正确性（两道闸 + bit-identical）

- 量化闸不受影响；kernel 两道闸 136 non-slow × {qk_hide, full} 全绿。
- 设计保证（softmax/pack/merge 操作数与执行序不变，仅 wgmma commit/wait 编排
  不同）实测成立：9 形状（单块 trip_count=1/尾块/causal/GQA/s4096）×
  {qk_hide, full} 输出与 serial **逐位一致**（harness/smoke_pipe.py）。

## 4. bench 裁决（GPU7 H200，kernel-only，3 轮同卡交错中值，TFLOPS）

主表（non-causal；`模式:N` = SAGE_MINCTASM=N；serial 默认 = :3(d128)/:4(d64)；
d128 的 qk_hide/full 默认即 :3；同构对 serial vs serial:3 互差 ~2.5% 为会话噪声下界）：

| seq | d | CUDA | serial | qk_hide 默认 | full 默认 | qk_hide:2 | full:2 | serial:2 | qk_hide:3 | full:3 | serial:3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 128 | 819.9 | **816.8** | 497.0* | 380.1* | 776.6 | 646.7 | 724.4 | 497.0* | 380.1* | 839.3 |
| 16384 | 128 | 848.7 | **835.7** | 517.8* | 397.7* | 797.3 | 686.1 | 760.7 | 517.8* | 397.7* | 873.0 |
| 4096 | 64 | 582.4 | **556.4** | 439.5* | 359.9* | 427.3 | 401.7 | 414.4 | 504.2 | 497.8 | 497.5 |
| 16384 | 64 | 610.5 | **580.5** | 454.1* | 364.1* | 454.0 | 418.7 | 434.3 | 526.1 | 518.9 | 520.5 |

（* = 该配置 spill）

**裁决：d128、d64 均 serial 全胜**。最优流水配置 vs serial：
d128 qk_hide:2 = 0.925/0.913，d64 qk_hide:3 = 0.906/0.906。

**实验 A 翻正验证（同占用对照，重叠机制本身有效）**：
- d128 @2CTA：qk_hide:2 vs serial:2 = **+7.2%/+4.8%**（s4096/s16384）——
  merge ∥ QK 重叠真实兑现（当年 temp 复用让 QK 等 merge 的 +11.5% 教训的反向
  确认）；且 qk_hide:2 指令数 611.7M 略低于 serial 614.0M（编排零冗余）。
- d64 @3CTA：qk_hide:3 vs serial:3 = **+1.3%/+1.1%**；full:3 ≈ serial:3（d64
  merge 仅 32 FFMA，softmax 掩盖也无净增益——issue 0.61 尚有空隙，标量链不是
  d64 的约束）。
- full 在任何配置都劣于 qk_hide：softmax 重叠要求 temp+acc_qk 双在飞，
  2CTA@256 也 spill（16.8M sectors）+654M 指令。

## 5. ncu 归因（主形状 b4 h32 s4096，-s 3 -c 1）

默认占用（168/128 预算）——spill 是直接死因：

| | serial d128 | qk_hide d128 | full d128 |
|---|---:|---:|---:|
| duration | 1.663 ms | 2.694 ms | 3.370 ms |
| local ld/st sectors | **0 / 0** | 188.7M / 194.6M | 336.6M / 345.5M |
| 指令总数 | 614.0M | 726.8M (+18%) | 828.5M (+35%) |
| issue_active | 0.520 | 0.380 | 0.340 |
| launch regs | 167 | 168 | 168 |

寄存器墙的算术：merge 点恒有 acc_pv(64)+temp(64) 活跃，qk_hide 再加在飞
acc_qk(64) ≈ 232 regs > 168；full 的 softmax 点 acc_s+acc_pv+在飞 temp+在飞
acc_qk ≈ 296 > 256（2CTA 也 spill）。ptxas 对在飞 wgmma 的 acc 从 issue 到可
证明退休的 wait_group 全程 pinned，生命期错开洗不掉。
无 spill 配置后瓶颈转为占用（--set full 前后对照，GPU7）：

| | serial d128 | qk_hide:2 d128 | serial d64 | qk_hide:3 d64 |
|---|---:|---:|---:|---:|
| duration (ncu) | 1.665 ms | 1.866 ms | 1.306 ms | 1.443 ms |
| regs；local | 167；0 | 252；0 | 126；0 | 166；0 |
| warps_active | 18.46% | **12.37%** | 24.47% | **18.44%** |
| issue_active | 0.521 | 0.462 | 0.608 | 0.550 |
| 指令总数 | 614.0M | 611.7M | 563.4M | 562.1M |
| stalled_gmma | 0.672 | **0.561** | 0.810 | 0.800 |
| stalled_not_selected | 0.690 | 0.375 | 1.113 | 0.739 |
| tensor pipe (imma, of active) | 22.76% | 20.29% | 14.49% | 13.13% |

归因链条自洽：d128 qk_hide:2 的 gmma-stall 确实下降（0.672→0.561，merge ∥ QK
兑现）、指令还略少，但 3→2 CTA 让 warps_active 18.5→12.4%、issue 0.52→0.46，
时长净 +12%；d64 的 gmma-stall 几乎不动（0.810→0.800，merge 仅 32 FFMA），
占用 4→3 CTA 纯亏。

## 6. 结论与教训

1. **serial 维持合入态**（kernel 代码回退到 7034143 状态）；三模式实现存档
   `softpipe-three-modes.patch`（基于 7034143，含 SAGE_PIPE_MODE/SAGE_MINCTASM
   开关与 harness），未来若寄存器预算变化（CTA_K=64 变体、sm100 tmem）可复活。
2. FIFO 约束本身可解（§2 的 commit 序+W(1) 编排、恒等 merge 垫位），**真正的
   墙是寄存器**：这颗 kernel 的 serial 峰值恰好贴满预算（167/168、126/128），
   属于「零寄存器空隙」形态——与 ping-pong 轮「占用损失吃掉流水收益」同一律。
3. 量化收益上限重估：重叠全兑现也只有 +5~7%（d128 iso-2CTA 实测），而占用
   代价 -11~13%；除非未来把 merge/softmax 的寄存器足迹砍半，此路线关闭。
4. 副产物：三模式输出逐位一致证明「只动 wgmma 编排」的数值零改动论断可实测
   钉死（bit-identical 而非阈值比对），后续同类实验可沿用 smoke_pipe 手法。

## 7. Artifacts

本目录：`softpipe-three-modes.patch`（负者实现全文）；`harness/`
（bench_pipe.py 交错 A/B、smoke_pipe.py bit-identical 闸、profile_kernels.py）；
`reports/*.ncu-rep`（softpipe_{serial,qkhide,qkhide2}_d128、
softpipe_{serial,qkhide3}_d64，--set full + PmSampling）；`analysis/`
（analyze_sm90.py、并排对比、全量 metric JSON ×5）。远端副本：hyper01
`/data02/triplemu/workspace/SageAttention/profile/2026-07-13-softpipe/`。
过程记录：`.superpowers/sdd/softpipe-design.md`、`softpipe-report.md`（未跟踪）。
