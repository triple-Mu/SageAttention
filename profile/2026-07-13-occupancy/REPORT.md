# NCU Profile：占用收官——128 线程全 consumer + 3/4 CTA/SM（H200 / sm90）

**TL;DR**（主形状 b=4, h=32, s=4096, non-causal）：把 256 线程（producer WG +
setmaxnreg）结构改为 **128 线程单 WG 全 consumer**（CUDA 版同构，warp0 兼职发
TMA），d128 3 CTA/SM、d64 4 CTA/SM：**d128 duration 1812.8 → 1667.5 μs
（vs CUDA 1685.0，0.990×——首次追平并反超）**；d64 1480.2 → 1307.5 μs
（vs CUDA 1260.8，**1.037×**，上轮 1.174×）。bench 全矩阵 24 点：d128 0.935–1.169
（12 点中 8 点 ≥0.99）、d64 0.952–1.046；端到端（triton 量化）全部快于 CUDA。
上一轮判定的「剩余差距 100% 在发射/占用」全部兑现：issue 0.464→0.520（CUDA
0.519）、warps_active 15.3%→18.5%（=CUDA）。**占用维度就此收官。**

## 1. Setup（可复现）

- 机器/容器/ncu 与前几轮一致（hyper01 H200 132 SM，容器 `sglang-diffusion-qwenimage`，
  ncu 2025.1.1）；harness 逐字节复用 `harness/profile_kernels.py`（预量化输入、
  warmup 3 + 5 次调用、`-s 3 -c 1` 抓稳态第 4 次 launch）。全程 GPU7（空闲 0 MiB）。
- 被试：
  - `full_dsl5_*`：本轮最终版（e95fd6c；128 线程全 consumer，d128 kv_stage=3 /
    minctasm=3，d64 kv_stage=5 / minctasm=4）
  - `full_dsl4sel_*`：上轮基线（97137a8，dequant-fold 报告最终版，逐字节副本）
  - `prev_full_cuda_*`：CUDA 基线（首轮报告逐字节副本）
- 采集：`ncu --set full --section PmSampling --section PmSampling_WarpStates`；
  分析 `analysis/analyze_sm90.py`（并排对比 ×2 + 全量 metric JSON ×6）。

## 2. Headline（full set，d128 主形状）

| metric | **dsl5（最终）** | dsl4sel（上轮） | CUDA |
|---|---:|---:|---:|
| duration | **1667.5 μs** | 1812.8 | 1685.0 |
| vs CUDA duration | **0.990×** | 1.076× | — |
| block / launch regs | 128 / **167** | 256 / 128 | 128 / 167 |
| CTA/SM（regs+smem limit） | **3 + 3** | 2 + 2 | 3 + 3 |
| smem/CTA | 66.56 KB | 82.94 KB | 40.96 KB |
| warps_active | **18.46%** | 15.34% | 18.49% |
| eligible warps/smsp | 0.879 | 0.649 | 0.964 |
| issue_active | **0.520** | 0.464 | 0.519 |
| warp 指令总数 | 614.0M | 597.1M | 613.6M |
| local ld/st（spill） | **0 / 0** | 139,264 / 131,072 | 0 / 0 |

d64 副形状：duration **1307.5 μs**（dsl4sel 1480.2，−11.7%；CUDA 1260.8，
**1.037×**）；**4 CTA/SM**（126 regs、smem 54.27KB，CUDA 同为 4）；warps_active
**24.46%**（CUDA 24.50%）；issue 0.608 vs 0.637（残差 ~4.6% 即最后差距）；
指令 563.4M vs 564.0M（0.999×）。

交叉验证：d128 指令比 614.0/613.6=1.001 × 发射率比 0.519/0.520=0.998 → 0.999
（duration 比 0.990，残差 ~1%）。指令面积维持持平（producer 逻辑内联进 math WG
新增 ~17M 被 dequant-fold 的余量吸收），赢的全部来自发射/占用——与上轮预测一致。

## 3. 结构改动（1e87a34 + e95fd6c）

1. **去独立 producer WG 与 setmaxnreg**：128 线程单 WG；warp0 prologue 发 Q +
   预填充 min(kv_stage, 2·trip_count) 个 K/V token（发射序=消费序 K0,V0,K1,…）；
   mainloop 中 K 在 QK wgmma 消费后立即 release 并补发下一 token，V 在 PV wgmma
   后同理（producer 状态作为 loop-carried 值穿过 compute）。pipeline 结构不变
   （PipelineTmaAsync，producer_group=Thread×1——DSL 允许 producer/consumer 同 WG）。
2. **占用参数**：d128 minctasm=3 + kv_stage 4→3（smem 66.56KB ≤ 75.7KB/CTA）；
   d64 minctasm=4 + kv_stage 10→5（54.27KB ≤ 56.7KB/CTA）。ptxas 在 168/128 reg
   预算内 0 spill（167/126 实测）。
3. mainloop/epilogue 数值代码逐字未变 → 数值语义零改动。

## 4. 为什么不是 setmaxnreg 146 或寄存器复用（负结果，已回退）

- **256 线程 + minctasm≥3 编译即失败**（NVVM backend error，无日志）。二分定位
  （const_expr 门控消融）：罪魁是 PV RS-wgmma 块（A fragment 16 reg + acc 64 reg
  pinned）在 launch 期 ~80 reg/线程预算下无法分配；trivial MRE（同 launch 配置 +
  setmaxnreg）可编译 → 后端对 wgmma 的寄存器可行性检查不认 setmaxnreg 区域。
- **寄存器复用**（acc_qk→acc_s 原位 recast、acc_pv_temp 复用 acc_s 存储，CUDA 版
  RS_f32/RO_temp 手法）在 224 reg/2 CTA 下无收益：spill 本来只有 4 reg/线程
  （全部来自 temp 活区）；temp 复用虽清零 spill 但让下迭代 QK wgmma 串行等待
  merge FFMA 链（d128 +11.5%）；原位 recast 在 5 CTA/96 reg 高压下反而 spill
  翻倍（LLVM 别名分析退化）。**最终版不含任何复用改造**——168/128 预算下
  LLVM/ptxas 自动完成活区复用。过程数据：`.superpowers/sdd/occupancy-3cta-report.md`。
- **d64 5 CTA/SM**（minctasm=5、kv_stage=3、96 reg）：duration 与 4 CTA 持平
  （1.31ms）但 17M 条 local ld/st——占用增益恰好被 spill 抵消，不取。

## 5. bench 口径最终（GPU7，do_bench 中值，CuTeDSL/CUDA）

- --quick 4 点：d64 s4096/16384 **0.954/0.959**（上轮 0.842/0.851），
  d128 **1.013/0.991**（上轮 0.960/0.932）。
- 全矩阵 24 点：d128 non-causal 0.972–1.103、causal 0.935–1.169；
  d64 non-causal 0.952–0.987、causal 0.968–1.046。短序列（s=1024）DSL 显著更快
  （d128 1.10/1.17）；仅存 <5% 差距集中在 d64 长序列（issue 0.608 vs 0.637）
  与 d128 causal s32768（0.935，波尾效应）。
- 端到端（含 triton 量化，d128）：CUDA/CuTeDSL = 1.03–1.70（全部 DSL 更快）。

## 6. 正确性

146 tests 两道闸全绿（quant bit-identical + kernel 两级比对，含 slow/FP22 长序列），
d128/d64 × full/causal × fp16/bf16 全组合。数值语义零改动（结构层，无运算顺序变化）。

## 7. 剩余空间与收官结论

1. d128 已 0.990×（反超），维度闭卷。
2. d64 1.037×：残差在 issue 0.608 vs 0.637（eligible 1.284 vs 1.477）。CUDA d64
   的 smem 占用限为 6 CTA（20.5KB/CTA、寄存器限 4），DSL smem 54KB 限 4——若要
   追，需 kv_stage 压到 3（37.9KB）同时不引起 spill（96 reg 下当前会 17M spill，
   见 §4），即需把 mainloop 峰值活区再压 ~30 reg——收益上限 ~3.7%，投入产出比低，
   停在此处。
3. stall 结构（d128）：DSL gmma-stall 0.671 vs CUDA 0.076（DSL 等 wgmma 更多）、
   CUDA not_selected 0.858 vs DSL 0.690（CUDA 等调度更多）——不同的等法，同样的
   发射率；无单点可攻。

## 8. Artifacts

本目录：`harness/profile_kernels.py`；`reports/`（full_dsl5 d128+d64、
full_dsl4sel ×2 与 prev_full_cuda ×2 逐字节副本）；`analysis/`（并排对比 ×2、
全量 metric JSON ×6、脚本）。远端副本：hyper01
`/data02/triplemu/workspace/SageAttention/profile/2026-07-13-occupancy/`
（容器内 `/data/workspace/...`）。实验过程与负结果记录：
`.superpowers/sdd/occupancy-3cta-report.md`。
