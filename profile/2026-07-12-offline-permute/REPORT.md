# NCU Profile：V offline permute（消除 P fragment 在线重排）归因（H200 / sm90）

**TL;DR**（主形状 b=4, h=32, s=4096, d=128, non-causal）：把 `make_acc_into_op` 的
shuffle_sync+prmt 在线重排拆成「线程内 τ 字节交织（免费，f32 gather 折进 F2FP 源操作数）
+ V 量化时 offline φ 置换（零成本地址重排）」后，**SHFL.IDX 16.8M 与 SEL 53.4M 全部清零、
PRMT 减半（33.6M→16.8M）、总指令 760.4M → 692.4M（−8.9%）、duration 1.9991 → 1.8225 ms
（−8.8%）**；vs CUDA 从 1.186× 收敛到 **1.082×**（指令面积 1.128× × 发射率比 0.965 ≈ 1.088，
分解闭合）。d64 duration −5.3%（vs CUDA 1.261×，占用/发射短板开始主导）。
剩余指令差 +78.8M 的清单见 §5。

## 1. Setup（可复现）

- 机器/容器/ncu 与 `profile/2026-07-12-cutedsl-vs-cuda-sm90/` 一致（hyper01 H200 132 SM，
  容器 `sglang-diffusion-qwenimage`，ncu 2025.1.1）；harness 同款复用
  （`harness/profile_kernels.py`，预量化输入、warmup 3 + 5 次调用、`-s 3 -c 1` 抓稳态第 4 次
  launch）。**采集期间 GPU2 被其他作业占用，本轮新报告改在空闲 GPU7 采集**
  （同型号 H200，空载 SM clock 同为 1980MHz；指令计数与设备状态无关，duration 跨卡可比性
  按 ±1% 看待）。harness 喂随机 V——V 置换不影响 kernel-only 口径（kernel 不感知置换）。
- 被试：
  - `full_dsl3b_*` / `source_dsl3b_d128`：**offline permute 最终版**（变体 B，load 侧 τ）
  - `source_dsl3_d128`：中间变体 A（store 侧 τ），仅 source 集，codegen 对比存档
  - `prev_full_dsl2_*` / `prev_source_dsl2_d128`：两级累加版基线（HEAD 5005f6d，上轮
    `2026-07-12-two-level-accum` 报告逐字节副本）
  - `prev_full_cuda_*` / `prev_source_cuda_d128`：CUDA 基线（首轮报告逐字节副本）
- 采集命令（与上轮相同）：
  ```
  ncu --set full --section PmSampling --section PmSampling_WarpStates --devices 0 \
      -k regex:SageAttnSm90 -s 3 -c 1 -o reports/<tag> python harness/profile_kernels.py
  ncu --set source --section SourceCounters --devices 0 -k regex:SageAttnSm90 -s 3 -c 1 ...
  ```
- 分析：`analysis/analyze_sm90.py`（并排 metric + 全量 JSON）、
  `analysis/extract_source_stalls.py`（SASS opcode 聚合 + pcsamp 热点）、
  `analysis/dump_sass.py` / `dump_fsel2.py`（按 PC 序 dump SASS，本轮新增，用于打包簇与
  FSEL 链归因）。

## 2. Headline（full set，d128 主形状）

| metric | dsl3b（本轮） | dsl2（基线） | CUDA | dsl3b vs dsl2 |
|---|---:|---:|---:|---:|
| duration | **1.8225 ms** | 1.9991 ms | 1.6850 ms | **−8.8%** |
| warp 指令总数 | **692.38M** | 760.43M | 613.58M | **−68.05M（−8.9%）** |
| vs CUDA duration / 指令比 | **1.082× / 1.128×** | 1.186× / 1.239× | — | — |
| 发射率 issue_active/cyc | 0.538 | 0.541 | 0.519 | ≈ |
| eligible warps/smsp | 0.750 | 0.762 | 0.964 | ≈ |
| achieved occupancy | 15.31% | 15.26% | 18.49% | 不变 |
| tensor pipe active (of active) | **20.85%** | 19.11% | 22.68% | +1.7pp |
| launch regs/thread；CTA/SM | 128；2 | 128；2 | 167；3 | 不变 |
| local ld / st（条） | 139,264 / 131,072 | 同 | 0 | 不变（两级累加 temp） |

交叉验证：vs CUDA duration 比 1.082 ≈ 指令比 1.128 × 发射率比 (0.519/0.538)=0.965
（乘积 1.088，残差 0.6% 在 GPU7/GPU2 跨卡口径内）——上两轮的分解框架依旧闭合；
**指令面积减少几乎 1:1 转化为 duration**（发射率不变）。

d64 副形状：duration 1.5894 vs dsl2 1.6775 vs CUDA 1.2608 ms（−5.3%；vs CUDA 1.261×）；
指令 642.53M vs 710.58M vs 564.04M（−68.06M，与 d128 完全同量——P 路径与 d 无关）；
发射率 0.570 vs 0.604 vs 0.637——d64 指令比只剩 1.139× 但发射率反而下降，
说明 d64 的短板已转为占用/延迟侧（CUDA 4 CTA/SM vs DSL 2）。

## 3. 两个实现变体的 codegen 对照（source set，d128）

τ 交织可以放在 cast 的 store 侧（字节位置换写入）或 load 侧（f32 寄存器 gather）。实测：

| opcode | dsl2（shuffle 版） | 变体 A（store 侧 τ） | **变体 B（load 侧 τ，最终）** | CUDA |
|---|---:|---:|---:|---:|
| SHFL.IDX | 16.78M | 0 | **0** | 0 |
| SEL | 53.41M | 34.54M | **0** | 0 |
| FSEL | 2.20M | 2.20M | 34.73M | 2.10M |
| PRMT | 33.55M | 38.80M | **16.78M** | 0 |
| F2FP | 33.55M | 33.55M | 33.55M | 33.55M |
| LOP3.LUT | 7.69M | 4.55M | 3.50M | 0 |
| IMAD.MOV.U32 | 10.33M | 6.13M | 6.13M | 0 |
| SHF.R.U32.HI | 2.37M | 0 | 0 | 0 |
| **总指令（source 口径）** | **767.52M** | 724.49M | **699.48M** | 613.58M |

- 变体 A 的字节级置换写入让 DSL 打包 codegen 退化（PRMT +5.2M、SEL 链残留 34.5M）；
- 变体 B 的 f32 gather 完全折进 F2FP 源操作数选择：**主循环打包 = 32 F2FP + 16 PRMT
  /线程/迭代，与置换前的自然打包完全相同、零 SHFL/SEL**（`dump_sass.py` SASS 复核：
  F2FP 簇内无任何 select/shuffle/额外 MOV）。

### 变体 B 唯一残留：FSEL 34.73M 的归因（dump_fsel2.py）

按 exec-count 分类：`{1,015,808: 2 条, 32,768: 998 条}`——
- 2 条 ×1.02M：主循环每行 `local_max = (s_max==-inf) ? 0 : s_max` 既有逻辑（dsl2 同款）；
- 998 条 ×32.8K = 32.7M：**mask 尾块**（`compute(masked=True)` 实例化，每 CTA 仅执行一次）
  的 residual-mask `if k_pos >= seqlen_k: acc_s[i]=-inf` 展开成 FSEL 链
  （SASS 见 dump：`ISETP.GE` + 每元素 2~3 条链式 FSEL）。dsl2 同一块用 ~606 条 SEL
  （19.9M）表达同一语义——mask 路径的 codegen 形态变化 +12.8M（占全 kernel +1.8%），
  属尾块既有问题，非本改动主径；已入 §5 残留清单。

## 4. 指令数分解 vs CUDA（source set，d128；基准 67.1M = 每元素每迭代 1 条）

| 功能 | dsl3b | CUDA | 差 | 备注 |
|---|---:|---:|---:|---|
| dequant I2FP | 67.1M | 67.1M | 0 | |
| dequant ×scale FMUL | 67.1M | 0 | **+67.1M** | CUDA 把 dequant 折进 exp2 参数 FFMA（行 max 在原始 S 上做再缩放 max）；DSL 未做，最大残留单项 |
| FFMA（exp2 参数+两级 merge） | 138.4M | 138.4M | 0 | 两级累加轮已对齐 |
| 行 max FMNMX | 71.3M | 73.4M | −2.1M | |
| exp2 MUFU.EX2 | 69.2M | 69.2M | 0 | |
| 行 sum FADD | 69.3M | 67.2M | +2.1M | |
| PV rescale FMUL | 7.3M | 8.4M | ≈0 | |
| fp8 pack F2FP | 33.6M | 33.6M | 0 | |
| fp8 pack PRMT | 16.8M | 0 | **+16.8M** | CUDA 用 F2FP.MERGE_C 直接拼高低半字，DSL 每 u32 仍 1 条 PRMT |
| mask 尾块 FSEL | 32.7M | ~0 | **+32.7M** | §3 归因；每 CTA 一次 |
| quad reduce SHFL.BFLY | 4.3M | 4.3M | 0 | |
| 地址/uniform（UMOV/UIADD3/ULEA/IMAD 等） | ~53M | ~19M | **+34M** | DSL 地址生成，含 kv_stage 环形索引 |
| 本轮消除项（SHFL.IDX+SEL+半数 PRMT） | 0 | 0 | 0 | dsl2 时为 +67.1M |
| **合计（source 口径）** | 699.5M | 613.6M | **+85.9M（1.140×）** | full 口径 +78.8M（1.128×） |

## 5. 结论与残留清单（下一步优化目标，按量排序）

1. **dequant FMUL 67.1M**：把 q_scale·k_scale 折进 exp2 参数 FFMA（行 max 改在原始
   S_f32 上做、m 再乘 dequant，与 CUDA 同构）。做完理论指令 ~625M ≈ CUDA 的 1.02×。
2. **mask 尾块 FSEL 32.7M**：mask 双条件（residual+causal）逐元素 select 链；可改
   fmin(acc, ±inf) 或按列向量化。每 CTA 只执行一次，收益上限 ~4.5% 指令 / ~2% duration。
3. **pack PRMT 16.8M**：需要 DSL 生成 F2FP.MERGE_C 目标寄存器复用（高半字直写），
   目前 `.to(Element)` 向量化路径每 u32 仍留 1 条 PRMT。
4. **地址/uniform ~+34M**：DSL 代码生成固有，改动性价比低。
5. d64 的短板已切换到占用/发射（指令比 1.139 但发射率 0.570 vs CUDA 0.637）：
   下一步是 P2b（第二 math WG / 3 CTA/SM 路线）而非继续抠指令。

**本轮结论**：offline permute 以零量化成本消掉了在线重排的全部 67.1M 条指令
（SHFL/SEL 清零、PRMT 减半，再加杂项 −68.0M 总量），duration 收益与指令面积严格 1:1
（发射率不变），kernel-only 差距 1.186×→1.082×（bench 口径 d128 稳态 0.93–0.95×，
causal 最高 0.97×；e2e 含量化已全面 ≥ CUDA）。

## 6. Artifacts

本目录：`harness/profile_kernels.py`；`reports/`（`full_dsl3b_{d128,d64}` +
`source_dsl3b_d128` + 变体 A `source_dsl3_d128` + prev 逐字节副本 ×6）；`analysis/`
（并排对比 ×2、SASS opcode 聚合 ×4、dump 脚本 ×2、全量 metric JSON）。
远端副本：hyper01 `/data02/triplemu/workspace/SageAttention/profile/2026-07-12-offline-permute/`
（容器内 `/data/workspace/...`）。φ 的实验反解记录与实现细节：
`.superpowers/sdd/offline-permute-report.md`。
