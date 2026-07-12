# NCU Profile：2 math WG ping-pong 实验归因（H200 / sm90）——为什么不如 2 CTA/SM

**TL;DR**（主形状 b=4, h=32, s=4096, non-causal）：ping-pong（CTA_Q=128=2×64、384 线程、
1 CTA/SM，fmha.py 底本结构）**最优变体也比现行 2 CTA/SM 慢 11.7%（d128）/ 14.5%（d64）**，
错峰 barrier 只会更慢（fmha 式 +2.9%、逐迭代式再 −7%）。**裁决：保留 2 CTA/SM，kernel 不合入**。
归因三项：① 冷路径指令面积 +4.5%（DSL 嵌套 jit ICE 迫使 peel 内联复制 masked 块，SEL
53.4M→87.9M，全部在每 warp 执行一次的冷路径）；② 发射率 0.541→0.518（活跃 warp 9.77→8.81/SM）；
③ 两 math WG 经共享 KV pipeline 锁进**同相**，softmax/EX2 与 wgmma 成簇爆发——mio_throttle
×2.75（0.122→0.336）、tensor pipe 19.1%→17.5%（不升反降）。ping-pong 唯一兑现的红利是
K/V 每 SM 只装载一次（L2 吞吐 38.7%→17.4%），但本 kernel 离 memory bound 极远（DRAM 2%），
换不来时间。

## 1. Setup（可复现）

- 机器/容器/GPU/ncu 与 `profile/2026-07-12-two-level-accum/` 完全一致（H200 132 SM，GPU 2，
  ncu 2025.1.1，容器 `sglang-diffusion-qwenimage`）；harness 同款复用
  （`harness/profile_kernels.py`，预量化输入、warmup 3 + 5 次调用、`-s 3 -c 1` 抓稳态第 4 次）。
- 被试：
  - `full_pp_d128` / `full_pp_d64` / `source_pp_d128`：**ping-pong 最优变体**
    （错峰=None、regs 224、kv_stage 10(d128)/16(d64)、q_stage 2、384 线程 1 CTA/SM）。
    kernel 源码存档：`.superpowers/sdd/core_pingpong_experiment.py`（未合入 main）
  - `full_ppfmha_d128`：fmha.py 式错峰变体（首块 QK + epilogue 的 PipelineOrder）
  - `full_dsl2_d128` / `full_dsl2_d64` / `source_dsl2_d128` / `prev_full_cuda_d128`：
    上轮报告（4340ec0，2 CTA/SM 两级累加）与 CUDA 的**逐字节副本**，保证本目录自包含
- 采集命令同上轮：
  ```
  ncu --set full --section PmSampling --section PmSampling_WarpStates --devices 0 \
      -k regex:SageAttnSm90 -s 3 -c 1 -o reports/<tag> python harness/profile_kernels.py
  ncu --set source --section SourceCounters --devices 0 -k regex:SageAttnSm90 -s 3 -c 1 ...
  ```
- 分析：`analysis/analyze_sm90.py`（并排对比）、`analysis/extract_source_stalls.py`（SASS
  opcode 聚合）、`analysis/dump_sel_context.py`（本轮新增：按 PC 定位 SEL 膨胀来源）。

## 2. Bench 全景（H200，bench_compare --quick，TFLOPS，non-causal）

| 配置 | d64 s4096 | d64 s16384 | d128 s4096 | d128 s16384 |
|---|---:|---:|---:|---:|
| **2 CTA/SM（HEAD 4340ec0，保留）** | **434.6** | **474.1** | **729.3** | **781.8** |
| ping-pong 无错峰 kv10/20 regs224 | 380.1 | 443.5 | 652.9 | 750.6 |
| ping-pong 无错峰 kv8/16 | 382.7 | 451.5 | 646.5 | 741.7 |
| ping-pong 无错峰 kv10/20 regs240 | 373.6 | 432.8 | 641.1 | 726.2 |
| ping-pong fmha 式错峰（+epilogue 串行） | 291.3 | 437.7 | 634.3 | 727.5 |
| ping-pong 逐迭代错峰（裸 mbarrier 严格交替） | 372.6 | 436.5 | 608.4 | 687.0 |
| （CUDA 对照） | 578.0 | 601.0 | 843.4 | 895.3 |

规律：**错峰越强越慢**（None > fmha > periter）；寄存器 240 比 224 慢；kv_stage 8↔10 在 ±1.5%。
最优 ping-pong 仍落后 2 CTA/SM **4.7%~11.9%**。正确性：所有变体 `-m "not slow"` 130 passed 全绿
（含 s=64/100/337 qo 尾块、causal、GQA、scale stress）。

## 3. Headline：ping-pong vs 2 CTA/SM vs CUDA（d128 主形状，ncu full）

| metric | ping-pong | 2 CTA/SM | CUDA | pp vs 2CTA |
|---|---:|---:|---:|---:|
| duration | 2.2323 ms | 1.9991 ms | 1.6850 ms | **+11.7%** |
| warp 指令总数 | 794.58M | 760.43M | 613.58M | **+4.5%** |
| 发射率 issue_active/cyc | 0.518 | 0.541 | 0.519 | **−4.3%** |
| eligible warps/smsp | 0.735 | 0.762 | 0.964 | −3.5% |
| 活跃 warp/SM | 8.81 | 9.77 | 11.84 | −9.8% |
| achieved occupancy | 13.77% | 15.26% | 18.49% | ↓ |
| CTA/SM；线程/CTA | 1；384 | 2；256 | 3；128 | — |
| launch regs；smem/CTA | 168；193KB | 128；82.9KB | 167；40KB | — |
| tensor pipe active (of active) | **17.46%** | 19.11% | 22.68% | **↓ 不升反降** |
| DEPBAR 热点占 pcsamp | 2 处共 ~31k 样本 | 结构同 | — | 不变 |
| L2 吞吐 lts | **17.4%** | 38.7% | 47.7% | K/V 只装一次的红利 |
| DRAM read 占峰值 | 1.84% | 2.05% | 2.43% | 全都远离 DRAM bound |

交叉验证：duration 比 1.117 ≈ 指令比 1.045 × 发射率比 (0.541/0.518=1.044) = 1.091，
残差 ~2.4% 来自 cycles active/elapsed 与 smsp 平均口径。d64 更差：duration 1.9201 vs
1.6775 ms（**+14.5%**），issue 0.546 vs 0.604，tensor pipe 10.1% vs 11.4%，
mio_throttle 0.357 vs 0.149。

## 4. 归因 ①：冷路径指令面积 +4.5%（DSL 结构性代价）

source set 的 SASS opcode 对比（pp 797.2M vs 2CTA 767.5M）里唯一的大项是
**SEL 53.41M → 87.88M（+34.5M）**。`dump_sel_context.py` 按 PC 分桶后定位干净：

| | 热循环 SEL PC（每 warp 30/31 次） | 冷路径 SEL PC（每 warp 1 次） |
|---|---:|---:|
| 2 CTA/SM | 18 个 × 1,015,808 | 1,072 个 × 32,768 |
| ping-pong | 18 个 × 983,040 | **2,142 个** × 32,768 |

热循环体两版完全一致（18 个 SEL，来自 make_acc_into_op 的 8-bit shuffle 区）；差异全部是
**冷路径 masked 块的两份内联拷贝**（peel 首块 + 尾块）——CuTe-DSL 4.6.0 把循环体抽成嵌套
`@cute.jit` 函数放进动态 while 会触发 IR dominance ICE（`operand does not dominate this use`，
pipeline/barrier 状态无法跨 region 线程化），fmha.py 底本因此也是内联展开首迭代，
这份代价是结构性的。s=4096 时冷路径占 2/32 个块 ≈ 6% 指令面积，随 s 增长摊薄——这解释了
pp 差距从 s4096 的 −10.4% 收窄到 s16384 的 −4.7%。

## 5. 归因 ②③：同相共振——发射率与 tensor pipe 双降

- **warp 供给**：1 CTA×12 warp vs 2 CTA×16 warp（load WG 少 4 个 warp），活跃 warp
  9.77→8.81/SM，eligible 0.762→0.735，issue 0.541→0.518。
- **同相锁定**：两 math WG 消费**同一条 KV pipeline**（每 stage 8-warp 共同 wait/release），
  没有错峰时它们经共享 mbarrier 锁进同相：同时 softmax（MUFU.EX2 爆发→
  **mio_throttle 0.122→0.336，×2.75**；EX2 热点的首要 stall 从 wait 变为 mio_throttle）、
  同时 wgmma（tensor pipe 更簇状：active-of-active 19.1%→17.5%）、同时 FFMA
  （math_pipe_throttle 0.151→0.199）。2 CTA/SM 的两个独立 CTA 会自然漂移错开，混合更平滑。
- **变好的项救不回来**：long_scoreboard 0.843→0.465（kv_stage 4→10 缓冲更深）、
  L2 吞吐减半、L1 hit 92.5%→94.6%——但 kernel 是发射/指令面积 bound，不是访存 bound。

## 6. 错峰（order barrier）为何反而更慢

- **fmha 式**（首块 QK + epilogue 串行，`full_ppfmha_d128`）：duration +2.9%（2.2968 ms），
  issue 0.518→0.504，barrier stall 0.750→0.782，mio_throttle 0.336→0.387，
  tensor pipe 17.46%→16.98%。一次性反相位撑不过共享 KV 等待的重新同相，epilogue 串行
  还让 WG2 的收尾直接排队（d64 s4096 掉到 291 TFLOPS）。
- **逐迭代式**（裸 mbarrier + 显式 phase，严格交替；PipelineOrder 状态进不了 DSL 动态
  while，须手工展开）：bench 再 −7%（608/687）。每迭代 2 次 mbarrier 往返压在 QK 发射的
  关键路径上，且把两 WG 的抖动互相放大。
- 根因：tensor pipe 只有 ~19% 占用、**不是争用瓶颈**，错峰解决的是「MMA 互相排队」的问题
  ——这里不存在，于是只剩同步开销。ping-pong 的教科书收益（fp16 GEMM 类 MMA 饱和场景）
  在 int8+fp8 SageAttention 的指令面积 bound 形态下没有着力点。

## 7. 结论与下一步

- **保留 2 CTA/SM（HEAD 4340ec0），ping-pong kernel 不合入**；实验版全文存
  `.superpowers/sdd/core_pingpong_experiment.py`（错峰三模式由 `pingpong_order` 切换）。
- 上轮结论维持：与 CUDA 的剩余 1.19× 差距在标量指令面积（SEL/PRMT/SHFL），
  优先级应回到 make_acc_into_op 的 8-bit fragment 重排（热循环的 53.4M SEL + 33.6M PRMT +
  16.8M SHFL）而非执行模型。
- 若未来热循环指令面积压到 CUDA 水平、tensor pipe 逼近饱和，ping-pong 可重新评估
  （届时错峰才有争用可解）。

## 8. 工件

本目录：`harness/profile_kernels.py`；`reports/{full_pp_d128,full_pp_d64,source_pp_d128,
full_ppfmha_d128}.ncu-rep` + 上轮副本 `{full_dsl2_d128,full_dsl2_d64,source_dsl2_d128,
prev_full_cuda_d128}.ncu-rep`；`analysis/`（三方对比表、opcode 聚合、SEL 定位与上下文、全量
metric JSON）。远端副本：hyper01 `/data02/triplemu/workspace/SageAttention/profile/2026-07-12-pingpong/`。
