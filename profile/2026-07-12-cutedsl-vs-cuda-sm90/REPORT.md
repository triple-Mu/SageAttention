# NCU Profile：CuTe-DSL vs CUDA C++ SageAttention kernel（H200 / sm90）

**TL;DR**：主形状（b=4, h=32, s=4096, d=128, non-causal）下 CuTeDSL 3.4525 ms vs CUDA 1.6850 ms（0.488×，与 bench 0.40–0.60× 区间一致）。差距可乘法分解为 **2.05× = 1.68×（指令膨胀）× 1.22×（发射效率）**：
1. **指令膨胀是最大单一因素**：DSL 执行 1,031.3M 条 warp 指令 vs CUDA 613.6M（+68%），多出的 ~420M 集中在 softmax 段标量循环——`cutlass.max` 被编译成 NaN 语义的 FSETP+FSEL+SEL（DSL 334M 条 vs CUDA 用 FMNMX 仅 73M），3 个独立 FMUL 循环未融合（DSL 211M FMUL vs CUDA 8.4M，CUDA 融进 138M FFMA）。
2. **占用率过低使一切延迟裸露**：DSL 1 CTA/SM（regs=240、smem=113KB 双限制）、每 smsp 仅 1 个 math warp（not_selected=0.007），WGMMA wait_group(0)（18.3% 样本）与每 CTA 一次的 pipeline-fill 等待（10.7% 样本）完全无掩盖；CUDA 相同的串行结构（15.8% DEPBAR 样本）靠 3 CTA/SM 互相覆盖（not_selected=0.858）。

---

## 1. Setup（可复现）

- 机器：hyper01 H200（132 SM，SM 1.35 GHz / DRAM 3.2 GHz），容器 `sglang-diffusion-qwenimage`，GPU 2（`CUDA_VISIBLE_DEVICES=2`，ncu 内 `--devices 0`）；ncu 2025.1.1，`ncu_report` 位于 `/opt/nvidia/nsight-compute/2025.1.1/extras/python`。
- Harness：`harness/profile_kernels.py`（口径同 `cutedsl_sage/bench_compare.py`：预量化输入、per_warp、fuse v_scale；warmup 3 + 精确 5 次调用，ncu 用 `-s 3 -c 1` 抓稳态第 4 次 launch）。`PROF_KERNEL=cuda|dsl` 选 kernel，`PROF_D=128|64` 选形状。
- 采集命令（每 kernel 两轮）：
  ```
  ncu --set full --section PmSampling --section PmSampling_WarpStates --devices 0 \
      -k "regex:<NAME>" -s 3 -c 1 -o reports/full_<tag> python harness/profile_kernels.py
  ncu --set source --section SourceCounters --devices 0 \
      -k "regex:<NAME>" -s 3 -c 1 -o reports/source_<tag> python harness/profile_kernels.py
  ```
  kernel 名：CUDA `qk_int8_sv_f8_attn_kernel<64u,128u,128u,128u,(QuantGranularity)2,...>`；CuTeDSL `kernel_cutlass_kernel_coreSageAttnSm90_object_at_...`（regex 分别用 `qk_int8_sv_f8_attn_kernel` / `SageAttnSm90`）。
- 无 lineinfo（两侧均 JIT/预编译产物），source 归因在 SASS 级（opcode + 指令计数模式反推源码循环，见 §4）。ncu 抓 CuTeDSL kernel（tvm-ffi/cuda-bindings 启动路径）**无需任何 workaround**。
- 分析脚本：`analysis/analyze_sm90.py`（全量 metric + 并排对比）、`analysis/extract_source_stalls.py`（per-PC stall → SASS opcode 聚合）。注意 ncu 2025.1 Python API 与 skill 文档差异：无 `rule_results_as_dicts()`（rule 建议改从 `--page details` 取）、`sass_by_pc(addr)` 需传地址。

## 2. Headline 对比（主形状 d=128；副形状 d=64 附后）

| metric | CuTeDSL | CUDA | 比值 |
|---|---:|---:|---:|
| duration | 3.4525 ms | 1.6850 ms | **2.049×** |
| 折算 TFLOPS | 318 | 652 | 0.488× |
| warp 指令执行总数 | 1,031.3M | 613.6M | **1.681×** |
| 每 scheduler 发射率（issue_active/cyc） | 0.426 | 0.519 | 0.821× |
| eligible warps / scheduler | 0.43 | 0.96 | |
| active warps / scheduler | 1.22 | 2.96 | |
| SM throughput SOL | 42.0% | 51.2% | |
| Memory SOL（L1/L2 主导） | 17.9% | 47.6% | |
| DRAM read（占峰值） | 1.19% | 2.43% | 都远离 DRAM bound |
| tensor pipe active（hmma/imma, of active） | 11.06% | 22.68% | |
| regs/thread；dyn smem/CTA | 240；113.0 KB | 167；40 KB | |
| CTA/SM（Block Limit regs / smem） | **1**（1 / 1） | **3**（3 / 3） | |
| theoretical / achieved occupancy | 12.5% / 7.63%（4.88 warps） | 18.75% / 18.49%（11.84 warps） | |
| grid；block | 8192；256 | 8192；128 | waves 62.1 vs 20.7 |

交叉验证：duration 比 2.049 ≈ 指令比 1.681 × 发射率比 (0.519/0.426)=1.219（乘积 2.049，严格吻合）——差距完全由"指令更多"ד发出更慢"构成，与访存带宽无关（DRAM read 两者 <2.5%）。

## 3. 六维度证据

1. **Occupancy/launch**：DSL Block Limit Registers=1 且 Shared Mem=1 → 1 CTA/SM，8 warps 理论；achieved 仅 4.88——core.py:432-434 中 load WG 只留 warp 0 发 TMA、其余 3 warp 直接退出，故稳态 = 4 math + 1 load warp。CUDA 3 CTA/SM，achieved/theoretical = 98.6%。NCU rule（两侧同类）："theoretical occupancy … limited by registers and shared memory"，DSL Est. Speedup 57.43%、CUDA 48.15%。
2. **均衡/tail**：无 tail。PM timeline（`analysis/pm_timeline_d128.txt`）两 kernel 均平顶净落；waves 62.1/20.7。
3. **Stall 分布**（占 active warp-cycles，由 ratio×issue 换算）：DSL——issuing 34.8%、wait 21.6%、long_scoreboard 19.5%、barrier 17.4%；CUDA——issuing 17.5%、wait 22.2%、barrier 15.4%、not_selected 15.1%、mio_throttle 7.0%。DSL 的 not_selected≈0.7%（无富余并行度），CUDA 15.1%（富余）。
4. **Tensor core**：两侧 GMMA 指令数完全相同（IGMMA 4.19M + QGMMA 4.19M，同 64x128x32 tile），tensor pipe active 11.06% vs 22.68%——tensor 工作量相同，DSL 用 2.05× 时间摊薄。即使 CUDA 也只有 22.7%，说明两者都是"非 tensor 段"主导。
5. **SM 利用率 timeline**：稳态平坦（DSL inst_executed 38.7% avg / CUDA 44.2% avg），无周期性锯齿（1.5µs 采样间隔无法分辨 ~1.8µs 的单次迭代相位，见 caveat）。
6. **访存**：DSL L1 hit 85.8%/L2 92.5%；global ld sectors/req=1.09（合并好），但 rule 指"7.0/32 bytes/sector 被利用，Est. Speedup 14%"（scale 类 LDG 步长访问）；shared st bank conflict 仅 7,501 次（可忽略）；两侧 local ld/st=0（无溢出）。CUDA 自身有 41.65% Est. Speedup 的 uncoalesced 规则（O store 16.0/32 bytes/sector）——CUDA 也有可挖空间。

## 4. SASS 级指令归因（source set，`analysis/sass_opcode_source_*_d128.txt`）

基准单位：67.1M = 每元素每迭代 1 条（8192 CTA × 32 kv-iter × 4 warp × 64 元素/线程）。

| 功能 | CuTeDSL | CUDA | 源码位置（core.py） |
|---|---|---|---|
| dequant int→f32 | I2FP 67.1M | I2FP 67.1M | L647 |
| dequant ×scale | FMUL 67.1M | 融进 FFMA | L649-650 |
| 行 max | FSETP.GTU 67.1M + FSETP.NAN 67.1M + FSEL 136.4M + SEL 53.4M（NaN 语义 compare+select） | **FMNMX 73.4M** | L708 `cutlass.max` |
| exp2 参数 | FFMA 69.2M | FFMA（138.4M 含 dequant） | L717-719 |
| exp2 | MUFU.EX2 69.2M | MUFU.EX2 69.2M | L718 |
| 行 sum | FADD 69.3M | FADD 67.2M | L724 |
| ×448 | FMUL 67.1M | 0（折叠/FMNMX clamp） | L668-669 |
| PV rescale | FMUL 67.1M | FMUL 8.4M | L726-727 |
| fp8 pack | F2FP 33.6M + **PRMT 33.6M** | F2FP 33.6M | make_acc_into_op |
| quad reduce | SHFL.IDX 16.8M + BFLY 4.3M | BFLY 4.3M | L709-711 |

NCU rule 佐证：DSL "280.1M non-fused FP32 vs 69.2M fused"（CUDA 75.6M non-fused vs 138.4M fused）。

**Stall 热点（pcsamp）**：DSL 总样本 296,024——Top1/2 均为 `WARPGROUP.DEPBAR.LE`（wait_group(0)，各 27.1K/26.0K，合计 **18.35%**，stall 原因 barrier）；4 个 prologue mbarrier 等待 BRA（每 warp 每 CTA 执行 1 次，long_scoreboard，合计 29.5K ≈ **10.0%**，再加 math WG 专属 1 个 2.1K → 10.7%）；其余 ~56% 散布在上表标量 opcode（wait/short_scoreboard 为主，即依赖链+发射时间）。CUDA 总样本 144,452——DEPBAR 15.81%、MUFU.EX2 21.95%（mio/wait，靠 not_selected 掩盖）。

## 5. 三个结构性怀疑：证实/证伪

- **① 单 math WG 串行（每次 WGMMA 后 wait_group(0) 全等待）——部分证实，需修正**。DEPBAR（wait_group）确为 DSL 第 1、2 号 stall 热点（18.35% 样本；stalled_gmma 仅 0.015 说明等待都压在显式全屏障上）。但 **CUDA 每 warpgroup 内同样串行**（每迭代同样 2 次 DEPBAR，占其样本 15.81%）——串行结构本身不是差距来源；差别在 CUDA 有 3 CTA/SM 提供并发掩盖（not_selected 0.858 vs 0.007）。结论：①独立成立的部分很小，应与③合并理解为"串行 + 零掩盖"。
- **② softmax 3 个 64 次标量循环——证实，且为最大单一因素**。量化见 §4：DSL 每元素 ~13 条指令 vs CUDA ~6 条；总指令 +417.7M（+68%），其中 `cutlass.max` 的 NaN 语义展开贡献 ~261M（334M−73M）、未融合 FMUL 贡献 ~150M、PRMT+SHFL.IDX ~50M。编译期 DSLOptimizationWarning 指向的正是这些循环。
- **③ 256 线程 + 240 reg + 113KB smem 压 occupancy——证实**。regs 和 smem 双双把 Block Limit 压到 1（CUDA 167 reg/40KB → 3）；achieved 7.63% vs 18.49%。直接后果有数：eligible 0.43 vs 0.96、发射率 0.426 vs 0.519、DEPBAR 18.35% 和 prologue-fill 10.7% 全裸露。

**新发现**：(a) 每 CTA 一次的 pipeline-fill 等待占 10.7% 样本——62 waves × 1 CTA/SM 意味着每 SM 串行经历 62 次冷启动、无相邻 CTA 覆盖（CUDA 对应 PC 仅 0.8%）；(b) load WG 4 warp 中 3 个立即退出（资源上无害，但 launch 仍按 256 线程 × 240 reg 计占用）；(c) 两 kernel 都远离 DRAM bound（read ≤2.4%），DSL 追平 CUDA 后 L2（47.7% busy@CUDA）也尚不是墙；(d) 无 tail、无 bank conflict、无寄存器溢出。

## 6. 优化建议（按预期收益排序）

1. **P1 消除 softmax 标量膨胀（预期 −30~45% 时长，effort 低-中，纯 DSL 源改动）**
   - `cutlass.max`（core.py:708）改用能 lower 成 FMNMX 的 max（省 ~261M 条，占 DSL 总指令 25%）；
   - ×448（L668）折进 exp2 常数：`exp2(x·log2e·s − m + log2 448)`，省 67M FMUL；dequant ×scale（L649）折进 exp2 的 FFMA（把 q_scale·k_scale 并进 scale_softmax_log2，注意 k_scale 逐块变化时行 max 的量纲），再省 67M；
   - fp8 pack 布局对齐 CUDA 消 33.6M PRMT；quad-reduce 的 16.8M SHFL.IDX 复查。
   - 依据：duration 与指令数近似成正比（发射率恒定下 1.68× 指令 ↔ 2.05× 时长的主项）；全做完指令数≈CUDA，单此项理论上 3.45→~2.1 ms。
2. **P2 提供并发掩盖（预期再 −20~30%，effort 高）**：优选 FA3 式第二个 math WG（384 线程 ping-pong，2×232+24 regs 可容），或砍资源跑 2 CTA/SM（kv_stage 6→3 省 ~54KB smem，math regs 240→≤232）。目标：掩盖 DEPBAR 18.35% + prologue-fill 10.7%，把发射效率从 0.426 抬向 ≥0.52（CUDA 靠 3 CTA 做到 0.519 且 not_selected 0.858）。
3. **P3 scale 类 LDG 合并性（NCU Est. Speedup 14%，实际预期 ≤5%）**：7.0/32 bytes/sector；global ld 相关 stall 样本占比小，收益有限，最后做。

P1+P2 合计理论上到 CUDA 持平或更好（CUDA 自身还挂着 41.65% uncoalesced-store 规则未修）。

## 7. Confidence & caveats

- 指令归因高置信：67.1M=每元素 1 条的模式在 10 个 opcode 上严格成立，且与 core.py 循环一一对应；duration 分解式严格闭合（2.049=1.681×1.219）。
- 无 lineinfo → per-PC 归因基于 SASS；prologue BRA 判定基于 PC 序（0x…9e980-9f1c0 < 主循环 0x…a0100-a3960）与 exec 计数（每 warp 每 CTA 1 次）。
- PM 采样间隔 1.5µs > 单次 kv 迭代 ~1.8µs/2，无法直接目视 QK/softmax/PV 相位交替；串行性的证据来自 DEPBAR 热点而非 timeline。
- ncu replay 下的 duration 与 do_bench 中值不同源，但比值 0.488 与 bench 中值一致。

## 8. Artifacts

本目录：`harness/profile_kernels.py`；`reports/{full,source}_{dsl,cuda}_d128.ncu-rep` + `full_{dsl,cuda}_d64.ncu-rep`（均 <5MB，已入 git）；`analysis/`（并排对比表、SASS opcode 聚合、details 页 dump、全量 metric JSON、PM timeline）。远端副本：hyper01 `/data02/triplemu/workspace/SageAttention/profile/2026-07-12-cutedsl-vs-cuda-sm90/`（容器内 `/data/workspace/...`）。

**附：d=64 副形状**——同一故事且更差：DSL 3.00ms vs CUDA 1.26ms（0.42×）；CUDA 升到 4 CTA/SM（achieved 24.5%、3.92 warps/smsp、发射率 0.637），DSL 仍 1 CTA/SM（7.56%、1.21、0.458）。
