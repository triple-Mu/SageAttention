# 收尾会话总报告(2026-08-30,feat/varlen 45aadd7 → 收口)

一轮「最后整理 + 优化」campaign 的完整账目。全部数字为实机测量,口径:同卡独占、双向交替 ≥3 轮、每形状 median(功耗 cap 机 min-of-N)、全表几何均值、>0.5% 才算信号。

## 1. 性能收益表(落袋,已在默认路径生效)

| 改动 | arch | 实测收益 | bench 口径 | 数据产物 |
|---|---|---|---|---|
| C1 WS kernel + auto 启发式(d128 非 causal ≥16k / causal ≥32k 自动走 WS) | sm100 | auto/old d128 geomean **+1.07%**,命中段逐形状 +0.6~+2.7%,其余段 =old;ws/cudnn 0.732(old/cudnn 0.738) | cdsl_bench_fwd 22 点 ×3 轮四方轮换(old/ws/auto/cudnn),B200 | computelab `logs-w4/bench/`,本机 scratchpad `w4run/` |
| P2 quant dense/varlen 拆实例(per_warp + per_thread 家族) | sm100 | quant 组合 wall **+2.2~2.3%**(quant_q +1.4~1.8%、quant_k +2.4~3.2%) | quant_qk dense,b4h32 d128 seq{4k,16k,32k},min-of-30 ×3 轮交替,B200 | `logs-w4/p2/` |
| 同上 | sm90 | per_warp Q **+4.6%**、per_thread Q +2.4~3.7%(per-block 家族拆分实测劣化已回退,见 §3) | 同口径,H200 | hyper01 `/workspace/p2-sm90/` |
| 同上 | sm89 | 无回退(±0.4%,低于信号线) | kernel_breakdown b4h32 d128,L20 | computelab `logs-w4l/kbd/` |
| P4 V 融合门限 per-arch(sm89 4096→**12288**) | sm89 | 6-12k 段 V 前处理 kernel 级 **+15~20%**(融合比分离 0.80-0.85);两路 bit 等价,golden 免重 dump | b·h {32,64,128,256} × kv {4k..24k},L20 | bench/P4_VFUSE_L20.md §5,`bench/microbench/vfuse_l20/` |
| P4 门限(sm100/110 4096→**24576**) | sm100 | 4-24k 段融合快 **13~31%**(0.69-0.87,全范围无交叉);bitcheck 12/12 等价 | 同上,B200 | `logs-w4/p4_*.json` |
| C1 r4 collective-wait 交错 | sm100 | ws/old 0.9827→0.9890(**+0.64%**,非 causal 逐形状 ~+1%) | 同 C1 口径 | `logs-w4/bench/` |

C1 迭代史:r1 两遍读 0.883 → r2 单遍驻留 0.947 → r3 2×x64 宽 ld + f32x2 0.983 → r4 交错 0.989(对 old);auto 启发式把正收益段固化为默认。C1 完整 WS 对 cudnn 的差距(0.73)未如 cutedsl 先验(1.07-1.18)收敛——warp 供给兑现(Active/sched 2.0→3.4)但 tcgen05.ld 暴露延迟未被 ptxas 调度吸收(r4 的 PTX 交错在 sm_100a 被复沉,6/8 站点串行),继续深挖需源级在 ld 与首个消费点间填独立工作(重开 bit-exact 论证)或接受现状。

## 2. 工程收益表

| 项 | 结果 | 证据 |
|---|---|---|
| 代码整理(批 1-4) | **净 −1191 行**(45 文件,+1700/−2891):过渡 op 退役 −753、死代码 −582、base launcher −295、launcher/宏/fused 收拢 −402、kernel body 去重 −106、Python 整理与命名 +附带;2005 个 kernel 与基线 SASS 逐字节全同 | scratchpad `sassd_gateB/gateC`、`build_w4` 对比 |
| fwd 加 `backend=` 覆写 | 对拍跨家族覆盖保住,五机 golden 零重 dump(RETIRED markers 兼容层) | tools/compare_reference.py |
| Track D gencode 裁剪(`SAGE_PRUNE_GENCODE`,**已转正默认 ON**) | build CPU **−36.1%**、so 体积 **−38.0%**(118.7→73.6MB)、cubin 51→40 与决策表逐条一致;已转正;敞口=cc12 上 sm89 家族对拍需 OFF 重建 | bench/microbench/PRESCREEN_REPORT.md §4、`prescreen_data/trackd_*` |
| resolve 显式 backend 干净报错 | 撞未编 SASS 由 no-kernel-image 变 TORCH_CHECK 带 cure | csrc/sageattn/plan.cpp `backend_serves` |
| 测试补全 | test_varlen_sm89/sm120 各 62 kernel 级 packed 用例(sm120@pro-5k、sm89@L20 全过);sm89 fp8 accuracy 数值采齐(最差 cos 0.999296) | test/、HARDWARE_CHECKLIST §1b |
| varlen 零开销审计 | attention 侧本就是独立实例零开销(SASS/param 布局审计固化);quant 侧残余税裁决维持,超阈值的实例已由 P2 拆分 | PRESCREEN_REPORT.md Track D 节 |

## 3. 判负与回退(全部有实测,防重复立项)

| 方向 | 结论 | 证据 |
|---|---|---|
| E1 f16x2 softmax | 无硬件 packed MUFU 通路(四 arch SASS 均拆 2×MUFU+PRMT),sm86 实测与 f32 同速;sm90 不因此重开 | PRESCREEN_REPORT.md |
| P6 PDL | kernel 间 gap 全形状 ≤1.59%,PDL 可及部分 0.25µs/边;nsys 短调用注入伪 gap 的测量教训在案 | bench/P6_PDL_PRESCREEN.md |
| sm90 process_tile 收拢 | dense 持平、varlen kernel 锁基频 +1.47%(6/6)→ revert;教训:功耗 cap 下亚 1% 信号必须锁基频仲裁 | hyper01 `/workspace/sm90-7a64fc0/REPORT.md` |
| P2 per-block 家族拆分 | sm90 实测 plain +21%/sub_mean +3.4%(ptxas 分支域重构杀 LDG 并行)→ dense 回退 kVarlen=true 实例;静态 SASS 条数跨 toolkit 不可移植 | `/workspace/p2-sm90/`(SASS+A/B) |
| P8 sm80 fold | sm_86 +0.9% 不迁移:L20 0.9995、A100 0.9988;三平台合议「可回收 hunk 换简单性」建议在案 | bench/P8_SM80_FIX_{L20,A100}.md |
| C1 x128 宽 ld | ptxas C7602(单指令 dst 超 512 线程 kernel 的 128 入口 reg,函数级检查),x64 是结构上限 | C1_DESIGN.md §6.3 |

## 4. 挂账(明确悬置)

- P9 `fp32+fp16` 退役:等 4090 跑 `bench/microbench/mma_rate.cu`(org 内无卡,消费级 Ada 2× 速率变数);
- 4090 的 P4 门限复核(72MB L2 < L20 的 96MB,拐点可能早于 12288);

- sm80 fold hunk 回收(三平台合议建议,17 行换简单性);
- C1 下一 lever:ld 与首消费点间填独立工作(需重开 bit-exact 论证);sm100 varlen(等 C1 定型);
- 清理终审(2026-08-30 用户拍板):computelab 7 个历史构建树已删(~1.7G);hyper01 `SageAttention-rowsum`(独有试验 diff)与 pro-5k `/workspace/SageAttention`(v2g 未 push 工作)保留。证据类目录(kbd_*/profiles/golden_fixq/cdsl_src)保留。
