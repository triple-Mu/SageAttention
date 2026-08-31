# 收尾 + sm100 C1 战役总报告(2026-08-31,feat/varlen 45aadd7 → e3e65b5)

「最后整理 + 优化」campaign 与后续 sm100 C1 战役(wave9-25)的完整账目。全部数字为实机测量,口径:同卡独占、双向交替 ≥3 轮、每形状 median(功耗 cap 机 min-of-N)、全表几何均值、>0.5% 才算信号;sm100 判据模板 = geomean >1.005 且无形状 <0.995,**判据以自由时钟 bench 为准**(ncu 锁频 duration 可与 bench 反向,wave23 实证)。B200 会话纪律与门禁口径见 `HANDOFF.md` §1-2。

## 1. 性能收益表(落袋,已在默认路径生效)

sm100 C1 链(每行 = 一个独立验收的 lever,比值为该 wave 同场实测;golden 硬闸除 G1 外全部 diff=0):

| 改动(wave) | 实测收益 | 累计战线 | 数据产物 |
|---|---|---|---|
| G1 raw 域 softmax(w10 实现/w11 验收;ae84b8b,含 G4 prefetch、G5 packed rescale) | ws/old 0.989→**1.1308**(d128 1.1518,20/20 全正:s1024 1.18-1.37、s4096 1.13-1.15、≥16k 1.10-1.13);duration@16k 16.35→14.69 ms(−10.2%);accuracy 最差 cos 0.999242 / rel_l1 0.0390,golden 切轨 `golden-sm100-g1ws` | ws/cudnn 0.8346(d128 0.8621) | `logs-w11/`,C1_DESIGN §9.6 |
| G2 k_scale 经 smem 预载(w12 实现/w14 验收;11dbbcc) | G2/G1 **1.0112**(min 0.9997);duration@16k 14.69→14.48 ms;bitwise diff=0 硬闸(纯数据源替换) | ws/old 1.1436,ws/cudnn 0.8412 | `logs-w14/`,C1_DESIGN §10.4 |
| correction ballot 全跳(w14 实现/w16 验收;65d30f7) | ballot/G2 **1.0593**(22/22 全正,min 1.0079);duration@16k 13.71 ms;diff=0 硬闸(×1.0 恒等论证) | ws/old 1.2107(d128 1.2356),ws/cudnn 0.9019;**s≥32768 段 ws/cudnn 1.0520 首破 cudnn** | `logs-w16/`,C1_DESIGN §11.4 |
| persistent CTA(w20;f84c25f + auto 非 causal fc03183) | wsp/ws d128 c0 全段 **1.084**(s1024/s4096/s16k/s32k/s131k = 1.108/1.104/1.072/1.068/1.070);序幕摊薄 12.7×(238.8K→18.8K cycles/SM @b4s4096),launch waves 13.84→1;c1 十点全负 0.881 判负,auto 维持 causal→per-tile ws;golden 三轨 diff=0 | wsp/cudnn c0 s16384/32768/131072 = 1.012/1.059/1.096 | `logs-w20/`,C1_DESIGN §13.5 |
| d64 vec_full 交付(w24;8b5f814,int max + issue wall gate 到 head_dim==64) | d64 ws/old 0.980→**1.0302**、wsp/old 1.0537(12 点);d128 8/8 实例 SASS 逐字节恒等,抽查零影响;ncu XU 54.3→56.8、eligible 0.484→0.536 | d64 组合形态 12/12 ≥1.0319 | `logs-w24/`,D64_DESIGN §8.5.3 |
| auto 放宽到 d64(w25;f428eb3,host-only) | auto/old d64 **1.0590**(12/12 ≥1.0304)、d128 1.2863(与 w20/23 带 max dev 0.35pp);路由自检 4/4、压测 3/3 | **终表 auto/cudnn:d128 c0 1.0005 / c1 0.9020,d64 c0 0.7112 / c1 0.6442** | `logs-w25/`,D64_DESIGN §8.6.3 |
| varlen sm100(w12 classic kernel ca5e4b1;w16 fence f3e617b;w17 挂死修复 75b8012 + 回 plan b78ea2a) | 挂死(~1/50-1/2000 causal launch)根因 = mask 链混进 S 排空循环,修复后 isolate 三臂(a2b/a1/a2)各 6000 全绿;bench ragged .25-1x **1.13-1.54×**、等长 0.91-0.95×(vs dense classic WS=0) | sm100 回 `_VARLEN_BACKENDS`,pytest 207+436 全过 | `logs-a3/p2_varlen_bench.*`,SM100_VARLEN_DESIGN §6.4.6 |

C1 迭代史(全档 C1_DESIGN):r1 两遍读 0.883 → r2 单遍驻留 0.947 → r3 宽 ld+f32x2 0.983 → r4 交错 0.989 → G1 1.131 → G2 1.144 → ballot 1.211 → persistent(nc)/vec_full-d64/auto-d64 → 终态 auto/old d128 1.286、d64 1.059。淘汰链同样完整:lever A 单变量 +4.8~7.1% 劣化被 G1 吸收结案;P-chunk、lazy max、wave22 三 lever 全部倒在预注册判据(§3)。

收尾会话原有条目(全部仍在默认路径生效):

| 改动 | arch | 实测收益 | bench 口径 | 数据产物 |
|---|---|---|---|---|
| P2 quant dense/varlen 拆实例(per_warp + per_thread 家族) | sm100 | quant 组合 wall **+2.2~2.3%**(quant_q +1.4~1.8%、quant_k +2.4~3.2%) | quant_qk dense,b4h32 d128 seq{4k,16k,32k},min-of-30 ×3 轮交替,B200 | `logs-w4/p2/` |
| 同上 | sm90 | per_warp Q **+4.6%**、per_thread Q +2.4~3.7%(per-block 家族拆分实测劣化已回退,见 §3) | 同口径,H200 | hyper01 `/workspace/p2-sm90/` |
| 同上 | sm89 | 无回退(±0.4%,低于信号线) | kernel_breakdown b4h32 d128,L20 | computelab `logs-w4l/kbd/` |
| P4 V 融合门限 per-arch(sm89 4096→**12288**) | sm89 | 6-12k 段 V 前处理 kernel 级 **+15~20%**(融合比分离 0.80-0.85);两路 bit 等价,golden 免重 dump | b·h {32,64,128,256} × kv {4k..24k},L20 | bench/P4_VFUSE_L20.md §5,`bench/microbench/vfuse_l20/` |
| P4 门限(sm100/110 4096→**24576**) | sm100 | 4-24k 段融合快 **13~31%**(0.69-0.87,全范围无交叉);bitcheck 12/12 等价 | 同上,B200 | `logs-w4/p4_*.json` |
| C1 r4 collective-wait 交错 | sm100 | ws/old 0.9827→0.9890(**+0.64%**,非 causal 逐形状 ~+1%) | 同 C1 口径 | `logs-w4/bench/` |

## 2. 工程收益表

| 项 | 结果 | 证据 |
|---|---|---|
| 代码整理(批 1-4) | **净 −1191 行**(45 文件,+1700/−2891):过渡 op 退役 −753、死代码 −582、base launcher −295、launcher/宏/fused 收拢 −402、kernel body 去重 −106;2005 个 kernel 与基线 SASS 逐字节全同 | scratchpad `sassd_gateB/gateC`、`build_w4` 对比 |
| fwd 加 `backend=` 覆写 | 对拍跨家族覆盖保住,五机 golden 零重 dump(RETIRED markers 兼容层) | tools/compare_reference.py |
| Track D gencode 裁剪(`SAGE_PRUNE_GENCODE`,默认 ON) | build CPU **−36.1%**、so 体积 **−38.0%**(118.7→73.6MB);敞口 = 对拍需 OFF 重建 | bench/microbench/PRESCREEN_REPORT.md §4 |
| resolve 显式 backend 干净报错 | 撞未编 SASS 由 no-kernel-image 变 TORCH_CHECK 带 cure | csrc/sageattn/plan.cpp `backend_serves` |
| 测试补全 | test_varlen_sm89/sm120 各 62 kernel 级 packed 用例全过;sm100 varlen 用例随 M2 合入(B200 全量 436 passed / 395 skipped) | test/、HARDWARE_CHECKLIST §1b |
| varlen 零开销审计 | attention 侧独立实例零开销(SASS/param 审计固化);quant 侧超阈值实例已由 P2 拆分 | PRESCREEN_REPORT.md Track D 节 |
| 显式 fmaf 值契约硬化(7a7c84d) | 把 nvcc 收缩出的 denom FFMA 写死在源码,控制流改动不再破 bit-exact;衍生硬规则 = FP opcode 逐类对账进本地闸 | C1_DESIGN §15.2,memory `sageattention-fma-contraction-contract` |
| varlen sm100 mbarrier init 发布 fence(f3e617b) | 封了 Programming Guide 要求的 init→async proxy 缺口(挂死根因另在,fence 无害保留);dense/ws/sm90 同型缺口挂账 §4 | SM100_VARLEN_DESIGN §6.4.3 |
| B200 方法论沉淀 | 节点健康门脚本、cuda-gdb from-launch+SIGINT 调试姿势、分角色 stall 归因脚本、固定形状分钟级复现臂 | `scripts-w19/`、`scripts-a3/`,HANDOFF §2 |

## 3. 判负与回退(全部有实测,防重复立项;黑名单总表另见 BEYOND_CUDNN_PLAN 附录 A)

| 方向 | 结论 | 证据 |
|---|---|---|
| E1 f16x2 softmax | 无硬件 packed MUFU 通路(四 arch SASS 均拆 2×MUFU+PRMT),sm86 实测与 f32 同速 | PRESCREEN_REPORT.md |
| P6 PDL | kernel 间 gap 全形状 ≤1.59%,PDL 可及部分 0.25µs/边 | bench/P6_PDL_PRESCREEN.md |
| sm90 process_tile 收拢 | varlen kernel 锁基频 +1.47% → revert;功耗 cap 下亚 1% 信号必须锁基频仲裁 | hyper01 `/workspace/sm90-7a64fc0/REPORT.md` |
| P2 per-block 家族拆分 | sm90 实测 plain +21%(ptxas 分支域重构杀 LDG 并行)→ dense 回退 kVarlen=true 实例 | `/workspace/p2-sm90/` |
| P8 sm80 fold | sm_86 +0.9% 不迁移:L20 0.9995、A100 0.9988;已回收 | bench/P8_SM80_FIX_{L20,A100}.md |
| C1 x128 宽 ld | ptxas C7602(函数级 128 入口 reg 检查),x64 是结构上限 | C1_DESIGN §6.3 |
| lever A ld-shadow 预取(单项) | 四态归因坐实长序列 +4.8~7.1% 劣化;G1 重写吸收,不再单独立项 | C1_DESIGN §9.6,`logs-w11/states/` |
| P-chunk(P 64 列分块交付,常量 parity) | 真机 tile 0 即 100% 死锁(两次完成/步的 wait/parity 错配),phase-alias 论证被证伪;revert 494d4f5 | C1_DESIGN §11.4,`logs-w16/` |
| A′ RESCALE_THRESHOLD 惰性 max(T=4) | bench 0.9967(判据 >1.005),6/22 <0.995;ncu correction issued 无崩落 = bench 协议 randn 刻度下命中集近空;「skip 命中率 × 分段收益」乘数外推不可作立项依据;revert 910a831 | C1_DESIGN §12.5,`logs-w18/bench_gate_corrected.txt` |
| wave22 vec_full 交付(int max + issue wall,d128 形态)| ws/old −0.8%、wsp/old −3.3%,s≥16k 不回退线破;revert 2fa877e。**d64-only gate 形态 wave24 重启并 KEEP**(8b5f814,见 §1) | C1_DESIGN §14/§15.4,`logs-w23/` |
| wave22 persist causal EX2 phase gate | wsp/ws 0.8949 < 0.90 判负线(还叠着 §14 顺风);revert b6c4569 | C1_DESIGN §13.6/§15.4 |
| wave22 d64 EX2 phase gate(moved wait 造 stagger) | 融合树 1.0269 是混杂读数;单改动复验 0.9575 < 0.96,mio_throttle 0.95→1.06、XU 53.7→51.9 反向恶化;revert d9e1afc | C1_DESIGN §15.5、D64_DESIGN §8.4,`logs-w23b/` |
| FA3 式 varlen persistent scheduler | 偏斜浪费按 KV tile 归一后 −9.0~+3.6%(判据 ≥5%),ncu 空 CTA 被硬件即时回填(idle 增量 −3.0~−0.1pp);不立项 | bench/DIVE4_SCHED_PRESCREEN.md(sm120) |
| varlen 挂死的调度差假设与 C2 族 | E5 两臂(peeled k_scale 位置 / LDTM 间距串行化)全红出局;等长/偏斜/solo 全挂 ⇒ 裸退与 CTA churn 出局;病灶 = mask 链在 S 排空区间(位置轴,非指令形态,V7/V9 佐证) | SM100_VARLEN_DESIGN §6.4.5/6.4.6,`logs-a3/` |

## 4. 挂账(明确悬置;详细版 HANDOFF §6)

- **d128 c1 残差**:causal persist 判负后 c1 走 per-tile ws,auto/cudnn s1024-16384 0.65-0.99 是当前主谷地;W=1 c1 斜率 +13% 的动态相位残差待上机 per-barrier ledger 归因(C1_DESIGN §13.6.1);`q_empty` 提前 commit 候选记档(§13.6.2)。
- **d64**:XU roofline 束缚(追平需 80-89%,演示上限 67-72%,D64_DESIGN §1.2/§7.1);G3 批量 drain 被 A5 红线封锁;C2(2 CTA/SM)唯一保留结构候选;fp8-QK d64 形态是唯一 ws/cudnn>1 路径,产品决策出界。
- **cga2(Phase C)**:启动条件 = KV 供给/L2 成新顶(BEYOND_CUDNN_PLAN §4.6)。
- **init-fence 同型缺口**:dense sm100 / sm100 ws / sm90 三处 mbarrier init 后无发布 fence,各需带 SASS gate 的独立 commit(SM100_VARLEN_DESIGN §6.4.3)。
- **A3 机理**:mask 链在排空区间引发 completion 丢失的微架构机理未定位,按禁区绕行(HARDWARE_CHECKLIST 约束表)。
- **varlen sm100 WS 版**:未立项。
- 清理终审(2026-08-30 用户拍板)不变:computelab 7 个历史构建树已删(~1.7G);hyper01 `SageAttention-rowsum` 与 pro-5k `/workspace/SageAttention`(v2g 未 push 工作)保留;证据类目录保留。
- ~~P9 / 4090 系列~~(2026-08-30 关闭:部署面只有服务器级 GPU)。
