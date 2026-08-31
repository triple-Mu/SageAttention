# D64 设计侦察:差距构成、结构候选与路线(wave18,纯本地)

日期 2026-08-31,基线 2903e45(wave16 ballot 态),分支 wave18/d64-design。
全部结论来自已有实测数据与读码,无新上机;每个数字标来源(附录 C)。

口径约定:XU = ncu 的 xu pipe(超越函数 + 数据转换),「XU 忙时」= duration ×
XU%;「同结构差距」= 与 cutedsl 分支(同 16-warp WS 结构的 CuTe-DSL 实现,七月
调优终态)之间的执行效率差;TF = TFLOPS,FLOPs 口径 4·b·h·s²·d(与 bench json
一致,已用 cudnn d128 s16384 反推核对);bench 形状默认 b4h32 non-causal。

## 一屏结论

1. **d64 输 cudnn 的根子是算术强度,不是 tile 几何**。head_dim 减半把 FLOPs 砍
   半,但 int8-QK 的 softmax XU 工作量(I2F 1 + EX2 1 + e4m3 pack 0.5 ≈ 2.5
   op/S 元素)只随 s² 走,与 d 无关——XU 忙时在 b4h32s16384 恒 ~8.0 ms,d64 与
   d128 同值。由此追平 cudnn 需要 XU 利用率 78%(s16k)/ 85-88%(s4096),而同
   形态任何实现的演示上限是 67-72%(cutedsl)。int8-QK 形态内 d64 全面超越
   cudnn 不可达;可达带 ws/cudnn ≈ 0.79(s4k)/ 0.92(s16k)(§1.2)。
2. 现状离可达带还有 33%:cutedsl 同结构 d64 s16k 793 TF,我们 596(w16 ballot
   态)。历史归因「TMEM 512 列只用 384 / tile 几何差」不成立——cutedsl 的 tile
   几何与 TMEM 用量与我们逐项相同(§1.3)。真正要抓的是同结构执行差(登记在册
   的只剩 G3/LDTM 批量读,d64 因 MMA 阴影减半吃双倍伤害)加一个未归因的 d64 反
   常:ws 每 128×128 tile-block 的耗时 d64 比 d128 还长 8%,经典 kernel 同口径
   反而快 11%(§1.4)。
3. 路线(§4):先 C0(d64 ncu 首采 + A5 根因定位解锁 G3 + A′ 顺带收益)吃同结
   构差,ws/old 转正后把 d64 翻进 auto;结构候选只保留 C2(2 CTA/SM,依赖 A′
   落袋);q_stage=4 与 CTA_K=256 纸面即可排除(§3)。ws/cudnn > 1.0 的唯一路
   径是 fp8-QK d64 形态(XU 降到 1.5 op/元素,54% 利用率即 1.15×),产品决策,
   出界挂账。
4. 生产权重(§6):d64 不是边缘需求——本仓库自家 demo CogVideoX(README 头图
   模型)与 LTX-Video 都是 head_dim 64、数万 token、non-causal,正落在 hd64 nc
   s16k 点位;但 LLM 主流与新一代视频模型(HunyuanVideo/Wan/Mochi)全在 d128。
   优先级排序维持:d128 Phase B(persistent)之后,C2 按需求启动。

---

## 1. 差距构成

### 1.1 现值(w16 同场三方,B200,cudnn 9.24)

三轮 median(轮间 spread ≤0.1%,cudnn s4096 除外:9.2%,三轮 941/938/855 TF):

| 形状(hd64 nc b4h32) | cudnn | old(经典) | ws(ballot) | old/cudnn | ws/cudnn | ws/old |
|---|---:|---:|---:|---:|---:|---:|
| s4096 | 938.1 TF / 0.586 ms | 562.6 / 0.977 | 558.5 / 0.984 | 0.600 | 0.595 | 0.993 |
| s16384 | 862.4 / 10.200 | 606.9 / 14.492 | 596.3 / 14.752 | 0.704 | 0.691 | 0.982 |
| 几何均值 | — | — | — | **0.650** | 0.642 | 0.988 |

数据:主控 scratchpad `w16/bench/{cudnn,old,wsa}-r{1,2,3}.json`(wsa = ballot
树 65d30f7),集群 `SageAttention_refactor/logs-w16/bench/`。

两处对既有口径的修正:

- 「ws 比 old 慢 7%」是 w9/w11 的 s32768 读数(4 形状几何均值 0.9312,
  C1_DESIGN §6.5;G1 后 0.908-0.933,§9.6),**ballot 之前**的状态。ballot 给
  d64 +3.3~4.4%(w16 lever 分解),两个 bench 形状上 ws/old 已到 0.98-0.99;
  s32768 在 ballot/A′ 后未复测,是 auto 翻转判断的缺口(§5 采集项)。
- launcher auto 注释(qk_int_sv_f8_cuda_sm100.cu:85-94)把 d64 排除归因为
  「only 384 of the 512 TMEM columns used」——§1.3 证伪,翻转时一并改写。

### 1.2 根子:FLOPs 减半,XU 工作量不减半

int8-QK 形态里 softmax 的 XU 工作量按 S 元素计:每元素 I2F 1(s32→f32)+
EX2 1 + e4m3 pack 0.5 = 2.5 op,只依赖 (b,h,s),与 head_dim 无关。证据:

- d128 三代实现 XU 忙时同为 ~8.0 ms @ b4h32s16384(BEYOND_CUDNN_PLAN §1.4);
- d64 直接锚点:cutedsl 七月 ncu d64s4096f XU 67.5% × 767 µs = 0.518 ms,
  与 8.0/16 = 0.50 ms 的换算差 +3%(cutedsl REPORT「诊断 1」)。

d64 的 FLOPs 是 d128 的一半,同一份 XU 忙时摊到一半的 FLOPs 上——「追平
cudnn 所需 XU 利用率」直接翻倍:

| 形状(nc b4h32) | cudnn ms(w16) | XU 忙时 | 追平所需利用率 | 我们现值(ballot) | cutedsl 同结构实测 |
|---|---:|---:|---:|---:|---:|
| d64 s4096 | 0.586 | ~0.50-0.52 ms | **85-88%** | 51-53%(0.984 ms) | 67.5%(ncu 直读) |
| d64 s16384 | 10.200 | ~8.0 ms | **78%** | 54%(14.752 ms) | 72%(11.10 ms 反推) |
| d128 s16384(参照) | 13.662 | ~8.0 ms | 58% | 58%(13.71 ms,w16 ncu) | 67%(11.92 ms) |

d128 段「追平线 58% ≈ 我们已到 58%」正是 s≥32k 翻正的原因;d64 段追平线在任
何同形实现的演示带之外(cutedsl 自己的教训:XU 62-67% 时「管线组合已均衡」,
无套利空间)。结论:

- 可达带 = 把利用率做到 cutedsl 演示值:s16k 793 TF(0.92× cudnn)、
  s4k 742 TF(0.79×)。这是结构+执行全部到位后的期望,不是下限承诺。
- 超越 cudnn 需要换形态:fp8-QK(e4m3 QK MMA 直接出 f32,消 I2F)把 XU 降到
  1.5 op/元素,忙时 ×0.6 → s16k 4.8 ms,维持今天的 54% 利用率即 993 TF =
  1.15× cudnn。这是 d64 段唯一能过 1.0 的路径,属产品级量化形态决策
  (BEYOND_CUDNN_PLAN §3 已出界),本文只记账不排期。

### 1.3 同结构差距:cutedsl 比我们快 33%,几何归因不成立

cutedsl 分支与本 kernel 的结构逐项相同(C1_DESIGN §2:16 warp、双 Q tile、
TMEM 512 布局、kv_stage=4、mma 程序序),d64 下同样只用 384/512 列 TMEM。
七月它的 d64 实测(b4h32,cudnn 同场锚点与 w16 偏差 ≤1%,附录 C):

| 实现 | d64 s4096 | d64 s16384 | d64 s32768-131072 |
|---|---:|---:|---:|
| cutedsl(七月终态) | 728.9 TF | 792.8 | 802-817 |
| ws ballot(w16) | 558.5 | 596.3 | 未测(G1 前 573-616,w9) |
| 差 | +30.5% | +33.0% | — |

同结构差距(596→793,+33%)大于剩余的结构性差距(793→862,+8.7%)。对账
(C1_DESIGN §7)登记在册的执行差只剩 G3:cutedsl 的 S drain 是 4×x32 批量发
射单次 wait(1 个暴露窗口/块),我们是 2×x64 且 ptxas 把第二条沉回消费点后
(6/8 站点,实测 2 个串行窗口/块)。d128 上这笔账值 ~13%(1289 vs 1474);
d64 的 MMA 阴影减半,同一笔暴露延迟伤害加倍——与 30-33% 的差距量级吻合,但
G3 与编译器调度各占多少未归因(§5 采集项)。

### 1.4 d64 反常:ws 每 tile-block 比 d128 还慢

归一化口径:SM·µs / (128 Q 行 × 128 KV 列) tile-block =
duration × 148 ÷ (q_tiles·b·h·kv_blocks),s16384 nc b4h32(分母 2.097e6;
per-CTA 推导见附录 B)。d64 每 tile-block 的工作量严格少于 d128(softmax 同,
MMA/correction/epilogue 减半),比值理应 <1:

| 实现 | d128 | d64 | d64/d128 |
|---|---:|---:|---:|
| cudnn | 0.964 µs | 0.720 µs | 0.75 |
| 经典 kernel(2 CTA/SM) | 1.149 | 1.023 | 0.89 |
| cutedsl(七月) | 0.842 | 0.783 | 0.93 |
| **ws ballot** | **0.963** | **1.041** | **1.08** |

ws 是唯一比值 >1 的实现:d64 比 d128 每单位工作**多花** 8% 时间。候选解释
(本地无法裁决,§5 的首要问题):correction/epilogue 工作减半后 warp 空转,
eligible warps 掉档;PV 的 N=64 UMMA 每指令周期未随 N 减半(固定开销);
TMEM 口在 384 列集中脚印下的争用变化。经典 kernel 无此反常(0.89),说明这
是 WS 结构 × d64 的特有交互,不是形态必然。

---

## 2. cudnn 的 d64 侧写

- **绝对值不高**:cudnn d64/d128 = 0.67-0.69(w16:862/1288、938/1366)。
  d64 的算术强度减半打击所有实现,cudnn 也一样(它的 fp16 形态在 d128 是
  tensor 占 82% 的形状,d64 下 tensor 工作减半、softmax 不减)。目标线因此比
  d128 低三成——这是 d64 段唯一的好消息。
- **目标线在移动**:d64 s4096 cudnn 七月 845.5 TF → w16 938.1(median,+11%;
  但 w16 三轮 941/938/855 有 9% 抖动,一轮与七月持平)。s16384 稳定
  (869.3 → 862.4,−0.8%)。验收会话必须同场重测 cudnn,不引历史值
  (与 d128 段既定规则相同)。
- **DSL kernel zoo 没有 d64**:nvidia-cudnn-frontend 1.27.0 的
  `cudnn/sdpa/fwd/kernels/` 只有 d128(Llama 类)、d192_d128(DSv3 MLA)、
  d256(Qwen 类)、d512(DSv4 类)五个 flavor——cudnn 团队的 DSL 产品线不为
  d64 投资,d64 由闭源 C++ kernel 兜底。含义有二:没有现成源码可抄
  (d128 的 §2 开箱情报对 d64 失效);对手在 d64 上也只是「够用」水平。
- **闭源 d64 kernel 名/几何未知**:w14 只采到 d128 的名字
  (`..._128x128x128_4x1x1_cga1x1x1_...`,tile 几何编码在 knob 段)。d64 的
  名字、tile 形状、是否 persistent/cga 都待采(§5)。
- **横向对照**:FA4 fp16 d64 nc 只有 710-735 TF——低于 cutedsl 的 793-817。
  七月场 d64 nc 的排序是 cudnn 866 > cutedsl-sage 793-817 > FA4 730 > 我们
  596。我们的 d64 差距主要输给自己的结构上限,不是输给什么 d64 专属魔法。

---

## 3. 结构候选(资源账与判定)

黑名单对照见附录 A;寄存器/SASS 门禁口径同 C1_DESIGN §5。

| 候选 | 机制 | 收益上限 | 工作量 | 判定 |
|---|---|---|---|---|
| C0 同结构追赶(ncu 首采 + A5 根因 → G3/交错 + A′ 顺带) | 消 S drain 暴露窗口、correction 近全跳 | +10~25%(硬上限 = 同结构差 +33%,793 TF) | 小-中 | **立项,先行** |
| C1a q_stage=4(CTA 512 行,TMEM 全 512) | 序幕/尾声摊薄 ×2、KV L2 减半 | +0~5%(XU 发射侧无新机制) | 大 | 不立项 |
| C1b CTA_K=256(K tile 加倍) | 每块固定开销减半 | — | — | 纸面排除(寄存器) |
| C2 2 CTA/SM(256 线程 8 warp,correction 融入 softmax) | SMSP 层两条独立 softmax/mma 流交错 | ≤+35%(cutedsl 估) | 很大 | 保留,条件启动 |
| C3 经典 kernel fork + G1/ballot 移植 | 已有 2 CTA/SM,补串行链 | +7~19%(650-720 TF,仍低于 ws+C0 可达带) | 中-大 | 不立项 |
| fp8-QK d64 形态 | XU 1.5 op/元素 | 1.15× cudnn(§1.2) | 产品决策 | 出界挂账 |

### 3.1 C0:同结构追赶(推荐先行)

1. **d64 ncu 首采**(零风险,§5):裁决 §1.4 反常与 G3/编译器归因,后续一切
   立项的证据基础。w11/w14/w16 的 ncu 全部只有 d128(附录 C),d64 从未采过。
2. **A5 根因定位 → G3(4×x32 批量 S drain)**。A5 红线(4+ outstanding
   tcgen05.ld 挂死,根因未定位)的黑名单条目自带翻案条件「根因定位」;wave16
   已把可靠的死锁调试姿势记档(cuda-gdb from-launch + SIGINT 打断,attach 在
   ComputeLab 拿不到 CUDA 态,C1_DESIGN §11.4)。做法:probe TU 复现 A5(批量
   4×x32 单 wait),outstanding 2→3→4 二分,断点态读各 warp PHASECHK。定位成
   功 → G3 落地(cutedsl 同款,每块 1 个暴露窗口);失败 → 退而求其次
   sm_100a 交错 lever:在两条 x64 ld 与首个消费点之间填独立工作(mask/scale
   预计算重排)强制真 in-flight,2 outstanding 不越 A5 边界,但要重做
   bit-exact 论证(C1_DESIGN §6.4 后续 lever 段的预案)。
3. **A′ lazy max 顺带收益**(wave17 已实现,验收本就排在下一 B200 会话)。
   d64 的 correction 流量是 d128 的一半,ballot 实测给 d64 +3.3~4.4%
   (d128 +3.9~6.6%),A′ 把命中率推到 ~100%,d64 预计再 +1~3%。
4. **auto 翻转**:ws/old 两形状已 0.98-0.99,A′ + C0 后预计全面 ≥1 →
   `sm100_ws_auto_pick` 把 d64 翻进 ws,d64 用户立得 ws 相对经典的全部后续
   收益。判据需要 s32768 d64 复测(§5),翻转时同步改注释(§1.1)。

### 3.2 C1a:q_stage=4(不立项)

CTA 覆盖 4×128 行,TMEM 用满:S 2×128(4 tile 共享 2 slot,tile t 用 slot
t%2)+ O 4×64 = 512 列;vec/P 仍叠 S 内。资源账全部过得去——寄存器预算不变
(softmax/correction 每 warp 串行处理 2 个 tile,行缓冲复用,状态 ×2 约 +6
reg,余量够);barrier 8 族 ×4 + kv 8 + dealloc = 41;smem d64 = sQ 32K +
ring 32K + sO 64K = 128KB ≤ 227KB;grid.x = qo_len/512。

但收益机制经不起推敲:softmax 还是 8 个 warp、每 warp 对每 tile 严格串行
(行驻留 128 reg 决定了 tile 间无寄存器空间做 ILP),XU 发射结构与今天逐拍
相同;S slot 共享让 QK(tile t+2) 依赖 PV(tile t) 退役(P 叠 S 别名),流水
解耦弱于表面。剩下的只有序幕/尾声摊薄 ×2 与 KV L2 流量减半——前者是短序列
项(且 grid 再减半让 b1 尾波更糟,w9 已见 b1 对 wave quantization 敏感),
后者在 DRAM ~10% 的画像下无从兑现。预期 +0~5%,工作量大(账本重写、mma 轮
转重排),挂死风险与 P-chunk 同类(wave16 的 barrier 改造首块即死锁的前车之
鉴)。判定:不立项;若未来 ncu 显示 d64 卡序幕/KV 供给再议。

### 3.3 C1b:CTA_K=256(纸面排除)

S 行驻留是 softmax 结构的地基:RS_f32[CTA_K] = 256 寄存器 > 192 预算,必然
退回两遍 TMEM 读——round 1 实测 0.887× 的老路(C1_DESIGN §6.1,tcgen05.ld
流量 1.75× 暴露延迟);TMEM 侧 S0 变 256 列后 S0+S1+O 超 512,双 tile 结构
也放不下。且 XU 工作量与分块无关(2.5 op/元素不变),每块固定开销的减半换
不回这两笔。判定:排除,不再评估。

### 3.4 C2:2 CTA/SM,256 线程 8 warp(唯一保留的结构候选)

cutedsl REPORT 遗留的 d64 结构候选(「q_stage=1 + 2 CTA/SM,交错掩盖 XU 等
待,预期 ≤+35%,改动大」)在 512 线程 CTA 下不可行(寄存器 ×2 超 RF),可行
形态是 256 线程:

- **TMEM**:每 CTA S 128 + O 64 = 192 → alloc 256(幂次),2 CTA = 512 恰满。
- **寄存器**:4 softmax @192 + 4 其余 @40 = 29696 ≤ 32768/CTA,余 3072
  (softmax 可升到 208 或留给 miss 路径)。
- **warp 角色的硬约束**:tcgen05.ld/st 只能触达本 warp 的 lane 象限
  (warp id%4,PTX 9.7.17.8.1)——128 lane 的 correction 必须恰好 4 个 warp,
  8-warp CTA 里放不下独立 correction 组。解法:**correction 融入 softmax**。
  依赖 A′ lazy max:rescale 近恒跳,softmax warp 自己 vote,miss 时(randn
  ~0-2%,对抗 ramp 最坏 24%,lazy_max_sim §12.2)自修 64 列 O(2×x32 ld +
  FMUL2 + st 骑在自己 lane 象限上,行归属天然对齐)。correction/vec 两族
  barrier 整体删除;epilogue 数学与 sO staging 也归 softmax,warp 4-7 做
  mma/load/epi/spare(sched 位为 persistent 预留)。
- **smem**:每 CTA sQ 8K + ring 32K + sO 16K = 56KB,2 CTA 112KB。
- **收益机制**:两个独立 CTA 的 softmax/mma 流在 SMSP 层交错——CTA A 卡
  LDTM 回程时 CTA B 发 XU,这是今天 1 CTA/SM 结构给不了的真并发,也是经典
  kernel d64 每 tile-block 反而比 ws 快的机制来源(§1.4 表)。
- **风险**:双 CTA TMEM alloc 握手有前科(HARDWARE_CHECKLIST §5f triage 清
  单;经典 kernel 的 alloc→relinquish 序是现成解法,p0b 型 probe 先行);
  miss 路径把 rescale 搬上 softmax 关键路径,对抗性输入(ramp 类)有尾部劣
  化,精度/性能耦合要在 sim 里扩场景;16 warp→8 warp 的账本全部重写。
- **黑名单对照**:与 sm90 判过不可行的 cluster TMA multicast 是不同机制
  (无跨 CTA 数据共享,两 CTA 完全独立);不触 A5;correction 融合无前科。

判定:收益上限最高的结构项,但依赖 A′ 落袋 + C0 的 ncu 归因先行(若 §1.4
反常另有根源,+35% 的估计要重算)。启动条件见 §4 M3。

### 3.5 C3:经典 kernel fork + 移植(不立项)

经典 kernel d64 已有 2 CTA/SM(TMEM 右尺寸化 143a027 的遗产)但数学还是
G1 前状态(串行 max fold + 串行 d_sum,链深 ~200 float op/块;correction 无
条件全跑)。把 G1/ballot/lazy 移植过去能拿 +7~19%(估 650-720 TF),但:
经典 TU 是全仓 SASS 逐字节对拍的 oracle,动它必须 fork 新 TU + 新 golden
轨;天花板仍在 ws+C0 可达带(793)之下;两个性能载具并存违背 C1 立项以来
「ws 是唯一性能方向」的维护策略。判定:不立项。它的价值是做 C2 的活参照
(2 CTA/SM 的 eligible/issue 画像,§5 采集项)。

### 3.6 微候选(记档,不单独立项)

- **S 第三缓冲**(d64 空置的 [384,512) 列做 S ping-pong):稳态下 softmax 是
  落后方、s_full 等待近零,纸面收益≈0;且 TMEM 腾挪在 128 线程地基有 0.6508
  的先例(边界:仅限该地基)。翻案条件:ncu 显示 softmax 卡 s_full。
- **vec 双缓冲**(同一片空列):C1_DESIGN §6.3 定位顺序里的既有条目,条件
  「correction 卡 vec」,d64 ncu 顺带看。
- CTA_Q=CTA_K=64 + 4 CTA/SM:M=64 MMA 浪费半个 tensor datapath、TMEM alloc
  128 列(S 64 + P 叠 + O 64)可行,但 64 线程 CTA 的 warp 角色摆不开,收益
  机制与 C2 重复而资源账更差。不展开。

---

## 4. 推荐路线与 milestone

**总判断**:d64 段的合理目标不是超越 cudnn(形态内不可达,§1.2),而是
(a)把 ws d64 从 0.64 抬进 0.8+ 可达带,(b)让 auto 在 d64 上也交付 ws,
(c)把 fp8-QK d64 的换形态账递给产品决策。投入排序仍在 d128 Phase B
(persistent)之后——d128 覆盖的模型面更大(§6),且 persistent 的
scheduler 基建未来对 d64 同样生效。

| milestone | 内容 | 验收判据 | 依赖 |
|---|---|---|---|
| M0(下一 B200 会话,与 A′ 验收同场,~1.5h) | §5 采集清单全跑 | 数据齐(d64 三方 ncu、cudnn d64 kernel 名、s32768 复测、XU 忙时直测) | 无 |
| D1 决策(读数后) | 裁决:§1.4 反常归因;G3 vs 编译器占比;8.0 ms 不变量 d64 直测是否成立(不成立则本文 roofline 结论重开);A′ 后 ws/old 是否全面 ≥1 | 归因写回本文 §1 | M0 |
| M1 | C0.2:A5 复现定位 → G3 或交错 lever | d64 两形状 ws/old ≥1.05,或 ncu softmax long_scoreboard 减半;d128 不劣化(同 lever 双段共享) | D1 |
| M2 | auto 翻转 d64 + launcher 注释改写 | d64 全 sweep(含 s32768 b1/b4 causal 两态)ws/old ≥1,auto 抽查同 w11 口径 | M1(或 A′ 单独达标) |
| M3(条件启动) | C2:256 线程 2 CTA/SM | 先 p0b 型双 CTA alloc probe → 单形状冒烟 → golden/压测/22 点全套 | A′ 落袋 + D1 归因支持 +35% 估计 + 产品侧确认 d64 需求 |

不做清单:C1a、C1b、C3、软件 exp2 / f16x2(黑名单维持)、S 第三缓冲与 vec
双缓冲(等 ncu 证据)。fp8-QK d64:挂产品决策账,本路线不排期。

---

## 5. 下一 B200 会话采集清单(交付物)

与 A′(wave17)验收同场跑,新增预算 ~1h。全部命令模板沿用 w14
(`bench/sm100_review/ws_prof.py` + `--set full --section PmSampling
--section PmSampling_WarpStates -s 2 -c 1`,ncu 锁频口径)。

### 5.1 ncu(6 发,全部 hd64 nc b4h32)

| # | 引擎 × 形状 | 回答什么 |
|---|---|---|
| 1 | ws s16384(`SAGEATTN_SM100_WS=1`) | XU 忙时直测(预期 ~8.0 ms;显著偏离则 §1.2 重算);分角色 stall 对照 #2 |
| 2 | ws s4096 | 中段谷地画像;与 d128 s4096(w14 已有)对照 |
| 3 | old s16384(`=0`) | 2 CTA/SM 的 eligible/issued/stall 画像,C2 的 +35% 实测锚 |
| 4 | cudnn s16384 | 限制器(tensor? XU? issue?)、launch 三件套(grid vs tile 数 → persistent?;cluster_size → cga?;reg/smem) |
| 5 | cudnn s4096 | 同上 + s4k 抖动(w16 三轮 9%)复核 |
| 6 | ws s16384 **d128 同发对照**(可复用 w16 的 ballot 采集,若容器/频率同场则免) | §1.4 的 d64/d128 每 tile-block 比值在 ncu 锁频口径下复算 |

判读点(按优先级):

1. **§1.4 反常归因**:PmSampling_WarpStates 分 warp 角色(softmax/corr/
   mma/epi)对照 d64 vs d128——correction/epilogue 减半后是 not_selected 涨
   (warp 空转掏空 eligible)还是 softmax long_scoreboard 绝对值涨(TMEM 口
   争用变化)。
2. **tensor pipe active 是否随 d 减半**:不减半 ⇒ N=64 UMMA 有固定开销
   (PV 4×N64 步的隐性损耗,q_stage 双 tile 结构的 d64 税),C2 设计要计入。
3. **G3 vs 编译器占比**:softmax 视角 long_scoreboard/issue 与 d128 同指标
   的比值;若 d64 显著更高,G3 类 lever 的 d64 弹性坐实。
4. **cudnn d64 kernel 名与几何**:torch profiler(w14 的
   `cudnn_kernel_names.txt` 流程)抓名字,knob 段读 tile 形状(d128 是
   `128x128x128_4x1x1_cga1x1x1`);grid 数对比 tile 数判 persistent。
5. M4/M6 类既有问题不重复采(w14 已裁)。

### 5.2 bench 补充(与 A′ 验收 bench 合并跑)

- d64 **s32768 × causal 两态 × b{1,4}**(w9 网格的 d64 行),A′ 树与 old、
  cudnn 三方同场——auto 翻转(M2)的判据数据。
- 22 点常规回归照旧(A′ 验收本身要求);d64 两点单列读数。

### 5.3 顺带

- cudnn DSL 无 d64 flavor(§2),不安排 DSL 对表实验(d128 的 §7.4 计划照旧)。
- 若 A5 复现 probe 已备好(M1 前置),同场跑定位;不阻塞采集主线。

---

## 6. d64 生产权重

结论先行:**d64 是「仓库自家 demo 在用、行业增量在别处」的段**——不挂零权
重,也不升主战场;C0(与 d128 共享的 lever)照做,C2 等需求信号。

- 仓库自证:`example/cogvideox_infer.py`、`parallel_sageattn_cogvideo.py`、
  `example/ltx_infer.py`;README 头图对比就是 CogVideoX1.5-5B。CogVideoX 全
  家(2b/5b/1.5)attention head_dim = 64(diffusers config
  attention_head_dim=64),LTX-Video 同为 64。负载形态:视频 DiT,
  non-causal,单次生成 attention 序列数万 token(随时长/分辨率,~1.7-4.5 万
  量级)——正是 bench 的 hd64 nc s16384 点位。对这类用户,d64 段今天交付的
  是 0.65× cudnn 的经典 kernel。
- 行业面:新一代视频模型 HunyuanVideo/Wan/Mochi/Flux 全是 d128;LLM 主流
  d128+(Llama/Qwen 128,DSv3 MLA 192/128),head_dim 64 的 LLM 只剩少数
  (如 gpt-oss 族)。cudnn 自己的 DSL 投资清单(§2)是同一判断的旁证:
  d64 不在其四个 flavor 之列。
- 判断含义:若 CogVideoX 类负载被确认为重点支持对象,M3(C2)的启动条件即
  成立;否则 d64 停在 M2(auto 翻转 + 同结构追赶的搭车收益)是合理稳态。

---

## 附录 A:黑名单对照(立项前必查;总表在 BEYOND_CUDNN_PLAN 附录 A)

| 本文候选 | 相关黑名单条目 | 对照结论 |
|---|---|---|
| C0.2 G3 批量 S drain | A5(4+ outstanding tcgen05.ld 挂死,根因未定位) | 合规路径:黑名单自带翻案条件「根因定位」,M1 做的正是定位;定位失败则只做 2-outstanding 交错(不越界) |
| C0.2 交错 lever | ptxas 复沉(r4/r5 两轮实证,源级不可控) | 已知限制,方案本身即针对它(填独立工作);bit-exact 论证重做 |
| C1a q_stage=4 | 无直接条目;P-chunk(wave16 首块死锁)为同类 barrier 改造前科 | 不立项主因是收益机制缺失,前科佐证风险 |
| C1b CTA_K=256 | 两遍 TMEM 读 0.887×(C1_DESIGN §6.1) | 寄存器账必然退回两遍读,视同已判不可行 |
| C2 2 CTA/SM | sm90 cluster TMA multicast(判不可行,sm90 专属);双 CTA TMEM alloc 卡死(probe 前科,§5f triage) | 机制不同(两 CTA 独立,无数据共享);alloc 序有现成解法 + probe 先行 |
| C3 经典 fork | C2 TMEM 腾挪 0.6508(128 线程地基)、CTA_Q=128 单 tile 变体(cutedsl 侧) | 不立项与黑名单无关(维护策略);fork 才能绕 SASS 恒等门禁本身就是成本 |
| 软件 exp2 / f16x2 EX2 | 全线劣化(七月)/ XU 计数平手 + M6 已裁 | 维持封存,d64 不例外 |

## 附录 B:per tile-block 归一化(§1.4)

定义:SM·µs per tile-block = duration_ms × 148 ÷ N_unit × 1000,
N_unit = (s/128)·b·h·(s/128)。b4h32 s16384:N_unit = 2,097,152。等价的
per-CTA 推导:ws 与经典在该形状下 wave 数同为 55.35(ws 8192 CTA × 1/SM,
经典 16384 CTA × 2/SM),除法后与上式一致。自检:ws d64/old d64 的每
tile-block 比值 1.041/1.023 = 1.018,与 bench 的 ws/old 0.982 互为倒数。

## 附录 C:数据来源

- w16 bench(本文 §1.1/§1.4 主数据):主控 scratchpad
  `w16/bench/{cudnn,old,wsa,wsg2}-r{1,2,3}.json`;集群
  `SageAttention_refactor/logs-w16/bench/`。arm 释义:wsa = ballot-only 树
  65d30f7(wave16 落袋态),wsg2 = G2 基线 a88057d。三轮 median。
- w16 lever 分解与 ncu(d128):C1_DESIGN §11.4;`logs-w16/ncu/`。
- w14 ncu(d128 四形状 + cudnn kernel 名):`logs-w14/ncu/`,主控
  `w14/logs/ncu/`。**d64 的 ncu 三个 wave 都没有**(w11 仅
  g1_ws_s16384_c0,w14 仅 d128 8 发,w16 仅 ballot d128 一发)——§5 由此。
- XU 忙时不变量:BEYOND_CUDNN_PLAN §1.4(d128 三实现 8.0 ms);d64 锚点
  cutedsl REPORT「诊断 1」(d64s4096f XU 67.5%,基线 767 µs)。
- cutedsl 七月四引擎 32 点大表与 d64 结构候选:
  `git show eb1d3e7:profile/sm100-tuning-baseline/REPORT.md`(cudnn 锚点
  跨场偏差:d64 s16k −0.8%,d64 s4k +11%,§2)。
- w9/w11 的 d64 s32768:C1_DESIGN §6.5 表(geomean 0.9312)/ §9.6
  (0.908-0.933);原始 `logs-w9/bench`、`logs-w11/gap`。
- ws kernel 结构与预算:`csrc/qattn/qk_int_sv_f8_cuda_sm100_ws.cu`
  (TMEM 布局 :504-511,寄存器预算 :497-502,barrier 账 :588-601);
  经典 kernel:`csrc/qattn/qk_int_sv_f8_sm100_impl.cuh`(TMEM 右尺寸化
  :218-227);auto 注释:`csrc/qattn/qk_int_sv_f8_cuda_sm100.cu:85-103`。
- cudnn DSL flavor 清单:本机 torch env
  `site-packages/cudnn/sdpa/fwd/kernels/`(nvidia-cudnn-frontend 1.27.0)。
- 经典 kernel 2 CTA/SM 与占用画像:`test/HARDWARE_CHECKLIST.md` §5f。
