# C1 设计笔记:sm100 完整 warp specialization + 双 Q tile

对象:`csrc/qattn/qk_int_sv_f8_cuda_sm100_ws.cu`(新 kernel,`SAGEATTN_SM100_WS=1`
选入,默认走现 128 线程 kernel)。蓝本 `cutedsl-sage-sm100:cutedsl_sage/core_sm100.py`
(B200 实测 2.31× 的 16-warp 结构),judged-负的方向(128 线程地基上的 B1 KV ring
加深、C2 TMEM 腾挪,test/HARDWARE_CHECKLIST.md:1164-1166)不再触碰——本设计是
地基整体换掉,不是增量。

barrier 全账本、死锁自由与 TMEM 别名 hazard 证明在
`bench/sm100_review/barrier_ledger.md` 的 16-warp 章节;本文只放架构、对照、
milestone 与风险。

## 1. 架构

```
grid = (ceil(qo_len/256), num_qo_heads, batch)      1 CTA = 512 线程 = 16 warp
CTA 覆盖 Q 行 [256*bx, 256*bx+256) = tile0 + tile1(各 128 行)

  warp 0-3   softmax0   S0 拉取/在线 softmax/P0 打包        setmaxnreg 192
  warp 4-7   softmax1   同上,tile1                          setmaxnreg 192
  warp 8-11  correction O0/O1 rescale + 末段 epilogue 数学    setmaxnreg 88
                        (r5 起末段写 sO staging,见 §6.6)
  warp 12    mma        单 elect 线程发全部 tcgen05.mma       setmaxnreg 40
  warp 13    load       单 elect 线程发全部 TMA load          (同 wg3)
  warp 14    epilogue   TMA store sO(r5 lever B 落地)       (同 wg3)
  warp 15    empty

TMEM(恒 alloc 512 列;lane = tile 内行号,两 tile 复用全部 128 lane):
  S0[0,128)  s32          vec0@[0,2) f32 复用   P0@[32,64) e4m3 复用
  S1[128,256)             vec1@[128,130)        P1@[160,192)
  O0[256,256+HD) f32      O1[256+HD,256+2HD)    (d64 时 [384,512) 空置)

smem(dynamic):sQ ×2 tile + K/V 共用 4-slot ring(item n -> slot n%4,
  K_i=item 2i / V_i=item 2i+1)+ sO ×2 staging tile(r5,每 tile
  head_dim/64 个 128 行 ×64 列 box,TMA 128B swizzle 布局),d128 共
  160KB / d64 80KB;static:mbarrier 数组 bars[25] + tmem 槽 + sV_scale。

数据流(每 KV 块 i,双 tile 交错;mma warp 的发射序):
  QK0(i) -> [PV1(i-1)] -> QK1(i) -> PV0(i)
  causal 差异化 trip:trip0=min(2bx+1,kblk), trip1=min(2bx+2,kblk);
  多出的 (trip1-trip0) 轮只跑 PV1+QK1,尾部收一发 PV1(trip1-1)。
```

qo_len 尾块:tile1 全越界时由 TMA OOB 零填充跑白算(有限值,不落存储,
q_scale 块索引 clamp 防越界读),与 cutedsl 同策略。

## 2. 与 cutedsl 逐项对照

| 项 | cutedsl(core_sm100.py) | 本 kernel | 理由 |
|---|---|---|---|
| 16-warp 分工 | L141-148 | 照抄 | — |
| 双 Q tile / grid.x /256 | L138, L368-370 | 照抄 | — |
| TMEM 512 列布局 | L158-167 | 照抄(vec/P 复用 S 区同偏移) | — |
| kv_stage=4 ring | L178 | 照抄(K/V 交错共 ring) | — |
| 差异化 causal trip | L395-401 | 照抄公式 | — |
| mma warp 程序序 | L621-757 | 照抄(prologue/主循环/S1-only/尾) | — |
| setmaxnreg | 192/192/96/32(L169-172) | **192/192/88/40** | 实测:correction 区 ptxas 上限 77-78 reg(88 下余 10);裸 mbarrier 版 mma/load 需 ~38,32 时 ptxas 把溢出压栈(24B spill)。总和仍恰 64K,kernel 内 static_assert 锁死 |
| mma_corr pipeline | 共享 2-stage ring,o0/o1 交替 | 每 O tile 专用 1-stage ×2 | S1-only 轮打破交替后共享 ring 的 slot=item%2 错位;专用 pipe 语义等价(账本 §1)且 phase 记账是纯标量 |
| mma_s 尾部 dummy commit | L745-746(softmax 末段多等一拍) | 无 | 本实现 softmax 末段不等 mma_s,计数天然平衡(账本 §3);少两次 commit |
| softmax 数学 | raw 域 tree-max + packed FMA/exp2 + 0.5 种子 packed 求和 + P448 常量域 | **G1 已对齐**(wave10,§9):raw 域 tree max + 域折叠 fma + d_sum 4 链;row_max/vec/o_scale 仍与旧 kernel 逐位一致,P/denom 变位 | golden 口径自 G1 起切换为 accuracy gate(§9);M0-r5 的 bit-exact 硬闸条款仅存于历史章节 |
| softmax 寄存器策略 | 整行 128 f32 驻留 | 照抄:单遍读、整行驻留(见 §3;初版两遍读实测 0.887× 已回退,见 §6.1) | 行驻留 128 reg + 状态 ~30,ptxas 收在 170-173/192、零 spill(§6.3);每 KV 块每线程 2 次 64 列 tmem_ld(round 2 是 4 次 x32,两遍版 7 次) |
| vec 提前信号(max 后、exp2 前发) | L1104-1109 | 照抄(全部 S 读已完成,别名列 [0,2) 已死) | correction 的 O rescale 藏进 softmax 的 exp2 段,是重叠收益主源 |
| k_scale smem 预载(sKScale) | L544-552(per-block gran) | 不搬;r5 lever A 改为寄存器预取(下块 k_scale 在本块 ld shadow 里 LDG,§6.6),smem 预载仍开放(§7 G2) | 我们默认 per-thread gran(每 tile 4 标量);cutedsl 注释记载 gmem 广播读曾被 ncu 判为 long_scoreboard 主因之一 |
| q_scale 粒度 | BLKQ=64/WARPQ=16 契约 | 现 kernel 的 BLKQ=128 契约,块索引改 2bx+tile + clamp | 量化侧零改动 |
| epilogue | sO smem + TMA store(epilogue warp) | **r5 起对齐**:correction 写 sO(128B swizzle)+ fence.proxy.async + epi_full,warp 14 TMA store(账本 §8 as-built;数值经 TMA 逐字节搬运,bit-exact 保持) | M0-M2 曾直接寄存器->global(:567-599 序);r5 lever B 换搬运路径 |
| LSE | 无 | 从现 kernel 搬(softmax 侧存,:601-608 序) | 功能奇偶性 |
| PV_FROMSMEM SS twin | 无 | 不搬(TS only) | oracle 用旧 kernel(开关不设时)承担 |
| USE_SEQ_GATE(S0/S1 MUFU 错拍闸) | 默认关(L119) | 不实现 | 蓝本自己默认关;翻开条件见蓝本注释 |
| pipeline 抽象 | cutlass.pipeline 对象 | 裸 mbarrier + 显式 phase | C++ 侧无该抽象;账本承担正确性论证 |

## 3. bit-exact 论证(M0-M2 硬闸)

浮点运算序全部从 `qk_int_sv_f8_cuda_sm100.cu` 逐行拷贝并在代码里标注源行号
(:412-474 softmax、:454-458 在线更新、:479-491 O correction、:567-599
epilogue、:601-608 LSE)。**选拷贝不选抽共享 inline**:两 kernel 的线程结构不
同(行归属、barrier 驱动、exp2/pack 融合),抽取必然带参数化改写,做不到"旧
TU 预处理文本等价";拷贝 + 行号注释 + golden 双 gate 反而是可核对的。已验证:旧 TU 在
`-DSAGE_SM100_DEVICE_ONLY` 下预处理文本与 45aadd7 逐字节一致(env 开关全在
host 侧),SASS 恒等由此保证。

与旧 kernel 的三处结构差异,均为值恒等:

1. **softmax exp2 与 pack 融合**。load/dequant/mask/max 循环与旧 kernel
   :407-449 完全同构(单遍读、整行 RS_f32 驻留);差异只剩 exp2 段:旧
   kernel 先整行 in-place exp2 再单独一轮 pack,新 kernel 每 4 列算完
   p[0..3] 立即 `floatx4_to_e4m3x4` pack(压 RS_f32+RP 并存的峰值)。恒等
   理由:p 的 exp2 表达式同输入同序,d_sum 单累加器升序不变,pack 是逐 4
   元素独立转换,先后不影响任何值。
   (历史:初版为压寄存器采用两遍 TMEM 读 + chunk0 驻留,B200 实测几何均值
   0.887× 回退,归因 tcgen05.ld 流量翻倍暴露延迟,已改回单遍,见 §6.1。)
2. **o_scale 经 TMEM 转手**。vec=(m_prev,row_max) 两个 f32 位保真进出
   TMEM,correction 里 `ptx_exp2(m_prev-row_max)` 与旧 kernel 同表达式同输
   入 → 同位。denom 同理(最终 vec),`ptx_rcp` 同位。
3. **epilogue 折叠双数组**。旧 kernel RO_u32[32]→RO_f32[32] 两数组;新
   kernel 逐元素对折叠(mul d_rcp、mul v_scale、cvt 的单元素序不变,元素间
   无依赖)。动机:双数组顶破 correction 88 reg 预算(实测 spill)。

## 4. 寄存器预算(实测,nvcc 13.3 / sm_100a)

| 区 | 预算 | ptxas 区内峰值(4 实例最大) | 余量 |
|---|---|---|---|
| softmax(warp 0-7) | 192 | 173(round 2 单遍驻留 189;round 3 f32x2 pack 后 170-173,§6.3) | 19 |
| correction(warp 8-11) | 88 | R77 → 78 | 10 |
| mma/load(warp 12-15) | 40 | 线性段混入冷块不可直读;40 下 0 spill,32 下 24B spill | ~2-6 |
| 合计 | 2·192+88+40 = 512/线程列 ×128 = 64K | — | 恰满 |

softmax 区细账(单遍驻留版):行驻留 128(round 3 起是 uint32_t 数组,
raw→f32 原位覆盖,LDTM in-flight 块与驻留行共用)+ RP_u32 打包 32(pack
每出 1 word 杀 4 个行元素,联合活跃度自 max 点后单调降)+ 标量状态
(row_max/denom/相位/4 个 barrier 地址/tmem_row/scale 组)~30,强制平台
≈ 158,其余是 ptxas 调度自由度;f32x2 的寄存器对约束反而帮 ptxas 收敛
(189→170-173)。cutedsl 同形(整行 f32 + 32 P word)同样收在 192
(其账面 ~190)。

偏离任务书的 192/192/96/32 的原因:zero-spill 门禁与 32-reg mma/load 区冲突
(裸 mbarrier 记账 + 描述符循环不变量 ~38 live)。88/40 调剂后四实例
(d64/d128 × causal+lse+perthread / nonc+perwarp,fp16/bf16)在 sm_100a 与
sm_110a 全部 0 spill、0 stack、入口 128 reg。SASS 验证
`USETMAXREG TRY_ALLOC 0xc0 / DEALLOC 0x58 / DEALLOC 0x28` 各 4 处。

编译期三个硬前提(全部进了 probe/静态断言):

- `__launch_bounds__(512, 1)`:缺 minBlocks 时 ptxas 报 C7508
  “'setmaxnreg' ignored”,预算全废。
- 本地数组禁止运行时下标(phase 数组、指针选择的寄存器数组必 spill);
  barrier 全放进单一 `bars[23]` 数组,各角色一个 u32 基址 + 立即数偏移。
- mbarrier/TMA 一律 32 位 shared-space 地址(`ws::` 封装):generic 指针每次
  cvta 会把 64 位 generic 基址拖过 setmaxnreg 边界,低预算区必 spill。

## 5. 编译门禁(全过;复现命令)

```
# ws kernel 门禁 TU(4 实例;sm_110a 同样跑)
nvcc -std=c++17 -O3 --use_fast_math -cubin -Xptxas -v \
     -gencode arch=compute_100a,code=sm_100a -I csrc/qattn \
     -o /tmp/ws_probe.cubin bench/sm100_review/qk_int_sv_f8_cuda_sm100_ws_probe.cu
# M0 点火 probe(host main 上机直接跑,PASS/FAIL)
nvcc -std=c++17 -O3 -cubin -Xptxas -v -arch=sm_100a          -o /tmp/p0a.cubin bench/sm100_review/p0a_setmaxnreg.cu
nvcc -std=c++17 -O3 -cubin -Xptxas -v -arch=sm_100a -I csrc  -o /tmp/p0b.cubin bench/sm100_review/p0b_tmem512.cu
nvcc -std=c++17 -O3 -cubin -Xptxas -v -arch=sm_100a -I csrc  -o /tmp/p0c.cubin bench/sm100_review/p0c_umma_pipeline.cu
```

| TU | 入口 reg | spill/stack | 备注 |
|---|---|---|---|
| ws probe ×4 实例(sm_100a & sm_110a) | 128 | 0 / 0 | smem static 1024B |
| p0a_setmaxnreg | 128 | 0 / 0 | 无 C7508;区内近满载校验和 |
| p0b_tmem512 | 14 | 0 / 0 | 512 列 alloc/dealloc + 384 到达握手 |
| p0c_umma_pipeline | 48 | 0 / 0 | i8 QK 链 8 轮 phase 翻转 + 全量比对 |

host 侧:ws TU 与旧 TU 均已用 torch 2.13/CUDA 13.2 头完整编译过(-O0 语法
级);全量构建与 golden gate 由主控在合并点跑。

## 6. M0-M3 milestone 与 ncu 预期信号

基线(128 线程 kernel,B200 profile):eligible warps/cycle 0.60、
no_eligible 54%、SM throughput 44.8%。cutedsl 同结构上限:XU 62-67% SOL。

| 阶段 | 内容 | 通过判据 |
|---|---|---|
| M0 点火(上机第一步) | p0a/p0b/p0c 三 probe PASS;ws kernel d128 non-causal s=512 单点不挂、输出有限 | probe 全 PASS;kernel 无 hang/trap |
| M1 正确性 | `SAGEATTN_SM100_WS=1` 过 golden 双 gate:d64/d128 × causal × lse × per-warp/per-thread × GQA × 奇数长度(qo_len 非 256 倍数、kv 非 128 倍数、kv≤128 退化) | 与现 kernel 输出逐位一致(bit-exact 契约) |
| M2 性能采样 | bench 扫 s∈{512..16K};ncu 采关键 kernel | eligible ≥1.2、no_eligible <40%、SM throughput 55-65%;几何均值相对 128 线程基线的加速落在 1.5×+(cutedsl 包络 2.31×) |
| M3 epilogue TMA store + 收尾 | **r5 已实现**(§6.6 lever B:epi_full + warp14,单次使用无需双缓冲,账本 §8 as-built);k_scale 走了寄存器预取(lever A),smem 预载留 §7 G2 | 重过 M1 gate;实现为纯搬运路径替换,bit-exact 口径**不必放宽**(放宽留给 §7 G1) |

### 6.1 M2 首轮实测:两遍读判负,已改单遍驻留

B200 b4h32s16384 d128、cdsl_bench 22 点:ws/old 几何均值 **0.887**(22/22
全慢)。ncu 归因:供给侧变好(Active/scheduler 1.99→3.43、Eligible
0.65→0.88、Issued 0.50→0.54)但单 warp 变慢(Warp Cycles Per Issued
Instruction 4.01→6.39),逐项 stall(cycles/issued-insn)里唯一大头是
`long_scoreboard` 0.83→2.76(+1.93),`mio_throttle` 0.05→0.40 同涨;
Memory Throughput 17.9→8.3,是延迟暴露不是带宽。与两遍 TMEM 读的判断吻合:
每 KV 块每线程 7 次 32 列 tcgen05.ld(两遍 4+3)对 128 线程 kernel 的 4 次,
流量 1.75×、暴露延迟链 7 条。

修复 = 预案 1 落地:改回单遍读整行驻留(结构即 128 线程 kernel 的
:407-449,寄存器账在 §4),每块 4 次 ld、每 ld 一 wait 的节奏不变——不碰
A5 判负的「批量发射后集中 wait」模式。方案取舍:
* A1 单遍驻留(选定):bit-exact 保持,消灭多余流量,零新机制;
* A3 两遍保留 + fragment 级交错 ld:流量仍 1.75×,只藏延迟,且 2 条
  outstanding ld 落在 A5 挂死(4 条 outstanding,根因未定位)的未证安全区;
* A2 fragment 局部 max + 行级补偿:破 bit-exact 换寄存器,A1 的账算得过来,
  不需要。

预期信号(下轮上机):`long_scoreboard` 回落、ws/old 转正;golden 双 gate
输出应与两遍版逐位一致(值恒等论证 §3)。

### 6.2 M2 第二轮实测(round 2):单遍驻留落地,仍慢 5.3%

数据(B200;22 点全表在主控 scratchpad,此处只记口径与结论):

- cdsl_bench 22 点,ws/old d128 几何均值 **0.947**(0/22 翻正);长序列
  non-causal ≈0.98,短序列/causal 0.83-0.93;d64 0.89-0.90。
- ncu b4h32s16384 non-causal d128:duration 16.86 ms(old 16.31 ms);
  Active/scheduler 3.43、Eligible 0.77;`long_scoreboard` 仍是最大 stall
  项。注意 per-issue 口径的 stall 比值因总发射数下降不可直接跨版本比,
  绝对口径 duration 相对 round 1(0.887×)已改善。
- 归因:softmax 每 KV 块 4 条 ld+wait 串行 tcgen05.ld(每条 32 列),
  4 段暴露延迟;softmax 区寄存器 189/192(余 3),批量/交错方向没有
  寄存器空间。

### 6.3 round 3:宽 tcgen05.ld + f32x2 pack(已实现,待上机)

两个独立 commit,每个都过 §5 门禁(nvcc 13.3,4 实例 × sm_100a/sm_110a,
全部 0 spill / 0 stack / 入口 128 reg):

1. **S drain 加宽 4×x32 → 2×x64**(commit 58a6a11)。
   - x128(整行一条 ld)不可行:`tcgen05.ld.32x32b.x128` 单条指令自身
     128 个 dst + 地址操作数超过 `__launch_bounds__(512,1)` 钉死的 128
     入口寄存器目标,ptxas 报 C7602;该检查用函数级目标,setmaxnreg 区
     预算(192)救不了(实测)。x64 是可行上限(指令脚印 65)。
   - raw→f32 原位转换(uint32_t 行数组 + 位转换),LDTM in-flight 块与
     驻留行共用寄存器;逐元素值序不变(bit-exact 保持)。
   - SASS:softmax 区 LDTM 8→4/实例,区内峰值 189 不变,无 MOV 膨胀。
   - 预期:暴露 TMEM-load 延迟链 4→2 段,`long_scoreboard` 相应回落。
2. **f32x2 pack**(commit d405309;prescreen P3 的 go 判据落地,
   bench/microbench/PRESCREEN_REPORT.md)。
   - dequant mul 与 exp2 输入 fma 逐相邻列对 pack 成
     `mul/fma.rn.ftz.f32x2`(相邻两列共享 k_scale,乘数是 splat);
     per-lane 是独立 IEEE fp32 运算,舍入/ftz 与标量 `mul.ftz.f32` /
     `fma.rn.ftz.f32` 一致 → 逐位一致。max 链(无 max.f32x2)与 d_sum
     串行加法链保持标量原序。
   - PTX 层对账:add/max 计数不变,mul 3672 = 2648 标量 + 512×2 packed,
     fma 1040 = 16 标量 + 512×2 packed——严格 1:1 lane 替换。
   - sm_100a SASS:FMUL2.FTZ / FFMA2.FTZ 各 128/实例,splat scale 折成
     单寄存器广播、零 MOV 开销;静态指令 -4.8%~-7.0%;softmax 区寄存器
     峰值 189→170-173(余量 3→19+)。sm_110a 无 packed f32 ALU,ptxas
     拆回两条标量,中性。
   - 预期:FMA pipe issue 压力 -5~7%;XU(MUFU)不动,收益上限受 XU
     串行链约束(§6 基线画像)。

方向 3(chunk 交错,2-outstanding)按 round 3 任务书跳过:方向 1 已可行,
且 SASS 显示 ptxas 在 peeled 实例里已自发把两条 x64 背靠背发射
(2 outstanding)——PTX 级 per-ld wait 不钉死 SASS 调度。若下轮
`long_scoreboard` 仍主导,f32x2 后的 19+ 寄存器余量已够源级交错立项,
先例是本观察 + cutedsl 的 4×x32 批量发射(其 2.31× 实测即此模式);重开
前仍须遵守 A5 规则(先复现定位挂死)。

资源与静态指令表(round 2 → round 3,sm_100a;sm_110a 同为 0 spill):

| 实例 | 总指令 | softmax 区 LDTM | softmax 区寄存器峰值 |
|---|---|---|---|
| hd64 per-thread causal+lse | 4872→4608 | 8→4 | 189→173 |
| hd64 per-warp 非 causal | 3904→3624 | 8→4 | 189→170 |
| hd128 per-warp 非 causal | 4576→4312 | 8→4 | 189→170 |
| hd128 per-thread causal+lse | 5224→4960 | 8→4 | 189→173 |

上机判据(叠加不变):golden 双 gate 逐位一致(两处改动的值恒等论证在
kernel 注释与本节;硬件 FMUL2/FFMA2 的 per-lane IEEE 语义由 golden gate
终审);bench 22 点 ws/old 转正;ncu `long_scoreboard` 显著回落。

M2 若仍不达标的定位顺序:
1. vec/corr 1-stage 串扰:`smsp__warp_issue_stalled_barrier` 分角色看;若
   correction 卡 vec,考虑 vec 双缓冲(TMEM 有空列:d64 显然,d128 可借
   P 区错峰,需重开别名论证)。
2. mma warp 发射间隙:`sm__pipe_tensor_op_cycles_active` 占空;若 QK/PV 间
   有大空洞,检查 s_empty 到达延迟(softmax 关键路径)。
3. softmax 2 条串行 x64 ld 链仍是热点:源级交错(2 outstanding)按 §6.3
   方向 3 段的先例与 A5 规则立项(已落地,见 §6.4)。

### 6.4 round 4:WS 三态启发式 + chunk 交错(2 ld 共用一次 wait)

r3 上机小结(B200,数据在主控 r3 轮记录;口径同 §6.2):

- bench:d128 且 qo_len ≥ 16384 的 10/10 形状 ws/old 1.006-1.027×;
  s ≤ 4096 全慢(ws 的 grid.x = qo_len/256,是旧 kernel 的一半,小 grid
  填不满 148 SM 的 wave);d64 慢 8-9%(512 列 TMEM 只用 384)。
- ncu b4h32s16384 non-causal d128:duration 16.17 ms(old 16.29 ms),
  Eligible 0.66;softmax 仍受 tcgen05.ld 暴露延迟支配。
- 压测:2×8000 次连续发射零挂死(r3 的两条 x64 结构,含 ptxas 在
  per-thread peeled 实例里自发背靠背发射的 2-outstanding 调度)。

两个独立 commit:

1. **SAGEATTN_SM100_WS 三态**(host-only,双 TU probe SASS 逐字节不变)。
   未设/`auto` = 启发式:HEAD_DIM==128 且 qo_len ≥ 16384 走 ws(16384 是
   实测全正下界,4096-16384 间无采样点,保守取);`1`/`on` 强制 ws;
   `0`/`off` 强制旧。env 仍进程内读一次,只缓存模式,启发式分支每次调用
   按 runtime 的 qo_len 判断。
2. **chunk 交错:两条 x64 ld 背靠背 + 单次 collective wait**。
   - wait 语义结论:`tcgen05.wait::ld.sync.aligned` 无操作数、无 per-op
     形式,PTX ISA 定义为阻塞到本线程此前**全部** tcgen05.ld 完成
     (cccl `tcgen05_wait.h` 同一形式)。任务书里 ld0/wait0/ld1/wait1 的
     按序交错表达不出来,按预案改为两条 ld 共用一次 wait、顺序处理两个
     chunk——总暴露延迟与按序 wait 版相同(两条 ld 同时在飞)。
   - bit-exact:PTX 逐 opcode 对账,4 实例唯一变化是 wait::ld 各 -2
     (main + peeled 两处展开各省 1),浮点 op 计数全部不变;处理循环
     源码未动,列序/fold 序不变 → 值序逐位一致。
   - A5 边界:2 outstanding 正是 r3 SASS 已出现且 2×8000 压测覆盖过的
     调度;A5 判负的是 4+ 条批量发射(根因未定位),不越界。
   - 门禁(nvcc 13.3,4 实例 × sm_100a/sm_110a):0 spill / 0 stack /
     入口 128 reg;softmax 区峰值不变(sm_100a 170-173,sm_110a 189)。
   - **ptxas 复沉实测(要点)**:PTX 钉住的背靠背结构到 SASS 不保真。
     sm_100a:8 个 pair 站点里 6 个 ptxas 仍把 ld1 沉回 chunk0 首个消费
     点之后(串行 round trip,与 r3 完全同构;in-flight 的 2 个站点也
     与 r3 相同),净变化只有 2 个实例各 -8 条 NOP——上机预期持平。
     sm_110a:per-thread peeled 站点从 gap 410 变背靠背 in-flight,
     main 站点 gap 143-152 → 77-107——有真实交错,或有小改善。
   - 后续 lever(若 long_scoreboard 仍主导):r3/r4 里唯一稳定 in-flight
     的站点(per-thread peeled)是因为 mask 的 ISETP 链填在 ld 与首个
     消费点之间;要在 sm_100a 上强制交错,得给 main 站点也填独立工作
     (如 mask/scale 预计算重排),需重开 bit-exact 论证,或等 ptxas
     调度器修正。

### 6.5 auto 拐点细化 sweep(wave9,B200 补采;启发式源码未动)

§6.4 的两个 cut(非 causal 16384、causal 32768)是当时 sweep 的全正下界,
4096-16384 与 16384-32768 两段无采样点。本轮在 gap 内补采 d128 b{1,4} h32,
另补 d64 s32768 复核 d64 长序列结论。

口径:tree 5af2e06(git archive 覆盖 sage-w4 重建,`TORCH_CUDA_ARCH_LIST=10.0a`);
B200 单卡独占(umb-b200-260,JID 4026426,pytorch_26.07-py3 容器,
torch 2.13.0a0 nv26.07 / CUDA 13.3);driver
`bench/sm100_review/ws_auto_sweep.py`(fwd op 直测 + CUDA event + 800 ms
budget,协议同 §6.2 起沿用的 cdsl_bench_fwd);WS 模式进程内只读一次 env,
故 ws(`SAGEATTN_SM100_WS=1`)与 old(`=0`)各占一进程,3 轮换序
(ws-old / old-ws / ws-old)逐形状取 median,轮间 spread ≤0.115%。
前置自检:重建后 ws 与 old 输出逐位一致(`ws_auto_sweep_sanity.py`,
b2h32s8192 d128 × causal 两态,SANITY_PASS)。原始数据:集群
`SageAttention_refactor/logs-w9/bench/*.json`。

ws/old(>1 = ws 快;ms 为 median):

非 causal d128:

| seq | b1 old ms | b1 ws ms | b1 ws/old | b4 old ms | b4 ws ms | b4 ws/old |
|---|---|---|---|---|---|---|
| 6144 | 0.674 | 0.698 | 0.9658 | 2.440 | 2.444 | 0.9985 |
| 8192 | 1.073 | 1.061 | 1.0111 | 4.236 | 4.205 | 1.0072 |
| 10240 | 1.690 | 1.663 | 1.0163 | 6.524 | 6.441 | 1.0130 |
| 12288 | 2.392 | 2.402 | 0.9960 | 9.303 | 9.153 | 1.0164 |
| 14336 | 3.274 | 3.273 | 1.0000 | 12.530 | 12.339 | 1.0155 |

causal d128:

| seq | b1 old ms | b1 ws ms | b1 ws/old | b4 old ms | b4 ws ms | b4 ws/old |
|---|---|---|---|---|---|---|
| 20480 | 3.452 | 3.473 | 0.9939 | 13.174 | 13.220 | 0.9965 |
| 24576 | 4.868 | 4.874 | 0.9988 | 18.734 | 18.718 | 1.0008 |
| 28672 | 6.537 | 6.523 | 1.0022 | 25.286 | 25.178 | 1.0043 |

d64 s32768:

| causal | b1 old ms | b1 ws ms | b1 ws/old | b4 old ms | b4 ws ms | b4 ws/old |
|---|---|---|---|---|---|---|
| 0 | 14.500 | 15.481 | 0.9366 | 57.157 | 61.358 | 0.9315 |
| 1 | 7.548 | 8.134 | 0.9279 | 29.293 | 31.535 | 0.9289 |

结论(阈值现值都成立,是否下调是收益 <1% 的取舍):

- 非 causal:拐点实际在 8192 附近但 b1 侧非单调——8192/10240 全正
  (+0.7~+1.6%),12288 b1 又 −0.4%、14336 b1 打平;b4 自 8192 起单调全正。
  8192-14336 段 8 形状 geomean 1.0094。b1 的振荡与 ws grid 减半后的 wave
  quantization 一致(b1 总 CTA 数 = 32·s/256,尾波填充率随 seq 在
  0.11-0.92 摆动;b4 波数 ×4,尾波摊薄——推断,未做 ncu 归因)。
  若下调 cut 到 8192:该段净收益 ~+0.9%,最大单点代价 −0.4%。
- causal:20480 双负(−0.6/−0.3%),24576 混合(b1 −0.1%),28672 起全正
  (+0.2/+0.4%,大于轮间抖动)。可下调至 28672,收益 <0.5%。
- d64 s32768 仍输 6.3-7.2%(4 形状 geomean 0.9312):d64 不进 auto 的
  结论在长序列继续成立。

若未来下调,取 `qo_len >= (is_causal ? 28672 : 8192)` 并接受上表两处
小负点;本轮只记录,源码阈值不动。

## 7. 风险与回退
### 6.5 r4 上机小结(wave8,口径同 §6.2)

- r4 两个 commit(WS 三态 + chunk 交错)上机后,主口径:**ws/cudnn 0.732**
  (bench 几何均值);cutedsl 同结构先验 1.07-1.18(XU 62-67% SOL,
  我们 ~50%)。剩余差距的结构对账在 §7。
- ptxas 复沉已在 §6.4 记录:PTX 钉住的背靠背 ld 在 sm_100a 8 站点里 6 个
  被沉回消费点后;r5 本地复核(5af2e06 基线重编)结论不变,逐站数据见
  §6.6。

### 6.6 round 5(wave9):ld shadow 预取 + epilogue TMA store(已实现,待上机)

两个独立 commit,均过 §5 门禁(nvcc 13.3,4 实例 × sm_100a/sm_110a,全部
0 spill / 0 stack / 入口 128 reg);区内峰值(SASS 寄存器号上界,r5 后):
softmax 164-170(预算 192)、correction 71-77(88)、mma/load/epi 38(40)。

1. **lever A:下块 k_scale 预取进 ld shadow**(commit 5e89994)。
   - 动机:基线 SASS 里 k_scale LDG 落在 vec_empty wait 与 s_full wait 之
     间(距 s_full wait 仅 1-3 条指令,首个消费 dequant FMUL 紧随其后);
     softmax 是落后方时 s_full 直落,per-warp 顺序发射让首条 LDTM 卡在等
     LDG 的 FMUL 后面——每 KV 块一次 LDG->LDTM 串行 round trip。改法:把块 iter+1 的 k_scale 原始字
     在块 iter 的两条 tcgen05.ld 之后、wait::ld 之前 LDG 进寄存器;
     q_scale 乘法留在消费步顶部(同输入同序 → 逐位一致;PTX 对账:全部
     fp opcode 计数不变,ld.global.nc 28 = 28)。
   - SASS 实测(sm_100a):ptxas 不保留 PTX 钉的 gap 位置,把预取 LDG 沉
     到 exp2/pack 段中部——仍比基线早 ~220 条指令(hd128-pw:LDG@683,
     s_full wait@907;基线 LDG@907 wait@910),LDG 延迟改为骑在 MUFU 段
     + P store + 两个 barrier wait 上。「填充落在 ld 与 LDTM 消费之间」的
     原目标**未达成**:LDTM 站点结构不变(6/8 沉没同 r4),故 lever A 的
     收益机制是消除步首 LDG 暴露,不是 ld 交错。
   - 预期信号:softmax 视角 `long_scoreboard`(步首段)回落;causal/短序列
     (trip 小、LDG 占比高)改善大于长序列;`ws/old` 全形状不劣化。
2. **lever B:epilogue 改 sO staging + warp14 TMA store**(commit 10b8cef,
   M3 项落地)。
   - 结构:correction 末段数学不变(:572-599 值序),目的地从直接
     global 改为 sO(TMA 128B swizzle 布局,16B 单元 u 在行 r 的地址
     r*128 + (u^(r%8))*16,STS.128 无 bank 冲突)→ 每线程
     fence.proxy.async → arrive epi_full[t];warp 14 等 phase 0,每 64 列
     box 一发 cp.async.bulk.tensor.4d S2G + commit_group,尾部
     wait_group.read 0(smem 生命期,账本 H13)。行界保护从逐行
     `q_idx < qo_len` 断言换成 tensor map dim1=qo_len 的 TMA store 裁剪,
     可观察行为相同。数值:同乘同转换,TMA 逐字节搬运 → bit-exact 保持,
     M1 golden 双 gate 口径不放宽。
   - 指令账(SASS,per 实例):hd128 的 128 条/线程 STG.32(线程按行分布,
     每条 warp 级 32 个 4B 散点 = 32 sector,写放大 8×)→ 32 条 STS.128 +
     4 发 UTMASTG.4D/CTA;hd64 64 STG → 16 STS + 2 发。总静态指令
     hd128-pt 4952→4840、hd128-pw 4312→4192。v_scale 读顺带并宽:
     LDS 126→70 条(主体 LDS.64×112 → LDS.128×64)。
   - smem 96→160KB(d128)/ 48→80KB(d64),<227KB;新 barrier
     epi_full ×2(单相),账本 §1/§3/§5(H11-H13)/§8 已更新为 as-built。
   - 预期信号:L2 写 sector 数(`lts__t_sectors_op_write`)显著回落;
     correction 尾段 LSU 压力消失;CTA 收尾延迟(最后一发 PV 到 kernel 结
     束)缩短——短序列(epilogue 占比大)改善大于长序列;`launch__*` 无变
     化(occupancy 不受 smem 影响:本 kernel 恒 1 CTA/SM)。
   - 风险:swizzle XOR 若与 tensor map 期望不符,输出按 16B 单元错排,
     M1 gate 立刻可见(整改点只有 epilog 的地址式);TMA store 对
     stride_seq_o*2 的 16B 对齐要求已由 head_dim∈{64,128} 保证。

上机判据(两 lever 叠加):
- M1:golden 双 gate 逐位一致(bit-exact 口径不变;lever B 的 swizzle 与
  OOB 裁剪由 gate 终审,专置奇数长度用例覆盖尾 CTA);
- M2:bench 22 点 ws/old 相对 r4 几何均值 >1,且无形状 <0.98;短序列与
  causal 点位单独看(两 lever 的预期主受益面);
- ncu(b4h32s16384 + 一个短序列点):`long_scoreboard` 步首分量回落
  (lever A)、`lts__t_sectors_op_write` 回落(lever B)、duration 下降;
- 压测口径不变(outstanding tcgen05.ld 仍为 2,未触 A5 边界)。

## 7. 与 cutedsl 的执行结构对账(wave9,r4 之后的剩余差距)

口径:我方 = r5 双 lever 后的 ws kernel,数据来自 cuobjdump(nvcc 13.3,
sm_100a,4 实例);cutedsl 侧无法本地编译(JIT 需 B200 环境),计数从
core_sm100.py 源结构推导,记为「推导」;其 profile 数据(XU 62-67% SOL、
2.31×)沿用历史实测。逐项过完 §2 对照表后,执行结构的差异只剩下表;
「cutedsl 有而我们没有」按预期收益排序:

| # | 项 | cutedsl(推导) | 我们(cuobjdump 实测) | 预期收益/代价 |
|---|---|---|---|---|
| ~~G1~~ | ~~softmax 值域改写:raw 域 tree-max + P448 常量域 exp2 + 0.5 种子 4 路 packed 求和(L1094-1183)~~ | 每块每线程:127 FMNMX 树(深 ~7)+ 1 FMUL(max 回 deq 域)+ 64 FFMA2(raw 直接进 exp2 arg)+ 128 MUFU + 64 FADD2 分 4 条独立链(深 16)+3 收束;**不物化 dequant 行**(I2F 128 两边都有,平项) | **wave10 已落地**(§9;实测 SASS 链深 198-200 → 35-36) | 已关闭(实现与数值分析、上机判据全在 §9;0.5 种子折 denom 一项未搬,denom 乘加结构保持旧序,省的是 1 条标量 FADD,不值得再动值序) |
| G2 | sKScale smem 预载(L107, L544-552;其注释记载 gmem 广播读是 ncu long_scoreboard 主因之一) | kernel 头一次性搬 ≤1024 块标量进 smem(4KB),softmax 每块 1 次 LDS | 每块每线程 1-4 次 LDG 广播(lever A 已把发射点前移 ~220 指令,暴露延迟基本盖住;L2 sector 浪费仍在:per-thread 粒度每 warp 每块 16B/32B sector) | lever A 后剩余收益 = 残余延迟窗口 + L2 流量;实现小(4KB static smem + 预载循环)。若上机 ncu 显示步首 long_scoreboard 已平,则收益趋零,不立项 |
| G3 | 4×x32 tcgen05.ld 批量发射(4 outstanding;cute.copy 单发整行) | 每块 1 个 LDTM 等待窗口(4 条并飞) | 2×x64,r4/r5 SASS:6/8 站点 ld1 沉回消费点后 → 2 个串行窗口;in-flight 只在 pt-peeled 站点 | 每块省 ~1 个 LDTM round trip。被 A5 规则挡住(4+ outstanding 挂死根因未定位)+ ptxas 复沉不受源级控制(r4/r5 两轮实证);翻案条件:A5 根因定位,或 ptxas 调度修正 |
| ~~G4~~ | ~~TMA descriptor prefetch(L432-436,prefetch_descriptor ×4)~~ | load warp 起手预取 Q/K/V/O 四张 descriptor | **wave10 已落地**(load warp 预取 Q/K/V、epilogue warp 预取 O;SASS UTMACCTL.PF ×4 站点;bit-exact) | 已关闭 |
| ~~G5~~ | ~~correction rescale 用 mul_packed_f32x2(L1237-1239)~~ | 每 O tile 每块 64 FMUL2(d128) | **wave10 已落地**(per-lane IEEE 恒等,bit-exact;correction trip 循环钉 unroll 1,否则前端把 runtime-trip 循环展开 ~4×,静态指令 +45%) | 已关闭 |
| ~~G6~~ | ~~epilogue sO+TMA store 流水~~ | epi_stage=2 + corr_epi pipe | **r5 lever B 已对齐**(单次使用省掉 empty barrier) | 已关闭 |

不构成差距的项(对账过程中排除):correction 与 softmax 的重叠深度
(vec 在 max 后、exp2 前发,rescale 藏进 exp2 段;vec/corr 管线级数与信
号位置逐行同构,§2)、mma warp 发射节奏(QK0|PV1|QK1|PV0 程序序照抄,
且我们省 2 发尾部 dummy commit,§2)、每 tile barrier 次数(9 管线 1:1
等价,账本 §1-§3)、q_scale 供给(均为每 tile 一读)、USE_SEQ_GATE(蓝
本默认关)、软件 exp2(蓝本自己判负,EX2_EMU_PER4=0)、mma_corr 共享
2-stage vs 专用 1-stage ×2(边界等价,账本 §1 偏差条)。

结论:剩余 ~27 点(ws/cudnn 0.732 → cutedsl 包络对应 ~0.95+)的结构性
来源集中在 G1(softmax 串行链与多余 dequant 发射)。G1 是「128 线程
kernel 值序逐行拷贝」这一 bit-exact 硬闸的直接代价,继续压 XU/隐延迟的
增量 lever(r3-r5)都绕不开它。**wave10 已按既定政策(精度换性能,双级
门禁)落地 G1 并顺路捎带 G4/G5,见 §9**;G2 视上机 ncu 步首
long_scoreboard 再议,G3 维持 A5 红线不动。

## 8. 风险与回退

| 风险 | 迹象 | 预案 |
|---|---|---|
| 挂死(新 barrier 协议) | kernel 不返回 | 见下方 triage 清单 |
| setmaxnreg 运行时行为与编译期不符 | p0a FAIL/hang | p0a 单独定位(TRY_ALLOC 自旋 = dec 没执行到:查 wg 分支覆盖);极端回退:去掉 setmaxnreg(全员 128 reg,softmax 会 spill,仅作正确性载具) |
| TMA OOB 零填充假设(int8 fill=0) | 尾块 CTA 输出错 | M1 专置奇数长度用例;若假设破产,改 launcher 侧 pad 或 tile1 掩蔽加载 |
| in-order UMMA 假设(H2) | P 被下一发 QK 踩,数值错但不挂 | 与 CUTLASS FMHA 同假设,风险极低;若疑,临时在 QK 前多等一拍 corr_full(牺牲重叠的诊断开关) |
| 尾 CTA 白算拖平均 | 短序列加速比差 | 已知代价(cutedsl 同);不单独修,归入 M2 评估 |
| 88 reg correction 将来加逻辑顶破 | 新增代码后 spill 复现 | r5 lever B(sO staging)后区内峰值 71-77,余量 11+,§5 门禁已重跑 |
| 192 reg softmax 将来加逻辑顶破 | 新增代码后 spill 复现 | G1 后区内峰值 sm_100a 167-177(余 15-25)、sm_110a 188-189(余 3,packed 拆标量后 4 条 d_sum 链的活跃度全暴露);softmax 区任何改动都要重跑 §5 门禁,顶破时先把 d_sum 4 链降 2 链(深 16→32,省 4 寄存器),再考虑 correction 让 reg |

挂死 triage(继承 quant-occupancy/sqsh 战场经验 + 本设计专项):

1. 先跑三 probe:p0a 挂 = setmaxnreg;p0b 挂 = alloc/dealloc 握手(前科:
   dynamic-smem opt-in 两 bug、双 CTA alloc 卡死——本 kernel 恒 1 CTA/SM,
   若挂优先查 dealloc 到达计数);p0c 挂 = commit/parity 或 smem 喂 MMA 缺
   `fence.proxy.async`(前科:SS twin)。三 probe 绿则问题在完整 pipeline。
2. `cuda-gdb` attach 看各 warp PC:停在哪个 `SYNCS.PHASECHK` 对回账本 §2 的
   事件序,直接读出是谁没到——账本的 #n/phase 记法就是为这一步准备的。
3. 二分:环境变量选新 kernel 后,按 tile 退化(qo_len≤128 使 tile1 全 OOB
   白算)、按 trip 退化(kv≤128 走 §3 的 T=1 轨迹)缩小到最小挂死形态。
4. 若 A2/A5 型未定位挂死复现(k_scale 预载/TMEM 批量读的前科),规则不变:
   不重开,绕开该结构。

## 9. G1:softmax 值域改写(wave10,已实现,待上机)

精度换性能改动(既定政策,双级门禁):golden bitwise gate 预期破,验收口径
= accuracy gate + bench 双闸,上机验收后重 dump golden(流程见 §9.5)。
三个 commit:G4 descriptor prefetch、G5 rescale packed mul(两者各自
bit-exact,独立可存活)、G1 本体。

### 9.1 设计

softmax step 不再物化 dequant 行,S 行全程呆在 raw 整数域(f32 表示,
I2F 原位覆盖,寄存器结构不变):

1. **块 rowmax = raw 域平衡 FMNMX 树**,每 k_scale class 一次乘法回 deq 域。
   per-warp:单棵 128 元素树 + 1 FMUL;per-thread:4 棵 32 元素树(class c
   拥有列 {8k+2c, 8k+2c+1})+ 4 FMUL + 3 FMNMX 合并。乘法舍入单调
   (x≤y ⟹ rnd(x·d)≤rnd(y·d), d≥0),所以「类内先 max 再乘」与旧「逐元素
   乘完再串行 max」**逐位同值**;补一个 -5e6 floor 复刻旧 m_local 初值语义
   (兼收 zero-amax class 全 mask 时 d·(-inf)=NaN 的角落:fmaxf 丢 NaN 操作
   数)。=> row_max、vec、o_scale、correction 的 rescale 因子与旧 kernel
   在**所有**输入下逐位一致(9.3 的 sim 断言覆盖全部 corner)。
2. **exp2 输入域折叠**:a = fma(raw, c_raw, -row_max),
   c_raw = sm_scale·log2e·q_scale·k_scale 每块每 class 预乘一次;-row_max
   本来就带着 S_FP8_OFFSET(row_max = c·m - 8.807 ⟹ -row_max =
   8.807 - c·m,即 cutedsl neg_off = LOG2_448 - c·m 的同款常量域)。逐元素
   dequant FMUL2(64 条/块)整段消失;P 只动在舍入位置上——旧
   rnd(raw·d) 后 fma,新 rnd(c·d) 后 fma,arg 差 ≤1 ulp 量级。mask 改在
   raw 域写 -inf(peeled step 专属);c_raw 在 peeled step 加 FLT_MIN
   clamp,防 zero-amax(dequant=0)块里 -inf·0=NaN——活 lane 的
   |raw|·FLT_MIN ≤ 2e-32 进 fma 加数即消失,clamp 只在 c_raw=0(全行乘积
   本来就 underflow)时才改值。
3. **d_sum 分链**:4 条独立 packed f32x2 累加链(quad w 喂链对 w&1,深
   16)+ 3 条 packed 收束 + 1 标量收束,替换 128 深串行 FADD;纯重结合。
   denom 的乘加结构(denom·o_scale 与 += d_sum 两步)保持旧序;cutedsl
   的 0.5 种子 trick 只省 1 条标量 FADD,不搬。

改位清单(golden bitwise 破的全部来源):P 的舍入位置(上面 2)、denom
重结合(上面 3)、两个只在病理输入下可见的语义差——mask 哨兵 -5e6→-inf
(旧哨兵在全行 deq 域 logits < -5e6 时会当 floor 用)与 LSE 随 denom 动。
row_max/vec/o_scale 不在清单里(逐位保持)。

### 9.2 SASS/资源(nvcc 13.3;基线 = r5+G4+G5)

| 指标(每 softmax body,sm_100a) | 旧 | G1 |
|---|---|---|
| 关键链深(FMNMX/FADD 依赖链,float op 数) | 198-200(64 FMNMX3 串行 + 128 FADD 串行) | **35-36**(树 ~9 级 + FADD2 深 ~19) |
| dequant FMUL2 | 64 | 0(FMUL 4-9 条:per-class 标量预乘) |
| rowmax | 64 FMNMX3 串行 | ~87 FMNMX/FMNMX3 平衡树 |
| d_sum | 128 FADD 串行 | 67 FADD2 |
| MUFU.EX2 | 128+1 | 128+1(XU 不动) |

估算口径(§7 G1 行的推导落实):~200×4 ≈ 800 cyc/块 → ~36×4 ≈ 145
cyc/块,消掉的正是基线画像里压在 XU 天花板之下的串行 ALU 链;每块另省
64 FMUL2 + 61 FADD 发射。总静态指令(G5 后→G1):hd64-pt 4288→4168、
hd64-pw 3528→3424、hd128-pw 4072→3968、hd128-pt 4656→4528。

门禁(4 实例 × sm_100a/sm_110a):全部 0 spill / 0 stack / 入口 128 reg。
softmax 区峰值:sm_100a 167-177(基线 162-169,+8 ≈ 4 条 packed 累加链),
sm_110a 188-189(packed 拆标量,余量 3,§8 风险表已更新)。correction
(G5)与 mma/load 区不变。

### 9.3 数值分析(bench/sm100_review/g1_softmax_sim.py,numpy 位级仿真)

口径:int32 S → 新旧两条 softmax 路径(fma 用 f64 乘加一次下转模拟,
exp2/rcp 用 libm f32 站位 MUFU 近似,e4m3 转换按 cvt.rn.satfinite 精确
实现;两条路径同一套模型,A/B 本身精确)→ e4m3 P → PV(f32)→ epilogue;
参考 = fp64 SDPA(量化前浮点 Q/K/V)。场景:randn(±causal)、大动态范围、
全常数行、outlier 块、zero-amax 块(含 masked peel 的 NaN 角落)、全行
logits < -5e6(旧哨兵 floor 角落)、kv=17 短块;每场景 × per-warp/
per-thread,128 行 × 32 块 × 128 列。

| 指标 | 结果 |
|---|---|
| row_max 位一致(新 vs 旧) | 18/18 组合全部成立(硬断言) |
| P 的 f32 级变位率 | 0-12.6%(变位幅度典型 ≤7.5e-7 相对,病理场景 ≤1.8e-3) |
| **P 的 e4m3 改写率** | **0 / 8360448 live 元素**(3-bit 尾数吞掉 arg 的 ulp 级差) |
| denom 相对移动 | 典型 ≤3e-7,病理 ≤1.6e-3 |
| O:新 vs 旧 | rel_l1 典型 ≤2e-6,最坏 8.5e-4(neg_5e6);cos ≥ 0.99999976 |
| O:新/旧 vs fp64 参考 | 逐场景到小数第 6 位相同(量化误差主导);randn cos 0.9993 / rel_l1 0.037-0.038,与 test_accuracy.py 记录的 fp8-PV 后端实测(0.99926/0.039)吻合 |
| accuracy gate 预检 | 除 neg_5e6 外全过(cos>0.995, rel_l1≤0.046);neg_5e6 新旧**同样**不过(l1 0.087/0.120,int8 attention 表示域外,非 G1 劣化) |
| LSE(新 vs 旧) | 典型 ≤2e-6,病理 ≤3.9e-3(test_accuracy 的 LSE 闸是 rtol 2e-2) |

### 9.4 预期收益与信号

- bench:softmax 是 §6 基线画像的双卡之一(tensor/XU 串行 + occupancy);
  G1 拆掉 XU 天花板下的串行 ALU 链,预期 d128 长序列 ws/old 在 r5 基础上
  继续抬升,短序列(softmax 占比更高)弹性更大。cutedsl 同结构包络
  ws/cudnn ~0.95+(§7)。
- ncu(绝对判据 duration;-k 过滤 kernel 名):`smsp__pipe_fma_cycles_active`
  回落(dequant FMUL2 消失),XU(MUFU)占空比向 62-67% SOL 靠拢,
  softmax 视角 selected/not-selected 结构改善;`long_scoreboard` 与 G1 无关
  (LDTM 结构未动),不作为本轮判据。
- G4:短序列/多 wave 形状的首块延迟略降(每 CTA 一次性 ~百 ns);G5:
  correction 发射数减半,非关键路径,预期中性偏正。

### 9.5 上机验收(下一会话,与 r5 结果合流)

1. **正确性(accuracy gate)**:`SAGEATTN_SM100_WS=1` 跑
   `pytest test/test_accuracy.py`(COS_MIN 0.99 / REL_L1_MAX 0.06 / LSE
   rtol 2e-2)+ `test/test_ops.py` 功能项;`SAGEATTN_SM100_WS=0` 全量
   golden `--check` 必须仍 diff=0(证明旧 kernel TU 逐字节未动,§5 的
   SASS 恒等已在本地核过)。
2. **ws=1 golden 差异确认**:对 r5 期 golden 跑 `--check`,预期 attn 段
   diff(P/denom 变位),quant 段 diff=0;差异形态若超出 9.3 的包络
   (如出现 NaN/Inf 或 cos<0.99)即回退。
3. **bench 双闸**:cdsl_bench 22 点 + §6.5 的 ws_auto_sweep 口径,判据
   = 相对 r5 几何均值 >1 且无形状 <0.98;auto 拐点是否下调按 §6.5 规则
   另记。
4. **压测**:`bench/sm100_review/ws_stress.py` 2×8000 连续发射零挂死
   (outstanding tcgen05.ld 仍为 2,未触 A5 边界;G1 未动 LDTM 结构)。
5. **重 dump golden**:全部通过后,`SAGEATTN_SM100_WS=1` 环境
   `tools/compare_reference.py --dump --golden-dir <新目录>` 固化 G1 后
   基线(双路口径不变:0 与 1 各一份);HARDWARE_CHECKLIST 记录切换点
   commit,此后 ws 路的 bitwise gate 以新 golden 为准。
