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
  warp 12    mma        单 elect 线程发全部 tcgen05.mma       setmaxnreg 40
  warp 13    load       单 elect 线程发全部 TMA               (同 wg3)
  warp 14    epilogue   M3 前空转(预留 TMA-store epilogue)
  warp 15    empty

TMEM(恒 alloc 512 列;lane = tile 内行号,两 tile 复用全部 128 lane):
  S0[0,128)  s32          vec0@[0,2) f32 复用   P0@[32,64) e4m3 复用
  S1[128,256)             vec1@[128,130)        P1@[160,192)
  O0[256,256+HD) f32      O1[256+HD,256+2HD)    (d64 时 [384,512) 空置)

smem(dynamic):sQ ×2 tile + K/V 共用 4-slot ring(item n -> slot n%4,
  K_i=item 2i / V_i=item 2i+1),d128 共 96KB / d64 48KB;static:mbarrier
  数组 bars[23] + tmem 槽 + sV_scale。

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
| softmax 数学 | raw 域 tree-max + packed FMA/exp2 + 0.5 种子 packed 求和 + P448 常量域 | 逐元素值序照抄现 kernel(:412-474);round 3 起 dequant mul 与 exp2 输入 fma 走 f32x2 packed 指令,per-lane IEEE 语义与标量逐位一致(§6.3),不是 cutedsl 那种改值域的 packed 化 | bit-exact 硬闸保持;cutedsl 式常量域改写仍留给 M3 后(需重开 golden 口径) |
| softmax 寄存器策略 | 整行 128 f32 驻留 | 照抄:单遍读、整行驻留(见 §3;初版两遍读实测 0.887× 已回退,见 §6.1) | 行驻留 128 reg + 状态 ~30,ptxas 收在 170-173/192、零 spill(§6.3);每 KV 块每线程 2 次 64 列 tmem_ld(round 2 是 4 次 x32,两遍版 7 次) |
| vec 提前信号(max 后、exp2 前发) | L1104-1109 | 照抄(全部 S 读已完成,别名列 [0,2) 已死) | correction 的 O rescale 藏进 softmax 的 exp2 段,是重叠收益主源 |
| k_scale smem 预载(sKScale) | L544-552(per-block gran) | 不搬,保持现 kernel 逐块 global 读 | 我们默认 per-thread gran(每 tile 4 标量);M2 若 long_scoreboard 热点再加 |
| q_scale 粒度 | BLKQ=64/WARPQ=16 契约 | 现 kernel 的 BLKQ=128 契约,块索引改 2bx+tile + clamp | 量化侧零改动 |
| epilogue | sO smem + TMA store(epilogue warp) | M0-M2:correction 直接寄存器->global(现 kernel :567-599 序) | bit-exact;M3 换 TMA store(corr_epi pipe 设计已入账本 §8) |
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
| M3 epilogue TMA store + 收尾 | corr_epi pipe + warp14 + sO 双缓冲(账本 §8);视 M2 profile 决定 k_scale smem 预载 | 重过 M1 gate(此步起允许放宽 bit-exact 为容差 gate,由主控拍板) |

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
   方向 3 段的先例与 A5 规则立项。

## 7. 风险与回退

| 风险 | 迹象 | 预案 |
|---|---|---|
| 挂死(新 barrier 协议) | kernel 不返回 | 见下方 triage 清单 |
| setmaxnreg 运行时行为与编译期不符 | p0a FAIL/hang | p0a 单独定位(TRY_ALLOC 自旋 = dec 没执行到:查 wg 分支覆盖);极端回退:去掉 setmaxnreg(全员 128 reg,softmax 会 spill,仅作正确性载具) |
| TMA OOB 零填充假设(int8 fill=0) | 尾块 CTA 输出错 | M1 专置奇数长度用例;若假设破产,改 launcher 侧 pad 或 tile1 掩蔽加载 |
| in-order UMMA 假设(H2) | P 被下一发 QK 踩,数值错但不挂 | 与 CUTLASS FMHA 同假设,风险极低;若疑,临时在 QK 前多等一拍 corr_full(牺牲重叠的诊断开关) |
| 尾 CTA 白算拖平均 | 短序列加速比差 | 已知代价(cutedsl 同);不单独修,归入 M2 评估 |
| 88 reg correction 将来加逻辑顶破 | 新增代码后 spill 复现 | 余量 10;M3 改 epilogue 时重跑 §5 门禁 |
| 192 reg softmax 将来加逻辑顶破 | 新增代码后 spill 复现 | round 3 f32x2 后余量 19+(§6.3);softmax 区任何改动都要重跑 §5 门禁,顶破时先看 exp2/pack 融合段可否再压,再考虑 correction 让 reg |

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
