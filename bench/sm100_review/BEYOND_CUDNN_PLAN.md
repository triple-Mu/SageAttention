# sm100 超越 cudnn:目标线、技巧侦察与路线(wave13 纯本地侦察)

日期 2026-08-31,基线 97c5b2b(ws kernel G1 终态)。本文全部结论来自本地读码与
已有实测数据的对账,无新上机数据;每个数字都标了来源。

**一屏结论**:ws kernel 的 XU 忙时(b4h32s16k d128 nc)恒为 ~8.0 ms,三代实现
(旧 kernel / ws-G1 / cutedsl)全部吻合——int8 形态的 XU 工作量(I2F+EX2+E4M3
pack ≈ 320 op/行/块,cudnn fp16 只有 ~192)是形态常数,比拼的只剩 XU 利用率:
我们 54.5%,cutedsl 同结构做到 62-67%,超越 cudnn 只需 59%。路线:
Phase A(correction 按 ballot 跳过 + sKScale smem 预载 + P 分块提前发 PV,均小改)
→ Phase B(persistent CTA + CLC + LPT,拿中短序列与 causal 尾部)
→ Phase C(tcgen05 cta_group::2,结构级)。d64 与 fp8-QK 形态单列,不进本役。

情报增量:**pip 包 nvidia-cudnn-frontend 1.27.0 内含 cudnn 团队自己的 sm100
CuTe-DSL prefill kernel 全源码**(`site-packages/cudnn/sdpa/fwd/kernels/`,
含 fp8 变体),其注释直接引用闭源 C++ 实现
(`mma_pipeline_op_native_sdpa_prefill_sm100_nonfp8.cpp`)并声明布局一致——
cudnn 的结构不再是黑盒,逐项 diff 见 §5。

---

## 1. 目标线:cudnn 逐形状绝对值与所需提升

数据 = wave11 B200 实测(umbriel JID 4027436,torch 2.13.0a0 nv26.07 / CUDA 13.3,
kernel-only 预量化输入,3 轮取 median;原始 json 在主控 scratchpad
`w11/bench/{cudnn,ws,old}-r{1,2,3}.json`,集群 `SageAttention_refactor/logs-w11/bench/`)。
`need` = ws 要追平 cudnn 还需快多少。

### 1.1 d128(主战场;ws = G1 终态)

| 形状(hd128) | cudnn ms | ws ms | ws/cudnn | ws TFLOPS | cudnn TFLOPS | need |
|---|---:|---:|---:|---:|---:|---:|
| nc b1 s1024 | 0.0255 | 0.0246 | 1.036 | 699 | 675 | 已超 |
| nc b1 s4096 | 0.2147 | 0.2867 | 0.749 | 959 | 1280 | +33.5% |
| nc b1 s16384 | 3.201 | 3.692 | 0.867 | 1191 | 1374 | +15.3% |
| nc b1 s32768 | 13.402 | 14.574 | 0.920 | 1207 | 1313 | +8.7% |
| nc b1 s131072 | 218.43 | 229.56 | 0.952 | 1226 | 1289 | +5.1% |
| nc b4 s1024 | 0.0652 | 0.0904 | 0.721 | 760 | 1055 | +38.7% |
| nc b4 s4096 | 0.7995 | 1.0052 | 0.795 | 1094 | 1375 | +25.7% |
| nc b4 s16384 | 13.576 | 14.701 | 0.923 | 1197 | 1296 | +8.3% |
| nc b4 s32768 | 54.703 | 57.804 | 0.946 | 1217 | 1286 | +5.7% |
| nc b4 s131072 | 878.14 | 915.84 | 0.959 | 1229 | 1282 | +4.3% |
| causal b1 s1024 | 0.0217 | 0.0236 | 0.922 | 365 | 396 | +8.5% |
| causal b1 s4096 | 0.1181 | 0.1843 | 0.641 | 746 | 1163 | +56.0% |
| causal b1 s16384 | 1.6115 | 2.0284 | 0.794 | 1084 | 1365 | +25.9% |
| causal b1 s32768 | 6.822 | 7.546 | 0.904 | 1166 | 1289 | +10.6% |
| causal b1 s131072 | 111.51 | 114.22 | 0.976 | 1232 | 1262 | +2.4% |
| causal b4 s1024 | 0.0468 | 0.0733 | 0.638 | 469 | 734 | +56.7% |
| causal b4 s4096 | 0.4665 | 0.5992 | 0.779 | 918 | 1179 | +28.4% |
| causal b4 s16384 | 7.059 | 7.622 | 0.926 | 1154 | 1246 | +8.0% |
| causal b4 s32768 | 28.470 | 29.251 | 0.973 | 1203 | 1236 | +2.7% |
| causal b4 s131072 | 444.80 | 453.78 | 0.980 | 1241 | 1266 | +2.0% |

逐 seq 行几何均值(4 形状/行):s1024 0.814、s4096 0.739、s16384 0.876、
s32768 0.935、s131072 0.967;d128 全段 0.862、22 点全段 0.835
(与 C1_DESIGN §9.6 记录一致)。分层读法:

- **s≥32k 只差 2-9%**:Phase A 的小改预计就能把 b4 侧翻正,b1 侧靠 Phase B。
- **s16k 差 8-26%**:b4 侧 8%,b1 侧 15-26%(b1 只有 13.8 waves,尾波+重启占比高;
  另见 §7 的 b1/b4 L2 对照采集项——cudnn 在 b1 比 b4 快 6%,我们持平,说明它有
  我们没拿到的 L2 复用或调度收益)。
- **s4096 是全表最深谷(0.64-0.80)**:wave quantization 只解释 ~1%(b4 13.8 waves,
  尾波 84% 满),大头是每波 pipeline 充放与 kernel 重启的固定开销 × 14 个串行波,
  加 causal 的 trip 不均;这正是 persistent+LPT 的主受益面(§4.4)。
- s1024 b1 已超(cudnn 小 grid 自己也填不满);b4 s1024 的 39-57% 缺口同属
  Phase B 主场。

wave14 复测(2026-08-31,B200 同容器同卡,`logs-w14/bench`;cudnn 9.24.0,
torch 2.13.0a0 nv26.07):cudnn 22 点与本表偏差 geomean 0.3%(最大单形状
5.8% @ nc b4 s1024,微秒级形状抖动),**目标线沿用本表**。ws 侧 G2 落袋后
ws/cudnn 22 点 0.835→**0.841**(d128 0.862→0.869);逐 seq 行几何均值
s1024 0.827、s4096 0.754、s16384 0.884、s32768 0.934、s131072 0.962。

### 1.2 d64(单列,不进本役主线)

| 形状(hd64 nc) | cudnn ms | 现行(old kernel)ms | 比值 | need |
|---|---:|---:|---:|---:|
| b4 s4096 | 0.579 | 1.024 | 0.566 | +76.8% |
| b4 s16384 | 10.057 | 15.630 | 0.643 | +55.4% |

d64 的 ws 比 old 还慢 7%(auto 已停用 ws),cutedsl 同结构也只有 0.92-0.94
vs 七月 cudnn。已知唯一结构候选:q_stage=1 + 2 CTA/SM(TMEM 192×2=384≤512,
cutedsl 分支 `profile/sm100-tuning-baseline/REPORT.md` 估 ≤+35%)——即使兑现也
够不到 cudnn,需要独立立项重新设计,本文不排期。

### 1.3 参照系:cutedsl 包络与 FA4(七月四引擎大表,commit eb1d3e7)

四引擎 32 点表(b4h32,cudnn/FA4fp16/FA4bf16/cutedsl-sage)在
`git show eb1d3e7:profile/sm100-tuning-baseline/REPORT.md`。要点:

- cutedsl-sage d128 nc(b4):s4096 1311 TF、s16k 1474、s32k 1502、s131k 1522;
  causal s16k 1300、s131k 1473。对 w11 cudnn 重算比值:s16k nc 1.14、
  s32k 1.17、s131k 1.19、causal s16k 1.043——长序列的「可行性证明」仍然成立。
- **但 cudnn 在两轮之间变快了(目标线在移动)**:s4096 nc cudnn 七月 1227 TF →
  w11 1375(+12%),causal s4096 1051→1179。cutedsl 的 s4096 nc 1311 对新 cudnn
  只有 0.95——中段谷地即使做到 cutedsl 水平也不够,必须要 cudnn 自己的
  调度机制(§4.4/§4.6)。两容器 cudnn 版本不同(sglang-diffusion vs
  pytorch_26.07-py3);验收会话必须同场重测 cudnn,不能引用历史值。
- FA4 fp16/bf16 在 B200 b4h32 全部 d128 点位低于 cudnn(如 s16k nc 1054 vs 1293),
  且长序列掉速;FA4 源码未 vendor 进本仓库(七月是直传容器),本轮以 cudnn DSL
  源码为主参照,FA4 不再单独追。

### 1.4 XU 恒量:长序列的换算尺

ncu b4h32s16384 nc d128(w11,`scratchpad/w11/ncu/g1_ws_s16384_c0.raw.csv`):
duration 14.687 ms,SOL 顶 = XU pipe 54.48%(tensor 26.8/tc 29.7、ALU 31.8、
FMA 21.6、LSU 0.3、mem 10.1)。三代实现的 XU 忙时(duration × XU%):

| 实现 | duration | XU% | XU 忙时 |
|---|---:|---:|---:|
| 旧 128 线程 kernel(七月) | 12.885 ms | 62.3% | 8.03 ms |
| ws-G1(w11) | 14.687 ms | 54.5% | 8.00 ms |
| cutedsl 调优后(七月) | 12.14 ms | 62.3% | 7.56 ms |

XU 忙时 ≈ 8.0 ms 是形态常数(int8 QK 的 softmax 每行每块 I2F 128 + EX2 128 +
E4M3 pack 64,SASS 计数见 §3)。由此:

- 追平 cudnn 13.58 ms ⟺ XU 利用率 ≥ 58.9%——低于 cutedsl 已示范的 62-67%,
  **长序列超越在本形态内已被同结构实现证明可行**;
- cutedsl 终态 11.92 ms ⟺ 67%;100% 理想下界 8.0 ms,不可达,别当目标;
- b1 s16k 需 ≥62.5%,恰在示范带边缘 → b1 侧还要吃到调度类收益(Phase B)。

---

## 2. cudnn 开箱情报:nvidia-cudnn-frontend 1.27.0 的 sm100 DSL kernel

位置:`/home/ubuntu/miniconda3/envs/torch/lib/python3.12/site-packages/cudnn/sdpa/fwd/`
(kernels/prefill_d128_f16_sm100.py 2122 行、prefill_d128_fp8_sm100.py 2047 行、
config_sm100.py、_common_sm100.py;调度器在 `cudnn/frost/tile_dsl/scheduler.py`)。

可信度口径:这是 cudnn 团队的 python DSL 产品线,不是 torch SDPA 实际调用的
闭源二进制;但 (a) 文件头与注释多处声明与 C++ 参考实现布局一致
("matches the C++ SM100 reference layout"),(b) 注释直接引用闭源文件名与行号
(`mma_pipeline_op_native_sdpa_prefill_sm100_nonfp8.cpp:916-921`),(c) warp 分工、
寄存器预算(192/88/40)与我们从 ncu 反推的 cudnn 行为吻合。按「高置信设计情报,
上机 ncu 复核」使用;复核清单见 §7(含把这个 DSL kernel 直接跑起来对表的项)。

d128 flavor 的结构参数(config_sm100.py CfgD128):

- **cga2 + tcgen05.mma cta_group::2**:CGA_M=CTA_MMA=2,MMA 的 M=256 跨两 CTA
  collective;K 按 seq 行对半(每 CTA 存 64 行)、V 按 d_v 列对半,per-CTA
  K/V smem 与 TMA 流量减半(`K_ROW_OFFSET_PEER`/`V_COL_OFFSET_PEER`,
  prefill_d128_f16 tmaldg 段)。fp16 下 smem 192KB = sQ64+sK32+sV32+sO64。
- **persistent + CLC try_cancel**:warp 15 是 scheduler,循环
  `clusterlaunchcontrol.try_cancel`(multicast 到 cga 两 CTA)偷未启动 cluster 的
  blockIdx 写进 smem,2-stage 流水;各角色 warp 以 `read_tile_id_arrive` 取下一
  tile。调度序:NATURAL / LPT(Q tile 倒序,causal 长 trip 先跑)/ LPT_L2。
- **16 warp 分工与我们同构**:softmax 8 + correction 4 + mma 12 + tma-ldg 13 +
  tma-stg 14 + sched 15;寄存器 192/192/88/40(correction 88 与我们的偏离一致)。
- **RESCALE_THRESHOLD 惰性 max**:fp16 阈 8.0、fp8 阈 4.0(`rescale_threshold()`);
  块 max 超过 running max 不足阈值就不更新 → alpha ≡ 1.0;correction 用
  `vote.all(alpha==1.0)` **整段跳过 O 的 TMEM 读改写**(O_CHUNK=16×8)。
- **P 分块交付**:softmax 先 exp2+pack 前 64 列、arrive `bmm2_ready[chunk0]`,
  MMA 立刻发 PV chunk0,再交付后 64 列;chunk0 的 barrier 计数同时含 softmax 和
  correction(一个 barrier 合并「P 就绪」与「O rescale 完成」两个依赖)。
- KV stages:fp16 2、**fp8 4**(1 字节输入换更深预取);fp8 P 存 e4m3、
  PV 走 F8F6F4,per-tensor descale 折进 softmax 标量,无逐元素开销。
- softmax S drain = 2×`tcgen05.ld` x64 背靠背(与我们 r3 后同构,**不是** 4 条
  outstanding——A5 红线没有反例)。
- wg0/wg1 之间有 named-barrier 错拍(wg1 顶部等 wg0 底部)。
- TMEM:S0@0 S1@128、O 256..511、stats 骑在各 S slot 头两列、P 叠 S 尾
  (fp16 @64/192,fp8 @96);`mb_stats_read` 单独一个 barrier 防下一 tile 的
  BMM1 覆写 stats(persistent 特有 hazard)。

---

## 3. 差距分解:XU 工作量帐(SASS 计数)

本地重编 ws probe(`nvcc 13.3 -arch=sm_100a`,§复现命令同 C1_DESIGN §5)后
cuobjdump 统计,4 实例合计:MUFU.EX2 1042、I2FP.F32.S32 1024、
F2FP.SATFINITE.E4M3.PACK 512、F2FP.F16/BF16.PACK 各 192、MUFU.RCP 20。
折回每 softmax 行每 KV 块:

| XU 家族 op(ncu 的 xu pipe = 超越函数 + 数据转换) | 我们(int8 QK) | cudnn(fp16) | cudnn(fp8 QK) |
|---|---:|---:|---:|
| I2F(S 的 s32→f32) | 128 | 0(MMA 直接出 f32) | 0 |
| MUFU.EX2 | 128+1 | 128+1 | 128+1 |
| P pack(F2FP e4m3/f16 x2) | 64 | 64 | 64 |
| 合计 | **~320** | ~192 | ~192 |

三点推论:

1. int8-QK 形态自带 1.67× 的 XU 工作量(I2F 是 int8 MMA 输出 s32 的形态税,
   逐元素无从省);这就是 §1.4 的 8.0 ms 恒量。**能动的只剩利用率。**
2. **f16x2 EX2 想法在纸面上就是平手,不立项**:把 128 条 EX2.f32 换成
   64 F2FP.F16.PACK + 64 EX2.f16x2 + 64 F2FP.E4M3(f16x2 源)= 128+64+64+64
   = 320,XU 计数不变(f32→f16 的转换也在 XU)。除非上机证明 I2F/F2FP 不占
   xu pipe(§7 采集项 M6),否则封存。
3. fp8-QK(e4m3 QK MMA 直接出 f32)可把 XU 工作量降到 cudnn 同级,但那是
   换量化形态(int8 QK 是 SageAttn2++ 的精度卖点),属产品级决策,本役出界,
   仅记录。

结论:长序列的仗 = 把 54.5% 利用率抬到 59%+,手段是拆 softmax/correction 侧的
串行等待与 TMEM 流量(§4.1-4.3),以及消灭 kernel 重启/尾波(§4.4)。

---

## 4. 技巧清单(按预期收益排序)

每条:机制 / 证据 / 预期量级 / 精度口径 / 工作量 / 上机判据 / 黑名单对照。
黑名单总表见附录 A。

### 4.1 correction 按 ballot 跳过 O rescale(cudnn 同款,去阈值版)

- 机制:o_scale = exp2(m_prev−row_max),块没刷新 max 时恒等于 1.0;
  correction 当前无条件做 128 列 O 的 LDTM→FMUL2→STTM 往返
  (`qk_int_sv_f8_cuda_sm100_ws.cu` rescale lambda,:1033-1068)。
  改为 `vote.all(m_prev==row_max)` 命中时整段跳过(barrier 收发保留)。
  correction 的 TMEM 列流量占全 kernel 的 ~61%(softmax 128ld+32st,
  corr 128ld+128st,每块每 tile),跳过直接减 TMEM 口争用(与 softmax 的
  S drain、MMA 写回同口),corr_empty 也更早到 → PV 等待缩短。
- 证据:cudnn d128 f16/fp8 默认开(`all_alpha_one` ballot,
  prefill_d128_f16_sm100.py:1719);它还配了阈值让 ballot 几乎恒命中。
- 预期量级:跳过率随块序号 i 按 (i/(i+1))^32 上升(warp 32 行都不刷新才跳),
  randn 数据 s16k(128 块)平均 ~40-50%,s131k ~80%+;换算 TMEM 流量省
  25-50%。对 duration 的弹性未知(TMEM 口压力 ncu 无直读指标),
  按 +2~5% 长序列挂账,上机定夺。
- 精度:o_scale==1.0 时跳过唯一的位差来源是 FTZ——mul.ftz 会把 O 里的
  f32 denormal 刷零,跳过则保留。P·V 累加出 f32 denormal 需要病理性相消,
  golden `--check` 大概率 diff=0;若有 diff 走 accuracy gate 口径
  (G1 已把 ws 路切到该口径,§9.5 流程复用)。
- 工作量:小(correction 区 +ballot +分支;寄存器余量 11+,重跑编译 gate)。
- 上机判据:golden 对照 → bench 22 点(尤其 s≥16k)→ ncu duration 与
  `not_selected`/`long_scoreboard`(softmax 视角)回落。
- 黑名单对照:无先例冲突(correction 数值路径不动,只是条件执行)。

### 4.2 G2:sKScale smem 预载(C1_DESIGN §7 遗留项,照 cutedsl L544-552)

- 机制:kernel 序幕把 ≤1024 块的 k_scale 一次合并读进 4KB smem,softmax 每块
  改读 LDS;消灭 per-block 广播 LDG 的 L2 sector 浪费与残余延迟窗口。
- 证据:cutedsl 实测 +5.8%(d128s16384f 12.885→12.140 ms,REPORT「实施与结果」);
  我们的 lever A(寄存器预取)已吃掉发射点前移的部分,但其收益机制被 G1 改写
  softmax 后已不可分辨,且 L2 sector 浪费仍在(ncu details 曾判
  "每 sector 32B 仅 4.9B 被利用")。
- 预期量级:+1~3%(cutedsl 的 5.8% 打折:lever A 已在、G1 后步首等待变短)。
- 精度:bit-exact(同值同序,只换存储层级)。
- 工作量:小(静态 smem 4KB + 预载循环;smem 160→164KB,远低于 227KB)。
- 上机判据:softmax 步首 long_scoreboard 分量回落;bench 全形状不劣化。
- 黑名单对照:无;C1 §7 明确留作 G2,条件「若步首 long_scoreboard 已平则收益
  趋零」——预载后顺手把 lever A 的 shadow LDG 删掉(减一份 L2 读放大,
  w11 ncu 记过 L2 read +14%)。

### 4.3 P 分块交付,PV chunk0 提前发(cudnn 同款)

- 机制:softmax 先算完前 64 列的 exp2+pack 并 arrive,MMA 即发 PV 的前半
  (V 行 0-63 的 2 个 K=32 子步),后 64 列跟上;s_empty 一分为二。
  cudnn 还把 correction 完成合并进 chunk0 的 barrier 计数(省一次独立 wait)。
- 证据:cudnn d128 softmax `bmm2_ready[qs*2+chunk]`(:1387-1400);
  我们现在是整行 P 齐活才 arrive(pv lambda 的 bar_p_ready)。
- 预期量级:+0~2%(tensor 26.8% 不是顶;收益在缩短 softmax→PV 的关键路径拍,
  短序列/低 trip 弹性更大)。
- 精度:P 数值不动;若顺带把 exp2 处理序改成前半行/后半行,d_sum 4 链的
  归组会变(重结合)→ golden 变位、accuracy gate 口径;若保持现有列序只拆
  arrive,bit-exact 可保(优先做后者)。
- 工作量:中(barrier +2、softmax 主循环重排、账本 §1/§3 更新)。
- 上机判据:bench 短序列点位;ncu `sm__pipe_tensor` 空洞收窄。
- 黑名单对照:无。

### 4.4 persistent CTA + CLC try_cancel + LPT(cudnn 同款,中短序列主攻)

- 机制:grid 不变,CTA 常驻:专用 scheduler warp(现 warp 15 空转,位子现成)
  用 `clusterlaunchcontrol.try_cancel` 偷未启动 block 的 blockIdx,smem 广播,
  各 warp 外层加 tile 循环;LPT 策略把 Q tile 倒序发(causal 长 trip 先跑,
  cudnn `decode_linear_tile_lpt`)。消灭:每波 kernel 重启/序幕(TMEM alloc、
  barrier init、descriptor prefetch)、尾波空转、causal 尾部 straggler。
- 证据:cudnn d128 默认 persistent(SCHEDULER_STAGES=2);其 sm120 kernel 注释
  也点名「避免昂贵 CTA 落在最后几波」。我们 s4096 谷地 0.739、b4s1024 0.72/0.64
  与波数强相关(§1.1);cutedsl REPORT 遗留清单里 persistent+CLC 是唯一
  未验证的开放项。
- 预期量级:s4096 +5~15%,s1024 b4 +15~25%,causal 中段再加(LPT);
  长序列 +1~3%(13.8-55 waves 的重启摊薄)。合计仍不一定填平 s4096 的
  25-56% 缺口——中段全面超越可能还要 §4.6。
- 精度:逐 tile 数学完全不动,bit-exact;golden 不动。
- 工作量:大(launcher 加 cluster 属性与 CLC 特性检测、16 warp 全员外层循环、
  barrier phase 跨 tile 连续性论证进账本、q/k_scale 指针按 tile 重推导)。
  分两步降险:先 NATURAL 顺序(纯 persistent),再 LPT。
- 上机判据:M0 级 probe(p0d:cluster(1,1,1) 上 try_cancel 的最小往返)→
  golden 双 gate → bench 分段(s1024/4096 主看)→ ncu `launch__waves`、
  尾波空转(PM sampling 时间线)。
- 黑名单对照:**不撞**。已判不可行的是「CTA_Q=128 单 tile 变体」(调度粒度,
  非驻留)与 sm90 的 WS 四件套(架构不同);CLC 是 sm100 新机制,无前科。
  风险:CLC 在 cluster=(1,1,1) 启动下的可用性需 probe 先行(cudnn 全部
  cga2 启动,没有 cga1 反例;PTX 手册允许,但按本仓库惯例上机自证)。

### 4.5 RESCALE_THRESHOLD 惰性 max(cudnn 同款;4.1 的加强版,accuracy 换性能)

- 机制:块 max 超过 running max 不足 T(log2 域)不更新 → alpha≡1,
  4.1 的 ballot 命中率从 (i/(i+1))^32 提到 ~100%;P 的动态范围上界变
  2^T 倍,448 常量域要让位:row_max 映射点从 448 降到 448·2^-T(T=4 时 28),
  P 底部提前 T 个 binade 进 e4m3 subnormal。
- 证据:cudnn fp16 T=8、fp8 T=4(config `rescale_threshold`);其 fp8 kernel
  与我们同为 e4m3 P + F8F6F4 PV,T=4 是现成的先例参数。
- 预期量级:在 4.1 之上再 +1~3%(把 40-80% 跳过率补到 ~100%,并消掉
  正态数据下 max 缓慢爬升期的 rescale)。
- 精度:**golden 必破**,走既定「精度换性能双级 gate」政策;先扩
  `bench/sm100_review/g1_softmax_sim.py` 仿真(P 位变率、denom/O/LSE 包络、
  zero-amax 与全 mask 角落,阈值扫 T∈{2,3,4}),包络进 9.3 同款表再上机。
- 工作量:中(softmax max 更新条件 + 常量域偏移改 448·2^-T + sim)。
- 上机判据:accuracy gate(cos≥0.99/rel_l1≤0.06/LSE rtol 2e-2)+ bench;
  过线后重 dump golden。
- 黑名单对照:无先例;与 G1 的 row_max 位一致声明冲突(§9.1 的「row_max
  逐位一致」将不再成立)→ 文档与 golden 切换点要同步改。

### 4.6 tcgen05 cta_group::2(cudnn 的地基;结构级,Phase C)

- 机制:2-CTA cluster,MMA M=256 collective;K 按 seq 行、V 按 d_v 列各半,
  per-CTA K/V smem 与 TMA/L2/DRAM 流量减半,mma warp 发射与 barrier 数减半;
  释放的 smem 32KB(int8/fp8 下 KV ring 4 slot 只占 64KB)可把 KV ring 加深到
  4 个 KV 块在飞(cudnn fp8 正是 STAGES_KV=4)。
- 证据:cudnn d128 是 cga2-only(config 断言);七月 ncu 里 cudnn 同 shape
  tensor 44% vs 我们 9.2%(d128s1024c)的粒度差一半来自它。
- 预期量级:诚实说——在我们当前瓶颈像(XU 顶、tensor 27%、DRAM 10%)下,
  MMA/流量减半的直接收益有限;它的价值是 (a) b1/长序列的 L2 压力减半
  (§1.1 的 b1 异常),(b) 给 KV 深预取腾 smem,(c) 与 persistent/CLC 的
  cluster 语义天然契合。列为 Phase C:Phase A+B 不达标或 ncu 显示 KV 供给/
  L2 成为新顶时启动。
- 精度:bit-exact 可保(MMA 数值同序,K/V 切半不改 reduction 序;
  collective MMA 的 accumulate 语义与单 CTA 相同)。
- 工作量:很大(TMA 切半坐标、跨 CTA smem descriptor、leader/quiet mma、
  multicast barrier 家族、奇数 grid.x 的尾 cluster 处理、账本重写)。
- 上机判据:M0 probe(p0e:双 CTA collective MMA 最小链)→ golden → bench;
  ncu `launch__cluster_size`、KV DRAM/L2 流量减半自证。
- 黑名单对照:**sm90 的 cluster 记录是 TMA multicast 路径**(数据多播给两份
  smem,4-warp kernel,已判不可行)——cta_group::2 是不同机制(数据各存一半,
  MMA 跨 CTA 读),该结论不迁移。反过来:单独的「TMA multicast 共享整份 KV」
  与 sm90 输掉的机制同款,且我们 DRAM 只有 10%——不立项,除非 §7 的 L2/DRAM
  采集证明 b1 长序列真的卡供给。

### 4.7 wg0/wg1 named-barrier 错拍(cudnn 有,cutedsl 已判不可行)——不立项

cudnn softmax wg1 顶部等 wg0 底部(barrier_id 8);cutedsl 的同类 S0/S1 MUFU
错拍(USE_SEQ_GATE)实测删掉更快(d128s16384f +4.8%),默认关。两家结论相反,
说明收益依赖各自的发射时序;我们与 cutedsl 同构,沿用其结论。翻案条件:
PM sampling 显示两 softmax wg 的 MUFU 突发同相踩踏。

### 4.8 其余对账结论(不立项,防止重复立项)

- **软件 exp2**:七月实测全线回退(I-cache 溢出 + ALU 本已高位),f32 域不再碰;
  f16x2 域的变体也被 §3 的 XU 平手账封存。
- **G3 批量 4×x32 tcgen05.ld**:A5 红线不动(4+ outstanding 挂死根因未定位);
  cudnn 也只用 2 条 x64 背靠背——没有对手证据要求我们冒险。
- **KV ring 加深(不带 cga2)**:B1 已在 128 线程地基证伪(geomean 1.0004,
  方向不一致);ws 地基上单独加深的前提是 mma warp 实测卡 `wait_kv`
  (§7 采集项 M4),有证据再议,且优先作为 4.6 的附带项。
- **L2 persistence policy(cudaAccessPolicyWindow)**:b1 长序列 KV 每 head
  33.5MB、512 CTA 复用,天然流式;w11 ncu L2 读 = 每 CTA 全量 KV 重读、
  DRAM 仅 10%——L2 已经在扛,policy 无从改善,只保留 §7 的对照测量。
- **d64 TMEM/warp 重排**:见 §1.2,独立立项。
- **softmax 与 PV 重叠的其余部分**:与 cutedsl 逐项同构已核(C1 §2/§7 的
  「不构成差距」清单);cudnn 之外再无已知重叠机制,增量都在 4.1/4.3/4.5。

---

## 5. 结构 diff:cudnn DSL d128 vs 我们 ws kernel(交付物 2)

| 维度 | cudnn(prefill_d128_{f16,fp8}_sm100.py) | 我们(qk_int_sv_f8_cuda_sm100_ws.cu G1 态) | 差距归类 |
|---|---|---|---|
| 输入形态 | fp16/bf16;fp8 变体 e4m3 QK+PV,per-tensor descale | int8 QK + fp8 PV,per-block/warp/thread scale | 我们精度机制更重(I2F+逐块 scale),XU 税 §3 |
| 线程/warp | 512 线程 16 warp:softmax 8 + corr 4 + mma/tma-ldg/tma-stg/sched | 同构:softmax 8 + corr 4 + mma/load/epi + 空转 warp15 | 平;warp15 位子可给 scheduler(§4.4) |
| 寄存器 | 192/192/88/40(setmaxnreg 同款) | 192/192/88/40 | 平(巧合到 88 都一致) |
| grid/调度 | persistent + CLC try_cancel + NATURAL/LPT/LPT_L2 | 每 tile 一 CTA,自然序,kernel 重启 | **§4.4** |
| cluster/MMA | cga2,cta_group::2,M=256 collective;K seq 对半 V d_v 对半 | 无 cluster,cta_group::1,M=128 ×2 tile | **§4.6** |
| KV pipeline | K/V 各自 ring,f16 2 段、fp8 4 段 | K/V 共用 4 slot(=2 块在飞) | §4.8(单独加深已有 B1 前科;随 §4.6 重估) |
| Q | TILES_Q=2,mb_q_full ×2,QO_ALIAS 关 | 双 Q tile 同构 | 平 |
| softmax S drain | 2×tcgen05.ld x64 背靠背 + 手工树 max | 同构(r3/r4 已对齐;G1 树 max) | 平(A5 红线两家一致) |
| max 更新 | RESCALE_THRESHOLD 惰性(f16 8.0 / fp8 4.0) | 每块必更新(row_max 位一致) | **§4.5** |
| alpha/rescale | softmax 算好 alpha 写 TMEM stats;corr `vote.all(alpha==1)` 全跳 | corr 读 vec 自算 exp2,无条件 128 列往返 | **§4.1** |
| P 交付 | 64 列一档,PV chunk0 先行;chunk0 barrier 合并 corr 完成 | 整行 P 齐活单次 arrive | **§4.3** |
| P 存储 | f16:fp16 P 叠 S 尾 @64;fp8:e4m3 @96 | e4m3 @S+32 | 平 |
| stats 通道 | (total_max,total_sum) 骑 S slot 头,mb_stats_read 防覆写 | vec=(m_prev,row_max) 走 S 头两列,vec_full/empty | 平(细节不同,拍数相同) |
| wg 错拍 | wg1 顶等 wg0 底(named barrier 8) | 无(cutedsl 反例结论沿用) | §4.7 不立项 |
| epilogue | corr 写 sO(swizzle)+ TMA-stg warp;LSE 自然对数 | 同构(r5 lever B) | 平 |
| 掩码/varlen | MASK 家族 + THD varlen 同 kernel;dead-Q-tile KV 塌缩 | causal 差异化 trip;varlen 另设计 | 借鉴:persistent 化时把 empty-tile 塌缩路径一并搬(varlen 融合的伏笔) |
| smem(d128) | f16 192KB;fp8 sQ32+sK32+sV32+sO64=160KB | 160KB(sQ32+ring64+sO64) | 平;cga2 后我们余 ~32KB 可加深 ring |
| TMEM | S0@0 S1@128 O@256/384 共 512;P/stats 叠 S | 同款(vec/P 叠 S) | 平 |

结论:数学与流水的「密度」两家已同级;**cudnn 多出的全是调度层
(persistent/LPT/cga2)与惰性化(threshold+ballot skip)**。这与 §1 的形状
分布吻合——我们输最多的恰是调度敏感段。

---

## 6. 路线

**Phase A(小改,先长序列翻正;全部可独立回退)**
1. §4.1 correction ballot skip(准 bit-exact,golden 大概率不动)
2. §4.2 G2 sKScale smem 预载 + 删 lever A 残留(bit-exact)
3. §4.3 P 分块交付(保列序版,bit-exact)

预算:+3~8% 长序列。对照 §1.1:b4 s32k/131k(需 4-6%)与 causal b4 长序列
(需 2-3%)应全部翻正;b4 s16k(8.3%)看运气;b1 侧(9-26%)大概率不够。
XU 利用率预计 54.5→57-60%。

**Phase A′(sim 先行的精度档)**
4. §4.5 threshold 惰性 max(T=4 起步,g1_softmax_sim 扩展 → accuracy gate)

**Phase B(persistent + LPT,拿下剩余长序列与中段)**
5. §4.4:p0d probe → NATURAL persistent → LPT。预算:长序列 b1 +2~5%、
   s4096 +5~15%、s1024 b4 +15~25%、causal 中段更多。
   目标:d128 s≥16k 全点 ≥1.0(含 causal b1 s16k 的 25.9% 大坑:
   4.1+4.5+4.4 三项都对它有效,若仍不够,这个点就是 Phase C 的立项证据)。

**Phase C(结构级,全面超越)**
6. §4.6 cga2:s4096 谷地(cutedsl 水平也只 0.95)与 b1 供给问题的最终解;
   与 persistent 已铺的 cluster 语义合并实施。

**不进本役**:d64(独立重设计)、fp8-QK 形态(产品决策)、G3/A5、软件 exp2、
TMA multicast、wg 错拍。

验收口径(每 phase 相同):golden(bitwise 或 accuracy 按各项标注)→
bench 22 点 + gap sweep(§6.5 协议,cudnn 同场重测)→ ncu 三件套
(duration、XU%、分角色 stall)→ 压测 2×8000。全绿才进下一 phase。

---

## 7. 留给下个 B200 会话:cudnn ncu 采集清单(交付物 3)

目的:把 §2 的 DSL 情报升级成闭源 kernel 的实测事实,并补齐我们自己的
瓶颈像缺口。全部在验收会话一次跑完,预算 ~1.5h。

### 7.1 kernel 名发现(-k 过滤前置步)

```
# 1) 先 nsys 拿 cudnn kernel 名(torch SDPA cudnn backend)
nsys profile -o /tmp/cudnn_name --stats=true python bench/sm100_review/ws_prof.py ... (改 --engine cudnn 版)
nsys stats --report cuda_gpu_kern_sum /tmp/cudnn_name.nsys-rep | head
# 预期名族(cudnn-frontend 注释泄露):*native_sdpa_prefill_sm100* / cudnn_generated_fort_*
# 2) ncu 用 regex 过滤
ncu -k "regex:sdpa|fort|cudnn" --launch-skip 2 --launch-count 1 ...
```

驱动:把 `bench/sm100_review/ws_prof.py` 加 `--engine {ws,cudnn}`(cudnn 分支照
`cutedsl-sage-sm100:profile/sm100-tuning-baseline/harness/prof_harness.py` 的
SDPBackend.CUDNN_ATTENTION 写法,~15 行)。

### 7.2 形状(6 个,ncu 每形状 sage/cudnn 各一发)

| 形状 | 看什么 |
|---|---|
| d128 nc b4 s16384 | 主锚点(历史序列可比) |
| d128 nc b1 s16384 | b1/b4 的 L2 与调度差(cudnn +6% 我们 0%,§1.1) |
| d128 nc b4 s4096 | 中段谷地归因(波重启 vs 其他) |
| d128 causal b1 s16384 | 最深长序列坑(0.794) |
| d128 causal b4 s4096 | causal 中段 + LPT 证据 |
| d64 nc b4 s16384 | d64 立项素材(顺带) |

### 7.3 metrics 集(一条命令拿全)

```
--set full --section PmSampling --section PmSampling_WarpStates -s 2 -c 1
```
重点读数(cudnn 侧首采,sage 侧对照):

- 调度证据:`launch__grid_size`(≪tile 数 ⇒ persistent;=tile 数且时长 ≫wave
  也可能 CLC——再看 PM 时间线首尾)、`launch__cluster_size`/`launch__cluster_dim_x`
  (=2 ⇒ cga2)、`launch__waves_per_multiprocessor`、
  `launch__registers_per_thread`(=128 ⇒ setmaxnreg 全特化)、
  `launch__shared_mem_per_block`(≈192KB fp16 ⇒ §2 布局坐实)。
- pipe 像:`sm__throughput`、`sm__inst_executed_pipe_xu`、
  `sm__pipe_tensor_cycles_active`(注明整条 pipe 口径)、alu/fma/lsu、
  `dram__bytes_read.sum`、`lts__t_sectors_op_{read,write}.sum`。
  **cudnn 的 XU% 是 §3 形态税论证的终审**(预期 ~35-45%,若也 55%+ 则
  I2F-税理论要修)。
- 停顿像:`smsp__warp_issue_stalled_*`(long_scoreboard/barrier/wait/
  not_selected)+ PmSampling 时间线(cudnn 尾波形状,LPT 是否可见)。
- M4(我们侧,定 KV ring 议题):mma warp 视角 `wait_kv` 是否成 stall
  (PM sampling 按 warp 角色分道读 barrier 停顿)。
- M6(我们侧,定 §3 taxonomy):`sm__inst_executed_pipe_xu` 与
  `smsp__sass_inst_executed_op_*`(I2F/F2FP/MUFU 计数)相除,确认转换指令
  是否计入 xu pipe。

### 7.4 开箱对表实验(强烈建议,~20min)

容器里 `pip install nvidia-cudnn-frontend==1.27.0`,直接跑它的 DSL d128 kernel
(`cudnn.sdpa` python API)同 6 形状 bench + ncu:

- DSL vs 闭源 cudnn 的 duration 差 <5% ⇒ §2 情报升级为「结构等价实现」,
  后续所有结构问题直接读源码,不再猜;
- 顺带把 SCHEDULER_POLICY 三档(NATURAL/LPT/LPT_L2)各跑一遍 causal s4096/s16k,
  白拿 cudnn 自己的调度消融数据,直接标定 §4.4 的 LPT 预期收益。

### 7.5 我们侧回归锚

Phase A 每合一项重采 d128 nc b4 s16384 全 set(历史序列:r5 16.35 → G1 14.69 →
目标 ≤13.5);XU% 目标带 57-62%。bench 协议照 §6.5(双向轮换 3 轮 median,
cudnn 同场重测,不引历史值)。

### 7.6 wave14 采集实录(B200,cudnn 9.24.0;§7.2 清单的 b4 四形状)

数据:集群 `logs-w14/ncu/`(cudnn_* 与 ws_* 各 4 形状,sections 版 +
XU metrics 版;ws s16384c0 的 sections = `g2_ws_s16384_c0`)。kernel 名
(torch profiler):`cudnn_generated_fort_native_sdpa_sm100_flash_fprop_f16_
knob_1_128x128x128_4x1x1_cga1x1x1_kernel0_0`。

**调度证据(§7.3 首采,四形状一致)**:grid == tile 数(b4h32:8192/2048)
→ **非 persistent**;`cga1x1x1` → **无 cluster**(§2 的 cta_group::2 情报只
适用其 DSL 变体,闭源 prefill 没开);512 线程 / 128 reg(setmaxnreg 全特
化)/ dyn smem 232.45KB(fp16 KV ring,比我们 163.84KB 多 68.6KB)。

| 指标(b4h32,c0/c1 = causal) | cudnn s16k c0 | ws s16k c0 | cudnn s16k c1 | ws s16k c1 | cudnn s4k c0 | ws s4k c0 | cudnn s4k c1 | ws s4k c1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| duration(ncu 锁频) | 12.03 ms | 14.48 ms | 6.33 ms | 7.55 ms | 768 µs | 975 µs | 430 µs | 594 µs |
| SM throughput % | 84.1 | 55.3 | 84.1 | 53.4 | 81.7 | 51.5 | 75.4 | 43.6 |
| tensor pipe active % | **82.0** | 27.2 | 81.8 | 26.3 | 79.3 | 25.3 | 72.5 | 21.4 |
| XU % of peak | **62.0** | 55.3 | 62.1 | 53.4 | 60.3 | 51.5 | 55.6 | 43.6 |
| eligible / issued per sched | 0.73/0.51 | 0.50/0.40 | 0.75/0.53 | 0.48/0.40 | 0.72/0.52 | 0.48/0.39 | 0.74/0.52 | 0.46/0.38 |
| stall long_scoreboard / wait(per issue) | 3.97/0.92 | 5.04/1.83 | 3.64/0.89 | 5.03/1.82 | 3.76/0.87 | 5.01/1.80 | 3.77/0.89 | 5.03/1.77 |
| L2 read sectors | 1.087G | 1.087G | 1.139G | 0.569G | 90.0M | 65.2M | 84.9M | 39.5M |
| inst executed(总) | 5.33G | 6.22G | 2.81G | 3.18G | 341M | 404M | 199M | 221M |
| 其中 pipe_xu | 0.81G(15.3%) | 1.09G(17.5%) | 0.41G | 0.55G | 50.9M | 68.3M | 27.2M | 35.3M |

**M6 裁决(§3 形态税 / I2F-税理论)**:cudnn XU 55.6-62.1% of peak,
**高于**我们的 43.6-55.3%,远超预期带(35-45%)→ **I2F/F2FP 税不是差距
主因,理论修正**。两侧 XU 指令占比接近(15.3% vs 17.5%);我们总指令多
14-17%,但真正的分界在:(1) cudnn 的 SOL 限制器是 tensor pipe
(72-82% active,fp16 MMA 每 FLOP 占 2× tensor 周期,把非 tensor 工作全部
藏进 MMA 阴影里);我们 int8/fp8 MMA 只占 27%,阴影太短,XU/发射效率直接
暴露成关键路径。(2) 发射效率:cudnn eligible 0.72-0.75 / issued
0.51-0.53,我们 0.46-0.50 / 0.38-0.40;per-issue 依赖等待 cudnn
long_scoreboard 3.6-4.0 + wait 0.9,我们 5.0 + 1.8——**每发射的定长依赖
等待(wait)是 cudnn 的 2 倍**,softmax 串行链仍是主攻面。(3) causal 侧
cudnn L2 read 反而比 nc 多(1.139G vs 1.087G,我们 0.569G 减半)——它用
多读换调度均匀,佐证 §4.4 的 LPT/重排方向。persistent/CLC 与 cga2 在闭源
prefill kernel 里都没开,§4.4/§4.6 的预期收益不能再拿「cudnn 有」背书,
要靠自己的消融立项。

## 附录 A:黑名单对照总表(立项前必查)

| 已判不可行项 | 判据出处 | 边界(什么情况不适用) |
|---|---|---|
| 软件 exp2(f32 Horner) | REPORT「实施与结果」全线回退 | MUFU >90% 饱和 + SASS 级验证才可重开 |
| 4+ outstanding tcgen05.ld(A5) | 挂死,根因未定位 | 根因定位或 ptxas 修正;cudnn 也只用 2 条 |
| B1 KV ring 加深 | 0ba30c1(128 线程地基,geomean 1.0004 方向不一致) | ws 地基 + mma 实测卡 wait_kv(§7 M4)可重估;优先随 cga2 |
| C2 TMEM 腾挪 | 0ba30c1(0.6508,22 点全慢) | 仅限 128 线程地基的腾挪;ws 的 TMEM 布局已是 cutedsl 同款 |
| CTA_Q=128 单 tile 变体 | REPORT 追加(crossover 0.70-1.01) | 针对调度粒度;persistent(§4.4)不属此类 |
| sm90 WS 四件套 / cluster TMA multicast | sm90 记忆与 REPORT(H1/H4) | sm90 专属;sm100 的 cta_group::2 是不同机制(§4.6);但 sm100 上纯 TMA multicast 共享 KV 同样不立项 |
| S0/S1 MUFU 错拍(USE_SEQ_GATE) | REPORT(删除净优,默认关) | PM sampling 出现同相踩踏证据可重开(§4.7) |
| lever A ld-shadow 预取单独立项 | C1 §9.6 四态归因(长序列 +4.8~7.1% 劣化) | G1 已吸收;G2 落地时顺手删残留 |
| f16x2 EX2(本轮新增) | §3 XU 计数平手(320=320) | §7 M6 若证明 F2FP/I2F 不占 xu pipe,重算 |

## 附录 B:本轮情报文件索引

- cudnn DSL:`site-packages/cudnn/sdpa/fwd/kernels/prefill_d128_{f16,fp8}_sm100.py`、
  `config_sm100.py`、`_common_sm100.py`、`cudnn/frost/tile_dsl/scheduler.py`
  (nvidia-cudnn-frontend 1.27.0;本机 torch env)
- 四引擎大表:`git show eb1d3e7:profile/sm100-tuning-baseline/REPORT.md`
- w11 逐形状 ms:主控 scratchpad `w11/bench/*.json`(median 表即 §1.1)
- w11 ncu:`w11/ncu/g1_ws_s16384_c0.{raw,details}.csv`
- SASS XU 计数:本 worktree 重编 `bench/sm100_review/qk_int_sv_f8_cuda_sm100_ws_probe.cu`
  (nvcc 13.3 sm_100a,cuobjdump 统计;命令同 C1_DESIGN §5)
