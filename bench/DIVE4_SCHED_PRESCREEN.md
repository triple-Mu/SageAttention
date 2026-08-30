# Dive 4 前筛:varlen 偏斜 batch 的 CTA 早退浪费(sm120)

判定:**不立项(no-go)**,FA3 式 persistent scheduler / CTA 重排不做。

判据是偏斜 .25-1x / .1-1x 档 attention kernel 级调度浪费 ≥5% 才立项。两个独立口径都没到线:

1. 时长口径(kineto 主测,nsys 交叉):任务口径的原始上界(与同 total token 的等长对照直接比,下表 raw 列)在 20 个判据档读数里 9 个 ≥5%、最高 11.7%;但按 kernel 实际执行的 KV tile 数归一(norm 列)后,20 个读数全部落在 −9.0% 到 +3.6%,0 个 ≥5%。raw 超线的部分全部来自工作量差:同 token 数下偏斜 batch 的 Σnᵢ² 更大,判据档要多做 5%–12% 的 KV tile(tile 比列),persistent scheduler 不减少 tile 数,这部分本来就收不回。
2. SM 空转口径(ncu):同一 grid 下,equal(空 CTA 0%)对偏斜(空 CTA 21.9%–32.8%)的 idle 占比不升反降(causal n1024:16.0%→13.1%),8 个判据档读数的 idle 增量全在 −3.0 到 −0.1 个百分点(pp)。idle 的大小由 kernel 波数决定,与偏斜无关——早退 CTA 被硬件 work distributor 即时回填,成本在测量精度内为零。

另外两点进一步压缩立项空间:causal kernel 已内建重 CTA 先发(`qk_int_sv_f8_cuda_sm89.cuh` 中 causal 时 `cta_idx_q = gridDim.x-1-blockIdx.x`),FA3 scheduler 的主要均衡手段已部分存在;人为构造的极端双峰负载(1×4096 + 7×128,空 CTA 84.8%,判据档之外的补充点)下 norm 仍为负(−6.8% / −9.2%),idle 超出同尺寸等长对照 1.4 / 5.2 pp——即便全记作可收回,也只有 kernel 的 ~5%、e2e 的 ~3.6%。attention kernel 在这些负载里占 e2e 的 41%–85%,kernel 级 <5% 意味着 e2e 级更小。

## 名词约定

- 空 CTA:varlen grid.x 开到 ceil(max_seqlen_q/128)(`csrc/qattn/qk_int_sv_f8_varlen_launcher_sm89.cuh`),序列自身长度之外的 query-block CTA 在 kernel 入口 `cta_idx_q * CTA_Q >= qo_len` 处直接 return。
- eqtok 对照:与偏斜档 total token 相同、各序列等长(除法余数逐条 +1 摊开)的 varlen 负载;max_seqlen 随之变小,grid 也变小。
- raw:1 − t_eqtok / t_skew,任务规定的调度浪费上界(时长均为 attention kernel 的 kineto GPU 时长均值)。
- tile 比:偏斜与 eqtok 两侧 KV tile 总数之比。KV tile 数按 kernel 的迭代公式精确计数(causal 截断到对角线,`qk_int_sv_f8_cuda_sm89.cuh` 的 num_iterations),即 kernel 真实执行的 MMA 工作量。
- norm:1 − t_eqtok × tile比 / t_skew,即「每 tile 效率与 eqtok 相同、完美打包的 kernel 跑偏斜工作量所需时间」对实测时长的差。这才是调度(早退 + 尾波不均)可收回部分的上界。
- idle:1 − sm__cycles_active.avg / sm__cycles_elapsed.max,kernel 时长内平均每 SM 的空转占比(含 launch ramp 与收尾 drain)。

## 测试条件

| 项 | 值 |
|---|---|
| 机器 / GPU | pro-5k GPU 0(NVIDIA Graphics Device,sm120,cc 12.0,110 SM) |
| GPU 独占 | 是。测前/测后 nvidia-smi 全 8 卡 0 MiB,测量期仅本进程 |
| 容器 / venv | `sglang-diffusion-triplemu-inference`,`/workspace/sgl-env` |
| torch / nsys / ncu | 2.13.0+cu132 / 2026.3.1.117 / 2026.2.1.0 |
| 被测树 | `/workspace/sage-dive4` = 本仓 5af2e06 的 git archive,TORCH_CUDA_ARCH_LIST=12.0 `build_ext --inplace`(测完已删,build.log 保留) |
| 被测 kernel | `sage::sm120_varlen::qk_int_sv_f8_attn_kernel`,CTA_Q 128 / CTA_K 64;实测 2 CTA/SM 共驻(1024 CTA ÷ 4.65 波 = 220/波 = 110 SM × 2) |
| 负载 | bench/bench_varlen.py 的 seqlens(seed 0,最长序列钉在 seq_len,同形状各 profile 的 grid 相同);fp16 randn;5 个 SHAPES × causal 0/1 × profile equal / .5-1x / .25-1x / .1-1x,另加双峰补充点 |
| 迭代 | 时长口径每配置 warmup 10 + 无 profiler wall-clock 30 + kineto 30(kernel 时长取均值);ncu 跳过前 12 个 launch 取 3 个平均(默认锁频);nsys 交叉 40 launch 均值 |
| 脚本 | bench/dive4_sched_driver.py(测量)、bench/dive4_sched_analyze.py(归约) |

## 口径 1:attention kernel 时长(kineto,µs/调用)

判据档是 .25-1x 与 .1-1x(共 5 形状 × 2 causal × 2 档 = 20 个读数);.5-1x 与双峰补充点一并列出。

| shape | causal | profile | skew µs | eqtok µs | tile 比 | raw% | norm% | 空 CTA% |
|---|---|---|---|---|---|---|---|---|
| b8 h16 n1024 d128 | 0 | equal | 176.7 | — | — | — | — | 0.0 |
| b8 h16 n1024 d128 | 0 | ragged .5-1x | 135.3 | 131.0 | 1.084 | 3.2 | −4.9 | 10.9 |
| b8 h16 n1024 d128 | 0 | ragged .25-1x | 108.3 | 95.6 | 1.106 | 11.7 | 2.4 | 21.9 |
| b8 h16 n1024 d128 | 0 | ragged .1-1x | 90.7 | 87.2 | 1.013 | 3.8 | 2.5 | 29.7 |
| b16 h8 n2048 d64 | 0 | equal | 375.7 | — | — | — | — | 0.0 |
| b16 h8 n2048 d64 | 0 | ragged .5-1x | 256.7 | 258.0 | 1.040 | −0.5 | −4.5 | 16.8 |
| b16 h8 n2048 d64 | 0 | ragged .25-1x | 209.3 | 201.2 | 1.003 | 3.9 | 3.6 | 27.3 |
| b16 h8 n2048 d64 | 0 | ragged .1-1x | 177.3 | 170.1 | 1.010 | 4.0 | 3.1 | 35.2 |
| b8 h16 n4096 d128 | 0 | equal | 2527.2 | — | — | — | — | 0.0 |
| b8 h16 n4096 d128 | 0 | ragged .5-1x | 1821.1 | 1772.7 | 1.050 | 2.7 | −2.3 | 17.2 |
| b8 h16 n4096 d128 | 0 | ragged .25-1x | 1493.2 | 1371.0 | 1.091 | 8.2 | −0.2 | 26.6 |
| b8 h16 n4096 d128 | 0 | ragged .1-1x | 1310.5 | 1224.8 | 1.074 | 6.5 | −0.4 | 32.8 |
| b8 h16 n4096 d128 | 0 | 双峰 1×4096+7×128 | 335.4 | 69.5 | 5.155 | 79.3 | −6.8 | 84.8 |
| b8 h32 n4096 d64 | 0 | equal | 2941.7 | — | — | — | — | 0.0 |
| b8 h32 n4096 d64 | 0 | ragged .5-1x | 2174.0 | 2117.3 | 1.050 | 2.6 | −2.3 | 17.2 |
| b8 h32 n4096 d64 | 0 | ragged .25-1x | 1785.5 | 1653.6 | 1.091 | 7.4 | −1.1 | 26.6 |
| b8 h32 n4096 d64 | 0 | ragged .1-1x | 1567.6 | 1468.0 | 1.074 | 6.4 | −0.6 | 32.8 |
| b4 h16 n8192 d128 | 0 | equal | 5079.4 | — | — | — | — | 0.0 |
| b4 h16 n8192 d128 | 0 | ragged .5-1x | 4218.8 | 4190.0 | 1.008 | 0.7 | −0.1 | 11.3 |
| b4 h16 n8192 d128 | 0 | ragged .25-1x | 3714.5 | 3588.6 | 1.031 | 3.4 | 0.4 | 17.6 |
| b4 h16 n8192 d128 | 0 | ragged .1-1x | 3487.2 | 3284.0 | 1.074 | 5.8 | −1.1 | 21.1 |
| b8 h16 n1024 d128 | 1 | equal | 116.5 | — | — | — | — | 0.0 |
| b8 h16 n1024 d128 | 1 | ragged .5-1x | 92.8 | 91.2 | 1.057 | 1.7 | −3.8 | 10.9 |
| b8 h16 n1024 d128 | 1 | ragged .25-1x | 74.8 | 73.0 | 1.116 | 2.4 | −9.0 | 21.9 |
| b8 h16 n1024 d128 | 1 | ragged .1-1x | 64.2 | 69.9 | 0.973 | −8.8 | −5.9 | 29.7 |
| b16 h8 n2048 d64 | 1 | equal | 222.2 | — | — | — | — | 0.0 |
| b16 h8 n2048 d64 | 1 | ragged .5-1x | 159.6 | 154.2 | 1.059 | 3.4 | −2.3 | 16.8 |
| b16 h8 n2048 d64 | 1 | ragged .25-1x | 129.8 | 131.6 | 0.988 | −1.4 | −0.2 | 27.3 |
| b16 h8 n2048 d64 | 1 | ragged .1-1x | 111.5 | 112.7 | 0.976 | −1.1 | 1.3 | 35.2 |
| b8 h16 n4096 d128 | 1 | equal | 1385.7 | — | — | — | — | 0.0 |
| b8 h16 n4096 d128 | 1 | ragged .5-1x | 983.1 | 947.6 | 1.057 | 3.6 | −1.9 | 17.2 |
| b8 h16 n4096 d128 | 1 | ragged .25-1x | 803.9 | 755.8 | 1.100 | 6.0 | −3.4 | 26.6 |
| b8 h16 n4096 d128 | 1 | ragged .1-1x | 705.4 | 691.5 | 1.053 | 2.0 | −3.3 | 32.8 |
| b8 h16 n4096 d128 | 1 | 双峰 1×4096+7×128 | 225.9 | 55.3 | 4.458 | 75.5 | −9.2 | 84.8 |
| b8 h32 n4096 d64 | 1 | equal | 1590.1 | — | — | — | — | 0.0 |
| b8 h32 n4096 d64 | 1 | ragged .5-1x | 1145.2 | 1107.8 | 1.057 | 3.3 | −2.3 | 17.2 |
| b8 h32 n4096 d64 | 1 | ragged .25-1x | 946.7 | 880.9 | 1.100 | 6.9 | −2.3 | 26.6 |
| b8 h32 n4096 d64 | 1 | ragged .1-1x | 835.2 | 808.3 | 1.053 | 3.2 | −2.0 | 32.8 |
| b4 h16 n8192 d128 | 1 | equal | 2645.2 | — | — | — | — | 0.0 |
| b4 h16 n8192 d128 | 1 | ragged .5-1x | 2207.2 | 2215.5 | 1.005 | −0.4 | −0.9 | 11.3 |
| b4 h16 n8192 d128 | 1 | ragged .25-1x | 1970.8 | 1921.4 | 1.025 | 2.5 | 0.1 | 17.6 |
| b4 h16 n8192 d128 | 1 | ragged .1-1x | 1852.0 | 1718.4 | 1.078 | 7.2 | 0.0 | 21.1 |

raw 与 norm 的差恰好是 tile 比:raw ≥5% 的 9 行 tile 比都在 1.05–1.10(双峰行 4.5–5.2,raw 在那里完全失真),norm 之后没有一行 ≥5%。norm 为负的行(最深 −9.0%)是偏斜侧每 tile 效率反而更高:偏斜保留了钉住的最长序列,其 CTA 的 KV 循环更长、prologue 摊薄更好,而 eqtok 全是中等长度;负值同时标出本口径单点的噪声/结构效应量级(几个百分点)。

## 口径 2:SM 空转(ncu,3 launch 均值)

看点是同 grid 的 equal 行对 skew 行(空 CTA 从 0% 升到 33%,idle 不升);eqtok 行 grid 更小,列作同工作量小 kernel 的 ramp/drain 参照。busy cycle/tile = sm__cycles_active.avg × 110 ÷ KV tile 总数。

| shape | causal | 配置 | grid | 空 CTA% | 波数 | idle% | 最慢 SM busy% | busy cycle/tile |
|---|---|---|---|---|---|---|---|---|
| b8 h16 n1024 d128 | 0 | equal | (8,16,8) | 0.0 | 4.65 | 8.0 | 88.0 | 2816 |
| b8 h16 n1024 d128 | 0 | .25-1x skew | (8,16,8) | 21.9 | 4.65 | 8.0 | 87.4 | 2855 |
| b8 h16 n1024 d128 | 0 | .1-1x skew | (8,16,8) | 29.7 | 4.65 | 7.9 | 86.7 | 2883 |
| b8 h16 n1024 d128 | 0 | .25-1x eqtok | (6,16,8) | 0.0 | 3.49 | 4.4 | 83.2 | 2933 |
| b8 h16 n1024 d128 | 0 | .1-1x eqtok | (6,16,8) | 0.0 | 3.49 | 3.8 | 84.0 | 2955 |
| b8 h16 n1024 d128 | 1 | equal | (8,16,8) | 0.0 | 4.65 | 16.0 | 73.0 | 3040 |
| b8 h16 n1024 d128 | 1 | .25-1x skew | (8,16,8) | 21.9 | 4.65 | 13.1 | 76.2 | 3144 |
| b8 h16 n1024 d128 | 1 | .1-1x skew | (8,16,8) | 29.7 | 4.65 | 13.3 | 74.9 | 3207 |
| b8 h16 n1024 d128 | 1 | .25-1x eqtok | (6,16,8) | 0.0 | 3.49 | 17.4 | 69.6 | 3248 |
| b8 h16 n1024 d128 | 1 | .1-1x eqtok | (6,16,8) | 0.0 | 3.49 | 16.3 | 71.0 | 3228 |
| b8 h16 n4096 d128 | 0 | equal | (32,16,8) | 0.0 | 18.62 | 2.1 | 97.1 | 2593 |
| b8 h16 n4096 d128 | 0 | .25-1x skew | (32,16,8) | 26.6 | 18.62 | 1.6 | 97.6 | 2603 |
| b8 h16 n4096 d128 | 0 | .1-1x skew | (32,16,8) | 32.8 | 18.62 | 1.9 | 96.7 | 2606 |
| b8 h16 n4096 d128 | 0 | .25-1x eqtok | (23,16,8) | 0.0 | 13.38 | 1.5 | 95.6 | 2617 |
| b8 h16 n4096 d128 | 0 | .1-1x eqtok | (22,16,8) | 0.0 | 12.80 | 1.9 | 95.5 | 2627 |
| b8 h16 n4096 d128 | 1 | equal | (32,16,8) | 0.0 | 18.62 | 4.9 | 91.6 | 2638 |
| b8 h16 n4096 d128 | 1 | .25-1x skew | (32,16,8) | 26.6 | 18.62 | 4.1 | 92.7 | 2666 |
| b8 h16 n4096 d128 | 1 | .1-1x skew | (32,16,8) | 32.8 | 18.62 | 3.9 | 93.1 | 2672 |
| b8 h16 n4096 d128 | 1 | .25-1x eqtok | (23,16,8) | 0.0 | 13.38 | 7.0 | 88.1 | 2685 |
| b8 h16 n4096 d128 | 1 | .1-1x eqtok | (22,16,8) | 0.0 | 12.80 | 6.5 | 88.7 | 2692 |
| b8 h16 n4096 d128 | 0 | 双峰 skew | (32,16,8) | 84.8 | 18.62 | 7.5 | 79.4 | 2692 |
| b8 h16 n4096 d128 | 0 | 双峰 eqtok | (5,16,8) | 0.0 | 2.91 | 6.1 | 81.6 | 3075 |
| b8 h16 n4096 d128 | 1 | 双峰 skew | (32,16,8) | 84.8 | 18.62 | 25.9 | 61.1 | 2782 |
| b8 h16 n4096 d128 | 1 | 双峰 eqtok | (5,16,8) | 0.0 | 2.91 | 20.7 | 65.4 | 3456 |

要点:

1. n4096:equal idle 2.1%(c0)/ 4.9%(c1),偏斜档 1.6%–4.1%——空 CTA 到三成,idle 无增量。n1024 causal 更明显:equal 16.0%,偏斜 13.1%/13.3%,偏斜反而低。idle 的主导变量是波数与 kernel 大小(ramp/drain 占比),不是偏斜。
2. busy cycle/tile 非双峰行全落在 2593–3248,skew 与其 eqtok 差 ≤2.5%(n4096 c1 .25:2666 对 2685,0.7%)——每 tile busy 成本几乎不随长度分布变化,norm 口径用 eqtok 每 tile 效率外推的假设成立。
3. 双峰补充点是全篇唯一的疑似正信号:c1 skew idle 25.9%,但同尺寸 eqtok 参照本身 20.7%,超额 5.2 pp(c0 为 1.4 pp);而时长口径的 norm 在这点是 −9.2%(每 tile 效率优势盖过了空转)。空 CTA 84.8% 时 3472 个空 CTA 的发射/回填本身仍测不出成本。

## 交叉验证

P6(bench/P6_PDL_PRESCREEN.md)的教训是 nsys 会给短调用注入伪 gap;本次测的是 kernel 时长不是 gap,时长由 GPU 侧时间戳决定,理论上不受提交端拖慢影响,仍按任务要求双工具交叉(b8 h16 n4096 d128,causal=1,nsys 为 40 launch 均值):

| profile | kineto µs | nsys µs | 差 |
|---|---|---|---|
| equal | 1385.7 | 1397.8 | +0.9% |
| ragged .1-1x | 705.4 | 704.9 | −0.1% |

ncu 侧同样对得上:n4096 c0 .25-1x 的 skew/eqtok elapsed cycle 比 1.087,kineto 时长比 1.089。

## 立项判据核对

| 口径 | 判据档读数 | 范围 | ≥5% 行数 |
|---|---|---|---|
| raw(任务口径上界,含工作量差) | 20(5 形状 × 2 causal × 2 档) | −8.8% … +11.7% | 9 |
| norm(tile 归一,调度可收回上界) | 20 | −9.0% … +3.6% | 0 |
| ncu idle 增量(skew − 同 grid equal) | 8(2 形状 × 2 causal × 2 档) | −3.0 … −0.1 pp | 0 |

raw 的 9 行超线全部被 tile 比解释;两个可收回口径都是 0 行 ≥5%。**no-go,归档。**

## 没做到的

- ncu 只覆盖 n1024 / n4096 两个 d128 形状,d64 与 b4 n8192 没采 idle。d64 的 norm 最高(+3.6%,b16 h8 n2048 c0 .25-1x)但离 5% 有距离,同行 raw 也只有 3.9%。
- 时长口径每配置单次 30 iter 均值,没做重复采样;norm 负值幅度(至 −9%)说明单点有几个百分点的结构效应。判定不依赖单点:20 个判据档读数全体 <5%,且与 ncu 口径互证。
- 只测了 sm120。sm80/89/90 的 varlen 同样把 grid 开到 max_seqlen + 入口早退,work distributor 行为同代,外推方向一致,但没实测;若在别的 arch 重开此项,先复跑本 driver。
- 双峰补充点只测 b8 h16 n4096 d128。其 c1 的 idle 超额 5.2 pp 是全篇最大的疑似可收回量(约 kernel 5%、e2e 3.6%);若未来 serving 负载真是「单条长序列带一群短序列」形态,重开前筛从这行入手。
- 没做 persistent scheduler 原型;前筛未过线,不上原型。

## 可复现命令(容器内)

```bash
source /workspace/sgl-env/bin/activate
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/sage-dive4   # 5af2e06 archive + build_ext --inplace

# 口径 1:全矩阵(kineto),归约出 raw/norm 表
bash /workspace/dive4-logs/run_matrix.sh > matrix.jsonl
python bench/dive4_sched_analyze.py matrix.jsonl

# 口径 2:ncu 20 配置;双峰补充点(kineto + ncu)
bash /workspace/dive4-logs/run_ncu.sh
bash /workspace/dive4-logs/run_extreme.sh

# nsys 交叉(2 配置)
bash /workspace/dive4-logs/run_nsys.sh
```

## 产物路径

全部在 pro-5k 容器 `sglang-diffusion-triplemu-inference` 的 `/workspace/dive4-logs/`,未回传本地(报告表格已含全部结论数字):

- `matrix.jsonl`、`extreme.jsonl`:时长口径逐配置 JSON(kernel 时长、KV tile 数、lens、逐 kernel 分解)
- `ncu/*.csv`:24 个 ncu 配置的原始 metric CSV;`ncu/device.txt` 记录 110 SM
- `nsys_{eq,p10}.nsys-rep` 与同名 `_cuda_gpu_kern_sum.csv`:nsys 交叉原始 trace
- `run_{matrix,ncu,extreme,nsys}.sh`:各步骤的原样脚本
- `build.log`:构建日志(双峰补测前树删过一次,含两次构建)
