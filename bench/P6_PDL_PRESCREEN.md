# P6 PDL 前筛:sm120 kernel 间 gap 测量

判定:**不立项(no-go)**。

判据是 kernel 间 gap 合计 ≥2% e2e 才立项(背景见 `test/HARDWARE_CHECKLIST.md` 5c:seq ≤1024 时前处理占 e2e 45–53%,但那是 kernel busy 时间;PDL 只能缩 gap,不动 busy)。实测 b4 h32 d128 HND 下,gap 合计占 GPU e2e 的 0.13%(seq4096)到 2.07%(seq512),有 kineto 交叉的形状取较小读数、其余取 nsys 读数(profiler 只会高估真 gap)后,全形状 ≤1.59%。gap 的绝对量是每调用 2–4 µs 的固定 launch 开销,与 seq 无关;其中真正 PDL 可及的 kernel→kernel 邻接边只有 1.0–1.7 µs/调用,最大的单一来源(1 KB memset 前后两条边合计 ~3.4 µs)不是 kernel 对 kernel,PDL 机制上够不到。

两个越过 2% 的 nsys 读数(seq512 非 causal 2.07%、seq512 causal 3.48%)被 kineto 交叉验证证实是 nsys 拖慢提交端注入的伪 gap,见「交叉验证」一节。

## 名词约定

- gap:GPU 时间线上相邻两个 GPU op(kernel 或 memset)之间流内空闲的时间,memcpy 已排除(稳态窗口内本来就没有 H2D/D2H)。
- 调用内 gap:一次 `sageattn` 调用内部各依赖边上的 gap,PDL 的目标区。
- 跨调用 gap:上一调用 attention kernel 结束到下一调用第一个 GPU op 之间的 gap。

## 测试条件

| 项 | 值 |
|---|---|
| 机器 / GPU | pro-5k,GPU 0(NVIDIA Graphics Device,sm120,cc 12.0) |
| GPU 独占 | 是。开跑前 `nvidia-smi` 全部 8 卡 0 MiB / 0% util,测量期间仅本进程,跑完复查仍空载 |
| 容器 / venv | `sglang-diffusion-triplemu-inference`,`/workspace/sgl-env` |
| torch / nsys | 2.13.0+cu132 / 2026.3.1.117 |
| 被测树 | `/workspace/sage-w2` 预构建树(容器内非 git,无 hash;kernel 结构与本分支 `sageattention/core.py` 一致,每调用 6 个 GPU op) |
| 测量脚本 | 本分支 `bench/p6_pdl_driver.py` + `bench/p6_pdl_gaps.py` |
| 形状 | batch 4,heads 32,head_dim 128,HND,fp16,seq ∈ {512, 1024, 2048, 4096},causal 主测否、补测 512/1024 两点 |
| 数据 | `torch.randn`,seed 0 |
| 迭代 | warmup 30 + 测量 100,循环内不 sync(自由跑,launch 队列排满);分析时掐头去尾,取中间 98 次调用 |
| 重复次数 | 每形状 nsys 采 1 次;512/1024/512-causal 另各采 1 次 kineto 交叉;wall-clock 每形状另跑 1 次无 profiler 对照 |

循环内不逐次 sync 是有意的:CPU 把 launch 排在前头,GPU 时间线上的 gap 才反映 launch/调度延迟本身——这正是 PDL 能作用的部分。若真实 pipeline 是 CPU launch-bound,对策是 CUDA graphs,不是 PDL。

## 每形状 gap 占比(nsys 主口径)

占比分母是 GPU window(中间 98 次调用首 op 开始到末 op 结束);e2e 列是无 profiler 的 wall-clock 对照。

| seq | causal | e2e(µs/调用) | GPU window(µs/调用) | gap 合计 | 调用内 gap | 跨调用 gap | 调用内 gap(µs/调用) |
|---|---|---|---|---|---|---|---|
| 512 | 否 | 101.8 | 102.4 | 2.07% | 1.82% | 0.25% | 1.9 |
| 1024 | 否 | 339.2 | 339.4 | 1.29% | 0.72% | 0.58% | 2.4 |
| 2048 | 否 | 991.3 | 996.4 | 0.45% | 0.25% | 0.20% | 2.5 |
| 4096 | 否 | 3462.0 | 3481.6 | 0.13% | 0.07% | 0.06% | 2.4 |
| 512 | 是 | 91.9 | 93.5 | 3.48% | 3.20% | 0.29% | 3.0 |
| 1024 | 是 | 279.4 | 278.8 | 1.59% | 0.87% | 0.72% | 2.4 |

## 交叉验证:nsys 的伪 gap

kineto(`torch.profiler`,同 driver 同形状)给出系统性更低的 gap 读数:

| seq | causal | gap 合计(nsys) | gap 合计(kineto) | 调用内 gap(nsys) | 调用内 gap(kineto) |
|---|---|---|---|---|---|
| 512 | 否 | 2.07% | 1.39% | 1.82% | 1.12% |
| 1024 | 否 | 1.29% | 1.15% | 0.72% | 0.76% |
| 512 | 是 | 3.48% | 1.55% | 3.20% | 1.24% |

差异全部落在 op 边界的两条边上:seq512 causal 时 nsys 读到 k.mean→quant Q 1.17 µs、quant K→V transpose 1.22 µs,同两条边 kineto 只有 0.27–0.34 µs,与大 seq 下所有 kernel→kernel 边的稳态值一致。调用越短,nsys 对 CUDA API 的逐调用记录越拖慢提交端,伪 gap 越大;nsys 下 wall-clock 也被抬高(seq512 非 causal +4.1%)。两种 profiler 的读数都只会高估真 gap,所以判定取每形状两者较小值,即便如此也全部 <2%。

## gap 落在哪些依赖边(nsys,非 causal,均值 µs/边)

| 依赖边(前 op → 后 op) | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| attention → 下一调用首 op(跨调用) | 0.26 | 1.98 | 2.03 | 1.94 |
| memset → K mean | — | 1.43 | 1.46 | 1.44 |
| K mean → Q quant | 0.76 | 0.25 | 0.25 | 0.25 |
| Q quant → K quant | 0.30 | 0.25 | 0.25 | 0.26 |
| K quant → V transpose | 0.52 | 0.26 | 0.25 | 0.25 |
| V transpose → attention | 0.28 | 0.25 | 0.24 | 0.25 |

账能对上,gap 只有两种来源:

1. kernel→kernel 邻接边 0.25 µs/边(sm120 队列排满时的调度延迟;512 列偏大的 0.3–0.8 µs 是上节的 nsys 伪 gap,kineto 下同边 0.27–0.34 µs)。每调用 4–5 条边,合计 1.0–1.7 µs——这是 PDL 可及收益的全部,seq1024 上占 e2e 0.3–0.5%。
2. 1 KB memset 前后两条边 1.4 + 2.0 µs(seq ≥1024 才有这个 memset;来源未查)。它两侧的邻接方有一侧不是 kernel,PDL 机制上不适用;真要省这 3.4 µs,方向是消掉这个 memset 本身。
3. python 循环开销在队列排满时被完全遮住:seq4096 时 GPU 积压 ~340 ms 工作,跨调用边仍只有 1.94 µs 且贴着 memset;seq512(无 memset)跨调用边只有 0.26 µs。

## 每 kernel 用时(nsys,非 causal,均值 µs / 占 window)

| kernel | seq512 | seq1024 | seq2048 | seq4096 |
|---|---|---|---|---|
| attention(`qk_int_sv_f8_attn_kernel`) | 48.1(47.0%) | 176.6(52.0%) | 656.6(65.9%) | 2660.6(76.4%) |
| V transpose + fp8 quant(`TransposeQuantFp8Kernel`) | 25.8(25.2%) | 50.3(14.8%) | 115.5(11.6%) | 367.9(10.6%) |
| K int8 quant(`QuantPerThreadKInt8Kernel`) | 10.7(10.4%) | 38.6(11.4%) | 89.9(9.0%) | 179.2(5.2%) |
| K mean(ATen `reduce_kernel`,smooth_k) | 8.4(8.2%) | 38.4(11.3%) | 68.8(6.9%) | 125.8(3.6%) |
| Q int8 quant(`QuantPerThreadQInt8Kernel`) | 7.2(7.1%) | 30.2(8.9%) | 60.2(6.0%) | 142.7(4.1%) |
| memset 1 KB | — | 0.9(0.3%) | 0.9(0.1%) | 0.9(<0.1%) |

前处理 busy 占比 47–51%(seq ≤1024),与 checklist 5c 的 45–53% 对得上;但这些是 busy 时间,前处理提速属于 kernel 优化的事,与 PDL 无关。与 checklist 5c 相比,本树 V 路只剩一个 `TransposeQuantFp8Kernel`(独立的 MeanScale kernel 与 fp8 零填充 kernel 已不存在),故每调用 op 数是 6 不是 7–8。

## 没做到的

- 没做 PDL 原型:前筛已 no-go,不再上机验证。
- attention prologue 能被 PDL 提前的部分没有实测量化。定性推断(未验证):attention kernel 第一批依赖 load(q/k int8、scale)之前只有地址计算,提前量亚 µs 级,不改变结论。
- seq ≥1024 那个 1 KB memset 的来源没查(它两侧 3.4 µs 的 gap 是最大单一来源,若有人想追,这是比 PDL 更直接的 3 µs/调用级别线索)。
- `/workspace/sage-w2` 无 commit hash,被测二进制的源码状态只能对齐到「与本分支 core.py 的调用结构一致」。

## 限制与偏差方向

- 两种 profiler 都拖慢提交端、抬高 gap 读数,偏差方向对「立项」有利;即便取被抬高的读数,唯二越线的点也被交叉验证排除。结论向 no-go 方向是稳的。
- 每形状 nsys 只采 1 次,但 6 个形状 × 2 工具的调用内 gap 绝对量一致落在 1–3 µs/调用,且与逐边求和对账吻合,单次采样风险低。
- 只测了 b4 h32 d128。更小 batch 或更短 seq 时 e2e 更短,gap 占比会更高,但可及绝对量仍是每调用 1–2 µs 的固定值(推算),e2e 需短到 ~50 µs 才够到 2% 线。

## 可复现命令(容器内)

```bash
source /workspace/sgl-env/bin/activate
cd /workspace/p6_pdl   # 或任意放了 bench/ 两个脚本的目录
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/sage-w2

# wall-clock 对照(无 profiler)
python bench/p6_pdl_driver.py --seq 1024 --warmup 30 --iters 100 --time-only

# nsys 主口径
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
    -t cuda -s none --cpuctxsw=none -o seq1024 --force-overwrite true \
    python bench/p6_pdl_driver.py --seq 1024 --warmup 30 --iters 100
nsys stats --report cuda_gpu_trace --format csv --output seq1024 seq1024.nsys-rep
python bench/p6_pdl_gaps.py seq1024_cuda_gpu_trace.csv --iters 100 --label seq1024

# kineto 交叉
python bench/p6_pdl_driver.py --seq 1024 --warmup 30 --iters 100 --kineto kineto_seq1024.json
python bench/p6_pdl_gaps.py kineto_seq1024.json --iters 100 --label kineto_seq1024
```

causal 点加 `--causal`。

## 产物路径

全部在 pro-5k 容器 `sglang-diffusion-triplemu-inference` 内,未回传本地(报告表格已含全部结论数字):

- `/workspace/p6_pdl/results.txt`:非 causal 四形状的 time-only + nsys 分析全文
- `/workspace/p6_pdl/results_causal.txt`、`results_kineto.txt`:causal 两点与 kineto 三点的分析全文
- `/workspace/p6_pdl/seq{512,1024,2048,4096}{,c}.nsys-rep` 及同名 `.sqlite`、`_cuda_gpu_trace.csv`:原始 trace
- `/workspace/p6_pdl/kineto_seq{512,1024,512c}.json`:kineto chrome trace
