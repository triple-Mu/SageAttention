# 上机验证清单(sm89 / sm120 / sm100)

本机(sm_86)只能运行 sm80 路径与 fused 量化 kernel 的 fp16 部分。
**sm90 已在 H200 上全部验证完毕**(dense 2026-08-28、varlen 2026-08-29,
见 §5),**sm120 已在 RTX PRO 6000 Blackwell 上验证完毕**(2026-08-29,
见 §5b),**sm89 已在 L20 上验证完毕**(2026-08-29,见 §5d),
**sm100 已在 B200 上验证完毕**(2026-08-29,见 §5f)。四个 arch 的
bitwise 对拍与 varlen 都有硬件证据了。
sm89 与 sm120 都过了,删除过渡期
`torch.ops.sageattention.qattn_smXX_*` 低层 op(它们是对拍工具)的条件
已经达成,见 §7。

## 0. 一键环境与对拍工具

```bash
pip install -e . --no-build-isolation   # TORCH_CUDA_ARCH_LIST 按机器设置
pytest test/ -q
```

bitwise 对拍用 `tools/compare_reference.py`(单文件,可整个拷走;输入 CPU
固定 seed 跨机复现;两进程用法见其 docstring):

```bash
# baseline 环境(重构前 commit 0a5d2e4 构建,或 pip 旧版):
python compare_reference.py --dump --golden-dir <dir> --backend legacy
# 新构建环境:
python compare_reference.py --check --golden-dir <dir> --backend new
# 预期 ok 全部 diff=0;equiv 段(fwd vs 低层 op 全等、quant_v_fp8 pad 下沉
# 等价、v_fp8 尾部全零、varlen vs dense 全等)只在 new 侧跑,不需要 baseline。
```

varlen 的性能对照用 `python bench/bench_varlen.py`(packed vs 补齐到
max_seqlen 的 dense,同数据同 API,`--csv` 存表)。

## 1. bitwise 对拍(剩余机器)

- [x] sm89(L20,sm_89,2026-08-29):`--dump`/`--check` 全绿,
      `ok=2004 diff=0 env_mismatch=0 missing=0`,equiv 345/345。A4-1 主循环
      合并(见 §3)与 `quant_v_fp8(pad_multiple)` 修复由 equiv 段覆盖。
      数字见 §5d。注:精度准入检查已按 backend 参数化(eb510e5),L20 那轮
      跑在参数化之前——下次上 sm89 卡顺跑 `pytest test/test_accuracy.py`
      补三个 pv_accum_dtype 的实测数值。
- [x] sm89+ 任一实机(sm_120,2026-08-29):fp8 zero-amax 回归(`test_ops.py` 的
      `test_quant_v_fp8_zero_amax_channel` / `test_quant_v_fp8_subnormal_amax_channel`
      / `test_sageattn_zero_v_channel_fp8`,以及 `test_varlen.py` 的
      `test_quant_v_fp8_varlen_zero_amax_channel`,本机 sm86 全部 skip)。
      修复前 V 的全零 channel(或 smooth_v 下常数 channel、bf16 subnormal
      amax)量化成 fp8 NaN 行,污染对应输出 channel;修复后应为精确 0。
      dense 与 packed 是同一个 `MeanScaleKernel`,所以两条路径一起修好。
      对 §1 bitwise 对拍无影响:degenerate channel 只出现在 hd=96 e2e case 的
      head_dim pad 区,dump 前已被 slice 掉。e2e 的 smooth_v 只在 sm120 真正
      生效,其余 fp8 backend 降级。这四个测试的 16 个 case 在 sm_120 上
      全部 PASSED。
- [x] sm120(RTX PRO 6000 Blackwell,2026-08-29):`--dump`/`--check` 全绿。
      cc 12.0 上 sm89 家族与 sm120 家族的 cubin 同时可跑,attn 段两族分别
      对拍(sm120 是同一批 sm89 TU 的 `SAGEATTN_ARCH_NS=sm120` 双编译);
      公共 API 没有 backend 参数,sm89 fallback 只能从低层 `qattn_sm89_*`
      op 这一侧覆盖。数字见 §5b。
- [x] sm100(B200,2026-08-29):SS twin 与 TS 各对同一份 baseline golden
      `--check`,两轮都是 `ok=2280 diff=0`。tcgen05 kernel 第一次上真硬件就
      暴露了两个 launcher bug(都不在 kernel 里,见 §5f),修完才全绿。
      数字见 §5f。SDPA 精度 62/62 ×2 轮已实测;参数化后的
      `test_accuracy.py` 全量数值下次上机顺跑。

## 1b. varlen(packed cu_seqlens 布局)

本机 sm_86 只能跑 sm80 的 packed kernel,以及 fp8 V 流水里唯一 fp16 进
fp16 出的那一级(transpose)。sm120 与 sm89 的 packed attention 都已在
2026-08-29 上机(分别见 §5b、§5d)。

sm90 已在 H200(GPU 5)验完,fp8 V^T 量化这一级也随之有了硬件证据:

- [x] sm90(H200,2026-08-29):packed kernel 与 `sageattn_varlen` API 全绿,
      数字见 §5。
- [x] `quant_v_fp8_varlen` 的 fp8 用例在 H200 跑通。**它在 sm_86 上是
      skip 的,第一次真跑就暴露了断言写错**:`mma_k16` 把 token 在 16 个一组
      内做了置换,跨过段长的那一组里真值和零是交错的,所以「段长之后每个字节
      全零」对 dense 也不成立;`smooth_v=True` 时 padding 的 0 量化成
      `(0 - v_mean) / v_scale`,本来就不是零。断言已按布局和 smooth_v 改写。
- [x] `segment_mean_varlen`(smooth_k 的分段均值 kernel,替换 ATen 的
      repeat_interleave + index_add_ 组合)在 H200 验完(2026-08-29):
      `pytest test/ -q` 339 passed / 319 skipped;`bench/bench_varlen.py`
      等长 batch 对 dense 从 0.49-0.80x 回到 0.93-0.99x,ragged .1-1x
      1.16-1.77x。kernel 在 fused 组、与 arch 无关,sm89/sm120 由下面那条
      pytest 项顺带覆盖。
- [x] sm120(2026-08-29):`pytest test/test_varlen.py test/test_varlen_utils.py -q`
      138 passed / 97 skipped。`sageattn_varlen` 在 cc 12.0 上 resolve 到
      sm120 packed kernel(`pv_accum_dtype="fp32"`、`v_layout="mma_k16"`、
      `v_pad_multiple=64`),等长 batch 对 dense 的 `torch.equal`、ragged 的
      分段 SDPA 精度、bottom-right causal、空 KV 段、cudagraph 换分段
      replay 全过。skip 的 97 个是 pin 在 sm80/sm90 的 kernel 级用例。
- [x] sm89(L20,2026-08-29):`pytest test/test_varlen.py test/test_varlen_utils.py -q`
      138 passed / 97 skipped,与 sm120 同一批。`sageattn_varlen` 在 cc 8.9 上
      resolve 到 sm89 packed kernel(`pv_accum_dtype="fp32+fp16"`、
      `v_layout="mma_k16"`、`v_pad_multiple=64`);`sm89_varlen` 命名空间的
      kernel 这是第一次真在硬件上执行。API 级 41 个用例(等长 `torch.equal`、
      ragged 分段 SDPA、bottom-right causal、空 KV 段、GQA + head_dim pad、
      cudagraph 换分段 replay、compile 无 graph break)全过。
- [x] sm100(B200,2026-08-29):**设计内不支持**,只验报错可读。tcgen05 kernel
      的 K/V 走每个 batch entry 一份 tensor map,packed 布局要另写一份,所以
      `resolve()` 直接拒绝而不是静默降级。实机确认抛
      `ValueError: varlen is not supported by the sm100 backend`。
- [ ] sm89 / sm120 都缺 kernel 级 packed 用例。`test_varlen.py` 里 `fwd_varlen`
      那一组写死 `pv_accum_dtype="fp32"` + `v_layout="seq"`,所以 pin 在 sm80;
      sm90 另有 `test_varlen_sm90.py`。这两个 arch 目前只有 API 级覆盖,补
      `test_varlen_sm89.py` / `test_varlen_sm120.py` 才能直接测它们的 tile
      几何与 fp8 V^T 布局。

口径提醒:varlen 每个 arch 只实例化它自己的默认 `pv_accum_dtype`(sm80
`"fp32"`、sm89 `"fp32+fp16"`、sm90 `"fp32+fp32"`、sm120 `"fp32"`,见
plan.cpp)。别的组合会明确报错,不会静默降级,所以对拍和 bench 脚本里不要
顺手换 pv。

## 2. A4-1 合并的性能复核(sm89/sm120)

本机已验证:SASS 算术/MMA/访存指令 132/132 实例逐条相同,spill 合计净减
(sm89 3680→3544 B、sm120 亦降),但个别实例升:

- sm89 `f32_fuse_v_scale_attn_inst_buf` spill 592/424 → 656/480
- sm120 `f16_fuse_v_scale_attn_inst_buf` spill 320/312 → 416/416

跑 fwd bench(交替中位数,>3% 才算信号)确认无劣化;有劣化则只回退对应
TU 的合并。

- [x] sm120(2026-08-29):kernel 级 36 个配置最差 +2.73%,e2e 24 个配置
      最差 +0.53%,都在噪声内,无劣化。数字见 §5b。sm89 家族的 TU 在
      cc 12.0 上编成 sm_120 cubin,所以上表 sm89 那一行的 spill 数字(sm_89
      cubin)要等 Ada 实机才能复核。
- [x] sm89(L20,2026-08-29):sm_89 cubin 的 spill 数字复核成立——
      `f32_fuse_v_scale_attn_inst_buf` 的 head_dim=128 组正是 **656 / 480 B**,
      与上表合并后的那一列逐字相符。**但它跑得更快**:kernel 级 3 轮交替中位数
      里这个 kernel 是 −4.25% ~ +0.42%(head_dim 128),全 64 个配置最差
      +3.69%(`accum_f16` inst_buf 的一个 causal 点,同 kernel 的另外 7 个配置
      在 −1.07% ~ +2.25%,是噪声)。e2e 28 个配置 −30.7% ~ +1.94%。
      **无劣化,不回退**。多出来的 64 B spill store / 56 B spill load 在 ncu 里
      是 524 KB 的 local 流量、`long_scoreboard` 只占 10.23 cycle/issued 里的
      0.21,被 tensor 管线完全遮住(见 §5e)。数字见 §5d。

## 3. C-1 inst_buf(方向已改,源码循环序方案作废)

实测结论(2026-08-28,本机 ptxas):head_dim=128 的 inst_buf 实例寄存器顶
在 255 上限,把活跃缓冲从 [2][8][8] 压到 [2][8] 后 ptxas 重新调度把压力原
样拼回(两种相差 2 倍的缓冲尺寸给出完全相同的 spill,且比基线差)。
上机方向:对这些实例试 `__launch_bounds__` / `-maxrregcount` 约束,判据
不变——f32 inst_buf 1576 B / f16 inst_buf 2088 B 的 spill 归零 + bench
`pv_accum_dtype=fp32+fp16, per_warp` 预期 +15-30%。

- [x] sm120 上机判定:**NO-GO**(2026-08-29,数据见 §5c)。两条独立理由:
  1. **前提不成立**。1576 / 2088 B 那两个数字是 sm_89 cubin 的。同一批 TU
     编成 sm_120 后,每个 head_dim=128 实例只剩 12-84 B spill,而且 ncu 的
     `Local Memory Spilling Requests` 是 0——spill 全在冷路径,一次都没执行。
  2. **约束方向是负的**。真正的限制是寄存器数本身:所有 head_dim=128 实例
     顶 255 reg,`Block Limit Registers=2`,occupancy 16.7%。要拿到
     3 block/SM 得压到 ≤170 reg;`-maxrregcount=168` 让 spill 从 ~50 B 涨到
     ~2350 B,head_dim=128 慢 **190-207%**(dense)/ **185-205%**(varlen)。
  3. **head_dim=64 是不带 spill 的对照组,同样没有收益**。它基线 187-216 reg、
     零 spill,压到 168 只花 12-88 B spill,occupancy 实测 16.5% → 24.5%
     (+49%),吞吐只动 2%。**这个 kernel 不是 occupancy 受限的**——ncu 也是
     这么说的(Compute SM 76.7%,`math_pipe_throttle` 占 stall 的 38.6%)。
  dense 与 varlen 两路分别测过,结论一致;sm89(Ada)那一侧的 spill 数字仍未
  复核,但机制一样,方向大概率不变。
- [x] sm89 上机判定:**NO-GO**(L20,2026-08-29,数据见 §5e)。这次前提是
  成立的——sm_89 cubin 上 spill 是实数(f32 inst_buf 656/480 B、f16 inst_buf
  1304/1200 B、`sm89_varlen` 984/1088 B,head_dim=64 全为 0),ncu 也确实
  记到 524 KB 的 local 流量。**判决落在对照组上**:head_dim=64 那一组零 spill,
  `-maxrregcount=168` 把 occupancy 从 16.50% 抬到 24.59%(+49%),16 个配置的
  吞吐是 −2.3% ~ +3.2%(中位 ~0)。head_dim=128 付出 524 KB → 139 MB 的
  spill 流量换同样的 occupancy,慢 **18% ~ 98%**。两个 arch 各自独立地
  得到同一个结论:**这个 kernel 不是 occupancy 受限的**,C-1 关掉。

## 4. 批次 D 宏(CMake option,默认 OFF)

本机 sm86 已采数(scratchpad batchd/,精度统计见会话报告):三宏 OFF 时
SASS byte-identical;D-1 diff ≤ 1 输出 ULP 但本机 bench 略慢(整数管线);
D-2 实测比参考实现更不准(rms_rel ×1.41),不建议开;D-4 在 sm_89 上
SASS 逐字节相同(nvcc 已自动融合,纯 no-op)。

- [x] sm120(2026-08-29):`-DSAGE_OPT_FUSED_EPILOGUE=ON` 后 sm120 三个 TU
      的 spill_st 合计 960→792(ld 1104→944),与本机 ptxas 的预估一致。
      golden `--check` 仍是 `ok=2578 diff=0` + equiv 265/265,说明 nvcc 本来
      就融合了这一步,宏只改寄存器分配。kernel 级 bench 36 个配置全在
      ±2.3% 内,−168 B spill 测不出来。**结论:宏留档,默认保持 OFF。**
- [ ] 高 tensor-core 配比卡上若 conversion 单元成为瓶颈,可重测 D-1
      (`SAGE_OPT_MAGIC_I2F`,sm80 kernel 专用)。

## 5. sm90 —— 已完成(H200 GPU 5)

两轮:08-28 是重构后的 dense 验证,08-29 是 varlen 提交上去之后的复验。

| 项 | 结果 | 轮次 |
|---|---|---|
| bitwise 对拍(vs 0a5d2e4,1488 case) | diff=0 | 两轮 |
| equiv 段(fwd 全等 / pad 下沉等价 / 尾部全零) | 105/105 | 两轮 |
| batch stride > 2^32 隔离(`test_large_seq_batch_isolation`) | PASSED | 08-28 |
| pytest(含 compile/cudagraph) | 95 passed / 223 skipped | 08-28 |
| e2e bench(12 配置,3 轮交替中位数) | 全部加速 1.02-1.79×,零劣化 | 08-28 |
| pytest(加 varlen 用例) | 308 passed / 319 skipped,含 65 个 sm90 packed kernel 用例与 29 个 `sageattn_varlen` API 用例 | 08-29 |
| dense SASS 逐字节 | 0 处差异(纯搬家 1940 kernel、varlen 提交 1643 kernel) | 08-29 |

skip 是这个 sm_90-only 构建里缺席的其他 arch 家族,不是被跳过的检查。

## 5b. sm120 —— 已完成(RTX PRO 6000 Blackwell,sm_120,110 SM,73 GB)

2026-08-29 一轮。机器不是 5090,但同属 GB202 / cc 12.0,`compiled_archs`
是 `[80, 89, 120]`(`TORCH_CUDA_ARCH_LIST=12.0`:sm80 与 sm89 家族都会为
sm_120 出 cubin),所以三族 qattn op 在同一张卡上都能跑。

| 项 | 结果 |
|---|---|
| bitwise 对拍(vs 0a5d2e4,2578 case) | `ok=2578 diff=0 env_mismatch=0 missing=0` |
| 对拍分布 | attn sm80 792 / sm89 792 / sm120 594,e2e sm120 160,quant 240 |
| equiv 段(fwd 全等 / pad 下沉等价 / 尾部全零 / plan 镜像) | 265/265 |
| pytest 全量 | 289 passed / 397 skipped |
| pytest varlen 两文件 | 138 passed / 97 skipped |
| fp8 zero-amax 四个测试 | 16 case 全 passed(sm86 上全是 skip,这里首次真跑) |
| kernel 级 bench(6 kernel × 3 shape × 2 causal,3 轮交替中位数) | 最差 +2.73%,无劣化 |
| e2e bench(6 shape × 2 causal × 2 pv,3 轮交替中位数) | 0.99-1.12×,最差 +0.53% |
| `bench_varlen.py` | 等长 0.97-1.01× dense,ragged .25-1x 1.35-1.68×,ragged .1-1x 1.43-1.88× |
| D-4 宏 ON | spill_st 960→792,数值 diff=0,bench ±2.3%(见 §4) |

pytest 的 397 个 skip 全是 pin 在别的 backend 的用例(cc 12.0 resolve 到
sm120,`requires_backend("sm80"/"sm89"/"sm90")` 一律跳过),外加
`test_large_seq_batch_isolation` 要 ~100 GB 显存、这张卡 73 GB。

两个口径问题,都不影响本轮结论:

- `test_accuracy.py` 整个文件是 `pytestmark = requires_backend("sm80")`,
  所以任何 fp8 卡上都不跑。临时去掉这一行在 sm_120 上跑,54 个用例过
  (cos_sim > 0.99、rel_l1 < 0.06),72 个报 `ValueError: pv_accum_dtype`
  ——它们写死了 sm80 才有的 `"fp16"` / `"fp16+fp32"`。
  **已解决(2026-08-29)**:pv 按 resolve 到的 backend 参数化(合法集合镜像
  plan.cpp 的 `pv_supported`/`smooth_v_supported`)。实跑:sm120 单文件
  94 passed(全量 383 passed / 271 skipped),sm90(H200)54 passed /
  8 skipped(skip 全是 smooth_v——sm90 无 fused smooth_v kernel),本机
  sm86 全量 548 passed / 138 skipped 不变。实测最差:sm80 cos 0.99992 /
  rel_l1 0.013,fp8(sm90、sm120 全部合法 pv)cos 0.99926 / rel_l1 0.039,
  0.99/0.06 阈值对所有实测 backend 成立,数值不动。sm89/sm100 尚无实测数值(见 §1)。
- 跑测试的容器里有 `pip install -e /workspace/SageAttention`,它的
  MetaPathFinder 排在 `sys.meta_path` 末尾:待测树里缺席的子模块会静默落到
  那个安装上(本轮就撞到 `sageattention._qattn_sm90`)。用 PYTHONPATH 指向
  待测树之前先把这个 finder 摘掉。

## 5c. sm120 profiling(2026-08-29,ncu 2026.2.1 / nsys 2026.3.1)

容器是特权模式(`CapEff` 全开),ncu 硬件计数器可用,不需要降级到 nsys-only。
**dense 与 varlen 两路都采、都测。** dense 的形状网格取
`bench/bench_qk_int8_pv_fp8_cuda.py` 的那一套:batch 4、heads 32、
head_dim 64/128、seq 1024-32768、NHD、causal 两态。varlen 在等长(与 dense
同形状对照)之外再加三档偏斜(.5-1x / .25-1x / .1-1x),形状覆盖小 batch 长
序列(b1 h16 n32768、b2 h16 n16384)到大 batch 短序列(b16 h8 n2048)。

### 全流程时间占比(nsys,d128,batch 4 heads 32,每个 kernel 25 次的合计)

dense(`sageattn`,smooth_k 的均值走 ATen `reduce_kernel`):

| seq / causal | attention | transpose_pad | quant K | quant Q | MeanScale | `k.mean` | fp8 零填充 |
|---|---|---|---|---|---|---|---|
| 1024 / 0 | 47.2% | 13.7% | 9.6% | 7.9% | 9.0% | 9.7% | 1.7% |
| 1024 / 1 | 36.7% | 16.3% | 11.7% | 9.5% | 10.8% | 11.6% | 2.0% |
| 4096 / 0 | 74.4% | 7.0% | 5.2% | 4.2% | 4.2% | 3.5% | 1.1% |
| 4096 / 1 | 61.2% | 10.6% | 7.9% | 6.3% | 6.3% | 5.3% | 1.7% |
| 16384 / 0 | 91.7% | 2.1% | 1.6% | 1.5% | 1.6% | 1.0% | 0.4% |
| 16384 / 1 | 84.7% | 3.9% | 2.9% | 2.7% | 2.9% | 1.9% | 0.8% |

varlen(`sageattn_varlen`,非 causal;均值走 `SegmentMeanKernel`。
`equal` 是等长 batch,`.25-1x` 是最短序列取 0.25 倍的偏斜):

| seq / 偏斜 | attention | transpose_pad | quant K | quant Q | MeanScale | `segment_mean` | fp8 零填充 |
|---|---|---|---|---|---|---|---|
| 1024 / equal | 46.2% | 13.7% | 9.5% | 8.2% | 10.3% | 8.7% | 1.7% |
| 1024 / .25-1x | 40.9% | 16.6% | 6.4% | 9.4% | 19.3% | 3.3% | 1.8% |
| 4096 / equal | 73.9% | 7.1% | 5.3% | 4.3% | 4.2% | 3.3% | 1.1% |
| 4096 / .25-1x | 69.9% | 8.5% | 6.3% | 4.7% | 4.6% | 4.0% | 1.3% |
| 16384 / equal | 91.6% | 2.1% | 1.6% | 1.5% | 1.6% | 1.0% | 0.4% |
| 16384 / .25-1x | 90.1% | 2.5% | 1.9% | 1.8% | 1.9% | 1.2% | 0.5% |

数字是 nsys 自己的 Time(%),分母里还含一个生成测试数据的 RNG kernel
(1.1% / 0.5% / 0.1%),所以每行合计不到 100。**两路的形状完全一样**:
seq ≥ 4096 时 attention kernel 是绝对大头(61-92%),前处理要到 seq ≤ 2048
才值得看。varlen 的 `segment_mean` 在偏斜大时比 dense 的 ATen `k.mean` 便宜
(它只读真实 token),但省下的比例又被 attention kernel 同步缩短抵掉,占比反而
看不出优势。

### attention kernel(d128 n4096 非 causal,dense 与 varlen packed 各一份)

| 指标 | dense | varlen packed |
|---|---|---|
| Duration | 2.69 ms | 2.65 ms |
| Compute (SM) Throughput | 76.7% | 78.3% |
| Memory Throughput | 28.1% | 28.0% |
| 寄存器 / occupancy | 255 → 16.67%(理论)、16.51%(实测) | 255 → 16.67% / 16.48% |
| `Block Limit` registers / smem / warps | 2 / 3 / 12 | 2 / 3 / 12 |
| Warp cycles per issued instruction | 6.49 | 6.24 |

stall 拆解(dense,单位是每条已发射指令的周期,合计 6.49):
`math_pipe_throttle` 2.50、`wait` 1.57、`not_selected` 0.32、
`long_scoreboard` 0.29、`barrier` 0.25、`short_scoreboard` 0.20。
`Local Memory Spilling Requests` 为 0。

结论:kernel 已经 tensor pipe 主导(tensor 管线 78.2% of peak),最大的 stall
是管线本身占满。剩下那 ~22% 要靠把 softmax / scaling 的 ALU 工作与 mma 重叠
才能吃到——和 sm90 的 H1/H4 同级,属于重写。**varlen packed kernel 与 dense
是同一个模板、同一套资源画像**,varlen 开发期记的那份高 spill(REG 255 /
stack 208 B)同样是 sm_89 目标下的数字,sm_120 上最差只有 84 B 且不执行。

### C-1 的判决实验:head_dim=64 是不带 spill 的对照组

`-maxrregcount=168` 在两种 head_dim 上的代价完全不同,正好构成一组对照:

| | 寄存器 | spill(st/ld) | 理论 occupancy | 实测 occupancy | Compute (SM) | Duration |
|---|---|---|---|---|---|---|
| d64 基线 | 187-216 | 0 | 16.67% | 16.47% | 67.9% | 1.53 ms |
| d64 `-maxrregcount=168` | 168 | 12-88 B | **25%** | 24.51% | 69.4% | 1.50 ms |
| d128 基线 | 255 | 12-84 B | 16.67% | 16.51% | 76.7% | 2.69 ms |
| d128 `-maxrregcount=168` | 168 | ~2350 B | 25% | — | — | 3× 变慢 |

head_dim=64 那一行是关键:寄存器压下去几乎不花 spill,occupancy 实打实涨了
**49%**(16.5% → 24.5%),吞吐只动了 2%。**这个 kernel 不是 occupancy 受限的。**
head_dim=128 想拿同样的 occupancy 要付 ~2350 B spill,于是 3× 变慢。两条腿
都不通,C-1 的 launch_bounds 方向到此为止。

bench 两路分开报(3 轮交替中位数,base = 无约束):

| 路径 | head_dim | 配置数 | 相对基线 |
|---|---|---|---|
| dense(`qattn_sm120_*` + `qattn_sm89_*`,6 个 kernel) | 128 | 24 | +190% ~ +207% |
| dense | 64 | 12 | −0.4% ~ +3.2% |
| varlen(`fwd_varlen`,sm120_varlen 实例) | 128 | 36 | +185% ~ +205% |
| varlen | 64 | 36 | −5.6% ~ +9.7%(中位 +1.5%) |

sm89 家族的 varlen 实例(`sm89_varlen`,fp32+fp16 inst_buf)在 cc 12.0 上够不
着:`sageattn_varlen` resolve 到 sm120,而 `fwd_varlen` 对每个 arch 只接受它
自己的默认 `pv_accum_dtype`。那一组要等 Ada 实机。

### 前处理 kernel(d128 n4096,ncu SpeedOfLight)

| kernel | Duration | DRAM Throughput | 寄存器 | Achieved Occupancy |
|---|---|---|---|---|
| `TransposePadPermuteKernel` | 206.5 µs | 83.8% | 26 | 63.6% |
| `QuantPerThreadKInt8Kernel` | 140.9 µs | 87.2% | 54 | 68.3% |
| `QuantPerThreadQInt8Kernel` | 126.5 µs | 93.8% | 34 | 83.6% |
| `MeanScaleKernel` | 122.6 µs | 94.4% | 40 | 95.8% |
| `at::native::reduce_kernel`(`k.mean`) | 121.0 µs | 89.1% | 50 | 64.5% |

五个都贴在 DRAM roofline 上(84-94%),单 kernel 调优没有空间,**只能删掉多余
的字节**。dense 与 varlen 用的是同一批 kernel(只有 smooth_k 的均值不同:
dense 走 ATen `reduce_kernel`,varlen 走 `SegmentMeanKernel`),所以这个结论
两路通用。已落地两处(见下),还剩一处提案:

1. varlen 的 fp8 零填充(占 0.4-1.8%):让 `MeanScaleKernel` 对空序列写零而
   不是提前 return,`quant_v_fp8_varlen` 就能跟 dense 一样换成 `at::empty`。
   动的是一段有明确正确性理由的提前返回,收益又在 2% 以内,没做。

### dense 补齐 vs varlen packed(3 轮交替中位数,单位 ms,括号是相对 dense 的加速)

dense 那一列是把同一批 token 补齐到 `seq_len` 后跑 `sageattn`,即没有 packed
布局的调用方今天付的价。`b1` 那两行只有一个序列,偏斜档位在它上面退化成等长。

| shape | causal | dense | equal | .5-1x | .25-1x | .1-1x |
|---|---|---|---|---|---|---|
| d64 b4 h32 n1024 | 0 | 0.176 | 0.190 (0.93×) | 0.158 (1.12×) | 0.151 (1.17×) | 0.156 (1.13×) |
| d64 b4 h32 n4096 | 0 | 1.865 | 1.962 (0.95×) | 1.341 (1.39×) | 1.144 (1.63×) | 1.231 (1.52×) |
| d64 b8 h32 n4096 | 0 | 3.924 | 4.084 (0.96×) | 3.210 (1.22×) | 1.991 (1.97×) | 1.685 (2.33×) |
| d64 b16 h8 n2048 | 0 | 0.582 | 0.603 (0.96×) | 0.378 (1.54×) | 0.294 (1.98×) | 0.286 (2.04×) |
| d64 b2 h16 n16384 | 0 | 6.108 | 6.683 (0.91×) | 4.714 (1.30×) | 4.160 (1.47×) | 5.350 (1.14×) |
| d64 b1 h16 n32768 | 0 | 12.275 | 12.751 (0.96×) | — | — | — |
| d128 b4 h32 n1024 | 0 | 0.391 | 0.396 (0.99×) | 0.313 (1.25×) | 0.297 (1.32×) | 0.304 (1.28×) |
| d128 b4 h32 n4096 | 0 | 3.474 | 3.494 (0.99×) | 2.462 (1.41×) | 2.080 (1.67×) | 2.142 (1.62×) |
| d128 b8 h32 n4096 | 0 | 7.349 | 7.431 (0.99×) | 5.833 (1.26×) | 3.681 (2.00×) | 3.240 (2.27×) |
| d128 b16 h8 n2048 | 0 | 1.103 | 1.088 (1.01×) | 0.728 (1.51×) | 0.578 (1.91×) | 0.561 (1.97×) |
| d128 b2 h16 n16384 | 0 | 11.261 | 11.360 (0.99×) | 8.041 (1.40×) | 7.011 (1.61×) | 9.319 (1.21×) |
| d128 b1 h16 n32768 | 0 | 22.005 | 21.764 (1.01×) | — | — | — |

causal=1 的 12 行同形状(等长 0.93-1.03×,偏斜最高 2.25×),完整 120 行在
scratchpad 的 `varlen_state_table.txt`。等长时 packed 与 dense 打平,偏斜越大
收益越大,批次越大收益越大——`b8 h32 n4096` 在 .1-1x 上是 2.3×。
偏斜列不单调(`b2 h16 n16384` 的 .1-1x 比 .25-1x 慢)是因为最长序列被钉死在
`seq_len`、batch 只有 2,随机抽到的总 token 反而更多。

### 落地的优化一:fp8 V 缓冲的 at::zeros

`quant_v_fp8` 里 fp8 V 缓冲的 `at::zeros`:量化 kernel 只写到 64 对齐边界,
所以 `pad_multiple=128` 且 kv_len 的 128 对齐边界更大时,尾巴必须先清零。
sm89 / sm120 的 `pad_multiple` 是 64,两个边界重合,kernel 覆盖每一个字节,
那次 memset 是纯浪费。改成两边界重合时走 `at::empty`。

- golden `--check`:`ok=2578 diff=0`,equiv 265/265(逐位不变);本机 sm86
  双门禁 `ok=1493 diff=0` + `548 passed / 138 skipped`
- nsys:`FillFunctor<Float8_e4m3fn>` kernel 从时间线上消失,其余 kernel 不变;
  单次迭代 GPU kernel 合计 n=1024 −1.74%、n=4096 −0.44%,随 seq 增大趋近 0
- **dense 墙钟**:官方网格(b4 h32,24 配置)−2.0% ~ +1.25%,没过阈值;换到
  V 字节占比更高的形状(b8 h32 n4096 causal=1、b16 h8 n2048 causal=1)有
  −3.3% ~ −4.4%,过了阈值
- **varlen 墙钟**:120 配置 −4.4% ~ +5.4%,散在两侧,就是噪声。**这是预期的**
  ——`quant_v_fp8_varlen` 的 `at::zeros` 没动,那边空序列不产生任何写入,
  它的 slab 必须靠分配时的零来兜底,主机侧又不能读 `cu_seqlens` 判断有没有
  空序列(会引入 D2H 同步,cudagraph / compile 路径不接受)

一句话:这个优化只对 dense 生效,varlen 那一路的同名 memset 不是冗余的。

### 落地的优化二:transpose 融进 fp8 量化(2026-08-30)

`quant_v_fp8` / `quant_v_fp8_varlen` 内部改成一个 `TransposeQuantFp8Kernel`,
中间那份 fp16 V^T 不再落地:读 V → 写 fp8,V 的字节从 7 份降到 3 份。CMake
`-DSAGE_FUSED_V_QUANT=OFF` 退回两趟(`transpose_pad_v` + `(mean_)scale_fuse_quant`
两个低层 op 原样保留)。

**逐位一致靠 thread mapping 而不是靠对数。** block 保持 `MeanScaleKernel`
的形状——256 线程,线程 t 拥有转置后的 token pack `[i*2048 + t*8, +8)`——所以
每一个归约叶子、blockReduce 树的每一步、统计的 ceil16 口径与量化的 ceil64
口径都原封不动;变的只是这 8 个值从哪来:不再从 V^T 连续读,而是按 16-token
permute 的逆映射从 V 原布局 gather。代价是放弃整行合并访问:一个 block 只
covers 16 个 channel,每次 gather 是 token 行里的 32 字节(一个完整 DRAM
sector,不浪费),而不是 128 字节的整行。

| 证据 | 结果 |
|---|---|
| `test_quant_v_fp8_matches_two_kernel_path` | 融合 vs 两趟逐位相同,224 组(layout × v_layout × smooth_v × dtype × head_dim × 7 个 seq) |
| golden `--check`(sm120) | `ok=2578 diff=0`,equiv 265/265 |
| pytest(sm120) | 305 passed / 397 skipped |
| 本机 sm86 双门禁 | `ok=1493 diff=0`(extra=48 是 varlen equiv);源映射对拍 96 组全等 |
| ptxas | 8 个实例 0 spill,64-94 寄存器 |

varlen 那一路的逐位一致是接力得来的:`test_quant_v_fp8_varlen_matches_per_sequence_dense`
已经钉死「packed 每段 == 同长度 dense」,加上上面这条「dense 融合 == dense
两趟」,ragged batch 的每一段也就等于两趟版。

**ncu**(d128 n4096,batch 4 heads 32,与上表同一组配置):

| kernel | Duration | DRAM Throughput | 寄存器 | Achieved Occupancy |
|---|---|---|---|---|
| `TransposePadPermuteKernel` + `MeanScaleKernel` | 198.5 + 117.3 = 315.8 µs | 84.6% / 94.8% | 26 / 40 | 63.5% / 95.7% |
| `TransposeQuantFp8Kernel` | 251.0 µs | 55.6% | 78 | 47.3% |

字节少搬 57%、时间只省 20%,差额就是 gather 的效率:DRAM 从 ~90% 掉到 56%。
**这条曲线有拐点**,所以主机侧按 padded token 数收口在 4096(两个 round),
再长就走两趟。V 前处理单独计时(b8 h16 d128,3 轮中位数,单位 µs):

| n | 两趟 | 融合 | Δ |
|---|---|---|---|
| 512 | 55.4 | 32.4 | −41% |
| 1024 | 73.1 | 46.8 | −36% |
| 2048 | 169.3 | 77.2 | −54% |
| 3072 | 292.1 | 161.9 | −45% |
| 4096 | 398.8 | 362.8 | −9% |
| 5120 | 510.3 | 561.6 | +10% |
| 8192 | 827.3 | 1202.6 | +45% |
| 16384 | 1716.5 | 2628.7 | +53% |

拐点在 4096 与 5120 之间:block 的一个 round 是 2048 个 token,两个 round
以内时第二趟还能在 L2 里找到第一趟读过的字节,超过就是实打实的第二次 DRAM
读,而 gather 的 DRAM 效率只有整行访问的六成,换不回来。门限之上两棵树
逐点同速(±1%),即回退生效。

**e2e 墙钟**(`sageattn`,3 轮交替中位数):

| shape | 两趟 (µs) | 融合 (µs) | Δ |
|---|---|---|---|
| b8 h16 n512 d128 | 136.1 | 111.8 | −17.9% |
| b8 h16 n1024 d128 | 383.1 | 347.2 | −9.4% |
| b8 h16 n2048 d128 | 1060.0 | 1001.4 | −5.5% |
| b8 h16 n4096 d128 | 3544.6 | 3500.0 | −1.3% |
| b8 h32 n4096 d64 | 3963.2 | 3862.3 | −2.5% |
| b16 h8 n2048 d64 | 573.4 | 560.1 | −2.3% |
| b1 h24 n512 d128 | 47.6 | 43.5 | −8.7% |
| b1 h24 n1024 d128 | 79.9 | 74.3 | −7.0% |
| b1 h24 n2048 d128 | 192.7 | 189.0 | −1.9% |
| b4 h16 n8192 d128 | 6228.5 | 6238.1 | +0.2% |
| b2 h24 n16384 d128 | 17253.8 | 17291.3 | +0.2% |

短 seq 过阈值(−5% ~ −18%),长 seq 走回退、留在噪声里。`smooth_v` 那一路
同向:V 前处理 n=512 −33%、n=2048 −46%、n=4096 −41%。

## 5d. sm89 —— 已完成(L20,sm_89,92 SM,46 GB)

2026-08-29 一轮,ComputeLab 的 L20(AD102,cc 8.9)。不是 4090,但同为
Ada、同一个 `sm_89` cubin 目标,`compiled_archs` 是 `[80, 89]`
(`TORCH_CUDA_ARCH_LIST=8.9`),两族 qattn op 在同一张卡上都能跑。
**卡型注意**:L20 是数据中心 Ada,fp8 的 fp16 累加不是 2× 速率(见 §6),
所以凡是牵涉 `pv_accum_dtype="fp32+fp16"` 收益的结论不能直接搬到 4090。

| 项 | 结果 |
|---|---|
| bitwise 对拍(vs 0a5d2e4,2004 case) | `ok=2004 diff=0 env_mismatch=0 missing=0` |
| 对拍分布 | attn sm80 792 / sm89 792,e2e sm89 180,quant 240 |
| equiv 段(fwd 全等 / pad 下沉等价 / 尾部全零 / plan 镜像 / varlen vs dense) | 345/345 |
| pytest 全量 | 289 passed / 397 skipped |
| pytest varlen 两文件 | 138 passed / 97 skipped |
| fp8 zero-amax 四个测试 | 16 case 全 passed |
| kernel 级 bench(8 kernel × 2 head_dim × 2 seq × 2 causal,3 轮交替中位数) | 64 配置最差 +3.69% |
| e2e bench(2 head_dim × 6 seq × 2 causal + pv 扫描,28 配置) | −30.7% ~ +1.94% |
| `bench_varlen.py`(96 行) | 等长 0.96-1.01× dense,ragged .1-1x 最高 1.93× |

对拍与 pytest 的口径跟 sm120 那一轮完全一致(289/397、138/97 两组数字逐一
相同),说明两个 arch 走的是同一批用例、只是 backend 不同。

e2e 那个 −30.7% 不是 kernel 变快,是重构后新加的 `quant_v_fp8` 免 memset
(commit 78cf506)在 seq 1024 这种前处理占一半时间的形状上放大了:同一列
seq ≥ 4096 就回到 ±2% 以内。

## 5e. sm89 profiling(2026-08-29,ncu 2026.2.1 / nsys 2026.3.1)

容器 ncu 硬件计数器可用。**dense 与 varlen 两路都采、都测**,形状网格与
§5c 逐格对齐,好让两个 arch 并排读。

### 全流程时间占比(nsys,d128,batch 4 heads 32,每个 kernel 30 次的合计)

dense(`sageattn`,`per_thread` 量化,smooth_k 的均值走 ATen `reduce_kernel`):

| seq / causal | attention | transpose_pad | quant K | quant Q | MeanScale | `k.mean` |
|---|---|---|---|---|---|---|
| 1024 / 0 | 54.3% | 12.4% | 7.4% | 10.5% | 5.9% | 8.9% |
| 1024 / 1 | 42.2% | 15.6% | 9.2% | 13.4% | 7.5% | 11.2% |
| 4096 / 0 | 78.5% | 6.1% | 4.3% | 3.3% | 3.8% | 3.7% |
| 4096 / 1 | 66.2% | 9.6% | 6.8% | 5.2% | 5.9% | 5.7% |
| 16384 / 0 | 92.7% | 2.0% | 1.6% | 1.4% | 1.3% | 0.9% |
| 16384 / 1 | 86.5% | 3.7% | 2.9% | 2.6% | 2.5% | 1.6% |

varlen(`sageattn_varlen`,非 causal;均值走 `SegmentMeanKernel`):

| seq / 偏斜 | attention | transpose_pad | quant K | quant Q | MeanScale | `segment_mean` | fp8 零填充 |
|---|---|---|---|---|---|---|---|
| 1024 / equal | 51.7% | 12.3% | 7.3% | 10.5% | 6.5% | 8.3% | 1.2% |
| 1024 / .25-1x | 50.6% | 13.6% | 5.4% | 10.6% | 8.7% | 7.2% | 1.3% |
| 4096 / equal | 76.5% | 6.3% | 4.2% | 3.3% | 3.9% | 3.6% | 1.2% |
| 4096 / .25-1x | 74.3% | 6.7% | 4.5% | 3.6% | 4.3% | 4.0% | 1.3% |
| 16384 / equal | 92.0% | 2.1% | 1.6% | 1.4% | 1.4% | 0.9% | 0.4% |
| 16384 / .25-1x | 90.7% | 2.4% | 1.8% | 1.6% | 1.6% | 1.0% | 0.4% |

分母里还有一个生成测试数据的 RNG kernel(0.1-0.9%)与 varlen 侧打包用的
`cat`(0.2-1.2%),所以每行合计不到 100。形状规律与 sm120 一模一样:
seq ≥ 4096 起 attention kernel 是绝对大头(66-93%)。dense 那张表没有
「fp8 零填充」这一列,因为 78cf506 已经把它删了;varlen 还留着(0.4-1.3%)。

### attention kernel(d128 n4096 非 causal)

| 指标 | dense | varlen packed 等长 | varlen packed .25-1x |
|---|---|---|---|
| Duration | 5.40 ms | 5.35 ms | 3.84 ms |
| Compute (SM) Throughput | 85.8% | 86.6% | 85.9% |
| DRAM Throughput | 8.0% | 8.1% | 9.0% |
| 寄存器 / occupancy | 255 → 16.67%(理论)、16.52%(实测) | 255 → 16.67 / 16.52 | 255 → 16.67 / 16.50 |
| `Block Limit` registers / smem / warps | 2 / 3 / 12 | 2 / 3 / 12 | 2 / 3 / 12 |
| Warp cycles per issued instruction | 10.23 | 10.04 | 10.00 |
| `Local Memory Spilling Requests` | 524 KB | 492 KB | 411 KB |

stall 拆解(每条已发射指令的周期):dense d128 合计 10.23,
`math_pipe_throttle` **6.03**、`wait` 1.64、`selected` 1.00、`barrier` 0.62、
`not_selected` 0.34、`short_scoreboard` 0.21。head_dim=64 合计 7.13,
`math_pipe_throttle` 4.07。

结论与 sm120 同向、程度更极端:**tensor 管线自己占满就是最大的 stall**
(占 59%,sm120 是 38.6%),Compute (SM) 85.8% 已经贴顶。head_dim=128 那
524 KB 的 spill 流量没有代价——`long_scoreboard` 只有 0.21 cycle,而且
d128(2× 的算力工作量)刚好用了 d64 的 2.00× 时间(5.40 / 2.70),两者
TOPS 相同。**varlen packed 与 dense 是同一个模板、同一套资源画像**,
偏斜档位只改 Duration,不改任何一项每周期指标。

### C-1 的判决实验:head_dim=64 是不带 spill 的对照组

sm_89 上 spill 是实数(不像 sm_120),所以这组对照才是 C-1 的正式判据。
`-maxrregcount=168` 是唯一能把 128 线程的 block 从 2/SM 抬到 3/SM 的档位
(65536 / (3 × 128) = 170.6)。

| | 寄存器 | cubin spill(st/ld) | ncu local 流量 | 理论 occ | 实测 occ | Compute (SM) | Duration |
|---|---|---|---|---|---|---|---|
| d64 基线 | 242-255 | 0 | 0 | 16.67% | 16.50% | 85.9% | 2.70 ms |
| d64 `-maxrregcount=168` | 168 | 832-1632 B | 311 KB | **25%** | 24.59% | 85.6% | 2.72 ms |
| d128 基线 | 255 | 656-1304 B | 524 KB | 16.67% | 16.52% | 85.8% | 5.40 ms |
| d128 `-maxrregcount=168` | 168 | 6288-10624 B | 139 MB | 25% | 24.61% | 82.8% | 5.65 ms |

head_dim=64 那一行是判据:occupancy 实打实涨了 **49%**(16.50% → 24.59%),
kernel 级 bench 16 个配置只动了 −2.3% ~ +3.2%(中位 ~0)。**这个 kernel 不是
occupancy 受限的。** head_dim=128 想拿同样的 occupancy 要把 spill 流量放大
265×,于是慢下去:

| 路径 | head_dim | 配置数 | 相对基线(3 轮交替中位数) |
|---|---|---|---|
| dense(`qattn_sm89_*` 4 个 kernel) | 128 | 16 | +18.4% ~ +98.2% |
| dense | 64 | 16 | −2.3% ~ +3.2% |

单次 ncu 采样里 d128 只慢 4.6%,bench 是 +18% 起;以 bench 为准(ncu 会锁频
并串行化 replay)。**C-1 在 sm_89 上同样 NO-GO**,两个 arch 的理由互补:
sm_120 是「spill 根本不存在也不执行」,sm_89 是「spill 存在但被 tensor 管线
遮住,而 occupancy 本来就不是瓶颈」。

### 前处理 kernel(d128 n4096,ncu SpeedOfLight)

| kernel | Duration | DRAM Throughput | 寄存器 | Achieved Occupancy |
|---|---|---|---|---|
| `TransposePadPermuteKernel` | 327 µs | 89.8% | 26 | 63.6% |
| `QuantPerThreadQInt8Kernel` | 234 µs | 93.4% | 34 | 85.0% |
| `QuantPerThreadKInt8Kernel` | 233 µs | 94.1% | 110 | 29.8% |
| `MeanScaleKernel` | 217 µs | 95.2% | 40 | 95.7% |
| `at::native::reduce_kernel`(`k.mean`) | 194 µs | 92.7% | 44 | 78.2% |
| `SegmentMeanKernel`(varlen) | 156 µs | 95.1% | 44 | 63.3% |
| `SegmentMeanFinishKernel`(varlen) | 2.9 µs | 11.3% | 40 | 11.1% |

前六个都贴在 DRAM roofline 上(90-95%),和 sm120 的结论一样:**单 kernel
调优没有空间,只能删掉多余的字节**(`SegmentMeanFinishKernel` 只有 2.9 µs,
不在讨论范围)。`QuantPerThreadKInt8Kernel` 的 110 寄存器把 occupancy 压到
29.8%,但它照样跑到 94.1% 的 DRAM 峰值,所以那也不是问题。

### dense 补齐 vs varlen packed(单位 ms,括号是相对 dense 的加速,causal=0)

dense 那一列是把同一批 token 补齐到 `seq_len` 后跑 `sageattn`。`b1` 只有一个
序列,偏斜档位在它上面退化成等长。

| shape | dense | equal | .5-1x | .25-1x | .1-1x |
|---|---|---|---|---|---|
| d64 b4 h32 n1024 | 0.347 | 0.357 (0.97×) | 0.323 (1.08×) | 0.299 (1.15×) | 0.285 (1.22×) |
| d64 b4 h32 n4096 | 3.417 | 3.491 (0.98×) | 2.846 (1.20×) | 2.558 (1.34×) | 2.433 (1.43×) |
| d64 b8 h32 n4096 | 6.949 | 7.144 (0.97×) | 5.189 (1.36×) | 4.235 (1.64×) | 3.791 (1.85×) |
| d64 b16 h8 n2048 | 1.044 | 1.075 (0.97×) | 0.773 (1.36×) | 0.632 (1.61×) | 0.557 (1.84×) |
| d64 b2 h16 n16384 | 11.477 | 11.625 (0.99×) | 10.797 (1.07×) | 10.558 (1.09×) | 10.277 (1.13×) |
| d64 b1 h16 n32768 | 22.338 | 22.551 (0.99×) | — | — | — |
| d128 b4 h32 n1024 | 0.735 | 0.741 (0.99×) | 0.644 (1.14×) | 0.588 (1.25×) | 0.549 (1.34×) |
| d128 b4 h32 n4096 | 6.907 | 6.988 (0.99×) | 5.724 (1.20×) | 5.154 (1.34×) | 4.839 (1.43×) |
| d128 b8 h32 n4096 | 13.874 | 14.088 (0.98×) | 10.124 (1.37×) | 8.419 (1.65×) | 7.477 (1.86×) |
| d128 b16 h8 n2048 | 2.096 | 2.128 (0.99×) | 1.529 (1.37×) | 1.254 (1.68×) | 1.091 (1.93×) |
| d128 b2 h16 n16384 | 23.034 | 22.959 (1.00×) | 21.236 (1.08×) | 20.768 (1.11×) | 20.129 (1.14×) |
| d128 b1 h16 n32768 | 44.529 | 44.208 (1.01×) | — | — | — |

causal=1 的 48 行同形状(等长 0.96-1.00×,偏斜最高 1.77×),完整 96 行在
scratchpad 的 `varlen_state_table.csv`。规律与 sm120 一致:等长时打平,偏斜越大、
批次越大收益越大;`b2`/`b1` 那两组因为最长序列被钉死在 `seq_len`、batch 又小,
偏斜档位省不下多少 token,所以只有 1.1× 上下。

### 采到但没落地的三条

1. **sm89 默认 `pv_accum_dtype="fp32+fp16"` 在 L20 上是净亏**。见 §6 的 sm_89
   指令速率表:`f8f8f16` 与 `f8f8f32` 同速。kernel 级 bench 上
   `accum_f16` inst_buf 在 head_dim=128 比 `accum_f32` inst_buf **慢 3.0-7.0%**
   (head_dim=64 快约 1%),e2e 的 pv 扫描也是 `fp32+fp16` 最慢
   (hd128 n4096:6.684 / 4.293 ms,对 `fp32` 的 6.621 / 4.149 与
   `fp32+fp32` 的 6.532 / 4.113)。同时它只保 10 位尾数,`fp32` 保 13 位。
   **没有改默认**:2× 的 `f8f8f16` 速率是消费级 Ada(4090)的特性,改 plan.cpp
   的默认要先在 4090 上复核,不能拿 L20 一张卡定 sm89 全族。
2. transpose 与 MeanScale 是对 V 的两趟(读 V → 写 fp16 V^T → 读 V^T → 写 fp8),
   融成一个 kernel 少搬 28% 的字节。n=4096 上这两个 kernel 合计 9.9%(dense)
   / 10.2%(varlen),省 ~90 µs。与 sm120 那条是同一个提案,要新写 kernel 且
   要保证按 channel 的 amax 与 16-token permute 补位逐位一致。
3. varlen 的 fp8 零填充(0.4-1.3%):理由与 sm120 那条相同(空序列不产生写入),
   没做。

## 5f. sm100 —— 已完成(B200,sm_100,183 GB,单卡)

2026-08-29,ComputeLab `umb-b200-237`,NGC `pytorch_26.07-py3`
(torch 2.13.0a0 / CUDA 13.3 / nvcc 13.3)。`TORCH_CUDA_ARCH_LIST=10.0`,
两侧同容器构建,`compiled_archs=[80, 89, 100]`,`SAGEATTN_SM100_TCGEN05=1`
开门(baseline 与 new 同一个 env 门,baseline 走 core.py 的同名判断)。

**对拍口径**:golden 只 dump 一份——baseline(0a5d2e4)的 TS 路径,2280 case。
new 的 SS twin 和 TS 各自对这一份跑 `--check`。三者全等,同时说明
(a) 重构没动数值,(b) TS 的 TMEM A-operand 布局与走 smem 的 SS oracle 一致。
再 dump 一份 baseline-SS 只能回答「重构有没有动 SS twin」,而 SS 不是发布路径,
所以没做。

| 项 | 结果 |
|---|---|
| SS twin `--check`(vs 0a5d2e4,2280 case) | `ok=2280 diff=0 env_mismatch=0 missing=0` |
| TS `--check`(默认发布路径,同一份 golden) | `ok=2280 diff=0 env_mismatch=0 missing=0` |
| 对拍分布 | attn 1980(sm80/sm89/sm100 三族)/ e2e 60(sm100)/ quant 240 |
| equiv 段(SS 与 TS 各一轮) | 105/105 |
| SDPA 精度(cos_sim > 0.99、rel_l1 < 0.06) | SS 62/62、TS 62/62 |
| pytest 全量 | 307 passed / 441 skipped,0 failed |
| varlen 设计内拒绝 | `ValueError: varlen is not supported by the sm100 backend` |
| `test_large_seq_batch_isolation`(batch stride > 2^32) | PASSED(183 GB 够) |
| e2e bench(40 配置,3 轮交替中位数) | 0.995-1.128×,中位 1.051×,**零劣化** |

`repeat=2` 下 `unstable=0`、`raised=0`:kernel 在硬件上是确定性的。
精度用例是 `test_accuracy.py` 的等价改写(`pytestmark` 换 sm100、`pv` 只留
sm100 唯一合法的 `"fp32"`、smooth_v 那个用例去掉写死的 `pv="fp16"`),
不是新写的判据。bench 形状网格 = 官方 `bench/` 那套(batch 4、heads 32、
head_dim 64/128、seq 1k-32k、causal 两态)加两个角:小 batch 长 seq
(b1 s65536、b2 s49152)与大 batch 短 seq(b64 s1024、b32 s2048)。

### 上机第一天暴露的两个 bug(都在 launcher,不在 kernel)

两个都是重构引入的,`csrc/sageattn/launch_helpers.cuh` 的
`set_max_dynamic_smem_once` 一个函数里。0a5d2e4 每次 launch 都无条件调
`cudaFuncSetAttribute`,重构把它换成「≤48 KB 就跳过 + 按字节数记忆化」,
两条捷径各自都是错的:

1. **`smem_bytes <= 48 * 1024` 直接 return**。48 KB 的默认额度是把静态
   `__shared__` 一起算的,而 sm100 kernel 有 5 个 mbarrier + 1 个 uint32。
   head_dim=128 的动态部分正好是 49152 B = 48 KB,加上静态就超了,于是
   **默认 TS 路径每一个 head_dim=128 的 launch 都挂**
   `CUDA error: invalid argument`。这是发布路径,不是 debug 路径。
2. **记忆化的 key 是 `smem_bytes`**。`KernelT` 只是函数指针**类型**,一族
   kernel 的所有 HEAD_DIM / gran / causal / lse 实例共用同一个类型,于是共用
   同一个 `static` 记忆槽:第一个申请到 N 字节的 kernel 把 N 写进去,后面
   所有同样要 N 字节的**别的** kernel 都被跳过,一个都没开成。SS twin
   (head_dim=128 要 65536 B)上表现为 16 个 hd128 用例只有先跑的那 4 个
   (per_warp + causal=False)过,其余全挂。

修法是回到 0a5d2e4 的语义:去掉 48 KB 捷径,记忆化改用 kernel 地址做 key
(`smem_bytes` 是实例的编译期常量,地址已经唯一确定它)。为什么别的 arch
没踩到:sm80 40960 B、sm89 32768 B、sm90 40960 B、sm100-TS 49152 B,全都
`<= 48 KB`,永远走第一条捷径 return 掉,函数体一次都没进过——所以 sm90 /
sm120 / sm89 三轮上机全绿,遮住了这个洞。它对今天的发布路径只影响 sm100,
但对任何以后超过 48 KB 的 kernel 都是地雷。

本机 sm_86 门禁:改动只在 host 侧,`cuobjdump -sass` 对改前改后的
`_C.abi3.so` **逐字节相同**;golden `ok=1493 diff=0`(+48 个 varlen equiv
extra),`pytest test/ -q` 548 passed / 138 skipped。

### profiling(ncu 2026.2.1;nsys 在这个容器里起不来)

nsys 的 launcher 在容器里 fork 出来就变 defunct、目标进程根本没起,换
torch profiler(Kineto)拿全流程占比,ncu 本身硬件计数器可用,不需要降级。

全流程时间占比(每次调用的 GPU 时间,b4 h32 d128):

| seq / causal | attention | QuantInt8(K) | transpose_pad | QuantInt8(Q) | MeanScale | `k.mean` | 其他 |
|---|---|---|---|---|---|---|---|
| 1024 / 0 (d64) | 65.5% | 5.2% | 5.4% | 4.3% | 11.6% | 5.6% | 2.5% |
| 1024 / 0 | 54.9% | 7.8% | 7.7% | 5.8% | 14.4% | 6.9% | 2.4% |
| 4096 / 0 | 84.8% | 4.1% | 3.7% | 2.5% | 2.2% | 1.8% | 0.9% |
| 4096 / 1 | 76.7% | 6.3% | 5.7% | 3.8% | 3.4% | 2.8% | 1.4% |
| 16384 / 0 | 95.9% | 1.2% | 1.1% | 0.8% | 0.4% | 0.4% | 0.2% |
| 16384 / 1 | 92.4% | 2.2% | 2.1% | 1.4% | 0.8% | 0.7% | 0.4% |

seq ≥ 4096 时前后处理合计 < 15%,seq=1024 时到 45%。按用户的口径
(占比 < 5% 的 kernel 不超过一轮),前后处理只有在短 seq 才值得动,而那正是
sm120 / sm89 两篇已经提过的同一个提案(transpose + MeanScale 融成一趟)。

attention kernel 本身(ncu,d128 seq4096 causal=0):

| 指标 | 值 |
|---|---|
| Compute (SM) Throughput | 28.1% |
| Memory / DRAM Throughput | 11.5% / 2.1% |
| Achieved Occupancy | 12.2%(7.83 warp/SM) |
| Active / Eligible warps per scheduler | 1.96 / **0.30** |
| No Eligible | 71.4% |
| Warp Cycles Per Issued Instruction | 6.84,其中 **37.7% 是 barrier** |

既不吃带宽也不吃算力,是纯延迟问题:一个 CTA 只有 1 个 warpgroup(4 warp,
每个 scheduler 1 个 warp),没有 warp specialization、没有多级流水,
每个 scheduler 3.5 个 cycle 才发一条指令。这与 §5b/§5e 上 sm89/sm120 那种
「tensor pipe 已经打满、拿寄存器换 occupancy 换不到吞吐」是**相反**的结论:
sm100 这个 MVP kernel 离硬件上限还很远(d128 seq32768 约 634 TFLOPS)。

- [ ] **C-9 主方向:warp specialization + 多级流水**。数据支持(barrier 占
      stall 37.7%、eligible warp 0.30),但是重写级工作量,这一轮没做。
- [ ] **TMEM 右尺寸化:测出 1.18-1.64× 但没有合入,原因见下。**

### 未合入的优化:TMEM 右尺寸化(head_dim=64)

`TMEM_COLS_TOTAL` 写死 512,而一个 SM 总共就 512 列,所以**一次只有一个 CTA
能持有 TMEM**:occupancy 允许 2 个 CTA 同时驻留,第二个只能堵在
`tcgen05.alloc` 里等第一个 dealloc。列计划实际要的是
`S(128) + P(32) + O(head_dim)`,head_dim=64 只要 224 列,凑到 2 的幂是 256,
两个 CTA 就都能拿到(head_dim=128 要 288,还是得进位到 512,所以只有 d64 受益)。

实测(TS 路径,3 轮交替中位数,40 配置):

| | 结果 |
|---|---|
| head_dim=64 | **1.18-1.64×**(seq 越长收益越大) |
| head_dim=128 | 1.000×(如预期完全不变) |
| 数值 | `ok=2280 diff=0`,精度 62/62 |
| ncu 机理确认(d64 seq4096) | Duration 1.62 → 1.02 ms;Compute SM 31.3% → 50.4%;eligible warp/scheduler 0.33 → 0.63;No Eligible 68.8% → 51.1% |

**没有合入**:同一份改动编进 SS twin 之后,`compare_reference.py --check` 的
`attn` 段(1980 个低层 kernel 用例)**挂死**——GPU 100% 占用、进程空转,
10 分钟不返回,可稳定复现。同一个 SS 构建单独跑都是好的:直接调
`qattn_sm100_*` 的 fuse_v_scale 与 base 两个 launcher、hd 64/128 ×
qo/kv {128,256,512,300/777,1024} × per_warp/per_thread × causal 两态 ×
`return_lse` 两态 × fp16/bf16 输出、以及 4096 CTA 的大 grid,全部 0.0x 秒通过;
`quant` / `e2e` / `equiv` 三段也都 `diff=0`。只有 attn 段整段跑才挂。
TS 路径带这个改动跑同样的 attn 段是过的(`ok=2280 diff=0`)。

假设:让两个 CTA 真正同时持有 TMEM 之后,SS 路径的 mbarrier / TMEM 交接里
有一个只在 CTA 并发时才现形的竞争(SS 比 TS 多一段 `sP` 的 smem staging 与
对应的 barrier)。SS twin 存在的意义就是当 TS 的 oracle,oracle 挂了就不能
放行被它盯着的那个改动,所以这一轮把它撤回,留复现命令等下一次上机:

```bash
# 复现(容器内,new 树带 TMEM_COLS_TOTAL 右尺寸化):
NVCC_APPEND_FLAGS=-DSAGE_SM100_PV_FROM_SMEM python setup.py build_ext --inplace
SAGEATTN_SM100_TCGEN05=1 python compare_reference.py --check --backend new \
    --golden-dir <golden> --section attn      # 挂在这里,TS 同命令是过的
```

## 6. 性能决策点

| 项 | 机器 | 命令/指标 | 决策 |
|---|---|---|---|
| C-8 QMMA.SF go/no-go | sm120 | `bash bench/microbench/run_microbench.sh`,看 QMMA.SF(fp32 accum)相对 f8f8f32 的比率;同时记录 f8f8f16:f8f8f32 | full-rate → 立项 sm120 v2(block-scaled mma);f8f8f16 与 f8f8f32 同速 → 退役 fp32+fp16 路径。**已采数,见下表** |
| H1/H4 sm90 wgmma 异步化 | H100/H200 | `ncu` 采 `sm__pipe_tensor_op_cycles_active / sm__cycles_active` 与 `smsp__warp_issue_stalled_barrier` | **按实验证据关闭,见下节**。重叠四件套已在 `cutedsl-sage-sm90` 上全部实测判负;提高 occupancy 的替代方向也判负 |

C-8 实测(RTX PRO 6000 Blackwell,110 SM,nvcc 13.3,2026-08-29):

| 指令(m16n8k32) | 精度 | 持续吞吐 | 对 f8f8f32 的比 |
|---|---|---|---|
| `s8s8s32` | 到 2^30 精确 | 588.6 TOPS | 1.00 |
| `f8f8f32` | 保 23 位尾数 | 588.6 TFLOPS | 1.00 |
| `f8f8f16` | 保 10 位尾数 | 588.6 TFLOPS | 1.00 |
| `QMMA.SF`(block-scaled,fp32 accum) | 保 23 位尾数 | 588.2 TFLOPS | 1.00 |

四条指令在这张卡上同速。按上表的判据:QMMA.SF 没有 1.7× 以上的
优势,sm120 v2(block-scaled mma)不成立;`f8f8f16` 与 `f8f8f32` 同速,
sm120 的 `pv_accum_dtype="fp32+fp16"` 拿不到 Ada 上那 2× 的速度,只剩
精度损失。`f8f8f32` 保满 23 位,证实 sm120 默认 `"fp32"` 是精确的。
`ptxas -arch=sm_120f` 接受 `kind::mxf8f6f4`,真要做 v2 时一个
`compute_120f` cubin 能同时覆盖 12.0 和 12.1。

同一份 `bench/microbench/mma_rate.cu` / `mma_accum_precision.cu` 换
`-gencode arch=compute_89,code=sm_89` 后在 L20 上的结果(92 SM,nvcc 13.3,
2026-08-29):

| 指令(m16n8k32) | 精度 | 持续吞吐 | 对 f8f8f32 的比 |
|---|---|---|---|
| `s8s8s32` | 到 2^30 精确 | 237.3 TOPS | 1.00 |
| `f8f8f32` | 保 13 位尾数 | 237.2 TFLOPS | 1.00 |
| `f8f8f16` | 保 10 位尾数 | 237.2 TFLOPS | 1.00 |

**L20 上也是同速**,`f8f8f16` 拿不到 Ada 消费卡那 2×。`f8f8f32` 只保 13 位
尾数(fp22 累加),这正是 sm89 需要 inst_buf 两级累加的原因,和 sm120 的
23 位不是一回事。这张表只代表 L20 这一个 SKU:4090 的 2× 速率是消费级 Ada
的特性,`pv_accum_dtype="fp32+fp16"` 该不该退役必须在 4090 上再采一次。

### H1/H4 sm90 终版结论(2026-08-29,H200 GPU 2)

**现行 kernel 的地位:不动。** `qk_int8_sv_f8_attn_kernel`(CTA_Q=64、CTA_K=128、
128 线程、167-168 reg、40 KB smem、3 CTA/SM)在已知的全部对照里是最快的一档:
`cutedsl-sage-sm90` 那条线用 CuTe-DSL 重写同一个算法,一路优化到最后也只是
**追平**它(d128 kernel-only 0.990-0.998×)。

**重叠四件套(producer WG + setmaxnreg / CTA_Q 64→128 / 多 stage 环 +
`warpgroup_wait<1>` / 双 consumer ping-pong)已全部实测判负**,证据在
`cutedsl-sage-sm90`:

| 判负项 | 出处 | 结论 |
|---|---|---|
| producer WG + CTA_Q=128 + ping-pong(384 线程,1 CTA/SM) | `921c5b6` `profile/2026-07-12-pingpong/REPORT.md` | 最优变体比 2 CTA/SM 慢 4.7-11.9%,错峰越强越慢 |
| 多 stage 环 + `warpgroup_wait<1>`(软件流水) | `b151142` `profile/2026-07-13-softpipe/REPORT.md` | 寄存器墙;同占用下重叠真兑现也只有 +4.8~7.2%(d128)、+1.1~1.3%(d64) |
| warp specialization 本身 | `1e87a34` | 主动**去掉** producer WG 与 setmaxnreg、退回 128 线程单 WG,才追平 CUDA |

两条与直觉相反、但被实测钉死的前提:① CTA_Q=128 省下的 K/V L2 流量换不来时间
(L2 吞吐 38.7%→17.4%,但 DRAM 只用 2%,离 memory bound 极远);② tensor pipe
只有 ~19-23% 的 imma 占用,**不是争用瓶颈**,ping-pong 要解的「MMA 互相排队」
在这里不存在,只剩同步开销。

**替代方向 A(降寄存器换 occupancy)判定:同样判负。** 本 kernel 卡在 167 reg
(4 CTA/SM 需 ≤128),给它加 `__launch_bounds__(NUM_THREADS, 4)`:

| | 寄存器 | spill(st/ld) | stack |
|---|---|---|---|
| d128 基线 | 167-168 | **0 / 0** | 16 B |
| d128 `__launch_bounds__(128, 4)` | 128 | **500 B / 456 B** | 264 B |
| d64(两版相同,对照组) | 128 | 0 / 0 | 16 B |

bench(GPU 2,`TORCH_CUDA_ARCH_LIST=9.0` 双树,4 轮交替 min-of-30,单位 µs):

| seq | causal | 基线 | launch_bounds | 基线 TFLOPS | lb TFLOPS | lb/基线 |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 0 | 1267.6 | 1591.7 | 867.4 | 690.8 | **1.256×** |
| 4096 | 1 | 703.6 | 893.8 | 781.3 | 615.1 | **1.270×** |
| 16384 | 0 | 19304.8 | 27327.5 | 911.3 | 643.8 | **1.416×** |
| 16384 | 1 | 9704.8 | 13634.4 | 906.4 | 645.1 | **1.405×** |

两道闸都判负:spill 500 B 远超 200 B 阈值,实测慢 26-42%。这正是 §5c 里 C-1 在
sm120 上的同款教训的 sm90 版本——d128 想要 occupancy 就得付 spill,ptxas 会把
省下的寄存器用 local memory 拼回来;d64 那一行是干净的对照,它本来就是 128 reg,
加不加约束都不动。**这个 kernel 不是 occupancy 受限的。**

**tensor pipe 的两个数字是口径差,不是矛盾。** 同一份
`ncu_hd128_s4096_c0.ncu-rep`、同为 `pct_of_peak_sustained_active`:
`sm__pipe_tensor_cycles_active` = **45.35%**(整条 tensor pipe),
`sm__pipe_tensor_op_imma_cycles_active` = **22.68%**(仅 imma 子管线),
比值正好 2.000——int8 QK 走 imma、fp8 PV 走另一半,各占一半 tensor cycles。
判断「tensor pipe 是否饱和」该看 45.35% 这个数,结论仍是**未过半**。

## 7. 完成后的收尾

- [ ] **条件已达成,尚未执行**:sm89(§5d)与 sm120(§5b)对拍都已全绿,
      可以删除 `qattn_smXX_*` 过渡 op(csrc/sageattn/ops.cpp 两处注册表 +
      sageattention/ops.py 的 `_QATTN_OPS`/fake + test/ 对应用例 +
      tools/compare_reference.py 的 equiv fwd 映射表)。删掉之后
      `compare_reference.py` 的 attn 段就没有低层入口了,所以要么同时改成
      走 `fwd`,要么接受对拍粒度降到 e2e——单独立项做
- [ ] 按 C-8 结论处置 fp32+fp16 路径:sm120 与 L20 上都与 `fp32` 同速且更不准
      (§6),但 sm89 家族还有 4090 这个 SKU 没量。**下一台 Ada 消费卡上先跑
      `mma_rate.cu`**,同速就两边一起退役,2× 就把默认改成按 SKU 选
- [ ] bench/ 脚本迁移到 torch.ops.sageattention.*(除 `bench_varlen.py` 外,
      其余还 import 已删除的 pybind 模块)
- [x] transpose + MeanScale 融成一趟:2026-08-30 落地(§5c「落地的优化二」),
      sm120 上短 seq e2e −5% ~ −18%,padded token 数 > 4096 回退两趟。**还没在
      sm89 / sm100 上量过**——两边的 kernel 占比与 sm120 一致(§5e / §5f),
      但 4096 这个门限是按 sm120 的 L2 定的,换卡要重新扫拐点
- [ ] attention kernel 的 softmax / mma 重叠(sm_89 上
      `math_pipe_throttle` 已占 stall 的 59%,sm120 同类):sm90 那一路已按
      实验证据关闭(§6 H1/H4 终版结论),sm89/sm120 立项前先读那三份判负
      报告——arch 不同不自动继承,但「重叠收益 < 占用代价」与「降 reg 换
      occupancy 会被 spill 拼回来」两条在 sm90/sm120/sm89 上各自都已实测成立
