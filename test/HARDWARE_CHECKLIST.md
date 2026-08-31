# 上机验证清单(sm89 / sm120 / sm100)

本机(sm_86)只能运行 sm80 路径与 fused 量化 kernel 的 fp16 部分。
**sm90 已在 H200 上全部验证完毕**(dense 2026-08-28、varlen 2026-08-29,
见 §5),**sm120 已在 RTX PRO 6000 Blackwell 上验证完毕**(2026-08-29,
见 §5b),**sm89 已在 L20 上验证完毕**(2026-08-29,见 §5d),
**sm100 已在 B200 上验证完毕**(2026-08-29 首轮、2026-08-30 补 TMEM
右尺寸化,见 §5f)。四个 arch 的 bitwise 对拍与 varlen 都有硬件证据了。
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
- [x] sm100(B200,2026-08-29,已过时):当时**设计内不支持**,只验了报错可读
      (`ValueError: varlen is not supported by the sm100 backend`)。
- [ ] sm100 varlen Phase A M3(wave12/sm100-varlen-m2 起):packed kernel 已
      落地(经典 128 线程 kernel 的 `#ifdef SAGE_VARLEN` 分支 + 独立 TU
      `sm100_varlen` 命名空间;WS kernel 无 varlen 路,`SAGEATTN_SM100_WS`
      只影响 dense),`resolve()` 不再拒绝。B200 待跑:
      `pytest test/test_varlen_sm100.py test/test_varlen.py -q`。gate 组同
      sm90 文件(等长 batch 对 dense classic kernel `torch.equal`、ragged
      非 causal 逐段对 dense、CAUSAL_RAGGED 五组 bottom-right(含 dead row
      O==0 / lse==-inf)、空 KV 段、cudagraph、opcheck),外加
      (blk_q=128, warp_q=32, blk_k=128, warp_k=128) 四元组 quant 对拍;
      几何 CTA_Q=CTA_K=128、`v_layout="linear"`、`pad_multiple=128`、
      pv 只有 `"fp32"`。注意:该文件 import 时把 `SAGEATTN_SM100_WS`
      setdefault 成 "0"(dense 参照必须走同 kernel 体);显式设 WS=1/auto
      跑它会整文件 skip。上机清单见
      bench/sm100_review/SM100_VARLEN_DESIGN.md §6 M3。
- [x] sm89 / sm120 的 kernel 级 packed 用例已补:`test_varlen_sm89.py` /
      `test_varlen_sm120.py`,套用 `test_varlen_sm90.py` 的 gate 组(等长
      batch 对共享 body 的 dense kernel `torch.equal`、ragged 非 causal 逐段
      对 dense、bottom-right causal 精度、空 KV 段、opcheck),几何
      CTA_Q 128 / CTA_K 64、`pad_multiple=64`,pv 各用自己的默认(sm89
      `"fp32+fp16"` 配 V `scale_max=2.25`,sm120 `"fp32"` 配 448)。
      sm120 已上机(pro-5k RTX PRO 6000,2026-08-30):
      `pytest test/test_varlen_sm89.py test/test_varlen_sm120.py -q`
      62 passed / 62 skipped,skip 的 62 个全是 sm89 文件在 cc 12.0 上按
      resolved backend 让路。sm89 侧 62 个用例待 L20 复验轮顺跑(sm_86 与
      cc 12.0 两处均确认 collection 干净、只 skip)。

口径提醒:varlen 每个 arch 只实例化它自己的默认 `pv_accum_dtype`(sm80
`"fp32"`、sm89 `"fp32+fp16"`、sm90 `"fp32+fp32"`、sm100 `"fp32"`、sm120
`"fp32"`,见 plan.cpp)。别的组合会明确报错,不会静默降级,所以对拍和 bench
脚本里不要顺手换 pv。

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
| varlen 设计内拒绝 | `ValueError: varlen is not supported by the sm100 backend`(当日行为;wave12 起拒绝已删,sm100 有 packed kernel,见 §1) |
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
- [x] **TMEM 右尺寸化:hd64 快 1.19-1.64×,已合入。**上一轮卡住的 SS 挂死
      找到根因了(缺 async proxy fence),见下。

### TMEM 右尺寸化(head_dim=64)—— 已合入

`TMEM_COLS_TOTAL` 原先写死 512,而一个 SM 总共就 512 列,所以**一次只有一个
CTA 能持有 TMEM**:occupancy 允许 2 个 CTA 同时驻留,第二个只能堵在
`tcgen05.alloc` 里等第一个 dealloc。列计划实际要的是
`S(128) + P(32) + O(head_dim)`,head_dim=64 只要 224 列,凑到 2 的幂是 256,
两个 CTA 就都能拿到(head_dim=128 要 288,还是得进位到 512,所以只有 d64 受益)。

2026-08-30 复测(B200 `umbriel-b200-094`,TS 路径,3 轮交替取每形状中位数,
40 配置;baseline = 同一棵树把 `TMEM_COLS_TOTAL` 改回 512 重编):

| head_dim | 加速比 |
|---|---|
| 64 | **1.19-1.64×**(中位 1.53×,seq 越长收益越大) |
| 128 | 0.999-1.002×(中位 1.000×,如预期完全不变) |

hd64 分档:seq 1k 1.19-1.22×、4k 1.45-1.53×、16k 1.59-1.62×、
32k 以上 1.62-1.64×。hd128 整列 1.000× 正好当这次 A/B 的自检对照。

**A/B 的坑**:上一轮留在机器上的 `new/` 树其实**没有**把这个改动撤掉
(只是没进 git),所以 `_C.ts.abi3.so` 已经是右尺寸化的构建,拿它当
baseline 量出来是 1.00×。确认口径:`cuobjdump -sass` 里
`UTCATOMSWS.FIND_AND_SET.ALIGN` 前那条 `UMOV` 的立即数 = 列数/32
(8 = 256 列、16 = 512 列),量之前先核这个数。

ncu 机理(上一轮采的,d64 seq4096):Duration 1.62 → 1.02 ms;
Compute SM 31.3% → 50.4%;eligible warp/scheduler 0.33 → 0.63;
No Eligible 68.8% → 51.1%。

### 上一轮的 SS 挂死:根因是缺 `fence.proxy.async.shared::cta`

现象回顾:右尺寸化编进 SS twin 后 `compare_reference.py --check --section attn`
(1980 个用例)挂死,GPU 100% 占用、进程空转;TS 带同样改动是过的,SS 单独调
各 launcher 也都过,只有 attn 段整段跑才挂。

根因不是 mbarrier / TMEM 竞争,是 **smem 的 proxy 可见性**:SS twin 用普通
`st.shared`(generic proxy)把 P 写进 `sP`,而 `tcgen05.mma` 读 smem operand
走的是 **async proxy**。`tcgen05.fence::before_thread_sync` 只排序 TMEM 访问,
`__syncthreads()` 只排序 generic proxy 对自己,两个都不跨 proxy——所以 MMA
读到的 `sP` 是未定义的。TS 路径把 P 写进 TMEM,不经过 smem,所以一直没暴露。
1 CTA/SM 时时序上碰巧看不出来,2 CTA 真并发后才现形。

修法:`sP` 写完、`__syncthreads()` 之前加一条
`fence.proxy.async.shared::cta`(即 `tcgen05::fence_async_shared()`,
对应 CUTLASS 的 `fence_view_async_shared`)。

对照实验(两个构建的 TMEM 列数**完全相同**,hd64 都是 256,只差这条 fence):

| 构建 | `FENCE.VIEW.ASYNC.S` 条数 | `--section attn` |
|---|---|---|
| 右尺寸化,无 fence | 64 | **rc=124 挂死**(600 s 超时) |
| 右尺寸化,有 fence | 192 | `ok=1980 diff=0` |

这一轮的验收(同一份 golden-sm100,2280 case):

| 项 | 结果 |
|---|---|
| TS `--check`(发布路径) | `ok=2280 diff=0 env_mismatch=0 missing=0` |
| SS twin `--check`(oracle) | `ok=2280 diff=0 env_mismatch=0 missing=0` |
| SDPA 精度 | TS 62/62、SS 62/62 |
| pytest 全量 | 307 passed / 441 skipped,0 failed(与上一轮同数) |

因为 oracle 自己被修好了,**不需要**按 `PV_FROM_SMEM` 给 TMEM 列数分支:
TS 和 SS 用同一套右尺寸化,SS 继续当 TS 的数值 oracle。

留给下次的一条通则:任何**手写进 smem 再喂给 `tcgen05.mma` / `wgmma` 的
operand**,写完都要过一条 `fence.proxy.async.shared::cta`;TMA 填的 buffer
不需要(TMA 本身就在 async proxy 里,靠 mbarrier 完成)。本仓目前只有 SS twin
的 `sP` 属于前者。

### cutedsl 反向移植 Phase 1(2026-08-30,B200 `umbriel-b200-017`)

这一节引用的脚本(`cdsl_build.sh` / `cdsl_verify.sh` / `cdsl_bench.py` /
`cdsl_ab.sh` / `cdsl_report.py` / `cdsl_stress.py` / `cdsl_triage_a2.py` /
`cdsl_ncu.sh`)不在本仓,放在集群工作目录
`/home/scratch.sonlin_wwfo/workspace/nvidia/SageAttention_refactor/scripts/`,
原始 json / 日志在同目录的 `logs/` 下。

四项按序上机,每项独立 commit + 双路 golden + 同卡双向 A/B。基线是
feat/varlen tip `117d63c`,即上面那轮右尺寸化之后的树。

| 项 | 结论 | 数据 |
|---|---|---|
| A1 P 叠进 S 区 | **合入** | 全表几何均值 **1.513×**,d128 段 1.577×(1.283-1.687×) |
| A2 k_scale 寄存器预载 | **回退** | golden attn 段挂死,TS/SS 各吃满 1800 s timeout |
| A3 v_scale 预载 smem | **合入** | fuse_v_scale kernel 上几何均值 **1.0124×**(17 快 / 0 慢) |
| A5 TMEM 读批量发射 | **回退** | 间歇挂死,同一形状两次分别在第 1808、386 次发射停住 |

合入的两项(A1 + A3)是分支上最终留下的代码;A2、A5 只在实验里存在过,没有
留在历史里。

同族第三条经验约束(wave17 varlen 定位,SM100_VARLEN_DESIGN §6.4.6):
**S 排空的 `tcgen05.ld`/`wait::ld` 区间内不得混入 mask 的 compare/select
指令流**。varlen causal 实例把 bottom-right mask 编译成排空循环内约 500 条
谓词化 ISETP/FSEL(逐 tile 发射,谓词真假无关),混合/小 trip 的 causal
网格下 ~1/50-1/2000 launch 丢一个 mbarrier completion,单 CTA 永停在主循环
`wait(barrier_V)`;把 mask 挪到排空循环之后(寄存器上补打,数值逐位不变)
即转绿。B200 单变量矩阵(13 臂)与机理边界见 §6.4.6——A2/A5/本条同为
"机理未定位、按禁区绕行"的记录:golden 全绿不足以放行,必须过定点压测。

bench 口径:低层 `qattn_sm100_*_attn` op 直接对测(不含量化前后处理),形状
d128 × seq{1k,4k,16k,32k,128k} × batch{1,4} × causal 两态 = 20 点,外加 d64
两点作不回退对照;3 轮 A→B / B→A 交替、每形状取中位数、每点测前测后各查一次
`nvidia-smi --query-compute-apps`(22 点全部独占)。A1/A5 用不带 v_scale 的
launcher,A3 必须用 `fuse_v_scale` 的那个——不带 v_scale 的实例里 `sV_scale`
根本不会被写也不会被读,第一轮拿它测 A3 量出 1.0004×,是测错了对象。
cudnn 分母是同一份形状网格上的 `SDPBackend.CUDNN_ATTENTION`。

#### A1:P 叠进 S 区(commit `21bce17`)

`TMEM_COL_P` 从 128 改到 32,即 P 复用 S 的 [32,64) 列,`TMEM_COL_O` 128,
列计划 `128 + head_dim`:d128 从 288 降到 256、d64 从 224 降到 192,两个
head_dim 都进位到 256,于是 **d128 也能两个 CTA 同时持有 TMEM**。别名安全性
写在源码 `TMEM_COL_P` 上方,三条:TMEM lane == thread(线程只碰自己那条 lane,
无跨线程依赖)、tile 内 S 整行先 `tmem_ld` + `wait::ld` 读进寄存器再写 P、
跨 tile 的下一发 QK MMA 在所有线程过了 `wait(barrier_O_done)` 之后才发出。
`bench/sm100_review/barrier_ledger.md` 的跨 tile 冒险表补了对应两行。

SASS 自检(probe TU,sm_100a):`UTCATOMSWS.FIND_AND_SET.ALIGN` 前那条 `UMOV`
立即数,两个 hd128 实例 `0x10 → 0x8`(512 → 256 列),hd64 保持 `0x8`;
寄存器 255/255/254/254 不变,零 spill;整个 TU 指令数 13496 条不变——A1 的
收益全部来自 occupancy,不是指令面积。

| 验收 | 结果 |
|---|---|
| TS `--check`(发布路径) | `ok=2280 diff=0 env_mismatch=0 missing=0` |
| SS twin `--check`(2 CTA 并发是 SS 竞争的现形条件) | `ok=2280 diff=0 env_mismatch=0 missing=0` |
| A/B 全表几何均值 | **1.5126×**(20 点快 / 0 点慢 / 2 点持平) |
| d128 段 | geo **1.577×**,单点 1.283-1.687×(seq 越长收益越大) |
| d64 段(自检对照,列数本就是 256) | geo 0.998×,如预期不变 |
| 对 cudnn(d128 全段几何均值) | **0.471× → 0.743×**,最好的点 0.866× |

ncu(d128 seq4096 causal=0,ncu 2026.2.1):

| 指标 | 改前 | 改后 |
|---|---|---|
| Duration | 1.91 ms | 1.21 ms |
| Compute (SM) Throughput | 28.17% | 44.21% |
| Memory Throughput | 11.50% | 19.02% |
| Eligible Warps / scheduler | 0.30 | 0.58 |
| No Eligible | 71.35% | 55.04% |
| Registers / thread | 255 | 255 |
| Achieved Occupancy | 12.24% | 12.33% |

**occupancy 几乎不动是对的**:改前第二个 CTA 也已经驻留(Achieved Active
Warps/SM 7.83),只是堵在 `tcgen05.alloc` 里空转,warp 计数上算活着;改后它
真的在干活,所以变的是 Compute Throughput 和 eligible warp,不是 occupancy。
d64 两条 ncu 曲线改前改后完全重合(1.02 ms / 50.4%),与 bench 的 0.998× 一致。

#### A2:k_scale 寄存器预载 —— 挂死,已回退

改法是零 smem 的寄存器双缓冲:prologue 预载 tile 0 的标量,每个 tile 在顶部
发出 tile i+1 的 load。ptxas 侧看不出问题(零 spill,一个实例 254 → 255 寄存器,
LDG 条数不变,纯调度改动)。上机后 **golden 的 attn 段挂死**:

| 构建 | attn 段(1980 case) |
|---|---|
| A1 基线 TS | 20 s 跑完,`ok=1980 diff=0` |
| A2 TS | 1800 s timeout,日志零输出 |
| A2 SS | 1800 s timeout,日志零输出 |

quant / e2e / equiv 三段在 A2 下都是 `diff=0` 正常返回,只有 attn 段整段挂。

收窄到的**确定性复现**:`scripts/cdsl_stress.py`(单形状反复发射,发射前打印
序号,发射后 `synchronize()`),d128 / b4 h32 / per_warp,seq 扫描:

| seq | causal=0 | causal=1 |
|---|---|---|
| 2048 | 20 次干净 | 20 次干净 |
| 4096 | **挂在第 2 次** | 20 次干净 |
| 8192 | **挂在第 1 次** | **挂在第 0 次** |
| 16384 | **挂在第 0 次** | **挂在第 0 次** |
| 32768 | **挂在第 0 次** | **挂在第 0 次**(600 s timeout) |

同一形状同一脚本,A1 构建在 seq 32768 causal 连跑 8000 次干净。小形状也不复现:
`scripts/cdsl_triage_a2.py` 的 192 个 case(hd × gran × kv≤1024 × causal × lse ×
v_scale)在 A2 下全过。**这条线上的门限在 seq ≈ 4096**。

注意这还不是全部触发条件:golden attn 段最大的形状只有 qo=kv=2048 / b2 h8,
比上表里干净的那个点还小,却照样挂——说明除了 seq,至少还有一条触发路径没被
这个扫描覆盖(attn 段额外变的维度是 layout NHD、bfloat16 输出、GQA hq≠hk、
per_thread gran)。

越界读已逐条排除(causal 下 `num_iterations ≤ num_ctas_k`,预取下标
`(iter+1)*offset+cls` 恒在 `num_ctas_k*offset` 内;`T=1` 时走 peeled 分支不
预取),barrier 记账也一行没动。根因**仍未定位**。
**在根因清楚之前不要重开这一项**;重开时直接用上面那条复现命令。

#### A3:v_scale 预载 smem(commit `9567bf4`)

per-channel v_scale 原来在 epilogue 里逐通道 `__ldg`:CTA 的每一行都要读同一份
head_dim 个 float,等于最后一发 PV MMA 之后再发 head_dim 条广播 LDG,后面已经
没有活能盖住这段延迟。改成 prologue 里一次性搬进 smem(d128 是 512 B,由
TMEM alloc 后面那条 `__syncthreads()` 发布),epilogue 读 LDS。

静态 smem 仍是 1024 B——这个向量正好落进 barrier 本来就有的对齐填充里,所以
动态 smem 预算和每 SM 的 CTA 数一点没动。probe TU 四个实例指令数
13496 → 13360(`LDG.E.CONSTANT` 216 → 26,新增 48 条 `LDS.128`),零 spill。

| 验收 | 结果 |
|---|---|
| TS `--check` | `ok=2280 diff=0 env_mismatch=0 missing=0` |
| SS twin `--check` | `ok=2280 diff=0 env_mismatch=0 missing=0` |
| A/B(fuse_v_scale kernel)全表几何均值 | **1.0124×**(17 点快 / 0 点慢 / 5 点持平) |
| d128 段 | geo 1.0131×,单点 0.999-1.032× |
| d64 段 | geo 1.0057× |

短 seq 收益最大(d128 s1024 causal 到 +3.1%),长 seq +0.2-1.2%,与
「epilogue 在总时间里的占比随 seq 变小」一致。**方向一致性 17/0** 比 1.24%
的均值本身更硬。

#### A5:TMEM 读批量发射 —— 间歇挂死,已回退

改法是 O 修正与 epilogue 两处把 `num_tiles_o` 条 `tcgen05.ld` 一次发完再等一条
`wait::ld`(`wait::ld` 覆盖本线程此前所有 `tcgen05.ld`)。静态指标全是好的:
零 spill,指令数再降 16 条(正好是省掉的 wait),golden 双路 2280 `diff=0`。

但**定点压测挂死**。压测口径:d128 causal b4 s32768 单形状反复发射,每次
`torch.cuda.synchronize()`,发射前先打印序号,`timeout` 判死:

| 构建 | 结果 |
|---|---|
| baseline(A1 之前) | 2000 次干净 |
| A1 TS | 2000 次 + 8000 次,都干净 |
| A1 SS twin | 8000 次干净 |
| A1+A3 TS | 8000 次干净 |
| **A1+A3+A5 TS** | **两次独立挂死,分别停在第 1808、386 次** |

同一形状在 A/B bench 里也停过一次(5 分钟无进展)。golden 的 2280 个 case 全过
说明这不是数值问题、也不是小形状能碰到的;根因**没有定位**,`wait::ld` 的语义
(等本线程此前全部 `tcgen05.ld`)按 PTX ISA 读是够的,所以嫌疑指向 4 条
`tcgen05.ld` 同时在飞时的某个未写明的约束。**重开这一项之前先复现挂死并定位。**

两次挂死(A2、A5)有个共同点值得记:**golden 全绿不足以放行 sm100 的改动**,
这个 kernel 的挂死只在长 seq、大 batch、连续发射下现形,必须补一轮定点压测
(`scripts/cdsl_stress.py`,8000 次量级)。

#### 合入后的最终状态(A1 + A3)

最终构建 = A1 + A3(A2、A5 都不在分支上)。

| 门禁 | 结果 |
|---|---|
| golden TS `--check` | `ok=2280 diff=0 env_mismatch=0 missing=0` |
| golden SS twin `--check` | `ok=2280 diff=0 env_mismatch=0 missing=0` |
| 压测 seq 扫描(d128 b4 h32,seq 2048/4096/8192/16384/32768 × causal 两态,每点 20 次) | 10/10 全干净 |
| 压测长跑(d128 causal b4 s32768) | 8000 次干净 |
| 本机 sm_86 golden | `ok=1493 diff=0`(+48 varlen equiv extra,预期) |
| 本机 `pytest test/ -q` | 552 passed / 154 skipped / 0 failed |
| 全 arch SASS 对比(8.6;8.9;9.0;10.0;12.0,20 个 object / 2068 个 kernel / 5,548,880 条指令) | 只有 `qk_int_sv_f8_cuda_sm100.cu.o` 变了,其余逐字节相同 |

d128 对 cudnn 从 **0.471× 提到 0.743×**(全段几何均值),最好的点 0.866×;
Phase 1 的计划预期是 ~0.78×,实测落在这个量级上,差的那点是 A2 / A5 没进来。

### cutedsl 反向移植 Phase 3:两个结构候选都判负(2026-08-30,B200 `umb-b200-250`)

Phase 1 之后剩下的两个结构改动互相冲突,只能各做一版跟 A1 基线三方 A/B。
基线是 feat/varlen tip `6ae9fe6`(A1 + A3),两个候选各自独立分支,不叠加。
脚本 `p3_stage.sh` / `p3_ab3.sh` / `p3_report.py` / `p3_stress_scan.sh` /
`p3_phaseA.sh` / `p3_phaseB.sh` / `p3_phaseC.sh` / `p3_ncu.sh` /
`p3_ncu_retry.sh` / `ncu_table.py` / `stalls.py` 与 Phase 1 的脚本同目录
(集群 `SageAttention_refactor/scripts/`),日志在同目录 `logs/`。

| 候选 | 改动 | 结论 |
|---|---|---|
| **B1** KV 统一 ring | K/V 共用 `KV_STAGES=4` 个 smem slot(load 2i = K tile i,2i+1 = V tile i,slot = idx % 4);TMA 在 slot 空出时发,提前量从 1 发 MMA 变成 2 个 KV tile。smem/CTA 48 → 80 KB | **回退**,全表几何均值 1.0004,方向不一致 |
| **C2** S 双缓冲 ping-pong | TMEM 改 S0[0,128) + S1[128,256) + O[256,256+d),P 各叠进对应 S 区;QK(0) 剥进 prologue,tile i 先发 QK(i+1) 再做自己的 softmax;K 双缓冲配套 | **回退**,全表几何均值 0.6508,22 点全慢 |

#### 三方 A/B(同卡、3 轮轮换顺序、每形状取中位数,22 个形状全部独占)

`p3base-ts` = A1 基线。比值是 base / 候选,>1 表示候选更快。基线自己对 cudnn
的 d128 全段几何均值这一轮是 **0.760**(范围 0.539-0.887);Phase 1 记的 0.743
是另一台 B200 上另一轮的数,同一份代码,不要当成变化。

| 分段 | B1 vs 基线 | C2 vs 基线 |
|---|---|---|
| **全表几何均值** | **1.0004**(低于 0.5% 阈值) | **0.6508**(有信号) |
| 方向一致性(>0.5% 计) | 10 快 / 12 慢 | 0 快 / 22 慢 |
| d128 non-causal | 1.0140(0.977-1.062) | 0.6582 |
| d128 causal | 0.9878(0.985-0.995) | 0.6474 |
| d64 non-causal(对照) | 0.9973 | 0.6308 |
| hd128 s1024 / s4096 | 0.9962 / 1.0138 | 0.7465 / 0.6753 |
| hd128 s16384 / s32768 / s131072 | 0.9995 / 0.9973 / 0.9968 | 0.6234 / 0.6166 / 0.6119 |
| d128 对 cudnn(全段几何均值) | 0.761 | 0.496 |

**B1 不是「没效果」,是收益和代价互相抵消**:non-causal 全段 +1.4%(最好的点
d128 / batch 1 / s4096,+6.2%),causal 全段 −1.2%,两边都过 0.5% 阈值但符号相反,
所以按 BENCH_PROTOCOL 的方向一致性检验判负。四个形状的 ncu 给出的是同一张脸:

| 每形状 base → B1 | s4096 c0 | s4096 c1 | s16384 c0 |
|---|---|---|---|
| L1 命中率 | 88.40 → 92.14% | 88.46 → 91.19% | 88.75 → 93.01% |
| `long_scoreboard` | 0.97 → 0.89 | 1.11 → 0.96 | 0.83 → 0.80 |
| `wait` | 1.00 → 0.92 | 1.00 → 0.94 | 0.99 → 0.91 |
| **`short_scoreboard`** | 0.43 → **0.51** | 0.48 → **0.63** | 0.36 → **0.39** |
| eligible warps / scheduler | 0.60 → 0.60 | 0.55 → 0.55 | 0.65 → 0.65 |
| SM Throughput | 44.82 → 46.11% | 39.84 → 40.60% | 49.46 → 51.01% |

预取那一半确实奏效:L1 命中率涨 2.5-4.3 个点,`long_scoreboard` 和 `wait` 都降。
但**省下来的延迟没人接盘**——eligible warp 三个形状全部一动不动,只有 0.55-0.65,
腾出来的空档没有别的 warp 可以填。代价那一半是 `short_scoreboard` 上涨:ring
把 barrier 和 smem 基址都变成了动态下标(`barrier_kv[st]`、`sKV + st*TILE`),
这些访问走 MIO。涨幅最大的正是 causal s4096(+0.15),也正是 bench 上亏得最多的
那一段;non-causal 长序列涨得最少(+0.03),也正是赢的那一段。**这条归因是相关
关系,没有做消融**(比如把 `KV_STAGES` 固定成 2 让下标退回常量),重开时先补。

**C2 是教科书式的 occupancy 交换**:TMEM 从 256 列涨到 384 列 → 进位到 512
→ 每个 SM 只剩一个 CTA 真正在干活。ncu 上 Duration 1.19 → 1.83 ms(比值
0.650,跟 bench 的 0.6508 对得上),Compute (SM) Throughput 44.78% → 30.09%。
拿掉的正是 A1 花 1.51× 换来的东西,而 QK/softmax 重叠换回来的远不够。
cutedsl 单 tile 变体在 `core_sm100.py:1310` 列的翻案条件里,「S 双缓冲」这一条
到此关闭;剩下那条(2 CTA/SM + `setmaxnreg`)要求先有完整 warp specialization。

#### 门禁(两个候选都过,判负是纯性能结论)

| 门禁 | B1 | C2 |
|---|---|---|
| golden TS `--check` | `ok=2280 diff=0 env_mismatch=0 missing=0` | 同左 |
| golden SS twin `--check` | `ok=2280 diff=0 env_mismatch=0 missing=0` | 同左 |
| 压测 seq 扫描(d128 b4 h32,seq 2048/4096/8192/16384/32768/65536/131072 × causal 两态,每点 10 次) | 14/14 全干净 | 14/14 全干净 |
| 定点压测(d128 b4 h32 s32768,causal 两态各 8000 次) | 两态都干净 | 未跑(性能已判负) |
| 全 arch object 对比(8.6;8.9;9.0;10.0;12.0,23 个 object) | 只有 `qk_int_sv_f8_cuda_sm100.cu.o` 变 | 同左 |

四段 golden 的分段数是 quant 240 / attn 1980 / e2e 60 / equiv 0,equiv 段另有
105 个 case 不在 golden 里(`extra=105`),三份构建完全一致,是 golden 目录的
覆盖范围问题,不是改动引入的。

全 arch 对比这次没用 SASS 反汇编,直接比 23 个编译产物的 md5:**先用同一份源码
重建一次,23 个 md5 全部复现**,确认 nvcc 在这台机器上是确定性的,再拿这个
基准去比两个候选。两个候选都只有 sm100 那一个 object 变。构建树固定在同一个
路径下原地换 `.cu` 重建——换目录会让 `TORCH_CHECK` 里的 `__FILE__` 变化,
制造出假的差异。

#### ncu(d128 b4 h32 s4096 non-causal,ncu 2026.2.1,单次发射)

| 指标 | 基线(A1) | B1 | C2 |
|---|---|---|---|
| Duration | 1.191 ms | 1.200 ms | 1.834 ms |
| SM Throughput | 44.82% | 46.11% | 30.09% |
| Memory Throughput | 19.55% | 19.43% | 12.01% |
| Warps active / scheduler | 1.97 | 1.97 | 1.96 |
| **Eligible warps / scheduler** | **0.60** | **0.60** | 0.32 |
| L1 命中率 | 88.40% | 92.14% | 88.76% |
| L2 命中率 | 83.64% | 84.47% | 79.49% |
| DRAM 字节 | 303.6 MB | 302.8 MB | 304.9 MB |
| Registers / thread | 255 | 168 | 254 |
| Dynamic smem / block | 48 KB | 80 KB | 64 KB |
| occupancy 上限(寄存器 / smem) | 2 / — | 3 / 2 | 2 / 2 |

`--set full` 那一轮另外给出:基线 No Eligible 54.31%、Achieved Active Warps/SM
7.87;C2 No Eligible 69.38%、Achieved Active Warps/SM 7.83。C2 的 warp 计数
仍是两个 CTA 的量,但第二个 CTA 堵在 `tcgen05.alloc` 里空转——跟 Phase 1 记的
512 列旧版同一个现象,warp 计数上活着、实际不干活。**基线那一栏的
`Block Limit Shared Mem` 报 0,不可信**:动态 smem opt-in 超过 ncu 默认假设的
48 KB carveout 之后这一栏就失真,判 CTA 数要看 `Achieved Active Warps / SM`。

**B1 在 ncu 下有一个形状挂死,这本身是一条结论**。指定 metric 的少 pass 采集
在四个形状上跑了一遍:

| 形状(d128 b4 h32) | 基线 | B1 |
|---|---|---|
| s4096 non-causal | 正常 | 正常 |
| s4096 causal | 正常 | 正常 |
| s16384 non-causal | 正常 | 正常 |
| **s16384 causal** | 正常 | **只写出一行 `Connected to process`,吃满 600 s timeout(两次复现)** |

`--set full` 在 B1 上也两次跑满十几分钟后报 `==ERROR== An error occurred while
trying to profile`。同一条命令在基线和 C2 上每个形状都正常,所以嫌疑在 kernel
不在工具。正常发射下 B1 是干净的(golden 2280 双路、seq 扫描 14/14 含这个形状、
8000 次定点两态),但 A2/A5 的前科正是「常规路径全绿、换个发射节奏就挂」,
ncu 恰好就是换发射节奏。**这条没定位之前不要重开 B1**;复现命令是
`p3_ncu_retry.sh b1-ts 1 16384`。

#### warp stall 分解(每条已发射指令的 warp cycle,同一批采集)

| stall 原因 | 基线(A1) | B1 | C2 | 基线 causal | C2 causal |
|---|---|---|---|---|---|
| selected(发射本身) | 1.000 | 1.00 | 1.000 | 1.00 | 1.00 |
| wait(定长延迟依赖) | 0.996 | 0.92 | 1.034 | 0.99 | 1.02 |
| long_scoreboard | 0.969 | 0.89 | 0.529 | 0.88 | 0.46 |
| short_scoreboard | 0.429 | 0.51 | 0.402 | 0.37 | 0.38 |
| not_selected | 0.306 | 0.27 | 0.049 | 0.31 | 0.05 |
| **barrier** | **0.142** | 0.13 | 2.549 | 0.13 | 2.55 |
| no_instruction | 0.138 | 0.13 | 0.141 | 0.13 | 0.15 |
| branch_resolving | 0.124 | — | 0.279 | — | — |
| sleeping | 0.000 | 0.00 | 0.310 | 0.00 | 0.31 |
| membar | 0.000 | — | 0.000 | — | — |
| 合计(`--set full` 全项) | 4.304 | — | 6.392 | — | — |

后两列是 causal 的对照(d128 b4 h32 s16384):基线 8.643 ms / SM 47.72% /
eligible 0.64,C2 13.938 ms / SM 31.14% / eligible 0.33,比值 0.620,跟 bench
的 causal s16384 段 0.621 对得上。causal 与 non-causal 的 stall 形状几乎一样,
说明 C2 的亏损跟 mask 无关,就是那一个 CTA。

读这张表要注意一件事:这个 kernel 的 MMA 交接走的是 `mbarrier.try_wait.parity`
自旋,**不计进 `barrier`**;`barrier` 只统计每个 tile 那一次 `__syncthreads()`。
所以「基线的 barrier 只占 3.3%」的正确读法是「CTA 内的线程同步本身很便宜」,
不能直接读成「等 MMA 不花时间」。

#### Phase 4 / C1(完整 warp specialization)还值不值得做:值得,但立项理由要换

Phase 3 这两条负结果把 C1 的论证方式改了。原来的说法是「barrier 开销大,
warp specialization 去掉它」;实测下来基线的 `__syncthreads()` 只占 3.3%,
**这个理由不成立**。真正的天花板是另一件事:

- 每个 SM 只有 **1.97 个 active warp per scheduler**(2 CTA × 4 warp = 8 warp,
  4 个 scheduler),其中只有 **0.60 个 eligible**,54% 的 scheduler cycle
  无指令可发;
- 最大的两块 stall 是 `wait`(0.996,定长延迟依赖,softmax 的 FFMA/exp2 链)
  和 `long_scoreboard`(0.969,TMEM/global 读),两块加起来占 46%,**都是靠
  「有别的 warp 可以发」来盖的延迟,不是靠调结构能消掉的**;
- B1 证明了这一点的反面:它确实把 `long_scoreboard` 压到 0.89、`wait` 压到
  0.92、L1 命中率提到 92%,但 eligible warp 一点没动(0.60 → 0.60),
  端到端因此只有 +1.4%(还只在 non-causal 段),净收益归零;
- C2 证明了另一面:在 128 线程的地基上腾挪 TMEM,拿走 occupancy 的代价
  (0.65×)远大于结构重叠拿回来的。

所以 C1 的立项理由应该是**把 warp 数从每 scheduler 2 个抬到 8 个**——
cutedsl 用 16 warp 全特化跑到 62-67% SOL,对照基线的 44.8%,这才是那 2.31×
的来源。**代价是 128 线程地基假设全改**(thread == S/O 行 == TMEM lane 的
一一对应没了,softmax / correction / epilogue 全部要重写),属于重写级,
不是增量优化。建议:**单独立项,别挂在 Phase 3 后面继续做增量**;立项前先
在小 kernel 上验 `setmaxnreg` 在 sm100a 上编得过(sm90 有编译阻塞前科)。

同时关掉两条:B1 这类「加深 KV 预取」和 C2 这类「TMEM 内腾挪」在 128 线程
地基上都已实测判负,不要再试第三种排列。

### C1 ws G1 验收与 golden 切换点(wave11,B200 JID 4027436)

G1(softmax 值域改写,commit ae84b8b,tree dfaebb4)双级门禁全过,收为
ws kernel 最终态;lever A(5552fc2)的长序列劣化经四态单变量归因坐实后随
G1 一并消失。全部实测(golden 246 diff 分布、accuracy 0.999242/0.038967、
2×8000 压测、bench ws/old 1.1308、ncu −10.2%)见
`bench/sm100_review/C1_DESIGN.md` §9.6,原始数据在集群
`SageAttention_refactor/logs-w11/`。

**golden 切换点**:自 dfaebb4 起,sm100 ws 路(`SAGEATTN_SM100_WS=1`)的
bitwise gate 以 `golden-sm100-g1ws`(wave11 重 dump,自检 `ok=2107 diff=0`)
为准;旧路(`=0`)继续对 `golden-sm100`(`ok=2082 diff=0`)。auto 启发式
同轮改为 d128 全开(commit a1ca3b7),auto 抽查 d128 1.12-1.52×、d64 恰
1.0000。

## 6. 性能决策点

| 项 | 机器 | 命令/指标 | 决策 |
|---|---|---|---|
| C-8 QMMA.SF go/no-go | sm120 | `bash bench/microbench/run_microbench.sh`,看 QMMA.SF(fp32 accum)相对 f8f8f32 的比率;同时记录 f8f8f16:f8f8f32 | full-rate → 立项 sm120 v2(block-scaled mma);f8f8f16 与 f8f8f32 同速 → 退役 fp32+fp16 路径。**已采数,见下表** |
| H1/H4 sm90 wgmma 异步化 | H100/H200 | `ncu` 采 `sm__pipe_tensor_op_cycles_active / sm__cycles_active` 与 `smsp__warp_issue_stalled_barrier` | **按实验证据关闭,见下节**。重叠四件套已在 `cutedsl-sage-sm90` 上全部实测判负;提高 occupancy 的替代方向也判负 |
| sm90 行和交给 tensor core | H200 | `bench/sm90_rowsum/bench_rowsum.py` 双向 A/B | **判负,见下节**。d64 慢 7.3~7.8%、d128 慢 17%,三个变体零个点变快 |
| sm100 结构改动(128 线程地基上) | B200 | `p3_ab3.sh` 三方 A/B + `p3_report.py` | **判负,见 5f 的 Phase 3**。KV 统一 ring 几何均值 1.0004 且方向不一致,S 双缓冲 ping-pong 0.6508。要再往上走只剩完整 warp specialization(C1),单独立项 |

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

### sm90 行和交给 tensor core —— 判负(2026-08-30,H200 GPU 5)

`cutedsl-sage-sm90` 的 `c78c486` 在 DSL 侧把 online softmax 的行和从 CUDA core 的
FADD 归约换成 e4m3 wgmma(B 是全 1 的 tile),d64 实测 −2.1%、96/96 全快。同一想法
搬到本仓的 CUDA kernel 上,**三个变体全部变慢**,不落地。

**前验(通过)**:`bench/sm90_rowsum/probe_mma_layout.cu` 把同一份 `RS_f32_to_f8`
输出同时喂给 wgmma 的 RS 型 A 操作数和 `mma.sync.m16n8k32` 的 A 操作数。mma 的行和
与 host 端 e4m3 量化行和**逐位相同**(maxrel 0);wgmma(B 全 1)与 mma 差 1.66e-4,
那是 Hopper wgmma fp8 累加器尾数比 mma.sync 短,不是布局错位。所以「wgmma RS 型 A
布局 == mma.m16n8k32 A 布局」成立,方案不必改走 smem 全 1 tile。

顺带挖出一个潜伏 bug:`mma::rowsum_f8f8f32` 用 `_` 丢弃 D 的第 1、3 列,**ptxas 在
sm_90a 上拒收**(`Result discard mode is not allowed for instruction 'add'`),
sm_89 上接受。这个函数此前从没被任何 arch 实例化过(sm89 launcher 只用 kCudaCore
分母),所以没人撞到。已改成两个局部 `.reg .f32` 承接——这是本轮唯一留下的代码改动。

**A/B**:同卡独占、CUDA event min-of-20、A→B 与 B→A 各一轮、全表几何均值;
kernel-only 打 `qattn_sm90_..._fuse_v_scale_attn_inst_buf`;每档 28 点
= seq{1k,2k,4k,8k,16k,32k,64k} × gran{per_warp,per_thread} × causal{0,1},
batch/heads 随 seq 缩(64k 用 1×8)。

| 变体 | 档 | r1 | r2 | 变快/变慢 |
|---|---|---|---|---|
| 分母上 tensor core(mma 发在 V barrier 之前)+ rowmax 树 | d64 | 1.0775 | 1.0725 | 0/28、0/28 |
| 同上,mma 改发在 `commit_batch` 与 `wait<0>` 之间 + `__launch_bounds__(128,4)` | d64 | 1.0813 | 1.0809 | 0/28、0/28 |
| 只做 rowmax 树形化,分母仍在 CUDA core | d64 | 1.0113 | 1.0063 | 3/25、4/24 |
| 分母上 tensor core | d128 | 1.1694 | 1.1728 | 0/28、0/28 |
| 对照:d128 kernel 逐字未改 | d128 | 1.0012 | 1.0009 | 13/15、10/18 |

比值 = 新 / 基线,大于 1 是变慢。最后一行是噪声标定:同一份 d128 机器码两轮几何
均值 1.0012 / 1.0009,单点落在 0.977~1.020。d64 最好的那个点是 1.034,**比对照组
最差的单点还差**,所以这个量级不用讨论显著性。

**归因**:rowmax 树形化自己就要 0.6~1.1%(它把 num_tiles_k 个 tile max 同时留活,
d128 上多占 1 个寄存器);剩下的 6.5% 是那 4 条 `mma.sync.m16n8k32` 本身。两个发射
位置都试过,**发在 wgmma 的 commit/wait 空档里反而更差 0.9pp**——DSL 那条「同组提交
省 2.8%」的教训不能照搬,因为它的行和是 wgmma、和 PV 同组,而这里是 warp 级 HMMA
插进在飞的 wgmma 中间,两类指令在同一条 tensor pipe 上要排空重灌。**要复现 DSL 的
收益,行和也得是 wgmma(B = 1 KB 全 1 smem tile),不是 mma.sync**——另立项。

**寄存器**(`cuobjdump -res-usage`;d64 的生产实例是 `fuse_v_scale=true`):

| 变体 | d64 | d128 |
|---|---|---|
| 基线 | 128 × 48 | 167 × 12、168 × 36 |
| mma 发在 V barrier 之前 | 128 × 44、129 × 2、168 × 2(后 4 个都是 varlen per_thread causal) | 与基线相同 |
| mma 发在 commit/wait 空档 | dense 138~162 × 16、其余 128 | 与基线相同 |
| 空档 + `__launch_bounds__(128, 4)` | 128 × 48,但 12 个实例 4~16 B spill | 6 个 167→168 |
| d128 也开 | — | 173~209,3 CTA/SM 掉到 2 |

d64 基线正好卡在 128 = 4 CTA/SM 的线上,加 mma 就漂过去。把寄存器按回 128 之后
那一版仍然是最慢的,所以 occupancy 不是主因。

**精度**(顺带证实 DSL 的说法):分母从量化前 fp32 P 换成量化后 e4m3 P 之后,d64
对 fp32 SDPA 的指标全线略好——10 个形状 cos_sim 无一下降,平均 0.999343 → 0.999347,
rel_l1 平均 0.03687 → 0.03679;d128 逐位不变。`return_lse` 漂移(d64,单位 nat):
中位 ~1e-3,p99 3~9e-3,最大 2.4e-2,集中在 causal 的前几行(rows[0,32) 1.9e-2 vs
rows[512,1024) 6.0e-3)——短行的分母只由少数几个 e4m3 值相加,单值 3% 级的量化误差
没被平均掉。ring attention 这类跨 chunk 比 lse 的消费方要知道这个量级;本轮判负,
这条漂移没有进仓。

**作用面核对**:改动版对 `/workspace/sage-golden-sm90`(1488 case)的 diff 精确落在
sm90 head_dim=64 一档——attn 198 个、e2e 20 个,共 218 个;d96/d128、sm80 段、quant
段、equiv 段逐位不变,pytest 429 passed / 213 skipped。

原始数据 `bench/sm90_rowsum/data/*.jsonl`,采集脚本 `bench/sm90_rowsum/bench_rowsum.py`
与 `bench/sm90_rowsum/acc_rowsum.py`,前验 kernel `bench/sm90_rowsum/probe_mma_layout.cu`。

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
