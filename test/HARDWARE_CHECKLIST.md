# 上机验证清单(sm89 / sm120 / sm100)

本机(sm_86)只能运行 sm80 路径与 fused 量化 kernel 的 fp16 部分。
**sm90 已在 H200 上全部验证完毕**(dense 2026-08-28、varlen 2026-08-29,
见 §5),**sm120 已在 RTX PRO 6000 Blackwell 上验证完毕**(2026-08-29,
见 §5b),剩余项需要 Ada(4090/L40S)与 B200/GB200 实机。
sm89 与 sm120 全部通过后才可以删除过渡期的
`torch.ops.sageattention.qattn_smXX_*` 低层 op(它们是对拍工具)。

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

- [ ] sm89(4090/L40S):`--dump`/`--check` 全绿;A4-1 主循环合并(见 §3)与
      `quant_v_fp8(pad_multiple)` 修复由 equiv 段覆盖。
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
- [ ] sm100(B200/GB200):顺序必须是 `SAGE_SM100_PV_FROM_SMEM=1` 的 SS twin
      先验数值 → 切 TS 路径对拍 → 才谈性能。`SAGEATTN_SM100_TCGEN05=1` 开门。

## 1b. varlen(packed cu_seqlens 布局)

本机 sm_86 只能跑 sm80 的 packed kernel,以及 fp8 V 流水里唯一 fp16 进
fp16 出的那一级(transpose)。sm120 的 packed attention 已在 2026-08-29 上机
(见 §5b),sm89 的还一行都没在硬件上执行过。

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
- [ ] sm89(4090/L40S):同上。
- [ ] sm120 缺 kernel 级 packed 用例。`test_varlen.py` 里 `fwd_varlen` 那
      一组写死 `pv_accum_dtype="fp32"` + `v_layout="seq"`,所以 pin 在 sm80;
      sm90 另有 `test_varlen_sm90.py`。sm120 目前只有 API 级覆盖,补一个
      `test_varlen_sm120.py` 才能直接测它的 tile 几何与 fp8 V^T 布局。

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

## 3. C-1 inst_buf(方向已改,源码循环序方案作废)

实测结论(2026-08-28,本机 ptxas):head_dim=128 的 inst_buf 实例寄存器顶
在 255 上限,把活跃缓冲从 [2][8][8] 压到 [2][8] 后 ptxas 重新调度把压力原
样拼回(两种相差 2 倍的缓冲尺寸给出完全相同的 spill,且比基线差)。
上机方向:对这些实例试 `__launch_bounds__` / `-maxrregcount` 约束,判据
不变——f32 inst_buf 1576 B / f16 inst_buf 2088 B 的 spill 归零 + bench
`pv_accum_dtype=fp32+fp16, per_warp` 预期 +15-30%。

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
  ——它们写死了 sm80 才有的 `"fp16"` / `"fp16+fp32"`。把 pv 按 backend 参数化
  就能把这套精度门禁铺到 fp8 路径上。
- 跑测试的容器里有 `pip install -e /workspace/SageAttention`,它的
  MetaPathFinder 排在 `sys.meta_path` 末尾:待测树里缺席的子模块会静默落到
  那个安装上(本轮就撞到 `sageattention._qattn_sm90`)。用 PYTHONPATH 指向
  待测树之前先把这个 finder 摘掉。

## 6. 性能决策点

| 项 | 机器 | 命令/指标 | 决策 |
|---|---|---|---|
| C-8 QMMA.SF go/no-go | sm120 | `bash bench/microbench/run_microbench.sh`,看 QMMA.SF(fp32 accum)相对 f8f8f32 的比率;同时记录 f8f8f16:f8f8f32 | full-rate → 立项 sm120 v2(block-scaled mma);f8f8f16 与 f8f8f32 同速 → 退役 fp32+fp16 路径。**已采数,见下表** |
| H1/H4 sm90 wgmma 异步化 | H100/H200 | `ncu` 采 `sm__pipe_tensor_op_cycles_active / sm__cycles_active` 与 `smsp__warp_issue_stalled_barrier` | 确认 tensor core 空闲主导后再投入(重写级) |

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

## 7. 完成后的收尾

- [ ] sm89 + sm120 对拍全绿后:删除 `qattn_smXX_*` 过渡 op(csrc/sageattn/ops.cpp
      两处注册表 + sageattention/ops.py 的 `_QATTN_OPS`/fake + test/ 对应用例
      + tools/compare_reference.py 的 equiv fwd 映射表)。sm120 已过(§5b),
      还差 sm89 实机
- [ ] 按 C-8 结论处置 sm120 的 fp32+fp16 路径:同速且更不准,建议退役;
      退之前要确认 sm89(Ada)那一侧还需要它
- [ ] bench/ 脚本迁移到 torch.ops.sageattention.*(除 `bench_varlen.py` 外,
      其余还 import 已删除的 pybind 模块)
