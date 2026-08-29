# 上机验证清单(sm89 / sm120 / sm100)

本机(sm_86)只能运行 sm80 路径与 fused 量化 kernel 的 fp16 部分。
**sm90 已于 2026-08-28 在 H200 上全部验证完毕**(见 §5),剩余项需要
Ada(4090/L40S)、Blackwell consumer(5090)、B200/GB200 实机。
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
# 等价、v_fp8 尾部全零)只在 new 侧跑,不需要 baseline。
```

## 1. bitwise 对拍(剩余机器)

- [ ] sm89(4090/L40S):`--dump`/`--check` 全绿;A4-1 主循环合并(见 §3)与
      `quant_v_fp8(pad_multiple)` 修复由 equiv 段覆盖。
- [ ] sm120(5090):同上,并且 `backend="sm89"` fallback 路径与
      `backend="sm120"` 各自对拍(sm120 是同一批 sm89 TU 的
      `SAGEATTN_ARCH_NS=sm120` 双编译)。
- [ ] sm100(B200/GB200):顺序必须是 `SAGE_SM100_PV_FROM_SMEM=1` 的 SS twin
      先验数值 → 切 TS 路径对拍 → 才谈性能。`SAGEATTN_SM100_TCGEN05=1` 开门。

## 1b. varlen(packed cu_seqlens 布局)

本机 sm_86 只能跑 sm80 的 packed kernel,以及 fp8 V 流水里唯一 fp16 进
fp16 出的那一级(transpose)。fp8 的 V^T 量化和 sm89/sm120 的 packed
attention 一行都没在硬件上执行过。H200 也顶不上:sm90 resolve 到
`kSm90F8`,还没有 packed kernel(P3 的事)。

- [ ] sm89(4090/L40S)与 sm120(5090):`pytest test/test_varlen.py -q`。
      `sageattn_varlen` 那一段的开关是「本机 resolve 到哪个 packed
      backend」,所以在这两张卡上它跑的就是新的 fp8 packed kernel——等长
      batch 对 dense 的 `torch.equal`、ragged 的分段 SDPA 精度、
      bottom-right causal、空 KV 段、cudagraph 换分段 replay。
- [ ] 同两张卡:`quant_v_fp8_varlen` 的三个 fp8 用例(逐段对 dense 全等、
      每段 `[n_b, 段 padded 末)` 的 fp8 字节全零、opcheck)。这条白盒不变量
      是 sm89 V 加载能不带边界 predicate 的前提。
- [ ] 口径提醒:sm89 的 varlen 只实例化 `pv_accum_dtype="fp32+fp16"`、
      sm120 只实例化 `"fp32"`(plan.cpp 的默认值)。别的组合会明确报错,
      不会静默降级,所以对拍脚本里不要顺手换 pv。

## 2. A4-1 合并的性能复核(sm89/sm120)

本机已验证:SASS 算术/MMA/访存指令 132/132 实例逐条相同,spill 合计净减
(sm89 3680→3544 B、sm120 亦降),但个别实例升:

- sm89 `f32_fuse_v_scale_attn_inst_buf` spill 592/424 → 656/480
- sm120 `f16_fuse_v_scale_attn_inst_buf` spill 320/312 → 416/416

跑 fwd bench(交替中位数,>3% 才算信号)确认无劣化;有劣化则只回退对应
TU 的合并。

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

- [ ] sm120:`-DSAGE_OPT_FUSED_EPILOGUE=ON` 后 spill_st 合计 960→792,跑
      golden 确认数值 + fwd bench 看 −168 B spill 是否可测;无收益则宏留
      档不推荐。
- [ ] 高 tensor-core 配比卡上若 conversion 单元成为瓶颈,可重测 D-1
      (`SAGE_OPT_MAGIC_I2F`,sm80 kernel 专用)。

## 5. sm90 —— 已完成(H200, 2026-08-28)

| 项 | 结果 |
|---|---|
| bitwise 对拍(vs 0a5d2e4,1488 case) | diff=0 |
| equiv 段(fwd 全等 / pad 下沉等价 / 尾部全零) | 105/105 |
| pytest(含 compile/cudagraph) | 95 passed / 223 skipped |
| batch stride > 2^32 隔离(`test_large_seq_batch_isolation`) | PASSED |
| e2e bench(12 配置,3 轮交替中位数) | 全部加速 1.02-1.79×,零劣化 |

## 6. 性能决策点(未变)

| 项 | 机器 | 命令/指标 | 决策 |
|---|---|---|---|
| C-8 QMMA.SF go/no-go | 5090(sm120) | `bash bench/microbench/run_microbench.sh`,看 QMMA.SF(fp32 accum)相对 f8f8f32 的比率;同时记录 f8f8f16:f8f8f32 | full-rate → 立项 sm120 v2(block-scaled mma);f8f8f16 与 f8f8f32 同速 → 退役 fp32+fp16 路径 |
| H1/H4 sm90 wgmma 异步化 | H100/H200 | 先 `ncu` 采 `sm__pipe_tensor_op_cycles_active / sm__cycles_active` 与 `smsp__warp_issue_stalled_barrier`,确认 tensor core 空闲主导后再投入(重写级) |

## 7. 完成后的收尾

- [ ] sm89 + sm120 对拍全绿后:删除 `qattn_smXX_*` 过渡 op(csrc/sageattn/ops.cpp
      两处注册表 + sageattention/ops.py 的 `_QATTN_OPS`/fake + test/ 对应用例
      + tools/compare_reference.py 的 equiv fwd 映射表)
- [ ] 按 C-8 结论处置 sm120 的 fp32+fp16 路径
- [ ] bench/ 脚本迁移到 torch.ops.sageattention.*(bench 现在还 import 已删除的 pybind 模块)
