# P8:sm80 k_scale fold(e85834d)L20 复测

结案对象:[SM80_NONCAUSAL_FIX_REPORT.md](SM80_NONCAUSAL_FIX_REPORT.md) §4 的
「实跑未验」——那轮只有 sm_86 笔记本(降频 + 共享编码器),fix 收回 0.9%
勉强越过噪声标尺。本轮在 L20(sm_89)复测,sm80 组 kernel 由同一份源码
编成 sm_89 SASS。

**结论:L20 / sm_89 SASS 上 fold 无收益——46 形状非 causal 合计
base/fix = 0.9995(fix 反而慢 0.05%,两轮一致);causal 控制组 1.0001。
sm_86 上的 0.9% 收益不迁移到 L20。fold 的存留理由仍是 sm_80/sm_86 SASS,
在 sm_89 上它无害(≤0.1%)但也无功。**

## 1 口径

- 树:fix = `eaf23cf`(含 fold commit `e85834d`,612ab3d 同款);
  base = 同一棵 `eaf23cf` 逆向 `e85834d` 的 hunk(只动
  `csrc/qattn/qk_int_sv_f16_sm80_impl.cuh`)。与任务书「base=父提交树」
  字面口径的差别:e85834d 之后还有 3 个清理 commit(b12d8b9/08682c5/
  e53f509)也动过该头文件,取字面父提交会把它们混进对比;本口径 =
  tip ± fold hunk,与 sm_86 报告的 fix/new 配对同构,严格隔离 fold 本身。
- 构建:CMake `SAGE_CUDA_ARCHS=8.9`,同一 build 目录增量重编(删 sm80
  组 .o 强制,绕开 ninja 空 deps 记录的坑),`_C.abi3.so` 双侧各存一份,
  轮间只换文件不重编;每轮换前记 sha256(so_swap.log,base
  `4804ff96…` / fix `76eafb47…` 交替无误)。
- backend:L20 默认 resolve 到 sm89 fp8;
  [kernel_breakdown_data/sm80_noncausal_fix_l20/kb_sm80.py](kernel_breakdown_data/sm80_noncausal_fix_l20/kb_sm80.py)
  把 `get_plan` / `fwd` 强制到 `backend="sm80"`(plan 落在 per_thread +
  fp32 accum,与 sm_86 那轮默认 plan 同一 kernel 实例族;记录里实跑
  kernel 为 `sage::sm80::qk_int_sv_f16_attn_kernel<…, float, false,
  __half, …, MaskMode 0/1, …>`,已逐 JSON 验过),其余全走
  `bench/kernel_breakdown.py` 原口径:50 形状(46 非 causal + 4 causal),
  warmup 3,iters 10/5/3,attention role 每调用 µs。
- 轮次:base r1 → fix r1 → base r2 → fix r2 交替,4 轮全部 ok=50;
  ratio = base/fix 按轮配对,>1 为 fix 更快,聚合 = 组内 µs 求和后取比,
  两轮几何平均。

## 2 SASS 自检(sm_89 SASS,CUDA 13.3,fatbin 全 160 kernel 对比)

fix 与 base 的 `.so` 里 `qk_int_sv_f16` 家族 160 实例,**15 个
S2R/指令数变化,全部 MaskMode 0 且 return_lse=false**(dense + varlen、
hd64/128、f32/f16 accum、含 fuse_v_mean 变体);causal 与 lse 实例计数
零变化——手术边界与 sm_86 报告一致。实跑热实例(dense hd128、f32 accum、
非 inst_buf、half):fix 88 S2R / 3744 条,base 89 S2R / 3776 条,
**base 侧的 remat 在 sm_89 SASS 里同样存在,fold 也确实把它消掉了**。
逐实例计数:[sm80_noncausal_fix_l20/s2r2_fix.tsv](kernel_breakdown_data/sm80_noncausal_fix_l20/s2r2_fix.tsv)
/ [s2r2_base.tsv](kernel_breakdown_data/sm80_noncausal_fix_l20/s2r2_base.tsv)。

## 3 结果(ratio = base/fix,>1 为 fix 更快)

| seq_len | 形状数 | r1 | r2 | 两轮几何平均 |
|---|---|---|---|---|
| 1024 | 2 | 0.9984 | 0.9982 | 0.9983 |
| 4096 | 12 | 0.9982 | 0.9989 | 0.9985 |
| 8192 | 2 | 0.9993 | 0.9994 | 0.9994 |
| 16384 | 3 | 0.9993 | 0.9995 | 0.9994 |
| 32768 | 12 | 0.9991 | 0.9995 | 0.9993 |
| 65536 | 12 | 0.9994 | 0.9995 | 0.9995 |
| 131072 | 3 | 0.9996 | 0.9996 | 0.9996 |
| **非 causal 合计** | 46 | 0.9994 | 0.9996 | **0.9995** |
| causal 合计(SASS 不动的控制组) | 4 | 1.0001 | 1.0001 | 1.0001 |

- 轮间散布:逐形状 |r1-r2| 中位 0.0002、最大 0.0076(小形状);
  大形状(如 b8h40s8192,75.6 ms/调用)两轮重合到 ±0.02%,-0.05% 的
  方向是可复现信号,不是噪声。
- 逐形状几何平均最差 0.9966(b2h3s4096),最好 0.9996;
  全表 [sm80_noncausal_fix_l20/p8_ratios.csv](kernel_breakdown_data/sm80_noncausal_fix_l20/p8_ratios.csv)。

## 4 与 sm_86 那轮的对照

| | sm_86(RTX 3080 Ti Laptop) | L20(sm_89 SASS,本轮) |
|---|---|---|
| base 侧循环内 remat | 有(主循环 861 vs 860) | 有(全函数 89 vs 88 S2R) |
| fold 实测效果(非 causal) | fix/new 0.992,收回 ~0.9% | base/fix 0.9995,**无收益且 -0.05%** |

同一份源码、同一处 SASS 手术,收益不迁移:L20 的 SM(Ada,更高
occupancy/带宽配比)把 base 侧那条 S2R 依赖链的延迟藏掉了,省下 1 条
S2R + 32 条指令换不来周期;sm_86 笔记本(降频、访存饥饿)才吃这口。
fold 的立项依据本来就是 sm_80/sm_86(A100 未验仍待机会),本轮确认它
在 sm_89 上无害(合计 ≤0.05%,逐形状最差 -0.34% 在 b2h3s4096,两轮同向,
量级仍在半个百分点以内),不需要动作;若后续 A100 复测同样无收益,可以
考虑回收该 hunk 换简单性。

## 5 产物

| 产物 | 路径 |
|---|---|
| bench 原始 JSON(base/fix × r1/r2) | [kernel_breakdown_data/sm80_noncausal_fix_l20/](kernel_breakdown_data/sm80_noncausal_fix_l20/) |
| 逐形状 ratio 表 | 同上 `p8_ratios.csv`(生成脚本 `p8_merge.py`) |
| backend 强制 shim | 同上 `kb_sm80.py` |
| SASS 逐实例计数 | 同上 `s2r2_{fix,base}.tsv` |
| 远端树 / 构建日志 / so 三份 | computelab `/home/scratch.sonlin_wwfo/workspace/nvidia/SageAttention_refactor/l20w3/` |
