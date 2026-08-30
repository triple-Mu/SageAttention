# P8:sm80 k_scale fold(e85834d)A100 实卡复测

结案对象:[SM80_NONCAUSAL_FIX_REPORT.md](SM80_NONCAUSAL_FIX_REPORT.md) §4 的
「sm_80(A100)实跑未验」。这是 sm80 组的本命 arch:sm_80 SASS 跑在
GA100 上(此前只有 sm_86 笔记本与 L20 的 sm_89 SASS)。

**结论:A100 上 fold 无收益也无劣化——46 形状非 causal 合计,8 轮顺序
平衡后 base/fix = 0.9974,再除掉 causal 控制组(SASS 逐字节一致,量到
0.9986)的漂移后 = 0.9988,逐轮校正值 [-0.49%, +0.16%] 跨零。sm_86 的
0.9% 收益不迁移;三平台合议见 §5。**

## 1 口径

- 树:fix = `a264fc0` 的 git archive(含 fold commit `e85834d`);
  base = 同一棵树逆向 `e85834d` 的 hunk(只动
  `csrc/qattn/qk_int_sv_f16_sm80_impl.cuh`)。与 L20 报告同款
  tip ± fold hunk 口径。
- 机器:ComputeLab 单卡节点 ipp1-1782,NVIDIA A100 80GB PCIe
  (cc 8.0,单 GPU 档整节点独占,exclusive_check 干净),
  driver 595.58.03,pytorch_26.07-py3 容器(torch 2.13.0a0,CUDA 13.3)。
- 构建:CMake `SAGE_CUDA_ARCHS=8.0`(fatbin 只含 sm_80 目标码),base 侧
  删 `*sm80*.o` 强制重编(绕 ninja 空 deps 记录);`_C.abi3.so` 双侧各存
  一份,轮间只换文件(so_swap.log 8 次交替 sha256 无误,base `2b648ea2…`
  / fix `d54e1574…`)。
- backend:A100 默认 resolve 已是 sm80,仍走
  [kernel_breakdown_data/sm80_noncausal_fix_l20/kb_sm80.py](kernel_breakdown_data/sm80_noncausal_fix_l20/kb_sm80.py)
  强制 `backend="sm80"`,与 L20 口径逐字一致(远端副本 sha256
  `94fa4bfc…` 同一份);plan 落 per_thread + fp32 accum,实跑热 kernel
  `sage::sm80::qk_int_sv_f16_attn_kernel<…, float, false, __half, …,
  MaskMode 0, false, false>`。其余全走 `bench/kernel_breakdown.py`
  原口径:50 形状(46 非 causal + 4 causal),warmup 3,iters 10/5/3,
  attention role 每调用 µs,8 轮全部 ok=50。

## 2 这台 A100 的热漂移(为什么要 8 轮顺序平衡)

前 4 轮照 L20 排法 base 先跑(base r1 → fix r1 → base r2 → fix r2),
结果 causal 控制组也「慢」了 1.1%——SASS 逐字节一致的代码不可能变慢,
是机器在漂:

- 逐轮 50 形状合计 µs 单调上行,冷启动首轮最快,之后 +2.2%/+0.7%/+0.7%
  渐近;隔 5 分钟空闲(回落 210 MHz / 33 °C)再跑后 4 轮,曲线原样重演,
  谁先跑谁快。
- SM clock 全程顶格 1410 MHz,温度 33→67 °C(clock_snap.log):不是降频,
  是随板卡升温的整体变慢,量级 ~2-3%,比 fold 的预期效应大一个量级。

处置:后 4 轮反序(fix r3 → base r3 → fix r4 → base r4),两种排序各 2 对,
一阶漂移在 4 对几何平均里对消;causal 控制组单独给出残余漂移的标尺,
再用它除非 causal 合计得校正值。

## 3 结果(ratio = base/fix,>1 为 fix 更快)

| seq_len | 形状数 | 前 4 轮 gm(base 先跑) | 后 4 轮 gm(fix 先跑) | 8 轮 gm |
|---|---|---|---|---|
| 1024 | 2 | 1.0097 | 1.0058 | 1.0078 |
| 4096 | 12 | 0.9897 | 0.9987 | 0.9942 |
| 8192 | 2 | 0.9875 | 1.0067 | 0.9971 |
| 16384 | 3 | 0.9875 | 1.0094 | 0.9984 |
| 32768 | 12 | 0.9831 | 1.0104 | 0.9967 |
| 65536 | 12 | 0.9841 | 1.0106 | 0.9973 |
| 131072 | 3 | 0.9873 | 1.0082 | 0.9977 |
| 非 causal 合计 | 46 | 0.9856 | 1.0094 | 0.9974 |
| causal 合计(SASS 不动的控制组) | 4 | 0.9893 | 1.0081 | 0.9986 |
| **非 causal ÷ 控制组(漂移校正)** | 46 | 0.9963 | 1.0013 | **0.9988** |

- 校正后逐对:r1 0.9951 / r2 0.9975 / r3 1.0009 / r4 1.0016——越热越靠近
  1.0 且跨零,方向不稳定,判无信号。
- 每档 seq_len 的前/后 4 轮 gm 都随排序整体翻面(0.98x ↔ 1.01x),
  与控制组同步:表内结构全是漂移,不是形状效应。
- 逐形状 8 轮 gm:最差 0.9752(b1h5s4096,4 轮同向)、最好 1.0178
  (b1h3s4096)——两者都是 s=4096 小形状,该档噪声带 ±2%;
  s≥32768 大形状全落 0.9937-1.0020。
  全表 [sm80_noncausal_fix_a100/p8_ratios.csv](kernel_breakdown_data/sm80_noncausal_fix_a100/p8_ratios.csv)。

## 4 SASS 自检(sm_80 SASS,CUDA 13.3,fatbin 160 kernel 对比)

`qk_int_sv_f16` 家族 160 实例(dense 128 + varlen 32),**20 个变化,
全部 MaskMode 0 且 return_lse=false**(16 dense + 4 varlen,含
fuse_v_mean 变体);causal 与 lse 实例计数零变化,手术边界与 sm_86 /
L20 两轮一致。热实例(dense hd128、f32 accum、half、非 inst_buf):
fix 88 S2R / 3752 条,base 89 S2R / 3776 条——base 侧循环内 remat 在
sm_80 SASS 同样存在,fold 也同样消掉了它,但 A100 的 SM 把这条延迟藏掉了。
逐实例计数:[sm80_noncausal_fix_a100/s2r2_fix.tsv](kernel_breakdown_data/sm80_noncausal_fix_a100/s2r2_fix.tsv)
/ [s2r2_base.tsv](kernel_breakdown_data/sm80_noncausal_fix_a100/s2r2_base.tsv)。

## 5 三平台合议与建议

| 平台 | SASS | fold 实测(非 causal 合计) | 判定 |
|---|---|---|---|
| RTX 3080 Ti Laptop(sm_86) | sm_86 | fix/new 0.992(收回 ~0.9%) | 有收益 |
| L20(sm_89) | sm_89 | base/fix 0.9995 | 无收益无害 |
| A100 80GB PCIe(sm_80,本轮) | sm_80 | base/fix 0.9988(漂移校正) | 无收益无害 |

fold 的立项依据是 sm_80/sm_86;本轮确认在本命 arch A100 上它既不赚也
不赔(|效应| ≤ 0.2%,低于该机校正后分辨率)。三平台里唯一实证收益只剩
sm_86 笔记本(降频、访存饥饿的边缘场景)。**建议:可回收 e85834d 这个
hunk(17 行)换简单性**——代价仅 sm_86 一档的 0.9%;若 sm_86/87/88
(laptop / Jetson 类)仍是服务对象则保留,反正数据中心两卡上它无害。
本轮不执行回收,留给 merge 决策。

## 6 产物

| 产物 | 路径 |
|---|---|
| bench 原始 JSON(base/fix × r1-r4) | [kernel_breakdown_data/sm80_noncausal_fix_a100/](kernel_breakdown_data/sm80_noncausal_fix_a100/) |
| 逐形状 ratio 表 | 同上 `p8_ratios.csv`(生成脚本 `p8_merge8.py`,含控制组校正) |
| 轮前后 SM clock/温度/功耗快照 | 同上 `clock_snap.log` |
| SASS 逐实例计数 | 同上 `s2r2_{fix,base}.tsv` |
| 远端树 / 构建日志 / so / swap log | computelab `/home/scratch.sonlin_wwfo/workspace/nvidia/SageAttention_refactor/sage-a100/` |
| ComputeLab A100 profile(复用) | computelab `/home/scratch.sonlin_wwfo/workspace/nvidia/scripts/profiles/a100x1.yaml` |
