# P4:V 前处理融合门限在 L20(sm89)的实测

`quant_v_fp8` 的 V 前处理有两条位级等价的路径:融合单 kernel
(`TransposeQuantFp8Kernel`,V 读两遍、免掉 V^T 中间缓冲)与分离双 kernel
(`TransposePadPermuteKernel` + `MeanScaleKernel`)。切换门限
`kFusedVQuantMaxTokens = 4096` 按 sm_120 实测定(quant_cuda.cu 注释:
-54% @2048 → -9% @4096 → +45% @8192)。融合路径的前提是第二遍读还命中
L2;L20 L2 = 96 MB(`prop.L2_cache_size` 实测),门限是否该抬,本轮回答。

**结论:L20 上融合路径一直赢到 12288(-20%),16384 才翻负(+3%);
4096 门限在 sm89 上把 6144-12288 段 1.7-1.25 倍的 V 前处理加速留在了桌上。**

(follow-up 2026-08-30:b·h 敏感性已补,四档拐点一致;门限已按 arch 分化
落地,sm89 = 12288,见 §5。)

## 1 口径

- 机器:ComputeLab L20 单卡独占(job 4023344,ipp1-1820),
  torch 2.13.0a0+nv26.07,CUDA 13.3,树 `eaf23cf`,CMake
  `SAGE_CUDA_ARCHS=8.9` 产品构建。
- 形状:b4 h32 d128 HND fp16,kv_len ∈ {1024,2048,3072,4096,6144,8192,
  12288,16384}(全部 64 对齐,padded == kv_len)。
- 参数:sm89 默认 plan(`mma_k16`、scale_max 2.25、smooth_v=False、
  pad_multiple 64)。
- 两条路怎么强制([microbench/vfuse_l20/p4_vfuse.py](microbench/vfuse_l20/p4_vfuse.py)):
  - 分离:直接调低层 op `transpose_pad_v` + `scale_fuse_quant`,与
    composite else 分支同分配、同参数;任意 kv_len 可跑。
  - 融合 >4096:`kFusedVQuantMaxTokens` sed 成 2^60 重编 `quant_cuda.cu`
    单 TU(门限是 host 侧常量,cubin 不变),composite 全程走融合。
- 计时:torch.profiler 每 kernel device µs(主口径,单遍 200/100/50 iters
  取每调用均值)+ CUDA event wall(辅,5 次重复取中位),warmup 5。
  两个构建交叉核对:
  融合 ≤4096(同代码路径)双侧差 ≤0.7%,分离全程(同 kernel)差 ≤1.6%。
- 分发自检:产品构建 composite 在 ≤4096 只见 `TransposeQuantFp8Kernel`,
  ≥6144 只见 `TransposePadPermuteKernel`+`MeanScaleKernel`,门限行为符合。

## 2 结果(kernel device µs,fused/separate <1 为融合更快)

| kv_len | 融合 µs | 分离 µs(transpose + meanscale) | 融合/分离 |
|---|---|---|---|
| 1024 | 49.0 | 76.9(36.6 + 40.2) | 0.64 |
| 2048 | 88.4 | 278.8(186.3 + 92.4) | **0.32** |
| 3072 | 289.9 | 473.3(287.7 + 185.6) | 0.61 |
| 4096 | 408.4 | 664.8(387.8 + 277.0) | 0.61 |
| 6144 | 619.2 | 1035.2(572.7 + 462.6) | 0.60 |
| 8192 | 877.1 | 1387.1(790.7 + 596.5) | 0.63 |
| 12288 | 1705.4 | 2117.8(1164.6 + 953.2) | 0.81 |
| 16384 | 2927.2 | 2832.4(1566.7 + 1265.8) | **1.03** |

带宽视角(字节按 V 大小 B_V = 32768·kv 折算,分离 ≈ 4.5·B_V,融合 ≈
2.5·B_V):分离路径随 kv 增大逼近 DRAM 上限(3072 时 957 GB/s 还有部分
L2 命中,16384 时 853 ≈ L20 标称 864),时间随 kv 基本线性;融合路径
2048 时 ~1900 GB/s(V 67 MB + fp8 34 MB 与 96 MB L2 同量级,基本全程
cache 内,0.32 的深谷就是这么来的),3072-8192 回落到 ~770-870 GB/s
(第二遍读仍大量命中 L2),12288 退到 ~590 GB/s,16384 掉到 ~460 GB/s
——第二遍读落出 L2 后,少搬的字节补不回 32 B gather 的效率,翻负。

## 3 拐点与建议

- L20 拐点在 **12288 与 16384 之间**(b·h=128 口径,与 sm_120 注释数字
  b8 h16 同基准);sm_120 的拐点(4096-5120)在 L20 上完全不成立。
- e2e 值多少:按 8/29 L20 sm89 kernel_breakdown
  ([kernel_breakdown_data/kb_new_sm89_20260829T181514Z.json.gz](kernel_breakdown_data/kb_new_sm89_20260829T181514Z.json.gz)),
  V 前处理对占 sageattn 总 kernel 时间 s=8192 时 5.8%、16384 时 3.1%、
  32768 时 1.6%;按本轮融合节省折算,抬门限到 12288 在 6-12k 段能拿回
  **e2e 约 1-2%**,更长序列 <0.5%。
- 建议:门限值得按 arch 分化——sm89(L20 96 MB L2)取 **8192(稳妥)
  或 12288(激进)**;分化落地前需补一次 b·h 敏感性(拐点跟并发
  (batch, head) 数与每 (b,h) 切片大小相关,本轮只测了 b·h=128 一档)。
  本任务纯测量,不动产品代码;数字供 kFusedVQuantMaxTokens 分 arch 化
  立项用。

## 4 产物

| 产物 | 路径 |
|---|---|
| microbench 脚本 | [microbench/vfuse_l20/p4_vfuse.py](microbench/vfuse_l20/p4_vfuse.py)、[p4_analyze.py](microbench/vfuse_l20/p4_analyze.py) |
| 原始 JSON(两个构建) | [microbench/vfuse_l20/p4_fix.json](microbench/vfuse_l20/p4_fix.json)、[p4_fuseall.json](microbench/vfuse_l20/p4_fuseall.json) |
| accuracy 数值 | [microbench/vfuse_l20/acc_sweep.csv](microbench/vfuse_l20/acc_sweep.csv)(附录 A) |
| 远端树与构建日志 | computelab `/home/scratch.sonlin_wwfo/workspace/nvidia/SageAttention_refactor/l20w3/` |

## 5 b·h 敏感性扫描与门限落地(2026-08-30)

§3 留的敏感性缺口本轮补齐,并把门限按 arch 分化落进产品代码。

### 5.1 口径

- 机器:ComputeLab L20 单卡独占(job 4025375,ipp1-1818),同 §1 环境
  (torch 2.13.0a0+nv26.07,CUDA 13.3);树 = `wave7/p4-vfuse-threshold`
  分支,CMake `SAGE_CUDA_ARCHS=8.9`。
- 形状:h32 d128 HND fp16,b ∈ {1,2,4,8} 给出 b·h ∈ {32,64,128,256},
  kv_padded ∈ {4096,6144,8192,12288,16384,24576};V 参数取 sm89 默认
  plan(mma_k16 / pad 64 / scale_max 2.25 / smooth_v=False)。
- 脚本:[p4_vfuse_sweep.py](p4_vfuse_sweep.py)(p4_vfuse.py 的
  arch 参数化版,V 参数从 `torch.ops.sageattention.plan` 取);方法同
  §1——分离走低层 op,>门限的融合数字用两个 return 全 sed 成 2^60 的
  fuseall 单 TU 重编构建;profiler per-kernel device µs 为主口径,
  CUDA event wall 交叉核对(两口径每格点方向一致,比值差 ≤0.01)。
- 构建间交叉:分离路两构建全格点、融合路 ≤8192(同代码路径)双侧差
  均 <3%;落地后 final 构建的融合 ≤12288 与 fuseall 差 ≤1.5%。

### 5.2 结果(融合/分离,kernel device µs 之比,<1 融合更快)

| b·h | 4096 | 6144 | 8192 | 12288 | 16384 | 24576 |
|---|---|---|---|---|---|---|
| 32(b1) | 0.85 | 0.42 | 0.31 | 0.81 | 1.05 | 1.28 |
| 64(b2) | 0.31 | 0.71 | 0.74 | 0.85 | 0.99 | 1.27 |
| 128(b4) | 0.62 | 0.60 | 0.64 | 0.81 | 1.04 | 1.30 |
| 256(b8) | 0.58 | 0.58 | 0.60 | 0.80 | 1.10 | 1.31 |

- 四档拐点一致:12288 全赢(0.80-0.85),16384 翻负(1.04-1.10;
  b·h=64 的 0.99 与两构建 ≤3% 的测量噪声同量级,不当获胜算),24576
  全负(1.27-1.31)。b·h=128 行与 §2(b4h32)逐点吻合(±0.01),
  两轮互为复现。小尺寸档(b·h≤64 且 kv=4096,绝对量 44-270 µs)比值
  在 0.31-0.85 间跳动,launch 占比高,不影响 ≥ 拐点段的判断。
- **定值:sm89 = 12288**——每档「最后获胜格点」的保守下界
  (b·h=64 名义到 16384,其余三档都是 12288)。
- 绝对值与原始行在 [microbench/vfuse_l20/p4s_fuseall.json](microbench/vfuse_l20/p4s_fuseall.json)
  (fuseall 构建)、[p4s_prod.json](microbench/vfuse_l20/p4s_prod.json)
  (门限 8192 的中间构建,用于交叉)、[p4s_final.json](microbench/vfuse_l20/p4s_final.json)
  (落地构建,b·h∈{64,256} 分发自检)。

### 5.3 两路 bit 等价(golden 不需重 dump)

fuseall 构建上全格点(4 档 × 6 kv = 24 个非平凡组合)同输入跑
composite(融合)与强制分离两路,`v_fp8` 按字节、`v_scale` 按 fp32
逐位比较:**全部相等**;final 构建 ≤12288 的 12 个非平凡格点同样全等。
与 `test_ops.py::test_quant_v_fp8_matches_two_kernel_path`(kv≤4096)
的既有结论一致,换路不改变任何输出位,golden 无需重 dump。

### 5.4 落地与门禁

- 代码:`csrc/sageattn/quant_cuda.cu` 的 `kFusedVQuantMaxTokens` 常量
  改为 `fused_v_quant_max_tokens(device)`:cc 8.9 → 12288,其余 arch
  维持 4096(sm120 为实测原值,sm90/sm100 未扫描,注释已标)。dense 与
  varlen 两个 composite 都走该函数。
- 门禁(L20,final 构建):`compare_reference.py --check`(golden-sm89,
  legacy baseline dump)**ok=2004 diff=0** missing=0,equiv 25/25;
  `pytest test/ -q` 434 passed / 264 skipped / 0 failed(cc8.9-only
  构建,sm80 家族用例 skip)。分发自检:composite ≤12288 只见
  `TransposeQuantFp8Kernel`,≥16384 只见分离双 kernel。
- e2e 预期不变:按 §3 的 kernel_breakdown 折算,6-12k 段拿回约 1-2%。
- B200(sm100,linear/pad128)与 sm120 复扫用同一脚本
  `bench/p4_vfuse_sweep.py`(--bitcheck 顺带出等价性),fused>门限
  需按脚本 docstring 重编 fuseall 变体。
- 远端产物:computelab `/home/scratch.sonlin_wwfo/workspace/nvidia/SageAttention_refactor/l20p4/`
  (tree/so/out:构建日志、golden/pytest 日志、三份 sweep JSON)。

## 附录 A:sm89 accuracy 实测数值(HARDWARE_CHECKLIST 待办)

同一 L20 会话、同一产品构建顺跑。`pytest test/test_accuracy.py -v`:
**126 passed / 0 failed / 0 skipped**(resolved backend sm89,Python 3.12,
torch 2.13.0a0+nv26.07,CUDA 13.3;日志在远端 l20w3/out/pytest_accuracy.log)。

实测数值来自 [microbench/vfuse_l20/acc_sweep.py](microbench/vfuse_l20/acc_sweep.py)
(与 `test_accuracy_pv_sweep` / `test_accuracy_smooth_v` 完全同参数、同
reference、同 metric,只是打印数值而非 assert),全表 104 case 见
[microbench/vfuse_l20/acc_sweep.csv](microbench/vfuse_l20/acc_sweep.csv):

| case | pv_accum_dtype | head_dim | n | cos_sim 最差 | rel_l1 最差 |
|---|---|---|---|---|---|
| pv_sweep | fp32 | 64 | 16 | 0.999297 | 0.037431 |
| pv_sweep | fp32 | 128 | 16 | 0.999296 | 0.037476 |
| pv_sweep | fp32+fp32 | 64 | 16 | 0.999297 | 0.037429 |
| pv_sweep | fp32+fp32 | 128 | 16 | 0.999296 | 0.037472 |
| pv_sweep | fp32+fp16 | 64 | 16 | 0.999318 | 0.036869 |
| pv_sweep | fp32+fp16 | 128 | 16 | 0.999313 | 0.036885 |
| smooth_v | fp32 | 64 | 8 | 0.999309 | 0.037138 |

全局最差 case:HND / hd128 / per_warp / fp32 / smooth_k=F,
cos_sim 0.999296 / rel_l1 0.037476。与 test_accuracy.py 注释里 sm90(H200)/
sm120(RTX PRO 6000)的 0.99926 / 0.039 同一水位,离门限
(cos > 0.99,rel_l1 < 0.06)余量充足。sm89 可以从「unmeasured」名单划掉,
三个 pv_accum_dtype 的数字都有了。
