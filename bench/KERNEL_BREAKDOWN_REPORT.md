# Kernel breakdown:重构前后逐 kernel 对照

把 `sageattn` 的一次调用拆成角色（quant、V 预处理、attention 等），在同一台机器上背靠背测 baseline 与 new 两侧，四个 arch 各 50 个形状。

- 采集日期 2026-08-29（UTC）；harness `bench/kernel_breakdown.py` @ `b8145d2`。
- 四机各两侧共八次正式运行，全部 50/50 形状成功，无 OOM、无未归类 kernel、独占检查全部通过。
- 结论一句话：**V 预处理是本次重构确认的收益点**（sm100 2.5–3.3×、sm90 2.3–2.8×、sm120 1.2–2.5×、sm89 1.0–1.8×）；attention kernel 两侧模板参数逐字相同，差异只来自函数签名，sm89/sm120 上系统性慢 1.5–1.7%；三项回退已立案（第 5 节）。

---

## 1 实验设计

### 1.1 目的

定位重构（`0a5d2e4` 的 per-arch pybind 模块 → `feat/varlen` 的单 `_C` 扩展 + `torch.ops.sageattention.*`）带来的逐 kernel 性能差。端到端时间只能给一个总数，看不出哪一路变快、哪一路变慢，所以按角色拆开分别对照。

两侧的唯一变量是安装的 sageattention 包：同一台机器、同一块 GPU、同一份 harness 脚本、同一组形状，相邻几分钟内先后执行。

### 1.2 形状矩阵

生成规则（`bench/kernel_breakdown.py` 的 `SHAPE_GROUPS`），`head_dim` 全部 128、`dtype` fp16、layout HND：

| group | b | h | s | causal | 形状数 |
|---|---|---|---|---|---|
| core | 1, 2 | 56, 40, 24, 10, 5, 3 | 4096, 32768, 65536 | 否 | 36 |
| longctx | 1 | 56, 40, 24 | 16384, 131072 | 否 | 6 |
| serving | 8 | 40, 24 | 1024, 8192 | 否 | 4 |
| causal | 1 | 56, 40 | 4096, 32768 | 是 | 4 |

设计依据：

- **head 数**取 sglang 侧 DiT 模型的锚点：56 = MiniMax-H3，40 = Wan，24 = HunyuanVideo / Flux / Qwen-Image。
- **10 / 5 / 3** 是上述锚点经 Ulysses 序列并行整除切分后单个 rank 剩下的 head 数（40÷4、40÷8、24÷8）。
- **s 档**按 DiT 的 token 量级取：1024 图像；4096 480p 短视频；8192、16384 720p；32768、65536、131072 长视频与长上下文。
- **d = 128** 是库默认，也是这批模型的实际 head_dim。
- **causal 单列一组**，因为 MaskMode 走的是另一个 kernel 实例，两侧各自编译 MaskMode 0 与 1 两份。

50 个形状的完整枚举：

| # | group | b | h | s | causal | shape_id |
|---|---|---|---|---|---|---|
| 1 | core | 1 | 56 | 4096 | 0 | `b1h56s4096` |
| 2 | core | 2 | 56 | 4096 | 0 | `b2h56s4096` |
| 3 | core | 1 | 56 | 32768 | 0 | `b1h56s32768` |
| 4 | core | 2 | 56 | 32768 | 0 | `b2h56s32768` |
| 5 | core | 1 | 56 | 65536 | 0 | `b1h56s65536` |
| 6 | core | 2 | 56 | 65536 | 0 | `b2h56s65536` |
| 7 | core | 1 | 40 | 4096 | 0 | `b1h40s4096` |
| 8 | core | 2 | 40 | 4096 | 0 | `b2h40s4096` |
| 9 | core | 1 | 40 | 32768 | 0 | `b1h40s32768` |
| 10 | core | 2 | 40 | 32768 | 0 | `b2h40s32768` |
| 11 | core | 1 | 40 | 65536 | 0 | `b1h40s65536` |
| 12 | core | 2 | 40 | 65536 | 0 | `b2h40s65536` |
| 13 | core | 1 | 24 | 4096 | 0 | `b1h24s4096` |
| 14 | core | 2 | 24 | 4096 | 0 | `b2h24s4096` |
| 15 | core | 1 | 24 | 32768 | 0 | `b1h24s32768` |
| 16 | core | 2 | 24 | 32768 | 0 | `b2h24s32768` |
| 17 | core | 1 | 24 | 65536 | 0 | `b1h24s65536` |
| 18 | core | 2 | 24 | 65536 | 0 | `b2h24s65536` |
| 19 | core | 1 | 10 | 4096 | 0 | `b1h10s4096` |
| 20 | core | 2 | 10 | 4096 | 0 | `b2h10s4096` |
| 21 | core | 1 | 10 | 32768 | 0 | `b1h10s32768` |
| 22 | core | 2 | 10 | 32768 | 0 | `b2h10s32768` |
| 23 | core | 1 | 10 | 65536 | 0 | `b1h10s65536` |
| 24 | core | 2 | 10 | 65536 | 0 | `b2h10s65536` |
| 25 | core | 1 | 5 | 4096 | 0 | `b1h5s4096` |
| 26 | core | 2 | 5 | 4096 | 0 | `b2h5s4096` |
| 27 | core | 1 | 5 | 32768 | 0 | `b1h5s32768` |
| 28 | core | 2 | 5 | 32768 | 0 | `b2h5s32768` |
| 29 | core | 1 | 5 | 65536 | 0 | `b1h5s65536` |
| 30 | core | 2 | 5 | 65536 | 0 | `b2h5s65536` |
| 31 | core | 1 | 3 | 4096 | 0 | `b1h3s4096` |
| 32 | core | 2 | 3 | 4096 | 0 | `b2h3s4096` |
| 33 | core | 1 | 3 | 32768 | 0 | `b1h3s32768` |
| 34 | core | 2 | 3 | 32768 | 0 | `b2h3s32768` |
| 35 | core | 1 | 3 | 65536 | 0 | `b1h3s65536` |
| 36 | core | 2 | 3 | 65536 | 0 | `b2h3s65536` |
| 37 | longctx | 1 | 56 | 16384 | 0 | `b1h56s16384` |
| 38 | longctx | 1 | 56 | 131072 | 0 | `b1h56s131072` |
| 39 | longctx | 1 | 40 | 16384 | 0 | `b1h40s16384` |
| 40 | longctx | 1 | 40 | 131072 | 0 | `b1h40s131072` |
| 41 | longctx | 1 | 24 | 16384 | 0 | `b1h24s16384` |
| 42 | longctx | 1 | 24 | 131072 | 0 | `b1h24s131072` |
| 43 | serving | 8 | 40 | 1024 | 0 | `b8h40s1024` |
| 44 | serving | 8 | 40 | 8192 | 0 | `b8h40s8192` |
| 45 | serving | 8 | 24 | 1024 | 0 | `b8h24s1024` |
| 46 | serving | 8 | 24 | 8192 | 0 | `b8h24s8192` |
| 47 | causal | 1 | 56 | 4096 | 1 | `b1h56s4096c` |
| 48 | causal | 1 | 56 | 32768 | 1 | `b1h56s32768c` |
| 49 | causal | 1 | 40 | 4096 | 1 | `b1h40s4096c` |
| 50 | causal | 1 | 40 | 32768 | 1 | `b1h40s32768c` |

### 1.3 测量方法

| 项 | 口径 |
|---|---|
| 被测调用 | `sageattn(q, k, v, tensor_layout="HND", is_causal=…)`，其余参数全部走库默认（smooth_k 开、quant 粒度按设备解析） |
| profiler | `torch.profiler`，activities = CPU + CUDA（CPU 侧是 kineto 关联 launch 所必需，只读回 CUDA 行） |
| warmup / 计时 | warmup 3 次；计时 N 次，N = 10（s ≤ 8192）/ 5（s ≤ 32768）/ 3（更大） |
| 单位 | `key_averages()` 按 kernel 名聚合 `self_device_time_total`，除以 N。每个数字都是**一次 sageattn 调用的设备侧微秒** |
| 保留项 | Memset / Memcpy 行保留，它们是真实设备工作 |
| 失败处理 | 单个形状 OOM 或报错只记录 `status` 并跳过，不中断整轮 |

**角色映射机制。** `ROLE_PATTERNS` 是一张「正则 → 角色」表，按顺序首次命中生效。两侧共用同一张表：重构把 attention kernel 挪进 `sage::smXX::`、把 fused kernel 挪进匿名 namespace，但没有改名，所以正则刻意不锚定前缀。命中不到的名字落进 `other` 并单独记入 JSON 的 `unmatched_kernels` —— 四机八次运行该字段全部为空。每份 JSON 同时保存**未经加工的 kernel 原始名 → (us_per_call, 调用数) 全表**，角色聚合只是它的派生视图。

**独占检查。** 采集前先用 `nvidia-smi --query-compute-apps` 拍一张快照，发现别的计算进程就直接拒绝测量。快照必须在 CUDA 初始化**之前**取：否则自己的 context 会被算成一个计算进程，而容器里 NVML 报的是宿主 PID、`os.getpid()` 报的是 namespace PID，两者无法比对，没法按 PID 过滤。八次运行的 `exclusive_check` 全部是 `status=exclusive, foreign=[]`。

**融合阈值。** new 侧的 V 预处理在 `padded_tokens ≤ 4096` 时走单 kernel 的 `TransposeQuantFp8Kernel`，超过则回落成 `TransposePadPermuteKernel` + `MeanScaleKernel` 两个 kernel（`csrc/sageattn/quant_cuda.cu:49`，`kFusedVQuantMaxTokens`）。所以主表里 **s ≤ 4096 与 s ≥ 8192 两组列测的不是同一条代码路径**，V-prep 行按「三个角色相加」对齐口径才可比。

### 1.4 两侧构建口径

| arch | baseline 构建 | new 构建 |
|---|---|---|
| sm89 | `setup.py build_ext --inplace`，per-arch pybind 模块 | `cmake -DSAGE_CUDA_ARCHS=8.9 -DSAGE_NVCC_THREADS=2`，`ninja --parallel 8` |
| sm90 | 同上，多 arch 全量 | `setup.py build_ext --inplace`（内部走 CMake），目标 9.0，`ninja -j12` |
| sm100 | 同上 | `cmake`，`SAGE_NVCC_THREADS=2`，链接耗时 1m57s |
| sm120 | 同上，多 arch 全量 | `cmake`，目标 12.0 |

`compiled_archs` 见第 2 节：baseline 侧在 sm90 / sm120 上是多 arch 全量构建，new 侧按单 arch 构建。两侧 `resolve_arch()` 解析出的执行 arch 相同（第 2 节「dispatch arch」列），所跑的 kernel 因此一致。

---

## 2 环境

硬件与运行时：

| arch | GPU | GPU UUID | 显存 | driver | torch | CUDA | 主机 | GPU idx |
|---|---|---|---|---|---|---|---|---|
| sm89 | NVIDIA L20 | `f5cca453-21fe-cfa8-f33a-76fd76879316` | 47.67 GB | 595.58.03 | 2.13.0a0+9186a08b2c.nv26.07 | 13.3 | a1u1g-rome-0090 | 0 |
| sm90 | NVIDIA H200 | `d3c43aa0-d97b-e971-d194-d8e3dc852f96` | 150.11 GB | 595.71.05 | 2.13.0+cu132 | 13.2 | node-radixark-16-0001 | 2 |
| sm100 | NVIDIA B200 | `09fefc33-9adf-9328-d62c-b77255734953` | 191.50 GB | 595.58.03 | 2.13.0a0+9186a08b2c.nv26.07 | 13.3 | umb-b200-263 | 0 |
| sm120 | NVIDIA Graphics Device | `68ea9307-0178-74a0-0c9d-25835e55b637` | 76.37 GB | 580.95.05 | 2.13.0+cu132 | 13.2 | R6KD-CX8aaS-GPU-07 | 0 |

sm120 那台的驱动没上报型号串，JSON 里就是 `NVIDIA Graphics Device`（cc 12.0，显存 76.37 GB）。

构建与源码：

| arch | 侧 | 源 commit | dirty | compiled_archs | dispatch arch | tcgen05 | 包路径 |
|---|---|---|---|---|---|---|---|
| sm89 | baseline | unknown | — | 80, 89 | 89 | 0 | `/workspace/SageAttention_refactor/sm89/baseline` |
| sm89 | new | unknown | — | 80, 89 | 89 | 0 | `/workspace/SageAttention_refactor/sm89/new` |
| sm90 | baseline | `a07d7fcc50f6` | 是 | 80, 89, 90, 100, 120 | 90 | 0 | `/workspace/SageAttention` |
| sm90 | new | `45d4f11ab158` | 是 | 80, 90 | 90 | 0 | `/workspace/SageAttention-kbd` |
| sm100 | baseline | unknown | — | 80, 89, 100 | 100 | 1 | `/workspace/SageAttention_refactor/baseline` |
| sm100 | new | unknown | — | 80, 89, 100 | 100 | 1 | `/workspace/SageAttention_refactor/new-kbd` |
| sm120 | baseline | unknown | — | 80, 89, 90, 120 | 120 | 0 | `/workspace/SageAttention-refactor/baseline` |
| sm120 | new | unknown | — | 80, 89, 120 | 120 | 0 | `/workspace/SageAttention-refactor/new-kbd` |

commit 一栏为 `unknown` 的，是因为那些树是 tarball 解出来的、没有 `.git`，harness 的 `source_commit()` 只能报 unknown。实际内容：

- **new 侧**：sm89 / sm100 / sm120 都是 `45d4f11` 的源码树，harness 单独换成了 `b8145d2` 版本（sm89 上用 `md5sum` 对过两份 `kernel_breakdown.py`）。sm100 的构建日志第一行写着 `NEW-KBD BUILD (tip 45d4f11, arch 10.0)`。
- **baseline 侧**：sm89 / sm100 / sm120 按 `0a5d2e4` 口径部署；sm90 那台用的是机器上已有的 checkout，commit `a07d7fcc`（dirty），不在本仓库历史里。四台 baseline 的 kernel 原始名逐字一致（未加 namespace、`unsigned int` stride、模板参数相同，见附录 A），因此按「pre-refactor per-arch pybind」同一代码口径对待。这是本报告最弱的一环，见 7.7。
- **容器镜像**：日志里没有记镜像名。sm100 走 Slurm + Pyxis/Enroot（`clab.py -p b200x4`，job 4015071）；sm89 / sm100 用容器自带的 `/usr/bin/python`（NGC 风格的 `nv26.07` torch），sm90 / sm120 用 venv（`/workspace/.sglang`、`/workspace/sgl-env`）里的 pip wheel torch。

`SAGEATTN_SM100_TCGEN05=1` 只在 sm100 上设置，且**两侧都设**——不设的话 tcgen05 kernel 根本不会被 dispatch。

---

## 3 主表

`ratio = baseline / new`，大于 1 表示 new 侧更快。每列是该 seq_len 档内所有形状的**求和**后再取比值（形状数：s=1024 → 2、4096 → 14、8192 → 2、16384 → 3、32768 → 14、65536 → 12、131072 → 3）。「50 形状合计」列是全矩阵求和，被最大的几个形状主导，只适合看总体方向。

`V-prep` = `transpose_pad` + `mean_scale_fp8` + `transpose_quant_fused` 三个角色相加。s ≤ 4096 走融合单 kernel、s ≥ 8192 走两 kernel，所以这三行单看不可比，只有相加后可比；它们的「50 形状合计」标 n/a。

### 3.1 四 arch 汇总

| arch | 指标 | s=1024 | s=4096 | s=8192 | s=16384 | s=32768 | s=65536 | s=131072 | 50 形状合计 |
|---|---|---|---|---|---|---|---|---|---|
| sm89 | V-prep | 1.832 | 1.270 | 1.062 | 1.035 | 1.036 | 1.024 | 1.030 | 1.038 |
| sm89 | attention | 0.992 | 0.988 | 0.984 | 0.983 | 0.987 | 0.984 | 0.983 | 0.984 |
| sm89 | TOTAL | 1.114 | 1.014 | 0.989 | 0.987 | 0.988 | 0.984 | 0.984 | 0.984 |
| sm90 | V-prep | 2.773 | 2.138 | 2.445 | 2.424 | 2.326 | 2.306 | 2.347 | 2.329 |
| sm90 | attention | 1.021 | 1.007 | 1.001 | 1.008 | 0.992 | 1.003 | 0.994 | 0.997 |
| sm90 | TOTAL | 1.364 | 1.115 | 1.073 | 1.048 | 1.014 | 1.013 | 0.999 | 1.008 |
| sm100 | V-prep | 3.254 | 2.816 | 2.778 | 2.724 | 2.693 | 2.545 | 2.529 | 2.607 |
| sm100 | attention | 0.993 | 0.995 | 0.995 | 0.995 | 0.995 | 0.995 | 0.995 | 0.995 |
| sm100 | TOTAL | 1.241 | 1.079 | 1.037 | 1.018 | 1.008 | 1.001 | 0.998 | 1.001 |
| sm120 | V-prep | 2.511 | 1.302 | 1.254 | 1.263 | 1.227 | 1.205 | 1.201 | 1.220 |
| sm120 | attention | 1.006 | 0.996 | 0.986 | 0.982 | 0.986 | 0.982 | 0.988 | 0.985 |
| sm120 | TOTAL | 1.227 | 1.030 | 1.006 | 0.993 | 0.992 | 0.986 | 0.989 | 0.988 |

### 3.2 sm89（L20）

ratio（baseline / new）：

| role | s=1024 | s=4096 | s=8192 | s=16384 | s=32768 | s=65536 | s=131072 | 50 形状合计 |
|---|---|---|---|---|---|---|---|---|
| k_mean | 0.995 | 1.035 | 0.999 | 1.009 | 1.002 | 1.000 | 1.000 | 1.002 |
| quant_q | 1.011 | 1.049 | 1.003 | 1.048 | 1.007 | 0.989 | 0.980 | 0.995 |
| quant_k | 1.025 | 1.015 | 0.989 | 1.036 | 0.998 | 0.980 | 0.971 | 0.986 |
| transpose_pad | 仅 baseline | 仅 baseline | 1.101 | 1.099 | 1.097 | 1.081 | 1.071 | n/a |
| mean_scale_fp8 | 仅 baseline | 仅 baseline | 1.011 | 0.951 | 0.959 | 0.965 | 0.996 | n/a |
| transpose_quant_fused | 仅 new | 仅 new | — | — | — | — | — | n/a |
| fill_memset | 1.034 | 0.906 | 1.103 | 0.983 | 0.992 | 0.991 | 0.930 | 0.968 |
| attention | 0.992 | 0.988 | 0.984 | 0.983 | 0.987 | 0.984 | 0.983 | 0.984 |
| V-prep | 1.832 | 1.270 | 1.062 | 1.035 | 1.036 | 1.024 | 1.030 | 1.038 |
| TOTAL | 1.114 | 1.014 | 0.989 | 0.987 | 0.988 | 0.984 | 0.984 | 0.984 |

绝对值（baseline µs → new µs）：

| role | s=1024 | s=4096 | s=8192 | s=16384 | s=32768 | s=65536 | s=131072 | 50 形状合计 |
|---|---|---|---|---|---|---|---|---|
| k_mean | 261.3→262.6 | 1016.1→982.2 | 1570.6→1572.0 | 878.9→871.1 | 6489.0→6473.4 | 9975.5→9971.3 | 5550.6→5551.8 | 25742.0→25684.4 |
| quant_q | 278.0→275.0 | 1043.8→995.2 | 2245.7→2239.4 | 974.3→930.0 | 8784.8→8724.1 | 14556.2→14711.4 | 8611.6→8791.7 | 36494.4→36666.7 |
| quant_k | 298.3→290.9 | 1005.3→990.3 | 2370.6→2396.1 | 1175.6→1134.4 | 9332.9→9355.7 | 15047.2→15347.9 | 8699.0→8956.1 | 37928.8→38471.3 |
| transpose_pad | 444.4→0.0 | 1768.2→0.0 | 3459.9→3142.0 | 1641.4→1492.9 | 13668.9→12463.6 | 22082.2→20433.6 | 12821.0→11976.0 | n/a |
| mean_scale_fp8 | 207.6→0.0 | 616.9→0.0 | 2486.9→2459.0 | 1094.9→1151.3 | 9597.9→10004.0 | 18754.3→19439.5 | 14271.1→14328.5 | n/a |
| transpose_quant_fused | 0.0→356.0 | 0.0→1877.6 | — | — | — | — | — | n/a |
| fill_memset | 2.0→2.0 | 11.3→12.4 | 2.1→1.9 | 2.3→2.3 | 10.5→10.6 | 7.0→7.0 | 1.8→2.0 | 37.0→38.3 |
| attention | 1390.7→1401.4 | 20103.9→20343.8 | 83688.5→85078.3 | 78737.9→80067.8 | 1210979.7→1227139.7 | 4315330.7→4387707.4 | 4995530.1→5080338.3 | 10705761.6→10882076.8 |
| V-prep | 652.0→356.0 | 2385.2→1877.6 | 5946.7→5601.0 | 2736.4→2644.2 | 23266.8→22467.5 | 40836.5→39873.1 | 27092.0→26304.5 | 102915.6→99123.9 |
| TOTAL | 2882.3→2587.9 | 25565.6→25201.6 | 95824.2→96888.6 | 84505.3→85649.8 | 1258863.7→1274171.0 | 4395753.1→4467618.2 | 5045485.1→5129944.3 | 10908879.4→11082061.4 |

### 3.3 sm90（H200）

ratio（baseline / new）：

| role | s=1024 | s=4096 | s=8192 | s=16384 | s=32768 | s=65536 | s=131072 | 50 形状合计 |
|---|---|---|---|---|---|---|---|---|
| k_mean | 0.992 | 0.996 | 0.998 | 0.996 | 0.998 | 0.997 | 0.999 | 0.998 |
| quant_q | 1.176 | 1.196 | 1.194 | 1.203 | 1.181 | 1.189 | 1.182 | 1.186 |
| quant_k | 0.861 | 0.775 | 0.807 | 0.821 | 0.806 | 0.791 | 0.777 | 0.793 |
| transpose_pad | 仅 baseline | 仅 baseline | 3.292 | 3.270 | 3.286 | 3.326 | 3.368 | n/a |
| mean_scale_fp8 | 仅 baseline | 仅 baseline | 0.955 | 1.081 | 1.036 | 1.000 | 0.998 | n/a |
| transpose_quant_fused | 仅 new | 仅 new | — | — | — | — | — | n/a |
| fill_memset | 0.939 | 0.953 | 0.986 | 1.004 | 1.037 | 0.988 | 1.047 | 0.993 |
| attention | 1.021 | 1.007 | 1.001 | 1.008 | 0.992 | 1.003 | 0.994 | 0.997 |
| V-prep | 2.773 | 2.138 | 2.445 | 2.424 | 2.326 | 2.306 | 2.347 | 2.329 |
| TOTAL | 1.364 | 1.115 | 1.073 | 1.048 | 1.014 | 1.013 | 0.999 | 1.008 |

绝对值（baseline µs → new µs）：

| role | s=1024 | s=4096 | s=8192 | s=16384 | s=32768 | s=65536 | s=131072 | 50 形状合计 |
|---|---|---|---|---|---|---|---|---|
| k_mean | 61.8→62.3 | 329.6→330.8 | 274.6→275.0 | 175.8→176.4 | 1323.4→1325.7 | 1933.0→1938.5 | 1010.4→1011.4 | 5108.5→5120.2 |
| quant_q | 87.5→74.4 | 357.4→299.0 | 673.8→564.6 | 314.9→261.6 | 2748.1→2327.0 | 4538.5→3817.6 | 2714.0→2295.9 | 11434.1→9640.0 |
| quant_k | 101.7→118.1 | 426.5→550.7 | 679.7→841.8 | 338.8→412.8 | 2766.7→3433.4 | 4333.2→5479.1 | 2457.7→3163.4 | 11104.2→13999.1 |
| transpose_pad | 332.4→0.0 | 1316.0→0.0 | 2600.2→789.8 | 1206.8→369.0 | 10637.8→3237.6 | 17642.2→5303.5 | 10676.6→3169.7 | n/a |
| mean_scale_fp8 | 136.7→0.0 | 258.5→0.0 | 429.2→449.4 | 251.2→232.4 | 2497.6→2410.3 | 4146.6→4146.1 | 2394.3→2399.9 | n/a |
| transpose_quant_fused | 0.0→169.2 | 0.0→736.3 | — | — | — | — | — | n/a |
| fill_memset | 2.4→2.5 | 13.9→14.5 | 2.5→2.6 | 3.1→3.1 | 14.6→14.1 | 10.7→10.8 | 2.9→2.8 | 50.0→50.4 |
| attention | 419.9→411.3 | 5104.7→5068.6 | 21143.8→21126.2 | 19199.4→19049.0 | 306418.7→308993.8 | 1112365.4→1109547.1 | 1327920.4→1336301.2 | 2792572.3→2800497.2 |
| V-prep | 469.1→169.2 | 1574.5→736.3 | 3029.4→1239.2 | 1458.0→601.4 | 13135.4→5647.9 | 21788.7→9449.6 | 13070.9→5569.6 | 54526.0→23413.2 |
| TOTAL | 1142.3→837.8 | 7806.6→6999.8 | 25803.8→24049.3 | 21490.0→20504.3 | 326406.8→321741.8 | 1144969.5→1130242.9 | 1347176.2→1348344.3 | 2874795.2→2852720.1 |

### 3.4 sm100（B200）

ratio（baseline / new）：

| role | s=1024 | s=4096 | s=8192 | s=16384 | s=32768 | s=65536 | s=131072 | 50 形状合计 |
|---|---|---|---|---|---|---|---|---|
| k_mean | 1.015 | 1.015 | 0.999 | 0.997 | 1.002 | 0.998 | 0.998 | 1.000 |
| quant_q | 0.944 | 0.943 | 0.935 | 0.934 | 0.936 | 0.935 | 0.935 | 0.936 |
| quant_k | 0.587 | 0.614 | 0.559 | 0.574 | 0.566 | 0.562 | 0.558 | 0.564 |
| transpose_pad | 仅 baseline | 仅 baseline | 3.428 | 3.398 | 3.414 | 3.427 | 3.430 | n/a |
| mean_scale_fp8 | 仅 baseline | 仅 baseline | 1.077 | 1.041 | 1.112 | 1.040 | 1.022 | n/a |
| transpose_quant_fused | 仅 new | 仅 new | — | — | — | — | — | n/a |
| fill_memset | 0.934 | 0.982 | 0.935 | 0.943 | 0.977 | 0.966 | 0.987 | 0.970 |
| attention | 0.993 | 0.995 | 0.995 | 0.995 | 0.995 | 0.995 | 0.995 | 0.995 |
| V-prep | 3.254 | 2.816 | 2.778 | 2.724 | 2.693 | 2.545 | 2.529 | 2.607 |
| TOTAL | 1.241 | 1.079 | 1.037 | 1.018 | 1.008 | 1.001 | 0.998 | 1.001 |

绝对值（baseline µs → new µs）：

| role | s=1024 | s=4096 | s=8192 | s=16384 | s=32768 | s=65536 | s=131072 | 50 形状合计 |
|---|---|---|---|---|---|---|---|---|
| k_mean | 54.5→53.6 | 301.9→297.6 | 203.2→203.4 | 146.4→146.9 | 1093.5→1091.7 | 1514.8→1517.7 | 734.8→736.4 | 4049.1→4047.2 |
| quant_q | 55.2→58.4 | 232.2→246.2 | 405.0→433.0 | 192.6→206.2 | 1628.2→1739.8 | 2620.1→2802.5 | 1513.3→1618.0 | 6646.7→7104.1 |
| quant_k | 56.6→96.6 | 232.3→378.5 | 391.8→700.4 | 192.1→334.7 | 1575.4→2783.7 | 2515.5→4475.3 | 1438.0→2577.5 | 6401.6→11346.6 |
| transpose_pad | 289.0→0.0 | 1176.7→0.0 | 2238.6→653.0 | 1060.0→312.0 | 8951.3→2621.7 | 14474.5→4224.0 | 8370.8→2440.5 | n/a |
| mean_scale_fp8 | 121.2→0.0 | 212.5→0.0 | 269.1→249.7 | 130.0→124.8 | 1330.3→1196.6 | 2571.1→2472.8 | 1492.6→1459.8 | n/a |
| transpose_quant_fused | 0.0→126.1 | 0.0→493.4 | — | — | — | — | — | n/a |
| fill_memset | 2.6→2.8 | 17.0→17.3 | 2.6→2.8 | 4.2→4.4 | 18.8→19.3 | 18.3→19.0 | 4.9→4.9 | 68.4→70.5 |
| attention | 641.5→645.8 | 7456.7→7495.3 | 28096.9→28226.3 | 25817.2→25939.2 | 392753.7→394558.4 | 1387192.9→1393616.7 | 1598962.9→1606563.4 | 3440921.9→3457045.2 |
| V-prep | 410.2→126.1 | 1389.2→493.4 | 2507.7→902.7 | 1190.0→436.8 | 10281.6→3818.2 | 17045.6→6696.8 | 9863.4→3900.4 | 42687.7→16374.4 |
| TOTAL | 1220.5→983.3 | 9629.3→8928.3 | 31607.3→30468.6 | 27542.4→27068.2 | 407351.3→404011.1 | 1410907.2→1409127.9 | 1612517.3→1615400.6 | 3500775.4→3495988.0 |

### 3.5 sm120（cc 12.0 单卡）

ratio（baseline / new）：

| role | s=1024 | s=4096 | s=8192 | s=16384 | s=32768 | s=65536 | s=131072 | 50 形状合计 |
|---|---|---|---|---|---|---|---|---|
| k_mean | 0.998 | 1.028 | 0.999 | 1.006 | 1.002 | 0.999 | 0.999 | 1.001 |
| quant_q | 1.041 | 1.024 | 1.000 | 1.006 | 1.001 | 1.000 | 1.000 | 1.001 |
| quant_k | 0.954 | 0.966 | 1.048 | 0.988 | 1.046 | 1.061 | 1.074 | 1.054 |
| transpose_pad | 仅 baseline | 仅 baseline | 1.453 | 1.485 | 1.464 | 1.453 | 1.448 | n/a |
| mean_scale_fp8 | 仅 baseline | 仅 baseline | 0.984 | 0.947 | 0.952 | 0.990 | 0.995 | n/a |
| transpose_quant_fused | 仅 new | 仅 new | — | — | — | — | — | n/a |
| fill_memset | 0.879 | 0.951 | 0.935 | 0.992 | 0.944 | 0.948 | 0.713 | 0.931 |
| attention | 1.006 | 0.996 | 0.986 | 0.982 | 0.986 | 0.982 | 0.988 | 0.985 |
| V-prep | 2.511 | 1.302 | 1.254 | 1.263 | 1.227 | 1.205 | 1.201 | 1.220 |
| TOTAL | 1.227 | 1.030 | 1.006 | 0.993 | 0.992 | 0.986 | 0.989 | 0.988 |

绝对值（baseline µs → new µs）：

| role | s=1024 | s=4096 | s=8192 | s=16384 | s=32768 | s=65536 | s=131072 | 50 形状合计 |
|---|---|---|---|---|---|---|---|---|
| k_mean | 136.1→136.4 | 565.9→550.5 | 929.8→930.8 | 469.4→466.5 | 3823.9→3816.8 | 6040.6→6046.2 | 3426.0→3428.7 | 15391.6→15375.9 |
| quant_q | 130.0→124.9 | 494.2→482.5 | 1318.9→1319.1 | 554.8→551.2 | 5096.6→5091.6 | 8562.9→8560.0 | 5105.0→5105.5 | 21262.3→21234.8 |
| quant_k | 171.2→179.4 | 628.0→650.3 | 1488.3→1419.8 | 667.7→675.8 | 5897.3→5640.5 | 9744.3→9182.4 | 5714.2→5318.0 | 24311.0→23066.2 |
| transpose_pad | 361.9→0.0 | 1446.1→0.0 | 2787.8→1918.8 | 1322.5→890.7 | 11144.1→7612.3 | 18030.1→12410.9 | 10431.6→7205.2 | n/a |
| mean_scale_fp8 | 155.4→0.0 | 348.8→0.0 | 1398.6→1420.8 | 594.2→627.4 | 6243.2→6555.0 | 14092.9→14236.1 | 8607.5→8647.2 | n/a |
| transpose_quant_fused | 0.0→206.0 | 0.0→1378.1 | — | — | — | — | — | n/a |
| fill_memset | 1.2→1.4 | 7.6→8.0 | 1.3→1.4 | 0.9→0.9 | 4.4→4.7 | 3.5→3.7 | 0.9→1.2 | 19.8→21.3 |
| attention | 735.5→730.8 | 9648.6→9685.3 | 42828.1→43430.5 | 38791.3→39491.7 | 603700.1→612305.6 | 2162894.1→2201517.0 | 2537948.6→2568925.3 | 5396546.4→5476086.2 |
| V-prep | 517.3→206.0 | 1794.9→1378.1 | 4186.5→3339.6 | 1916.7→1518.1 | 17387.3→14167.3 | 32123.0→26647.1 | 19039.1→15852.5 | 76964.8→63108.7 |
| TOTAL | 1691.4→1378.9 | 13139.0→12754.7 | 50752.8→50441.2 | 42400.8→42704.2 | 635909.7→641026.4 | 2219368.4→2251956.4 | 2571233.8→2598631.1 | 5534495.8→5598893.0 |

---

## 4 逐 arch 要点

### 4.1 sm89（L20）

V 预处理只在融合路径上有明显收益：s=1024 的 1.83×（652.0 → 356.0 µs），s=4096 降到 1.27×，s ≥ 8192 回落两 kernel 路径后只剩 1.02–1.06×。quant 两路基本持平（quant_q 0.98–1.05、quant_k 0.97–1.04），没有 sm90/sm100 上那种 128-token 档回退。attention 是这台机器的主线索：非 causal 的 46 个形状比值中位数 0.9837、标准差仅 0.21%（min 0.9833、max 0.9938），new 侧稳定多花约 1.7%；causal 的 4 个形状反过来是 1.005–1.015。attention 占 TOTAL 的 98%，所以整机 50 形状合计的 TOTAL 落在 0.984。同一台机器上约 6 分钟前还跑过一次完整的 50 形状（`prerun_45d4f11`，harness 是 `45d4f11` 版），两轮的 attention 合计比值都是 0.9838、逐形状差异中位数 −0.01%——这 1.7% 是系统性的，不是噪声。

### 4.2 sm90（H200）

V 预处理是四台机器里收益最稳的：全档 2.14–2.77×，50 形状合计 2.33×。拆开看，融合路径（s ≤ 4096）贡献一部分，但**回落路径上 new 侧的 `TransposePadPermuteKernel` 本身就快 3.27–3.37×**（s=8192：2600.2 → 789.8 µs），而两侧这个 kernel 的模板参数逐字相同、每次调用启动次数都是 1.0，只有函数签名不同。这个 3.3× 本实验只给出数字，成因需要单独 profiling。quant 两路反向分裂：quant_q 稳定 1.18–1.20，quant_k 稳定 0.78–0.86（new 耗时约 1.26×），见立案 1。attention 全档在 0.99–1.02 之间摆动，但逐形状标准差 2.65%（sm100 是 0.10%），所以只支持「持平」这一级结论；把同 3 个形状用 3 个独立进程连测 3 遍，spread 只有 0.4–0.9%，说明散布来自跨时间的时钟波动而非测量噪声。

### 4.3 sm100（B200）

V 预处理收益最大：2.53–3.25×，50 形状合计 2.61×；其中回落路径上 new 侧 `TransposePadPermuteKernel` 快 3.40–3.43×，与 sm90 同一现象。attention 在 7 个 seq 档里有 6 档都是 0.995（s=1024 是 0.993），逐形状标准差 0.10%——**这是四台里最干净的一组数**，也说明 0.5% 的差是真实且均匀的。1.00 附近属预期：本矩阵 `head_dim` 固定 128，而 sm100 的 TMEM 收益主要在 hd64 档，这里测不到。quant 是这台机器的问题所在：quant_k 全档 0.56–0.61（new 耗时约 1.77×）、quant_q 0.93–0.94，见立案 2。注意 sm100 的 QK 量化走 per_warp（`QuantGranularity 2`、`QuantInt8Kernel`），与其余三台的 per_thread（`QuantPerThread*Kernel`）不是同一个 kernel。

### 4.4 sm120

V 预处理 s=1024 有 2.51×，s=4096 起就只剩 1.20–1.30×；回落路径上 new 侧 `TransposePadPermuteKernel` 快 1.45–1.49×，明显低于 sm90/sm100 的 3.3×。quant 两路基本持平（quant_q 1.00–1.04、quant_k 0.95–1.07）。attention 与 sm89 同案：非 causal 46 个形状中位数 0.9865、标准差 1.04%，new 侧多花约 1.5%；causal 4 个形状是 1.005–1.017。TOTAL 从 s=1024 的 1.227 一路降到长序列的 0.986–0.989，交叉点在 s=8192 附近。

### 4.5 sm86 冒烟（本机 RTX 3080 Ti Laptop，dispatch arch sm80）

`--subset` 8 个形状，baseline `0a5d2e4`（clean）对 new `78988df`（dirty）。作用是先验证 harness 和角色映射，不作性能结论。

| role | 比值范围 |
|---|---|
| quant_q | 1.187 – 1.224 |
| quant_k | 1.603 – 1.643 |
| k_mean | 0.998 – 1.003 |
| attention | 0.933 – 1.067（本机噪声过大，不可用） |

sm80 路径不做 V 的 fp8 预处理，所以没有 V-prep 角色。另有一次刻意跑大形状的运行（`smoke/oom/`），`b2h56s131072` 在 16 GB 卡上 OOM，harness 记录 `status=OOM` 并继续跑完其余形状——OOM 路径本身也验证过。

---

## 5 发现与已立案项

### 立案 1 — sm90 quant_k 回退（WARP_TOKENS=128 档）

| 项 | 内容 |
|---|---|
| 现象 | `QuantPerThreadKInt8Kernel` 在 sm90 上全档 ratio 0.78–0.86，new 耗时约 1.26× |
| kernel | baseline `<128u, true, __half>` → new `<128u, 128u, true, __half>`：重构把 `warp_block_size` 从运行时参数提成模板参数 `WARP_TOKENS`，sm90 上取 128 |
| 对照 | 同一 kernel 在 sm89 / sm120 上 `WARP_TOKENS` 取 64，ratio 0.95–1.07，无回退 |
| 同侧收益 | 同机 quant_q 的 `WARP_TOKENS` 取 16，ratio 稳定 1.18–1.20（new 快约 16%） |
| 影响量级 | 占该 seq 档 baseline TOTAL：s=4096 +1.59%、s=8192 +0.63%、s=32768 +0.20%、s=131072 +0.05%。同档 V-prep 收益分别是 −10.74% / −6.94% / −2.29% / −0.56%，净 TOTAL 仍是 1.115 / 1.073 / 1.014 / 0.999 |

### 立案 2 — sm100 quant 回退

| 项 | 内容 |
|---|---|
| 现象 | `QuantInt8Kernel<128u, 128u, 2u, false, true, __half>`（quant_k）全档 ratio 0.56–0.61，new 耗时约 1.77×；quant_q `<128u, 32u, 1u, …>` ratio 0.93–0.94 |
| 关键点 | **两侧模板参数逐字相同**，只有函数签名从 `unsigned int` stride 变成 `long` stride + 多两个 varlen 参数（`int const*, unsigned int`）。回退不来自 tile 配置变化 |
| 与立案 1 的共同点 | 两个回退最重的 kernel，token 维度的模板参数都是 128（sm90 的是重构新加的，sm100 的两侧都有） |
| 影响量级 | 占该 seq 档 baseline TOTAL：s=1024 +3.27%、s=4096 +1.52%、s=8192 +0.98%、s=32768 +0.30%、s=131072 +0.07%。同档 V-prep 收益 −23.28% / −9.30% / −5.08% / −1.59% / −0.37%，净 TOTAL 仍是 1.241 / 1.079 / 1.037 / 1.008 / 0.998 |

### 立案 3 — sm89 家族非 causal attention 系统性回退

| 项 | 内容 |
|---|---|
| 现象 | 非 causal 形状的 attention kernel，new 侧稳定多花 1.5–1.7% |
| 数据 | sm89 中位数 0.9837（stdev 0.21%，n=46）；sm120 中位数 0.9865（stdev 1.04%，n=46） |
| causal 反向 | 同机 causal 4 个形状是 1.005–1.017，new 更快 |
| 关键点 | 两侧模板参数逐字相同（tile、`QuantGranularity`、`MaskMode`、全部 bool 开关都一致），差异只有 namespace 与签名：`float*` → `float const*`、`unsigned int` stride → `long` stride |
| 重复性 | sm89 上整轮 50 形状重跑过一次（间隔约 6 分钟），两轮 attention 合计比值都是 0.9838 |
| 影响量级 | attention 占 TOTAL 98%，所以直接决定整机结论：sm89 50 形状合计 TOTAL 0.984、sm120 0.988。长序列段（s ≥ 8192）TOTAL 全部 < 1 |
| 对照 | 同一改动在 sm100 上只有 0.995（+0.5%），sm90 落在噪声里。这两个 attention kernel 的入参是 TMA descriptor（`CUtensorMap_st`），推测指针与 stride 不参与主循环的地址计算——尚未验证 |

三项的共同背景是同一次签名改造（const 指针 + int64 stride + 多两个 varlen 参数）。影响并不均匀：**受影响重的是用裸指针做地址计算的 kernel**（sm89/sm120 的 attention、sm90/sm100 的 quant），入参走 TMA descriptor 的 attention kernel 受影响小得多（sm100 +0.5%，sm90 落在噪声里）。但同一个 quant kernel 为什么在 sm89/sm120 上不回退、在 sm90/sm100 上回退，本实验没有答案——三项立案都需要 SASS / ncu 层面的单独定位。

---

## 6 可复现指南

### 6.1 通用流程

harness 不 import 仓库里的任何东西，可以整份拷进没有 checkout 的 baseline 环境。三条硬约束：

1. **按路径运行，不要用 `-m`。** `-m` 会把当前目录插到 `sys.path` 头部，从 checkout 里跑就会 import 到源码树而不是被测的安装包。
2. **cwd 不能是 checkout。**上面几台机器都是 `cd /tmp` 之后再跑。
3. **两侧靠 `PYTHONPATH` 切换**，`--side` 强制标注（不传就自动探测：先试 `torch.ops.sageattention.compiled_archs()`，失败再试 `sageattention._fused`）。

```bash
# 1) baseline 环境（0a5d2e4，per-arch pybind 时代）
git worktree add --detach /path/baseline 0a5d2e4
cd /path/baseline
TORCH_CUDA_ARCH_LIST=<cc> MAX_JOBS=4 NVCC_APPEND_FLAGS=--threads=4 \
    python setup.py build_ext --inplace

# 2) new 环境（本分支，CMake 构建）
cd /path/new
cmake -S . -B build/cmake -G Ninja \
      -DPython_EXECUTABLE="$(which python)" \
      -DSAGE_CUDA_ARCHS=<cc> -DSAGE_NVCC_THREADS=4
cmake --build build/cmake -j 4
cp build/cmake/lib/sageattention/_C.abi3.so sageattention/

# 3) 新 arch 先验角色映射（一组一个形状，不落盘；有未命中就返回 1）
cd /tmp
PYTHONPATH=/path/baseline python /path/new/bench/kernel_breakdown.py --dump-names --device 0
PYTHONPATH=/path/new      python /path/new/bench/kernel_breakdown.py --dump-names --device 0

# 4) 正式采集，两侧写进同一个目录
PYTHONPATH=/path/baseline python /path/new/bench/kernel_breakdown.py \
    --side baseline --device 0 --out /shared/kb
PYTHONPATH=/path/new      python /path/new/bench/kernel_breakdown.py \
    --side new      --device 0 --out /shared/kb

# 5) 合并出报告（任一环境都行，不碰 GPU）
python /path/new/bench/kernel_breakdown.py --report /shared/kb \
    --csv /shared/kb/breakdown.csv --md /shared/kb/breakdown.md
```

`--subset`（每组首尾两个形状，共 8 个）是冒烟档；`--shapes 1x40x4096,2x56x32768c` 可以整体替换矩阵；`--allow-shared` 跳过独占检查（测出来的数不可用，只用于调试）。

### 6.2 四台机器的实际命令

**sm89 / L20**（`ROOT=/workspace/SageAttention_refactor/sm89`）

```bash
cd "$ROOT/new"
cmake -S . -B build/cmake -G Ninja -DPython_EXECUTABLE="$(which python)" \
      -DSAGE_CUDA_ARCHS="8.9" -DSAGE_NVCC_THREADS=2
cmake --build build/cmake --parallel 8
cp build/cmake/lib/sageattention/_C.abi3.so sageattention/

cd /tmp
PYTHONPATH="$ROOT/baseline" python "$ROOT/kernel_breakdown.py" \
    --side baseline --device 0 --out "$ROOT/kbd"
PYTHONPATH="$ROOT/new"      python "$ROOT/kernel_breakdown.py" \
    --side new      --device 0 --out "$ROOT/kbd"
python "$ROOT/kernel_breakdown.py" --report "$ROOT/kbd"
```

**sm90 / H200**（8 卡机，挑了空闲的 2 号卡；两侧都没传 `--side`，靠自动探测）

```bash
cd /workspace/SageAttention-kbd && python setup.py build_ext --inplace

cd /workspace
PYTHONPATH=/workspace/SageAttention \
  /workspace/.sglang/bin/python /workspace/SageAttention-kbd/bench/kernel_breakdown.py \
  --out /workspace/kbd_sm90 --device 2
PYTHONPATH=/workspace/SageAttention-kbd \
  /workspace/.sglang/bin/python /workspace/SageAttention-kbd/bench/kernel_breakdown.py \
  --out /workspace/kbd_sm90 --device 2
```

**sm100 / B200**（Slurm + Pyxis/Enroot；两侧都要开 tcgen05）

```bash
# 申请 1 卡 B200（QoS interactive-isolated-short），实测拿到 job 4015071 @ umb-b200-263
salloc --no-shell --account=<acct> --qos=interactive-isolated-short \
       --partition=<b200 分区列表> --nodes=1 --gres=gpu:b200:1 \
       --cpus-per-gpu=28 --mem=0 --time=03:59:59
clab.py -p b200x4 shell        # 进 Pyxis/Enroot 容器

cd /workspace/SageAttention_refactor/new-kbd
cmake -S . -B build/cmake -G Ninja -DPython_EXECUTABLE="$(which python)" \
      -DSAGE_CUDA_ARCHS="10.0" -DSAGE_NVCC_THREADS=2
cmake --build build/cmake
cp build/cmake/lib/sageattention/_C.abi3.so sageattention/

cd /tmp
export SAGEATTN_SM100_TCGEN05=1
KB=/workspace/SageAttention_refactor/new-kbd/bench/kernel_breakdown.py
PYTHONPATH=/workspace/SageAttention_refactor/baseline python "$KB" --dump-names --device 0
PYTHONPATH=/workspace/SageAttention_refactor/baseline python "$KB" \
    --side baseline --out /workspace/SageAttention_refactor/kbd_sm100 --device 0
PYTHONPATH=/workspace/SageAttention_refactor/new-kbd  python "$KB" \
    --side new      --out /workspace/SageAttention_refactor/kbd_sm100 --device 0
python "$KB" --report /workspace/SageAttention_refactor/kbd_sm100

clab.py -p b200x4 cancel     # 用完必须释放
```

这次是增量构建：日志头写的是 `arch 10.0`，但链接命令里带上了先前 configure 留下的 sm89 对象，所以成品 `.so` 的 `compiled_archs` 报 `[80, 89, 100]`。从零复现用 `-DSAGE_CUDA_ARCHS="10.0"` 即可，`resolve_arch()` 在 tcgen05 打开时一样解析到 100。

**sm120**

```bash
cd /tmp
KB=/workspace/SageAttention-refactor/new-kbd/bench/kernel_breakdown.py
PYTHONPATH=/workspace/SageAttention-refactor/baseline \
  /workspace/sgl-env/bin/python "$KB" --out /workspace/kbd_sm120 --side baseline --device 0
PYTHONPATH=/workspace/SageAttention-refactor/new-kbd \
  /workspace/sgl-env/bin/python "$KB" --out /workspace/kbd_sm120 --side new --device 0
```

### 6.3 数据溯源：JSON → CSV/MD → 本文档

链路三段，每段都可以单独重放：

| 段 | 产物 | 内容 |
|---|---|---|
| 采集 | `kb_{side}_sm{arch}_{stamp}.json` | `meta`（完整环境指纹 + argv + `role_patterns`）、`records[50]`（每形状的 `kernels{原始名 → us_per_call, count}` 与 `roles{角色 → us}`、`status`、`iters`）、`kernel_role_map`、`unmatched_kernels` |
| 合并 | `breakdown.csv` / `breakdown.md` | `--report` 按 `shape_id` 配对两侧，出「每形状 × 每角色」CSV 与「按 seq_len 聚合 + 每形状 total + 角色映射审计」的 markdown。每列都注明来源 JSON 文件名 |
| 本文档 | 第 3 节主表 | 对 `records` 按 `s` 分桶求和后取 `baseline/new`，逐格与各 `breakdown.md` 的聚合表核对一致；`V-prep` 行为本报告新增（三个角色相加），`breakdown.md` 里没有 |

回溯任意一格，例如「sm90，s=8192，quant_k，550.7 µs」：

```bash
# 该桶里每个形状的角色值
jq '.records[] | select(.s==8192) | {shape_id, roles}' kb_new_sm90_20260829T180529Z.json
# 这个角色由哪些 kernel 原始名组成
jq '.records[] | select(.shape_id=="b8h24s8192") | .kernels | keys' kb_new_sm90_20260829T180529Z.json
# 角色映射规则本身（存在 meta 里，不用回查代码版本）
jq '.meta.role_patterns' kb_new_sm90_20260829T180529Z.json
```

分桶求和即得主表数字；角色→原始名的对应见附录 A。

---

## 7 限制

1. **`v_pad_cat` 一次都没触发。**矩阵里所有 `s` 都是 128 的整数倍，baseline 在 sm90/sm100 上的 python 侧 V 零填充（`CatArrayBatchedCopy*`）因此完全没有出现在 profile 里。非 128 倍数的 seq 会多这一路开销，本报告对它没有数据。
2. **只测了 `head_dim = 128`。**sm100 的 TMEM 收益主要在 hd64 档，本矩阵测不到——这是 sm100 attention 停在 0.995 的直接解释之一。
3. **频率与功耗未锁、未记录。**四台都没锁 SM 频率，也没记功耗 cap。H200 上后果最明显：非 causal 逐形状 attention 比值 stdev 2.65%（sm100 0.10%、sm89 0.21%、sm120 1.04%），所以 sm90 的 attention 列只支持「持平」这一级结论，不支持 ±1% 的判读。把同 3 个形状用 3 个独立进程连测 3 遍，spread 只有 0.4–0.9%，说明散布来自跨时间的时钟波动而不是测量噪声。
4. **显存余量充足，没有 OOM。**四机全部 50/50 成功。矩阵最大的 `b2h56s65536` 与 `b1h56s131072`，q+k+v 各 5.25 GiB，最小的 L20（47.67 GB）也富余。16 GB 的笔记本 3080 Ti 在 `b2h56s131072` 上确实 OOM，但那是刻意的 OOM 路径验证，不构成本矩阵的边界。
5. **主表每格只测了一次。**重复性只抽查过两处：sm90 的 3 形状 × 3 连测，和 sm89 的整轮 50 形状重跑（间隔约 6 分钟，attention 合计比值两轮都是 0.9838）。
6. **两侧编译口径不完全对称。**baseline 在 sm90/sm120 上是多 arch 全量构建，new 侧按单 arch 构建。两侧解析出的 dispatch arch 相同，跑的 kernel 一致；但 `.so` 体积、代码布局、icache 行为不可比。
7. **sm90 的 baseline 不是 `0a5d2e4`。**那台用的是机器上已有的 checkout，commit `a07d7fcc`（dirty），不在本仓库历史里。判据只有 kernel 原始名与其余三台的 baseline 逐字一致（附录 A）。sm90 的数字因此比另外三台弱一档。
8. **只覆盖默认配置。**`sageattn` 的其余参数全走库默认，没有扫 `qk_quant_gran`、`pv_accum_dtype`、smooth_v、非 fp16 输入、varlen 接口。varlen 路径在本矩阵里只体现为多出来的两个 kernel 入参，没有被调用过。

---

## 附录 A 角色映射审计

四机两侧一共出现 36 个不同的 (kernel 原始名, 角色) 组合，全部列在下面。`arch` 列写「全部」表示四个 arch 上这个名字逐字相同。`unmatched_kernels` 四机八次运行全部为空。

| role | side | arch | 原始 kernel 名 |
|---|---|---|---|
| k_mean | 两侧 | 全部 | `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::Half, at::native::MeanOps<c10::Half, float, float, c10::Half>, unsigned int, c10::Half, 4, 8> >(at::native::ReduceOp<c10::Half, at::native::MeanOps<c10::Half, float, float, c10::Half>, unsigned int, c10::Half, 4, 8>)` |
| quant_q | baseline | sm100 | `void QuantInt8Kernel<128u, 32u, 1u, false, false, __half>(__half*, __half*, signed char*, float*, float, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int)` |
| quant_q | baseline | sm89,sm90,sm120 | `void (anonymous namespace)::QuantPerThreadQInt8Kernel<128u, __half>(__half*, signed char*, float*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int)` |
| quant_q | new | sm100 | `void QuantInt8Kernel<128u, 32u, 1u, false, false, __half>(__half*, __half*, signed char*, float*, float, unsigned int, long, unsigned int, long, long, long, long, unsigned int, long, long, long, int const*, unsigned int)` |
| quant_q | new | sm89,sm120 | `void (anonymous namespace)::QuantPerThreadQInt8Kernel<128u, 32u, __half>(__half*, signed char*, float*, unsigned int, long, unsigned int, long, long, unsigned int, long, long, long, int const*, unsigned int)` |
| quant_q | new | sm90 | `void (anonymous namespace)::QuantPerThreadQInt8Kernel<128u, 16u, __half>(__half*, signed char*, float*, unsigned int, long, unsigned int, long, long, unsigned int, long, long, long, int const*, unsigned int)` |
| quant_k | baseline | sm100 | `void QuantInt8Kernel<128u, 128u, 2u, false, true, __half>(__half*, __half*, signed char*, float*, float, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int)` |
| quant_k | baseline | sm89,sm90,sm120 | `void (anonymous namespace)::QuantPerThreadKInt8Kernel<128u, true, __half>(__half*, __half*, signed char*, float*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int)` |
| quant_k | new | sm100 | `void QuantInt8Kernel<128u, 128u, 2u, false, true, __half>(__half*, __half*, signed char*, float*, float, unsigned int, long, unsigned int, long, long, long, long, unsigned int, long, long, long, int const*, unsigned int)` |
| quant_k | new | sm89,sm120 | `void (anonymous namespace)::QuantPerThreadKInt8Kernel<128u, 64u, true, __half>(__half*, __half*, signed char*, float*, unsigned int, long, unsigned int, long, long, long, long, unsigned int, long, long, long, int const*, unsigned int)` |
| quant_k | new | sm90 | `void (anonymous namespace)::QuantPerThreadKInt8Kernel<128u, 128u, true, __half>(__half*, __half*, signed char*, float*, unsigned int, long, unsigned int, long, long, long, long, unsigned int, long, long, long, int const*, unsigned int)` |
| transpose_pad | baseline | sm100 | `void TransposePadPermuteKernel<128u, 64u, true, false, __half>(__half*, __half*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int)` |
| transpose_pad | baseline | sm89,sm90,sm120 | `void TransposePadPermuteKernel<128u, 64u, true, true, __half>(__half*, __half*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int)` |
| transpose_pad | new | sm100 | `void TransposePadPermuteKernel<128u, 64u, true, false, __half>(__half*, __half*, unsigned int, long, unsigned int, long, long, unsigned int, long, int const*, unsigned int)` |
| transpose_pad | new | sm89,sm90,sm120 | `void TransposePadPermuteKernel<128u, 64u, true, true, __half>(__half*, __half*, unsigned int, long, unsigned int, long, long, unsigned int, long, int const*, unsigned int)` |
| mean_scale_fp8 | baseline | 全部 | `void MeanScaleKernel<64u, false, __half>(__half*, signed char*, float*, float*, float, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int)` |
| mean_scale_fp8 | new | 全部 | `void MeanScaleKernel<64u, false, __half>(__half*, signed char*, float*, float*, float, unsigned int, long, long, long, long, long, long, long, long, long, long, int const*, unsigned int)` |
| transpose_quant_fused | new | sm100 | `void TransposeQuantFp8Kernel<16u, 64u, false, false, __half>(__half const*, signed char*, float*, float*, float, unsigned int, long, unsigned int, long, long, long, long, long, long, long, long, int const*, unsigned int)` |
| transpose_quant_fused | new | sm89,sm90,sm120 | `void TransposeQuantFp8Kernel<16u, 64u, false, true, __half>(__half const*, signed char*, float*, float*, float, unsigned int, long, unsigned int, long, long, long, long, long, long, long, long, int const*, unsigned int)` |
| fill_memset | 两侧 | 全部 | `Memset (Device)` |
| attention | baseline | sm100 | `void qk_int8_sv_f8_attn_kernel_sm100<128u, 128u, 128u, 128u, (QuantGranularity)2, (QuantGranularity)2, __half, (MaskMode)0, false, true, false>(CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, float*, float*, float*, __half*, float*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, float)` |
| attention | baseline | sm100 | `void qk_int8_sv_f8_attn_kernel_sm100<128u, 128u, 128u, 128u, (QuantGranularity)2, (QuantGranularity)2, __half, (MaskMode)1, false, true, false>(CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, float*, float*, float*, __half*, float*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, float)` |
| attention | baseline | sm120 | `void qk_int_sv_f8_attn_kernel<128u, 64u, 32u, 64u, 128u, (DataType)1, (QuantGranularity)3, (QuantGranularity)3, float, false, __half, (ComputeUnit)1, (MaskMode)0, false, true, false, false>(signed char*, signed char*, signed char*, __half*, float*, float*, float*, float*, float*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, float)` |
| attention | baseline | sm120 | `void qk_int_sv_f8_attn_kernel<128u, 64u, 32u, 64u, 128u, (DataType)1, (QuantGranularity)3, (QuantGranularity)3, float, false, __half, (ComputeUnit)1, (MaskMode)1, false, true, false, false>(signed char*, signed char*, signed char*, __half*, float*, float*, float*, float*, float*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, float)` |
| attention | baseline | sm89 | `void qk_int_sv_f8_attn_kernel<128u, 64u, 32u, 64u, 128u, (DataType)1, (QuantGranularity)3, (QuantGranularity)3, float, true, __half, (ComputeUnit)1, (MaskMode)0, false, true, false, true>(signed char*, signed char*, signed char*, __half*, float*, float*, float*, float*, float*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, float)` |
| attention | baseline | sm89 | `void qk_int_sv_f8_attn_kernel<128u, 64u, 32u, 64u, 128u, (DataType)1, (QuantGranularity)3, (QuantGranularity)3, float, true, __half, (ComputeUnit)1, (MaskMode)1, false, true, false, true>(signed char*, signed char*, signed char*, __half*, float*, float*, float*, float*, float*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, float)` |
| attention | baseline | sm90 | `void qk_int8_sv_f8_attn_kernel<64u, 128u, 128u, 128u, (QuantGranularity)3, (QuantGranularity)3, __half, (MaskMode)0, false, true>(CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, float*, float*, float*, __half*, float*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, float)` |
| attention | baseline | sm90 | `void qk_int8_sv_f8_attn_kernel<64u, 128u, 128u, 128u, (QuantGranularity)3, (QuantGranularity)3, __half, (MaskMode)1, false, true>(CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, float*, float*, float*, __half*, float*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, float)` |
| attention | new | sm100 | `void sage::sm100::qk_int8_sv_f8_attn_kernel_sm100<128u, 128u, 128u, 128u, (QuantGranularity)2, (QuantGranularity)2, __half, (MaskMode)0, false, true, false>(CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, float const*, float const*, float const*, __half*, float*, long, long, unsigned int, unsigned int, unsigned int, unsigned int, float)` |
| attention | new | sm100 | `void sage::sm100::qk_int8_sv_f8_attn_kernel_sm100<128u, 128u, 128u, 128u, (QuantGranularity)2, (QuantGranularity)2, __half, (MaskMode)1, false, true, false>(CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, float const*, float const*, float const*, __half*, float*, long, long, unsigned int, unsigned int, unsigned int, unsigned int, float)` |
| attention | new | sm120 | `void sage::sm120::qk_int_sv_f8_attn_kernel<128u, 64u, 32u, 64u, 128u, (DataType)1, (QuantGranularity)3, (QuantGranularity)3, float, false, __half, (ComputeUnit)1, (MaskMode)0, false, true, false, false>(signed char const*, signed char const*, signed char const*, __half*, float*, float const*, float const*, float const*, float const*, unsigned int, unsigned int, unsigned int, long, unsigned int, long, long, unsigned int, long, long, long, unsigned int, long, unsigned int, long, float)` |
| attention | new | sm120 | `void sage::sm120::qk_int_sv_f8_attn_kernel<128u, 64u, 32u, 64u, 128u, (DataType)1, (QuantGranularity)3, (QuantGranularity)3, float, false, __half, (ComputeUnit)1, (MaskMode)1, false, true, false, false>(signed char const*, signed char const*, signed char const*, __half*, float*, float const*, float const*, float const*, float const*, unsigned int, unsigned int, unsigned int, long, unsigned int, long, long, unsigned int, long, long, long, unsigned int, long, unsigned int, long, float)` |
| attention | new | sm89 | `void sage::sm89::qk_int_sv_f8_attn_kernel<128u, 64u, 32u, 64u, 128u, (DataType)1, (QuantGranularity)3, (QuantGranularity)3, float, true, __half, (ComputeUnit)1, (MaskMode)0, false, true, false, true>(signed char const*, signed char const*, signed char const*, __half*, float*, float const*, float const*, float const*, float const*, unsigned int, unsigned int, unsigned int, long, unsigned int, long, long, unsigned int, long, long, long, unsigned int, long, unsigned int, long, float)` |
| attention | new | sm89 | `void sage::sm89::qk_int_sv_f8_attn_kernel<128u, 64u, 32u, 64u, 128u, (DataType)1, (QuantGranularity)3, (QuantGranularity)3, float, true, __half, (ComputeUnit)1, (MaskMode)1, false, true, false, true>(signed char const*, signed char const*, signed char const*, __half*, float*, float const*, float const*, float const*, float const*, unsigned int, unsigned int, unsigned int, long, unsigned int, long, long, unsigned int, long, long, long, unsigned int, long, unsigned int, long, float)` |
| attention | new | sm90 | `void sage::sm90::qk_int8_sv_f8_attn_kernel<64u, 128u, 128u, 128u, (QuantGranularity)3, (QuantGranularity)3, __half, (MaskMode)0, false, true>(CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, float const*, float const*, float const*, __half*, float*, long, long, unsigned int, unsigned int, unsigned int, unsigned int, float)` |
| attention | new | sm90 | `void sage::sm90::qk_int8_sv_f8_attn_kernel<64u, 128u, 128u, 128u, (QuantGranularity)3, (QuantGranularity)3, __half, (MaskMode)1, false, true>(CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, float const*, float const*, float const*, __half*, float*, long, long, unsigned int, unsigned int, unsigned int, unsigned int, float)` |

---

## 附录 B 原始数据索引

采集产物不在本仓库里，由采集机汇总到一个 `kernel_breakdown/` 目录，结构 `{sm89,sm90,sm100,sm120,smoke}/`。SHA256 取前 8 位，用于核对手上的文件是否是本报告所依据的那一份。

| 目录 | 文件 | SHA256 前 8 | 字节 | 内容 |
|---|---|---|---|---|
| `sm89/` | `breakdown.md` | `228148e2` | 15397 | 合并报告 markdown：按 seq_len 聚合 + 每形状 total + 角色映射审计 |
| `sm89/` | `breakdown.csv` | `5fca1fb2` | 14088 | 合并报告 CSV：每形状 × 每角色两侧 µs 与 ratio |
| `sm89/` | `kb_baseline_sm89_20260829T181348Z.json` | `56b7c22e` | 165258 | baseline 侧原始采集，50 形状全量 |
| `sm89/` | `kb_new_sm89_20260829T181514Z.json` | `333c9b4e` | 152962 | new 侧原始采集，50 形状全量 |
| `sm89/` | `kb_baseline.log` | `673192ce` | 15436 | baseline 采集完整 stdout（meta JSON + 逐形状进度） |
| `sm89/` | `kb_new.log` | `e19094e0` | 15390 | new 采集完整 stdout |
| `sm89/` | `kb_cmake_configure.log` | `de3074ac` | 790 | new 侧 CMake configure，含目标 arch 判定 |
| `sm89/` | `kb_cmake_build.log` | `9d4cc9c7` | 329619 | new 侧完整编译日志 |
| `sm89/` | `prerun_45d4f11/kb_baseline_sm89_20260829T180801Z.json` | `2f2ccdb8` | 165159 | 重复性：45d4f11 版 harness 的 baseline 全量重跑 |
| `sm89/` | `prerun_45d4f11/kb_new_sm89_20260829T180928Z.json` | `d77d7a57` | 153065 | 重复性：45d4f11 版 harness 的 new 全量重跑 |
| `sm89/` | `prerun_45d4f11/breakdown.md` | `2b47b1a5` | 15397 | 上一轮的合并报告（与正式轮 attention 比值均为 0.9838） |
| `sm89/` | `prerun_45d4f11/breakdown.csv` | `e9139a3f` | 14088 | 上一轮的合并 CSV |
| `sm90/` | `breakdown.md` | `2d52f6b1` | 14593 | 合并报告 markdown：按 seq_len 聚合 + 每形状 total + 角色映射审计 |
| `sm90/` | `breakdown.csv` | `656c1294` | 13755 | 合并报告 CSV：每形状 × 每角色两侧 µs 与 ratio |
| `sm90/` | `kb_baseline_sm90_20260829T180453Z.json` | `c2ba947e` | 155494 | baseline 侧原始采集，50 形状全量 |
| `sm90/` | `kb_new_sm90_20260829T180529Z.json` | `41556ba3` | 144632 | new 侧原始采集，50 形状全量 |
| `sm90/` | `kbd_baseline.log` | `a726c990` | 15570 | baseline 采集完整 stdout |
| `sm90/` | `kbd_new.log` | `172e0639` | 15541 | new 采集完整 stdout |
| `sm90/` | `kbd_build.log` | `f67fb45c` | 47895 | new 侧 setup.py build_ext（内部 CMake + ninja）完整日志 |
| `sm90/` | `repeatability/kb_baseline_sm90_20260829T180647Z.json` | `67d5a9fb` | 13340 | baseline 侧原始采集，50 形状全量 |
| `sm90/` | `repeatability/kb_baseline_sm90_20260829T180706Z.json` | `70c96eaa` | 13332 | baseline 侧原始采集，50 形状全量 |
| `sm90/` | `repeatability/kb_baseline_sm90_20260829T180724Z.json` | `d26b0893` | 13297 | baseline 侧原始采集，50 形状全量 |
| `sm90/` | `repeatability/kb_new_sm90_20260829T180657Z.json` | `b2fa1baf` | 12768 | new 侧原始采集，50 形状全量 |
| `sm90/` | `repeatability/kb_new_sm90_20260829T180715Z.json` | `b57bbdba` | 12774 | new 侧原始采集，50 形状全量 |
| `sm90/` | `repeatability/kb_new_sm90_20260829T180734Z.json` | `fce61bd7` | 12780 | new 侧原始采集，50 形状全量 |
| `sm100/` | `breakdown.md` | `240e6f70` | 14697 | 合并报告 markdown：按 seq_len 聚合 + 每形状 total + 角色映射审计 |
| `sm100/` | `breakdown.csv` | `f93cf0f6` | 13806 | 合并报告 CSV：每形状 × 每角色两侧 µs 与 ratio |
| `sm100/` | `kb_baseline_sm100_20260829T182731Z.json` | `1b63cc7e` | 156215 | baseline 侧原始采集，50 形状全量 |
| `sm100/` | `kb_new_sm100_20260829T182757Z.json` | `28520339` | 145224 | new 侧原始采集，50 形状全量 |
| `sm100/` | `kbd_00_alloc.log` | `3919200d` | 1668 | Slurm 申请：QoS、分区、job 4015071 @ umb-b200-263 |
| `sm100/` | `kbd_01_recon.log` | `20ec13d8` | 1475 | 环境勘察：nvidia-smi / torch / nvcc / cmake / ninja 版本 |
| `sm100/` | `kbd_02_build.log` | `52763654` | 6078 | new 侧构建（tip 45d4f11，arch 10.0），含链接命令与耗时 |
| `sm100/` | `kbd_03_dump.log` | `4c6a77f6` | 11142 | 两侧 --dump-names 输出，角色映射预验证 |
| `sm100/` | `kbd_04_run.log` | `abe02eee` | 31735 | 两侧正式采集完整 stdout |
| `sm100/` | `kbd_05_report.log` | `03be1598` | 285 | --report 合并输出 |
| `sm120/` | `breakdown.md` | `37babb34` | 15393 | 合并报告 markdown：按 seq_len 聚合 + 每形状 total + 角色映射审计 |
| `sm120/` | `breakdown.csv` | `05d721af` | 14011 | 合并报告 CSV：每形状 × 每角色两侧 µs 与 ratio |
| `sm120/` | `kb_baseline_sm120_20260829T180043Z.json` | `cb71744c` | 165280 | baseline 侧原始采集，50 形状全量 |
| `sm120/` | `kb_new_sm120_20260829T180139Z.json` | `d94483d3` | 153297 | new 侧原始采集，50 形状全量 |
| `sm120/` | `baseline.log` | `a07e15fa` | 15484 | baseline 采集完整 stdout |
| `sm120/` | `new.log` | `f0bce9ee` | 15444 | new 采集完整 stdout |
| `smoke/` | `json/breakdown.md` | `e0c92fc4` | 9009 | sm86 冒烟合并报告（8 形状 --subset） |
| `smoke/` | `json/breakdown.csv` | `bf9361b3` | 2049 | sm86 冒烟 CSV |
| `smoke/` | `json/kb_baseline_sm80_20260829T174953Z.json` | `73f53be6` | 24160 | sm86 冒烟 baseline（commit 0a5d2e4，clean） |
| `smoke/` | `json/kb_new_sm80_20260829T174918Z.json` | `bb61026a` | 22959 | sm86 冒烟 new（commit 78988df，dirty） |
| `smoke/` | `oom/kb_new_sm80_20260829T175020Z.json` | `fb1c8c8f` | 9771 | OOM 路径验证：b2h56s131072 在 16 GB 卡上 OOM，其余形状继续 |
| `smoke/` | `oom/breakdown.md` | `7302dc4f` | 2895 | OOM 轮的合并报告 |
| `smoke/` | `oom/breakdown.csv` | `8f795dc4` | 662 | OOM 轮的 CSV |
| `smoke/` | `baseline_build.log` | `9c0cb3a2` | 144335 | 本机 baseline 编译日志（arch compute_86） |
| `smoke/` | `probe_names.py` | `1d9eb63a` | 1135 | 手写探针：按 qk_quant_gran 逐档打印 CUDA kernel 名，用于校准 ROLE_PATTERNS |
| `smoke/` | `probe_aten.py` | `58a7c9b0` | 785 | 手写探针：确认 ATen 侧 kernel 归属 |

此外还有两份体积较大的辅助产物未列入：`baseline_src/`（baseline 源码树快照，159 MB）与 `new_45d4f11.tar.gz`（new 侧源码归档，54 MB）。
