# E1/P3 前筛与 Track D 静态审计(wave1/prescreen)

| 项 | 结论 | 一句话依据 |
|---|---|---|
| E1(f16x2 softmax) | **no-go** | `ex2.approx.f16x2` 在 sm86-sm100 一律拆成 2 条 `MUFU.EX2.F16`+1 条 PRMT,element 吞吐与 f32 完全打平(15.99 vs 15.99 elem/SM/clk),还要多付 pack;数值不是瓶颈 |
| P3(sm100 packed f32x2 softmax) | **go** | `fma/add/mul.rn.f32x2` 在 sm_100a 落成单条 FFMA2/FADD2/FMUL2;相邻可并对面积 13.3-19.9%,合并后省 6.6-10.0% 静态指令,远超 4% 判据 |
| sm100 二选一 | **选 P3** | E1 机制在 ISA 层面就没有收益来源;P3 机制已验证、面积更大、数值零漂移 |
| Track D 裁剪规则 | 默认 resolve 路径下两条规则成立;反例集中在显式 backend/raw op 逃生门 | 详见下文,DRAFT patch 已落 `SAGE_PRUNE_GENCODE`(默认 OFF,commit b06d756,不合入) |
| P2 静态 baseline | quant kernel 哨兵块 13-60 条指令/kernel(占 0.6-28.1%),dense 热路径只付 2 ISETP+1 BRA | `prescreen_data/so_audit_sm86.txt` |

数据产物全部在 `bench/microbench/prescreen_data/`,probe 源码在 `bench/microbench/`。

---

## 一、E1 前筛:f16x2 softmax

设想:P 马上量化到 e4m3(3 位尾数),exp2 用 `ex2.approx.f16x2` 一条算两个,砍半 MUFU。
三条证据链都齐了,机制这条直接判死。

### 1. MUFU 吞吐(实测 sm86 + 四 arch SASS)

`ex2_rate.cu`,方法仿 `mma_rate.cu`(8 条独立依赖链纯寄存器循环,cudaEvent 计时,结果喂进不可能成立的比较防 DCE)。RTX 3080 Ti Laptop(sm_86,58 SM),跑前 `nvidia-smi --query-compute-apps` 确认 0 个 compute 进程。原始输出:`prescreen_data/ex2_rate_sm86.txt`。

| kernel | instr/SM/clk | elem/SM/clk | 说明 |
|---|---|---|---|
| ffma_f32 | 122.34 | 122.34 | FMA pipe 参照(名义 128,差值是 boost 漂移) |
| ex2_f32 | 15.99 | 15.99 | 现状 XU pipe,名义 16 |
| ex2_f16x2 | 7.99 | **15.99** | 每条 PTX 拆 2 条 SASS MUFU,element 吞吐与 f32 打平 |
| cvt_f16x2(f32→f16x2 pack) | 62.86 | 125.72 | E1 需要额外插入的 pack,本身不是瓶颈但纯属新增 |

SASS lowering(sm_86/sm_89/sm_90a/sm_100a 四份 cubin 均一致):一条 `ex2.approx.f16x2` = 2× `MUFU.EX2.F16`(逐半算)+ 1× PRMT 重打包。**这些 arch 上不存在 packed f16 MUFU,element 吞吐打平是 ISA 结构性的**,不是调参问题。sm89/90/100 没实机没跑速率,是本次的数据缺口;但要翻案需要 `MUFU.EX2.F16` 是 f32 的双倍速率,与 2 条 SASS 的拆法矛盾,风险极低。

### 2. 指令面积(判据 >=4%,全部通过——但机制不成立,面积白给)

device-only probe TU(`area_probe_sm89.cu` / `area_probe_sm90.cu` / 复用 `bench/sm100_review` 的 sm100 probe)+ `sass_area.py` 静态统计,生产 flags(-O3 --use_fast_math)。XU 链口径 = MUFU.EX2 + 喂它的 FFMA(def-use 回溯)+ FMAX。完整表:`prescreen_data/sass_area_results.txt`。

| arch(生产实例) | kernel 总指令 | XU 链面积 |
|---|---|---|
| sm89 hd128 per-thread ×3 实例 | 3922-4185 | 14.91-15.91% |
| sm90 hd128 per-thread ×2 实例 | 1802-1874 | 21.34-22.20% |
| sm100 hd64/128 ×4 实例 | 2977-3654 | 17.62-21.63% |

(静态计数含 prologue;XU 链全部住在主循环里,动态占比只会更高。)

### 3. 数值仿真(不是瓶颈,留作翻案素材)

`e1_fp16_exp2_sim.py`:S(int32)→float→fmaf(scale,-offset)→exp2(fp32 vs fp16)→e4m3→PV fp32 累加,row_max/denominator 保持 fp32。随机 + 对抗(spiky rowmax、uniform logits、outlier channels)。输出:`prescreen_data/e1_sim_results.txt`。

- 逐元素:arg∈[-30, 8.807] 内仅 0.26% 的点 e4m3 结果被 fp16 双重舍入改写,错者中位数 1 个 e4m3 ulp(~9%)。
- 端到端:cos(e1,ref) 与 cos(base,ref) 到小数点后 6-7 位一致(误差被 e4m3 量化本身淹没);e1↔base 漂移 cos>=0.99997,rel_l1<=0.66%,最差单行 cos 0.9984(spiky rowmax)。

**E1 结论:no-go。** 若未来某 arch 出现真 packed f16 MUFU,数值这关已经预先验过,可直接复用仿真脚本重开。

---

## 二、P3 前筛:sm100 packed f32x2 softmax

机制:PTX `add/mul/fma.rn.f32x2`(sm_100a)单条 SASS `FADD2/FMUL2/FFMA2`,已用 probe 实证(ptxas 13.3);**`max.f32x2`/`min.f32x2` 不存在**(ptxas 拒绝),row max 的 FMNMX3 不可 pack,exp2 仍是逐个 f32 MUFU——P3 减的是 FMA pipe/issue 压力,XU 不动。

sm100 四个生产实例(同一 `sass_area_results.txt`),口径:相邻、同 opcode、无依赖的 FFMA/FADD/FMUL 贪心配对(ptxas 调度序,保守下界):

| 实例 | 可并对指令 | 占 kernel | 合并后净省 |
|---|---|---|---|
| hd64 per-thread causal+lse | 430/976 | 13.26% | 6.63% |
| hd64 per-warp 非 causal | 510/1033 | 17.13% | 8.57% |
| hd128 per-warp 非 causal | 628/1161 | 19.13% | 9.56% |
| hd128 per-thread causal+lse | 728/1296 | 19.92% | 9.96% |

上界:f32 ALU(FFMA+FMUL+FADD)合计占 kernel 30-36%,softmax 主循环逐元素独立(除 d_sum 串行链),源码级 float2 重写理论上可逼近砍半(≈15-18% 静态指令)。判据 >=4%:**保守口径已 1.7-2.5 倍超线,go。**

实现风险(立项时要背):f32x2 要求操作数落在对齐寄存器对上,寄存器分配压力上升;d_sum 归约链要拆双部分和;数值语义逐 lane 仍是 IEEE fp32,输出零漂移。

**sm100 上 E1 vs P3:选 P3。** E1 在该 arch 连收益来源都没有(同一 2×MUFU 拆法);P3 机制实证、面积更大、无数值代价。

---

## 三、Track D 静态审计

### 1. resolve 选择表(plan.cpp:206-325 × SageArch.cmake:38-63)

默认路径(无显式 backend)下每个 cc 选中的 backend 家族,以及该 cc 现在被编进哪些组:

| 设备 cc | resolve 默认 | 依赖的组/SASS | 备注 |
|---|---|---|---|
| 8.0/8.6/8.7/8.8 | kSm80F16 | sm80 组同 major 低版 cubin | |
| 8.9 | kSm89F8 | sm89 组 8.9 | sm89 组未编时退 kSm80F16 |
| 9.0 | kSm90F8 | sm90 组 90a | sm90 组未编时退 kSm80F16 |
| 9.x(minor>0,假想) | **kSm80F16(无条件)** | 只有 sm80 组的 plain 9.0 cubin 能跑(90a 只认 9.0) | 「sm80 必须保 9.0」的硬依据 |
| 10.0/11.0 | 默认 **kSm89F8**(tcgen05 是 env opt-in,默认关) | sm89 组 10.0/11.0 是生产主路径,不是备胎 | opt-in 后才走 kSm100F8 |
| 10.3 等 10/11 其他 minor | kSm89F8 | sm89 组同 major 低版 cubin | |
| 12.x | kSm120F8 | sm120 组 | sm120 未编→退 kSm89F8(靠 sm89 组 12.x cubin)→再退 kSm80F16 |

### 2. 两条裁剪规则复核

**「sm89 组裁 major==12」:默认路径成立。** 12.x 一旦被请求,sm120 组必然入组(major==12 即成员),resolve 在 12.x 上无条件先选 kSm120F8;kSm89F8 只在 `!SAGEATTN_BUILD_SM120` 时可达,同一份 arch 列表下不可能。
反例(逃生门):`plan(backend="sm89")` 与 raw op `qattn_sm89_*`(ops.cpp:487-490,`sageattention/ops.py` 公开导出)在 12.x 设备上直接点名 sm89 家族;`backend_compiled()` 只查家族编译开关(SAGEATTN_BUILD_SM89),没有逐 arch 粒度,裁掉后这类调用从"能跑"变成 launch 时 `cudaErrorNoKernelImageForDevice`。

**「sm80 组裁 8.9/10.x/12.x、必须保 9.0」:保 9.0 成立且必要;裁 8.9 必须带条件;裁 10.x/12.x 默认路径成立。**
- 保 9.0:上表 9.x-minor 行,kSm80F16 是无条件终点,而 sm_90a SASS 只能在 9.0 上跑——sm80 组的 plain 9.0 cubin 是唯一能服务 9.x 的东西。
- 裁 8.9 的反例:`TORCH_CUDA_ARCH_LIST=8.9` 单 arch 构建(4090 用户常态)下无条件裁 8.9 会把 sm80 组裁空,fp16 PV 路径(kSm80F16 是唯一支持 pv fp16 的家族)在该构建里整体消失。安全形式是条件裁:仅当同列表存在更低的 8.x(同 major 二进制兼容,低版 cubin 顶上)才裁。
- 裁 10.x/12.x:默认路径 kSm80F16 在这些设备上只在 sm89 组缺席时可达,而任何 >=10 的请求都会让该 cc 入 sm89 组,不可达成立。反例同上:显式 kSm80F16/raw `qattn_sm80_*`(fp16 PV on Blackwell)从"能跑"变 no-kernel-image;若裁空整组则退化为干净的 "not in this build" 报错(反而更好)。
- 附带损失:被裁条目若带 +PTX,该组同时失去这份 PTX 的向前兼容 JIT。

### 3. DRAFT patch

`cmake/SageArch.cmake` 新增 `SAGE_PRUNE_GENCODE`(默认 OFF,**DRAFT 不合入**,commit b06d756):OFF 时成员表逐字节不变;ON 时按上述条件规则裁。已用提取真实代码块的 cmake -P 自检对照过 5 组 arch 列表(全量/单 8.9/8.0+8.9/单 12.0/9.0+10.0),OFF 全等、ON 全部命中预期(9.0 恒保留、孤儿 8.9 恒保留)。全量 arch 列表下收益:sm80 组 9→4 个 gencode 目标 × 2 TU,sm89 组 5→3 × 5 TU,合计少 20 次 ptxas 编译及对应 fatbin 体积(全 arch wheel 未实测,留待批 1 落地时量)。
落地前建议补的一件事:resolve() 对显式 backend 请求改查逐 arch 列表(`compiled_archs()` 已存在),把 no-kernel-image 变成干净报错。

### 4. `_C.abi3.so` 审计(P2 静态 baseline,sm_86 cubin,主 checkout 只读)

工具:`so_param_audit.py`,输出 `prescreen_data/so_audit_sm86.txt`。

**Dense attention kernel 无 varlen 残留**(抽 3 对同模板参数的 dense/varlen 实例):
- param 布局:dense 24 参数 160B,varlen 27 参数 180B,差值恰为 +2×8B cu_seqlens 指针 +3×4B stride −2×4B qo/kv_len = +20B;dense 侧 KPARAM_INFO 无任何 cu_seqlens 槽位(mangled 签名亦无 `PKi`)。
- SASS:dense LDG 普查 `LDG.E.CONSTANT`=4(scale 类),varlen=8,多出的 4 条正是 cu_seqlens_q/k 各两端的前缀和加载;LDGSTS(Q/K/V 数据面)两侧同为 44。SeqlenInfo<false> 全折叠成立。

**Quant/fused kernel 哨兵分支**(`cu_seqlens != nullptr`,自动定位 8B 参数判空 + `@!P BRA` 跳段):

| kernel 家族 | 总指令 | varlen-only 块 | 占比 |
|---|---|---|---|
| QuantPerThreadQInt8Kernel ×2 实例 | 256/392 | 57/59 | 22.3%/15.1% |
| QuantPerThreadKInt8Kernel ×2 | 776/816 | 57/59 | 7.3%/7.2% |
| QuantInt8Kernel ×2 | 264/296 | 60/59 | 22.7%/19.9% |
| TransposeQuantFp8Kernel ×2 | 2216/2224 | 13/13 | 0.6% |
| TransposePadPermuteKernel ×2 | 128/136 | 36/36 | 28.1%/26.5% |
| MeanScaleKernel ×2 | 584/712 | 34/34 | 5.8%/4.8% |
| SubMeanKernel | 80/64 | 无判空(本无 varlen 路径,符合预期) | — |

块内大头是 `blk_offset` 的两次运行时 u32 除法展开。dense 热路径动态只付 2 ISETP + 1 BRA(每线程一次),varlen 块对 dense 是纯 icache/体积占用——P2 若拆分 dense/varlen 实例,静态收益以此表为对照。

---

## 复现

- 速率:`nvcc -O3 -std=c++17 -arch=sm_86 ex2_rate.cu && ./a.out`(sm_89/90a/100a 用 `-cubin` 仅验指令合法)
- 面积:`bash build_area_probes.sh`(需 torch 头文件)→ `python sass_area.py --pairs <cubin...>`
- 数值:`python e1_fp16_exp2_sim.py`
- .so 审计:`python so_param_audit.py <path>/_C.abi3.so`
