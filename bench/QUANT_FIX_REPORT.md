# sm90/sm100 quant kernel 回退修复

2026-08-30。修复 `bench/KERNEL_BREAKDOWN_REPORT.md` 中记录的两项回退:H200 quant_k 慢 21%(`QuantPerThreadKInt8Kernel<128,128,·>`)、B200 quant_k 慢 1.8×(`QuantInt8Kernel<128,128,2,false,true>`)。

Commits(只改 `csrc/fused/quant_per_thread.cu` 与 `csrc/fused/fused.cu`):

- `05c33bd` quant: pin sm90/sm100 128-tile quant kernels back to 32 registers
- `a784344` quant: stop unrolling the reload form of the K kernel on sm90/sm100
- `378ef46` quant: re-convert the sub_mean image per use under the register pin

## 结论

根因是寄存器数跨过 32-register occupancy 悬崖,不是 int64 索引或 varlen 分支的执行成本本身。三个 commit 把目标实例 pin 回 32 registers 并消掉 pin 引出的 spill:

| 卡 | quant_k 修复前 | quant_k 修复后 | quant_q | golden |
|---|---|---|---|---|
| H200(sm90) | 0.775-0.861(合计 0.793) | 0.947-0.976(合计 0.967) | 1.15-1.23(保持领先) | 300/300 bit-exact |
| B200(sm100) | 0.55-0.59(~1.8× 慢) | 0.949-0.971(合计 0.952) | 0.932-0.942(未变,见残留) | 300/300 bit-exact |

ratio = baseline/new,>1 为 new 快。sm86/sm89/sm120 的 SASS 逐字节不变(修复宏对这三个 arch 预处理为空,本地 nvcc 13.3 cubin diff 验证),既有数字不受影响。

## 根因

`cuobjdump --dump-resource-usage` 静态对比(本地 nvcc 13.3,V13.3.73,与两台实卡工具链同版本):

| 实例 | baseline | 修复前 | 修复后 |
|---|---|---|---|
| `QuantPerThreadKInt8Kernel<128,128,±sub_mean>`(sm90,128-thread block) | REG 32 | REG 39-40 | REG 31-32,零 spill |
| `QuantInt8Kernel<128,128,2,false,true>`(sm100,1024-thread block) | REG 32 | REG 38-40 | REG 32,零 spill |

- sm90/sm100 每 SM 驻 2048 threads,满 occupancy 要求 regs ≤ 65536/2048 = 32。39 regs 让 H200 掉到 75% occupancy(对上 -21%);1024-thread block 从 2 CTA/SM 掉到 1(对上 B200 -1.8×)。
- sm86/89/120 每 SM 只驻 1536 threads,悬崖在 42 regs,39-40 regs 无损——这解释了「只有 sm90/sm100 的 128 档回退」:sm86 的 1.63× 提速来自 warp_k=64 档的寄存器缓存,L20/PRO6000 持平。
- 多出的 6-8 regs 来自 int64 stride 的地址计算:baseline 的 uint32 stride 允许 batch/head 偏移全走 uniform datapath(UIMAD,不占 vector 寄存器);int64 强制 `IMAD.WIDE.U32` 走 vector datapath(修复前 SASS 多 11 条 IMAD.WIDE、8 条 LDC.64)。quant_q 的实例恰好没超 32,所以 sm90 quant_q 反而快 18%(单遍读缓存)——三个嫌疑(寄存器缓存翻倍 / int64 / varlen 分支)中,前者被 row cap 排除,后两者只通过寄存器压力起效,直接执行成本很小。

## 修复方式与踩坑

全部用 `#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000)` 包住,其他 arch 源码级不变:

1. **`__launch_bounds__(1024, threads==1024 ? 2 : 1)`** pin 回 32 regs。踩坑记录:`(1024,1)` 不等价无标注(显式 maxntid 把 reg cap 从 255 压到 64);maxThreads 写小值(如 128)会放松其他实例的预算,它们 REG 乱涨;`__maxnreg__(255)` 也不等价无标注。因此 900/1000 上 warp_k=64 的实例会被 `(1024,1)` 臂变形——可接受,`plan.cpp fill_tiles` 在这两个 arch 固定 warp_k=128,64 档无调用路径。
2. **reload 形态不展开**(`kUnroll` 4→1):32-reg 预算下展开只买 spill(8B)与 I-cache 压力(SASS 912→336 条),不买 ILP。H200 实测不展开再快 1-3%;`kUnroll=2` 变体实测最差(0.83-0.91),已排除。
3. **sub_mean 的 fp32 mean 预转换数组改为 use 处重转换**:8 个长活跃寄存器正是 pin 后 16B spill 的来源;fp16→fp32 转换精确,数值不变。B200 quant_k 从 0.878-0.894(仅 1+2)提升到 0.949-0.971。

## 验证

- **golden bit-exact**:`tools/compare_reference.py --section quant --section e2e`,300 cases,H200 与 B200 均 `ok=300 diff=0`(golden 由修复前 45d4f11 build dump,check 修复后 build)。
- **quant bench**:`bench/kernel_breakdown.py` 50-shape 矩阵,baseline(a07d7fc)与 fixed 同卡背靠背。H200:GPU 5,3 轮(r1/r2 为 commit ×2,r3 为全部 3 commits;quant_k 走 per-thread 路径,第 3 个 commit 不影响该路径,3 轮可合并看重复性);B200:umb-b200-260,pytorch 26.07 容器(CUDA 13.3),2 轮全 3 commits。分档明细见 `kernel_breakdown_data/quantfix_20260830/*_breakdown.csv`。
- **SASS 不变性**:sm86/89/120 修复前后整 TU cubin 逐字节一致(仅匿名 namespace 哈希不同);900/1000 上被调度实例(warp_k=128、1024-thread)全部 REG ≤ 32、零 spill。
- **e2e**:H200 TOTAL 全档 0.996-1.409,修复不伤整体。

## 残留(超出本修复范围)

- B200 quant_q 0.932-0.942:该实例 REG 本来就是 32、无 spill,这 6-7% 是 varlen 支持的结构性指令开销(哨兵分支 + int64 头部),修复前后一致,quant_k 修复后已反超它。要消掉需要 dense/varlen 双实例分裂(会动全 arch 的 mangled name 与 SASS),建议单独立项。
- 个别档位在 0.947-0.950 边缘波动(H200 4096 档三轮 0.947/0.949/0.960,B200 131k 档 0.949/0.950),轮间波动 ±0.5-1pp,与 0.95 线的差距在噪声内。

## 原始数据

`bench/kernel_breakdown_data/quantfix_20260830/`:每轮 baseline/new 两个 JSON(gzip,含 GPU UUID、driver、commit、dirty 标记)+ per-shape CSV + 两卡 golden check log。
