# sm89 家族非 causal attention 回退:定位与修复

结案对象:[KERNEL_BREAKDOWN_REPORT.md](KERNEL_BREAKDOWN_REPORT.md) 立案 3(sm89/sm120 非 causal attention 稳定慢 1.5-1.7%)。
修复 commit:`93d79e8`(`csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh`,+16/-1 行)。

一句话:回退不是 int64 stride 签名本身,而是三副本合并(`033df4c`)把 `k_scale_off` 的活跃区间拉长到横跨三个 `process_tile` 展开点,寄存器顶格 255 时 ptxas 改成在主循环内用 `S2R SR_TID.X` 重算 `lane_id % 4`,正好落在每轮 `dequant_scale` 的依赖链上;把该偏移量预折进 `K_scale_base_ptr`(仅对非 causal 且非 return_lse 的实例)后,L20 非 causal 46 形状合计从 0.9836 恢复到 0.9993,causal 优势与全部其他实例的 SASS 逐字节不动。

## 1 根因

### 1.1 嫌疑拆解

`033df4c`(A4-1)在一个 commit 里做了四件事:三副本合并成 `process_tile`、int64 batch/head stride、causal grid 反向(heavy-CTA-first)、prologue 多余 `wait_group` 门控。用同一条 nvcc 命令行分别编译 baseline(`0a5d2e4`)与 new(`b8145d2` 等价源)的实跑 TU
`sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu`(sm_89 SASS),按实例对照:

| 实例(L20 实跑,hd128、kPerThread、lse=F) | baseline | new | 差异 |
|---|---|---|---|
| 非 causal(MaskMode 0) | 4144 条 / spill 32 | 4120 条 / spill 32 | 主循环 889 → 892 条,**循环内多出 `S2R SR_TID.X` + `LOP3`** |
| causal(MaskMode 1) | 4536 条 / spill 86 | 4200 条 / spill 10 | new 明显更优 |

静态指令数几乎打平甚至更少,但非 causal 主循环内出现了 baseline 没有的 `S2R R154, SR_TID.X` → `LOP3.LUT R154, R154, 0x3` → `IMAD R2, R2, 0x4, R154` 链:这是 `k_scale_off = warp_idx*4 + lane_id%4` 的循环内重算,喂给每轮一次的 K scale `LDG`,而那次 load 的结果进 `dequant_scale`、再进 `sm_scale`,是主循环逐轮依赖链的一环。baseline 的写法(单个 `uint32 k_scale_idx`)里该值常驻寄存器。

机制:合并后 `k_scale_off` 在 kBulk/kMask/kLast 三个展开点都活跃,活跃区间覆盖整个函数;255 寄存器顶格下 ptxas 的 remat 启发式选择「省一个寄存器、每轮付一次 S2R」。S2R 是长延迟特殊寄存器读,~14 cycles/iter 的代价与实测 1.6% 吻合。

### 1.2 立案 3 原推测的修正

原报告推测「`unsigned int` → `long` stride 的签名改造」是嫌疑。本次定位否定了它:stride 只进 prologue 的基址计算,主循环内的 K/V 推进两侧同为指针递增;真凶是活跃区间变化触发的 remat。causal 反快也与 kernel 无关——ncu 显示三方 causal 的 `sm__cycles_active` 相同(6.99M),加速全部来自 heavy-CTA-first grid 排序。

## 2 修复

```cpp
// prologue,在首个 K_scale 读取之前
if constexpr (mask_mode != MaskMode::kCausal && !return_lse) {
    K_scale_base_ptr += k_scale_off;
    k_scale_off = 0;
}
```

指针算术恒等(`(p + off)[x] == p[off + x]`),FP 运算序列不变,数值 bit-exact。`k_scale_off = 0` 经前端常量传播后,非 causal 实例的循环内索引只剩 `next_iter * k_scale_advance_offset`,S2R 消失,主循环 890 条(重构前 889)。

条件里两个排除项都是实测出来的,不是猜的:

| 变体 | 非 causal(实跑) | causal(实跑) | 非 causal + lse |
|---|---|---|---|
| 无条件预折(E2) | 4096 条 ✓ | 4200 → 4424 条,spill 10 → 61 ✗(吃回一半优势) | — |
| 排除 causal(E6) | 4096 条 ✓ | +0 ✓ | 4280 → 4392,spill 39 → 90 ✗ |
| 再排除 return_lse(E7,落地版) | 4096 条 ✓ | +0 ✓ | +0 ✓ |

## 3 验证

### 3.1 SASS 手术边界(本地,CUDA 13.3)

全部 6 个受影响 TU × {sm_89, sm_120} 三方对照(baseline / new / e7):每个组合 32 个 attention 实例中 **28 个与 new 逐字节相同**,变化的 4 个全部是「MaskMode 0 + kPerThread + lse=F」目标实例(half/bf16 × hd64/hd128),且算术 opcode 直方图不变。causal、kPerWarp、return_lse、fuse_v_mean 实例零变化。varlen TU 的同类非 causal 实例同步变化(±8-24 条),数值由 golden equiv + pytest 覆盖。

### 3.2 L20(sm89,ComputeLab a1u1g-rome-0091,单卡独占,同 allocation 背靠背)

- golden:`ok=2004 diff=0`,equiv 345/345(varlen 与 dense 等价 case)。
- pytest:435 passed / 271 skipped / 0 failed。
- bench(50 形状 × 各侧 2 轮,`ratio = baseline / side`,>1 为该侧更快):

| seq_len | 形状数 | baseline/new | baseline/e7 |
|---|---|---|---|
| 1024 | 2 | 0.9927 | 1.0086 |
| 4096 | 12 | 0.9857 | 1.0025 |
| 8192 | 2 | 0.9836 | 0.9997 |
| 16384 | 3 | 0.9840 | 1.0006 |
| 32768 | 12 | 0.9837 | 0.9999 |
| 65536 | 12 | 0.9837 | 0.9996 |
| 131072 | 3 | 0.9835 | 0.9990 |
| **非 causal 合计** | 46 | **0.9836** | **0.9993** |
| causal 合计 | 4 | 1.0122 | 1.0123 |

逐形状最差 baseline/e7 = 0.999,轮间波动 ≤0.24%。门禁(非 causal ≥0.995、causal 不回退)通过。

- ncu(dense d128 s4096,`--set full`):

| 指标 | baseline | new | e7 |
|---|---|---|---|
| 非 causal `sm__cycles_active` | 13.231M | 13.444M(+1.61%) | 13.228M(恢复) |
| 非 causal warp latency/inst | 10.086 | 10.227 | 10.092 |
| causal `sm__cycles_active` | 6.993M | 6.992M | 6.995M(三方同) |

### 3.3 RTX PRO 6000(sm120,pro-5k,8 卡机 device 0)

- golden:`ok=2578 diff=0`,equiv 265/265。
- bench(同 50 形状 × 各侧 2 轮):非 causal 46 形状合计 baseline/new = 1.0169、baseline/e7 = 1.0176,逐形状最差 baseline/e7 = 1.0064;causal new/e7 = 0.9939。这台机器轮间波动中位数 1.21%、最大 3.9%,本轮 new 对 baseline 的方向与 8/29 那轮(0.9865)相反,说明 1-2% 级别信号在此机上贴近噪声;e7 与 new 的 causal 实例 SASS 逐字节相同,0.6% 差值只能是噪声。门禁按「baseline/e7 ≥ 0.995 且 causal SASS 不动」通过。

## 4 残留与后续

- **sm80 同模式**:`qk_int_sv_f16_sm80_impl.cuh` 用同样的 `K_scale_base_ptr + k_scale_off` 形式,sm_86 SASS 下 2 个非 causal 实例(hd128、f32 accum、非 inst_buf)主循环同样出现 S2R remat(860 → 869 条)。影响未上机量化,已另立任务。
- **return_lse 路径**:保持 new 现状(与本修复前逐字节相同),即 lse=T 的非 causal 实例仍带 S2R;该路径不在 `sageattn` 标准推理面上,如后续要救,需要单独调寄存器压力而不是套本预折。
- 本轮 pro-5k 的 new 树(`new-kbd`)与 harness 均为 b8145d2 形态;e7 = new-kbd + `93d79e8` patch。

## 5 产物

| 产物 | 路径 |
|---|---|
| 修复 | commit `93d79e8` |
| bench 原始 JSON(L20/PRO 6000 × baseline/new/e7 × 2 轮) | [kernel_breakdown_data/sm89_noncausal_fix/](kernel_breakdown_data/sm89_noncausal_fix/) |
| ncu raw CSV(三方 × causal/非 causal) | 同上,`ncu_e7round_*_raw.csv.gz` |
| L20 侧远端树与 profile | computelab `/home/scratch.sonlin_wwfo/workspace/nvidia/SageAttention_refactor/sm89/{e7,kbd_e7round,profiles}` |
| PRO 6000 侧远端树与数据 | pro-5k 容器 `/workspace/SageAttention-refactor/e7-kbd`、`/workspace/kbd_sm120/e7round` |
