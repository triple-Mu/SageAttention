# sm100 varlen 支持设计(M1/M2 已落地;M3 上机:pytest 全绿、压测挂死,红)

基线 commit:5af2e06(master/feat/varlen 同点)。所有 file:line 以该点为准;M1 拆分后的新布局见 §3.1 末尾,M2 落地记录与 M3 上机清单见 §6.1/§6.2,M3 wave14 实录(含挂死画像与建议)见 §6.3,wave15 的 race 根因分析、已落地修复与 B200 裁决矩阵见 §6.4。

## 0. 结论

1. **先做旧 128 线程 kernel 的 varlen,WS kernel 悬置到 Phase B**。旧 kernel 的 8 处改动全部有 sm90 varlen 的逐行对应物;WS kernel 的 varlen 需要动 16-warp choreography 的 trip 计算,收益上限只有 dense 实测的 +1.07% geomean(`bench/FINAL_PASS_REPORT.md` L9),且仅命中 d128 长序列段。
2. **sm90 的 rank-4 tensor map batch=1 技巧可直接搬**,不需要 device-side tensormap(cuTensorMapReplace / tcgen05 时代的新设施)。sm90 与 sm100 共用同一个 host 侧 builder `create_tensor_map_4D`(csrc/tma.cuh:47-119)和同一条 `cp.async.bulk.tensor.4d` 指令(csrc/tcgen05.cuh:149 `using ::load_async_4D`;WS 的 u32 变体 csrc/qattn/qk_int_sv_f8_cuda_sm100_ws.cu:193-205),`make_qkv_tensor_maps_varlen`(csrc/qattn/launch_utils.cuh:571-603)对 CTA_Q=CTA_K=128 是模板参数,原样可用。
3. **quant 侧零新增 kernel**。`quant_qk_varlen` / `quant_v_fp8_varlen` / `segment_mean_varlen` 全部 arch 无关、几何参数化;sm100 需要的 (blk_q=128, warp_q=32) 已被 sm89/sm120 varlen 测试覆盖,(blk_k=128, warp_k=128) 已被 sm90 varlen 测试覆盖,V 的 linear 布局 + varlen 组合已有单测(test/test_varlen.py:343-406)。
4. **工作量:Phase A 约 4 个会话**(1 拆分+SASS gate、1-2 实现、1-2 B200 验证+bench);Phase B(WS varlen)另加 2-3 个会话,建议等 Phase A bench 数据说明长序列 varlen 是否真实存在后再立项。
5. 数值上 sm100 比 sm90 简单:sm100 不做 dequant scale 折进 sm_scale 的优化(逐元素乘,qk_int_sv_f8_cuda_sm100.cu:423),所以多 mask 一个 tile 不改变结果位,delta==0 的 dense 位等价(test_varlen.py:541-556 用 `torch.equal` 钉死)自动成立;sm90 为此专门做的 first_masked_tile 数值论证(qk_int_sv_f8_sm90_impl.cuh:138-151)在 sm100 只剩性能意义。

## 1. 现状与 fallback 代价

设计内拒绝在 plan.cpp:315-321:backend 解析到 `kSm100F8` 且 `varlen=true` 时直接置 `plan.error = "varlen is not supported by the sm100 backend"`。该行为被 test/test_varlen_utils.py:329-331(`test_varlen_rejects_sm100`)与 test/HARDWARE_CHECKLIST.md:99-103 钉死,B200 实机确认过报错可读(2026-08-29)。

B200 上 `sageattn_varlen` 今天落在哪,取决于两个开关:

| 条件 | 结果 | 代价 |
|---|---|---|
| `SAGEATTN_SM100_TCGEN05=1`(跑 sm100 的标准姿势) | plan 解析到 kSm100F8 → `ValueError`(sageattention/_plan.py:110-111) | varlen 完全不可用,dense 与 varlen 无法共存于同一进程配置 |
| 未设 TCGEN05,构建含 sm89 家族 PTX(如 `8.9+PTX`) | plan.cpp:297-299 forced_fallback → `kSm89F8`,走 sm89_varlen kernel(fwd_varlen_cuda.cu:172-194),默认 per_warp + fp32+fp16 | mma.sync 路径:无 TMA、无 tcgen05、PTX JIT 上 sm_100;仓内无 B200 varlen fallback 实测数字,量化留给 M4 bench |
| 未设 TCGEN05,构建无 sm89 PTX | `backend_serves` 检查报错(fwd_varlen_cuda.cu:97-104;plain cubin 8.9/12.x 与 cc 10.0 major 不合,plan.cpp:48-62) | varlen 完全不可用 |

即:B200 生产姿势(TCGEN05=1)下 varlen 是硬错误,没有静默 fallback。dense 参照系:sm100 old kernel 对 cudnn 是 0.738(FINAL_PASS_REPORT L9),sm89 kernel 在 B200 上的比值没有仓内记录。

## 2. 可移植资产:sm90 varlen 方案清单

sm100 复用的机制与出处,实施时逐条对照:

| 机制 | 出处 | sm100 适配点 |
|---|---|---|
| 闭式偏移(blk_offset/pad_offset/blk_total,两个 Property) | csrc/sageattn/varlen.h:56-101 | 直接用,kBlockQ=kBlockK=128 |
| SeqlenInfo(offset/seqlen/blk base/delta,签名 delta) | csrc/sageattn/seqlen_info.cuh:40-72 | `SeqlenInfo<true, 128, 128>` |
| rank-4 batch=1 tensor map(序列偏移走 token 坐标,cudagraph 安全) | launch_utils.cuh:564-603 | 模板参数 `<128, 128, HEAD_DIM>` 原样用 |
| varlen 布局解析(packed 3-D、scale 无 batch 维、padded_k) | launch_utils.cuh:362-521(kSVF8TMA 族) | 原样用 |
| scale 形状检查(blk_total 代数) | launch_utils.cuh:713-726 | Q_BLOCKS = blk_total(total_q,B,128)*4 |
| `#ifdef SAGE_VARLEN` 独立 TU + 独立 namespace(dense SASS 不动) | qk_int_sv_f8_sm90_impl.cuh:24-31;qk_int_sv_f8_cuda_sm90_varlen.cu:26-27 | 需先把 sm100 kernel 拆出 impl header(见 §3.1) |
| 空 grid 尾块提前退出 + 零 trip 零填充路径 | sm90_impl.cuh:123-127, 153-188 | 结构照搬,thread=row 使零填充更短 |
| bottom-right causal:签名 trip + first_masked_tile | sm90_impl.cuh:129-151 | 照搬;无 fold ⇒ 无数值约束(§3.3) |
| K 尾 tile 跨序列读 + mask 清除;V 靠 slab 永不跨序列 | sm90_impl.cuh:381-385, 479-484 | 照搬 |
| dead row(无可见 key 的行)强制 O=0 / lse=-inf | sm90_impl.cuh:616-650 | thread=row,可用 d_rcp=0 简化(§3.3 第 7 条) |
| launcher:grid.x=ceil(max_seqlen_q/CTA_Q)、padded_k 钉死、stride_batch_o=0 | qk_int_sv_f8_cuda_sm90_varlen.cu:88-150 | 克隆改名 |
| fwd_varlen 分发(per-backend case + pv 断言) | csrc/sageattn/fwd_varlen_cuda.cu:146-248 | 加 kSm100F8 case(pv=="fp32") |

TMA 对 packed 布局的约束只有两条,均已满足:batch stride 16 字节对齐(launch_utils.cuh:568-570 注释;packed q/k 强制 CHECK_CONTIGUOUS,launch_utils.cuh:382-383,stride_seq = heads*head_dim 字节,hd∈{64,128} 恒对齐)、各维 <2^32 / stride <2^40(tma.cuh:75-88 预检)。越界读由 TMA 填 0(int8),被 kv_len mask 或 V slab 零填充吸收。不需要逐序列 tensor map,也就不需要 device-side 修改 descriptor。

## 3. 旧 128 线程 kernel:varlen 路径

### 3.1 前置:impl header 拆分

sm100 kernel 体当前直接住在 csrc/qattn/qk_int_sv_f8_cuda_sm100.cu(不像 sm90 已拆 impl header)。varlen TU 复用 kernel 体的前提是先拆出 `qk_int_sv_f8_sm100_impl.cuh`(纯移动:kernel 模板 + 文件头注释,launcher 留原 TU),再新建 `qk_int_sv_f8_cuda_sm100_varlen.cu`:

```cpp
#define SAGE_VARLEN 1
#define SAGEATTN_ARCH_NS sm100_varlen   // 单 so ODR,同 sm90 模式
#include "qk_int_sv_f8_sm100_impl.cuh"
```

sm90 先例证明纯移动保 SASS 逐字节不变(sm90_impl.cuh:19-31)。注意 WS 文件头(_ws.cu:44-46)明确为保 SASS gate 而选择注释复刻不提取——那是针对跨 kernel 共享 helper 的决定,不否定同一 kernel 体的纯移动拆分;两条硬规则照 memory 执行:共享 body 加 `#ifdef` 时不提局部变量、gate 失败时按单 TU 二分。

`SAGE_SM100_DEVICE_ONLY`(ptxas probe 模式,qk_int_sv_f8_cuda_sm100.cu:66-99)随 kernel 体一起进 impl header,probe TU 不受影响。

**M1 已完成**(分支 wave10/sm100-varlen-m1,2026-08-30)。拆分后布局:

- `csrc/qattn/qk_int_sv_f8_sm100_impl.cuh`(新):文件头注释 + include 块(含 `SAGE_SM100_DEVICE_ONLY` 双支)+ `SAGEATTN_ARCH_NS` 默认 `sm100` + kernel 模板,全部自原 TU 逐字节移入;`SAGE_VARLEN` 只在注释里预留,kernel 体本 commit 零改动。
- `csrc/qattn/qk_int_sv_f8_cuda_sm100.cu`:只剩 host launcher 区(kPVFromSmem、SAGEATTN_SM100_WS 路由、fuse_v_scale launcher),include impl header;文件名不变,CMakeLists 零改动。
- `bench/sm100_review/qk_int_sv_f8_cuda_sm100_probe.cu`:include 改指 impl header(原先 include 整个 .cu)。

gate 实录:sm_100a+sm_110a 双 gencode 单 TU(同路径 build dir 全新重配)拆分前后 `cuobjdump -sass/-res-usage/-elf` 逐字节全同;test_ptxas_gate 4 例过;全 arch(8.6;8.9;9.0;10.0;12.0)构建绿。

### 3.2 kernel 内 8 处 `#ifdef SAGE_VARLEN` 改动点

对照 dense 行号(qk_int_sv_f8_cuda_sm100.cu)与 sm90 对应物:

| # | 位置(dense) | 改动 | sm90 对应 |
|---|---|---|---|
| 1 | :140-146 签名 | `qo_len/kv_len` 换成 `cu_seqlens_q/k + q_scale_stride_h + k_scale_stride_h + lse_stride_h`(独立 TU,dense 参数区不动) | impl.cuh:78-92 |
| 2 | :231-235 之后 | `SeqlenInfo<true,128,128>`;`cta_idx_q*128 >= qo_len` 整块退出(在 :297 barrier init、:308 TMA、:332 tmem_alloc 之前,块内 uniform) | :114-127 |
| 3 | :348-349 | trip 计算改签名 bottom-right:`kv_bound = causal ? min(kv_len, (cta+1)*128 + delta) : kv_len`,`num_iterations = div_ceil(int32)`;加 `first_masked_tile`(clamp 到 num_iterations-1);**并把这段提前到 :308 的 TMA 之前**,以支持第 4 条的零 trip 退出 | :129-151 |
| 4 | 同上 | `num_iterations <= 0` 时:每 thread 给自己的 row(`q_idx < qo_len` 时)写 head_dim 个零 + lse=-INFINITY,然后整块 return(仍在 barrier/TMA/TMEM 之前)。thread=row,无 sm90 那套 fragment 展开 | :153-188 |
| 5 | :312-314(prologue)、:389(K prefetch)、:547(V prefetch) | TMA 坐标:Q `(0, offset_q + cta*128, head, 0)`;K `(0, offset_k + iter*128, kv_head, 0)`;V `((blk_k_base + iter)*128, 0, kv_head, 0)`;batch 坐标恒 0 | :309-325, :381-388, :479-488 |
| 6 | :249-278 scale 索引 | 改为 `[heads, blocks]` 寻址:Q = `head_id*q_scale_stride_h + (blk_q_base + cta_idx_q)*per_cta + …`(per_warp per_cta=4、per_thread=32,行内项 row_id/32、row_id%8 不变);K base = `kv_head_id*k_scale_stride_h + blk_k_base*k_scale_per_cta`。dense 版依赖 `gridDim.x`(:254, :260)与 `div_ceil(kv_len,…)`(:267, :273),varlen 下 grid 开到 max_seqlen、块又有前置序列,两处都必须换参数 | :212-234 |
| 7 | :425-438 mask | 谓词换签名 shifted 行:`kv_idx > q_idx + delta \|\| kv_idx >= kv_len`(int32);触发条件从 `if constexpr (is_last)` 换成 varlen 侧 `if (iter >= first_masked_tile)`(runtime、块内 uniform;dense 分支保留原文本)。process_tile 的 prefetch 仍按 is_last 剥离 | :404-431, :534-549 |
| 8 | :561-601 epilogue | O 指针基址 `offset_q`、lse 改 `[heads, total_q]`:`head_id*lse_stride_h + offset_q + q_idx`;causal 下 dead row(`q_idx + delta < 0`)强制 `d_rcp = 0`(O 自然清零,无 NaN 风险:TMA OOB 填 0、V slab 零填充,全链路有限值)且 lse=-INFINITY | :616-650, :678-684, :729-733 |

改动点 3/7 的数值论证:sm100 的 dequant 是逐元素 `s = int2float(RS)*dequant_scale_j`(:423),不存在 sm90 的"未 mask tile 折 scale 进 sm_scale"分叉(sm90_impl.cuh:347-359),所以对一个无越界元素的 tile 跑 mask 谓词不改任何位。delta==0 时 first_masked_tile 恒等于 num_iterations-1,mask 元素集合与 dense is_last 完全一致,`torch.equal` 级 dense 等价自动成立。

### 3.3 host 侧与接线

| 项 | 内容 |
|---|---|
| launcher | 新 `sm100_varlen::qk_int8_sv_f8_accum_f32_fuse_v_scale_varlen_attn`,克隆 sm90 varlen launcher(qk_int_sv_f8_cuda_sm90_varlen.cu:37-159):CTA_Q=CTA_K=128、NUM_THREADS=128;padded_k 钉 `blk_total(total_k,B,128)*128`(:90-101 同款);scale 检查 `SAGEATTN_CHECK_QK_SCALE_SHAPES_VARLEN(blk_total(total_q,B,128)*4, blk_total(total_k,B,128))`;`make_qkv_tensor_maps_varlen<128,128,HD>`;grid `(div_ceil(max_seqlen_q,128), num_qo_heads, batch_size)`;stride_batch_o 传 0。**不接 SAGEATTN_SM100_WS 开关**(dense launcher :706-721 的路由 varlen 入口不复制,Phase A 无 WS varlen 可路由) |
| 头文件 | 新 `csrc/qattn/attn_cuda_sm100_varlen.h`(仿 attn_cuda_sm90_varlen.h) |
| plan.cpp | 删 :315-321 的拒绝块。smooth_v 降级(:346-349)、`v_pad_multiple==blk_k` 恒等式(fwd_cuda.cu:149-153,sm100 v_pad=128=blk_k,plan.cpp:211/236)均已就绪 |
| fwd_varlen_cuda.cu | include 加 SM100 分支(:40-51 处);switch 加 `kSm100F8` case,`TORCH_CHECK(plan.pv == PVAccum::kFp32)`(:146-248 同款) |
| CMakeLists.txt | `SAGE_BUILD_VARLEN` 下向 `SAGE_SRC_SM100` 追加 varlen TU(:137-139 sm90 同款;sm100 是 accel 组,gencode sm_100a+sm_110a 自动继承) |
| Python | sageattention/varlen.py:41 `_VARLEN_BACKENDS` 加 "sm100";docstring :182-198 改口径 |
| 测试更新 | test/test_varlen_utils.py:329-331 反转断言(或改为"不再拒绝");:344-345 的 sm100 豁免删除;test/HARDWARE_CHECKLIST.md:99-103、:673 行改状态 |

sm110(cc 11.0)随 plan.cpp:294 与 fatbin 的 sm_110a 条目自动获得同样支持,无额外工作。

## 4. WS kernel(C1,512 线程双 Q tile):varlen 路径

### 4.1 结构差异与改动点

WS 的 CTA 覆盖 256 行(2×128 tile),varlen 语义全部可表达,改动点(qk_int_sv_f8_cuda_sm100_ws.cu):

| # | 位置 | 改动 |
|---|---|---|
| 1 | :344-359 签名 | 同旧 kernel 第 1 条 |
| 2 | :492 `__syncthreads` 之前 | SeqlenInfo + 整块退出:`cta_idx_q*256 >= qo_len` 或 `trip_count(1) == 0` → 前 256 线程按 thread=row 零填 O(行有效时)+ lse=-INFINITY 后整 CTA return。必须在 barrier init(:456-473)与 tmem_alloc(:487-491)之前,512 线程一致 |
| 3 | :501-507 trip_count | 签名 bottom-right:`trip(tile) = div_ceil(max(0, min(kv_len, (2bx+tile+1)*128 + delta)), 128)`;再 `trip0 = max(trip0, 1)`(见 4.2) |
| 4 | :540-541 | qblk clamp 改按序列自己的块数:`min(2bx+tile, div_ceil(qo_len,128)-1)`,再加 `blk_q_base` 基址;Q/K scale 寻址同旧 kernel 第 6 条(:543-566 两处 `num_ctas_k` 换 k_scale_stride_h) |
| 5 | :638-654 is_oob | 谓词加 delta(signed);触发条件从 is_last 换 `iter >= first_masked_tile(tile)`(runtime;softmax_step 的 is_last 只管 mask,:602-655,无 prefetch 耦合,替换安全) |
| 6 | :1138-1175 load warp | TMA 坐标:Q0/Q1 加 offset_q,K 加 offset_k,V 换 `(blk_k_base + i)*128`,batch 坐标 0 |
| 7 | :864-913 correction epilog + :781-787 softmax lse | dead row(`q_idx + delta < 0`):epilog 里 `d_rcp = 0`,lse 写 -INFINITY;O/lse 寻址换 packed 形式(同旧 kernel 第 8 条) |
| 8 | :1192-1288 launcher | 克隆 + grid `(div_ceil(max_seqlen_q, 256), heads, batch)`;quant 块保持 128 行(scale 检查同旧 kernel varlen launcher) |

### 4.2 trip0==0 的处理:clamp,不改 choreography

bottom-right 下可能出现 `trip0 == 0 < trip1`(tile 0 整块 128 行都无可见 key,仅 causal 且 delta 足够负时)。mma warp 的 prologue 无条件做 QK00/PV00(:1022-1036),softmax0/correction 的 barrier 计数都假定 tile 0 至少一轮;按 tile 跳过会改 16-warp choreography,直接威胁 barrier_ledger.md 的死锁自由证明。

方案:**clamp `trip0 = max(trip0, 1)`,让 tile 0 跑一轮全 mask 的废轮**,dead-row 覆盖(第 7 条)把它的全部 128 行输出清洗掉。论证:

- choreography 完全不变:clamp 后仍满足 `trip0 >= 1` 且 `trip1 - trip0 ∈ {0, 1}`(两 tile 的 kv bound 差 128,ceil 差 ≤1;clamp 只在 trip1>=1 时把差从 1 收到 ≤1),现有 steady loop(:1042-1072)+ S1-only 轮(:1075-1093)+ tail(:1096-1106)照跑;
- 数值安全:全 mask 行的 P 元素是 `exp2(+8.807)≈448`(-5e6 sentinel 与 row_max 相消,只剩 S_FP8_OFFSET,正好顶到 e4m3 饱和值)而非 0,O0 是废值,但 trip0==0 ⇔ tile 0 的最后一行也无可见 key ⇔ 全 128 行都是 dead row,epilog 的 `d_rcp=0` 覆盖全 tile,废值不落地;lse 由 softmax 侧统一写 -INFINITY;
- 代价:仅该 CTA 一轮 KV tile 的废功,且只在极端 ragged causal 批出现;
- kv_len==0(trip1==0)不会走到这里,第 2 条整块退出已拦截。

废轮的 K scale 读块 0:trip1>=1 ⇒ kv_len>=1 ⇒ 序列至少有一个已写 scale 块,不越界。

### 4.3 判断:Phase B 缓做

- 收益上限:dense 实测 WS 只在 d128、非 causal >=16k / causal >=32k 段赢 0.6~2.7%,geomean +1.07%(FINAL_PASS_REPORT L9);varlen 工作负载以中短序列为主,命中概率更低;
- 风险不对称:choreography 相关改动的验证成本高(ws_stress.py SWEEP + 8000×2 定点是 sm100 通则,HANDOFF.md L22),寄存器预算(softmax 192 区间)加 SeqlenInfo/delta 计算后需重过 ptxas gate;
- 依赖:FINAL_PASS_REPORT L45 已注明 sm100 varlen 等 C1 定型,C1 的下一个 lever(ld 与首消费点间填独立工作)可能再动 softmax 结构。

Phase A 落地后 varlen 入口 plan 恒走旧 kernel,`SAGEATTN_SM100_WS` 只影响 dense,文档写明即可。

## 5. quant 侧现状(设计问题 2)

| 组件 | 现状 | sm100 需要的组合 | 覆盖证据 |
|---|---|---|---|
| `quant_qk_varlen`(csrc/sageattn/quant_cuda.cu:168-266) | arch 无关,blk/warp 全参数化,scale 输出 `[heads, blk_total 代数]`,per_warp/per_thread 双支 | blk_q=128, warp_q=32, blk_k=128, warp_k=128(plan.cpp:208-213),默认 gran=per_warp(plan.cpp:111-114) | Q 侧几何 = sm89/sm120 varlen(test_varlen_sm89/sm120.py 各 62 例已上机);K 侧 blk_k=128 双 gran = sm90 varlen(test_varlen_sm90.py:25 GEOM);DISPATCH_BLOCK_SIZE 本就含 64/128(csrc/dispatch_utils.h:86-98)。四元组本身在 test_varlen_sm100.py 里补 |
| smooth_k:`segment_mean_varlen`(:274-292)+ fuse_sub_mean varlen 实例(:239-242, :250-258) | chunked kernel,布局无关 | blk_k=128 的 fuse_sub_mean varlen 实例 | 编译已有(同 DISPATCH);dense 的 128 实例即 sm100 dense 日常路径,varlen 例进新测试文件 |
| `quant_v_fp8_varlen`(:390-474) | slab 按 pad_multiple 零填充(v_fp8 at::zeros,:421-426),fused/两段两路 bit 等价,融合门限 sm100=24576(:60-65) | v_layout="linear"(permute=False,:419)、pad_multiple=128、scale_max=448 | linear×varlen 与 pad=128 尾零已有单测(test/test_varlen.py:343-360, :366-406);128 slab 对齐 = sm90 已上机路径 |
| scale 布局对 CTA_Q=128/CTA_K=128 | launcher 检查即 blk_total 代数(launch_utils.cuh:713-726),kernel 侧 per_warp 4/块、per_thread 32/块(Q)与 1 或 4/块(K) | 与 dense sm100 kernel 的行内索引(row_id/32、(j%8)/2)完全同构,只换块基址 | §3.2 第 6 条 |

结论:quant/mean/V 链路一行 CUDA 不用写,Python 侧 `sageattn_varlen`(sageattention/varlen.py:240-299)按 plan 参数自动出正确几何。

## 6. 实施顺序、milestone 与 gate(设计问题 3)

顺序:**先旧 kernel(Phase A M1→M4),WS(Phase B)悬置**。

| 里程碑 | 内容 | gate | 场地 | 会话 |
|---|---|---|---|---|
| M1(已完成 2026-08-30) | impl header 拆分(纯移动) | dense sm100 TU 的 sm_100a/sm_110a SASS 逐字节等同(gensass 流程,memory「验证基建」);全 arch 构建绿 | 本机(nvcc 13.3 可编 sm_100a,无需硬件) | 1 |
| M2 | varlen TU(§3.2 八处)+ launcher/头/plan/dispatch/CMake/Python(§3.3)+ test_varlen_sm100.py 编写 | 本机全量编译;ptxas 无 spill 告警(test_ptxas_gate 模式);pytest 非 GPU 部分绿(plan 表、错误串) | 本机 | 1-2 |
| M3 | B200 correctness | `pytest test/test_varlen_sm100.py test/test_varlen.py -q` 全绿。test_varlen_sm100.py 套 sm90 文件的 gate 组(HARDWARE_CHECKLIST :103-108 口径):等长 batch 对 dense sm100 kernel `torch.equal`(golden 即 dense 同 kernel 体,无需新 golden dir)、ragged 非 causal 逐段对 dense、CAUSAL_RAGGED 四组 bottom-right、空 KV 段、dead row O==0/lse==-inf、opcheck、双 gran × 双 hd × causal | ComputeLab b200x4(--sqsh pytorch_26.07-py3,完毕 cancel) | 与 M4 合计 1-2 |
| M4 | bench + 文档 | `bench/bench_varlen.py`(packed vs padded dense,equal/ragged .25/.1 三 profile × 5 shape × causal)+ 同机 fallback 对照(TCGEN05 off → sm89 backend);口径按 BENCH_PROTOCOL(独占、双向交替、geomean>0.5%、方向一致);ncu 过滤 `-k regex:qk_int8_sv_f8_attn_kernel_sm100`;更新 HARDWARE_CHECKLIST/HANDOFF | 同 M3 分配 | — |
| Phase B(悬置) | WS varlen(§4)| 上述全部 + `ws_stress.py` SWEEP + 8000×2 定点 + ptxas 预算复核 + bench 证明命中段存在 | B200 | 2-3 |

Phase A 合计约 4 个会话(±1,取决于 M3 一次过与否)。

### 6.1 M2 落地记录(分支 wave12/sm100-varlen-m2,2026-08-31)

- kernel:§3.2 八处全部进 `qk_int_sv_f8_sm100_impl.cuh` 的 `#ifdef SAGE_VARLEN`
  分支,dense 分支原文本保留;`../sageattn/seqlen_info.cuh` 的 include 也套在
  `SAGE_VARLEN` 里(与 sm90 不同):无条件 include 时每个 kernel 自身的指令流
  与 ptxas 资源都不变,但 torch/cub 的 `EmptyKernel` 样板在 ELF 里的落点会移,
  违反本仓「整 dump 逐字节」的 gate 口径。与 sm90 的两处口径差:mask 触发条件
  是 `iter >= first_masked_tile`(runtime、块内 uniform,dense 的
  `if constexpr (is_last)` 只在 dense 分支保留);dead row 用 `d_rcp = 0` +
  `row_max = -INFINITY` 覆盖(thread=row,无 fragment 展开)。零 trip 退出的
  零填充是每 thread 一行 head_dim/2 个 uint32 store。
- host:launcher TU `qk_int_sv_f8_cuda_sm100_varlen.cu`(`sm100_varlen`
  命名空间,rank-4 batch=1 tensor map 走 `make_qkv_tensor_maps_varlen<128,128,HD>`,
  padded_k 钉 `blk_total*128`,不接 `SAGEATTN_SM100_WS`)+
  `attn_cuda_sm100_varlen.h` + CMake(`SAGE_BUILD_VARLEN` 下追加进
  `SAGE_SRC_SM100`,自动继承 sm_100a+sm_110a gencode)+
  fwd_varlen_cuda.cu 加 `kSm100F8` case(pv 钉 `"fp32"`)+ plan.cpp 删拒绝块 +
  varlen.py/conftest.py 的 backend 集合加 "sm100"。
- 测试:`test/test_varlen_sm100.py`(sm90 gate 组 + 四元组 quant 对拍;import
  时 `SAGEATTN_SM100_WS` setdefault "0",显式 WS=1/auto 则整文件 skip);
  test_varlen_utils.py 拒绝断言反转为 dense/varlen plan 等同;
  ptxas gate 新增 `qk_int_sv_f8_cuda_sm100_varlen_probe.cu`(4 实例)。
- 本机 gate 实录:dense TU 与 ws TU 的 `cuobjdump -sass/-res-usage`
  在 sm_100a+sm_110a 双 gencode 下改动前后逐字节全同(同路径 build dir
  全新重配,单 TU 对比);varlen TU 全量 CMake(`10.0;11.0` 与
  `8.6;10.0;11.0` 两种 arch 列表)编译链接绿;varlen probe 4 实例 ×
  sm_100a/sm_110a 全部 0 spill / 0 stack、254-255 reg(与 dense 同档),
  `test_ptxas_gate` 6/6;本机(sm_86)pytest 全套 542 passed / 347 skipped,
  test_varlen_sm100.py 69 例 collection 干净、全按 resolved backend skip。

### 6.2 M3 上机清单(B200,下一会话)

1. **构建**:ComputeLab b200x4(`--sqsh pytorch_26.07-py3`),全 arch 或至少
   `10.0;11.0`;并发乘积 ≤16;bdist 前清 `build/lib*`(memory 两条构建教训)。
2. **pytest**:`SAGEATTN_SM100_TCGEN05=1 pytest test/test_varlen_sm100.py
   test/test_varlen.py -q`(TCGEN05=1 是 resolve 到 sm100 的标准姿势,不设则
   两个文件按 resolved backend 全 skip)。不要显式设 `SAGEATTN_SM100_WS`
   (test_varlen_sm100.py 自己 setdefault "0";设了 1/auto 会整文件 skip)。
   test_varlen.py 的 API 组这次在 sm100 上 resolve 到 packed kernel,是
   `_VARLEN_BACKENDS` 加 "sm100" 后的首跑。
3. **等长 bitwise**:上面文件里的 test_fwd_varlen_equals_dense 即是 gate
   (dense classic kernel 同 kernel 体 `torch.equal`,无需新 golden dir);
   若 fail,先确认 dense 参照没被路由到 ws kernel(见 §6.1 的 env 说明)。
4. **dead-row 三层防线落地确认**(P=exp2(8.807)≈448 陷阱):
   CAUSAL_RAGGED 的 `([1000],[1])` 用例断言 dead row O==0 且 lse==-inf
   (防线一:epilogue `d_rcp=0`);空 KV 用例断言零 trip 提前退出路径
   (防线二);K 尾 tile 跨序列由 mask `kv_idx>=kv_len` 清除、V 走 slab
   (防线三)由 ragged 逐段对拍覆盖。
5. **压测**:`test_fwd_varlen_long_packed_tensor`(129536 token,64K 段)+
   cudagraph 换分段 replay(都在文件里);再顺跑一轮 `pytest test/ -q`
   看全套无回归。
6. **bench**:`bench/bench_varlen.py`(packed vs padded dense,equal/ragged
   .25/.1 三 profile × 5 shape × causal)+ 同机 fallback 对照(TCGEN05 off →
   sm89 backend);口径按 BENCH_PROTOCOL;ncu 过滤
   `-k regex:qk_int8_sv_f8_attn_kernel_sm100`。
7. **文档**:HARDWARE_CHECKLIST §1 的 M3 项打勾并填数字;HANDOFF 更新。

### 6.3 M3 上机实录(wave14,B200,tree a88057d;pytest 全绿,压测红)

口径:同 C1_DESIGN §10.4(umb-b200-262,JID 4028527,10.0a + PRUNE=OFF);
日志集群 `SageAttention_refactor/logs-w14/`。

| §6.2 项 | 结果 |
|---|---|
| pytest 三文件(TCGEN05=1,WS 不设) | **test_varlen_sm100 69/69 全过 0 skip**;test_varlen 89 passed / 97 skipped(skip 全是 pin 其他 backend);test_varlen_utils 49/49;合计 207 passed / 97 skipped / 12.4s |
| 等长 bitwise(`test_fwd_varlen_equals_dense`) | 全组合(双 gran × 双 hd × causal × 4 形状组)`torch.equal` 过 |
| dead-row 三层防线 | `bottom_right_causal[1000x1]` 四组过(防线一 d_rcp=0);`empty_kv_sequence` 两态过(防线二零 trip);ragged 逐段 `torch.equal` 过(防线三 mask+slab) |
| 长 packed / cudagraph | `long_packed_tensor`(129536 token,64K 段)causal 两态过;cudagraph 换分段 replay 过 |
| 全量回归 `pytest test/ -q` | **436 passed / 395 skipped,0 fail**(38s) |
| bench_varlen(等长 + .25-1x) | **红:挂死**(复跑一次在第 3 行 nc b8h16 n4096 等长处停;首跑 0 行——首跑距被杀的压测 wedged context 仅 5s,可能是连带) |
| ragged 压测(2000 次定点 driver,验收会话新增;不在 §6.2 清单) | **红:非确定性挂死**(详见下) |

**挂死画像(全部:GPU util 100% / ~247-248 W 低功耗自旋 = kernel 级
barrier 等待;driver/GPU 健康——同卡夹在中间的 dense ws_stress 16000+
launch 全绿)**:

| 复现试验 | 结果 |
|---|---|
| 压测 driver seed0 首跑 | iter=17 挂死(causal 等长 [7247,2865] b2 hd128) |
| `CUDA_LAUNCH_BLOCKING=1` 阶梯(同 seed 逐位重放 18 迭代 + 6 个 solo 变体:同形状 / smooth_k=0 / per_thread / 非 causal / b1 / 128 对齐) | **全部 clean → launch-blocking 掩蔽,是 race 不是形状 bug** |
| async 复跑 seed0 / seed1 / bench_varlen | **3/3 挂死**,位置各异(iter=225 causal b5 maxq4037;iter=53 causal b3 maxq7963;causal b8h16 n4096 等长)→ 非确定性,等长/ragged、空 KV 有无都中过,频率 ~每几十到几千次 varlen launch 一次 |
| 组件隔离:fwd_varlen-only(8 组预量化输入,纯 fwd 背靠背循环) | **挂死 ×2**(seed0 iter=1785、seed2 iter=45)→ 不需要 quant/segment_mean 在场 |
| 组件隔离:quant-only(segment_mean + quant_qk + quant_v 循环,不发 fwd) | 2000 次全绿 |
| dense 对照(同 pack 循环姿势,`SAGEATTN_SM100_WS=0` classic kernel,变形状 6000 次) | **全绿**(加上会话内 ws_stress 16000+ 次)→ 共享 impl body 的 dense 编译型不涉 |
| cuda-gdb attach(非 debug 进程) | 拿不到 resident kernel 名 |

**圈定**:race 在 **sm100 varlen fwd kernel(`SAGE_VARLEN` 分支 + varlen
launcher TU)自身**,quant 侧与 dense 编译型排除;triggering 条件是
async 背靠背 launch(每次 launch 后都有 `synchronize`,即单 launch 在飞,
串行化到 launch 级即消失)。**六次挂死全部落在 causal=1 迭代**(各 driver
两态交替,6/6 causal 的随机概率 1/64)→ runtime mask / 符号 trip /
差异化 trip 数是首要嫌疑面;其余差异只有 §3.2 八处(SeqlenInfo 读、K/V
坐标、block 表 scale 索引、零 trip 早退)加 launcher 的 rank-4 batch=1
tensor map。挂死形态是 util 100% / ~247 W 的 mbarrier 自旋。复现工具:集群
`SageAttention_refactor/scripts-w14/w14_isolate.py --component fwd`
(挂死概率单发 ~1/50-1/2000,跑 2 轮 6000 次内必中;stall 检测包装
`w14_21_isolate.sh`)。

**bench_varlen 部分数据**(挂死前 12 行,非 causal 全 10 行 + causal 2 行,
`logs-w14/repro/bench_vl.log`;dense 侧 d128 是 auto=WS 路):等长
0.71-0.86×(d128 0.71-0.80、d64 0.83-0.86),ragged .25-1x 0.99-1.37×——
显著低于 sm90/sm120 的记录(等长 0.93-1.01、.25-1x 1.35-1.68),d128 差距
一部分是 dense 参照换成了 WS kernel(+13-16%),d64 的 ~15% 等长差距是
varlen 路径自身开销;等 race 修复后按 M4 全量重测再定论。

**M3 判定:红**。建议二选一:(a) plan 侧把 sm100 撤出
`_VARLEN_BACKENDS`(退回 M2 前的 sm89 packed fallback),dense 侧 G2/G1
冻结不受影响;(b) 定位修复 race 后整轮重跑 M3(pytest 全绿态可沿用,
压测/bench 必须重来)。→ 走 (b),分析与修复见 §6.4。

### 6.4 race 根因分析与修复(wave15,本机 sm86 + nvcc 13.3 静态分析)

**结论:kernel 逻辑与 barrier 记账无 bug;首选根因候选 C1 = mbarrier init
后缺 async proxy 发布 fence(与 SS twin 同类的跨 proxy 可见性缺口),修复
已落地(varlen TU 单条 `fence.proxy.async.shared::cta`,dense SASS 逐字节
不变);C1 无法在本机拍死,§6.4.4 给出让一个 B200 会话一次裁决 C1/C2/C3
的实验矩阵。**

#### 6.4.1 排除表

逐候选排除,证据全部可本机复核:

| # | 候选 | 排除依据 |
|---|---|---|
| E1 | barrier 记账失衡(少发/多发 expect_tx、commit 或 wait) | §3.2 八处改动没有一处触碰 barrier 指令流:mask(第 7 条)只改 `s` 的值,trip 改动(第 3/4 条)只改 T 并保证 T>=1 才进 pipeline,坐标/索引/epilogue(5/6/8)全是地址算术。barrier_ledger.md 的计数表对 varlen 原样成立(把 T 读作该 CTA 自己的 T(cta)):每 barrier 每 tile 恰好一次 completion 一次 wait,T=1 退化列照走。且任何逻辑性失衡对固定形状是确定性的——wave14 同形状 `CUDA_LAUNCH_BLOCKING` 逐位重放 24 launch 全 clean,矛盾 |
| E2 | 数值链产生 Inf/NaN 引发挂死 | 全 mask 行走 P=exp2(8.807)≈448 的有限链(§6.2 第 4 条三层防线),denom>=1 恒有限;且数值错挂不了 mbarrier,只会错结果——69/69 pytest 含 bitwise 对拍全绿 |
| E3 | causal trip / first_masked_tile 符号错 | int32 负数除法逐段验过:kv_bound<=0 ⇒ div_ceil 截断后 num_iterations<=0 ⇒ 零 trip 早退;first_masked_tile 偏小只多跑逐元素谓词(幂等),偏大是数值错(E2 同理被测试排除)。谓词 `iter >= first_masked_tile` 块内 uniform,不产生分歧 |
| E4 | K 尾 tile / 跨序列 OOB 让 TMA 少记 tx | `cp.async.bulk.tensor` 对越界 box 仍按整 box 字节 complete_tx(OOB 行零填充,CUTLASS 全家依赖此语义);dense 自己的尾 tile(kv_len 非 128 倍数)走同一路径,16k+ 压测绿 |
| E5 | A2/A5 型 SASS 调度差(varlen TU 编译进禁区) | 本机把**生产实例**(per_warp、fuse_v_scale、no-lse、TS,causal/非 causal × 双 hd)双侧编到 sm_100a SASS 逐地标对照(TRYWAIT/ARRIVE/UTMALDG/UTCIMMA/UTCQMMA/UTCBAR/LDTM/STTM/LDG/BAR/FENCE 序列):骨架相似度 0.974-1.000,仅三处差异——varlen 序幕多 cu_seqlens LDG 与两个早退;peeled tile 的 k_scale LDG 从 wait(S_done) 前挪到后(**更保守**的方向,A2 是往前挪才挂);主循环 S 排空的 4 条 LDTM 发射间距 4 连发(dense)→ 2+1+1(varlen,**更少** outstanding,A5 是更多才挂)。无禁区形态 |
| E6 | TMEM 泄漏 / alloc-dealloc 失配 | 两个早退都在 alloc 之前;进了 pipeline 的 CTA 无一条路径绕过结尾 dealloc;alloc/dealloc 同一常量 256 列。泄漏还会让同 launch 后续 CTA 确定性卡 alloc,与 ~1/1000 非确定性矛盾 |
| E7 | 早退本身(块内非 uniform / 位置错) | 两个早退条件全由块 uniform 值构成,且都在 barrier init、TMA、tmem_alloc 之前(§3.2 第 2/4 条);CUTLASS 自家 Blackwell varlen FMHA(examples/77,persistent 调度)对空 tile 同样是"不 alloc 直接 continue/exit"的结构 |

E1-E7 合并的推论:挂死不在 kernel 的逻辑层,在硬件交互层——与 A2/A5 两桩
未定位前科同一空间(同一 kernel、同一形态:golden 绿 + 定点压测 ~1/400-1/2000
挂死;A5 停在第 1808/386 次 vs 本案 iter 1785/45,频率画像几乎重合)。

#### 6.4.2 候选根因(带论证)

**C1(首选,已修):mbarrier init 与 async proxy 之间缺发布 fence。**
`mbarrier.init` 是 generic proxy 写;而五个 barrier 里三个由 TMA 的
complete_tx、两个由 `tcgen05.commit` 的 arrive-on-retire 完成——全是
**async proxy** 对同一 smem 对象的访问。mbarrier 操作(arrive/expect_tx/
complete_tx/try_wait)彼此按 mbarrier 协议强序,但 **init 不是 mbarrier
操作**,是普通写:发布到 async proxy 需要显式
`fence.proxy.async.shared::cta`——CUDA Programming Guide 的 TMA 示例在
`init(&bar, ...)` 后紧跟 `cde::fence_proxy_async_shared_cta()`,CUTLASS 的
pipeline 构造在 init 后统一走 `fence_barrier_init()`。本 kernel(dense 与
varlen 同)只有 init → `__syncthreads()` → expect+TMA,`__syncthreads()`
不跨 proxy。SASS 佐证:ptxas 13.3 给 `mbarrier.init` 自动垫了 **init 前**
的 `FENCE.VIEW.ASYNC.S`(清掉槽位前任在同地址的 async proxy 残留状态),
**init 后**的发布方向没有任何 fence——工具链自己都认为这个对象跨 proxy
敏感,而我们缺的恰是文档要求的那半边。

时序窗口:init 的跨 proxy 传播 vs 第一次 complete_tx 的赛跑。窗口平时被
TMA 取数延迟(冷数据 ~1µs)盖死;varlen 压测是 8 组**固定输入**背靠背
launch(K/V tile L2 热,complete_tx 可 ~200ns 到达)+ 大量 µs 级生命期
CTA(block-skip 秒退、causal 1-tile trip)高频复用 SM 槽位——正好把窗口
压进可命中区。complete_tx 打在 init 未见的陈旧对象上 ⇒ tx 计数丢失 ⇒
全 CTA 永停在该 barrier 的 phase 0 wait ⇒ util 100% / 低功耗自旋,与
§6.3 画像逐条吻合。

与观测的对账:非确定性 ~1/50-1/2000 ✓(微架构窗口);同形状重放 clean ✓
(非形状函数);quant-only 绿 ✓(无 mbarrier);dense classic 6000 绿 ✓
(变形状 → L2 冷 + 无空 CTA 槽位churn);ws 16k 绿 ✓(init 到首 TMA 之间
隔着 512 线程序幕,天然余量大);causal 6/6 ✓(trip 减半 → 波次/槽位
churn 加倍、足迹减半 → L2 更热,是速率富集不是硬门控——这也兼容 bench
在非 causal 等长行停住的那次);pytest 绿 ✓(launch 数少,p~1/1000 命中
不了)。**本仓先例**:SS twin 的 sP staging 同为 generic 写喂 async proxy
读,缺同一条 fence,在 2 CTA 共驻时挂死,加 fence 后 8000×2 绿
(HARDWARE_CHECKLIST §5f)——同类缺口、同类形态、同款修法。

反对面(诚实记录):dense 同样缺这半边 fence 却 16k+ 绿,说明窗口极窄,
C1 的"varlen 时序才进窗"论证是速率级的,不是结构性必然;故 C1 必须由
B200 压测裁决,不能宣布结案。

**C2(候选):µs 级 CTA 槽位 churn 与 TMEM 分配器/CTA teardown 的硬件
交互。** varlen 是本仓第一个让大量 CTA 不触碰 tcgen05 就退出、且工作 CTA
生命期跨三个数量级的 tcgen05 workload;若硬件在"槽位回收 ↔ 邻居
UTCATOMSWS.FIND_AND_SET 重试 / barrier 流量"之间有未写明约束,形态同样是
永久自旋。反对面:CUTLASS varlen FMHA 结构上允许不 alloc 就退(见 E7,
虽然它是 persistent、churn 率低);且 causal 富集同样只能用速率解释。
若坐实,修法是"消灭裸退":`num_iterations = max(...,1)` + 两个早退改为
跑一轮全 mask 废 tile(§4.2 的 clamp 论证平移),需要两个补丁位——
q_scale 块索引按本序列块数 clamp(否则 block-skip CTA 的
`blk_q_base + cta_idx_q` 越界读 Q_scale),dead-row 覆盖条件扩到非 causal
的 `kv_len == 0`;代价是 ragged 批的空 CTA 从 µs 退出变成整轮废功,bench
必须重跑。**本轮不落这个补丁**(未证根因先不吃 perf 税)。

**C3(候选,已被 E5 削弱但未死):A2/A5 族未写明管线约束的 varlen 变体。**
地标级对照排除了粗粒度调度差,但 dual-issue 槽位、scoreboard 分配、代码
布局这类地标看不见的位形仍可能踩线。无本机手段,只能上机 A/B。

#### 6.4.3 已落地修复与本机 gate 实录

修复(本 commit):`qk_int_sv_f8_sm100_impl.cuh` 的 barrier init 块内、
`#ifdef SAGE_VARLEN` 下加一条 `tcgen05::fence_async_shared()`(thread 0,
init 五连之后、`__syncthreads()` 之前,与 Programming Guide 示例同位)。
一条指令覆盖全部五个 barrier(fence 语义按线程的全部先前 generic smem 写
生效)。SASS 里落为 `MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S`,与 SS twin
既有 fence 的降级完全一致。dense TU 不动:dense SASS 处于 G1/G2 冻结,
且其 16k 压测记录本身就是"dense 时序不进窗"的证据;若 B200 判 C1 成立,
dense/ws/sm90 三处同型潜伏缺口(均为 init 无发布 fence)另立 gate 周期
补齐。

| gate | 结果 |
|---|---|
| varlen probe 4 实例 × sm_100a | 0 spill / 0 stack,reg 255/254/255/255(与 M2 记录同档,未变) |
| varlen probe 4 实例 × sm_110a | 同上 |
| `test_ptxas_gate` | 6/6 过 |
| dense probe TU sm_100a | 改动前后 `cuobjdump -sass` 与 `-res-usage` 逐字节相同 |
| dense probe TU sm_110a | 同上(改动整体在 `#ifdef SAGE_VARLEN` 内,dense 预处理文本不变,字节相同是构造性保证,仍实测确认) |
| fence 落点 SASS 复核 | 四个 varlen 实例的 `SYNCS.EXCH.64 ×5` 与 `BAR.SYNC` 之间均出现 `MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S` |

#### 6.4.4 B200 一次裁决矩阵(w14_isolate.py 口径)

口径沿 §6.3:`w14_isolate.py --component fwd`,8 组预量化输入背靠背
launch,每 launch 后 synchronize,`w14_21_isolate.sh` 停滞检测;单发概率
~1/50-1/2000,每臂两轮 6000 次内必中(全绿臂跑满 2×6000 + 一轮 seed 扫)。

| 臂 | 构建/形状 | 判读 |
|---|---|---|
| A0 | fence 修复版,原 §6.3 复现姿势(含 seed0/seed2 定点) | 全绿 ⇒ C1 成立,进 M3 重跑(压测 + bench + `pytest test/ -q`);仍挂 ⇒ C1 不充分,进 A1-A3 |
| A1(裁 C2) | fence 版,causal,全序列等长 4096 × b8h16(**零** block-skip / 零 trip CTA,offset 非零) | 挂 ⇒ C2 出局(无裸退也挂),嫌疑回 C1 残余/C3;绿而 A2 挂 ⇒ C2 坐实 |
| A2(裁 C2) | fence 版,causal,极偏斜 ragged([7936,128,128,128] 类,max block-skip 占比) | 与 A1 对照读 |
| A3(裁 C3) | A0/A1/A2 都挂时:对 E5 仅存的两处调度差做源级 A/B(peeled tile 的 k_scale 读回挪到 wait 前 / `num_tiles_s` 循环 `#pragma unroll 1` 改 LDTM 间距),每臂 2×6000 | 哪臂转绿哪处即禁区,回填 A2/A5 那张约束表 |
| 随挂死附带 | 任何一次挂死时 `cuda-gdb -p` 抓 resident warp 的 PC(wave14 attach 失败的话换 `--ex "set cuda break_on_launch none"` 或 flush coredump 姿势),对照 gensass 产物定位自旋点 | PC 落 `SYNCS.PHASECHK...TRYWAIT`(tile 0 相位)⇒ C1 直接坐实;落 `UTCATOMSWS`/`NANOSLEEP` 重试环 ⇒ C2 直接坐实;落稳态 TRYWAIT(iter>0 相位寄存器非 0)⇒ C1 出局、C3 上位。一次成功抓取即可替代整张矩阵 |

判决后动作:C1 成立 → dense/ws/sm90 的同型 init fence 各起一个带 SASS
gate 的 follow-up;C2 成立 → 落 §6.4.2 的"消灭裸退"补丁(带 q_scale
clamp 与 kv_len==0 覆盖)重过 M3;C3 成立 → 按 A3 命中的位形改源并把
约束写进 HARDWARE_CHECKLIST 的 A2/A5 节。矩阵全绿超过 4×6000 而 A0 曾挂
⇒ 回到 §6.3 的建议 (a),plan 侧撤 sm100 等根因。

## 7. 风险(设计问题 4)

| 风险 | 内容 | 缓解 |
|---|---|---|
| bottom-right causal × 双 Q tile 的 trip 计算 | WS 两 tile trips 必须保持 `trip1-trip0 ∈ {0,1}` 且 ≥1,否则 16-warp choreography 死锁 | §4.2 clamp 方案不改 choreography,barrier_ledger.md 证明沿用;CAUSAL_RAGGED(test_varlen.py:632-637,含 delta=-999)+ 新 256 行粒度用例钉死;Phase A 不碰此风险 |
| 空序列 / 尾 tile × TMA OOB | K 尾 tile 越过序列尾读到下一序列 token(需 mask 清除)、越过 total_k(TMA 填 0);空 K 序列 trip=0 走不到 mask | 三层既有先例:mask 谓词含 `kv_idx>=kv_len`、V 走 slab 永不跨序列(varlen.h Property 1 + pad=blk_k=128,fwd_cuda.cu:149-153 恒等式)、零 trip 提前退出(§3.2 第 4 条);退出必须在 barrier init/TMA/tmem_alloc 之前且块内 uniform,WS 版还要在唯一一次 `__syncthreads`(:492)之前 |
| dead row 数值 | 全 mask 行的 P=exp2(+8.807)≈448 不为 0(sentinel 相消只剩 offset),不覆盖就输出 KV tile 的伪平均 | `d_rcp=0` + lse=-INFINITY 强制覆盖(§3.2 第 8 条);test_varlen.py:665-669 精确断言 O 全零、lse 恒 -inf;全链路有限值(TMA 填 0、slab 零填充),无 0×NaN |
| impl 拆分动 dense SASS | 提取即改 dense TU 文本 | M1 独立成会话,gensass 双 arch 逐字节 gate;`#ifdef` 不提局部变量、失败按单 TU 二分(memory 两条硬规则);varlen TU 本身不受 gate 约束(新 namespace 新 kernel) |
| grid 开到 max_seqlen 的空转 | ragged 批中大量 CTA 秒退,B200 148 SM 波次利用率下降 | 与 sm90/sm89 varlen 同构的既有代价,M4 ragged profile 直接量化;不在本设计内优化(persistent/动态调度另立项) |
| WS 寄存器预算 | softmax 192 区间加 SeqlenInfo/first_masked_tile 后可能 spill | Phase B 前置 ptxas 核对;预算恒等式 static_assert(:377-381)保总量 |
| 构建时长/OOM | sm100 fatbin 加第三个 TU(sm_100a+sm_110a) | 并发乘积 ≤16(memory「编译 OOM 教训」) |
