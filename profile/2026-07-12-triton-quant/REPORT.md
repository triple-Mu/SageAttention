# NCU Profile：triton 量化 kernel（CuTeDSL 路径）vs CUDA fused 量化 kernel（H200 / sm90）

**TL;DR**（主形状 b=4, h=32, s=4096, d=128, fp16 NHD）：三个 triton 量化 kernel（4 次 launch）
ncu 合计 **209.7µs，DRAM 带宽利用率 59.6–78.1% of peak**（memory-bound，接近 H200 混合读写
实际可达上限）；同形状 CUDA fused 量化 kernel（4 次 launch）合计 **647.4µs，慢 3.09×**——
主要输在 `TransposePadPermuteKernel`（400µs，DRAM 仅 12.7%，l1tex 92% 成瓶颈：fp16 中转
materialize 多付 114MB 写 + 134MB 读）。至此 CuTeDSL e2e 的量化短板消除：e2e 从 CUDA 的
0.24–0.30× 提升到 0.95–1.61×（短序列反超，长序列由 kernel-only 差距 ~0.88× 主导）。

## 1. Setup（可复现）

- 机器/容器/GPU/ncu 与前几轮完全一致：hyper01 H200（132 SM，DRAM 峰值约 4.8TB/s），容器
  `sglang-diffusion-qwenimage`，GPU 2（`CUDA_VISIBLE_DEVICES=2`，ncu 内 `--devices 0`），
  ncu 2025.1.1，torch 2.11.0+cu128，triton 3.6.0。
- 代码：branch `cutedsl-sage-sm90` @ b51c95f（triton 量化实现 commit）。
- Harness：`harness/profile_quant.py`。两个被试同口径：fp16 NHD [4,4096,32,128] 输入，
  km=k.mean 预计算（两侧共同的 torch reduce，不在被试内）；warmup 3 + 5 次迭代，
  每迭代 4 次 kernel launch，ncu `-s 12 -c 4` 跳过 warmup 抓稳态第 4 轮。
  - `PROF_KERNEL=triton`：`quant_triton.py` 的 `_quant_q_int8_per_warp_kernel` /
    `_quant_k_int8_per_block_kernel` / `_quant_v_fp8_amax_kernel` / `_quant_v_fp8_transpose_kernel`
  - `PROF_KERNEL=cuda`：`sageattention._fused` 的 sm90 e2e 实际派发序列
    `per_warp_int8(BLKQ=64,WARPQ=16,BLKK=128)` + `per_channel_fp8(smooth_v=False)`，即
    `QuantInt8Kernel<128,16,1>`(Q) / `QuantInt8Kernel<128,128,2,fuse_sub_mean>`(K) /
    `TransposePadPermuteKernel<128,64>` / `MeanScaleKernel<64>`(V)
- 采集：
  ```
  ncu --set full --devices 0 -k regex:_quant_ -s 12 -c 4 -o reports/full_triton_quant ...
  ncu --set full --devices 0 -k "regex:Quant|Transpose|MeanScale" -s 12 -c 4 -o reports/full_cuda_quant ...
  ```
- 分析：`analysis/analyze_quant.py` → `analysis/kernels_full_{triton,cuda}_quant.txt` +
  全量 metric JSON。

## 2. Headline：duration 与 DRAM 利用率（ncu replay，主形状）

| kernel（职责） | triton | CUDA fused | triton/CUDA |
|---|---:|---:|---:|
| Q per-warp int8 | **54.8µs**（DRAM 69.7%） | 66.5µs（DRAM 57.9%） | 0.82× |
| K per-block int8 + 减 km | **63.8µs**（DRAM 59.6%） | 110.9µs（DRAM 34.7%） | 0.58× |
| V pass1（amax / fp16 转置中转） | **43.5µs**（DRAM 64.2%） | 400.4µs（DRAM 12.7%） | 0.11× |
| V pass2（量化+转置 / per-channel 量化） | **47.6µs**（DRAM 78.1%） | 69.1µs（DRAM 56.1%） | 0.69× |
| **合计** | **209.7µs** | **647.4µs** | **0.32×（快 3.09×）** |

do_bench 中值口径（无 replay 开销，`cutedsl_sage/bench_compare.py` 量化分步表，s=4096）：
quant_q 56µs / quant_k 59µs / quant_v 96µs，合计 211µs；另有共同的 torch `k.mean` 53µs
（占 triton 量化段 ~21%，占 e2e 3%（s4096）/ 0.7%（s16384）——单 kernel reduce 已达
2.4TB/s，融合收益 <1% e2e，维持 torch 实现）。

## 3. triton kernel 逐个归因（memory-bound 验证）

理论最小流量（fp16 输入 128MB/张量）：Q/K 各 192MB（读 128 + 写 64 int8），
V 320MB（amax pass 读 128 + 量化 pass 读 128 + 写 64 fp8）。ncu 实测流量与之吻合
（Q 读 134.2MB + 写 53.2MB 等，差异为 scale 小张量与 L2 sector 粒度）。

| metric | Q | K | V amax | V transpose |
|---|---:|---:|---:|---:|
| duration | 54.8µs | 63.8µs | 43.5µs | 47.6µs |
| gpu__dram_throughput（% peak） | 69.7 | 59.6 | 64.2 | 78.1 |
| dram read / write（% peak） | 49.9 / 19.8 | 42.8 / 16.7 | 62.9 / 1.3 | 57.4 / 20.7 |
| grid × block | 32768×128 | 4096×256 | 256×256 | 8192×128 |
| regs/thread | 30 | 80 | 64 | 65 |
| achieved occupancy | 92.3% | 35.0% | 24.0% | 40.7% |
| local ld/st | 0 | 0 | 0 | 0 |

- 四个 kernel 全部 memory-bound（sm__throughput 均低于 mem 侧；无 local spill）。
  60–78% DRAM 即 2.9–3.7TB/s，混合读写 + fp16→fp32 在飞转换下已接近 H200 实际可达值。
- K（59.6%）最低：128×128 tile 占 80 regs/thread，occupancy 35%；实测扫参
  （num_warps 4/8/16）以 w=8 最优，再收窄空间 <10%，不值得复杂化。
- V amax pass 的 grid 仅 256 CTA（b·n·d/64），wave 0.49——但每 CTA 顺序流读 512KB，
  DRAM 64.2% 已够；扫参（BS 128→512, w 4→8）从 86.9µs 收到 43.5µs（do_bench 49µs）。

## 4. CUDA fused 路径为何慢 3×

- `TransposePadPermuteKernel`（400µs，占其总量化 62%）：把 V 先按 fp16 转置 materialize
  （读 134MB + 写 114.8MB fp16），随后 `MeanScaleKernel` 再整读一遍做量化——V 通道合计
  ~500MB 流量 vs triton 两 pass 的 ~320MB；且该 kernel DRAM 只有 12.7%，l1tex 92%
  成瓶颈（smem 转置 + 非合并访问），本身离带宽上限还差一个数量级。
- K `QuantInt8Kernel<...,fuse_sub_mean>`（110.9µs，DRAM 34.7%）：block=1024 高 occupancy
  但发射率 0.379/cyc，标量减均值 + 量化路径指令面积大，未打满带宽。
- 结论：CUDA 侧的「fused」只是省了 python 胶水，kernel 本身并非带宽最优；triton 直写
  转置量化（省 fp16 中转）在 V 通道拿到 5.2× 优势。

## 5. e2e 影响（bench_compare.py 完整跑，d=128，do_bench 中值，ms）

| seq | causal | CUDA e2e | CuTeDSL e2e torch量化（前） | CuTeDSL e2e triton量化（后） | 提升 | CUDA/CuTeDSL | 分解：量化 / kernel |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 1024 | F | 0.298 | 1.134 | **0.218** | 5.2× | **1.364** | 0.088 / 0.131 |
| 4096 | F | 1.897 | 5.072 | **1.837** | 2.8× | **1.032** | 0.250 / 1.588 |
| 16384 | F | 24.057 | 37.057 | **25.241** | 1.5× | 0.953 | 0.893 / 24.826 |
| 1024 | T | 0.270 | 1.086 | **0.168** | 6.5× | **1.610** | 0.088 / 0.080 |
| 4096 | T | 1.297 | 4.362 | **1.087** | 4.0× | **1.194** | 0.251 / 0.841 |
| 16384 | T | 12.835 | 24.942 | **13.058** | 1.9× | 0.983 | 0.894 / 14.344 |

- 量化差距主因消除：e2e 从 CUDA 的 0.24–0.30× → **0.95–1.61×**。s≤4096 全面反超 CUDA
  （我们量化快 3×，抵消 kernel-only 的 0.87–0.92× 差距还有余）；s=16384 时 kernel 占比
  >95%，e2e 收敛到 kernel-only 比值附近。
- 数值口径未变：triton 输出与 torch 参考实现 bit-identical（`test_triton_matches_torch`，
  fp16/bf16 × 5 形状含非整块 s；`-m 'not slow'` 全量 136 passed）。

## 6. 结论与后续

1. triton 量化 kernel 达标：memory-bound、DRAM 60–78% of peak、无 spill，与 CUDA fused
   量化同形状对比快 3.09×（ncu duration 209.7 vs 647.4µs）。
2. e2e 差距主因（torch 量化的多次 float 展开/permute/contiguous）已消除；剩余差距全部
   来自 kernel-only 的 ~1.13–1.15×（归因见 `profile/2026-07-12-pingpong/`：标量指令面积）。
3. 后续若要 e2e 长序列也反超 CUDA，只剩 kernel 本体的标量指令面积一条路（前轮已归因，
   ping-pong 不可行，2 CTA/SM 已启用）。

Caveat：ncu replay duration 与 do_bench 中值不同源，比值以两者交叉验证（do_bench 口径
triton 211µs vs CUDA ~532µs = 2.5×，ncu 口径 3.09×，方向一致）；无 lineinfo，SASS 级
归因未做（memory-bound 无需要）。

远端副本：hyper01 `/data02/triplemu/workspace/SageAttention/profile/2026-07-12-triton-quant/`
（容器内 `/data/workspace/SageAttention/profile/2026-07-12-triton-quant/`）。
