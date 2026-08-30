# sm80 非 causal attention:k_scale_off S2R remat 的同款修复

sm89 修复(2330989)的 sm80 跟进。同一根因:process_tile 三副本合并后
`k_scale_off`(含 `lane_id % 4`)活跃区间横跨 kBulk/kMask/kLast 三个展开点,
255 寄存器顶格下 ptxas 在 KV 主循环内用 `S2R SR_TID.X` 重算,落在每轮
`dequant_scale` 依赖链上。修复同款:非 causal 且 `!return_lse` 的实例把
`k_scale_off` 预折进 `K_scale_base_ptr`(commit 612ab3d,仅动
`csrc/qattn/qk_int_sv_f16_sm80_impl.cuh`)。

## 1 SASS 手术边界(sm_86,nvcc 13.3,tip 6ae9fe6 vs +预折)

- dense TU(129 kernel):**113 个逐字节一致,16 个变化**,全部是
  MaskMode 0 + lse=false(f32 / f16 / f16+inst_buf / f32+inst_buf /
  fuse_v_mean 五个家族 × hd64/128 × half/bf16)。causal 与 lse 实例零变化。
- varlen TU(33 kernel):29 个一致,4 个变化(f32-accum 家族)。
- 变化实例的 opcode 直方图 delta 全部是整数/地址/搬运类
  (IMAD/LEA/LOP3/S2R/U* 等),**FP/MMA 指令流不变** → 数值 bit-exact。
- 热实例(hd128 f32-accum,plan.cpp 默认路径):主循环 861→860 条,
  循环内 S2R 1→0;全函数 3776→3744 条。新增 LDL/STL 与 +8 B stack
  都在主循环外(loopscan 确认循环内 LDL=1/STL=0 不变)。
- 寄存器持平或下降:f32 hd64 243→230,f16 hd64 215→194;无实例升高。

## 2 sm86 实测(RTX 3080 Ti Laptop,12 轮三方 round-robin 配对)

kernel_breakdown.py,attention kernel 每调用 µs,每轮 base(0a5d2e4)/
new(tip)/ fix(tip+预折)轮换顺序抵消热漂移,取轮内配对比值的中位数。
笔记本会降频,且桌面编码进程(awesun_desktop,339 MiB)常驻,三方对称承受。

| shape(hd128 fp16) | base µs | new µs | fix µs | new/base | fix/new |
|---|---|---|---|---|---|
| 8×32×1024 | 2277 | 2318 | 2305 | 1.018 | 0.995 |
| 4×32×2048 | 4517 | 4624 | 4580 | 1.024 | 0.990 |
| 2×32×4096 | 8156 | 7631 | 7791 | 0.946* | 1.009* |
| 1×32×8192 | 14418 | 14828 | 14664 | 1.029 | 0.989 |
| 4×32×2048 causal | 1962 | 1953 | 1958 | 0.999 | 1.002 |
| 1×32×8192 causal | 7530 | 7449 | 7472 | 0.990 | 1.004 |

\* 4096 档轮间散布 min 0.796 / max 1.189(其余档 ±2% 以内),该档判为噪声,
不采信。causal 两档 fix 实例逐字节一致,其 fix/new(1.002/1.004)就是本机
噪声标尺。

剔除 4096 档:非 causal new/base 几何均值 1.024(tip 慢 2.4%),fix/new
0.992(**修复收回约 0.9%**),方向三档一致且越过噪声标尺。残余 fix/base
≈1.015 在本机(降频 + 共享编码器)分辨率内无法归因,上机口径见 §4。

## 3 门禁

- golden:`tools/compare_reference.py --check` vs cmp/full,
  **ok=1493 diff=0**(extra=48 varlen equiv 为预期),产物 `golden_check.log`。
- pytest:**552 passed / 154 skipped**,零失败(`pytest_fix.log`)。
- SASS 对比产物:scratchpad `s2r80/{dense_diff,varlen_diff}.txt`、
  `s2r80/kb/`(36 份 kernel_breakdown JSON)。

## 4 残留与后续

- 已结清——sm_80(A100)实跑:L20(sm_89 SASS)复测 base/fix 0.9995
  ([P8_SM80_FIX_L20.md](P8_SM80_FIX_L20.md));A100 80GB PCIe(sm_80
  实卡)8 轮顺序平衡 + causal 控制组漂移校正后 0.9988
  ([P8_SM80_FIX_A100.md](P8_SM80_FIX_A100.md))。两块数据中心卡上 fold
  无收益无害,实证收益仅剩本节的 sm_86;回收建议见 A100 报告 §5。
- varlen hd128 实例 stack 16→24(冷路径);varlen 路径未单独 bench,
  机制与 dense 相同,风险低。
- 构建系统一个坑(与本修复无关,已绕过):dense sm80 TU 在 ninja deps
  数据库里出现过 `#deps 0` 的空记录,改头文件后不重编,需删 .o 强制。
