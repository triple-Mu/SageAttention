# SageAttention 交接文档(2026-08-30,收尾会话后)

新会话读本文件即可接续。分支 `feat/varlen`(未 push),线性历史 = 两轮重构 + varlen + 全部优化/修复 + 本轮收尾(整理净 −1191 行、C1 WS kernel、P2/P4 落袋);工作树应干净。本轮完整账目(收益表、判负、挂账)在 `bench/FINAL_PASS_REPORT.md`,先读它。

## 1. 机器与环境

| 机器 | 连接 | GPU | 环境 | 工作目录(清理后) |
|---|---|---|---|---|
| 本机 | — | RTX 3080Ti Laptop(sm_86) | `/home/ubuntu/miniconda3/envs/torch/bin/python`(torch 2.13.0+cu132,nvcc 13.3;**别用 .venv**);cmake/ninja 用同 env 的 | 本仓库,editable install 就绪;**本机 golden:`~/sage-golden-local/{full,full2}`**(1493/1541) |
| hyper01(H200×8, sm_90) | `ssh hyper01 "docker exec sglang-diffusion-triplemu bash -c '<cmd>'"` | **卡有他人任务,现场挑空卡**;功耗 cap,亚 1% 信号必须 ncu 锁基频仲裁 | `source /workspace/.sglang/bin/activate` | `/workspace/{sage-w3,sage-w4}`(a264fc0 / 终验树);golden `/workspace/sage-golden-sm90`;证据 `/workspace/{sm90-7a64fc0,p2-sm90,sage-evidence-archive}`;待裁决 `SageAttention-rowsum` |
| pro-5k(PRO 6000×8, sm_120) | `ssh pro-5k "docker exec sglang-diffusion-triplemu-inference bash -c '<cmd>'"` | 常空,GPU 0 | `source /workspace/sgl-env/bin/activate` | `/workspace/SageAttention-refactor/{baseline,golden-sm120,archive/}`;**注意 `/workspace/SageAttention` 有 4 个未 push commit(v2g 工作),别动** |
| ComputeLab(L20/B200/A100) | `ssh computelab-sc`,csh 登陆节点,**非交互命令必 bash -c 包裹,多步写脚本 scp 执行** | Slurm+Pyxis | `clab.py -p {b200x4,l20x1,a100x1} --sqsh .../pytorch_26.07-py3.sqsh alloc`;**--sqsh 第一条命令就要带(默认 cu130 镜像编不了 sm_100a);多任务显式 --job-id;完毕必 cancel** | 容器把 `/home/scratch.sonlin_wwfo/workspace/nvidia` 挂为 `/workspace`;树 `SageAttention_refactor/{baseline,sage-w4,golden-sm100,sm89/,scripts/,logs*}` |

传代码:`git archive HEAD --prefix=<name>/ | gzip` 后 scp/管道解包(别 tar 工作树)。远端构建:`TORCH_CUDA_ARCH_LIST=<arch> python setup.py build_ext --inplace` + PYTHONPATH 跑。

## 2. 验证门禁(改动后的口径)

- **双级门禁**:等价改动走 bitwise golden diff=0;精度换性能改动走 accuracy gate(cos>0.99/rel_l1<0.06 对 SDPA fp32)+ BENCH_PROTOCOL 双闸,验收后重 dump golden。
- **bit-exact 对拍**:`python tools/compare_reference.py --check --golden-dir <G>`(别用 -m);attn 段走 `fwd(backend="smXX")` 跨家族覆盖;旧 golden 里退役 case 由 RETIRED_CASE_MARKERS 计 skip(本机 +320、H200/B200 +198,预期内)。
- pytest 本机基线:**544 passed / 278 skipped**(批 1 删 qattn 用例、P7 加 124 个 varlen 用例)。
- SASS 门禁:`tools/sass_diff.sh`(fresh vs fresh);kernel 数变化时用 scratchpad 的 `sass_subset_cmp.py` 口径;共享 body 别在条件编译里提局部变量;ninja 空 deps 假阴性要核对目标 TU 真重编;**mtime 保留式还原(copy2)后必须 touch 再重建**。
- sm100 通则:golden 全绿不够,必须 `bench/sm100_review/ws_stress.py`(SWEEP + 8000×2 定点);ncu 必须 `-k "regex:qk_int8_sv_f8_attn_kernel_sm100"` 过滤(张量初始化也是 launch);per-issue stall 比值跨版本不可比,绝对判据用 duration。
- bench 口径:BENCH_PROTOCOL(独占、双向交替、几何均值>0.5%、方向一致);工具 `scripts/cdsl_bench_fwd.py`(集群)、`bench/sm100_review/ws_{stress,prof}.py`、`bench/p4_vfuse_sweep.py`(均已入仓)。
- 构建 OOM:本机并发×NVCC 线程 ≤16;wheel 前 `rm -rf build/lib*`。

## 3. 本轮新增语义(必读)

- `fwd` 有 keyword-only `backend=None` 覆写(resolve 的 req_backend 透传);显式 backend 撞未编 SASS 会得到干净 TORCH_CHECK(plan.cpp `backend_serves`)。
- **`SAGEATTN_SM100_WS` 三态**:未设=auto(d128 且 qo_len≥16384 非 causal / ≥32768 causal 自动走 WS kernel)、`1`=强制 WS、`0`=强制旧(跑旧路 golden/SS oracle 必须显式 0)。
- V 融合门限 per-arch:sm89=12288、sm100/110=24576、其余 4096(quant_cuda.cu `fused_v_quant_max_tokens`);两路 bit 等价,换路不动 golden。
- quant kernel:per_warp/per_thread 家族 dense/varlen 双实例;per-block 家族 dense 仍走 kVarlen=true 实例(H200 判据,见 FINAL_PASS_REPORT §3)。
- `SAGE_PRUNE_GENCODE`(默认 OFF):GO 建议在案(build CPU −36%/体积 −38%),转正待用户拍板。

## 4. 已完成 / 判负 / 挂账

全部见 `bench/FINAL_PASS_REPORT.md`(收益表、判负表、挂账清单)与 `test/HARDWARE_CHECKLIST.md`(总账本)。C1 设计与迭代史:`bench/sm100_review/C1_DESIGN.md`。历史两轮重构与 varlen 的结论不变(五 arch bitwise 全 diff=0、varlen 语义与 cudagraph 安全)。

## 5. 速查

- memory 目录(`~/.claude/projects/-home-ubuntu-workspace-github-llm-SageAttention/memory/`)有全部教训索引;协作纪律:多会话 worktree 隔离、改动一成形就 commit、共享 kernel 文件先协调、**给 subagent 的 worktree 初始 HEAD 可能指错,第一步 reset --hard 到指定 SHA**。
- `sageattention3_blackwell` 不入 wheel;`native_quant_op.py`、`.idea/`、`refs/`、`sageattention-patches*` 不入库(git add -A 会误收,用显式路径)。
- feat/varlen 未 push;push/PR 待用户确认。
