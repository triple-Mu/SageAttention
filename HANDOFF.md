# SageAttention 交接文档(2026-08-30)

新会话读本文件即可接续全部工作。分支 `feat/varlen`(未 push),线性历史含两轮重构 + varlen + 全部优化/修复,工作树应干净。产出原则:最少字,砍表达不砍信息。

## 1. 机器与环境

| 机器 | 连接 | GPU | 环境启动 | 工作目录 |
|---|---|---|---|---|
| 本机 | — | RTX 3080Ti Laptop(sm_86, 16GB) | `/home/ubuntu/miniconda3/envs/torch/bin/python`(torch 2.13.0+cu132,nvcc 13.3;**别用 .venv**);cmake 用 pip 版 | 本仓库;editable install 已就绪 |
| hyper01(H200×8, sm_90) | `ssh hyper01 "docker exec sglang-diffusion-triplemu bash -c '<cmd>'"` | 141GB/卡;**卡有他人任务,现场 nvidia-smi 挑空卡** | `source /workspace/.sglang/bin/activate` | `/workspace/SageAttention`(=重构前 0a5d2e4 已构建,当 baseline);`/workspace/SageAttention-{final,kbd,varlen,...}`(历史树);sm90 golden:`/workspace/sage-golden-sm90`(1488 case) |
| pro-5k(RTX PRO 6000×8, sm_120) | `ssh pro-5k "docker exec sglang-diffusion-triplemu-inference bash -c '<cmd>'"` | 72GB/卡,常空闲,用 GPU 0 | `source /workspace/sgl-env/bin/activate` | `/workspace/SageAttention-refactor/{baseline,new*,golden-sm120}` |
| ComputeLab(L20 sm_89 / B200 sm_100) | `ssh computelab-sc`,**登陆节点是 csh,非交互命令必须 `bash -c` 包裹** | B200 183GB / L20 46GB,Slurm+Pyxis | `python3 /home/scratch.sonlin_wwfo/workspace/nvidia/scripts/clab.py -p {b200x4,l20x1} alloc [--gpus 1] --queue-timeout 3600`;容器 `--sqsh /home/scratch.sonlin_wwfo/workspace/nvidia/enroot/images/pytorch_26.07-py3.sqsh`;`exec -- bash -c '...'`;**state 按 profile 存,多任务并发必须显式 `--job-id`;完毕必 `cancel`(当前队列已空)** | `/home/sonlin/scratch/workspace/nvidia/SageAttention_refactor/{baseline,new*,golden-sm100,sm89/}` |

传代码:`tar --exclude={.git,build,dist,assets,example,refs,cmake-build-debug,'sageattention-patches*','*.egg-info',__pycache__,'*.so',.idea,.claude} -czf - . | ssh <机> "docker exec -i <容器> tar -xzf - -C <目录>"`。远端构建:`TORCH_CUDA_ARCH_LIST=<arch> python setup.py build_ext --inplace` + `PYTHONPATH` 跑(别污染 venv);容器内 torch 2.13.0a0+nv26.07/nvcc 13.3。

## 2. 验证工具与门禁(每次改 kernel 后的口径)

- **bit-exact 对拍**:`python tools/compare_reference.py --check --golden-dir <G>`;本机 G=scratchpad 的 `cmp/full`(1493)与 `cmp/full2`(1541),远端各机 golden 见上表;baseline 侧 `--dump --backend legacy`。equiv 段(fwd 全等/pad 等价/varlen_vs_dense)只在 new 侧跑。
- pytest 基线:本机 **552 passed / 154 skipped**;compile 用例含 fullgraph 0 break + cudagraph。
- SASS 门禁:改 kernel 前抓 baseline(scratchpad 各 `gensass` 类脚本;`bench/kernel_breakdown.py` 亦可),**fresh build 对 fresh build**(增量构建 sm90 会出不同 SASS);**ninja deps 有过空记录导致改共享头不重编→SASS 假阴性,必须核对目标 TU 真的重编了**。
- bench 口径:`cutedsl_sage/BENCH_PROTOCOL.md`(cutedsl-sage-sm90 分支)——同卡独占、双向交替、全表几何均值、>0.5% 信号、方向一致性;H200 有功耗 cap 用 min-of-N。
- **sm100 特有通则:golden 全绿不够,改动必须补单形状 8000 次定点压测**(A5 是 golden 全过、压测才挂死)。
- 构建 OOM:本机并发 × NVCC 线程 ≤16(4×4);打 wheel 前 `rm -rf build/lib*`(stale 产物会混入)。
- CUDA 支持矩阵(实测,commit c56bed3):sm80≥12.0、sm89≥12.4、sm90≥12.5、sm120≥12.8、sm100/110≥13.1;torch 2.13 自身要 nvcc≥12.4。

## 3. 已完成与结论

- **重构**(CMake 单 `_C.abi3.so`、torch.ops 统一 dispatch、0 graph break、cudagraph、int64 分层契约):五 arch 对 0a5d2e4 bitwise 全 diff=0(sm86 1493/sm89-L20 2004/sm90 1488×2/sm100 2280×2/sm120 2578)。
- **varlen**(packed+cu_seqlens,FA 语义 bottom-right causal,`sageattn_varlen`/`fwd_varlen`,sm100 设计内拒绝):四 arch 实机全绿;等长逐位=dense,ragged 对补齐 dense 1.4-2.1×;闭式偏移代数(csrc/sageattn/varlen.h)无前缀和张量,cudagraph/fake 安全;quant kernel dense/varlen 共享(残余税实测 ±0.05%,**裁决维持混合不拆实例**)。
- **优化落袋**:sm100 TMEM 右尺寸化 hd64 1.53× + **A1 P叠S区 d128 1.577×**(对 cudnn 0.47→0.76×)+ v_scale 预载;V transpose+量化融合(短 seq e2e −5~18%,门限 padded 4096 只在 sm120 扫过);segment_mean kernel(varlen 等长 0.93-0.99);fp8 V memset 消除;quant 寄存器悬崖修复(sm90/100);S2R 指针预折(sm89+sm80 非 causal);sm8x sub_mean pack;sm80 B 批(causal +3-5%)。
- **判负封路**(全有实测,别重试):sm90 全部方向(WS 四件套/降 reg/H6/行和 mma 化/cluster/软件流水/exp2 仿真——现行 911 TFLOPS 即上限);C-1 三 arch(非 occupancy 受限);sm100 的 B1 KV ring(净零,且 ncu 下有挂死隐患)与 C2 S 双缓冲(0.65×);A2 k_scale 预载与 A5 TMEM 批量读(sm100 挂死回退,根因未定位);D 宏三个(默认 OFF 留档);C-8 sm120 v2。证据:`test/HARDWARE_CHECKLIST.md`(总账本)、`profile/*/REPORT.md`(cutedsl 分支)、`bench/*_REPORT.md`。
- **kernel breakdown 实验**:`bench/KERNEL_BREAKDOWN_REPORT.md` + 原始数据 `bench/kernel_breakdown_data/`;工具 `bench/kernel_breakdown.py`(两侧自适应/角色映射/独占检查)。
- NaN 三兄弟修复(dense V 尾部/zero-amax/fa3 descale);修饰符与命名统一 + test_style 机检。

## 4. 未做 / 可挖(按价值)

1. **C1:sm100 完整 warp specialization + 双 Q tile(重写级,唯一大空间)**——现 0.76× cudnn,cutedsl 1.07-1.18×;立项理由已重定:每 scheduler 仅 2 active/0.6 eligible warp、54% cycle 无指令可发(不是 barrier,实测仅 3.3%);目标 = warp 供给 2→8;先验 `setmaxnreg` 在 sm100a 可编;参考存档分支 `cdsl-p3-b1/c2` 的 barrier ledger 与 cutedsl `core_sm100.py`(16 warp 特化结构);封路:128 线程地基上加深预取、TMEM 腾挪。
2. `fp32+fp16` 默认退役决策:L20/PRO6000 已证同速更不准,**等 4090 跑一次 `bench/microbench/mma_rate.cu`**(消费级 Ada 2× 速率是变数)。
3. V 融合门限(4096)只在 sm120 扫过,L20/B200 重扫;sm89 残余 ~1.5% 非 causal 差距 L20 复测(bench/SM80_NONCAUSAL_FIX_REPORT.md)。
4. B2 V 自然布局(MN-major 消 transpose_pad,e2e 1-8%,desc parity 要重写);A4 packed f32x2 softmax(sm100,需推导 per_thread 4-class rowmax)。
5. 二期:sm100 varlen、FA3 persistent scheduler(偏斜 batch R9);sm89/120 softmax/mma 重叠(先读三份判负报告)。
6. 工程:`feat/varlen` push/PR;删 `qattn_smXX_*` 过渡 op(条件已达成,对拍脚本需改走 fwd,单独立项);sm89/sm120 缺 kernel 级 packed varlen 用例;`test_accuracy` 参数化后的 fp8 卡数值未采;本机 /data 有 ~90GB CUDA 镜像可 `docker rmi`。

## 5. 其他速查

- memory 目录(`~/.claude/projects/-home-ubuntu-workspace-github-llm-SageAttention/memory/`)有全部教训索引;协作纪律:多会话并发时 worktree 隔离、改动一成形就 commit(clean worktree 有被误清风险)、共享 kernel 源文件改动先协调。
- `sageattention3_blackwell` 不打进 wheel(pyproject include 已收紧)但保留目录;`native_quant_op.py` 保留不入库;`.idea/`、`refs/`、`sageattention-patches*` 不入库。
