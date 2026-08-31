# SageAttention 交接文档(2026-08-31,wave25 收口后)

新会话读本文件即可接续。分支 `feat/varlen` @ e3e65b5(已 push origin),线性历史 = 两轮重构 + varlen + 收尾整理 + sm100 C1 战役(wave9-25 全部收编,判负项已 revert 只留档)。账目三层:总账 `bench/FINAL_PASS_REPORT.md`(收益/判负/挂账);sm100 设计与逐 wave 判决 `bench/sm100_review/{C1_DESIGN,D64_DESIGN,SM100_VARLEN_DESIGN,BEYOND_CUDNN_PLAN}.md`;硬件约束 `test/HARDWARE_CHECKLIST.md`。

## 1. 机器与环境

| 机器 | 连接 | GPU | 环境 | 工作目录 |
|---|---|---|---|---|
| 本机 | — | RTX 3080Ti Laptop(sm_86) | `/home/ubuntu/miniconda3/envs/torch/bin/python`(torch 2.13.0+cu132,nvcc 13.3;**别用 .venv**);cmake/ninja 用同 env 的 | 本仓库,editable install 就绪;本机 golden:`~/sage-golden-local/{full,full2}`(1493/1541) |
| hyper01(H200×8, sm_90) | `ssh hyper01 "docker exec sglang-diffusion-triplemu bash -c '<cmd>'"` | 卡有他人任务,现场挑空卡;功耗 cap,亚 1% 信号必须 ncu 锁基频仲裁 | `source /workspace/.sglang/bin/activate` | `/workspace/{sage-w3,sage-w4}`;golden `/workspace/sage-golden-sm90`;证据 `/workspace/{sm90-7a64fc0,p2-sm90,sage-evidence-archive}`;`SageAttention-rowsum` 保留(用户拍板) |
| pro-5k(PRO 6000×8, sm_120) | `ssh pro-5k "docker exec sglang-diffusion-triplemu-inference bash -c '<cmd>'"` | 常空,GPU 0 | `source /workspace/sgl-env/bin/activate` | `/workspace/SageAttention-refactor/{baseline,golden-sm120,archive/}`;**`/workspace/SageAttention` 有 4 个未 push commit(v2g 工作),别动** |
| ComputeLab(L20/B200/A100) | `ssh computelab-sc`,csh 登陆节点,**非交互命令必 bash -c 包裹,多步写脚本 scp 执行** | Slurm+Pyxis | `clab.py -p {b200x4,l20x1,a100x1} alloc`;**每条命令都带 `--sqsh .../pytorch_26.07-py3.sqsh`(默认 cu130 镜像编不了 sm_100a)与 `--job-id`;完毕必 cancel** | 容器把 `/home/scratch.sonlin_wwfo/workspace/nvidia` 挂为 `/workspace`;树 `SageAttention_refactor/`(布局见下) |

B200 会话纪律(wave19 起固化,违者当轮数据作废):

- **健康门**:alloc 后先跑 `scripts-w19/w19_05_health.sh`(空载 <70°C + 5 轮 matmul spread <5%),判坏 cancel 换节点,`--exclude` 累加。已知坏节点:umbriel-b200-019(过热掉频,w19)、umbriel-b200-069(load probe spread 5.8%,w24);留档 `logs-w19-node019-bad/`、`logs-w24-attempt1-node069/`。
- **集群 `SageAttention_refactor/` 布局**:`golden-sm100`(旧路 golden,ok=2082)、`golden-sm100-g1ws`(ws 路,ok=2107)、`sage-w*.tar.gz`(逐 wave 树 archive,最新 sage-w14 = wave25)、`logs-w{4,9,11,14,16,18,19,20,23,23b,24,25}/` 与 `logs-a3/`(逐 wave 原始数据:golden/stress/bench/ncu)、`scripts-w{11,19,23,24,25}/ scripts-a3/`(会话脚本,健康门、bench 驱动、ncu 归因都在)。
- 传代码:`git archive HEAD --prefix=<name>/ | gzip` 后 scp 解包;远端构建 `TORCH_CUDA_ARCH_LIST=10.0a python setup.py build_ext --inplace` + PYTHONPATH 跑;**golden 对拍构建必须 `SAGEATTN_CMAKE_ARGS=-DSAGE_PRUNE_GENCODE=OFF`**(默认 ON 裁掉 sm80 家族,missing=792)。

## 2. 验证门禁(wave25 现行口径)

- **双级门禁**:等价改动走 bitwise golden diff=0;精度换性能改动走 accuracy gate(cos>0.99/rel_l1<0.06 对 SDPA fp32)+ bench 双闸,验收后重 dump golden。
- **golden 双轨(sm100 三轨)**:`WS=0` 对 `golden-sm100`(ok=2082);`WS=1` 与 `WS=1 PERSIST=1` 对 `golden-sm100-g1ws`(ok=2107;wave11 G1 切轨,唯一一次精度换性能重 dump)。auto 态不做 golden,正确性由强制轨覆盖,路由走自检(`scripts-w25/w25_routing.py`,torch.profiler 认 kernel 名)。
- **bit-exact 对拍**:`python tools/compare_reference.py --check --golden-dir <G>`(别用 -m);attn 段走 `fwd(backend="smXX")`;RETIRED_CASE_MARKERS 计 skip 预期(本机 +320、H200/B200 +198)。
- **FFMA opcode 对账(wave23 新增硬规则)**:声称 bit-exact 的控制流改动(加 branch/spin/asm),本地闸必须做稳态循环 FP opcode 逐类计数对基线(全函数 + 最小含 128 MUFU 回跳段双口径);FFMA↔FMUL+FADD 计数漂移即 FAIL。根因:控制流切开 basic block 会破坏 nvcc 的 mul+add 收缩,denom 漂 1 ulp(golden 132 diff 破案,C1_DESIGN §15.2);修法是显式 `fmaf()` 固化(7a7c84d),不是哄编译器。
- **单改动复验条款(wave23b)**:多个改动同场上机,任何 KEEP 判决必须预注册单改动复验;融合树读数不作数(wave23 融合树 d64 +4.8pp 全是同场另一改动的,拆开后 gate 单独为负)。
- **bench 口径**:BENCH_PROTOCOL(独占、双向交替 ≥3 轮 median、几何均值 >0.5%、方向一致);sm100 判据模板 = 22 点主网格(d128 20 + d64 probe 2)与 d64 12 点网格,geomean >1.005 且无形状 <0.995;**判据以自由时钟 bench 为准**(wave23 实证 ncu 锁频 duration 可与 bench 反向)。
- **cycles 口径**:ncu w11/w14 是锁频 1.48 GHz,w16 起自由 boost;跨 wave 比较用 cycles,别混墙钟。
- **ncu**:必须 `-k "regex:qk_int8_sv_f8_attn_kernel_sm100"` 过滤(张量初始化也是 launch);per-issue stall 比值跨版本不可比,绝对判据用 duration。
- **sm100 压测**:golden 全绿不够,必须 `bench/sm100_review/ws_stress.py`(SWEEP + 8000×2 定点);varlen 判绿口径 = 6000 满轮,不认 iter 数(V7 曾在 691 次才击中);分钟级复现用 `w16_isolate2.py --mode a2b` 固定形状臂。B200 上 cuda-gdb attach 拿不到 CUDA 态(Yama+driver),可靠姿势 = from-launch 带跑 + 停滞后 SIGINT 打断。
- SASS 门禁:`tools/sass_diff.sh`(fresh vs fresh);共享 body 别在条件编译里提局部变量;mtime 保留式还原(copy2)后必须 touch 再重建。
- pytest:B200 全量 436 passed / 395 skipped(w18 实测);本机 544/278 是 sm100 varlen 用例合入前的口径,重跑以实测为准。
- 构建 OOM:本机并发×NVCC 线程 ≤16;wheel 前 `rm -rf build/lib*`。

## 3. 运行时语义(必读)

- **`SAGEATTN_SM100_WS` 三态**:未设/auto = head_dim 64 与 128 都走 ws 路(wave25 f428eb3 起;d64 判据见 D64_DESIGN §8.5.3/8.6);`1` = 强制 WS;`0` = 强制旧 kernel(跑旧路 golden/SS oracle 必须显式 0)。
- **`SAGEATTN_SM100_WS_PERSIST` 三态**:未设/auto = ws 路里非 causal 走 persistent、causal 走 per-tile ws(wave20 fc03183 起,causal persist 判负维持);`1` = 全部强制 persistent;`0` = 全部 per-tile ws。auto 组合 = nc→persistent、causal→ws,d64+d128 全段最优(wave25 组合态复验,34 点)。
- **varlen sm100 已恢复**:wave17 把 causal mask 链挪出 S 排空循环(75b8012)后 `_VARLEN_BACKENDS` 收回 sm100(b78ea2a);走 classic 128 线程 kernel 双实例,WS 版 varlen 未做。S 排空的 ld/wait 区间内禁混 mask/逐元素谓词链(第三条禁区,HARDWARE_CHECKLIST 约束表)。
- `fwd` 有 keyword-only `backend=None` 覆写;显式 backend 撞未编 SASS 得到干净 TORCH_CHECK(plan.cpp `backend_serves`)。
- V 融合门限 per-arch:sm89=12288、sm100/110=24576、其余 4096;两路 bit 等价,换路不动 golden。
- `SAGE_PRUNE_GENCODE` 默认 ON(build CPU −36%/体积 −38%);对拍场景见 §1 末条。

## 4. 战线图(wave25 终表,B200,`logs-w25/`,D64_DESIGN §8.6.3)

auto(零 env 配置)对 old(classic)与 cudnn 9.24,median of 3,全 exclusive:

| 段 | auto/old | auto/cudnn | 剩余谷地 |
|---|---:|---:|---|
| d128 c0(10 点) | 1.3332 | **1.0005** | s4096 0.84-0.93、b4 s1024 0.84;s≥16k 全部 ≥1.0(最高 1.11) |
| d128 c1(10 点) | 1.2410 | 0.9020 | s1024-16384 0.65-0.99;s≥32k 0.997-1.10 |
| d64 c0(6 点) | 1.0795 | 0.7112 | 全段;XU roofline 束缚,可达带 ~0.79(s4k)/0.92(s16k) |
| d64 c1(6 点) | 1.0389 | 0.6442 | 同上,causal 更深 |

关键里程:非 causal d128 整段追平 cudnn(auto/cudnn 1.0005),s≥32768 段自 wave16 起稳定 >1;d64 auto 用户侧净赚 +3.0~11.5%(vs classic)。cudnn 目标线必须同场重测,历史值会漂(BEYOND_CUDNN_PLAN §1.3)。

## 5. 判负黑名单(wave16-25 增量;全表 FINAL_PASS_REPORT §3 + BEYOND_CUDNN_PLAN 附录 A,立项前必查)

| 方向 | 结论 | 证据 |
|---|---|---|
| P 64 列分块交付(常量 parity,两次完成/步) | 真机 tile 0 即死锁(wait/parity 错配),revert 494d4f5 | C1_DESIGN §11.4,`logs-w16/` |
| A′ RESCALE_THRESHOLD 惰性 max(T=4) | bench 0.9967 fail;bench 协议 randn 刻度下命中集近空,correction issued 无崩落;revert 910a831 | C1_DESIGN §12.5,`logs-w18/` |
| wave22 vec_full 交付(int max + issue wall,d128 形态)| ws/old −0.8%、wsp/old −3.3%,长段线破;revert 2fa877e。d64-only 形态 wave24 重启并 KEEP(8b5f814) | C1_DESIGN §15.4,`logs-w23/` |
| wave22 persist causal EX2 phase gate | wsp/ws 0.8949 < 0.90 判负线;revert b6c4569 | C1_DESIGN §15.4 |
| wave22 d64 EX2 phase gate(moved wait 造 stagger) | 单改动复验 0.9575 < 0.96,mio/XU 反向恶化;revert d9e1afc | C1_DESIGN §15.5、D64_DESIGN §8.4,`logs-w23b/` |
| FA3 式 varlen persistent scheduler | 偏斜浪费按 KV tile 归一后 −9.0~+3.6%(判据 ≥5%);空 CTA 被硬件 work distributor 即时回填 | bench/DIVE4_SCHED_PRESCREEN.md(sm120) |
| varlen 挂死的调度差假设(E5 两臂)与 C2 族(裸退/CTA churn) | 13 臂单变量矩阵全部出局,病灶 = mask 链在排空区间 | SM100_VARLEN_DESIGN §6.4.5/6.4.6,`logs-a3/` |

历史项(E1 f16x2 softmax、P6 PDL、sm90 process_tile、P2 per-block、P8、C1 x128、lever A 单项、软件 exp2、A5 4+ outstanding、B1 ring 加深等)不赘,查上述两表。

## 6. 挂账(明确悬置)

- **d128 c1 残差(当前主谷地)**:causal persist 判负后 c1 走 per-tile ws,auto/cudnn s1024-16384 只有 0.65-0.99。已知线索:W=1 退化下 c1 每 kv 块斜率 +13% 的动态相位残差未归因(需上机 per-barrier ledger,C1_DESIGN §13.6.1);`q_empty` 提前 commit(攻 item 边界 runway)已记档待 ncu 指向边界气泡再立项(§13.6.2)。EX2 gate 两条路已判负,别重走。
- **d64 形态天花板**:int8-QK 形态内追平 cudnn 需 XU 利用率 80-89%,演示上限 67-72%(D64_DESIGN §1.2/§7.1);现值 auto/cudnn c0 0.711 / c1 0.644。可动的:G3 批量 drain 被 A5 红线(4+ outstanding tcgen05.ld 挂死根因未定位)封锁;C2(2 CTA/SM,256 线程)是唯一保留的结构候选,机制已被 w19 坐实(§3.4/§7.2);fp8-QK d64 形态(XU 降到 1.5 op/元素,54% 利用率即 1.15×)是唯一 ws/cudnn>1 路径,产品决策,出界挂账。
- **cga2(tcgen05 cta_group::2,Phase C)**:cudnn 的地基;启动条件 = ncu 显示 KV 供给/L2 成新顶,或 b1 长序列坐实卡供给;工作量大(BEYOND_CUDNN_PLAN §4.6)。
- **init-fence 同型缺口 follow-up**:varlen sm100 的 mbarrier init 发布 fence 已落(f3e617b);dense sm100、sm100 ws、sm90 三处同型缺口(init 后无 `fence.proxy.async`)未补,各需带 SASS gate 的独立 commit(SM100_VARLEN_DESIGN §6.4.3)。正确性卫生项,非性能。
- **A3 机理**:mask 链在 S 排空区间为何让邻近 CTA 丢 mbarrier completion,微架构层无答案;按 A2/A5 同格入禁区表绕行(SM100_VARLEN_DESIGN §6.4.6 遗留节)。
- **varlen sm100 WS 版**:未立项(classic 已达 ragged 1.13-1.54×,dense ws 结构移植的收益账未算)。
- 清理终审(2026-08-30 用户拍板)不变:computelab 7 个历史构建树已删;hyper01 `SageAttention-rowsum` 与 pro-5k `/workspace/SageAttention` 保留;证据类目录保留。

## 7. 速查

- memory 目录(`~/.claude/projects/-home-ubuntu-workspace-github-llm-SageAttention/memory/`)有全部教训索引;协作纪律:多会话 worktree 隔离、改动一成形就 commit、共享 kernel 文件先协调、**给 subagent 的 worktree 初始 HEAD 可能指错,第一步 reset --hard 到指定 SHA**。
- `sageattention3_blackwell` 不入 wheel;`native_quant_op.py`、`.idea/`、`refs/`、`sageattention-patches*` 不入库(git add -A 会误收,用显式路径)。
- feat/varlen 已 push origin(e3e65b5);PR 待用户确认。
