# CuteDSL SageAttention (Hopper)：torch 量化 + CuTe-DSL kernel + 端到端 API。
# kernel 部分见文件下半部（Task 3 追加）。
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# ===== torch 量化（正确性优先，后续可换 triton/融合 kernel）=====

_AMAX_EPS = 1e-7    # 防除零下限，与 CUDA 版一致（csrc/fused/fused.cu:147）


def quant_q_int8_per_warp(q: torch.Tensor):
    """Q per-warp int8 量化：seq 按 64 分块、块内 4 段各 16 行，amax over (16 行, d)。

    q: [b, s, n, d] fp16/bf16 -> (q_int8 [b,s,n,d] int8, q_scale [b, n, ceil(s/64)*4] fp32)
    scale 段序 = seq 顺序的 16 行段（行 r 对应索引 r//16），与 CUDA quant_per_warp_int8_cuda 一致。
    尾块补零行不抬高 amax；全 padding 段 scale 钳到 _AMAX_EPS/127（非 0/NaN）。
    """
    b, s, n, d = q.shape
    nblk = (s + 63) // 64
    s_pad = nblk * 64
    qf = q.float()
    if s_pad != s:
        qf = F.pad(qf, (0, 0, 0, 0, 0, s_pad - s))
    qv = qf.view(b, nblk * 4, 16, n, d)
    amax = qv.abs().amax(dim=(2, 4)).clamp_min(_AMAX_EPS)              # [b, nblk*4, n]
    q_scale = (amax / 127.0).permute(0, 2, 1).contiguous()             # [b, n, nblk*4]
    # torch.round 为 round-half-to-even，与 CUDA float_to_int8_rn (cvt.rni) 一致
    q_int8 = torch.round(qv * (127.0 / amax)[:, :, None, :, None]) \
        .clamp_(-128, 127).to(torch.int8)
    return q_int8.view(b, s_pad, n, d)[:, :s].contiguous(), q_scale


def quant_k_int8_per_block(k: torch.Tensor, km: Optional[torch.Tensor] = None):
    """K per-block int8 量化（BLKK=128），量化前减 km（smooth_k）。

    k: [b, s, n_kv, d]，km: [b, 1, n_kv, d] 或 None
    -> (k_int8 [b,s,n_kv,d] int8, k_scale [b, n_kv, ceil(s/128)] fp32)
    先减 km 再补零，保证 padding 行不污染尾块 amax。
    """
    b, s, n, d = k.shape
    nblk = (s + 127) // 128
    s_pad = nblk * 128
    kf = k.float()
    if km is not None:
        kf = kf - km.float()
    if s_pad != s:
        kf = F.pad(kf, (0, 0, 0, 0, 0, s_pad - s))
    kv = kf.view(b, nblk, 128, n, d)
    amax = kv.abs().amax(dim=(2, 4)).clamp_min(_AMAX_EPS)              # [b, nblk, n]
    k_scale = (amax / 127.0).permute(0, 2, 1).contiguous()             # [b, n, nblk]
    k_int8 = torch.round(kv * (127.0 / amax)[:, :, None, :, None]) \
        .clamp_(-128, 127).to(torch.int8)
    return k_int8.view(b, s_pad, n, d)[:, :s].contiguous(), k_scale


def quant_v_fp8_per_channel(v: torch.Tensor):
    """V per-channel e4m3 量化并 materialize 为 [b, n_kv, d, s_pad] contiguous。

    v: [b, s, n_kv, d]；s_pad = ceil(s/128)*128，尾部零补（e4m3 精确 0）。
    v_scale = amax/448（amax 仅统计有效 seq），v_fp8 = v·448/amax。无 token permute。
    """
    b, s, n, d = v.shape
    s_pad = (s + 127) // 128 * 128
    vt = v.permute(0, 2, 3, 1).float()                                 # [b, n, d, s]
    amax = vt.abs().amax(dim=-1).clamp_min(_AMAX_EPS)                  # [b, n, d]
    v_scale = (amax / 448.0).contiguous()
    v_scaled = vt * (448.0 / amax)[..., None]                          # 幅值 ≤448，e4m3 不溢出
    if s_pad != s:
        v_scaled = F.pad(v_scaled, (0, s_pad - s))
    v_fp8 = v_scaled.to(torch.float8_e4m3fn).contiguous()
    return v_fp8, v_scale


# ===== CuTe-DSL kernel（Hopper sm90a）=====
# 移植底本：cutlass examples/python/CuTeDSL/hopper/fmha.py（注释 fmha.py:NNN 为其行号）

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

try:
    from .cutedsl_sage import CutedslKernel
except ImportError:
    from cutedsl_sage import CutedslKernel

LOG2_E = math.log2(math.e)
LOG2_448 = math.log2(448.0)   # ×448 折进 exp2 的编译期常量偏移（P1.2）

_TORCH_TO_CUTLASS = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}


class SageAttnSm90(CutedslKernel):
    # 独立缓存，key = (head_dim, is_causal, out_dtype, gqa_ratio)
    compile_cache: dict = {}

    CTA_Q = 64
    CTA_K = 128

    def __init__(self, head_dim: int, is_causal: bool, o_dtype, gqa_ratio: int):
        super().__init__()
        assert head_dim in (64, 128)
        self.head_dim = head_dim
        self.is_causal = is_causal
        self.gqa_ratio = gqa_ratio

        self.q_dtype = cutlass.Int8
        self.k_dtype = cutlass.Int8
        self.v_dtype = cutlass.Float8E4M3FN
        self.o_dtype = o_dtype
        self.qk_acc_dtype = cutlass.Int32      # s8s8s32，|S|<2^24 转 f32 无损
        self.pv_acc_dtype = cutlass.Float32    # 单级 FP32（FP22）累加

        # fmha.py:160-170
        self.qk_mma_tiler = (self.CTA_Q, self.CTA_K, head_dim)
        self.pv_mma_tiler = (self.CTA_Q, head_dim, self.CTA_K)
        self.cta_tiler = self.qk_mma_tiler
        self.atom_layout_mnk = (1, 1, 1)
        self.cluster_shape_mnk = (1, 1, 1)

        # K/V 共用一个环形 smem 缓冲（fmha.py:601），每次 KV 迭代占 2 个 stage（K 一个 V 一个），
        # 一个 stage = 128*d 字节（K 块与 V 块字节数恒等）。
        # P2a：目标 2 CTA/SM，smem 预算 113KB/CTA —— d=128 用 4 stage（合计 ~81KB），
        # d=64 每 stage 仅 8KB、10 stage 合计 ~93KB 本就达标，维持不变
        self.q_stage = 1
        self.kv_stage = 4 if head_dim == 128 else 10
        self.epi_stage = 2

        self.num_threads_per_warp_group = 128
        self.num_warps_per_warp_group = 4
        self.threads_per_cta = 256            # WG0 = load, WG1 = math
        self.load_warp_group_id = 0
        self.math_warp_group_id = 1
        self.num_regs_load = 24
        # P2a：2 CTA/SM 要求 (mma+load)·128·2 ≤ 64K regs/SM → 240→224（(224+24)·256=63488）
        self.num_regs_mma = 224
        self.buffer_align_bytes = 1024

    # ---------------- host：编译缓存 ----------------
    @classmethod
    def from_args(cls, head_dim: int, is_causal: bool, out_dtype: torch.dtype,
                  gqa_ratio: int) -> "SageAttnSm90":
        key = (head_dim, is_causal, out_dtype, gqa_ratio)
        cached = cls.compile_cache.get(key)
        if cached is not None:
            return cached

        kernel = cls(head_dim, is_causal, _TORCH_TO_CUTLASS[out_dtype], gqa_ratio)

        # b/s/n 动态、d/dtype 静态；紧凑行主序 fake tensor（runtime.py:604），
        # stride_order 值 0 = 最小 stride（runtime.py:642 示例），(3,2,1,0) 即 torch contiguous
        sym = cute.sym_int
        b, s_q, s_k = sym(), sym(), sym()
        s_pad = sym(divisibility=128)
        n_q = sym()
        n_kv = n_q if gqa_ratio == 1 else sym()
        n_wb_q, n_blk_k = sym(), sym()

        def fake4(dtype, shape):
            return cute.runtime.make_fake_compact_tensor(
                dtype, shape, stride_order=(3, 2, 1, 0), assumed_align=16)

        def fake3(dtype, shape):
            return cute.runtime.make_fake_compact_tensor(
                dtype, shape, stride_order=(2, 1, 0), assumed_align=4)

        q_fake = fake4(cutlass.Int8, (b, s_q, n_q, head_dim))
        k_fake = fake4(cutlass.Int8, (b, s_k, n_kv, head_dim))
        # fp8 走 uint8 view workaround（interface.py:988），__call__ 内 recast 回 e4m3
        v_fake = fake4(cutlass.Uint8, (b, n_kv, head_dim, s_pad))
        o_fake = fake4(kernel.o_dtype, (b, s_q, n_q, head_dim))
        qs_fake = fake3(cutlass.Float32, (b, n_q, n_wb_q))     # ceil(s_q/64)*4
        ks_fake = fake3(cutlass.Float32, (b, n_kv, n_blk_k))   # ceil(s_k/128)
        vs_fake = fake3(cutlass.Float32, (b, n_kv, head_dim))

        kernel.compiled_kernel = cute.compile(          # interface.py:980 同款
            kernel,
            q_fake, k_fake, v_fake, qs_fake, ks_fake, vs_fake, o_fake,
            0.0,                                        # scale_softmax_log2（运行时标量）
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),  # runtime.py:712
            options="--enable-tvm-ffi",
        )
        cls.compile_cache[key] = kernel
        return kernel

    def run(self, q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale, o, sm_scale):
        if self.compiled_kernel is None:
            raise RuntimeError("SageAttnSm90 must be created via from_args before running")
        if self.is_causal:
            # causal mask 为 top-left 对齐且 KV 上界按 q 位置截断，s_q != s_k 时结果静默错误
            assert q_int8.shape[1] == k_int8.shape[1], "causal requires s_q == s_k"
        # TVM-FFI 编译产物直接吃 torch.Tensor，stream 取调用方当前 torch stream
        self.compiled_kernel(
            q_int8, k_int8, v_fp8.view(torch.uint8),    # interface.py:988 fp8 workaround
            q_scale, k_scale, v_scale, o,
            sm_scale * LOG2_E,
        )

    # ---------------- host jit：TiledMma / TMA / smem / launch（fmha.py:204-475）----------------
    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,          # (b, s_q, n_q, d)  int8
        k: cute.Tensor,          # (b, s_k, n_kv, d) int8
        v: cute.Tensor,          # (b, n_kv, d, s_pad) uint8(=e4m3)
        q_scale: cute.Tensor,    # (b, n_q, ceil(s_q/64)*4) f32
        k_scale: cute.Tensor,    # (b, n_kv, ceil(s_k/128)) f32
        v_scale: cute.Tensor,    # (b, n_kv, d) f32
        o: cute.Tensor,          # (b, s_q, n_q, d) fp16/bf16
        scale_softmax_log2: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        # uint8 → e4m3（recast_ptr 支持 dtype 重解释，core.py:3080）
        v = cute.make_tensor(
            cute.recast_ptr(v.iterator, dtype=self.v_dtype), v.layout
        )

        # 维度重排（对齐 fmha.py:226-253 的 (s, d, h, b) 约定）
        def _permute(t, order):
            return cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    tuple(t.shape[i] for i in order),
                    stride=tuple(t.stride[i] for i in order),
                ),
            )

        mQ = _permute(q, (1, 3, 2, 0))   # (s_q, d, n_q, b)
        mK = _permute(k, (1, 3, 2, 0))   # (s_k, d, n_kv, b)
        mV = _permute(v, (2, 3, 1, 0))   # (d, s_pad, n_kv, b)，s 连续 → B K-major
        mO = _permute(o, (1, 3, 2, 0))   # (s_q, d, n_q, b)

        self.q_layout = utils.LayoutEnum.from_tensor(mQ)    # fmha.py:290
        self.k_layout = utils.LayoutEnum.from_tensor(mK)
        self.v_layout = utils.LayoutEnum.from_tensor(mV)
        self.o_layout = utils.LayoutEnum.from_tensor(mO)

        self.epi_tile = sm90_utils.compute_tile_shape_or_override(   # fmha.py:286
            self.cta_tiler, self.o_dtype
        )

        # QK: s8s8s32 SS（MmaI8Op, mma.py:408；fmha.py:300）
        qk_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.q_dtype, self.k_dtype,
            self.q_layout.sm90_mma_major_mode(),
            self.k_layout.sm90_mma_major_mode(),
            self.qk_acc_dtype, self.atom_layout_mnk, self.qk_mma_tiler[:2],
        )
        # PV: e4m3 RS，P 在 RMEM（MmaF8Op, mma.py:322；fmha.py:310）
        pv_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.v_dtype, self.v_dtype,
            cute.nvgpu.OperandMajorMode.K,   # 4.6.0：warpgroup.OperandMajorMode 已弃用
            self.v_layout.sm90_mma_major_mode(),
            self.pv_acc_dtype, self.atom_layout_mnk, self.pv_mma_tiler[:2],
            warpgroup.OperandSource.RMEM,
        )

        # smem 布局（fmha.py:327-357；d=128→SW128 / d=64→SW64，V s 连续→SW128）
        q_smem_layout_staged = sm90_utils.make_smem_layout_a(
            self.q_layout, self.qk_mma_tiler, self.q_dtype, self.q_stage)
        k_smem_layout_staged = sm90_utils.make_smem_layout_b(
            self.k_layout, self.qk_mma_tiler, self.k_dtype, self.kv_stage)
        v_smem_layout_staged = sm90_utils.make_smem_layout_b(
            self.v_layout, self.pv_mma_tiler, self.v_dtype, self.kv_stage)
        o_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.epi_stage,
            cute.append(cute.append(self.epi_tile, self.epi_stage), 1),
            smem_order=(1, 0, 2, 3) if self.o_layout.is_m_major_c() else (0, 1, 2, 3),
        )

        # TMA atoms（fmha.py:359-401）
        tma_atom_q, tma_tensor_q = self._make_tma_atoms_and_tensors(
            mQ, q_smem_layout_staged,
            (self.qk_mma_tiler[0], self.qk_mma_tiler[2]), 1)
        tma_atom_k, tma_tensor_k = self._make_tma_atoms_and_tensors(
            mK, k_smem_layout_staged,
            (self.qk_mma_tiler[1], self.qk_mma_tiler[2]), 1)
        tma_atom_v, tma_tensor_v = self._make_tma_atoms_and_tensors(
            mV, v_smem_layout_staged,
            (self.pv_mma_tiler[1], self.pv_mma_tiler[2]), 1)

        o_smem_layout = cute.slice_(o_smem_layout_staged, (None, None, 0, 0))
        tma_atom_o, tma_tensor_o = cpasync.make_tiled_tma_atom(   # fmha.py:396
            cpasync.CopyBulkTensorTileS2GOp(), mO, o_smem_layout, self.epi_tile)

        q_smem_layout = cute.slice_(q_smem_layout_staged, (None, None, 0))
        # K/V 块字节数相等（128*d），共用 tx_count（fmha.py:403-406）
        self.tma_copy_q_bytes = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        self.tma_copy_kv_bytes = cute.size_in_bytes(
            self.k_dtype, cute.slice_(k_smem_layout_staged, (None, None, 0)))

        @cute.struct
        class SharedStorage:                              # fmha.py:414
            load_q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            load_kv_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.kv_stage * 2]
            sVScale: cute.struct.MemRange[cutlass.Float32, self.head_dim]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, cute.cosize(o_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(k_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # grid = (ceil(s_q/64), b, n_q)
        grid = (cute.ceil_div(mQ.shape[0], self.CTA_Q), mQ.shape[3], mQ.shape[2])

        self.kernel(                                       # fmha.py:446-475
            qk_tiled_mma, pv_tiled_mma,
            tma_atom_q, tma_tensor_q,
            tma_atom_k, tma_tensor_k,
            tma_atom_v, tma_tensor_v,
            tma_atom_o, tma_tensor_o,
            q_scale, k_scale, v_scale,
            scale_softmax_log2,
            q_smem_layout_staged, k_smem_layout_staged,
            v_smem_layout_staged, o_smem_layout_staged,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            # P2a：nvvm.minctasm=2（ptxas 按 2 CTA/SM 限制 launch 期寄存器 ≤128/线程，
            # 运行期由 warpgroup_reg_alloc/dealloc 重分配 224/24）+ smem carveout 提示
            min_blocks_per_mp=2,
        )

    # ---------------- device kernel ----------------
    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom, mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom, mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom, mV_dkl: cute.Tensor,
        tma_atom_o: cute.CopyAtom, mO_qdl: cute.Tensor,
        mQScale: cute.Tensor,
        mKScale: cute.Tensor,
        mVScale: cute.Tensor,
        scale_softmax_log2: cutlass.Float32,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx_q, b_idx, head_idx = cute.arch.block_idx()
        kv_head = head_idx // self.gqa_ratio            # GQA

        smem = utils.SmemAllocator()                    # fmha.py:570
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = self.make_and_init_load_q_pipeline(
            storage.load_q_mbar_ptr.data_ptr())
        load_kv_producer, load_kv_consumer = self.make_and_init_load_kv_pipeline(
            storage.load_kv_mbar_ptr.data_ptr())
        tma_store_pipeline = self.make_and_init_tma_store_pipeline()

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group)

        # smem 张量；V 与 K 共用同一缓冲、按 stage 交替（fmha.py:593-602）
        sQ = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner)
        sK = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner)
        sV = cute.make_tensor(
            cute.recast_ptr(sK.iterator, v_smem_layout_staged.inner, self.v_dtype),
            v_smem_layout_staged.outer)
        sO = storage.sO.get_tensor(
            o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner)
        sVScale = storage.sVScale.get_tensor(cute.make_layout(self.head_dim))

        seqlen_q = mQ_qdl.shape[0]
        seqlen_k = mK_kdl.shape[0]

        # KV 迭代次数；causal 上界 ceil((bx+1)*64/128)，且仅最后 1 块需要 mask（64<128 保证）
        k_blk_total = cute.ceil_div(seqlen_k, self.CTA_K)
        if const_expr(self.is_causal):
            trip_count = min(
                cute.ceil_div((bidx_q + 1) * self.CTA_Q, self.CTA_K), k_blk_total)
        else:
            trip_count = k_blk_total

        # TMA partition（fmha.py:615-647）
        gQ_qdl = cute.flat_divide(mQ_qdl, cute.select(self.qk_mma_tiler, mode=[0, 2]))
        qk_thr_mma = qk_tiled_mma.get_slice(tidx)
        tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
        tQsQ, tQgQ_qdl = cpasync.tma_partition(
            tma_atom_q, 0, cute.make_layout(1),
            cute.group_modes(sQ, 0, 2), cute.group_modes(tSgQ_qdl, 0, 3))

        gK_kdl = cute.flat_divide(mK_kdl, cute.select(self.qk_mma_tiler, mode=[1, 2]))
        tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
        tKsK, tKgK_kdl = cpasync.tma_partition(
            tma_atom_k, 0, cute.make_layout(1),
            cute.group_modes(sK, 0, 2), cute.group_modes(tSgK_kdl, 0, 3))

        gV_dkl = cute.flat_divide(mV_dkl, cute.select(self.pv_mma_tiler, mode=[1, 2]))
        pv_thr_mma = pv_tiled_mma.get_slice(tidx)
        tOsV_g = pv_thr_mma.partition_B(gV_dkl)
        tVsV, tVgV_dkl = cpasync.tma_partition(
            tma_atom_v, 0, cute.make_layout(1),
            cute.group_modes(sV, 0, 2), cute.group_modes(tOsV_g, 0, 3))

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk)   # fmha.py:654

        if warp_idx == 0:                                # fmha.py:657
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)
            cpasync.prefetch_descriptor(tma_atom_o)

        # ---------------- producer：load warpgroup ----------------
        if warp_group_idx == self.load_warp_group_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_load)
            if warp_idx == 0:
                # Q 单 stage 一次装载
                q_handle = load_q_producer.acquire_and_advance()
                tQgQ = tQgQ_qdl[(None, None, 0, head_idx, b_idx)]
                cute.copy(tma_atom_q, tQgQ[(None, bidx_q)],
                          tQsQ[(None, q_handle.index)],
                          tma_bar_ptr=q_handle.barrier)       # fmha.py:705-711

                tKgK = tKgK_kdl[(None, None, 0, kv_head, b_idx)]
                tVgV = tVgV_dkl[(None, 0, None, kv_head, b_idx)]
                kv_iter = Int32(0)
                while kv_iter < trip_count:                   # K/V 交替入队
                    k_handle = load_kv_producer.acquire_and_advance()
                    cute.copy(tma_atom_k, tKgK[(None, kv_iter)],
                              tKsK[(None, k_handle.index)],
                              tma_bar_ptr=k_handle.barrier)
                    v_handle = load_kv_producer.acquire_and_advance()
                    cute.copy(tma_atom_v, tVgV[(None, kv_iter)],
                              tVsV[(None, v_handle.index)],
                              tma_bar_ptr=v_handle.barrier)
                    kv_iter += 1

        # ---------------- consumer：math warpgroup ----------------
        if warp_group_idx == self.math_warp_group_id:
            cute.arch.warpgroup_reg_alloc(self.num_regs_mma)

            # per-warp q_scale：idx = ((b*n_q+h)*ceil(s_q/64) + bx)*4 + warp（sm90.cu:177）；
            # acc fragment 行 = 16*warp_local + lane/4 (+8) 恰在该 warp 的 16 行段内，每 warp 一个标量
            warp_local = warp_idx - self.math_warp_group_id * self.num_warps_per_warp_group
            q_scale_val = mQScale[(b_idx, head_idx, bidx_q * 4 + warp_local)]

            # v_scale 预存 smem，epilogue 按列取用；448 已在 row_sum448 分母中相消（P1.2）
            t_local = tidx - self.num_threads_per_warp_group
            if t_local < self.head_dim:
                sVScale[t_local] = mVScale[(b_idx, kv_head, t_local)]
            pipeline.arrive_and_wait(                          # named barrier（fmha.py:1171）
                barrier_id=self.math_warp_group_id,
                num_threads=self.num_threads_per_warp_group)

            # mainloop 前置（fmha.py:813-845）
            tSsQ = qk_thr_mma.partition_A(sQ)
            tSsK = qk_thr_mma.partition_B(sK)
            tSrQ = qk_thr_mma.make_fragment_A(tSsQ)
            tSrK = qk_thr_mma.make_fragment_B(tSsK)
            tOsV = pv_thr_mma.partition_B(sV)
            tOrV = pv_thr_mma.make_fragment_B(tOsV)

            q_handle = load_q_consumer.wait()                  # fmha.py:826

            # 全局 (q_idx, k_idx) 坐标（fmha.py:796/829 identity-tensor 技术）
            cP = cute.make_identity_tensor((seqlen_q, seqlen_k))
            gPcP = cute.local_tile(cP, self.qk_mma_tiler[:2], (None, None))
            ptScP = qk_thr_mma.partition_C(gPcP)

            pv_acc_shape = pv_thr_mma.partition_shape_C(
                (self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
            acc_pv = pv_thr_mma.make_fragment_C(pv_acc_shape)
            acc_pv.fill(0.0)
            qk_acc_shape = qk_thr_mma.partition_shape_C(
                (self.qk_mma_tiler[0], self.qk_mma_tiler[1]))

            s_max_layout = cute.make_layout(
                cute.size(self.layout_acc_mn(pv_tiled_mma, acc_pv.layout), mode=[0]))
            s_max = cute.make_rmem_tensor_like(s_max_layout, Float32)
            a_sum = cute.make_rmem_tensor_like(s_max, Float32)
            s_max.fill(-Float32.inf)     # 统一初值 → 无 is_first 特判：
            a_sum.fill(0.0)              # 首次 rescale 因子 exp2(-inf)=0，acc_pv 已清零

            # 前 trip_count-1 块不 mask，最后 1 块做 residual(+causal) mask
            kv_iter = Int32(0)
            load_kv_consumer, kv_iter, s_max, a_sum = self.compute(
                False, trip_count - 1, kv_iter,
                qk_thr_mma, qk_tiled_mma, pv_tiled_mma, load_kv_consumer,
                q_handle, tSrQ, tSrK, tOrV, acc_pv, s_max, a_sum,
                ptScP, bidx_q, qk_acc_shape,
                q_scale_val, mKScale, b_idx, kv_head,
                scale_softmax_log2, seqlen_k)
            load_kv_consumer, kv_iter, s_max, a_sum = self.compute(
                True, Int32(1), kv_iter,
                qk_thr_mma, qk_tiled_mma, pv_tiled_mma, load_kv_consumer,
                q_handle, tSrQ, tSrK, tOrV, acc_pv, s_max, a_sum,
                ptScP, bidx_q, qk_acc_shape,
                q_scale_val, mKScale, b_idx, kv_head,
                scale_softmax_log2, seqlen_k)

            cute.nvgpu.warpgroup.wait_group(0)
            q_handle.release()                                 # fmha.py:1037

            # ---- tail：行和归约 + O = acc · v_scale / row_sum448（fmha.py:1503-1561 改造）
            # acc = Σ P448_q·V_q，row_sum448 = 448·ΣP（量化前），448 相消 ----
            reduction_target_pv = self.reduction_target_n(pv_tiled_mma)
            for r in cutlass.range_constexpr(cute.rank(reduction_target_pv)):
                for i in cutlass.range_constexpr(cute.size(a_sum)):
                    a_sum[i] = cute.arch.warp_reduction_sum(
                        a_sum[i], threads_in_group=reduction_target_pv.shape[r])

            acc_mn = cute.make_tensor(
                acc_pv.iterator, self.layout_acc_mn(pv_tiled_mma, acc_pv.layout))
            cO = cute.make_identity_tensor(
                (self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
            tOcO = pv_thr_mma.partition_C(cO)
            tOcO_mn = cute.make_tensor(
                tOcO.iterator, self.layout_acc_mn(pv_tiled_mma, tOcO.layout))
            for i in cutlass.range_constexpr(cute.size(acc_mn, mode=[0])):
                inv_sum = cute.arch.rcp_approx(a_sum[i])       # fmha.py:1548
                if a_sum[i] == 0.0 or a_sum[i] != a_sum[i]:
                    inv_sum = 1.0
                for j in cutlass.range_constexpr(cute.size(acc_mn, mode=[1])):
                    col = tOcO_mn[(i, j)][1]                   # per-channel v_scale 列索引
                    acc_mn[(i, j)] = acc_mn[(i, j)] * inv_sum * sVScale[col]

            # ---- epilogue：r2s(StMatrix) + TMA store（fmha.py:1092-1190，单 math WG）----
            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                self.o_layout, elem_ty_d=self.o_dtype, elem_ty_acc=self.pv_acc_dtype)
            copy_atom_O = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(self.o_layout.is_m_major_c(), 4),
                self.o_dtype)
            tiled_copy_O_Atom = cute.make_tiled_copy_C_atom(copy_atom_O, pv_tiled_mma)
            tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_O_Atom)
            thr_copy_r2s = tiled_copy_r2s.get_slice(
                tidx % self.num_threads_per_warp_group)
            tRS_sD = thr_copy_r2s.partition_D(sO)
            tRS_rAcc = tiled_copy_r2s.retile(acc_pv)

            rD_shape = cute.shape(thr_copy_r2s.partition_S(sO))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor_like(tRS_rD_layout, self.pv_acc_dtype)
            size_tRS_rD = cute.size(tRS_rD)

            gD = cute.local_tile(mO_qdl, self.pv_mma_tiler[:2],
                                 (bidx_q, 0, head_idx, b_idx))   # fmha.py:1130
            sepi_for_tma_partition = cute.group_modes(sO, 0, 2)
            tcgc_for_tma_partition = cute.zipped_divide(gD, self.epi_tile)
            bSG_sD, bSG_gD = cpasync.tma_partition(
                tma_atom_o, 0, cute.make_layout(1),
                sepi_for_tma_partition, tcgc_for_tma_partition)
            epi_tile_num = cute.size(tcgc_for_tma_partition, mode=[1])

            for epi_idx in cutlass.range_constexpr(epi_tile_num):
                for epi_v in cutlass.range_constexpr(size_tRS_rD):
                    tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]
                tRS_rD_out = cute.make_rmem_tensor_like(tRS_rD_layout, self.o_dtype)
                acc_vec = tRS_rD.load()
                tRS_rD_out.store(acc_vec.to(self.o_dtype))
                epi_buffer = epi_idx % self.epi_stage
                cute.copy(tiled_copy_r2s, tRS_rD_out,
                          tRS_sD[(None, None, None, epi_buffer, 0)])
                # 4.6.0 的 fence_proxy 只接受字符串字面量（nvvm_wrappers.py:1038），
                # 语义同底本的 ProxyKind.async_shared + SharedSpace.shared_cta
                cute.arch.fence_proxy("async.shared", space="cta")
                pipeline.arrive_and_wait(
                    barrier_id=self.math_warp_group_id,
                    num_threads=self.num_threads_per_warp_group)
                if warp_idx == 4:            # math WG 首 warp 发 TMA store
                    cute.copy(tma_atom_o, bSG_sD[(None, epi_buffer, 0)],
                              bSG_gD[(None, epi_idx)])
                    tma_store_pipeline.producer_commit()
                    tma_store_pipeline.producer_acquire()
                pipeline.arrive_and_wait(
                    barrier_id=self.math_warp_group_id,
                    num_threads=self.num_threads_per_warp_group)
        return

    # ---------------- mainloop 迭代（fmha.py:1199-1299 改造：dequant + 运行时 scale）----------------
    @cute.jit
    def compute(
        self,
        masked: cutlass.Constexpr,
        k_tile_count: Int32,
        kv_iter: Int32,
        qk_thr_mma: cute.ThrMma,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        load_kv_consumer: pipeline.PipelineConsumer,
        q_handle,
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        tOrV: cute.Tensor,
        acc_pv: cute.Tensor,
        s_max: cute.Tensor,
        a_sum: cute.Tensor,
        ptScP: cute.Tensor,
        bidx_q: Int32,
        qk_acc_shape: cute.Shape,
        q_scale_val: Float32,
        mKScale: cute.Tensor,
        b_idx: Int32,
        kv_head: Int32,
        scale_softmax_log2: Float32,
        seqlen_k: Int32,
    ) -> Tuple[pipeline.PipelineConsumer, Int32, cute.Tensor, cute.Tensor]:
        while k_tile_count > 0:
            k_tile_count -= 1
            tScP = cute.slice_(ptScP, (None, None, None, bidx_q, kv_iter))  # fmha.py:1233
            # per-block k_scale：idx = (b*n_kv + kv_head)*ceil(s_k/128) + kv_iter（sm90.cu:188）
            k_scale_val = mKScale[(b_idx, kv_head, kv_iter)]
            kv_iter += 1

            acc_qk = qk_thr_mma.make_fragment_C(qk_acc_shape)   # Int32 acc
            k_handle = load_kv_consumer.wait_and_advance()

            cute.nvgpu.warpgroup.fence()                        # fmha.py:1242
            self.gemm_zero_acc(
                qk_tiled_mma,
                tSrQ[(None, None, None, q_handle.index)],
                tSrK[(None, None, None, k_handle.index)],
                acc_qk)
            cute.nvgpu.warpgroup.commit_group()
            tok = load_kv_consumer.try_wait()                   # fmha.py:1253
            cute.nvgpu.warpgroup.wait_group(0)

            # dequant：S_f32 = f32(S_i32) · q_scale(per-warp) · k_scale(per-block)
            # TensorSSA.to 支持 int→float（tensor.py:1772 sitofp），Int32→Float32 无损
            acc_s = cute.make_rmem_tensor_like(acc_qk, Float32)
            acc_s.store(acc_qk.load().to(Float32))
            dequant = q_scale_val * k_scale_val
            for i in cutlass.range_constexpr(cute.size(acc_s)):
                acc_s[i] = acc_s[i] * dequant

            if const_expr(masked):
                # residual mask（k 尾块）+ causal 对角 mask，均在 dequant 后 S_f32 上做
                for i in cutlass.range_constexpr(cute.size(acc_s)):
                    q_pos = tScP[i][0]
                    k_pos = tScP[i][1]
                    if k_pos >= seqlen_k:
                        acc_s[i] = -Float32.inf
                    if const_expr(self.is_causal):
                        if k_pos > q_pos:
                            acc_s[i] = -Float32.inf

            s_max, a_sum = self.softmax_step(
                acc_s, qk_tiled_mma, s_max, a_sum, acc_pv, pv_tiled_mma,
                scale_softmax_log2)

            # acc_s 已是 P448 = 448·P ∈ [0,448]（softmax_step 内 exp2 常量偏移），直接转 e4m3
            acc_p_op = self.make_acc_into_op(
                acc_s, pv_tiled_mma.tv_layout_A, self.v_dtype)  # fmha.py:1275
            v_handle = load_kv_consumer.wait_and_advance(tok)

            cute.nvgpu.warpgroup.fence()                        # fmha.py:1282
            pv_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            cute.gemm(pv_tiled_mma, acc_pv, acc_p_op,
                      tOrV[(None, None, None, v_handle.index)], acc_pv)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

            k_handle.release()
            v_handle.release()
        return load_kv_consumer, kv_iter, s_max, a_sum

    # ---------------- online softmax（fmha.py:1301-1397 改造：m 统一初值 -inf）----------------
    @cute.jit
    def softmax_step(
        self,
        acc_s: cute.Tensor,
        qk_tiled_mma: cute.TiledMma,
        s_max: cute.Tensor,
        a_sum: cute.Tensor,
        acc_pv: cute.Tensor,
        pv_tiled_mma: cute.TiledMma,
        scale_softmax_log2: Float32,
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        acc_s_mn = cute.make_tensor(
            acc_s.iterator, self.layout_acc_mn(qk_tiled_mma, acc_s.layout))
        acc_pv_mn = cute.make_tensor(
            acc_pv.iterator, self.layout_acc_mn(pv_tiled_mma, acc_pv.layout))
        reduction_target_qk = self.reduction_target_n(qk_tiled_mma)
        red_rank = cute.rank(reduction_target_qk)
        s_max_prev = cute.make_rmem_tensor_like(s_max, Float32)

        for i in cutlass.range_constexpr(cute.size(acc_s_mn, mode=[0])):
            s_max_prev[i] = s_max[i]
            for j in cutlass.range_constexpr(cute.size(acc_s_mn, mode=[1])):
                # cute.arch.fmax → FMNMX（cutlass.max 会编成 NaN 语义 FSETP+FSEL）；
                # S 在 mask 后只含普通值与 -inf、无 NaN，语义安全
                s_max[i] = cute.arch.fmax(s_max[i], acc_s_mn[(i, j)])
            for r in cutlass.range_constexpr(red_rank):        # quad reduction
                s_max[i] = cute.arch.warp_reduction_max(
                    s_max[i], threads_in_group=reduction_target_qk.shape[r])

            local_max = s_max[i]
            if s_max[i] == -Float32.inf:      # 整行被 mask 时避免 -inf 参与运算
                local_max = 0.0
            # ×448 折进 exp2（P1.2）：arg = c·S − (c·m − log2 448)，exp2 直接得 P448=448·P，
            # 消掉独立的 ×448 FMUL 循环；每元素仍是 1 条 FFMA + 1 条 MUFU.EX2
            scale_max = scale_softmax_log2 * local_max - LOG2_448
            for j in cutlass.range_constexpr(cute.size(acc_s_mn, mode=[1])):
                acc_s_mn[(i, j)] = cute.math.exp2(
                    scale_softmax_log2 * acc_s_mn[(i, j)] - scale_max, fastmath=True)

            # 首迭代 s_max_prev=-inf → scale_pv=0，旧 acc（已清零）与旧 sum 均被清除；
            # m 变化补偿项 scale_pv 不带 448 偏移（两次 448 在分子分母中相消）
            scale_pv = cute.math.exp2(
                (s_max_prev[i] - local_max) * scale_softmax_log2, fastmath=True)
            # a_sum 从此累加 P448：row_sum448 = 448·ΣP，与 epilogue O = acc·v_scale/row_sum448
            # 配对（acc 累加的也是 P448·V），448 代数上完全相消
            a_sum[i] = a_sum[i] * scale_pv + acc_s_mn[(i, None)].load().reduce(
                cute.ReductionOp.ADD, Float32.zero, 0)         # fmha.py:1393
            for j in cutlass.range_constexpr(cute.size(acc_pv_mn, mode=[1])):
                acc_pv_mn[(i, j)] *= scale_pv

        return s_max, a_sum

    # ---------------- 以下工具方法从 fmha.py 照抄 ----------------
    @cute.jit
    def reduction_target_n(self, tiled_mma):                   # fmha.py:1399
        separated = self.layout_separate(
            tiled_mma.shape_mnk[0],
            cute.make_layout(tiled_mma.tv_layout_C.shape[0]),
            tiled_mma.tv_layout_C.stride[0],
        )
        return separated[1]

    @staticmethod
    def convert_c_layout_to_a_layout(c, a):                    # fmha.py:1408
        return cute.make_layout(
            (a, c.shape[1], (c.shape[2], cute.size(c, mode=[0]) // cute.size(a))),
            stride=(
                c.stride[0],
                c.stride[1],
                (c.stride[2], cute.size(a, mode=[2]) * c.stride[0][2]),
            ),
        )

    @cute.jit
    def make_acc_into_op(self, acc, operand_layout_tv, Element):  # fmha.py:1419-1500
        operand = cute.make_rmem_tensor_like(
            self.convert_c_layout_to_a_layout(acc.layout, operand_layout_tv.shape[1]),
            Element,
        )
        operand_as_acc = cute.make_tensor(operand.iterator, acc.layout)
        acc_vec = acc.load()
        operand_as_acc.store(acc_vec.to(Element))

        if cutlass.const_expr(Element.width == 8):
            # 8-bit A fragment 的 warp 内重排（shuffle_sync + prmt），逐字照抄底本
            tidx, _, _ = cute.arch.thread_idx()
            tid = tidx % 4
            values_u32 = cute.recast_tensor(operand, cutlass.Uint32)
            for n in cutlass.range_constexpr(cute.size(values_u32, mode=[1])):
                for k in cutlass.range_constexpr(cute.size(values_u32, mode=[2])):
                    for ii in cutlass.range_constexpr(0, 8, 4):
                        values_tmp_0 = values_u32[ii // 2 + 0, n, k]
                        values_tmp_1 = values_u32[ii // 2 + 1, n, k]

                        v_to_send = 1
                        if tid == 1 or tid == 2:
                            v_to_send = 0
                        t_to_recv_from = (0x3021 >> (tid * 4)) & 0xF
                        values_tmp_a = values_tmp_1
                        if v_to_send == 0:
                            values_tmp_a = values_tmp_0
                        values_tmp_a = cute.arch.shuffle_sync_op(
                            values_tmp_a, t_to_recv_from, 0xFFFFFFFF, 7199)

                        v_to_send = 1 - v_to_send
                        t_to_recv_from = (0x2130 >> (tid * 4)) & 0xF
                        values_tmp_b = values_tmp_1
                        if v_to_send == 0:
                            values_tmp_b = values_tmp_0
                        values_tmp_b = cute.arch.shuffle_sync_op(
                            values_tmp_b, t_to_recv_from, 0xFFFFFFFF, 7199)

                        order = 0x5410
                        if v_to_send == 0:
                            order = 0x1054
                        values_u32[ii // 2 + 0, n, k] = cute.arch.prmt(
                            values_tmp_a, values_tmp_b, order)
                        order = 0x7632
                        if v_to_send == 0:
                            order = 0x3276
                        values_u32[ii // 2 + 1, n, k] = cute.arch.prmt(
                            values_tmp_a, values_tmp_b, order)
        return operand

    @staticmethod
    def layout_separate(thr, src, ref):                        # fmha.py:1563
        lt = cute.make_layout(())
        ge = cute.make_layout(())
        for k, v in enumerate(ref):
            if cutlass.const_expr(v < thr):
                lt = cute.append(lt, src[k])
            else:
                ge = cute.append(ge, src[k])
        r = None
        if cutlass.const_expr(cute.rank(lt) == 1):
            r = cute.append(lt, ge)
        else:
            r = cute.append(cute.append(cute.make_layout(()), lt), ge)
        return r

    @staticmethod
    @cute.jit
    def gemm_zero_acc(tiled_mma, A, B, C):                     # fmha.py:1581
        rA = cute.rank(A)
        rB = cute.rank(B)
        rC = cute.rank(C)
        if cutlass.const_expr(rA == 2 and rB == 2 and rC == 1):
            for k_block_idx in range(cute.size(A, mode=[1]), unroll_full=True):
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, k_block_idx != 0)
                cute.gemm(tiled_mma, C, A[None, k_block_idx], B[None, k_block_idx], C)
        elif cutlass.const_expr(rA == 3 and rB == 3 and rC == 3):
            for k_block_idx in range(cute.size(A, mode=[2]), unroll_full=True):
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, k_block_idx != 0)
                cute.gemm(tiled_mma, C, A[None, None, k_block_idx],
                          B[None, None, k_block_idx], C)
        else:
            assert 0

    @cute.jit
    def layout_acc_mn(self, tiled_mma, acc):                   # fmha.py:1610
        separated = self.layout_separate(
            tiled_mma.shape_mnk[0], acc[0], tiled_mma.tv_layout_C.stride[1])
        V_M = separated[0]
        V_N = separated[1]
        V_M1 = None
        V_N1 = None
        if cutlass.const_expr(cute.rank(V_M) == 1):
            V_M1 = cute.append(V_M, acc[1])
        else:
            V_M1 = cute.append(cute.append(cute.make_layout(()), V_M), acc[1])
        if cutlass.const_expr(cute.rank(V_N) == 1):
            V_N1 = cute.append(V_N, acc[2])
        else:
            V_N1 = cute.append(cute.append(cute.make_layout(()), V_N), acc[2])
        r = None
        if cutlass.const_expr(cute.rank(V_M1) == 1):
            r = cute.append(V_M1, V_N1)
        else:
            r = cute.append(cute.append(cute.make_layout(()), V_M1), V_N1)
        return r

    # ---------------- pipeline 构造（fmha.py:1636-1680；单 math WG）----------------
    def make_and_init_load_q_pipeline(self, load_q_mbar_ptr):
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_warps_per_warp_group)
        return pipeline.PipelineTmaAsync.create(
            barrier_storage=load_q_mbar_ptr,
            num_stages=self.q_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.tma_copy_q_bytes,
            defer_sync=True,
        ).make_participants()

    def make_and_init_load_kv_pipeline(self, load_kv_mbar_ptr):
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_warps_per_warp_group)
        return pipeline.PipelineTmaAsync.create(
            barrier_storage=load_kv_mbar_ptr,
            num_stages=self.kv_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.tma_copy_kv_bytes,
            defer_sync=True,
        ).make_participants()

    def make_and_init_tma_store_pipeline(self):                # fmha.py:1672
        tma_store_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        return pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=tma_store_producer_group,
        )

    @staticmethod
    def _make_tma_atoms_and_tensors(tensor, smem_layout_staged, smem_tile,
                                    mcast_dim):                # fmha.py:1696-1731
        op = (
            cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op, tensor, smem_layout, smem_tile, num_multicast=mcast_dim)
        return tma_atom, tma_tensor


# ===== 端到端入口 =====
def sageattn_qk_int8_pv_fp8_hopper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
) -> torch.Tensor:
    """SageAttention forward（Hopper，QK int8 per-warp + PV fp8 per-channel）。NHD 布局。"""
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    assert q.dtype in (torch.float16, torch.bfloat16) and q.dtype == k.dtype == v.dtype
    b, s_q, n_q, head_dim = q.shape
    _, s_k, n_kv, _ = k.shape
    assert head_dim in (64, 128)
    assert k.shape[-1] == head_dim and v.shape[-1] == head_dim
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    assert n_q % n_kv == 0
    if is_causal:
        assert s_q == s_k, "causal requires s_q == s_k"
    if sm_scale is None:
        sm_scale = head_dim ** -0.5

    km = k.mean(dim=1, keepdim=True) if smooth_k else None
    q_int8, q_scale = quant_q_int8_per_warp(q)
    k_int8, k_scale = quant_k_int8_per_block(k, km)
    v_fp8, v_scale = quant_v_fp8_per_channel(v)

    o = torch.empty(q.shape, dtype=q.dtype, device=q.device)

    kernel = SageAttnSm90.from_args(head_dim, is_causal, q.dtype, n_q // n_kv)
    kernel.run(q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale, o, sm_scale)
    return o
