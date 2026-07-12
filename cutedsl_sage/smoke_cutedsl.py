"""CuTe-DSL 环境冒烟。在 H200 容器内运行：python smoke_cutedsl.py"""
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute


class AddOne:
    def __init__(self):
        self.compiled = None

    @cute.jit
    def __call__(self, x: cute.Tensor, stream: cuda.CUstream):
        self.kernel(x).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx < cute.size(x.shape):
            x[tidx] = x[tidx] + 1.0


def check_apis():
    import cutlass.utils.hopper_helpers as sm90_utils
    import cutlass.pipeline as pipeline
    from cutlass.cute.nvgpu import cpasync, warpgroup
    checks = {
        "cute.runtime.make_fake_compact_tensor": hasattr(cute.runtime, "make_fake_compact_tensor"),
        "cute.runtime.make_fake_stream": hasattr(cute.runtime, "make_fake_stream"),
        "cute.sym_int": hasattr(cute, "sym_int"),
        "sm90_utils.make_trivial_tiled_mma": hasattr(sm90_utils, "make_trivial_tiled_mma"),
        "sm90_utils.make_smem_layout_a/b/epi": all(
            hasattr(sm90_utils, f"make_smem_layout_{s}") for s in ("a", "b", "epi")),
        "sm90_utils.compute_tile_shape_or_override": hasattr(sm90_utils, "compute_tile_shape_or_override"),
        "sm90_utils.sm90_get_smem_store_op": hasattr(sm90_utils, "sm90_get_smem_store_op"),
        "pipeline.PipelineTmaAsync": hasattr(pipeline, "PipelineTmaAsync"),
        "pipeline.PipelineTmaStore": hasattr(pipeline, "PipelineTmaStore"),
        "pipeline.pipeline_init_arrive/wait": hasattr(pipeline, "pipeline_init_arrive")
            and hasattr(pipeline, "pipeline_init_wait"),
        "pipeline.arrive_and_wait": hasattr(pipeline, "arrive_and_wait"),
        "pipeline.CooperativeGroup/Agent": hasattr(pipeline, "CooperativeGroup") and hasattr(pipeline, "Agent"),
        "cpasync.CopyBulkTensorTileG2SOp": hasattr(cpasync, "CopyBulkTensorTileG2SOp"),
        "cpasync.CopyBulkTensorTileS2GOp": hasattr(cpasync, "CopyBulkTensorTileS2GOp"),
        "cpasync.make_tiled_tma_atom": hasattr(cpasync, "make_tiled_tma_atom"),
        "cpasync.tma_partition": hasattr(cpasync, "tma_partition"),
        "cpasync.prefetch_descriptor": hasattr(cpasync, "prefetch_descriptor"),
        "warpgroup.MmaI8Op": hasattr(warpgroup, "MmaI8Op"),
        "warpgroup.MmaF8Op": hasattr(warpgroup, "MmaF8Op"),
        "warpgroup.OperandSource/OperandMajorMode": hasattr(warpgroup, "OperandSource")
            and hasattr(warpgroup, "OperandMajorMode"),
        "cute.make_rmem_tensor_like": hasattr(cute, "make_rmem_tensor_like"),
        "cute.recast_ptr/recast_tensor": hasattr(cute, "recast_ptr") and hasattr(cute, "recast_tensor"),
        "cute.arch.warp_reduction_max/sum": hasattr(cute.arch, "warp_reduction_max")
            and hasattr(cute.arch, "warp_reduction_sum"),
        "cute.arch.shuffle_sync_op/prmt": hasattr(cute.arch, "shuffle_sync_op")
            and hasattr(cute.arch, "prmt"),
        "cute.arch.warpgroup_reg_alloc/dealloc": hasattr(cute.arch, "warpgroup_reg_alloc")
            and hasattr(cute.arch, "warpgroup_reg_dealloc"),
        "cute.arch.rcp_approx": hasattr(cute.arch, "rcp_approx"),
        "cute.math.exp2": hasattr(cute.math, "exp2"),
        "cute.nvgpu.warpgroup.fence/commit/wait": hasattr(cute.nvgpu.warpgroup, "fence")
            and hasattr(cute.nvgpu.warpgroup, "commit_group") and hasattr(cute.nvgpu.warpgroup, "wait_group"),
        "cute.nvgpu.warp.StMatrix8x8x16bOp": hasattr(cute.nvgpu.warp, "StMatrix8x8x16bOp"),
    }
    print(f"cutlass version: {cutlass.__version__}")
    missing = [k for k, ok in checks.items() if not ok]
    for k, ok in checks.items():
        print(f"  [{'OK' if ok else 'MISSING'}] {k}")
    return missing


def main():
    missing = check_apis()
    op = AddOne()
    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (128,), stride_order=(0,), assumed_align=4)
    op.compiled = cute.compile(
        op, x_fake,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi")
    x = torch.zeros(128, dtype=torch.float32, device="cuda")
    op.compiled(x)
    torch.cuda.synchronize()
    assert (x == 1).all()
    print("tvm-ffi smoke: OK")
    assert not missing, f"missing APIs: {missing}"


if __name__ == "__main__":
    main()
