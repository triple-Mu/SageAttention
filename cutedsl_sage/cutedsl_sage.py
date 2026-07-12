import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from typing import Tuple


class CutedslKernel:
    compile_cache: dict[Tuple[object, ...], "CutedslKernel"] = {}

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        self.compiled_kernel = None

    @classmethod
    def from_args(
            cls,
            *args,
            **kwargs,
    ) -> "CutedslKernel":

        key = ...  # from args
        cached = cls.compile_cache.get(key)
        if cached is not None:
            return cached

        kernel = cls(*args, **kwargs)

        '''
        make_fake_tensor
        '''

        # 修复：位置参数（fake stream）必须在 **kwargs 解包之前
        kernel.compiled_kernel = cute.compile(
            kernel,
            *args,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
            **kwargs,
        )
        cls.compile_cache[key] = kernel
        return kernel

    def run(
            self,
            *args,
            **kwargs,
    ) -> None:
        if self.compiled_kernel is None:
            raise RuntimeError("CutedslKernel must be created via from_args before running")
        self.compiled_kernel(*args, **kwargs)

    @cute.jit
    def __call__(
            self,
            *args,
            stream: cuda.CUstream,
            **kwargs,
    ):
        ...

    @cute.kernel
    def kernel(
            self,
            *args,
            **kwargs,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        s_idx, b_idx, head_block_idx = cute.arch.block_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
