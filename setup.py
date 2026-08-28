"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import re
import sys
import subprocess
import threading
import warnings
from packaging.version import parse, Version

from setuptools import setup, find_packages

# Skip CUDA build in CI or when explicitly requested
SKIP_CUDA_BUILD = (
    os.getenv("SAGEATTN_SKIP_CUDA_BUILD", "0").upper() in {"1", "TRUE", "YES"}
    or ("sdist" in sys.argv)
)

ext_modules = []
cmdclass = {}

if not SKIP_CUDA_BUILD:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

    # Compiler flags.
    CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]
    # Arch-independent nvcc flags only; -gencode flags are chosen per extension
    # below and must never be added here.
    NVCC_BASE_FLAGS = [
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--use_fast_math",
        "--threads=8",
        "-Xptxas=-v",
        "-diag-suppress=174",
    ]

    # Append flags from env if provided
    cxx_append = os.getenv("CXX_APPEND_FLAGS", "").strip()
    if cxx_append:
        CXX_FLAGS += cxx_append.split()
    nvcc_append = os.getenv("NVCC_APPEND_FLAGS", "").strip()
    if nvcc_append:
        NVCC_BASE_FLAGS += nvcc_append.split()

    ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
    CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
    NVCC_BASE_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

    if CUDA_HOME is None:
        raise RuntimeError(
            "Cannot find CUDA_HOME. CUDA must be available to build the package.")

    def get_nvcc_cuda_version(cuda_dir: str) -> Version:
        """Get the CUDA version from nvcc.

        Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
        """
        nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                              universal_newlines=True)
        output = nvcc_output.split()
        release_idx = output.index("release") + 1
        nvcc_cuda_version = parse(output[release_idx].split(",")[0])
        return nvcc_cuda_version

    def get_nvcc_gpu_codes(cuda_dir: str) -> set:
        """The plain sm_* code targets this nvcc can emit.

        --list-gpu-code only lists plain codes; arch-specific variants like
        sm_90a follow the availability of their plain base target.
        """
        nvcc_output = subprocess.check_output(
            [cuda_dir + "/bin/nvcc", "--list-gpu-code"], universal_newlines=True)
        return {line.strip() for line in nvcc_output.splitlines() if line.strip()}

    def parse_arch_token(token: str):
        """Normalize one TORCH_CUDA_ARCH_LIST entry to ((major, minor), emit_ptx).

        Accepts '8.0', '80', 'sm_90', 'compute_120', '8.9+PTX', and 'a'/'f'
        suffixed forms (the suffix is ignored with a warning: per-extension
        suffixes are chosen automatically). Returns None for empty tokens and
        raises on anything unparsable instead of silently dropping it.
        """
        it = token.strip().lower().replace("sm_", "").replace("compute_", "")
        if not it:
            return None
        ptx = it.endswith("+ptx")
        if ptx:
            it = it[:-4]
        if it.endswith(("a", "f")):
            warnings.warn(
                f"ignoring {it[-1]!r} suffix on TORCH_CUDA_ARCH_LIST entry "
                f"{token.strip()!r}: per-extension arch suffixes are chosen automatically")
            it = it[:-1]
        if it.isdigit() and len(it) >= 2:
            it = f"{it[:-1]}.{it[-1]}"
        match = re.fullmatch(r"(\d+)\.(\d+)", it)
        if match is None:
            raise ValueError(
                f"Unparsable TORCH_CUDA_ARCH_LIST entry {token.strip()!r} "
                "(expected forms like '8.0', '80', 'sm_90', '12.0+PTX')")
        return (int(match.group(1)), int(match.group(2))), ptx

    # Determine target compute capabilities as {(major, minor): emit_ptx}.
    # Prefer TORCH_CUDA_ARCH_LIST if explicitly specified (works without GPUs),
    # otherwise detect from local GPUs.
    requested_ccs = {}
    arch_list_env = os.getenv("TORCH_CUDA_ARCH_LIST", "").strip()
    if arch_list_env:
        for item in arch_list_env.replace(",", ";").split(";"):
            parsed = parse_arch_token(item)
            if parsed is None:
                continue
            cc, ptx = parsed
            if cc < (8, 0):
                raise ValueError(
                    f"Compute capability {cc[0]}.{cc[1]} in TORCH_CUDA_ARCH_LIST is not "
                    "supported: SageAttention requires 8.0 or higher.")
            requested_ccs[cc] = requested_ccs.get(cc, False) or ptx
    else:
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        for i in range(device_count):
            cc = torch.cuda.get_device_capability(i)
            if cc < (8, 0):
                warnings.warn(f"skipping GPU {i} with compute capability {cc[0]}.{cc[1]}")
                continue
            requested_ccs.setdefault(cc, False)

    if not requested_ccs:
        raise RuntimeError(
            "No target compute capabilities. Set TORCH_CUDA_ARCH_LIST or build on a machine with GPUs.")
    print(f"Target compute capabilities: {sorted(requested_ccs)}")

    nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
    if nvcc_cuda_version < Version("12.0"):
        raise RuntimeError("CUDA 12.0 or higher is required to build the package.")
    nvcc_gpu_codes = get_nvcc_gpu_codes(CUDA_HOME)

    # First public toolkit per recent target, for friendlier errors than a bare
    # "not in --list-gpu-code".
    MIN_TOOLKIT_HINTS = {"sm_100": "12.8", "sm_120": "12.8",
                         "sm_103": "12.9", "sm_121": "12.9", "sm_110": "13.0"}

    def check_gpu_code_supported(cc):
        """Ensure this nvcc can emit plain SASS for cc, with actionable errors."""
        code = f"sm_{cc[0]}{cc[1]}"
        if code in nvcc_gpu_codes:
            return
        msg = f"nvcc {nvcc_cuda_version} cannot emit {code} (not in nvcc --list-gpu-code)."
        if code in MIN_TOOLKIT_HINTS:
            msg += f" CUDA {MIN_TOOLKIT_HINTS[code]} or higher is required."
        lower = sorted(c for c in nvcc_gpu_codes
                       if re.fullmatch(rf"sm_{cc[0]}(\d)", c) and int(c[-1]) < cc[1])
        if lower:
            msg += (f" Alternatively build for {lower[-1]}: plain {lower[-1]} cubins "
                    f"run on {code} (same major, lower minor).")
        raise RuntimeError(msg)

    # Per-extension target policy.
    #  - _qattn_sm80 / _fused use only sm_80-baseline instructions (guarded
    #    cp.async / ldmatrix / int8 & f16 mma / fp8 cvt) and are compiled
    #    natively for every requested arch. Plain codes keep same-major
    #    minor-forward compatibility (e.g. sm_120 cubins run on sm_121).
    #  - _qattn_sm89 is built around the plain fp8 mma.sync instruction
    #    ("sm_89 or higher"): native QMMA on sm_89 and 12.x Blackwell,
    #    HMMA-emulated but correct on sm_100-class parts. It must not be built
    #    for 8.0/8.6 (the mma.cuh guards degrade it to runtime traps there),
    #    and Hopper uses the specialized wgmma kernels instead.
    #  - _qattn_sm90 contains unguarded wgmma/TMA PTX that only the sm_90a
    #    target accepts (Blackwell dropped wgmma), so it is exactly sm_90a and
    #    never PTX (compute_90a PTX cannot JIT anywhere else).
    #  - _qattn_sm120 re-instantiates the sm89 kernel template with the modes
    #    that make sense on consumer Blackwell (exact fp8-mma fp32 accumulator:
    #    plain fp32 accumulation, no inst_buf workarounds). _qattn_sm89 still
    #    serves major >= 10 as the runtime fallback / on-device reference until
    #    the specialized extensions are hardware-validated.
    #  - _qattn_sm100 contains unguarded tcgen05 PTX (kind::i8 QK + kind::f8f6f4
    #    PV) that only the sm_100a / sm_110a targets accept: ptxas rejects
    #    kind::i8 on sm_103a and the family targets sm_100f/sm_110f (GB300
    #    dropped INT8 tensor cores), so it serves exactly (10,0) and (11,0) and
    #    never PTX ('a'-target PTX cannot JIT anywhere). sm103 and every other
    #    Blackwell stay on the sm89-class fallback. It ships dark (opt-in via
    #    SAGEATTN_SM100_TCGEN05=1) until hardware validation.
    EXT_SERVES = {
        "_qattn_sm80":  lambda cc: cc >= (8, 0),
        "_qattn_sm89":  lambda cc: cc == (8, 9) or cc[0] >= 10,
        "_qattn_sm90":  lambda cc: cc == (9, 0),
        "_qattn_sm100": lambda cc: cc in {(10, 0), (11, 0)},
        "_qattn_sm120": lambda cc: cc[0] == 12,
        "_fused":       lambda cc: cc >= (8, 0),
    }

    def gencode_flags_for(ext_name):
        """Per-extension -gencode list; an empty list skips the extension."""
        served = sorted(cc for cc in requested_ccs if EXT_SERVES[ext_name](cc))
        if not served:
            return []
        if ext_name == "_qattn_sm90":
            check_gpu_code_supported((9, 0))
            return ["-gencode", "arch=compute_90a,code=sm_90a"]
        if ext_name == "_qattn_sm100":
            # arch-specific cubins only, never PTX (see EXT_SERVES comment).
            flags = []
            for cc in served:
                check_gpu_code_supported(cc)
                num = f"{cc[0]}{cc[1]}"
                flags += ["-gencode", f"arch=compute_{num}a,code=sm_{num}a"]
            return flags
        flags = []
        for cc in served:
            check_gpu_code_supported(cc)
            num = f"{cc[0]}{cc[1]}"
            flags += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
            if requested_ccs[cc]:
                flags += ["-gencode", f"arch=compute_{num},code=compute_{num}"]
        return flags

    sm89_ccs = sorted(cc for cc in requested_ccs if EXT_SERVES["_qattn_sm89"](cc))
    if sm89_ccs:
        if nvcc_cuda_version < Version("12.4"):
            raise RuntimeError(
                "CUDA 12.4 or higher is required to build the FP8 (sm89-class) kernels "
                f"for compute capabilities {sm89_ccs}.")
        if nvcc_cuda_version < Version("12.8"):
            warnings.warn(
                "CUDA < 12.8: the fp16-accumulator FP8 kernels "
                "(qk_int8_sv_f8_accum_f16_*) compile to runtime traps on this "
                "toolkit; upgrade to CUDA 12.8+ for the full sm89 kernel set.")
    if (9, 0) in requested_ccs and nvcc_cuda_version < Version("12.3"):
        raise RuntimeError(
            "CUDA 12.3 or higher is required for compute capability 9.0.")

    sm100_ccs = sorted(cc for cc in requested_ccs if EXT_SERVES["_qattn_sm100"](cc))
    if sm100_ccs:
        # (10,0) needs >= 12.8, already implied by the --list-gpu-code probe in
        # check_gpu_code_supported; (11,0) additionally needs the CUDA 13.0
        # tcgen05 PTX support, gate it explicitly for a friendlier error.
        if (11, 0) in sm100_ccs and nvcc_cuda_version < Version("13.0"):
            raise RuntimeError(
                "CUDA 13.0 or higher is required to build the tcgen05 (sm110) kernels "
                f"for compute capabilities {sm100_ccs}.")

    EXT_SOURCES = {
        "_qattn_sm80": [
            "csrc/qattn/pybind_sm80.cpp",
            "csrc/qattn/qk_int_sv_f16_cuda_sm80.cu",
        ],
        "_qattn_sm89": [
            "csrc/qattn/pybind_sm89.cpp",
            "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn.cu",
            "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn.cu",
            "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf.cu",
            "csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu",
        ],
        "_qattn_sm90": [
            "csrc/qattn/pybind_sm90.cpp",
            "csrc/qattn/qk_int_sv_f8_cuda_sm90.cu",
        ],
        "_qattn_sm100": [
            "csrc/qattn/pybind_sm100.cpp",
            "csrc/qattn/qk_int_sv_f8_cuda_sm100.cu",
        ],
        "_qattn_sm120": [
            "csrc/qattn/pybind_sm120.cpp",
            "csrc/qattn/sm120_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn.cu",
            "csrc/qattn/sm120_qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn.cu",
            "csrc/qattn/sm120_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu",
        ],
        "_fused": [
            "csrc/fused/pybind.cpp",
            "csrc/fused/fused.cu",
            "csrc/fused/quant_per_thread.cu",
        ],
    }
    # -lcuda: cuTensorMapEncodeTiled (driver API) for the TMA-based kernels.
    EXT_LINK_ARGS = {"_qattn_sm90": ["-lcuda"], "_qattn_sm100": ["-lcuda"]}

    for ext_name in ["_qattn_sm80", "_qattn_sm89", "_qattn_sm90", "_qattn_sm100", "_qattn_sm120", "_fused"]:
        gencodes = gencode_flags_for(ext_name)
        if not gencodes:
            print(f"[sageattention] skipping {ext_name}: "
                  "no requested compute capability is served by it")
            continue
        print(f"[sageattention] {ext_name}: "
              + " ".join(gencodes[i + 1] for i in range(0, len(gencodes), 2)))
        ext_modules.append(
            CUDAExtension(
                name=f"sageattention.{ext_name}",
                sources=EXT_SOURCES[ext_name],
                extra_compile_args={"cxx": CXX_FLAGS,
                                    "nvcc": NVCC_BASE_FLAGS + gencodes},
                extra_link_args=EXT_LINK_ARGS.get(ext_name, []),
            )
        )

    # quant.py imports _fused unconditionally; its serve predicate covers every
    # supported arch, so it must be present in any successful configuration.
    assert any(ext.name == "sageattention._fused" for ext in ext_modules)

    # Resolve parallelism from env
    parallel = None
    if 'EXT_PARALLEL' in os.environ:
        try:
            parallel = int(os.getenv('EXT_PARALLEL'))
        finally:
            pass
    if parallel is None and 'MAX_JOBS' in os.environ:
        try:
            parallel = int(os.getenv('MAX_JOBS'))
        finally:
            pass
    # Defaults if not provided
    if parallel is None:
        parallel = 4
    # Ensure MAX_JOBS for underlying tooling if not explicitly set
    os.environ.setdefault('MAX_JOBS', '32')

    class BuildExtensionSeparateDir(BuildExtension):
        build_extension_patch_lock = threading.Lock()
        thread_ext_name_map = {}

        def finalize_options(self):
            if parallel is not None:
                self.parallel = parallel
            super().finalize_options()

        def build_extension(self, ext):
            with self.build_extension_patch_lock:
                if not getattr(self.compiler, "_compile_separate_output_dir", False):
                    compile_orig = self.compiler.compile

                    def compile_new(*args, **kwargs):
                        return compile_orig(*args, **{
                            **kwargs,
                            "output_dir": os.path.join(
                                kwargs["output_dir"],
                                self.thread_ext_name_map[threading.current_thread().ident]),
                        })
                    self.compiler.compile = compile_new
                    self.compiler._compile_separate_output_dir = True
            self.thread_ext_name_map[threading.current_thread().ident] = ext.name
            objects = super().build_extension(ext)
            return objects

    cmdclass = {"build_ext": BuildExtensionSeparateDir} if ext_modules else {}

setup(
    name='sageattention',
    version='2.2.0',
    author='SageAttention team',
    license='Apache 2.0 License',
    description='Accurate and efficient plug-and-play low-bit attention.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/thu-ml/SageAttention',
    packages=find_packages(),
    python_requires='>=3.9',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
