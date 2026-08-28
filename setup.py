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

---

setuptools entry point around the CMake build (see CMakeLists.txt). Supports
`pip install -e .`, `pip install .`, `python setup.py install|bdist_wheel`
and sdist (via SAGEATTN_SKIP_CUDA_BUILD=1, which never imports torch).

Environment knobs:
  TORCH_CUDA_ARCH_LIST   target archs, e.g. "8.0;8.9;9.0;12.0" (default: local GPUs)
  MAX_JOBS               cmake --build parallelism (default: min(nproc, 16))
  NVCC_THREADS           nvcc --threads per TU (default: 8)
  DEBUG=1                CMake Debug build
  SAGEATTN_PTXAS_VERBOSE=1 / SAGEATTN_LINEINFO=1
  SAGEATTN_CMAKE_ARGS    extra args appended to the cmake configure call
  SAGEATTN_BUILD_DIR     reusable CMake build dir (default: <repo>/build/cmake)
"""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).parent.resolve()
MIN_CMAKE = (3, 26)

SKIP_CUDA_BUILD = (
    os.getenv("SAGEATTN_SKIP_CUDA_BUILD", "0").upper() in {"1", "TRUE", "YES"}
    or "sdist" in sys.argv
)


def _find_cmake():
    candidates = []
    if os.getenv("CMAKE_EXECUTABLE"):
        candidates.append(os.environ["CMAKE_EXECUTABLE"])
    # pip-installed cmake lives next to the interpreter (also covers build
    # isolation, where PATH may not include the env's scripts dir)
    sibling = Path(sys.executable).parent / "cmake"
    if sibling.exists():
        candidates.append(str(sibling))
    exe = shutil.which("cmake")
    if exe:
        candidates.append(exe)
    for cmake in candidates:
        try:
            out = subprocess.check_output([cmake, "--version"], text=True)
            ver = tuple(int(x) for x in out.split()[2].split("-")[0].split(".")[:2])
            if ver >= MIN_CMAKE:
                return cmake
        except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
            continue
    raise RuntimeError(
        f"cmake >= {MIN_CMAKE[0]}.{MIN_CMAKE[1]} not found. "
        'Install it into the build environment: pip install "cmake>=3.31" ninja'
    )


class CMakeExtension(Extension):
    def __init__(self, name):
        # py_limited_api=True makes setuptools name the artifact _C.abi3.so;
        # the cp39-abi3 wheel tag comes from options={"bdist_wheel": ...} below.
        super().__init__(name, sources=[], py_limited_api=True)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        cmake = _find_cmake()
        build_dir = Path(os.getenv("SAGEATTN_BUILD_DIR") or ROOT / "build" / "cmake").resolve()
        build_dir.mkdir(parents=True, exist_ok=True)

        gen = ["-G", "Ninja"] if shutil.which("ninja") else []
        configure = [
            cmake,
            "-S",
            str(ROOT),
            "-B",
            str(build_dir),
            *gen,
            f"-DPython_EXECUTABLE={sys.executable}",
            # Always pass the arch list explicitly (an empty value triggers
            # env/auto-detection inside CMake) so a stale cache entry can never
            # silently win over a changed TORCH_CUDA_ARCH_LIST.
            f"-DSAGE_CUDA_ARCHS={os.getenv('TORCH_CUDA_ARCH_LIST', '')}",
            f"-DCMAKE_BUILD_TYPE={'Debug' if os.getenv('DEBUG') == '1' else 'Release'}",
            f"-DSAGE_NVCC_THREADS={os.getenv('NVCC_THREADS', '8')}",
            f"-DSAGE_PTXAS_VERBOSE={'ON' if os.getenv('SAGEATTN_PTXAS_VERBOSE') else 'OFF'}",
            f"-DSAGE_LINEINFO={'ON' if os.getenv('SAGEATTN_LINEINFO') else 'OFF'}",
            *shlex.split(os.getenv("SAGEATTN_CMAKE_ARGS", "")),
        ]
        subprocess.run(configure, check=True)

        jobs = self.parallel or int(os.getenv("MAX_JOBS") or 0) or min(os.cpu_count() or 4, 16)
        build = [cmake, "--build", str(build_dir), "--target", "_C", "--parallel", str(jobs)]
        if self.verbose or os.getenv("VERBOSE"):
            build.append("--verbose")
        subprocess.run(build, check=True)

        src = build_dir / "lib" / "sageattention" / "_C.abi3.so"
        dst = Path(self.get_ext_fullpath(ext.name))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def _runtime_requires():
    """Pin torch to the build-time major.minor: the extension links libtorch's
    C++ ABI, so installing the wheel against a different torch would fail at
    import time with undefined symbols. Falls back to an unpinned 'torch' when
    building the sdist (no torch import there)."""
    if SKIP_CUDA_BUILD:
        return ["torch"]
    try:
        import torch

        major, minor = torch.__version__.split(".")[:2]
        return [f"torch=={major}.{minor}.*"]
    except ImportError:
        return ["torch"]


setup(
    ext_modules=[] if SKIP_CUDA_BUILD else [CMakeExtension("sageattention._C")],
    cmdclass={} if SKIP_CUDA_BUILD else {"build_ext": CMakeBuild},
    install_requires=_runtime_requires(),
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
