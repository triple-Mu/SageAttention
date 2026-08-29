"""ptxas gate over the device-only probe TUs in bench/sm100_review/.

The probes are not part of the extension build, so nothing else ever compiles
them; without this test they silently rot when the kernel's namespace or
parameter list changes (it happened once). Each case shells out to nvcc with
the exact command from the probe's file header -- no GPU and no torch tensors
involved, ptxas machine-checks every asm string, descriptor and TMEM access.
Skips when nvcc is missing or too old to target the arch (sm_100a needs
CUDA >= 12.9, sm_110a needs CUDA >= 13.0).
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NVCC = shutil.which("nvcc")

# `--list-gpu-code` prints base codes (sm_100), the gencode targets carry the
# arch-specific suffix (sm_100a).
if NVCC is not None:
    SUPPORTED_CODES = frozenset(
        subprocess.run([NVCC, "--list-gpu-code"], capture_output=True, text=True).stdout.split()
    )
else:
    SUPPORTED_CODES = frozenset()

# (probe TU, include dir, expected `ptxas -v` entry-function count)
PROBES = [
    pytest.param("bench/sm100_review/qk_int_sv_f8_cuda_sm100_probe.cu", "csrc/qattn", 4, id="qk_int_sv_f8"),
    pytest.param("bench/sm100_review/tcgen05_probe.cu", "csrc", 1, id="tcgen05"),
]


@pytest.mark.parametrize("arch", ["100a", "110a"])
@pytest.mark.parametrize("probe,include_dir,num_kernels", PROBES)
def test_probe_tu_compiles(probe, include_dir, num_kernels, arch, tmp_path):
    if f"sm_{arch.rstrip('a')}" not in SUPPORTED_CODES:
        pytest.skip(f"nvcc missing or does not target sm_{arch}")
    result = subprocess.run(
        [
            NVCC, "-std=c++17", "-O3", "--use_fast_math", "-cubin", "-Xptxas", "-v",
            "-gencode", f"arch=compute_{arch},code=sm_{arch}",
            "-I", str(ROOT / include_dir),
            "-o", str(tmp_path / "probe.cubin"),
            str(ROOT / probe),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    compiled = result.stderr.count("Compiling entry function")
    assert compiled == num_kernels, result.stderr
