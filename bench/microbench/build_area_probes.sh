#!/bin/bash
# Build the E1/P3 instruction-area probe cubins (device code only, no GPU
# needed). Run from the repo root; cubins land in $OUT (default ./area_cubins).
#   TORCH_INC: torch include root (python -c "import torch.utils.cpp_extension as c; print(c.include_paths()[0])")
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${OUT:-$ROOT/area_cubins}"
TORCH_INC="${TORCH_INC:-$(python -c 'import torch.utils.cpp_extension as c; print(c.include_paths()[0])')}"
mkdir -p "$OUT"

FLAGS=(-std=c++17 -O3 --use_fast_math -cubin
  -diag-suppress=174
  -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
  -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
  --expt-relaxed-constexpr
  -I "$ROOT/csrc/qattn" -I "$ROOT/csrc"
  -I "$TORCH_INC" -I "$TORCH_INC/torch/csrc/api/include")

nvcc "${FLAGS[@]}" -arch=sm_89 "$ROOT/bench/microbench/area_probe_sm89.cu" -o "$OUT/area_sm89.cubin"
echo "sm89 OK"
nvcc "${FLAGS[@]}" -arch=sm_90a "$ROOT/bench/microbench/area_probe_sm90.cu" -o "$OUT/area_sm90a.cubin"
echo "sm90a OK"
# sm100 reuses the existing device-only probe TU (torch-free)
nvcc -std=c++17 -O3 --use_fast_math -cubin -arch=sm_100a \
  -I "$ROOT/csrc/qattn" \
  "$ROOT/bench/sm100_review/qk_int_sv_f8_cuda_sm100_probe.cu" -o "$OUT/area_sm100a.cubin"
echo "sm100a OK"
