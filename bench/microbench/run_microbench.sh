#!/bin/bash
# Step-0 microbench for the sm12x specialization work (run on the RTX PRO / 5090 box).
# Decides: (a) is plain fp32 fp8-mma accumulation exact on GB202 -> v0 default
#          (b) is QMMA.SF full-rate with fp32 accum -> v2 go/no-go
#          (c) does sm_120f accept kind::mxf8f6f4 -> v2 gencode plan A vs B
set -e
cd "$(dirname "$0")"
NVCC=${NVCC:-nvcc}
echo "== toolchain =="; $NVCC --version | grep release

echo "== build =="
$NVCC -O2 -std=c++17 -gencode arch=compute_120,code=sm_120 mma_accum_precision.cu -o precision_120
$NVCC -O2 -std=c++17 -gencode arch=compute_120a,code=sm_120a -DSAGE_SM120A mma_accum_precision.cu -o precision_120a
$NVCC -O3 -std=c++17 -gencode arch=compute_120,code=sm_120 mma_rate.cu -o rate_120
$NVCC -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -DSAGE_SM120A mma_rate.cu -o rate_120a

echo; echo "== accumulator precision (plain sm_120 binary) =="; ./precision_120
echo; echo "== accumulator precision (sm_120a binary, incl. QMMA.SF) =="; ./precision_120a
echo; echo "== sustained rate (plain) =="; ./rate_120
echo; echo "== sustained rate (sm_120a, incl. QMMA.SF) =="; ./rate_120a

echo; echo "== sm_120f probe: does the family target accept kind::mxf8f6f4? =="
PTXAS_VER=$($NVCC --version | grep -oP 'release \K[0-9]+\.[0-9]+')
cat > /tmp/bs_mxf8_120f.ptx <<'EOF'
.version 8.8
.target sm_120f
.address_size 64
.visible .entry probe()
{
  .reg .b32 %ra<5>; .reg .b32 %rb<3>; .reg .f32 %f<5>; .reg .b32 %s<3>;
  mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
    {%f1,%f2,%f3,%f4}, {%ra1,%ra2,%ra3,%ra4}, {%rb1,%rb2}, {%f1,%f2,%f3,%f4},
    {%s1}, {0, 0}, {%s2}, {0, 0};
  ret;
}
EOF
if ptxas -arch=sm_120f /tmp/bs_mxf8_120f.ptx -o /tmp/bs_mxf8_120f.cubin 2>/tmp/bs_mxf8_120f.err; then
  echo "sm_120f ACCEPTS kind::mxf8f6f4 -> v2 gencode plan A (single compute_120f cubin covers 12.0+12.1)"
else
  echo "sm_120f REJECTS kind::mxf8f6f4 -> v2 gencode plan B (emit both sm_120a and sm_121a)"
  cat /tmp/bs_mxf8_120f.err
fi
