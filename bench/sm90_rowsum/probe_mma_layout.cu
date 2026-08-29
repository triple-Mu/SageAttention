// Pre-verification for the sm90 row-sum-on-tensor-core port.
//
// Claim under test: the register layout that RS_f32_to_f8 produces (which the
// sm90 kernel feeds to the wgmma RS-form A operand) is also a valid A operand
// for mma.sync.aligned.m16n8k32.e4m3 - i.e. the same four registers name the
// same rows in both instructions.
//
// Method: feed one RS_f8 fragment to both instructions with an all-ones B and
// compare. wgmma with B = all ones gives the row sum in every output column;
// mma.sync (rowsum_f8f8f32) gives the row sum in denom. If the two agree, and
// both agree with a host-side row sum of the quantized P, the layouts match.
//
// Build and run (sm_90a needs the `a` target - plain sm_90 has no fp8 wgmma):
//   nvcc -O2 -gencode arch=compute_90a,code=sm_90a -I<repo>/csrc \
//        -o probe probe_mma_layout.cu && ./probe
//
// Result on H200 (2026-08-30): [1] bad=0 maxrel=0, [2] maxrel 1.66e-4 -> the
// layouts match; see test/HARDWARE_CHECKLIST.md section 6.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp8.h>

#include "mma.cuh"
#include "numeric_conversion.cuh"
#include "wgmma.cuh"

// ---- verbatim from csrc/qattn/attn_utils.cuh (copied to avoid torch headers) ----
template<uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void RS_f32_to_f8(float RS[][num_tiles_k][8], uint32_t RS_f8[][num_tiles_k / 2][4])
{
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
            floatx4_to_e4m3x4(RS_f8[fq][fk], RS[fq][fk * 2 + 0], RS[fq][fk * 2 + 0] + 4);
            floatx4_to_e4m3x4(RS_f8[fq][fk] + 1, RS[fq][fk * 2 + 0] + 2, RS[fq][fk * 2 + 0] + 6);
            floatx4_to_e4m3x4(RS_f8[fq][fk] + 2, RS[fq][fk * 2 + 1], RS[fq][fk * 2 + 1] + 4);
            floatx4_to_e4m3x4(RS_f8[fq][fk] + 3, RS[fq][fk * 2 + 1] + 2, RS[fq][fk * 2 + 1] + 6);
        }
    }
}

template<uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void accumulate_d_f8(uint32_t RS[][num_tiles_k / 2][4], float denom[][2])
{
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
            mma::rowsum_f8f8f32(denom[fq], RS[fq][fk]);
        }
    }
}
// --------------------------------------------------------------------------------

constexpr uint32_t CTA_K              = 128;
constexpr uint32_t HEAD_DIM           = 64;
constexpr uint32_t num_tiles_k        = CTA_K / 16;   // 8
constexpr uint32_t num_tiles_v        = HEAD_DIM / 16;  // 4
constexpr uint32_t num_tiles_pv_inner = CTA_K / 32;   // 4

__global__ __launch_bounds__(128) void probe(const float* __restrict__ P,  // [64][CTA_K] row major
                                             float* out_denom,             // [128][2]
                                             float* out_wgmma,             // [128][num_tiles_v*8]
                                             float* out_cuda,              // [128][2]
                                             uint32_t* out_rows,           // [128][2]
                                             const float* __restrict__ init)  // [2], host zeros
{
    extern __shared__ __align__(128) int8_t smem_[];
    int8_t* sV = smem_;

    // V^T slab, every byte the e4m3 encoding of 1.0 (0x38). All-ones B makes
    // the wgmma output layout-independent: every column is the row sum.
    for (uint32_t i = threadIdx.x; i < HEAD_DIM * CTA_K / 4; i += 128) {
        reinterpret_cast<uint32_t*>(sV)[i] = 0x38383838u;
    }
    __syncthreads();

    const uint32_t warp_idx = threadIdx.x / 32;
    const uint32_t lane_id  = threadIdx.x % 32;
    const uint32_t r        = warp_idx * 16 + lane_id / 4;
    const uint32_t c        = lane_id % 4;

    // Load P into the wgmma s32 accumulator layout the kernel's RS lives in.
    float RS_f32[1][num_tiles_k][8];
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
        RS_f32[0][fk][0] = P[r * CTA_K + fk * 16 + 2 * c + 0];
        RS_f32[0][fk][1] = P[r * CTA_K + fk * 16 + 2 * c + 1];
        RS_f32[0][fk][2] = P[(r + 8) * CTA_K + fk * 16 + 2 * c + 0];
        RS_f32[0][fk][3] = P[(r + 8) * CTA_K + fk * 16 + 2 * c + 1];
        RS_f32[0][fk][4] = P[r * CTA_K + fk * 16 + 8 + 2 * c + 0];
        RS_f32[0][fk][5] = P[r * CTA_K + fk * 16 + 8 + 2 * c + 1];
        RS_f32[0][fk][6] = P[(r + 8) * CTA_K + fk * 16 + 8 + 2 * c + 0];
        RS_f32[0][fk][7] = P[(r + 8) * CTA_K + fk * 16 + 8 + 2 * c + 1];
    }

    // cuda-core partial sums of the pre-quantized fp32 P, as the kernel does today
    float d_cuda[2] = {0.0f, 0.0f};
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
        d_cuda[0] += RS_f32[0][fk][0] + RS_f32[0][fk][1] + RS_f32[0][fk][4] + RS_f32[0][fk][5];
        d_cuda[1] += RS_f32[0][fk][2] + RS_f32[0][fk][3] + RS_f32[0][fk][6] + RS_f32[0][fk][7];
    }
#pragma unroll
    for (uint32_t e = 0; e < 2; e++) {
        d_cuda[e] += __shfl_xor_sync(0xffffffff, d_cuda[e], 0x1);
        d_cuda[e] += __shfl_xor_sync(0xffffffff, d_cuda[e], 0x2);
    }

    uint32_t RS_f8[1][num_tiles_pv_inner][4];
    RS_f32_to_f8<1, num_tiles_k>(RS_f32, RS_f8);

    // path A: mma.sync.m16n8k32 row sum. The seed comes from memory so the two
    // accumulators stay distinct registers; a literal 0 for both lets nvcc CSE
    // them into one, which ptxas then rejects in the mma's C operand.
    float denom[1][2];
    denom[0][0] = init[0];
    denom[0][1] = init[1];
    accumulate_d_f8<1, num_tiles_k>(RS_f8, denom);

    // path B: wgmma, same A registers, all-ones B
    float RO[1][num_tiles_v][8];
    wgmma::warpgroup_arrive();
    wgmma::wgmma_f8f8f32<HEAD_DIM, 0, CTA_K>(RO[0], RS_f8[0][0], &sV[0]);
#pragma unroll
    for (uint32_t v_it = 1; v_it < num_tiles_pv_inner; v_it++) {
        wgmma::wgmma_f8f8f32<HEAD_DIM, 1, CTA_K>(RO[0], RS_f8[0][v_it], &sV[v_it * 32]);
    }
    wgmma::warpgroup_commit_batch();
    wgmma::warpgroup_wait<0>();

    out_denom[threadIdx.x * 2 + 0] = denom[0][0];
    out_denom[threadIdx.x * 2 + 1] = denom[0][1];
    out_cuda[threadIdx.x * 2 + 0]  = d_cuda[0];
    out_cuda[threadIdx.x * 2 + 1]  = d_cuda[1];
    out_rows[threadIdx.x * 2 + 0]  = r;
    out_rows[threadIdx.x * 2 + 1]  = r + 8;
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
#pragma unroll
        for (uint32_t e = 0; e < 8; e++) {
            out_wgmma[threadIdx.x * (num_tiles_v * 8) + fv * 8 + e] = RO[0][fv][e];
        }
    }
}

#define CHECK(x)                                                                       \
    do {                                                                               \
        cudaError_t _e = (x);                                                          \
        if (_e != cudaSuccess) {                                                       \
            printf("CUDA error %s at %d\n", cudaGetErrorString(_e), __LINE__);          \
            return 1;                                                                  \
        }                                                                              \
    } while (0)

int main()
{
    const int M = 64;
    const int K = CTA_K;

    float* hP = (float*)malloc(sizeof(float) * M * K);
    srand(1234);
    for (int i = 0; i < M * K; i++) {
        // the kernel's P448 range: exp2 output scaled into (0, 448]
        hP[i] = 448.0f * (float)rand() / (float)RAND_MAX;
    }

    // host golden: row sums of the e4m3-quantized P, and of the raw fp32 P
    double* gq  = (double*)calloc(M, sizeof(double));
    double* graw = (double*)calloc(M, sizeof(double));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(hP[i * K + j], __NV_SATFINITE, __NV_E4M3);
            __half_raw          h = __nv_cvt_fp8_to_halfraw(q, __NV_E4M3);
            gq[i] += (double)__half2float(h);
            graw[i] += (double)hP[i * K + j];
        }
    }

    float *dP, *d_denom, *d_wgmma, *d_cuda;
    uint32_t* d_rows;
    CHECK(cudaMalloc(&dP, sizeof(float) * M * K));
    CHECK(cudaMalloc(&d_denom, sizeof(float) * 128 * 2));
    CHECK(cudaMalloc(&d_wgmma, sizeof(float) * 128 * num_tiles_v * 8));
    CHECK(cudaMalloc(&d_cuda, sizeof(float) * 128 * 2));
    CHECK(cudaMalloc(&d_rows, sizeof(uint32_t) * 128 * 2));
    CHECK(cudaMemcpy(dP, hP, sizeof(float) * M * K, cudaMemcpyHostToDevice));
    float* d_init;
    float  h_init[2] = {0.0f, 0.0f};
    CHECK(cudaMalloc(&d_init, sizeof(float) * 2));
    CHECK(cudaMemcpy(d_init, h_init, sizeof(float) * 2, cudaMemcpyHostToDevice));

    const size_t smem = HEAD_DIM * CTA_K;
    probe<<<1, 128, smem>>>(dP, d_denom, d_wgmma, d_cuda, d_rows, d_init);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    float*    denom = (float*)malloc(sizeof(float) * 128 * 2);
    float*    wg    = (float*)malloc(sizeof(float) * 128 * num_tiles_v * 8);
    float*    cc    = (float*)malloc(sizeof(float) * 128 * 2);
    uint32_t* rows  = (uint32_t*)malloc(sizeof(uint32_t) * 128 * 2);
    CHECK(cudaMemcpy(denom, d_denom, sizeof(float) * 128 * 2, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(wg, d_wgmma, sizeof(float) * 128 * num_tiles_v * 8, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(cc, d_cuda, sizeof(float) * 128 * 2, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(rows, d_rows, sizeof(uint32_t) * 128 * 2, cudaMemcpyDeviceToHost));

    int    n_wg_mismatch = 0, n_golden_bad = 0, n_raw_bad = 0;
    double max_rel_wg = 0.0, max_rel_golden = 0.0, max_rel_raw = 0.0;
    double max_rel_quant_vs_raw = 0.0;

    for (int t = 0; t < 128; t++) {
        for (int e = 0; e < 2; e++) {
            const uint32_t row = rows[t * 2 + e];
            const float    dv  = denom[t * 2 + e];

            // (1) mma.sync row sum vs host golden of the quantized P
            double rel = fabs((double)dv - gq[row]) / gq[row];
            if (rel > max_rel_golden) max_rel_golden = rel;
            if (rel > 1e-5) n_golden_bad++;

            // (2) wgmma output, every column of every n-tile, vs the mma.sync value.
            //     e in {0,1,4,5} is the row at +0, {2,3,6,7} the row at +8.
            for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
                for (uint32_t ee = 0; ee < 8; ee++) {
                    const int which = ((ee % 4) / 2);  // 0 -> row r, 1 -> row r+8
                    if (which != e) continue;
                    const float wv = wg[t * (num_tiles_v * 8) + fv * 8 + ee];
                    double      r2 = fabs((double)wv - (double)dv) / fabs((double)dv);
                    if (r2 > max_rel_wg) max_rel_wg = r2;
                    if (wv != dv) n_wg_mismatch++;
                }
            }

            // (3) the current cuda-core path vs the raw fp32 row sum
            double r3 = fabs((double)cc[t * 2 + e] - graw[row]) / graw[row];
            if (r3 > max_rel_raw) max_rel_raw = r3;
            if (r3 > 1e-5) n_raw_bad++;

            double r4 = fabs(gq[row] - graw[row]) / graw[row];
            if (r4 > max_rel_quant_vs_raw) max_rel_quant_vs_raw = r4;
        }
    }

    printf("=== sm90 rowsum layout probe (M=%d K=%d, head_dim=%d) ===\n", M, K, HEAD_DIM);
    printf("[1] mma.sync rowsum vs host quantized rowsum : bad=%d maxrel=%.3e\n", n_golden_bad, max_rel_golden);
    printf("[2] wgmma(all-ones B) vs mma.sync rowsum     : bitmismatch=%d maxrel=%.3e\n",
           n_wg_mismatch,
           max_rel_wg);
    printf("[3] cuda-core FADD path vs host raw rowsum   : bad=%d maxrel=%.3e\n", n_raw_bad, max_rel_raw);
    printf("[4] quantized vs raw denominator (semantics) : maxrel=%.3e\n", max_rel_quant_vs_raw);
    // [2] is not expected to be bit-equal: wgmma's fp8 f32 accumulate keeps
    // fewer mantissa bits than mma.sync does, so the two row sums differ in the
    // last bits. A layout mismatch would show up as an O(1) relative error, not
    // an O(1e-4) one.
    const bool ok = (n_golden_bad == 0) && (max_rel_wg < 1e-3);
    printf("VERDICT: %s\n", ok ? "LAYOUTS MATCH" : "LAYOUT MISMATCH");
    return ok ? 0 : 1;
}
