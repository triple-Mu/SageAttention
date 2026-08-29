/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../utils.cuh"
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/types.h>

#include "../dispatch_utils.h"
#include "../math.cuh"
#include "../wgmma.cuh"

#include "../tma.cuh"
#include "attn_utils.cuh"
#include "launch_utils.cuh"

// All sm90 kernels and launchers live in sage::sm90 (single-so ODR rule).
namespace sage {
namespace sm90 {

template<uint32_t         CTA_Q,
         uint32_t         CTA_K,
         uint32_t         NUM_THREADS,
         uint32_t         head_dim,
         QuantGranularity Q_GRAN,
         QuantGranularity K_GRAN,
         typename DTypeOut,
         MaskMode mask_mode    = MaskMode::kNone,
         bool     return_lse   = false,
         bool     fuse_v_scale = false>
__global__ void qk_int8_sv_f8_attn_kernel(const __grid_constant__ CUtensorMap tensorMapQ,
                                          const __grid_constant__ CUtensorMap tensorMapK,
                                          const __grid_constant__ CUtensorMap tensorMapV,
                                          const float* __restrict__ Q_scale,
                                          const float* __restrict__ K_scale,
                                          const float* __restrict__ V_scale,
                                          DTypeOut* O,
                                          float* __restrict__ Lse,
                                          const int64_t  stride_batch_o,
                                          const int64_t  stride_h_o,
                                          uint32_t       stride_seq_o,
                                          const uint32_t qo_len,
                                          const uint32_t kv_len,
                                          const uint32_t qo_per_kv_head,
                                          float          sm_scale)
{
    static_assert(NUM_THREADS == 128);
    static_assert(CTA_Q <= CTA_K);

    const uint32_t warp_idx = (threadIdx.x % 128) / 32;
    const uint32_t lane_id  = threadIdx.x % 32;

    constexpr uint32_t num_tiles_q        = CTA_Q / 64;
    constexpr uint32_t num_tiles_k        = CTA_K / 16;
    constexpr uint32_t num_tiles_qk_inner = head_dim / 32;
    constexpr uint32_t num_tiles_v        = head_dim / 16;
    constexpr uint32_t num_tiles_pv_inner = CTA_K / 32;

    const uint32_t batch_id     = blockIdx.z;
    const uint32_t cta_idx_q    = blockIdx.x;
    const uint32_t head_id      = blockIdx.y;
    const uint32_t num_qo_heads = gridDim.y;
    const uint32_t kv_head_id   = head_id / qo_per_kv_head;

    sm_scale *= math::log2e;

    extern __shared__ __align__(128) int8_t smem_[];

    int8_t* sQ = (int8_t*)smem_;
    int8_t* sK = (int8_t*)(smem_ + CTA_Q * head_dim * sizeof(int8_t));
    int8_t* sV = (int8_t*)(smem_ + CTA_Q * head_dim * sizeof(int8_t) + CTA_K * head_dim * sizeof(int8_t));
    half*   sO = (half*)smem_;

    int32_t RS[num_tiles_q][num_tiles_k][8];
    float   RO[num_tiles_q][num_tiles_v][8];
    float   row_max[num_tiles_q][2];
    float   denom[num_tiles_q][2];

    // Q scale is read once: the whole index may live in 64-bit. K scale is read
    // once per KV tile, so it is split into a 64-bit base pointer plus a 32-bit
    // running offset that the main loop advances.
    int64_t      q_scale_idx;
    const float* K_scale_base_ptr;
    uint32_t     k_scale_off;

    if constexpr (Q_GRAN == QuantGranularity::kPerBlock) {
        const uint32_t num_ctas_q = gridDim.x;
        q_scale_idx               = static_cast<int64_t>(batch_id) * num_qo_heads * num_ctas_q
                      + static_cast<int64_t>(head_id) * num_ctas_q + cta_idx_q;
    }
    else if constexpr (Q_GRAN == QuantGranularity::kPerWarp) {
        const uint32_t num_warp_tiles_q = gridDim.x * 4;
        q_scale_idx                     = static_cast<int64_t>(batch_id) * num_qo_heads * num_warp_tiles_q
                      + static_cast<int64_t>(head_id) * num_warp_tiles_q + cta_idx_q * 4 + warp_idx;
    }
    else if constexpr (Q_GRAN == QuantGranularity::kPerThread) {
        const uint32_t num_warp_tiles_q = gridDim.x * 4;
        q_scale_idx                     = static_cast<int64_t>(batch_id) * num_qo_heads * (num_warp_tiles_q * 8)
                      + static_cast<int64_t>(head_id) * (num_warp_tiles_q * 8) + cta_idx_q * (4 * 8) + warp_idx * 8
                      + lane_id / 4;
    }

    if constexpr (K_GRAN == QuantGranularity::kPerBlock || K_GRAN == QuantGranularity::kPerWarp) {
        const uint32_t num_ctas_k = div_ceil(kv_len, CTA_K);
        K_scale_base_ptr = K_scale + static_cast<int64_t>(batch_id) * (num_qo_heads / qo_per_kv_head) * num_ctas_k
                           + static_cast<int64_t>(head_id / qo_per_kv_head) * num_ctas_k;
        k_scale_off = 0;
    }
    else if constexpr (K_GRAN == QuantGranularity::kPerThread) {
        const uint32_t num_ctas_k = div_ceil(kv_len, CTA_K);
        K_scale_base_ptr = K_scale + static_cast<int64_t>(batch_id) * (num_qo_heads / qo_per_kv_head) * (num_ctas_k * 4)
                           + static_cast<int64_t>(head_id / qo_per_kv_head) * (num_ctas_k * 4);
        k_scale_off = lane_id % 4;
    }

    constexpr uint32_t k_scale_advance_offset =
        (K_GRAN == QuantGranularity::kPerBlock || K_GRAN == QuantGranularity::kPerWarp) ? 1 : 4;

    uint32_t Q_idx_lane_base = cta_idx_q * CTA_Q + warp_idx * 16 + lane_id / 4;

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        row_max[fq][0] = -5000000.0f;
        row_max[fq][1] = -5000000.0f;
        denom[fq][0]   = 1.0f;
        denom[fq][1]   = 1.0f;
    }

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
#pragma unroll
            for (uint32_t e = 0; e < 8; e++) {
                RO[fq][fv][e] = 0.0f;
            }
        }
    }

    __shared__ __align__(8) uint64_t barrier_Q;
    __shared__ __align__(8) uint64_t barrier_K;
    __shared__ __align__(8) uint64_t barrier_V;

    if (threadIdx.x == 0) {
        init_barrier(&barrier_Q, 1);
        init_barrier(&barrier_K, 1);
        init_barrier(&barrier_V, 1);
    }

    __syncthreads();

    // load Q, K, V
    if (threadIdx.x == 0) {
        expect_bytes<(CTA_Q * head_dim) * sizeof(int8_t)>(&barrier_Q);
        expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_K);
        expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_V);
        load_async_4D(sQ, &tensorMapQ, &barrier_Q, 0, cta_idx_q * CTA_Q, head_id, batch_id);
        load_async_4D(sK, &tensorMapK, &barrier_K, 0, 0, kv_head_id, batch_id);
        load_async_4D(sV, &tensorMapV, &barrier_V, 0, 0, kv_head_id, batch_id);
    }

    float q_scale           = Q_scale[q_scale_idx];
    float original_sm_scale = sm_scale;

    // wait for Q
    wait(&barrier_Q, 0);

    const uint32_t num_iterations =
        div_ceil(mask_mode == MaskMode::kCausal ? min(kv_len, (cta_idx_q + 1) * CTA_Q) : kv_len, CTA_K);

    int phase = 1;
    for (uint32_t iter = 1; iter < num_iterations; iter++) {
        phase ^= 1;

        float dequant_scale = q_scale * K_scale_base_ptr[k_scale_off + (iter - 1) * k_scale_advance_offset];
        sm_scale            = original_sm_scale * dequant_scale;

        // wait for K
        wait(&barrier_K, phase);

        // compute QK^T
        wgmma::warpgroup_arrive();
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
            int8_t* sQ_local = sQ + fq * 64 * head_dim;
            wgmma::wgmma_s8s8s32<CTA_K, 0, head_dim>(RS[fq], sQ_local, sK);
#pragma unroll
            for (int k_it = 1; k_it < num_tiles_qk_inner; k_it++) {
                wgmma::wgmma_s8s8s32<CTA_K, 1, head_dim>(RS[fq], &sQ_local[k_it * 32], &sK[k_it * 32]);
            }
        }
        wgmma::warpgroup_commit_batch();
        wgmma::warpgroup_wait<0>();

        // load K
        if (threadIdx.x == 0) {
            expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_K);
            load_async_4D(sK, &tensorMapK, &barrier_K, 0, iter * CTA_K, kv_head_id, batch_id);
        }

        // convert RS to float
        float RS_f32[num_tiles_q][num_tiles_k][8];
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
                for (uint32_t e = 0; e < 8; e++) {
                    RS_f32[fq][fk][e] = __int2float_rz(RS[fq][fk][e]);
                }
            }
        }

        update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(RS_f32, RO, row_max, denom, sm_scale);

        // accumulate denom on thread basis
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
                denom[fq][0] += (RS_f32[fq][fk][0] + RS_f32[fq][fk][1] + RS_f32[fq][fk][4] + RS_f32[fq][fk][5]);
                denom[fq][1] += (RS_f32[fq][fk][2] + RS_f32[fq][fk][3] + RS_f32[fq][fk][6] + RS_f32[fq][fk][7]);
            }
        }

        uint32_t RS_f8[num_tiles_q][num_tiles_pv_inner][4];
        RS_f32_to_f8<num_tiles_q, num_tiles_k>(RS_f32, RS_f8);

        // wait for V
        wait(&barrier_V, phase);

        float RO_tmp[num_tiles_q][num_tiles_v][8];
        wgmma::warpgroup_arrive();
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
            wgmma::wgmma_f8f8f32<head_dim, 0, CTA_K>(RO_tmp[fq], RS_f8[fq][0], &sV[0]);
#pragma unroll
            for (uint32_t v_it = 1; v_it < num_tiles_pv_inner; v_it++) {
                wgmma::wgmma_f8f8f32<head_dim, 1, CTA_K>(RO_tmp[fq], RS_f8[fq][v_it], &sV[v_it * 32]);
            }
        }

        wgmma::warpgroup_commit_batch();
        wgmma::warpgroup_wait<0>();

#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
#pragma unroll
                for (uint32_t e = 0; e < 8; e++) {
                    RO[fq][fv][e] += RO_tmp[fq][fv][e];
                }
            }
        }

        // load V
        if (threadIdx.x == 0) {
            expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_V);
            load_async_4D(sV, &tensorMapV, &barrier_V, iter * CTA_K, 0, kv_head_id, batch_id);
        }
    }

    {
        phase ^= 1;

        float dequant_scale = q_scale * K_scale_base_ptr[k_scale_off + (num_iterations - 1) * k_scale_advance_offset];
        sm_scale            = original_sm_scale;

        // wait for K
        wait(&barrier_K, phase);

        // compute QK^T
        wgmma::warpgroup_arrive();
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
            int8_t* sQ_local = sQ + fq * 64 * head_dim;
            wgmma::wgmma_s8s8s32<CTA_K, 0, head_dim>(RS[fq], sQ_local, sK);
#pragma unroll
            for (int k_it = 1; k_it < num_tiles_qk_inner; k_it++) {
                wgmma::wgmma_s8s8s32<CTA_K, 1, head_dim>(RS[fq], &sQ_local[k_it * 32], &sK[k_it * 32]);
            }
        }
        wgmma::warpgroup_commit_batch();
        wgmma::warpgroup_wait<0>();

        // convert RS to float
        float RS_f32[num_tiles_q][num_tiles_k][8];
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
                for (uint32_t e = 0; e < 8; e++) {
                    RS_f32[fq][fk][e] = __int2float_rz(RS[fq][fk][e]) * dequant_scale;
                }
            }
        }

        // masking
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
                for (uint32_t e = 0; e < 8; e++) {
                    const uint32_t q_idx = Q_idx_lane_base + fq * 64 + 8 * ((e % 4) / 2);
                    const uint32_t kv_idx =
                        (num_iterations - 1) * CTA_K + fk * 16 + 2 * (lane_id % 4) + 8 * (e / 4) + e % 2;

                    bool is_out_of_bounds;

                    if constexpr (mask_mode == MaskMode::kCausal) {
                        is_out_of_bounds = (kv_idx > q_idx) || (kv_idx >= kv_len);
                    }
                    else {
                        is_out_of_bounds = (kv_idx >= kv_len);
                    }

                    if (is_out_of_bounds) {
                        RS_f32[fq][fk][e] = -5000000.0f;
                    }
                }
            }
        }

        update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(RS_f32, RO, row_max, denom, sm_scale);

        // accumulate denom on thread basis
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
                denom[fq][0] += (RS_f32[fq][fk][0] + RS_f32[fq][fk][1] + RS_f32[fq][fk][4] + RS_f32[fq][fk][5]);
                denom[fq][1] += (RS_f32[fq][fk][2] + RS_f32[fq][fk][3] + RS_f32[fq][fk][6] + RS_f32[fq][fk][7]);
            }
        }

        uint32_t RS_f8[num_tiles_q][num_tiles_pv_inner][4];
        RS_f32_to_f8<num_tiles_q, num_tiles_k>(RS_f32, RS_f8);

        // wait for V
        wait(&barrier_V, phase);

        float RO_tmp[num_tiles_q][num_tiles_v][8];
        wgmma::warpgroup_arrive();
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
            wgmma::wgmma_f8f8f32<head_dim, 0, CTA_K>(RO_tmp[fq], RS_f8[fq][0], &sV[0]);
#pragma unroll
            for (uint32_t v_it = 1; v_it < num_tiles_pv_inner; v_it++) {
                wgmma::wgmma_f8f8f32<head_dim, 1, CTA_K>(RO_tmp[fq], RS_f8[fq][v_it], &sV[v_it * 32]);
            }
        }

        wgmma::warpgroup_commit_batch();
        wgmma::warpgroup_wait<0>();

#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
#pragma unroll
                for (uint32_t e = 0; e < 8; e++) {
                    RO[fq][fv][e] += RO_tmp[fq][fv][e];
                }
            }
        }
    }

    normalize_d<num_tiles_q, num_tiles_v, ComputeUnit::kCudaCore>(RO, row_max, denom);

    if constexpr (fuse_v_scale) {
        float        v_scale[4];
        const float* V_scale_base_ptr = V_scale
                                        + static_cast<int64_t>(batch_id) * (num_qo_heads / qo_per_kv_head) * head_dim
                                        + static_cast<int64_t>(head_id / qo_per_kv_head) * head_dim + (lane_id % 4) * 2;
#pragma unroll
        for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
            ((float2*)v_scale)[0] = *((float2*)(V_scale_base_ptr + fv * 16));
            ((float2*)v_scale)[1] = *((float2*)(V_scale_base_ptr + fv * 16 + 8));

#pragma unroll
            for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
                RO[fq][fv][0] *= v_scale[0];
                RO[fq][fv][1] *= v_scale[1];
                RO[fq][fv][2] *= v_scale[0];
                RO[fq][fv][3] *= v_scale[1];
                RO[fq][fv][4] *= v_scale[2];
                RO[fq][fv][5] *= v_scale[3];
                RO[fq][fv][6] *= v_scale[2];
                RO[fq][fv][7] *= v_scale[3];
            }
        }
    }

    DTypeOut* O_lane_ptr = O + static_cast<int64_t>(batch_id) * stride_batch_o
                           + static_cast<int64_t>(head_id) * stride_h_o
                           + static_cast<int64_t>(cta_idx_q * CTA_Q + warp_idx * 16 + (lane_id / 4)) * stride_seq_o
                           + static_cast<int64_t>((lane_id % 4) * 2);
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t fv = 0; fv < head_dim / 16; fv++) {
            if (Q_idx_lane_base + fq * 64 < qo_len) {
                if constexpr (std::is_same<DTypeOut, half>::value) {
                    ((half2*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16))[0] =
                        __float22half2_rn(((float2*)(RO[fq][fv]))[0]);
                    ((half2*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8))[0] =
                        __float22half2_rn(((float2*)(RO[fq][fv]))[2]);
                }
                else {
                    ((nv_bfloat162*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16))[0] =
                        __float22bfloat162_rn(((float2*)(RO[fq][fv]))[0]);
                    ((nv_bfloat162*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8))[0] =
                        __float22bfloat162_rn(((float2*)(RO[fq][fv]))[2]);
                }
            }

            if (Q_idx_lane_base + fq * 64 + 8 < qo_len) {
                if constexpr (std::is_same<DTypeOut, half>::value) {
                    ((half2*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8 * stride_seq_o))[0] =
                        __float22half2_rn(((float2*)(RO[fq][fv]))[1]);
                    ((half2*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8 + 8 * stride_seq_o))[0] =
                        __float22half2_rn(((float2*)(RO[fq][fv]))[3]);
                }
                else {
                    ((nv_bfloat162*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8 * stride_seq_o))[0] =
                        __float22bfloat162_rn(((float2*)(RO[fq][fv]))[1]);
                    ((nv_bfloat162*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8 + 8 * stride_seq_o))[0] =
                        __float22bfloat162_rn(((float2*)(RO[fq][fv]))[3]);
                }
            }
        }

        if constexpr (return_lse) {
            // only works for CTA_Q = 64
            uint32_t lse_idx      = cta_idx_q * CTA_Q + lane_id / 4 + 8 * (lane_id % 4) + 16 * warp_idx;
            float*   lse_lane_ptr = Lse + static_cast<int64_t>(batch_id) * (static_cast<int64_t>(qo_len) * num_qo_heads)
                                  + static_cast<int64_t>(head_id) * qo_len + lse_idx;
            uint32_t fq = (lane_id % 4) / 2;
            uint32_t e  = (lane_id % 4) % 2;

            if (lse_idx < qo_len && (lane_id % 4) < 2) {
                lse_lane_ptr[0] = (math::ptx_log2(denom[fq][e]) + row_max[fq][e]);
            }
        }
    }
}

torch::Tensor qk_int8_sv_f8_accum_f32_attn_inst_buf(torch::Tensor query,
                                                    torch::Tensor key,
                                                    torch::Tensor value,
                                                    torch::Tensor output,
                                                    torch::Tensor query_scale,
                                                    torch::Tensor key_scale,
                                                    int           tensor_layout,
                                                    int           is_causal,
                                                    int           qk_quant_gran,
                                                    float         sm_scale,
                                                    int           return_lse)
{
    const c10::cuda::CUDAGuard device_guard(query.device());
    cudaStream_t               stream = at::cuda::getCurrentCUDAStream();

    QKVLayout qkv = qkv_layout_parse<QKVFamily::kSVF8TMA>(query,
                                                          key,
                                                          value,
                                                          output,
                                                          query_scale,
                                                          key_scale,
                                                          /*value_scale_opt=*/nullptr,
                                                          /*value_mean_opt=*/nullptr,
                                                          tensor_layout,
                                                          return_lse);
    SAGEATTN_QKV_LAYOUT_LOCALS_FP8(qkv);

    auto out_dtype = output.scalar_type();

    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
            DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
                DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
                    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(out_dtype, DTypeOut, {
                        constexpr int CTA_Q       = 64;
                        constexpr int CTA_K       = 128;
                        constexpr int NUM_THREADS = 128;

                        constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

                        TORCH_CHECK(value.size(3) >= div_ceil(kv_len, CTA_K) * CTA_K,
                                    "value.size(3) >= div_ceil(kv_len, CTA_K) * CTA_K failed: value.size(3)=",
                                    value.size(3),
                                    ", kv_len=",
                                    kv_len,
                                    ", CTA_K=",
                                    CTA_K);

                        if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp)) {
                            CHECK_SHAPE(
                                query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (NUM_THREADS / 32));
                            CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K));
                        }
                        else if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread)) {
                            CHECK_SHAPE(query_scale,
                                        batch_size,
                                        num_qo_heads,
                                        div_ceil(qo_len, CTA_Q) * (NUM_THREADS / 32) * 8);
                            CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * 4);
                        }
                        else {
                            static_assert(QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp)
                                              || QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread),
                                          "Unsupported quantization granularity");
                        }

                        CUtensorMap tma_map_Q =
                            create_tensor_map_4D<CTA_Q, HEAD_DIM>(reinterpret_cast<int8_t*>(query.data_ptr()),
                                                                  batch_size,
                                                                  num_qo_heads,
                                                                  qo_len,
                                                                  HEAD_DIM,
                                                                  stride_batch_q,
                                                                  stride_h_q,
                                                                  stride_seq_q);
                        CUtensorMap tma_map_K =
                            create_tensor_map_4D<CTA_K, HEAD_DIM>(reinterpret_cast<int8_t*>(key.data_ptr()),
                                                                  batch_size,
                                                                  num_kv_heads,
                                                                  kv_len,
                                                                  HEAD_DIM,
                                                                  stride_batch_k,
                                                                  stride_h_k,
                                                                  stride_seq_k);
                        CUtensorMap tma_map_V =
                            create_tensor_map_4D<HEAD_DIM, CTA_K>(reinterpret_cast<int8_t*>(value.data_ptr()),
                                                                  batch_size,
                                                                  num_kv_heads,
                                                                  HEAD_DIM,
                                                                  value.size(3),
                                                                  stride_batch_v,
                                                                  stride_h_v,
                                                                  stride_d_v);

                        auto*  kernel     = qk_int8_sv_f8_attn_kernel<CTA_Q,
                                                                 CTA_K,
                                                                 NUM_THREADS,
                                                                 HEAD_DIM,
                                                                 static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                                 static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                                 DTypeOut,
                                                                 mask_mode,
                                                                 RETURN_LSE,
                                                                 false>;
                        size_t smem_bytes = CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t)
                                            + CTA_K * HEAD_DIM * sizeof(int8_t);
                        sage::set_max_dynamic_smem_once(kernel, smem_bytes, query.get_device());

                        dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
                        kernel<<<grid, NUM_THREADS, smem_bytes, stream>>>(
                            tma_map_Q,
                            tma_map_K,
                            tma_map_V,
                            reinterpret_cast<float*>(query_scale.data_ptr()),
                            reinterpret_cast<float*>(key_scale.data_ptr()),
                            nullptr,
                            reinterpret_cast<DTypeOut*>(output.data_ptr()),
                            (RETURN_LSE) ? reinterpret_cast<float*>(lse.data_ptr()) : nullptr,
                            stride_batch_o,
                            stride_h_o,
                            stride_seq_o,
                            qo_len,
                            kv_len,
                            qo_per_kv_head,
                            sm_scale);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    });
                });
            });
        });
    });

    return lse;
}

torch::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(torch::Tensor query,
                                                                 torch::Tensor key,
                                                                 torch::Tensor value,
                                                                 torch::Tensor output,
                                                                 torch::Tensor query_scale,
                                                                 torch::Tensor key_scale,
                                                                 torch::Tensor value_scale,
                                                                 int           tensor_layout,
                                                                 int           is_causal,
                                                                 int           qk_quant_gran,
                                                                 float         sm_scale,
                                                                 int           return_lse)
{
    const c10::cuda::CUDAGuard device_guard(query.device());
    cudaStream_t               stream = at::cuda::getCurrentCUDAStream();

    QKVLayout qkv = qkv_layout_parse<QKVFamily::kSVF8TMA>(query,
                                                          key,
                                                          value,
                                                          output,
                                                          query_scale,
                                                          key_scale,
                                                          &value_scale,
                                                          /*value_mean_opt=*/nullptr,
                                                          tensor_layout,
                                                          return_lse);
    SAGEATTN_QKV_LAYOUT_LOCALS_FP8(qkv);

    auto out_dtype = output.scalar_type();

    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
            DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
                DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
                    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(out_dtype, DTypeOut, {
                        constexpr int CTA_Q       = 64;
                        constexpr int CTA_K       = 128;
                        constexpr int NUM_THREADS = 128;

                        constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

                        TORCH_CHECK(value.size(3) >= div_ceil(kv_len, CTA_K) * CTA_K,
                                    "value.size(3) >= div_ceil(kv_len, CTA_K) * CTA_K failed: value.size(3)=",
                                    value.size(3),
                                    ", kv_len=",
                                    kv_len,
                                    ", CTA_K=",
                                    CTA_K);

                        if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp)) {
                            CHECK_SHAPE(
                                query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (NUM_THREADS / 32));
                            CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K));
                        }
                        else if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread)) {
                            CHECK_SHAPE(query_scale,
                                        batch_size,
                                        num_qo_heads,
                                        div_ceil(qo_len, CTA_Q) * (NUM_THREADS / 32) * 8);
                            CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * 4);
                        }
                        else {
                            static_assert(QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp)
                                              || QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread),
                                          "Unsupported quantization granularity");
                        }

                        CHECK_SHAPE(value_scale, batch_size, num_kv_heads, head_dim);

                        CUtensorMap tma_map_Q =
                            create_tensor_map_4D<CTA_Q, HEAD_DIM>(reinterpret_cast<int8_t*>(query.data_ptr()),
                                                                  batch_size,
                                                                  num_qo_heads,
                                                                  qo_len,
                                                                  HEAD_DIM,
                                                                  stride_batch_q,
                                                                  stride_h_q,
                                                                  stride_seq_q);
                        CUtensorMap tma_map_K =
                            create_tensor_map_4D<CTA_K, HEAD_DIM>(reinterpret_cast<int8_t*>(key.data_ptr()),
                                                                  batch_size,
                                                                  num_kv_heads,
                                                                  kv_len,
                                                                  HEAD_DIM,
                                                                  stride_batch_k,
                                                                  stride_h_k,
                                                                  stride_seq_k);
                        CUtensorMap tma_map_V =
                            create_tensor_map_4D<HEAD_DIM, CTA_K>(reinterpret_cast<int8_t*>(value.data_ptr()),
                                                                  batch_size,
                                                                  num_kv_heads,
                                                                  HEAD_DIM,
                                                                  value.size(3),
                                                                  stride_batch_v,
                                                                  stride_h_v,
                                                                  stride_d_v);

                        auto*  kernel     = qk_int8_sv_f8_attn_kernel<CTA_Q,
                                                                 CTA_K,
                                                                 NUM_THREADS,
                                                                 HEAD_DIM,
                                                                 static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                                 static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                                 DTypeOut,
                                                                 mask_mode,
                                                                 RETURN_LSE,
                                                                 true>;
                        size_t smem_bytes = CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t)
                                            + CTA_K * HEAD_DIM * sizeof(int8_t);
                        sage::set_max_dynamic_smem_once(kernel, smem_bytes, query.get_device());

                        dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
                        kernel<<<grid, NUM_THREADS, smem_bytes, stream>>>(
                            tma_map_Q,
                            tma_map_K,
                            tma_map_V,
                            reinterpret_cast<float*>(query_scale.data_ptr()),
                            reinterpret_cast<float*>(key_scale.data_ptr()),
                            reinterpret_cast<float*>(value_scale.data_ptr()),
                            reinterpret_cast<DTypeOut*>(output.data_ptr()),
                            (RETURN_LSE) ? reinterpret_cast<float*>(lse.data_ptr()) : nullptr,
                            stride_batch_o,
                            stride_h_o,
                            stride_seq_o,
                            qo_len,
                            kv_len,
                            qo_per_kv_head,
                            sm_scale);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    });
                });
            });
        });
    });

    return lse;
}
}  // namespace sm90
}  // namespace sage
