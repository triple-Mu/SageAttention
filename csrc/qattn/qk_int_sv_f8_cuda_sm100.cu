/*
 * Copyright (c) 2025 by SageAttention team.
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

// Host launchers for the sm100 attention kernel; the kernel body, the TMEM
// plan and the sage::sm100 namespace rationale live in the impl header.
// -DSAGE_SM100_PV_FROM_SMEM makes the launchers below instantiate the PV-SS
// twin path (on-device oracle for the TS TMEM layout) instead of the default
// TS one.
#include "qk_int_sv_f8_sm100_impl.cuh"
#include "launch_utils.cuh"

namespace sage {
namespace sm100 {

// ---------------------------------------------------------------------------
// Host launchers (clone of the sm90 launcher bodies; CTA_Q 64 -> 128; the
// kernels accumulate PV in true fp32 so there is no _inst_buf variant).
// ---------------------------------------------------------------------------

#ifdef SAGE_SM100_PV_FROM_SMEM
constexpr bool kPVFromSmem = true;  // SS cross-check build (hardware-day oracle)
#else
constexpr bool kPVFromSmem = false;  // default: TS (P fed from TMEM)
#endif

// C1 warp-specialized kernel (qk_int_sv_f8_cuda_sm100_ws.cu), fuse_v_scale
// variant only. Selected per call by SAGEATTN_SM100_WS (env read once, same
// contract as SAGEATTN_SM100_TCGEN05 in plan.cpp; only the mode is cached):
//   unset / "auto"      heuristic below (default)
//   "1" / "on"  (truthy) force the ws kernel
//   "0" / "off" (other)  force the classic kernel
// Note "on" bypasses the PV_FROM_SMEM twin: the ws kernel is TS-only, so run
// the SS oracle with the switch off.
torch::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_ws(torch::Tensor query,
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
                                                           int           return_lse);

// Persistent ws kernel (qk_int_sv_f8_cuda_sm100_ws_persist.cu, Phase B).
// Reached only when the ws path is selected AND SAGEATTN_SM100_WS_PERSIST is
// truthy (opt-in, default off; auto integration waits for B200 acceptance).
torch::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_ws_persist(torch::Tensor query,
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
                                                                   int           return_lse);

namespace {

enum class Sm100WsMode {
    kAuto,
    kOn,
    kOff
};

Sm100WsMode sm100_ws_mode()
{
    static const Sm100WsMode mode = [] {
        const char* v = std::getenv("SAGEATTN_SM100_WS");
        if (v == nullptr || std::strcmp(v, "auto") == 0 || std::strcmp(v, "AUTO") == 0) {
            return Sm100WsMode::kAuto;
        }
        if (std::strcmp(v, "1") == 0 || std::strcmp(v, "on") == 0 || std::strcmp(v, "ON") == 0
            || std::strcmp(v, "TRUE") == 0 || std::strcmp(v, "true") == 0 || std::strcmp(v, "YES") == 0
            || std::strcmp(v, "yes") == 0) {
            return Sm100WsMode::kOn;
        }
        return Sm100WsMode::kOff;  // "0"/"off"/anything else (keeps the old opt-in strictness)
    }();
    return mode;
}

// SAGEATTN_SM100_WS_PERSIST, three-state like SAGEATTN_SM100_WS (env read
// once, only the mode is cached):
//   unset / "auto"  persistent for non-causal ws calls only (default). The
//                   wave20 B200 acceptance (C1_DESIGN 13.5) has persistent
//                   winning every non-causal d128 shape (+6.8..19.2% vs ws,
//                   prologue amortized 12.7x) but losing 9-18% on causal
//                   (steady-state de-phasing across work items), so causal
//                   stays on the per-tile ws kernel until that is fixed.
//   "1" / "on"      force persistent for every ws-selected call
//   "0" / "off"     never
enum class Sm100PersistMode {
    kAuto,
    kOn,
    kOff
};

Sm100PersistMode sm100_ws_persist_mode()
{
    static const Sm100PersistMode mode = [] {
        const char* v = std::getenv("SAGEATTN_SM100_WS_PERSIST");
        if (v == nullptr || std::strcmp(v, "auto") == 0 || std::strcmp(v, "AUTO") == 0) {
            return Sm100PersistMode::kAuto;
        }
        if (std::strcmp(v, "1") == 0 || std::strcmp(v, "on") == 0 || std::strcmp(v, "ON") == 0
            || std::strcmp(v, "TRUE") == 0 || std::strcmp(v, "true") == 0 || std::strcmp(v, "YES") == 0
            || std::strcmp(v, "yes") == 0) {
            return Sm100PersistMode::kOn;
        }
        return Sm100PersistMode::kOff;
    }();
    return mode;
}

bool sm100_ws_persist_pick(int is_causal)
{
    const Sm100PersistMode mode = sm100_ws_persist_mode();
    return mode == Sm100PersistMode::kOn || (mode == Sm100PersistMode::kAuto && !is_causal);
}

// Heuristic for kAuto. After G1 (raw-domain softmax) the ws kernel wins on
// every measured d128 shape, so head_dim alone decides; the r3/r4-era qo_len
// cuts (16384/32768) are retired. Wave11 B200 sweeps (dfaebb4, logs-w11):
// 22-point bench + crossover-gap grid + long-row four-state sweep cover
// qo_len 1024..131072 x causal both x b{1,4}, ws/old 1.09-1.37 with the
// worst point 1.0911 (b1 s6144 non-causal); the old short-seq loss went away
// with the r5 epilogue TMA store, the old long-seq loss with G1. d64 joins
// after the wave24 vec_full delivery fix (D64_DESIGN 8.5): the combination
// this pick actually routes to (non-causal -> persistent, causal -> per-tile
// ws, see sm100_ws_persist_pick) beats the classic kernel on all 12 measured
// d64 shapes (>=1.032; nc persistent/old geomean 1.0803, causal ws/old
// 1.0388, combined 1.0593). The earlier "only 384 of the 512 TMEM columns"
// loss attribution was disproven (D64_DESIGN 1.3). The (tensor_layout,
// is_causal) parameters stay as the hook for re-cutting if a future
// regression needs it.
bool sm100_ws_auto_pick(const torch::Tensor& query, int tensor_layout, int is_causal)
{
    (void)tensor_layout;
    (void)is_causal;
    if (query.dim() != 4) {
        return false;  // malformed input: let the classic parse report it
    }
    const int64_t head_dim = query.size(3);
    return head_dim == 128 || head_dim == 64;
}

}  // namespace

torch::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(torch::Tensor query,
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
    const Sm100WsMode ws_mode = sm100_ws_mode();
    if (ws_mode == Sm100WsMode::kOn
        || (ws_mode == Sm100WsMode::kAuto && sm100_ws_auto_pick(query, tensor_layout, is_causal))) {
        if (sm100_ws_persist_pick(is_causal)) {
            return qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_ws_persist(query,
                                                                        key,
                                                                        value,
                                                                        output,
                                                                        query_scale,
                                                                        key_scale,
                                                                        value_scale,
                                                                        tensor_layout,
                                                                        is_causal,
                                                                        qk_quant_gran,
                                                                        sm_scale,
                                                                        return_lse);
        }
        return qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_ws(query,
                                                            key,
                                                            value,
                                                            output,
                                                            query_scale,
                                                            key_scale,
                                                            value_scale,
                                                            tensor_layout,
                                                            is_causal,
                                                            qk_quant_gran,
                                                            sm_scale,
                                                            return_lse);
    }

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
                        constexpr int CTA_Q       = 128;
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

                        SAGEATTN_CHECK_QK_SCALE_SHAPES(div_ceil(qo_len, CTA_Q) * (CTA_Q / 32), div_ceil(kv_len, CTA_K));

                        CHECK_SHAPE(value_scale, batch_size, num_kv_heads, head_dim);

                        QKVTensorMaps tma_maps = make_qkv_tensor_maps<CTA_Q, CTA_K, HEAD_DIM>(query, key, value, qkv);

                        auto*  kernel     = qk_int8_sv_f8_attn_kernel_sm100<CTA_Q,
                                                                       CTA_K,
                                                                       NUM_THREADS,
                                                                       HEAD_DIM,
                                                                       static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                                       static_cast<QuantGranularity>(QK_QUANT_GRAN),
                                                                       DTypeOut,
                                                                       mask_mode,
                                                                       RETURN_LSE,
                                                                       true,
                                                                       kPVFromSmem>;
                        size_t smem_bytes = CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t)
                                            + CTA_K * HEAD_DIM * sizeof(int8_t)
                                            + (kPVFromSmem ? CTA_Q * CTA_K * sizeof(int8_t) : 0);
                        sage::set_max_dynamic_smem_once(kernel, smem_bytes, query.get_device());

                        dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
                        kernel<<<grid, NUM_THREADS, smem_bytes, stream>>>(
                            tma_maps.q,
                            tma_maps.k,
                            tma_maps.v,
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

}  // namespace sm100
}  // namespace sage
