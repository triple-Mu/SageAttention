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

#include "plan.h"

#include <cstdlib>
#include <cstring>
#include <sstream>

#include "sageattn_build_config.h"

namespace sage {

namespace {

// tcgen05 kernels ship dark until hardware-validated (opt-in). Read once per
// process; consequence (documented): changing the env var after import has
// no effect, matching the import-time plan-table prime on the Python side.
bool sm100_tcgen05_enabled()
{
    static const bool enabled = [] {
        const char* v = std::getenv("SAGEATTN_SM100_TCGEN05");
        return v != nullptr
               && (std::strcmp(v, "1") == 0 || std::strcmp(v, "TRUE") == 0 || std::strcmp(v, "true") == 0
                   || std::strcmp(v, "YES") == 0 || std::strcmp(v, "yes") == 0);
    }();
    return enabled;
}

}  // namespace

bool backend_compiled(Backend b)
{
    switch (b) {
        case Backend::kSm80F16:
            return SAGEATTN_BUILD_SM80;
        case Backend::kSm89F8:
            return SAGEATTN_BUILD_SM89;
        case Backend::kSm90F8:
            return SAGEATTN_BUILD_SM90;
        case Backend::kSm100F8:
            return SAGEATTN_BUILD_SM100;
        case Backend::kSm120F8:
            return SAGEATTN_BUILD_SM120;
    }
    return false;
}

namespace {

std::string not_compiled_msg(Backend b, CC cc)
{
    std::ostringstream os;
    os << "sageattention was built for compute capabilities [" << SAGEATTN_BUILT_ARCHS_STR << "]; the " << name(b)
       << " kernels needed for sm_" << cc.major << cc.minor
       << " are not in this build. Rebuild with TORCH_CUDA_ARCH_LIST=" << cc.major << "." << cc.minor;
    return os.str();
}

QuantGran default_gran(Backend b)
{
    return b == Backend::kSm100F8 ? QuantGran::kPerWarp : QuantGran::kPerThread;
}

PVAccum default_pv(Backend b)
{
    switch (b) {
        case Backend::kSm80F16:
            return PVAccum::kFp32;
        case Backend::kSm89F8:
            return PVAccum::kFp32Fp16;
        case Backend::kSm90F8:
            return PVAccum::kFp32Fp32;
        case Backend::kSm100F8:
            return PVAccum::kFp32;
        case Backend::kSm120F8:
            return PVAccum::kFp32;
    }
    return PVAccum::kFp32;
}

// Legal (pv_accum, smooth_v) combinations per backend, with the historical
// "smooth_v silently ignored" downgrades preserved as smooth_v_ignored.
// Returns false if (pv) is flat-out unsupported.
bool pv_supported(Backend b, PVAccum pv)
{
    switch (b) {
        case Backend::kSm80F16:
            return pv == PVAccum::kFp32 || pv == PVAccum::kFp16 || pv == PVAccum::kFp16Fp32;
        case Backend::kSm89F8:
            return pv == PVAccum::kFp32 || pv == PVAccum::kFp32Fp32 || pv == PVAccum::kFp32Fp16;
        case Backend::kSm90F8:
            return pv == PVAccum::kFp32Fp32;
        case Backend::kSm100F8:
            return pv == PVAccum::kFp32;
        case Backend::kSm120F8:
            return pv == PVAccum::kFp32 || pv == PVAccum::kFp32Fp16;
    }
    return false;
}

// smooth_v is only fused into a kernel for these (backend, pv) pairs; all
// other legal pairs downgrade smooth_v to false (historically with a warning).
bool smooth_v_supported(Backend b, PVAccum pv)
{
    switch (b) {
        case Backend::kSm80F16:
            return pv == PVAccum::kFp16;
        case Backend::kSm89F8:
            return pv == PVAccum::kFp32;
        case Backend::kSm120F8:
            return pv == PVAccum::kFp32;
        default:
            return false;  // sm90 / sm100 have no v_mean kernels
    }
}

std::string pv_error_msg(Backend b, PVAccum pv)
{
    std::ostringstream os;
    if (b == Backend::kSm90F8 && pv == PVAccum::kFp32) {
        // historical wording (core.py: "Please use pv_accum_dtype='fp32+fp32' for sm90.")
        os << "pv_accum_dtype 'fp32' is not supported on sm90 (the wgmma fp32 "
              "accumulator is only 22-bit); use pv_accum_dtype=\"fp32+fp32\"";
    }
    else {
        os << "pv_accum_dtype \"" << name(pv) << "\" is not supported by the " << name(b) << " backend";
    }
    return os.str();
}

void fill_tiles(Plan& p, int head_dim)
{
    // Quantization tile geometry; must match the constexpr CTA_Q/CTA_K/
    // WARP_Q/WARP_K in the corresponding launcher (checked there via
    // CHECK_SHAPE on the scale tensors).
    switch (p.backend) {
        case Backend::kSm80F16:
            p.blk_q  = 128;
            p.warp_q = (head_dim == 128 && p.pv == PVAccum::kFp16Fp32) ? 16 : 32;
            p.blk_k  = 64;
            p.warp_k = 64;
            break;
        case Backend::kSm89F8:
        case Backend::kSm120F8:
            p.blk_q  = 128;
            p.warp_q = 32;
            p.blk_k  = 64;
            p.warp_k = 64;
            break;
        case Backend::kSm90F8:
            p.blk_q  = 64;
            p.warp_q = 16;
            p.blk_k  = 128;
            p.warp_k = 128;
            break;
        case Backend::kSm100F8:
            p.blk_q  = 128;
            p.warp_q = 32;
            p.blk_k  = 128;
            p.warp_k = 128;
            break;
    }
}

VLayout v_layout_of(Backend b)
{
    switch (b) {
        case Backend::kSm80F16:
            return VLayout::kSeq;
        case Backend::kSm100F8:
            return VLayout::kLinear;
        default:
            return VLayout::kMmaK16;
    }
}

int v_pad_of(Backend b)
{
    switch (b) {
        case Backend::kSm80F16:
            return 0;
        case Backend::kSm90F8:
        case Backend::kSm100F8:
            return 128;
        default:
            return 64;  // sm89 / sm120: per_channel_fp8's own 64 alignment
    }
}

}  // namespace

Plan resolve(CC                       cc,
             int                      head_dim,
             std::optional<Backend>   req_backend,
             std::optional<QuantGran> req_gran,
             std::optional<PVAccum>   req_pv,
             std::optional<bool>      req_smooth_v)
{
    Plan p{};

    // ---- 1. backend selection (mirrors core.py:139-170) ----
    // On the sm89 catch-all fallback the historical defaults change to
    // per_warp + fp32+fp16 (but, unlike the old code, an explicit user request
    // is honored rather than overwritten).
    bool forced_fallback = false;
    if (req_backend) {
        p.backend = *req_backend;
        if (!backend_compiled(p.backend)) {
            p.error = not_compiled_msg(p.backend, cc);
            return p;
        }
    }
    else if (cc < CC{8, 0}) {
        std::ostringstream os;
        os << "SageAttention requires compute capability >= 8.0, got sm_" << cc.major << cc.minor;
        p.error = os.str();
        return p;
    }
    else if (cc.major == 8 && cc.minor < 9) {  // sm80 / sm86 / sm87 / sm88
        p.backend = Backend::kSm80F16;
    }
    else if (cc == CC{8, 9}) {
        p.backend = backend_compiled(Backend::kSm89F8) ? Backend::kSm89F8 : Backend::kSm80F16;
    }
    else if (cc.major == 9) {
        p.backend = backend_compiled(Backend::kSm90F8) && cc.minor == 0 ? Backend::kSm90F8 : Backend::kSm80F16;
    }
    else if (cc.major == 12) {  // consumer Blackwell
        if (backend_compiled(Backend::kSm120F8)) {
            p.backend = Backend::kSm120F8;
        }
        else if (backend_compiled(Backend::kSm89F8)) {
            p.backend       = Backend::kSm89F8;  // sm89 fatbin carries sm_12x cubins
            forced_fallback = true;
        }
        else {
            p.backend = Backend::kSm80F16;
        }
    }
    else if (cc.major >= 10) {  // datacenter Blackwell+
        if ((cc == CC{10, 0} || cc == CC{11, 0}) && backend_compiled(Backend::kSm100F8) && sm100_tcgen05_enabled()) {
            p.backend = Backend::kSm100F8;
        }
        else if (backend_compiled(Backend::kSm89F8)) {
            p.backend       = Backend::kSm89F8;
            forced_fallback = true;
        }
        else {
            p.backend = Backend::kSm80F16;
        }
    }
    else {
        std::ostringstream os;
        os << "Unsupported CUDA architecture sm_" << cc.major << cc.minor;
        p.error = os.str();
        return p;
    }
    if (!backend_compiled(p.backend)) {
        p.error = not_compiled_msg(p.backend, cc);
        return p;
    }

    // ---- 2. defaults ----
    p.gran     = req_gran ? *req_gran : (forced_fallback ? QuantGran::kPerWarp : default_gran(p.backend));
    p.pv       = req_pv ? *req_pv : (forced_fallback ? PVAccum::kFp32Fp16 : default_pv(p.backend));
    p.smooth_v = req_smooth_v.value_or(false);

    // ---- 3. legality ----
    if (head_dim != 64 && head_dim != 128) {
        std::ostringstream os;
        os << "head_dim after padding must be 64 or 128, got " << head_dim;
        p.error = os.str();
        return p;
    }
    if (p.gran != QuantGran::kPerWarp && p.gran != QuantGran::kPerThread) {
        p.error = "qk_quant_gran must be \"per_warp\" or \"per_thread\"";
        return p;
    }
    if (!pv_supported(p.backend, p.pv)) {
        p.error = pv_error_msg(p.backend, p.pv);
        return p;
    }
    if (p.smooth_v && !smooth_v_supported(p.backend, p.pv)) {
        p.smooth_v         = false;
        p.smooth_v_ignored = true;  // Python warns once per config
    }

    // ---- 4. tile / V preprocessing parameters ----
    fill_tiles(p, head_dim);
    p.pv_fp8         = (p.backend != Backend::kSm80F16);
    p.v_layout       = v_layout_of(p.backend);
    p.v_pad_multiple = v_pad_of(p.backend);
    p.v_scale_max = ((p.backend == Backend::kSm89F8 || p.backend == Backend::kSm120F8) && p.pv == PVAccum::kFp32Fp16) ?
                        2.25 :
                        448.0;
    p.need_value_scale = p.pv_fp8;
    p.need_value_mean  = p.smooth_v;
    return p;
}

}  // namespace sage
