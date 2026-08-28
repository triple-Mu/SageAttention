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

// Single vocabulary for the string-typed op parameters. Every torch.ops
// entry point parses its string arguments through these helpers, so an
// unknown value fails loudly here instead of being silently defaulted the
// way the old int-flag pipeline did (e.g. tensor_layout="hnd" used to be
// treated as HND).

#pragma once

#include <c10/util/Exception.h>
#include <c10/util/string_view.h>

namespace sage {

enum class TensorLayout : int {
    kNHD = 0,
    kHND = 1
};

// Values match the historical magic ints consumed by DISPATCH_QK_QUANT_GRAN.
enum class QuantGran : int {
    kPerBlock  = 1,
    kPerWarp   = 2,
    kPerThread = 3
};

enum class PVAccum : int {
    kFp32     = 0,  // plain fp32 accumulation
    kFp16     = 1,  // plain fp16 accumulation (sm80 only)
    kFp16Fp32 = 2,  // fp16 short-term + fp32 long-term buffer (sm80 inst_buf)
    kFp32Fp32 = 3,  // fp22/fp32 short-term + fp32 long-term (sm89/sm90 inst_buf)
    kFp32Fp16 = 4,  // fp16 short-term at double mma rate (sm89/sm120 inst_buf)
};

// Physical layout of the value tensor handed to the fp8 kernels. Shapes and
// dtypes are identical between kMmaK16 and kLinear, so this cannot be
// inferred from the data — it must travel as an explicit op argument.
enum class VLayout : int {
    kSeq    = 0,  // sm80: value in original [B,N,H,D]/[B,H,N,D] order, fp16
    kMmaK16 = 1,  // sm89/sm90/sm120: transposed + 16-token mma k-order permuted
    kLinear = 2,  // sm100 tcgen05: transposed, linear token order
};

enum class Backend : int {
    kSm80F16 = 0,
    kSm89F8  = 1,
    kSm90F8  = 2,
    kSm100F8 = 3,
    kSm120F8 = 4,
};

inline TensorLayout parse_tensor_layout(c10::string_view s)
{
    if (s == "NHD")
        return TensorLayout::kNHD;
    if (s == "HND")
        return TensorLayout::kHND;
    TORCH_CHECK(false, "tensor_layout must be \"HND\" or \"NHD\", got \"", s, "\"");
}

inline QuantGran parse_quant_gran(c10::string_view s)
{
    if (s == "per_warp")
        return QuantGran::kPerWarp;
    if (s == "per_thread")
        return QuantGran::kPerThread;
    if (s == "per_block")
        return QuantGran::kPerBlock;
    TORCH_CHECK(false, "qk_quant_gran must be \"per_warp\" or \"per_thread\", got \"", s, "\"");
}

inline PVAccum parse_pv_accum(c10::string_view s)
{
    if (s == "fp32")
        return PVAccum::kFp32;
    if (s == "fp16")
        return PVAccum::kFp16;
    if (s == "fp16+fp32")
        return PVAccum::kFp16Fp32;
    if (s == "fp32+fp32")
        return PVAccum::kFp32Fp32;
    if (s == "fp32+fp16")
        return PVAccum::kFp32Fp16;
    TORCH_CHECK(false,
                "pv_accum_dtype must be one of \"fp32\", \"fp16\", "
                "\"fp16+fp32\", \"fp32+fp32\", \"fp32+fp16\", got \"",
                s,
                "\"");
}

inline VLayout parse_v_layout(c10::string_view s)
{
    if (s == "seq")
        return VLayout::kSeq;
    if (s == "mma_k16")
        return VLayout::kMmaK16;
    if (s == "linear")
        return VLayout::kLinear;
    TORCH_CHECK(false, "v_layout must be \"seq\", \"mma_k16\" or \"linear\", got \"", s, "\"");
}

inline Backend parse_backend(c10::string_view s)
{
    if (s == "sm80")
        return Backend::kSm80F16;
    if (s == "sm89")
        return Backend::kSm89F8;
    if (s == "sm90")
        return Backend::kSm90F8;
    if (s == "sm100")
        return Backend::kSm100F8;
    if (s == "sm120")
        return Backend::kSm120F8;
    TORCH_CHECK(false,
                "backend must be one of \"sm80\", \"sm89\", \"sm90\", "
                "\"sm100\", \"sm120\", got \"",
                s,
                "\"");
}

inline const char* name(TensorLayout v)
{
    return v == TensorLayout::kNHD ? "NHD" : "HND";
}
inline const char* name(QuantGran v)
{
    switch (v) {
        case QuantGran::kPerBlock:
            return "per_block";
        case QuantGran::kPerWarp:
            return "per_warp";
        default:
            return "per_thread";
    }
}
inline const char* name(PVAccum v)
{
    switch (v) {
        case PVAccum::kFp32:
            return "fp32";
        case PVAccum::kFp16:
            return "fp16";
        case PVAccum::kFp16Fp32:
            return "fp16+fp32";
        case PVAccum::kFp32Fp32:
            return "fp32+fp32";
        default:
            return "fp32+fp16";
    }
}
inline const char* name(VLayout v)
{
    switch (v) {
        case VLayout::kSeq:
            return "seq";
        case VLayout::kMmaK16:
            return "mma_k16";
        default:
            return "linear";
    }
}
inline const char* name(Backend v)
{
    switch (v) {
        case Backend::kSm80F16:
            return "sm80";
        case Backend::kSm89F8:
            return "sm89";
        case Backend::kSm90F8:
            return "sm90";
        case Backend::kSm100F8:
            return "sm100";
        default:
            return "sm120";
    }
}

}  // namespace sage
