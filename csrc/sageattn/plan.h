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

// The single source of truth for "which kernel configuration serves which
// device / request". It replaces three parallel copies of this knowledge in
// the old code: the per-arch ArchSpec tables in core.py, the EXT_SERVES
// predicates in setup.py, and the if/elif dispatch chain in sageattn().
//
// resolve() is pure host logic (no CUDA calls): given a compute capability
// and the user's (possibly unspecified) requests it either fills a Plan or
// sets Plan::error. The Python side primes a lookup table from it at import
// time; the CUDA impls re-run it for authoritative validation.

#pragma once

#include <optional>
#include <string>
#include <tuple>

#include "config.h"

namespace sage {

struct CC {
    int  major = 0, minor = 0;
    bool operator==(const CC& o) const
    {
        return major == o.major && minor == o.minor;
    }
    bool operator<(const CC& o) const
    {
        return std::tie(major, minor) < std::tie(o.major, o.minor);
    }
};

struct Plan {
    Backend     backend          = Backend::kSm80F16;
    QuantGran   gran             = QuantGran::kPerThread;
    PVAccum     pv               = PVAccum::kFp32;
    bool        smooth_v         = false;
    bool        smooth_v_ignored = false;  // request was downgraded (Python warns once)
    bool        pv_fp8           = false;
    VLayout     v_layout         = VLayout::kSeq;
    int         v_pad_multiple   = 0;  // 0 (fp16 path) / 64 / 128
    double      v_scale_max      = 448.0;
    bool        need_value_scale = false;
    bool        need_value_mean  = false;
    int         blk_q = 128, warp_q = 32, blk_k = 64, warp_k = 64;
    std::string error;  // non-empty => invalid request, message is user-facing
};

// req_* being nullopt means "use this backend's default". varlen asks for the
// packed [total_tokens, heads, head_dim] entry points, which serve a smaller
// set of configurations than the dense ones.
Plan resolve(CC                       cc,
             int                      head_dim,
             std::optional<Backend>   req_backend,
             std::optional<QuantGran> req_gran,
             std::optional<PVAccum>   req_pv,
             std::optional<bool>      req_smooth_v,
             bool                     varlen = false);

// Which kernel families were compiled into this extension (build_config.h).
bool backend_compiled(Backend b);

// Per-arch refinement of backend_compiled(): true when the family's fatbin
// carries a cubin (or forward-JITable PTX) the driver can load on this
// device. A compiled family can still miss a device when its gencode list
// was narrowed (SAGE_PRUNE_GENCODE) or never targeted it; a launch then dies
// with cudaErrorNoKernelImageForDevice, so the fwd entry points check this
// up front and fail with an actionable message instead.
bool backend_serves(Backend b, CC cc);

}  // namespace sage
