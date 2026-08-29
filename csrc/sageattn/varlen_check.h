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

#pragma once

#include <torch/types.h>

namespace sage {

// The host half of the cu_seqlens contract, shared by every varlen op.
//
// Only the tensor's type/shape/device are checked. Its *contents* are not:
// reading them would be a device-to-host copy on every call, which stalls the
// stream and makes the op uncapturable by a cudagraph - the one thing the
// closed-form block algebra in varlen.h was chosen to allow. A cu_seqlens
// whose last entry disagrees with total_tokens therefore reads out of bounds;
// that is documented on the Python entry point.
inline void check_cu_seqlens(const torch::Tensor& cu_seqlens, const torch::Tensor& packed, const char* name)
{
    TORCH_CHECK(cu_seqlens.is_cuda(), name, " must be on CUDA");
    TORCH_CHECK(cu_seqlens.device() == packed.device(), name, " must be on the same device as the packed tensors");
    TORCH_CHECK(cu_seqlens.scalar_type() == torch::kInt32, name, " must be int32");
    TORCH_CHECK(cu_seqlens.dim() == 1, name, " must be 1-D [batch_size + 1], got ", cu_seqlens.dim(), " dims");
    TORCH_CHECK(cu_seqlens.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(cu_seqlens.size(0) >= 2, name, " must have at least 2 entries (batch_size >= 1)");
}

}  // namespace sage
