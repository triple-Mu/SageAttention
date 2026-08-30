"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

Tensor-layout helpers shared by the entry points (core) and the FakeTensor
kernels (ops). A leaf module: it imports nothing from the package, so either
side can use it without a cycle.
"""

from typing import Tuple


def _seq_nh_dims(tensor_layout: str) -> Tuple[int, int]:
    """(seq_dim, num_heads_dim) for a 4D q/k/v tensor in `tensor_layout`."""
    return (1, 2) if tensor_layout == "NHD" else (2, 1)
