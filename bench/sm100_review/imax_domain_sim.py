# Copyright (c) 2025 by SageAttention team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bit-level check for the wave22 int-domain row max (C1_DESIGN.md 14.4).

Claim under test: for the int32 S values of the ws softmax step
(|S| <= 128 * 128 * 127 < 2^24, so int32 -> f32 is exact and strictly
monotone), one I2F of the integer max is bit-identical to the fmax fold
over the per-element I2F row - for the whole row (per-warp k granularity)
and per k-scale class (per-thread granularity, class c owning columns
{8k + 2c, 8k + 2c + 1}). fmax over exact values is associative, so any
tree shape folds to the same bits as numpy's reduction.
"""

import numpy as np

rng = np.random.default_rng(0)
BOUND = 128 * 128 * 127  # hd128 int8 QK accumulation bound

for it in range(10000):
    row = rng.integers(-BOUND, BOUND + 1, size=128, dtype=np.int64)
    if it % 7 == 0:
        row[rng.integers(0, 128)] = 0  # int 0 -> +0.0 (no -0.0 source)
    if it % 11 == 0:
        row[64:] = row[:64]  # duplicated maxima
    f = row.astype(np.float32)
    assert (f.astype(np.int64) == row).all(), "conversion not exact"
    # whole-row tree (per-warp granularity)
    assert np.float32(row.max()).tobytes() == f.max().tobytes()
    # per-class trees (per-thread granularity)
    for c in range(4):
        idx = np.array([8 * (i // 2) + (i % 2) + 2 * c for i in range(32)])
        assert np.float32(row[idx].max()).tobytes() == f[idx].max().tobytes()

print("PASS: 10000 rows, whole-row + 4-class int-domain max bit-identical")
