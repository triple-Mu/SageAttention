# `enable_D` truth table — `qk_int_sv_f8_cuda_sm100.cu` (design plan §5.4)

`tcgen05.mma`'s enable-input-D predicate: `enable_D = 0` means the
accumulator is **zero-initialized** (D = A·B), `enable_D = 1` means
accumulate (D += A·B). Wrapper argument `enable_input_d` in
`tcgen05::mma_*`.

## S accumulator (QK, `kind::i8`, TMEM cols [0,128))

S is a **per-tile scratch** — it must be re-initialized every KV tile and
accumulate only across the head_dim K-steps within one tile.

| KV tile `iter` | K-step `k_it` | `enable_D` (source: `k_it > 0`) | effect |
|---|---|---|---|
| every | 0 | **0** | S = Q·K₀ᵀ (zero-init this tile's S) |
| every | 1 .. HD/32-1 | 1 | S += Q·Kₖᵀ |

SASS witness: first `UTCIMMA … !UPT` then `UTCIMMA … UPT` (constant-false /
constant-true uniform predicates).

## O accumulator (PV, `kind::f8f6f4`, TMEM cols [160,160+HD))

O is **persistent across the whole sequence** — zero-initialized exactly
once (first tile, first K-step) and accumulated everywhere else. The
flash-attention rescaling is done by the *correction pass* (read-FMUL-write
of O by `alpha`), not by `enable_D`.

| KV tile `iter` | K-step `v_it` | `enable_D` (source: `iter > 0 \|\| v_it > 0`) | effect |
|---|---|---|---|
| 0 | 0 | **0** | O = P₀·V₀ (the one global zero-init) |
| 0 | 1 .. 3 | 1 | O += P₀·Vₖ |
| ≥ 1 | 0 .. 3 | 1 | O += Pᵢ·Vₖ |

SASS witness: first `UTCQMMA … UP0` (runtime uniform predicate = `iter > 0`)
then `UTCQMMA … UPT`.

## Consistency with the correction-before-PV ordering

Per-tile order is: softmax(i) → **correction O·=αᵢ (only if `iter > 0`)** →
P store → PV(i). The two "first-time" rules interlock:

- `iter == 0`: correction is **skipped** (O TMEM holds garbage from previous
  kernels — it must not be read), and PV's first K-step has `enable_D = 0`,
  which **overwrites** the garbage. No uninitialized read exists.
- `iter > 0`: O holds `α₁…αᵢ₋₁`-corrected partial sums; correction applies
  αᵢ first, then all four PV K-steps accumulate (`enable_D = 1`).
- Degenerate `num_iterations == 1`: only the peeled tile runs with
  `iter = 0` → correction skipped, PV k-step 0 zero-inits. Same rules, no
  special case.

Both truth tables are enforced at single call sites (no duplicated
loop/peeled copies of the enable logic): `tcgen05::mma_i8_ss(..., k_it > 0)`
and `enable_d = (iter > 0) || (v_it > 0)` inside the shared
`process_tile()` body.
