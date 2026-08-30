# mbarrier ledger — `qk_int_sv_f8_cuda_sm100.cu` (design plan §5.3)

Every mbarrier × iteration × phase for the MVP kernel. `T = num_iterations =
div_ceil(causal ? min(kv_len, (bx+1)·CTA_Q) : kv_len, CTA_K)`, iterations
`i = 0 .. T-1`; the last iteration (`i = T-1`) is peeled (fused mask, no
prefetch). Invariant proved below: **each barrier completes exactly one phase
per use, every completion has exactly one matching `try_wait.parity`, and
each barrier's phase bit flips exactly once per KV tile.**

All five barriers are `mbarrier.init(count = 1)` by thread 0, followed by
`__syncthreads()`.

## Barrier roles

| barrier | producer (completes a phase) | consumer (`wait(bar, phase)`) |
|---|---|---|
| `barrier_Q` | TMA: `expect_tx(QBYTES)` + `cp.async.bulk.tensor` complete_tx (thread 0, once) | all threads, once, phase 0 |
| `barrier_K` | TMA: `expect_tx(KBYTES)` + K-tile load (thread 0, once per tile) | all threads, once per tile |
| `barrier_V` | TMA: `expect_tx(KBYTES)` + V-tile load (thread 0, once per tile) | all threads, once per tile |
| `barrier_S_done` | `tcgen05.commit` after QK MMA chain (elected thread, once per tile) | all threads, once per tile |
| `barrier_O_done` | `tcgen05.commit` after PV MMA chain (elected thread, once per tile) | all threads, once per tile |

Note `expect_tx` + transaction-bytes arrival counts as the single completion
for the TMA barriers (init count 1 is consumed by the `arrive.expect_tx`
itself); for the commit barriers, `tcgen05.commit`'s `mbarrier::arrive::one`
is the single arrival, performed by hardware when all previously issued
`tcgen05.mma` of this CTA have retired.

## Per-tile event order (source order inside `process_tile(is_last, i)`)

1. `wait(barrier_K, k_phase)`; `k_phase ^= 1`
2. elected: QK MMA ×(HD/32) → `commit(barrier_S_done)`
3. `wait(barrier_S_done, s_phase)`; `s_phase ^= 1`
4. if `!is_last`: thread 0 `expect_tx` + TMA K(i+1)   ← K smem free (QK retired)
5. softmax: `tcgen05.ld` S, math, `tcgen05.ld/st` O-correction (i>0), P store
6. `fence::before_thread_sync` → `__syncthreads()` → `fence::after_thread_sync`
7. `wait(barrier_V, v_phase)`; `v_phase ^= 1`
8. elected: PV MMA ×(CTA_K/32) → `commit(barrier_O_done)`
9. `wait(barrier_O_done, o_phase)`; `o_phase ^= 1`   ← V smem + P free, O valid
10. if `!is_last`: thread 0 `expect_tx` + TMA V(i+1)

## Full ledger (phases the consumer waits on; all phase vars start at 0)

| tile | barrier_Q | barrier_K | barrier_V | barrier_S_done | barrier_O_done |
|---|---|---|---|---|---|
| pre-loop | load Q; **wait ph 0** | load K₀ (thread 0) | load V₀ (thread 0) | — | — |
| i = 0 | — | **wait ph 0**; load K₁ (if T>1) | **wait ph 0**; load V₁ (if T>1) | commit; **wait ph 0** | commit; **wait ph 0** |
| i = 1 | — | **wait ph 1**; load K₂ | **wait ph 1**; load V₂ | commit; **wait ph 1** | commit; **wait ph 1** |
| i = 2k | — | **wait ph 0**; load K | **wait ph 0**; load V | commit; **wait ph 0** | commit; **wait ph 0** |
| i = 2k+1 | — | **wait ph 1**; load K | **wait ph 1**; load V | commit; **wait ph 1** | commit; **wait ph 1** |
| **i = T-1 (peeled)** | — | **wait ph (T-1)&1**; *no load* | **wait ph (T-1)&1**; *no load* | commit; **wait ph (T-1)&1** | commit; **wait ph (T-1)&1** |
| epilogue | — | — | — | — | — (O already waited in tile T-1) |

### Count check (deadlock-freedom)

- `barrier_K`: T loads issued (1 pre-loop + T-1 in tiles 0..T-2, tile i loads
  K_{i+1}) = T completions; T waits (one per tile) on phases 0,1,0,1,…  ✓
- `barrier_V`: identical shape, loads issued at step 10 of tiles 0..T-2 plus
  the pre-loop V₀ = T completions, T waits. Because the V(i+1) load is issued
  only **after** `wait(barrier_O_done)` of tile i (step 9 → 10), the TMA can
  never overwrite `sV` while the PV MMA of tile i still reads it.  ✓
- `barrier_S_done` / `barrier_O_done`: exactly one commit and one wait per
  tile; commit is issued by the single elected thread, wait by all 128.  ✓
- `barrier_Q`: 1 completion, 1 wait (phase 0), never reused.  ✓
- **T = 1 degenerate case**: main loop body runs 0 times; the peeled tile is
  i = 0 and waits phase 0 on K/V/S/O — matching the single pre-loop K₀/V₀
  loads and the single commits. No prefetches are issued.  ✓

### Cross-tile hazards discharged by the ledger

| hazard | discharged by |
|---|---|
| next QK MMA overwrites S while softmax still reads it | softmax `tcgen05.ld` + `wait::ld` (own thread) happen before step 6's fence+`__syncthreads()`; the next tile's QK MMA is issued by the elected thread only after that barrier (and after `wait(barrier_O_done)` of this tile) |
| TMA K(i+1) overwrites sK while QK(i) reads it | step 4 runs after `wait(barrier_S_done)` (QK retired) |
| TMA V(i+1) overwrites sV while PV(i) reads it | step 10 runs after `wait(barrier_O_done)` (PV retired) |
| correction (tile i+1) reads O before PV(i) finished | `wait(barrier_O_done)` at step 9 of tile i |
| P store overwrites S cols [32,64) while softmax still needs them | same thread, same TMEM lane: every S `tcgen05.ld` + `wait::ld` is program-ordered before the P `tcgen05.st`, and no thread reads another thread's S |
| next QK MMA overwrites the columns P lives in while PV(i) still reads P | `wait(barrier_O_done)` at step 9 of tile i precedes the elected thread's next `mma_i8_ss` |
| PV(i) reads P before all 128 threads stored it | step 5 `wait::st` (own thread) + step 6 fence/`__syncthreads()`/fence handoff |
| PV(i) reads O before correction sts landed | same step 6 handoff (correction st is a prior tcgen05.st of the producer threads) |
| epilogue reads O before PV(T-1) done | `wait(barrier_O_done)` in the peeled tile |
| `tmem_dealloc` while other warps still read TMEM | epilogue `wait::ld` per thread + final fence/`__syncthreads()`/fence before warp 0 deallocs |
