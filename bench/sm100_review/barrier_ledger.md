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

---

# mbarrier ledger — `qk_int_sv_f8_cuda_sm100_ws.cu` (C1, 16 warps, dual Q tile)

Full account of the warp-specialized kernel's synchronization. Roles:
softmax0 = warps 0-3 (tile 0), softmax1 = warps 4-7 (tile 1), correction =
warps 8-11 (both O tiles), mma = warp 12 (single elected thread), load =
warp 13 (single elected thread), epilogue = warp 14 (single elected thread,
TMA-stores the sO staging tiles; r5 lever B); warp 15 idle. `trip0/trip1` are the
per-tile KV block counts (causal: `min(2bx+1, kblk)` / `min(2bx+2, kblk)`;
else both `kblk`). Steps are indexed `j = 0 .. trip_t-1` per tile `t`.

Notation: **#n** is the n-th completion of a barrier (0-based); a parity
wait on #n uses phase bit `n & 1`. "arrive x128" means each of the 128
threads of a warpgroup arrives once (init count 128 = one completion).

## 1. Pipeline inventory (the nine pipes)

| pipe | stages | full: completed by | empty: completed by | full consumer | empty consumer |
|---|---|---|---|---|---|
| load_q | 2 (one-shot) | TMA `expect_tx` (load, per tile) | — (never reused) | mma waits ph 0, once per tile | — |
| load_kv | 4-slot ring | TMA `expect_tx` (load; item n -> slot n%4) | `tcgen05.commit` by mma after the slot's last reader MMA | mma, ph `(n/4)&1` | load, ph `(n/4-1)&1`, items n>=4 only |
| mma_s0 / mma_s1 | 1 per tile | `tcgen05.commit` after QK_t chain | arrive x128 by softmax_t (S drained **and** P stored) | softmax_t, ph `j&1` | mma before PV_t(j), ph `j&1` |
| s0_corr / s1_corr (vec) | 1 per tile | arrive x128 by softmax_t (vec stored) | arrive x128 by correction (vec read) | correction, ph `j&1`, j = 0..trip_t | softmax_t at step end, ph `j&1`, j = 0..trip_t-1 |
| mma_corr | 2 = 1 per O tile | `tcgen05.commit` after PV_t chain | arrive x128 by correction (rescale stored) | correction, ph `j&1` | mma before PV_t(j), j>=1, ph `(j-1)&1` |
| epi_full | 2 = 1 per O tile, one-shot | arrive x128 by correction (O_t staged to sO, after `fence.proxy.async`) | — (sO never reused; no empty barrier) | epilogue warp, ph 0 | — |
| tmem_dealloc | 1 | 384 arrivals (softmax0/1 + correction, after each thread's last TMEM op) | — | mma warp (all 32), ph 0, then collective dealloc | — |

epi_full is the as-built shape of the §8 corr_epi design (r5 lever B): each
staging tile is filled exactly once per CTA, so the planned empty barrier
(buffer reuse acquire) has no waiter and is dropped.

Deviation from the cutedsl blueprint: cutedsl's `mma_corr` is one shared
2-stage ring carrying o0/o1 items alternately. Here each O tile owns a
dedicated 1-stage pipe (same barrier count). Reason: in the causal S1-only
rounds the shared ring's item sequence stops alternating, so slot = item%2
shifts parity between producer and consumer unless both track a running
item counter; dedicated pipes give the same (at the boundary: marginally
later, hence safe) acquire semantics with per-tile phase bits. Equivalence:
for the alternating prefix, shared acquire of item 2j (o0(j)) waits release
of item 2j-2 (o0(j-1)) — identical to the dedicated pipe's wait.

## 2. Event program (source order per role)

mma (elected thread; `[w]`=wait, `[c]`=tcgen05.commit):
```
[w] q_full#0, [w] kv_full K0      QK0(0)  [c] s0_full#0
[w] q_full#1                      QK1(0)  [c] s1_full#0   [c] kv_empty(K0)
[w] kv_full V0
[w] s0_empty#0                    PV0(0)  [c] corr0_full#0
for i = 1 .. trip0-1:
  [w] kv_full K_i                 QK0(i)  [c] s0_full#i
  [w] corr1_empty#(i-2 if i>=2)   -- skipped for the first PV1
  [w] s1_empty#(i-1)              PV1(i-1)[c] corr1_full#(i-1)  [c] kv_empty(V_{i-1})
                                  QK1(i)  [c] s1_full#i   [c] kv_empty(K_i)
  [w] kv_full V_i
  [w] corr0_empty#(i-1)
  [w] s0_empty#i                  PV0(i)  [c] corr0_full#i
for i = trip0 .. trip1-1:         -- causal S1-only rounds
  [w] kv_full K_i
  [w] corr1_empty#(i-2)           -- skipped if PV1 has not run yet
  [w] s1_empty#(i-1)              PV1(i-1)[c] corr1_full#(i-1)  [c] kv_empty(V_{i-1})
                                  QK1(i)  [c] s1_full#i   [c] kv_empty(K_i)
  [w] kv_full V_i
tail:
  [w] corr1_empty#(trip1-2)       -- skipped if trip1 == 1
  [w] s1_empty#(trip1-1)          PV1(trip1-1) [c] corr1_full#(trip1-1) [c] kv_empty(V_last)
(whole warp) [w] dealloc ph0 -> tcgen05.dealloc(512)
```

load (elected thread): issue order `Q0, K0, Q1, V0, (K_i, V_i)*` for
`i = 1..trip1-1`; before loading item n >= 4 into slot n%4, wait
`kv_empty` #(n/4 - 1) of that slot.

softmax_t (each of 128 threads; step j):
```
[w] s_full#j
ld chunks 0-3 (one ld + wait::ld each) -> dequant+mask row kept in regs, m_local
m/denom update; st vec=(m_prev,row_max) -> wait::st, fence, arrive vec_full#j
exp2+pack from the retained row (no TMEM reads);
  denom = fmaf(o_scale, denom, d_sum) (explicit since wave23: any control
  flow inserted between the old *= / += pair - both reverted wave22 changes
  did this - splits the basic block and stops the pair contracting on its
  own; the fmaf keeps the baseline's FFMA bit pattern in every code shape)
st P -> wait::st, fence, arrive s_empty#j
[w] vec_empty#j
```
after the loop: st final vec=(denom,row_max) on the slot acquired by the
last step's `[w] vec_empty#(trip_t-1)`; arrive vec_full#trip_t; (LSE store);
arrive dealloc.

correction (each of 128 threads):
```
discard: [w] vec0_full#0, arrive vec0_empty#0 ; same for vec1   -- j=0: O is
                                        overwritten by PV(0), no rescale
for j = 1 .. trip0-1:   rescale(0,j) ; rescale(1,j)
for j = trip0 .. trip1-1:   rescale(1,j)                        -- S1-only
epilog(0) ; epilog(1) ; arrive dealloc
```
where `rescale(t,j)` = [w] vec_full#j, ld vec, ld_wait, arrive vec_empty#j,
o_scale=exp2(v0-v1), warp vote `all(o_scale==1.0f)`, [w] corr_full#(j-1),
{ld/mul/st O_t, st_wait} skipped when the vote is unanimous (the multiply is
the identity; wave14 ballot skip), fence, arrive corr_empty#(j-1) — the
barrier traffic is unconditional, so every count/phase below is untouched
and corr_empty#(j-1) now certifies "rescale stored **or** skipped as the
identity"; and `epilog(t)` = [w] vec_full#trip_t, ld denom,
arrive vec_empty#trip_t, [w] corr_full#(trip_t-1), O_t -> sO[t] (swizzled
STS), `fence.proxy.async`, arrive epi_full[t].

epilogue warp 14 (elected thread):
```
[w] epi_full[0] ph0 -> TMA store sO[0] boxes -> commit_group
[w] epi_full[1] ph0 -> TMA store sO[1] boxes -> commit_group
cp.async.bulk.wait_group.read 0     -- sO must outlive the bulk reads
```

## 3. Count check (per CTA)

| barrier | completions | waits | balance |
|---|---|---|---|
| q_full[t] | 1 | 1 (mma, ph 0) | 1:1 |
| kv_full[slot] | one per item mapped to the slot; totals 2*trip1 | 2*trip1 (mma) | 1:1 per item |
| kv_empty[slot] | 2*trip1 commits (one per item) | max(2*trip1-4, 0) (load, items n>=4) | last <=4 completions unconsumed — harmless (only unmatched **waits** deadlock) |
| s_full[t] | trip_t | trip_t (softmax_t) | 1:1 |
| s_empty[t] | trip_t | trip_t (mma, one per PV_t) | 1:1 |
| vec_full[t] | trip_t + 1 | trip_t + 1 (correction) | 1:1 |
| vec_empty[t] | trip_t + 1 | trip_t (softmax_t) | final completion unconsumed — harmless |
| corr_full[t] | trip_t | trip_t (correction: trip_t-1 rescales + epilog) | 1:1 |
| corr_empty[t] | trip_t - 1 | trip_t - 1 (mma, PV_t(j), j>=1) | 1:1 |
| epi_full[t] | 1 (128 arrivals) | 1 (epilogue warp, ph 0) | 1:1 |
| dealloc | 1 (384 arrivals) | 1 (mma warp, ph 0) | 1:1 |

Degenerate traces:
* `trip0 = trip1 = 1` (kv <= 128): main and S1-only loops empty. mma:
  QK0(0), QK1(0), PV0(0), tail PV1(0) with all corr_empty waits skipped
  (`pv1_started` false; PV0(0) is the prologue call). softmax_t: one step +
  final vec. correction: two discards, no rescale, two epilogs waiting
  corr_full#0 each. Every count above holds with trip_t = 1.
* `trip1 = trip0` (non-causal, or causal tail clamped by kblk): S1-only
  loop empty; nothing else changes.
* tail CTA with `qo_len - 256*bx <= 128`: tile 1 is fully OOB. TMA
  zero-fills sQ1 (int8 OOB fill = 0), all tile-1 arithmetic runs on zeros
  (finite: row_max = -S_FP8_OFFSET after step 0, denom > 0), stores and LSE
  are suppressed by `q_idx < qo_len`, and the q-scale block index is
  clamped. The pipes run at full trip1 — correctness unaffected, the waste
  is bounded by one tile (same as cutedsl, core_sm100.py L995-1004).

## 4. Phase-alias freedom

`mbarrier.try_wait.parity` distinguishes only two phases, so a wait for
completion #n is correct only if completion #(n+2) cannot happen before the
waiter observes #n. Every pipe above has effective depth 1 with respect to
its own barrier (per slot / per tile): the producer of completion #(n+1)
first passes a wait that requires the consumer to have finished consuming
#n (mma_s/vec/mma_corr: the chains in section 5; kv ring: the load of lap
m+1 waits the empty completion of lap m, which the mma commits only after
consuming lap m's full). Hence #(n+2) is transitively behind the consumer's
#n wait and aliasing cannot occur. The one-shot barriers (q_full, dealloc)
complete once. (This is also why the probes use a dedicated one-shot
dealloc barrier instead of re-waiting a per-iteration barrier at an
arbitrary later phase — see p0c_umma_pipeline.cu.)

## 5. Cross-warp TMEM/smem hazards

| # | hazard | discharged by |
|---|---|---|
| H1 | QK_t(j+1) overwrites S_t while softmax_t still reads step j | mma waits `s_empty#j` before PV_t(j), and QK_t(j+1) is issued after that wait in the elected thread's program order; softmax arrives only after its last `tcgen05.ld` of step j completed (`wait::ld` per chunk) and its P `wait::st` + `fence::before_thread_sync` |
| H2 | QK_t(j+1) overwrites P_t cols [32,64) while PV_t(j) still reads P_t | both are UMMA ops issued by the same elected thread into one in-order UMMA queue; PV_t(j) is issued first, so it consumes P before QK_t(j+1) writes (same argument the CUTLASS sm100 FMHA mainloop and the 128-thread kernel's TMEM_COL_P overlay rely on) |
| **H3** | **QK_t(j+1) overwrites vec_t cols [0,2) before correction read vec_t(j)** | see section 6 |
| H4 | TMA K/V load overwrites a ring slot still read by an MMA | load waits `kv_empty` of the previous lap; that barrier completes via `tcgen05.commit` issued after the slot's last reader (K_i: QK1(i); V_i: PV1(i)), and commit fires only when those MMAs **retired** |
| H5 | PV_t(j) reads P_t(j) before all 128 softmax threads stored it | `s_empty#j` completes only after 128 arrivals, each preceded by that thread's P `wait::st` + fence; mma issues `fence::after_thread_sync` after the wait |
| H6 | PV_t(j) accumulates into O_t while correction's rescale is mid-flight | mma waits `corr_empty#(j-1)` (128 arrivals, each after `wait::st` + fence of the rescale) before issuing PV_t(j); a warp that ballot-skipped the rescale (wave14) has no in-flight TMEM write to order — its arrival is trivially safe |
| H7 | correction reads O_t (rescale j / epilog) before PV_t(j-1) finished accumulating | `corr_full#(j-1)` completes when the PV chain retired (tcgen05.commit semantics) |
| H8 | softmax_t's vec store of step j+1 (or the final vec) clobbers vec_t(j) before correction read it | softmax waits `vec_empty#j` at the end of step j; correction arrives only after its vec `tcgen05.ld` + `wait::ld` |
| H9 | softmax reads S cols [0,2) after its own vec store aliased them | vacuous by construction: all four S chunks are loaded (and the row retained in registers) before the vec store; the exp2/pack segment and the P store touch no S column afterwards |
| H10 | `tmem_dealloc` while any warp still touches TMEM | 384 dealloc arrivals, each after the thread's last TMEM op (+ fence); the final PV retired transitively before correction's epilog arrivals (H7); mma warp waits ph 0 then deallocs collectively |
| H11 | generic-proxy smem writes feeding async-proxy readers | sV_scale / sK_scale are written pre-`__syncthreads` and read by correction / softmax over the generic proxy — plain sync suffices (sK_scale: section 9). The one generic->async edge in the kernel body is correction's sO staging feeding the bulk-store engine: every thread issues `fence.proxy.async.shared::cta` after its STS and before its epi_full arrival, and the epilogue warp's TMA store is ordered behind the barrier wait (SS-twin lesson applied) |
| H12 | epilogue TMA store reads sO[t] before all 128 correction threads staged it | epi_full[t] completes only after 128 arrivals, each preceded by that thread's STS + `fence.proxy.async` |
| H13 | CTA exit reclaims smem while a bulk store still reads sO | epilogue warp ends with `cp.async.bulk.wait_group.read 0` (the CUTLASS `tma_store_wait<0>` tail); global visibility of the stores is the kernel-completion fence's job |

## 6. H3: the vec0/S0[0,2) alias chain, in full

Claim: for every tile t and step j >= 1, correction's read of vec_t(j) (the
`(m_prev, row_max)` pair softmax stored in step j) completes before
QK_t(j+1) can write TMEM columns [0,2) of S_t. For j = 0 the pair is never
read (discarded; O is overwritten by PV_t(0) with enable_D=0), so the
overwrite of vec_t(0) by QK_t(1) is vacuously safe. The final vec (denom)
has no subsequent QK.

The chain, entirely on the issue side (no reliance on MMA timing):

1. correction, rescale(t,j): wait `vec_full#j` -> `tcgen05.ld` vec ->
   `wait::ld` (values in registers) -> ... -> O rescale -> `wait::st` ->
   `fence::before_thread_sync` -> arrive `corr_empty#(j-1)` (release).
2. mma elected thread: wait `corr_empty#(j-1)` (acquire) — this is the
   `wait_corr_empty` immediately preceding `pv(t, ..., j)` in program order.
3. QK_t(j+1) is issued by the same thread strictly after that pv() call
   (next loop iteration / S1-only round / never).
4. A tcgen05.mma cannot touch TMEM before it is issued.

1->2 is an mbarrier release/acquire edge; 2->3 is single-thread program
order; 3->4 is the instruction's own semantics. Hence read(vec_t(j)) <
write(QK_t(j+1) over cols [0,2)). The read is anchored at step 1's
`wait::ld`, which precedes the arrive in correction's program order and is
made visible by the arrive's release semantics.

The companion hazard — PV_t(j) itself reading P_t while QK_t(j+1) rewrites
the S region — is NOT covered by this chain (PV_t(j) only *issues* before
QK_t(j+1)); it needs the in-order UMMA queue property (H2). The two
arguments are deliberately separate: H3 holds even if the UMMA queue were
out of order; H2 does not.

Cross-check against the blueprints: cutedsl reaches the same guarantee
through `mma_corr_producer.acquire_and_advance()` before each PV
(core_sm100.py L650/L673/L696: acquire of item 2j blocks until release of
item 2j-2, which correction performs after consuming the matching vec); the
CUTLASS sm100 FMHA example places the same order-correction pipeline
producer-acquire before the PV GEMM. This kernel makes the acquire explicit
as `wait_corr_empty` on the dedicated per-tile pipe.

## 7. Liveness (deadlock freedom)

Argument: exhibit a total order (witness schedule) consistent with all
waits; every mbarrier wait's matching completion is produced by events
strictly earlier in that order, so no wait cycle exists.

Rank the mma spine as written in section 2 (prologue, iterations
i = 1..trip0-1, S1-only rounds, tail). Induction over spine positions:

* `kv_full` waits: item n's load needs (for n >= 4) `kv_empty#(n/4-1)` of
  its slot = the mma's release of item n-4. Releases sit at spine
  positions: K_m released in iteration m, V_m in iteration m+1. The wait
  for K_i (item 2i) needs K_{i-2} released (iteration i-2 < i); for V_i
  (item 2i+1) needs V_{i-2} released (iteration i-1 < i). Both precede the
  waiting position; the load warp itself blocks only on those releases and
  otherwise runs ahead.
* `s_empty#j` waits: completed by softmax_t finishing step j, which needs
  `s_full#j` (QK_t(j), earlier in the spine) and `vec_empty#(j-1)`
  (correction consumed vec_t(j-1), which needs `corr_full#(j-2)` =
  PV_t(j-2), earlier in the spine).
* `corr_empty#(j-1)` waits: completed by correction's rescale(t,j), which
  needs `vec_full#j` (softmax step j, needing `s_full#j` = QK_t(j); in the
  spine QK0(i) precedes PV0(i), and QK1(i) precedes PV1(i) because PV1(i)
  is issued in iteration i+1 or the tail) and `corr_full#(j-1)`
  (PV_t(j-1), earlier).
* correction's and softmax's own waits unwind identically; `dealloc`'s 384
  arrivals depend only on events shown live above.

Every wait depends only on strictly-earlier witness events, and the only
other blocking construct — setmaxnreg.inc's TRY_ALLOC retry — resolves
once the two dec warpgroups execute their unconditional dec (the first
statement of their branches). The epilogue warp (r5) adds no cycle: its
only waits are epi_full[t] (completed by correction's epilog, shown live
above) and the bulk-group drain (hardware DMA, always completes), and no
other role waits on it. The kernel is deadlock-free.

## 8. epi_full / TMA-store epilogue (as built, r5 lever B)

Correction converts O_t to DTypeOut into sO[t] — head_dim/64 boxes of
CTA_Q x 64 in the CU_TENSOR_MAP_SWIZZLE_128B layout (16B unit u of row r at
r*128 + (u ^ (r%8))*16), +64KB at d128 / +32KB at d64 on top of the
96KB/48KB dynamic smem (160KB/80KB total, under 227KB) — then per thread
`fence.proxy.async.shared::cta` (the bulk-store engine reads through the
async proxy; SS-twin lesson) and arrives `epi_full[t]` (x128). Warp 14
waits ph 0, issues one `cp.async.bulk.tensor.4d` store per box +
`commit_group` per tile, and drains with `cp.async.bulk.wait_group.read 0`
before exit (smem lifetime, H13). Deviations from the original design
sketch: no `corr_epi_empty` (each buffer is written once per CTA — nothing
ever re-acquires it) and no sO double-buffer pressure for the same reason.
The old per-row `q_idx < qo_len` store guard is replaced by the tensor
map's dim1 = qo_len bound: the TMA store clips OOB rows (tail CTA's tile 1
stores nothing).

## 9. sK_scale preload (G2, as built) — zero new barriers

Producer: every thread of the CTA, before the kernel-start `__syncthreads`
(the sV_scale/A3 pattern), copies the (batch, kv-head) K_scale row prefix
(`min(num_ctas_k, 1024) * kNumKScales` f32 words) into static smem.
Consumers: the softmax warpgroups, LDS only, strictly after that
`__syncthreads`. The buffer is written exactly once and never re-acquired
(read-only for the CTA lifetime), so there is no full/empty pair, no phase,
and no ring — the ledger's count check (section 3) and liveness argument
(section 7) are untouched. The beyond-capacity fallback (blocks >= 1024)
reads gmem directly and synchronizes with nothing, like the old path.
Deliberately NOT tied to the kv ring's stage semantics: `kv_empty(K_i)`
fires when QK(i) retires, which can precede softmax's read of block i's
scale — a per-slot k_scale copy could be overwritten one ring lap early
(checked and rejected; the one-shot prefix copy has no such lifetime).

## 10. EX2 phase gate (wave22, d64 instances): the per-step vec_empty wait moved pre-exp2 — REVERTED

Design record. Implemented in wave22, reverted after the wave23b B200
final-tree re-verification: gate alone d64 ws/old 0.9575 geomean against
the pre-registered <0.96 revert line (the fused-tree 1.0269 was carried by
the also-reverted vec_full delivery change; C1_DESIGN.md 15.5). The §2
event program is back to the tail-position wait; the correctness argument
below stays as the record that the move was semantics-safe (wave23 golden
diff=0, stress zero hangs).

d64 instances only (`head_dim == 64`; d128 code is untouched). Both softmax
warpgroups' per-step `[w] vec_empty#j` (§2 event program, last line of the
step) moves to directly after `arrive vec_full#j`. (Branch-free by design:
a tile-predicated move was tried first and ptxas carried the `tile`
predicate across the setmaxnreg boundary on the stack — 8B frame, 6 LDL
reloaded at the hot loop top, d64 c1 sm_100a — the asymmetry comes from
correction's program order, not from code.) It is the SAME wait consuming
the SAME completion at the same cumulative index — every §3 count, §4 phase
argument and §5/§6 hazard proof holds verbatim; only the wait's position
inside the softmax step moved. Dynamics: correction consumes vec_0(j) first
(it spins on vec_full_0 in the d64 steady profile, 22.9%/1.5%,
D64_DESIGN.md 7.2), so tile 0's moved wait is already complete and free;
tile 1's completes only after correction also passed the O_0 rescale — tile
1's P store (s_empty_1#j, hence QK_1(j+1), hence softmax1's next step) runs
about one rescale behind tile 0, a structural per-step phase offset between
the two softmax warpgroups' EX2 bursts on each SMSP (D64_DESIGN.md 8; the
d64 in-phase EX2 verdict is D64_DESIGN.md 7.2). SASS check (nvcc 13.3): in
this branch-free form ptxas keeps the whole exp2 segment after the moved
wait, so the offset applies within the step; scheduling is not contractual
though (the rejected predicated form had the exp2 math hoisted above the
wait — no data edge into the FP chain), and either placement yields the
same steady-state stagger.

* Deadlock (tile t): the moved wait's completion (correction's `arrive
  vec_empty_t#j`) requires `vec_full_t#j` (arrived immediately above the
  moved wait) plus correction's upstream pipes (vec_fulls of the OTHER
  tile at step <= j, `corr_full`s of steps < j) — nothing softmax_t does
  after its moved wait, and the two softmax warpgroups' pre-wait segments
  depend only on mma/load progress. The §7 witness order takes one edit:
  softmax_t's step-j P store now ranks after correction's
  rescale/discard(t,j) vec read, whose producers all rank earlier — still
  well-founded. Step 0 degenerates to the discard pair (no rescale in
  front), trip=1 to discard+epilog — both immediate.
* H3/H8 unaffected: the wait still precedes the next vec store (step j+1's
  or the final one), which is all those hazards need.
* s_empty_t#j now completes after the moved wait: mma's QK_t(j+1) / PV_t(j)
  observe a later completion in wall time, but no wait edge changed — pure
  scheduling; tile 0's wait is free in the steady profile and tile 1's
  delay is absorbed by the tensor pipe's d64 slack (13.2% active,
  D64_DESIGN.md 7.1).

(A causal-gated twin of this move on the persistent TU was tried in wave22
and reverted after the wave23 B200 acceptance: c1 wsp/ws 0.8949 against the
0.90 revert line, C1_DESIGN.md 13.6/15.)

---

# mbarrier ledger — `qk_int_sv_f8_cuda_sm100_ws_persist.cu` (Phase B, persistent)

Extension of the 16-warp ledger above. The persistent kernel launches
`grid = min(total_tiles, #SMs)` CTAs; every warp role runs a grid-stride loop
over **work items** `e = blockIdx.x, +gridDim.x, …` (one work item = one
256-row Q tile × head × batch = the old blockIdx; qblk2 decoded REVERSED =
static LPT). Nothing is ever re-initialized between work items: all
barrier-phase variables are loop-carried, and the KV ring item counter runs
globally. Everything not restated here (per-step event programs, H1-H13, the
in-order UMMA queue argument) is inherited verbatim from the 16-warp ledger —
per work item the event program is exactly the one-shot kernel's, with the
deltas below. `trip0_w / trip1_w` are work item `w`'s trip counts;
`w = 0, 1, …` indexes a CTA's own work items in order.

## P1. Deltas to the pipeline inventory

| pipe | delta vs one-shot |
|---|---|
| load_q | reused: full completes once per tile per work item (TMA expect_tx); mma waits ph `w&1` (one carried toggle covers both tiles). NEW empty side: `q_empty` (count 1), `tcgen05.commit` by mma once per work item after the work item's last QK **issue** (fires on retire of every prior MMA = both tiles' last QK, in-order queue); load waits ph `(w-1)&1` before work item `w>=1`'s Q0 load |
| load_kv | unchanged expressions: item `n` is ABSOLUTE (running base `item0 += 2*trip1_w`), so slot `n%4`, full ph `(n/4)&1`, empty wait ph `(n/4-1)&1` for `n>=4` hold across work items; the per-work-item "last <=4 empty completions unconsumed" now ARE consumed by the next work item's loads (only the global-last <=4 dangle) |
| mma_s0/s1, s_empty, corr_full | unchanged per work item; phases carried |
| s_corr (vec) | unchanged completions; NEW final wait: softmax waits `vec_empty#trip_w` (ph carried) at the END of every work item — the completion that dangled in the one-shot kernel — so the next work item's step-0 vec store owns the slot. The pipe is now exactly 1:1 |
| mma_corr empty | NEW completion: correction's `epilog(t)` also arrives `corr_empty[t]` (after its O/vec `tcgen05.ld` + `wait::ld` + `tcgen05 fence`), making completions per work item `trip_t` (rescales j=1..trip_t-1, then epilog). NEW wait: mma waits `corr_empty[t]` (ph carried) before `QK_t(0)` of work item `w>=1` — consuming the PREVIOUS work item's epilog completion (H14); `PV_t(0)` takes no wait (`pv1_started` resets per work item), `PV_t(j>=1)` waits as before |
| epi_full | reused: completes once per tile per work item; epilogue warp waits ph `w&1` (one carried toggle covers both tiles) |
| epi_empty[t] (NEW) | count 1; epilogue warp arrives after `cp.async.bulk.wait_group.read 0` (this work item's both store groups drained); correction waits ph `(w-1)&1` at the top of `epilog(t)` for `w>=1` before restaging sO[t] |
| tmem_dealloc | unchanged one-shot: all arrivals moved AFTER the roles' work-item loops |

Non-barrier note: sV_scale is restaged per work item by the correction
warpgroup under a named-barrier sandwich (`bar.sync 1, 128` / write /
`bar.sync 1, 128`). Correction is its only reader, every corr thread's
old-row reads precede its own first bar arrival (program order), and the
second bar publishes the new row — plain CTA-scope smem, no mbarrier, no
entry in the counts below. sK_scale has no smem copy at all in this kernel
(softmax reads gmem; same words, so per-tile bits are unchanged).

## P2. Count check across the work-item sequence

Per pipe, list completions and waits in GLOBAL order over a CTA's work items
w = 0..W-1; each pipe's waits match completions 1:1 in order, so the carried
phase toggles stay aligned (phase = cumulative index & 1 on both sides).

| barrier | completions (global order) | waits (global order) | dangling at exit |
|---|---|---|---|
| q_full[t] | load expect_tx, one per work item: #0..#W-1 | mma, one per work item, ph w&1 | none |
| q_empty | mma commit, one per work item: #0..#W-1 | load before work item w>=1's Q0: consumes #w-1 | #W-1 (harmless: only unmatched waits deadlock) |
| kv_full/kv_empty[slot] | one per absolute item, exactly the one-shot expressions | same | last <=4 kv_empty completions |
| s_full/s_empty[t] | trip_t_w per work item, phases carried | same counts | none |
| vec_full[t] | (trip_t_w + 1) per work item | correction: 1 discard + (trip_t_w - 1) rescales + 1 epilog per work item | none |
| vec_empty[t] | (trip_t_w + 1) per work item (correction) | softmax: trip_t_w step-end waits + 1 end-of-work-item wait | none (now exactly 1:1) |
| corr_full[t] | trip_t_w per work item (PV commits) | correction: (trip_t_w - 1) rescales + epilog | none |
| corr_empty[t] | per work item: rescale #1..#trip_t-1, then epilog | mma: [w>=1: QK_t(0)-side wait consumes the PREVIOUS epilog completion], then PV_t(j) j=1..trip_t-1 consume this work item's rescales | last epilog completion |
| epi_full[t] | one per work item (128 corr arrivals) | epilogue warp, ph w&1 | none |
| epi_empty[t] | one per work item (epilogue warp, after group drain) | correction in epilog(t) of work item w>=1: consumes #w-1 | #W-1 |
| dealloc | 1 (384 arrivals, after all loops) | 1 (mma warp, ph 0) | none |

Boundary trace (the only new interleaving), tile t, work items w -> w+1:
```
corr_empty[t]:  … rescale#trip-1(w), epilog(w)   |  [mma QK_t(0)-wait](w+1), rescale#1(w+1) …
vec_empty[t]:   … #trip-1(w), #trip(w=epilog)    |  [softmax end-wait](w) consumes #trip(w); step-0 store(w+1) owns the slot
q_empty:        commit(w)                        |  [load wait](w+1)
epi_empty[t]:   drain-arrive(w)                  |  [correction wait in epilog(t)](w+1)
```

Degenerate cases: `trip0_w = trip1_w = 1` — corr_empty[t] completions per
work item = {epilog} only; mma waits per work item (w>=1) = {QK_t(0)-side}
only (`pv1_started` false at the tail skips the PV1 wait; the PV0(0) call
never waits) — 1:1 holds. W = 1 (persistent grid degenerates to one work
item per CTA): every new wait is skipped (`e == blockIdx.x`) except the
softmax end-wait, which its own epilog completes; the ledger reduces to the
one-shot kernel's plus that wait.

## P3. Phase-alias freedom (new/changed pipes)

Same criterion as §4: a wait on completion #n is safe iff #(n+2) cannot
complete before the waiter observes #n.

* `q_full[t]`: completion #(w+1) needs load's expect_tx of work item w+1,
  which sits behind load's `q_empty#w` wait, which needs mma's commit at w,
  issued AFTER mma's `q_full#w` waits. Depth 1. ✓
* `q_empty`: completion #(w+1) needs mma to issue work item w+1's QKs,
  behind mma's `q_full#(w+1)` wait -> load's Q(w+1) loads -> load consumed
  `q_empty#w`. ✓
* `corr_empty[t]` (with the epilog completion): completion #(n+1) is
  produced by correction only after passing `vec_full`/`corr_full` waits
  whose completions sit behind mma's consumption of #n (rescale case:
  unchanged §4 argument; epilog case: `corr_full#(trip-1)` = the last PV,
  which mma issues after consuming every rescale completion of the work
  item). One extra step for the wrap: rescale#1(w+1) — two completions
  after epilog(w) when trip_t=2 — needs `vec_full#1(w+1)` = softmax step 1
  of w+1, which needs `s_full#1(w+1)` = QK_t(1) of w+1, which mma issues
  after its QK_t(0)-side wait consumed epilog(w). ✓
* `vec_empty[t]`: completions are correction's, each behind a `vec_full`
  wait whose completion is softmax's store, each behind softmax's previous
  `vec_empty` wait (now including the end-of-work-item wait) — the pipe is
  1:1 alternating. ✓
* `epi_full[t]` / `epi_empty[t]`: epi_full#(w+1) needs correction's staging
  of w+1, behind its `epi_empty#w` wait; epi_empty#(w+1) needs the epilogue
  warp to pass `epi_full#(w+1)`, behind correction's staging of w+1, behind
  `epi_empty#w`'s consumption. Depth 1 both ways. ✓
* 128-count barriers and warp drift: a correction warp cannot fall a full
  phase behind on any x128 pipe because every next completion requires all
  128 threads' arrivals, including the laggard's (the §4 rate-limit
  argument, unchanged).

## P4. New cross-work-item hazards

| # | hazard | discharged by |
|---|---|---|
| H14 | `QK_t(0)` of work item w+1 overwrites vec_t (S_t cols [0,2)) before correction's epilog(t, w) read the final `(denom, row_max)`; and `PV_t(0)` of w+1 (enable_D=0) overwrites O_t while that epilog still reads it | epilog arrives `corr_empty[t]` only after its vec + O `tcgen05.ld` all completed (`wait::ld`) + `tcgen05 fence`; mma waits that completion (+ `fence_after_sync`) BEFORE issuing `QK_t(0)` of w+1, and `PV_t(0)` follows in the same thread's program order |
| H15 | TMA Q load of work item w+1 overwrites sQ_t while a QK of w still reads it | load waits `q_empty#w`, completed by mma's `tcgen05.commit` issued after the last QK issue of w — commit fires only when those MMAs **retired** (same mechanism as H4) |
| H16 | correction's sO[t] staging of w+1 races the bulk-store engine still reading sO[t] of w (WAR) | epilogue warp: `cp.async.bulk.wait_group.read 0` (engine finished reading) precedes its `epi_empty[t]` arrival (release); correction's STS sit behind its `epi_empty#w` wait (acquire) — the CUTLASS `tma_store_wait` reuse pattern |
| H17 | softmax's step-0 vec store of w+1 clobbers the final vec of w before epilog read it | softmax's end-of-work-item `vec_empty#trip(w)` wait; correction arrives it only after the epilog's vec `wait::ld` (H8 mechanism, extended to the final completion) |
| H18 | `QK_t(0)` of w+1 overwrites P_t cols while the tail `PV_1(w)` still reads P_1 | in-order UMMA queue, same thread: tail PV1 is issued before `QK_t(0)` of w+1 (H2 argument verbatim) |

## P5. Liveness

Extend the §7 witness schedule across work items: rank each CTA's spine as
(work item 0 spine) < (work item 1 spine) < …, with the epilogue warp's and
load warp's per-work-item programs interleaved as in §7. Every NEW wait's
completion is produced strictly earlier in that order:

* load's `q_empty#w`: mma's commit inside work item w's spine (fires on MMA
  retirement — hardware progress, no wait in front of it).
* mma's QK_t(0)-side `corr_empty` wait at w+1: correction's epilog(t, w),
  whose own waits (`vec_full#trip(w)`: softmax's final vec, `corr_full`:
  PV(w) retirement, `epi_empty#(w-1)`: the epilogue warp's drain of w-1)
  are all inside work items <= w. The epilogue warp itself only waits
  `epi_full` (correction's arrivals, shown live) and the bulk-group drain
  (hardware DMA).
* softmax's end-of-work-item `vec_empty#trip(w)`: correction's epilog(t, w)
  read_vec, live per the previous point.
* No role's post-boundary code is required for any of these producers, so
  the order is well-founded and the kernel remains deadlock-free.
