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
warp 13 (single elected thread); warps 14/15 idle. `trip0/trip1` are the
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
| corr_epi | 2 (**M3 only**) | arrive x128 by correction (O_t staged to sO) | arrive x32 by epilogue warp (TMA store retired) | epilogue warp | correction |
| tmem_dealloc | 1 | 384 arrivals (softmax0/1 + correction, after each thread's last TMEM op) | — | mma warp (all 32), ph 0, then collective dealloc | — |

corr_epi does not exist in the M0-M2 kernel (correction stores O to global
directly, mirroring the 128-thread epilogue); it is reserved for the M3
smem+TMA-store epilogue together with warp 14 and an sO double buffer.

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
exp2+pack from the retained row (no TMEM reads); denom += d_sum
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
o_scale=exp2(v0-v1), [w] corr_full#(j-1), ld/mul/st O_t, st_wait, fence,
arrive corr_empty#(j-1); and `epilog(t)` = [w] vec_full#trip_t, ld denom,
arrive vec_empty#trip_t, [w] corr_full#(trip_t-1), O_t -> global.

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
| H6 | PV_t(j) accumulates into O_t while correction's rescale is mid-flight | mma waits `corr_empty#(j-1)` (128 arrivals, each after `wait::st` + fence of the rescale) before issuing PV_t(j) |
| H7 | correction reads O_t (rescale j / epilog) before PV_t(j-1) finished accumulating | `corr_full#(j-1)` completes when the PV chain retired (tcgen05.commit semantics) |
| H8 | softmax_t's vec store of step j+1 (or the final vec) clobbers vec_t(j) before correction read it | softmax waits `vec_empty#j` at the end of step j; correction arrives only after its vec `tcgen05.ld` + `wait::ld` |
| H9 | softmax reads S cols [0,2) after its own vec store aliased them | vacuous by construction: all four S chunks are loaded (and the row retained in registers) before the vec store; the exp2/pack segment and the P store touch no S column afterwards |
| H10 | `tmem_dealloc` while any warp still touches TMEM | 384 dealloc arrivals, each after the thread's last TMEM op (+ fence); the final PV retired transitively before correction's epilog arrivals (H7); mma warp waits ph 0 then deallocs collectively |
| H11 | generic-proxy smem writes feeding async-proxy readers | sV_scale is written pre-`__syncthreads` and read by correction over the generic proxy — plain sync suffices. No hand-written smem feeds an MMA in this kernel (Q/K/V arrive via TMA, P via TMEM), so no `fence.proxy.async` is needed in the kernel body — the probe that does hand-fill smem (p0c) carries it |

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
statement of their branches). The kernel is deadlock-free.

## 8. corr_epi (M3, design only)

Planned shape when the epilogue moves to smem + TMA store: correction
converts O_t to DTypeOut into sO[t] (2 x CTA_Q x head_dim staging buffers;
+64KB at d128 on top of the current 96KB dynamic smem — fits under 227KB),
arrives `corr_epi_full[t]` (x128); warp 14 waits, issues
`cp.async.bulk.tensor` store + `commit_group`, waits the group read-visible
(`cp.async.bulk.wait_group.read`), arrives `corr_epi_empty[t]` (x32). Two
stages = one per O tile, each used once per CTA (phases 0 only).
Correction's generic-proxy writes to sO need
`fence.proxy.async.shared::cta` before the arrive (the bulk-store engine
reads through the async proxy) — same lesson as the SS twin.
