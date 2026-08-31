#!/usr/bin/env python3
"""Bit-level simulation of the A' RESCALE_THRESHOLD lazy row max
(BEYOND_CUDNN_PLAN.md section 4.5, C1_DESIGN.md section 12; cudnn DSL
reference: cudnn/sdpa/fwd/kernels/prefill_d128_fp8_sm100.py:1242-1260,
update_cond = is_first | ((current_max - total_max) > RESCALE_THRESHOLD),
threshold 4.0 for fp8 via config_sm100.py rescale_threshold()).

Both paths run the shipping G1 softmax value sequence
(csrc/qattn/qk_int_sv_f8_cuda_sm100_ws.cu, modeled in g1_softmax_sim.py's
"new" path - helpers are imported from there so the model is the same code):

  base:    row_max = max(row_max, fma(m_deq, c, -8.807))       (ships today)
  lazy(T): chal    = fma(m_deq, c, -(8.807 - T))
           row_max = chal if (chal - row_max) > T else row_max

The offset drop by T keeps the P ceiling at exp2(8.807) < 448: row_max >=
chal - T holds at every step (init -5e6 acts as a stale max; an update
restores row_max >= chal), so the exp2 argument never exceeds T + (8.807-T).
A fresh max now maps P to 448*2^-T and staleness climbs it back toward 448 -
P loses up to T binades at the bottom (earlier e4m3 subnormal/zero), which
is the accuracy cost this sim quantifies. A skipped update leaves row_max
bit-identical, so the correction's o_scale == 1.0 ballot (already shipping)
hits exactly; the ballot rate per (32-row warp, block) is the perf-side
proxy reported here.

Reported per scenario x granularity x T in {2,3,4,5}:
  * upd%:    row_max update rate over (row, block>=1) - correction work;
  * ballot:  base -> lazy all-32-rows o_scale==1.0 warp-vote skip rate
             (blocks >= 1: block 0's vec is discarded by correction);
  * P-rw:    e4m3 P bitwise rewrite rate vs base (scale change: expected large);
  * P-zero:  base -> lazy fraction of live P flushed to 0 (tail loss);
  * O:       cos_sim / rel_l1 lazy-vs-fp64-SDPA-ref and lazy-vs-base
             (base-vs-ref printed once per case as the yardstick);
  * lse-max: max |lse_lazy - lse_base| (log2 domain; scale-free, so this
             isolates rounding movement).

Self-check: lazy(T=0) must reproduce base bit-for-bit on every scenario
(same offset, update condition degenerates to strict max) - hard assert.

Pathological scenarios (big dynamic range / adversarial max ramps /
out-of-domain sentinels) are listed separately in the verdict; neg_5e6
fails the accuracy gate for base and lazy alike (outside the int8 attention
representation domain, C1_DESIGN.md section 9.3) and is excluded from it.

numpy only; runs on any host. Results table: C1_DESIGN.md section 12.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1_softmax_sim as g1  # noqa: E402  (model helpers + base scenarios)

F32 = np.float32
CTA_K = g1.CTA_K
THRESHOLDS = [2.0, 3.0, 4.0, 5.0]
# exp2 ceiling of both paths: T + (S_FP8_OFFSET - T); satfinite headroom vs 448
P_CEIL = float(np.exp2(np.float64(g1.S_FP8_OFFSET)))

SCENARIOS = [
    # (name, gran list, pathological?)
    ("randn", ("pw", "pt"), False),
    ("randn_causal", ("pw", "pt"), False),
    ("randn_long", ("pw",), False),  # 128 KV blocks: s16384-shaped skip rate
    ("big_range", ("pw", "pt"), True),
    ("sharp", ("pw", "pt"), True),
    ("const_rows", ("pw", "pt"), True),
    ("outlier_block", ("pw", "pt"), True),
    ("ramp_slow", ("pw", "pt"), True),  # adversarial: max creeps ~0.5/blk
    ("ramp_fast", ("pw", "pt"), True),  # adversarial: max jumps ~2/blk
    ("zero_amax", ("pw", "pt"), True),
    ("neg_5e6", ("pw", "pt"), True),  # out-of-domain: base fails the gate too
    ("short_kv", ("pw", "pt"), True),
]


def scenario_ext(name, rng, rows=128, nblk=32, hd=128):
    """A' additions on top of g1_softmax_sim.scenario."""
    if name == "randn_long":
        kv = 128 * CTA_K
        return (rng.standard_normal((rows, hd)), rng.standard_normal((kv, hd)),
                rng.standard_normal((kv, hd)), False)
    if name in ("ramp_slow", "ramp_fast"):
        # K block magnitudes grow linearly, so the running max keeps rising
        # in the exp2-argument domain by roughly +0.5 (slow) / +2 (fast) per
        # block: staleness sits inside (0, T] for most of the trip - the
        # worst case for both the ballot rate and the stale-scale P pack.
        Q = rng.standard_normal((rows, hd))
        K = rng.standard_normal((nblk * CTA_K, hd))
        V = rng.standard_normal((nblk * CTA_K, hd))
        rate = 0.12 if name == "ramp_slow" else 0.45
        gain = 1.0 + rate * np.arange(nblk)
        K = (K.reshape(nblk, CTA_K, hd) * gain[:, None, None]).reshape(-1, hd)
        return Q, K, V, False
    return g1.scenario(name, rng, rows, nblk, hd)


def run_path(S_raw, q_scale_row, k_scale_blk, sm_scale, mask_oob, gran,
             V_e4m3, v_scale, thr=None):
    """G1 softmax semantics (the shipping ws kernel path), optionally with
    the A' threshold. thr=None -> base; thr=T -> lazy. Value model identical
    to g1_softmax_sim.run_paths 'new' except the row_max update site."""
    rows, nblk, _ = S_raw.shape
    hd = V_e4m3.shape[2]
    c = F32(sm_scale * g1.LOG2E)
    off = g1.S_FP8_OFFSET if thr is None else F32(g1.S_FP8_OFFSET - F32(thr))
    T = None if thr is None else F32(thr)

    row_max = np.full(rows, g1.NEG5E6, np.float32)
    denom = np.ones(rows, np.float32)
    O = np.zeros((rows, hd), np.float32)
    P_log, p_log, upd_log, one_log = [], [], [], []
    for blk in range(nblk):
        raw = S_raw[:, blk, :].astype(np.float32)  # exact, |S| < 2^24
        mask = mask_oob if blk == nblk - 1 else None
        d_cls = (np.repeat(q_scale_row[:, None], 4, 1)
                 * k_scale_blk[blk][None, :]).astype(np.float32)

        rawm = raw.copy()
        if mask is not None:
            rawm = np.where(mask, -np.inf, rawm)
        if gran == "pt":
            m_deq = None
            for cl in range(4):
                m_raw = rawm[:, g1.CLS_COLS[cl]].max(axis=1)  # exact tree
                with np.errstate(invalid="ignore"):
                    cand = (d_cls[:, cl] * m_raw).astype(np.float32)
                m_deq = cand if m_deq is None else np.fmax(m_deq, cand)
        else:
            m_raw = rawm.max(axis=1)
            with np.errstate(invalid="ignore"):
                m_deq = (d_cls[:, 0] * m_raw).astype(np.float32)
        m_loc = np.fmax(m_deq, g1.NEG5E6).astype(np.float32)

        # ---- row_max update: the one site A' changes ----
        m_prev = row_max
        chal = g1.fma(m_loc, c, -off)
        if T is None:
            row_max = np.maximum(row_max, chal)
            upd = chal > m_prev  # reporting only
        else:
            upd = (chal - row_max).astype(np.float32) > T
            row_max = np.where(upd, chal, row_max).astype(np.float32)
        o_scale = g1.exp2_f32(m_prev - row_max)
        upd_log.append(upd)
        one_log.append(o_scale == F32(1.0))
        denom = (denom * o_scale).astype(np.float32)

        c_raw = (c * d_cls).astype(np.float32)
        if mask is not None:
            c_raw = np.maximum(c_raw, g1.FLT_MIN)
        c_col = c_raw[:, [g1.class_of_col(j) for j in range(CTA_K)]] if gran == "pt" \
            else np.repeat(c_raw[:, :1], CTA_K, 1)
        with np.errstate(invalid="ignore"):
            a = g1.fma(rawm, c_col, -row_max[:, None])
        p = g1.exp2_f32(a)
        acc = np.zeros((rows, 8), np.float32)
        for w in range(CTA_K // 4):  # 4 packed chains, quad w -> pair (w & 1)
            k0 = (w & 1) * 4
            for b in range(4):
                acc[:, k0 + b] = (acc[:, k0 + b] + p[:, 4 * w + b]).astype(np.float32)
        for k in range(4):
            acc[:, k] = (acc[:, k] + acc[:, k + 4]).astype(np.float32)
        acc[:, 0] = (acc[:, 0] + acc[:, 2]).astype(np.float32)
        acc[:, 1] = (acc[:, 1] + acc[:, 3]).astype(np.float32)
        d_sum = (acc[:, 0] + acc[:, 1]).astype(np.float32)
        denom = (denom + d_sum).astype(np.float32)

        P = g1.e4m3_rne_sat(p)
        P_log.append(P)
        p_log.append(p)
        assert np.isfinite(P).all(), f"non-finite P at blk {blk}"

        O = (O * o_scale[:, None]).astype(np.float32)
        O = (O + (P.astype(np.float64) @ V_e4m3[blk].astype(np.float64)
                  ).astype(np.float32))

    d_rcp = (F32(1.0) / denom).astype(np.float32)
    Of = ((O * d_rcp[:, None]).astype(np.float32) * v_scale[None, :]).astype(np.float32)
    lse = (np.log2(denom).astype(np.float32) + row_max).astype(np.float32)
    return dict(row_max=row_max, denom=denom, O=Of, lse=lse,
                P=np.stack(P_log, 1), p_f32=np.stack(p_log, 1),
                upd=np.stack(upd_log, 1), o_one=np.stack(one_log, 1))


def build_case(name, gran, seed=0):
    """Quantized inputs + fp64 reference (g1_softmax_sim.run_scenario's build,
    with the exact-f64 matmul shortcut: |dot| < 2^53 so BLAS f64 is exact)."""
    rng = np.random.default_rng(seed)
    Qf, Kf, Vf, causal = scenario_ext(name, rng)
    rows, hd = Qf.shape
    kv_len = Kf.shape[0]
    nblk = -(-kv_len // CTA_K)
    kv_pad = nblk * CTA_K
    sm_scale = 1.0 / np.sqrt(hd)

    Qi, q_scale = g1.quant_int8(Qf, axis=1)
    q_scale = q_scale[:, 0].astype(np.float32)
    Kpad = np.zeros((kv_pad, hd), np.float32)
    Kpad[:kv_len] = Kf
    Ki = np.zeros((kv_pad, hd), np.int32)
    k_scale_blk = np.zeros((nblk, 4), np.float32)
    for blk in range(nblk):
        kb = Kpad[blk * CTA_K:(blk + 1) * CTA_K]
        if gran == "pt":
            for cl in range(4):
                qk, sc = g1.quant_int8(kb[g1.CLS_COLS[cl]], axis=(0, 1))
                Ki[blk * CTA_K + g1.CLS_COLS[cl]] = qk
                k_scale_blk[blk, cl] = sc.reshape(())
        else:
            qk, sc = g1.quant_int8(kb, axis=(0, 1))
            Ki[blk * CTA_K:(blk + 1) * CTA_K] = qk
            k_scale_blk[blk, :] = sc.reshape(())

    S_raw = (Qi.astype(np.float64) @ Ki.astype(np.float64).T).astype(np.int32)
    S_raw = S_raw.reshape(rows, nblk, CTA_K)

    v_scale = (np.max(np.abs(Vf), axis=0) / 448.0).astype(np.float32)
    Vq = g1.e4m3_rne_sat(Vf / np.where(v_scale > 0, v_scale, 1))
    Vpad = np.zeros((kv_pad, hd), np.float32)
    Vpad[:kv_len] = Vq
    V_e4m3 = Vpad.reshape(nblk, CTA_K, hd)

    jcol = np.arange((nblk - 1) * CTA_K, nblk * CTA_K)
    q_idx = (nblk - 1) * CTA_K + np.arange(rows) if causal else np.arange(rows)
    mask_oob = np.broadcast_to(jcol[None, :] >= kv_len, (rows, CTA_K)).copy()
    if causal:
        mask_oob = mask_oob | (jcol[None, :] > q_idx[:, None])

    full_mask = None
    if causal:
        full_mask = np.arange(kv_len)[None, :] > q_idx[:, None]
    ref = g1.sdpa_ref(Qf, Kf, Vf, sm_scale, full_mask)

    live = ~np.concatenate([np.zeros((rows, (nblk - 1) * CTA_K), bool), mask_oob], 1)
    args = (S_raw, q_scale, k_scale_blk, sm_scale, mask_oob, gran, V_e4m3, v_scale)
    return args, ref, live


def ballot_rate(o_one):
    """all-32-rows-skip rate per (correction warp, block >= 1)."""
    rows, nblk = o_one.shape
    if nblk < 2:  # single-block trip: correction never rescales
        return float("nan")
    warps = o_one[:, 1:].reshape(rows // 32, 32, nblk - 1)
    return float(warps.all(axis=1).mean())


def assert_bitwise_equal(a, b, keys=("row_max", "denom", "O", "lse", "P")):
    for k in keys:
        av, bv = a[k], b[k]
        assert (av.view(np.uint32) == bv.view(np.uint32)).all(), f"T=0 != base on {k}"


def eval_case(name, gran, patho):
    args, ref, live = build_case(name, gran)
    base = run_path(*args, thr=None)
    assert_bitwise_equal(base, run_path(*args, thr=0.0))  # self-check

    P_base = base["P"].reshape(live.shape[0], -1)[live]
    out = dict(name=name, gran=gran, patho=patho,
               ballot_base=ballot_rate(base["o_one"]),
               cos_base=g1.cos_sim(base["O"], ref),
               l1_base=g1.rel_l1(base["O"], ref),
               per_T={})
    for T in THRESHOLDS:
        lz = run_path(*args, thr=T)
        p_max = float(lz["p_f32"].reshape(live.shape[0], -1)[live].max())
        assert p_max < 448.0, f"P ceiling broken: {p_max} ({name}/{gran}/T={T})"
        P_lz = lz["P"].reshape(live.shape[0], -1)[live]
        nblk = lz["upd"].shape[1]
        out["per_T"][T] = dict(
            upd=float(lz["upd"][:, 1:].mean()) if nblk > 1 else float("nan"),
            ballot=ballot_rate(lz["o_one"]),
            p_max=p_max,
            rw=float((P_lz.view(np.uint32) != P_base.view(np.uint32)).mean()),
            z_base=float((P_base == 0).mean()),
            z_lz=float((P_lz == 0).mean()),
            cos_ref=g1.cos_sim(lz["O"], ref),
            l1_ref=g1.rel_l1(lz["O"], ref),
            cos_b=g1.cos_sim(lz["O"], base["O"]),
            l1_b=g1.rel_l1(lz["O"], base["O"]),
            lse_max=float(np.max(np.abs(lz["lse"] - base["lse"]))),
        )
    return out


def main():
    results = []
    for name, grans, patho in SCENARIOS:
        for gran in grans:
            results.append(eval_case(name, gran, patho))

    print("base path (ships today) vs fp64 SDPA ref:")
    print(f"{'scenario':<14} {'gran':<4} {'cos b/ref':<10} {'l1 b/ref':<9} {'ballot':<7}")
    for r in results:
        print(f"{r['name']:<14} {r['gran']:<4} {r['cos_base']:<10.6f} "
              f"{r['l1_base']:<9.4f} {r['ballot_base']:<7.3f}")

    for T in THRESHOLDS:
        print(f"\nT = {T} (offset {float(F32(g1.S_FP8_OFFSET - F32(T))):.3f}, "
              f"fresh-max P at {P_CEIL / 2**T:.1f}):")
        print(f"{'scenario':<14} {'gran':<4} {'upd%':<6} {'ballot':<7} {'P-rw':<7} "
              f"{'P-zero b>l':<12} {'cos l/ref':<10} {'l1 l/ref':<9} "
              f"{'cos l/base':<11} {'l1 l/base':<10} {'lse-max':<8}")
        for r in results:
            m = r["per_T"][T]
            print(f"{r['name']:<14} {r['gran']:<4} {100*m['upd']:<6.2f} "
                  f"{m['ballot']:<7.3f} {m['rw']:<7.3f} "
                  f"{m['z_base']:.3f}>{m['z_lz']:.3f}  "
                  f"{m['cos_ref']:<10.6f} {m['l1_ref']:<9.4f} "
                  f"{m['cos_b']:<11.7f} {m['l1_b']:<10.2e} {m['lse_max']:<8.2e}")

    print("\nverdict per T (gate: cos > 0.99, rel_l1 < 0.06 vs ref; "
          "neg_5e6 excluded - base fails it too):")
    for T in THRESHOLDS:
        gate = [r for r in results if r["name"] != "neg_5e6"]
        wcos = min(r["per_T"][T]["cos_ref"] for r in gate)
        wl1 = max(r["per_T"][T]["l1_ref"] for r in gate)
        # worst added error vs what base already loses to the reference
        wdl1 = max(r["per_T"][T]["l1_ref"] - r["l1_base"] for r in gate)
        rn = [r for r in results if r["name"].startswith("randn")]
        bal = min(r["per_T"][T]["ballot"] for r in rn)
        bal0 = min(r["ballot_base"] for r in rn)
        ok = "PASS" if (wcos > 0.99 and wl1 < 0.06) else "FAIL"
        print(f"  T={T}: worst cos {wcos:.6f}, worst rel_l1 {wl1:.4f} "
              f"(added vs base {wdl1:+.4f}); randn ballot {bal0:.3f}->{bal:.3f}"
              f"  [{ok}]")


if __name__ == "__main__":
    main()
