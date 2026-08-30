#!/usr/bin/env python3
"""Bit-level simulation of the G1 softmax value-domain rewrite
(csrc/qattn/qk_int_sv_f8_cuda_sm100_ws.cu, C1_DESIGN.md section 9).

Simulates int32 S -> softmax -> e4m3 P -> PV -> epilogue for both paths:

  old: dequant row s = rnd(raw*d); mask -5e6; serial fold max (init -5e6);
       a = fma(s, c, -row_max); serial d_sum
  new: raw row; mask -inf; per-class fmax tree * d (+ -5e6 floor);
       a = fma(raw, rnd(c*d), -row_max); 4 packed d_sum chains

and checks/reports, over random + adversarial inputs:
  * row_max / o_scale bit-identity between the paths (hard assert - the
    kernel comments claim it, the sim proves it including the NaN corners);
  * P (e4m3) elementwise rewrite rate and denom relative movement;
  * end-to-end O and LSE: cos_sim / rel_l1 (test/conftest.py definitions)
    for new-vs-fp64-SDPA, old-vs-fp64-SDPA and new-vs-old.

Model fidelity notes (identical treatment for both paths, so the A/B is
exact even where the absolute model is approximate):
  * f32 fma is emulated as f64 multiply-add then one cast to f32 (the f64
    product of two f32 is exact; the double rounding through f64 differs
    from a fused f32 fma only on ~2^-29 ties);
  * exp2/log2/rcp use numpy's correctly-rounded f32 libm, standing in for
    MUFU.EX2 / LG2 / RCP (<= 2 ulp device error);
  * e4m3 conversion is round-to-nearest-even with satfinite (cvt.rn.satfinite
    semantics), implemented exactly;
  * the PV MMA and the correction rescale accumulate in f32; the MMA's
    internal accumulation order is modeled as exact-f64-then-round.

numpy only; runs on any host. Results table: C1_DESIGN.md section 9.
"""

import numpy as np

F32 = np.float32
S_FP8_OFFSET = F32(8.807)
LOG2E = F32(1.4426950408889634)
FLT_MIN = F32(np.finfo(np.float32).tiny)
NEG5E6 = F32(-5000000.0)
CTA_K = 128


def f32(x):
    return np.asarray(x, dtype=np.float32)


def fma(a, b, c):
    """f32 fma via exact-f64 then one cast (see module docstring)."""
    return (
        np.asarray(a, np.float64) * np.asarray(b, np.float64) + np.asarray(c, np.float64)
    ).astype(np.float32)


def exp2_f32(x):
    return np.exp2(np.asarray(x, np.float32)).astype(np.float32)


def e4m3_rne_sat(x):
    """cvt.rn.satfinite.e4m3x2.f32 semantics: RNE to e4m3, saturate to +-448."""
    x = np.asarray(x, np.float64)
    a = np.abs(x)
    s = np.sign(x)
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(np.where(a > 0, a, 1.0)))
    e = np.clip(e, -6, 8)
    q = np.where(a < 2.0**-6, 2.0**-9, 2.0 ** (e - 3))  # subnormal step below 2^-6
    v = np.round(a / q) * q  # np.round is round-half-even
    v = np.minimum(v, 448.0)
    return (s * v).astype(np.float32)


def quant_int8(x, axis, group=None):
    """amax/127 int8 quant along axis (whole-row group when group is None)."""
    amax = np.max(np.abs(x), axis=axis, keepdims=True)
    scale = (amax / 127.0).astype(np.float32)
    q = np.where(scale > 0, np.round(x / np.where(scale > 0, scale, 1)), 0)
    return np.clip(q, -128, 127).astype(np.int32), scale


def class_of_col(j):
    return (j % 8) // 2


CLS_COLS = [np.array([j for j in range(CTA_K) if class_of_col(j) == c]) for c in range(4)]


def run_paths(S_raw, q_scale_row, k_scale_blk, sm_scale, mask_oob, gran, V_e4m3, v_scale):
    """S_raw: (rows, nblk, 128) int32; q_scale_row: (rows,); k_scale_blk:
    (nblk, 4) f32 (per-warp: 4 equal columns); mask_oob: (rows, 128) bool for
    the LAST block only (peeled step); V_e4m3: (nblk, 128, hd) f32 (e4m3
    values); v_scale: (hd,). Returns per-path dict."""
    rows, nblk, _ = S_raw.shape
    hd = V_e4m3.shape[2]
    c = F32(sm_scale * LOG2E)

    out = {}
    for path in ("old", "new"):
        row_max = np.full(rows, NEG5E6, np.float32)
        denom = np.ones(rows, np.float32)
        O = np.zeros((rows, hd), np.float32)
        P_log, p_log = [], []
        for blk in range(nblk):
            raw = S_raw[:, blk, :].astype(np.float32)  # exact, |S| < 2^24
            is_last = blk == nblk - 1
            mask = mask_oob if is_last else None
            d_cls = np.repeat(q_scale_row[:, None], 4, 1) * k_scale_blk[blk][None, :]
            d_cls = d_cls.astype(np.float32)  # rnd(q*k), (rows, 4)
            d_col = d_cls[:, [class_of_col(j) for j in range(CTA_K)]] if gran == "pt" else \
                np.repeat(d_cls[:, :1], CTA_K, 1)

            if path == "old":
                s = (raw * d_col).astype(np.float32)  # rnd(raw*d) per element
                if mask is not None:
                    s = np.where(mask, NEG5E6, s)
                m_loc = np.maximum(s.max(axis=1), NEG5E6).astype(np.float32)
            else:
                rawm = raw.copy()
                if mask is not None:
                    rawm = np.where(mask, -np.inf, rawm)
                if gran == "pt":
                    m_deq = None
                    for cl in range(4):
                        m_raw = rawm[:, CLS_COLS[cl]].max(axis=1)  # exact tree
                        with np.errstate(invalid="ignore"):
                            cand = (d_cls[:, cl] * m_raw).astype(np.float32)
                        m_deq = cand if m_deq is None else np.fmax(m_deq, cand)
                else:
                    m_raw = rawm.max(axis=1)
                    with np.errstate(invalid="ignore"):
                        m_deq = (d_cls[:, 0] * m_raw).astype(np.float32)
                m_loc = np.fmax(m_deq, NEG5E6).astype(np.float32)  # floor absorbs NaN/-inf

            m_prev = row_max
            row_max = np.maximum(row_max, fma(m_loc, c, -S_FP8_OFFSET))
            o_scale = exp2_f32(m_prev - row_max)
            denom = (denom * o_scale).astype(np.float32)

            if path == "old":
                a = fma(s, c, -row_max[:, None])
                p = exp2_f32(a)
                d_sum = np.zeros(rows, np.float32)
                for j in range(CTA_K):  # serial fold, ascending j
                    d_sum = (d_sum + p[:, j]).astype(np.float32)
            else:
                c_raw = (c * d_cls).astype(np.float32)  # rnd(c*d), (rows, 4)
                if mask is not None:
                    c_raw = np.maximum(c_raw, FLT_MIN)
                c_col = c_raw[:, [class_of_col(j) for j in range(CTA_K)]] if gran == "pt" else \
                    np.repeat(c_raw[:, :1], CTA_K, 1)
                with np.errstate(invalid="ignore"):
                    a = fma(rawm, c_col, -row_max[:, None])
                p = exp2_f32(a)
                acc = np.zeros((rows, 8), np.float32)
                for w in range(CTA_K // 4):  # 4 packed chains, quad w -> pair (w & 1)
                    k0 = (w & 1) * 4
                    for b in range(4):
                        acc[:, k0 + b] = (acc[:, k0 + b] + p[:, 4 * w + b]).astype(np.float32)
                for k in range(4):  # acc[0..3] += acc[4..7]
                    acc[:, k] = (acc[:, k] + acc[:, k + 4]).astype(np.float32)
                acc[:, 0] = (acc[:, 0] + acc[:, 2]).astype(np.float32)
                acc[:, 1] = (acc[:, 1] + acc[:, 3]).astype(np.float32)
                d_sum = (acc[:, 0] + acc[:, 1]).astype(np.float32)
            denom = (denom + d_sum).astype(np.float32)

            P = e4m3_rne_sat(p)
            P_log.append(P)
            p_log.append(p)
            assert np.isfinite(P).all(), f"non-finite P in {path} blk {blk}"

            O = (O * o_scale[:, None]).astype(np.float32)
            O = (O + (P.astype(np.float64) @ V_e4m3[blk].astype(np.float64)).astype(np.float32))

        d_rcp = (F32(1.0) / denom).astype(np.float32)  # stands in for rcp.approx
        Of = ((O * d_rcp[:, None]).astype(np.float32) * v_scale[None, :]).astype(np.float32)
        lse = (np.log2(denom).astype(np.float32) + row_max).astype(np.float32)
        out[path] = dict(row_max=row_max, denom=denom, O=Of, lse=lse,
                         P=np.stack(P_log, 1), p_f32=np.stack(p_log, 1))
    return out


def sdpa_ref(Qf, Kf, Vf, sm_scale, mask_oob):
    S = (Qf.astype(np.float64) @ Kf.astype(np.float64).T) * sm_scale
    if mask_oob is not None:
        S = np.where(mask_oob, -np.inf, S)
    m = S.max(axis=1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(S - m)
    return e @ Vf.astype(np.float64) / e.sum(axis=1, keepdims=True)


def cos_sim(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def rel_l1(a, b):
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).sum()
                 / np.abs(b.astype(np.float64)).sum())


def scenario(name, rng, rows=128, nblk=32, hd=128):
    """Returns Qf, Kf, Vf float32 and a causal flag."""
    kv = nblk * CTA_K
    if name == "randn":
        return rng.standard_normal((rows, hd)), rng.standard_normal((kv, hd)), \
            rng.standard_normal((kv, hd)), False
    if name == "randn_causal":
        Q, K, V, _ = scenario("randn", rng, rows, nblk, hd)
        return Q, K, V, True
    if name == "big_range":  # per-row magnitudes spanning 4 decades
        Q = rng.standard_normal((rows, hd)) * 10 ** rng.uniform(-2, 2, (rows, 1))
        K = rng.standard_normal((kv, hd)) * 10 ** rng.uniform(-2, 2, (kv, 1))
        V = rng.standard_normal((kv, hd))
        return Q, K, V, False
    if name == "const_rows":  # constant S rows: max ties everywhere
        q = rng.standard_normal(hd)
        Q = np.tile(q, (rows, 1))
        K = np.tile(rng.standard_normal(hd), (kv, 1))
        V = rng.standard_normal((kv, hd))
        return Q, K, V, False
    if name == "outlier_block":  # one KV block 100x hotter: rescale stress
        Q, K, V, _ = scenario("randn", rng, rows, nblk, hd)
        K[5 * CTA_K:6 * CTA_K] *= 100.0
        return Q, K, V, False
    if name == "zero_amax":  # all-zero K blocks (k_scale = 0) + causal peel
        Q, K, V, _ = scenario("randn", rng, rows, nblk, hd)
        K[2 * CTA_K:4 * CTA_K] = 0.0
        K[(nblk - 1) * CTA_K:] = 0.0  # zero-amax in the masked peeled block
        return Q, K, V, True
    if name == "neg_5e6":  # every logit < -5e6: the old -5e6 floor bites
        q = np.abs(rng.standard_normal((rows, hd))) * 60.0
        K = -np.abs(rng.standard_normal((kv, hd))) * 60.0
        V = rng.standard_normal((kv, hd))
        return q, K, V, False
    if name == "sharp":  # 16x logit spread: larger fma cancellation noise
        return (rng.standard_normal((rows, hd)) * 4.0,
                rng.standard_normal((kv, hd)) * 4.0,
                rng.standard_normal((kv, hd)), False)
    if name == "short_kv":  # kv_len=17: peeled block almost fully masked
        return (rng.standard_normal((rows, hd)), rng.standard_normal((17, hd)),
                rng.standard_normal((17, hd)), False)
    raise ValueError(name)


def run_scenario(name, gran, seed=0):
    rng = np.random.default_rng(seed)
    Qf, Kf, Vf, causal = scenario(name, rng)
    rows, hd = Qf.shape
    kv_len = Kf.shape[0]
    nblk = -(-kv_len // CTA_K)
    kv_pad = nblk * CTA_K
    sm_scale = 1.0 / np.sqrt(hd)

    Qi, q_scale = quant_int8(Qf, axis=1)  # per-row q_scale (thread granularity)
    q_scale = q_scale[:, 0].astype(np.float32)
    Kpad = np.zeros((kv_pad, hd), np.float32)
    Kpad[:kv_len] = Kf
    Ki = np.zeros((kv_pad, hd), np.int32)
    k_scale_blk = np.zeros((nblk, 4), np.float32)
    for blk in range(nblk):
        kb = Kpad[blk * CTA_K:(blk + 1) * CTA_K]
        if gran == "pt":
            for cl in range(4):
                qk, sc = quant_int8(kb[CLS_COLS[cl]], axis=(0, 1))
                Ki[blk * CTA_K + CLS_COLS[cl]] = qk
                k_scale_blk[blk, cl] = sc.reshape(())
        else:
            qk, sc = quant_int8(kb, axis=(0, 1))
            Ki[blk * CTA_K:(blk + 1) * CTA_K] = qk
            k_scale_blk[blk, :] = sc.reshape(())

    S_raw = np.einsum("rd,kd->rk", Qi, Ki, dtype=np.int64).astype(np.int32)
    S_raw = S_raw.reshape(rows, nblk, CTA_K)

    v_scale = (np.max(np.abs(Vf), axis=0) / 448.0).astype(np.float32)
    Vq = e4m3_rne_sat(Vf / np.where(v_scale > 0, v_scale, 1))
    Vpad = np.zeros((kv_pad, hd), np.float32)
    Vpad[:kv_len] = Vq
    V_e4m3 = Vpad.reshape(nblk, CTA_K, hd)

    # OOB/causal mask of the peeled (last) block, kernel predicate. Causal
    # rows model the kernel's LAST 128-row tile of the sequence
    # (q_idx = (nblk-1)*128 + row): its trip count makes the peeled block the
    # diagonal one, and every earlier block is fully unmasked - the same
    # structure every tile sees.
    jcol = np.arange((nblk - 1) * CTA_K, nblk * CTA_K)
    q_idx = (nblk - 1) * CTA_K + np.arange(rows) if causal else np.arange(rows)
    mask_oob = np.broadcast_to(jcol[None, :] >= kv_len, (rows, CTA_K)).copy()
    if causal:
        mask_oob = mask_oob | (jcol[None, :] > q_idx[:, None])
    res = run_paths(S_raw, q_scale, k_scale_blk, sm_scale, mask_oob, gran, V_e4m3, v_scale)

    full_mask = None
    if causal:
        full_mask = np.arange(kv_len)[None, :] > q_idx[:, None]
    ref = sdpa_ref(Qf, Kf, Vf, sm_scale, full_mask)

    old, new = res["old"], res["new"]
    bitid = (old["row_max"].view(np.uint32) == new["row_max"].view(np.uint32)).all()
    live = ~np.concatenate([np.zeros((rows, (nblk - 1) * CTA_K), bool), mask_oob], 1)
    P_old = old["P"].reshape(rows, -1)[live]
    P_new = new["P"].reshape(rows, -1)[live]
    rewrite = float((P_old.view(np.uint32) != P_new.view(np.uint32)).mean())
    p_old = old["p_f32"].reshape(rows, -1)[live]
    p_new = new["p_f32"].reshape(rows, -1)[live]
    p_diff = float((p_old.view(np.uint32) != p_new.view(np.uint32)).mean())
    with np.errstate(invalid="ignore", divide="ignore"):
        p_rel = np.abs(p_new.astype(np.float64) - p_old) / np.maximum(p_old, 1e-30)
    p_relmax = float(np.max(np.where(p_old >= 2.0**-9, p_rel, 0.0)))  # e4m3-visible range
    dden = float(np.max(np.abs(new["denom"].astype(np.float64) - old["denom"])
                        / np.abs(old["denom"])))
    return dict(
        name=name, gran=gran, bitid=bool(bitid), rewrite=rewrite, p_diff=p_diff,
        p_relmax=p_relmax, n_live=int(P_old.size), dden=dden,
        cos_new_ref=cos_sim(new["O"], ref), cos_old_ref=cos_sim(old["O"], ref),
        cos_new_old=cos_sim(new["O"], old["O"]),
        l1_new_ref=rel_l1(new["O"], ref), l1_old_ref=rel_l1(old["O"], ref),
        l1_new_old=rel_l1(new["O"], old["O"]),
        lse_max=float(np.max(np.abs(new["lse"] - old["lse"]))),
    )


def main():
    names = ["randn", "randn_causal", "big_range", "sharp", "const_rows",
             "outlier_block", "zero_amax", "neg_5e6", "short_kv"]
    print(f"{'scenario':<14} {'gran':<4} {'rowmax==':<8} {'p-f32-diff':<11} "
          f"{'p-rel-max':<10} {'P-rewrite':<10} {'denom-rel':<10} {'cos n/ref':<10} "
          f"{'cos o/ref':<10} {'cos n/o':<11} {'l1 n/ref':<9} {'l1 o/ref':<9} "
          f"{'l1 n/o':<9} {'lse-max':<8}")
    worst = dict(cos=1.0, l1=0.0)
    n_live = 0
    n_flip = 0
    for name in names:
        for gran in ("pw", "pt"):
            r = run_scenario(name, gran)
            assert r["bitid"], f"row_max bit-identity broken: {name}/{gran}"
            n_live += r["n_live"]
            n_flip += int(round(r["rewrite"] * r["n_live"]))
            print(f"{r['name']:<14} {r['gran']:<4} {str(r['bitid']):<8} "
                  f"{r['p_diff']:<11.2e} {r['p_relmax']:<10.2e} {r['rewrite']:<10.2e} "
                  f"{r['dden']:<10.2e} "
                  f"{r['cos_new_ref']:<10.6f} {r['cos_old_ref']:<10.6f} "
                  f"{r['cos_new_old']:<11.8f} {r['l1_new_ref']:<9.4f} "
                  f"{r['l1_old_ref']:<9.4f} {r['l1_new_old']:<9.2e} {r['lse_max']:<8.2e}")
            worst["cos"] = min(worst["cos"], r["cos_new_ref"])
            worst["l1"] = max(worst["l1"], r["l1_new_ref"])
    print(f"\ntotal live P elements: {n_live}, e4m3 rewrites: {n_flip}")
    print(f"worst new-vs-ref: cos_sim {worst['cos']:.6f} (gate > 0.99), "
          f"rel_l1 {worst['l1']:.4f} (gate < 0.06)")


if __name__ == "__main__":
    main()
