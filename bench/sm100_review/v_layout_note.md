# sm100 V layout contract (updated)

The sm100 tcgen05 kernel consumes V from `per_channel_fp8(..., permute=False)`:
transposed to (head_dim, padded_kv_len) and zero-padded to a multiple of 128,
with **linear kv order** — NOT the within-16 seq permutation
`[0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15]` used by the sm89/sm90/sm12x paths.

Why: that permutation exists to match the k-order of the **register A-fragments**
of `mma.sync` / RS-form `wgmma` (the fp8 P fragment holds k interleaved, and
upstream chose to permute V rather than shuffle P registers). The tcgen05 kernel
instead packs P into TMEM in **linear k-order** (the TS A-operand layout, see
tmem_layout_sim.py), and its B operand (V^T) goes through a canonical K-major
smem descriptor. A permuted V against a linear P would misalign every dot
product's k pairing.

This retires the earlier risk-register item "per_channel_fp8 applies the sm90
seq-permutation to V; interaction with the tcgen05 descriptor is hardware-day
verifiable only": both MMA operands are now linear by construction.

Hardware-day note: the TS-vs-SS cross-check (PV_FROM_SMEM twin) is unaffected —
both paths consume the same linear sV. The correctness sweep must call the
sm100 wrapper (which passes permute=False), not hand-built per_channel_fp8
defaults.
