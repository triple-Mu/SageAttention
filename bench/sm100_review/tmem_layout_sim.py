# Copyright (c) 2025 by SageAttention team.
# Licensed under the Apache License, Version 2.0.
#
# TMEM layout simulator for qk_int_sv_f8_cuda_sm100.cu (design plan section 5.7).
#
# Runs on CPU, no GPU / torch needed:  python3 tmem_layout_sim.py
#
# Simulates, for the MVP kernel's CTA_Q=CTA_K=128 / 128-thread shape:
#   1. S consumption: tcgen05.mma D layout (M=128 cta_group::1 "Layout D":
#      lane = row, one 32b col per s32) vs the kernel's chunked
#      tcgen05.ld.32x32b.x32 index math -> asserts thread t sees exactly
#      S[t][j] in s_f32[j], i.e. softmax column j <-> kv position base+j
#      (the fused-mask mapping).
#   2. P production: the kernel's linear e4m3 pack (word w = elems 4w..4w+3,
#      byte b = elem 4w+b) + tcgen05.st.32x32b.x32 at column P_COL vs the
#      PTX-ISA-expected TS A-operand layout for kind::f8f6f4 (8-bit elements
#      packed contiguously in a 32b TMEM word, lane = M row - PTX ISA
#      9.7.17.10.4.3 + 9.7.17.10.5.4), including the per-K-step A address
#      offset (+8 columns per K=32 step).
#   3. O epilogue/correction: D layout of the PV accumulator vs the kernel's
#      chunked ld/st address math and the global-store channel indexing
#      (half2 pairs), for HEAD_DIM in {64, 128}.
#   4. PV_FROM_SMEM twin: the kernel's swizzled smem write of P vs the
#      canonical CU_TENSOR_MAP_SWIZZLE_128B layout that sK/sV are stored in
#      (whose K-major descriptor parameters are bit-for-bit parity-tested in
#      bench/sm100_review/test_desc_parity.cpp) -> the SS twin consumes P through the same
#      proven layout/descriptor pair.

CTA_Q = 128
CTA_K = 128
NUM_THREADS = 128

TMEM_COL_S = 0
TMEM_COL_P = 32               # P aliases S columns [32, 64)
TMEM_COL_O = CTA_K            # 128


def tmem_addr(lane, col):
    """PTX ISA 9.7.17.1.1: taddr = lane[31:16] | col[15:0]."""
    return (lane << 16) | col


def ld_st_32x32b_x32_cells(warp_idx, taddr):
    """Cells (lane, col, reg_owner_thread, reg_index) touched by one
    warp-collective tcgen05.{ld,st}.32x32b.x32 at taddr.
    Shape 32x32b: thread i of the warp owns lane base_lane+i; .x32 = 32
    consecutive columns per thread (PTX ISA 9.7.17.2.3.1.1 + 9.7.17.8.1:
    warp w may only access lanes [32w, 32w+32))."""
    base_lane = taddr >> 16
    base_col = taddr & 0xFFFF
    assert base_lane == 32 * warp_idx, "warp must address its own lane quadrant"
    cells = []
    for i in range(32):          # thread within warp
        for r in range(32):      # register index
            cells.append((base_lane + i, base_col + r, 32 * warp_idx + i, r))
    return cells


def mma_d_layout_cell(m, n, d_col_base):
    """tcgen05.mma cta_group::1 M=128 accumulator D cell for element (m, n):
    Layout D (PTX ISA 9.7.17.10.5.4): lane = m; D packing (9.7.17.10.4.1):
    s32/f32 = one element per 32b column -> col = base + n."""
    return (m, d_col_base + n)


def ts_a_expected_cell(m, k, a_col_base):
    """kind::f8f6f4 TS A-operand cell for 8-bit element A[m][k]:
    lane = m (Layout D, M=128 cta_group::1); 8-bit elements contiguously
    packed 4-per-32b-word along K (PTX ISA 9.7.17.10.4.2/.3 packing rule for
    8-bit containers) -> word col = base + k//4, byte-in-word = k%4."""
    return (m, a_col_base + k // 4, k % 4)


def test_s_consumption():
    # reference S produced by the QK MMA: S[m][n], n = kv position in tile
    S = [[(m * 131 + n * 7) & 0x7FFFFFFF for n in range(CTA_K)] for m in range(CTA_Q)]
    tmem = {}
    for m in range(CTA_Q):
        for n in range(CTA_K):
            tmem[mma_d_layout_cell(m, n, TMEM_COL_S)] = S[m][n]

    for t in range(NUM_THREADS):        # kernel: row = threadIdx.x
        warp_idx = t // 32
        row_base_addr = tmem_addr(32 * warp_idx, 0)
        s = [None] * CTA_K
        for c in range(CTA_K // 32):    # kernel: 4 chunks at col offsets 0/32/64/96
            taddr = row_base_addr + TMEM_COL_S + c * 32
            for (lane, col, owner, reg) in ld_st_32x32b_x32_cells(warp_idx, taddr):
                if owner == t:
                    s[c * 32 + reg] = tmem[(lane, col)]
        assert s == S[t], f"thread {t}: S row mismatch"
        # mask mapping: s_f32[j] is S[t][j] = score(q_row=t, kv=tile_base+j)
    print("PASS: S ld  - thread t reads S[t][0..127] in linear column order "
          "(mask column j <-> kv position base+j)")


def test_p_production():
    # p values the softmax produced: P[m][k], k = kv position in tile (K dim of PV)
    P = [[(m * 251 + k * 13) % 256 for k in range(CTA_K)] for m in range(CTA_Q)]

    # kernel pack: word w <- floatx4_to_e4m3x4(&p[4w], &p[4w+2])
    #   => byte 0 = p[4w+0], byte 1 = p[4w+1], byte 2 = p[4w+2], byte 3 = p[4w+3]
    def pack_word(row, w):
        return (P[row][4 * w + 0]
                | (P[row][4 * w + 1] << 8)
                | (P[row][4 * w + 2] << 16)
                | (P[row][4 * w + 3] << 24))

    # kernel store: tcgen05.st.32x32b.x32 at tmem_row_base + TMEM_COL_P
    tmem = {}
    for t in range(NUM_THREADS):
        warp_idx = t // 32
        taddr = tmem_addr(32 * warp_idx, TMEM_COL_P)
        for (lane, col, owner, reg) in ld_st_32x32b_x32_cells(warp_idx, taddr):
            if owner == t:
                tmem[(lane, col)] = pack_word(t, reg)

    # expected TS A layout, checked byte-for-byte for every (m, k)
    for m in range(CTA_Q):
        for k in range(CTA_K):
            lane, col, byte = ts_a_expected_cell(m, k, TMEM_COL_P)
            got = (tmem[(lane, col)] >> (8 * byte)) & 0xFF
            assert got == P[m][k], f"P[{m}][{k}]: TMEM byte mismatch"

    # per-K-step A address: kernel passes tmem_P_mma + v_it*8; step v consumes
    # K elems [32v, 32v+32) -> their word cols must be [P_COL+8v, P_COL+8v+8)
    for v in range(CTA_K // 32):
        cols = {ts_a_expected_cell(0, k, TMEM_COL_P)[1] for k in range(32 * v, 32 * v + 32)}
        assert cols == set(range(TMEM_COL_P + 8 * v, TMEM_COL_P + 8 * v + 8)), \
            f"K-step {v}: A column window mismatch"
    print("PASS: P st  - linear pack + 32x32b.x32 store == kind::f8f6f4 TS "
          "A-operand layout (incl. +8-col K-step offsets)")


def test_o_epilogue(head_dim):
    O = [[m * 1000.0 + n for n in range(head_dim)] for m in range(CTA_Q)]
    tmem = {}
    for m in range(CTA_Q):
        for n in range(head_dim):
            tmem[mma_d_layout_cell(m, n, TMEM_COL_O)] = O[m][n]

    for t in range(NUM_THREADS):
        warp_idx = t // 32
        row_base_addr = tmem_addr(32 * warp_idx, 0)
        out_row = [None] * head_dim  # global O[q_row][channel] written by thread t
        for c in range(head_dim // 32):
            taddr = row_base_addr + TMEM_COL_O + c * 32
            o_chunk = [None] * 32
            for (lane, col, owner, reg) in ld_st_32x32b_x32_cells(warp_idx, taddr):
                if owner == t:
                    o_chunk[reg] = tmem[(lane, col)]
            # correction path: st back to the SAME taddr -> exact aliasing by
            # construction (identical cell list); epilogue global store:
            for jj in range(16):  # half2 store at O_row_ptr + c*32 + 2*jj
                out_row[c * 32 + 2 * jj] = o_chunk[2 * jj]
                out_row[c * 32 + 2 * jj + 1] = o_chunk[2 * jj + 1]
        assert out_row == O[t], f"thread {t}: O epilogue channel mismatch (hd={head_dim})"
    print(f"PASS: O ld  - epilogue channel indexing == output layout (head_dim={head_dim})")


def test_pv_from_smem_swizzle():
    # CU_TENSOR_MAP_SWIZZLE_128B canonical layout for 128B rows (what TMA
    # produces for sK/sV and what the parity-tested K-major SW128 descriptor
    # describes): byte x of row r lives at r*128 + 16*((x//16) ^ (r%8)) + x%16
    def tma_sw128_offset(r, x):
        return r * 128 + 16 * (((x // 16) ^ (r % 8))) + (x % 16)

    # kernel write: word w of row r -> word index ((w>>2)^(r&7))<<2 | (w&3)
    def kernel_word_offset(r, w):
        return r * 128 + 4 * ((((w >> 2) ^ (r & 7)) << 2) | (w & 3))

    for r in range(CTA_Q):
        for w in range(CTA_K // 4):
            for b in range(4):
                x = 4 * w + b  # linear byte index within the row (K-major)
                assert kernel_word_offset(r, w) + b == tma_sw128_offset(r, x), \
                    f"sP swizzle mismatch at row {r}, word {w}, byte {b}"
    print("PASS: sP st - PV_FROM_SMEM staging writes the canonical 128B-swizzle "
          "layout (same layout/descriptor pair as TMA-loaded sK/sV)")


if __name__ == "__main__":
    test_s_consumption()
    test_p_production()
    test_o_epilogue(64)
    test_o_epilogue(128)
    test_pv_from_smem_swizzle()
    print("== tmem_layout_sim: ALL PASS ==")
