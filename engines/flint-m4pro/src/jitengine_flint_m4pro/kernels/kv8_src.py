"""kv8 profile of the decode megakernel: the 14 full layers' K and V at 8 bits in groups of 64.

The kv8 source is the main source with anchored textual edits (every anchor must match exactly the expected number of
times, or rendering fails), so render(kv8=False) stays the main profile's source byte for byte and the kv8 kernel
differs from it only in the pieces below. Sliding layers keep their bf16 rings and strided attention unchanged.

History (kv_cache.MKHistory8; bound in the megakernel's kf0 / kf1 / vf0 / vf1 slots, so the argument table is
unchanged):
  Kq, Vq    [14][4][cap][128] uint8: row p = the bytes mx.quantize(row, group_size=64, bits=8) packs for that 128-wide
            row (code of head dim d at byte d)
  Ksb, Vsb  [14][4][cap][2] uint32: bf16 scale | bf16 bias << 16 of 64-dim group g at word 2 p + g

Quantise on write (QKV epilogue of every full-layer K / V head, decode passes and verify rows): one whole simdgroup per
64-dim group runs MLX affine_quantize's fp32 arithmetic (0 code / scale / bias mismatches against mx.quantize on
16,384 varied rows): lane i owns values 64 g + 2 i and + 1; w_min seeded with the bf16 max and w_max with 0; simd
min / max; scale = max((w_max - w_min) / 255, 1e-7) signed toward the larger-magnitude edge; the q0 adjustment; codes
from the fp32 scale and bias; scale and bias stored as bf16 (RNE). The values quantised are exactly the bf16 K / V the
main profile stores (the threadgroup row hb after norm; full layers carry no RoPE).

Attention (att6q8 = kv8_fold.att6q8_src): layout 6d's item over the 8-bit history with the dequantisation folded out
of the stream, rather than dequantising every element as bf16(fma(s, q, b)) (on the M4 Pro, kv8 raw decode at 258K
is faster this way). K and V products run on the raw codes (unpacked to half as 1024 + c), and
each key's per-group scale / bias is applied once, at the score (s_g S_g + (b_g - 1024 s_g) Qsum_g) and at P (codes x
s_g p, plus a running sum of p (b_g - 1024 s_g) per head and group). Head-dim map: tile t = 8 g + tt holds dims
64 g + 8 i + tt, so each tile lies in one 64-dim group and a lane reads contiguous code bytes. Same chunk grid, causal
limit, partial rows (m, l, -, -, o[128] in natural dim order) and merge as att6, so a verify row is the M = 1 item at its
position (row-exact by construction). Not bit-identical to the per-element dequantising item (within 1 bf16 ulp of it
and of an fp64 reference at the attention output): kv8 is a separate arithmetic, held to its own R'-relative
tolerance.
"""

from __future__ import annotations

from jitengine_flint_m4pro.kernels import kv8_fold
from jitengine_flint_m4pro.kernels.metal_src import read_metal

# ---------------------------------------------------------------------------------------------- inserted fragments
KV8_HIST = read_metal("kv8_history.metal")

KV8_ATT = (
    "// Layout 6d-kv8, dequantisation folded (see kernels/kv8_fold.py): att6's item\n"
    "// over the 8-bit history with K and V products on the raw codes (half, 1024 + c) and the per-group scale / bias\n"
    "// applied per key at the score and at P; partial rows exactly as att6 (natural dim order), so the merge is unchanged.\n"
    + kv8_fold.att6q8_src("att6q8")
    + r"""// one layout-6d-kv8 item of full layer l, kv head kvh: q heads kvh * 8 .. + 7 of the row at q8, keys [kb, ke)
inline void att6q8_item(thread const Ctx& c, uint l, uint kvh, const device ushort* q8, uint kb, uint ke,
                        device float* rows) {
  att6q8(q8, kv8_codes(c, l, false, kvh), kv8_sb(c, l, false, kvh), kv8_codes(c, l, true, kvh), kv8_sb(c, l, true, kvh),
         kb, ke, rows, c.lane);
}
"""
)

# ---------------------------------------------------------------------------------------------- anchored edits
# (anchor, replacement, expected count); an insertion keeps its anchor after the inserted text
_M1 = (
    (
        "#define DIAG __DIAG__\n",
        "#define DIAG __DIAG__\n#define KV8 1                   // kv8 profile (kernels/kv8_src.py)\n",
        1,
    ),
    (
        "// ------------------------------------------------------------------ MoE block",
        KV8_HIST.lstrip("\n")
        + "// ------------------------------------------------------------------ MoE block",
        1,
    ),
    (
        "      else kv_head(c, l, kind == 2u, local)[size_t(slot) * 128u + c.tid] = f2bf(out);\n",
        "      else if (!full) kv_head(c, l, kind == 2u, local)[size_t(slot) * 128u + c.tid] = f2bf(out);\n"
        "      else if (c.tid < 64u) kv8_store(c, l, kind == 2u, local, slot, c.hb, c.tid >> 5);   // kv8: simdgroup = group\n",
        1,
    ),
    (
        "// one full-layer item (kv head, chunk, sub-stream) of the long stream",
        KV8_ATT + "// one full-layer item (kv head, chunk, sub-stream) of the long stream",
        1,
    ),
    (
        "      att6(QB(c, l) + kvh * 8u * 128u, kv_head(c, l, false, kvh), kv_head(c, l, true, kvh), kb, ke,\n"
        "           PART(c) + size_t(it) * 8u * PROW, c.lane);\n",
        "      att6q8_item(c, l, kvh, QB(c, l) + kvh * 8u * 128u, kb, ke, PART(c) + size_t(it) * 8u * PROW);\n",
        1,
    ),
)
_V = (
    (
        "          else kv_head(c, l, kind == 2u, local)[size_t(slot) * 128u + t] = f2bf(out);\n",
        "          else if (!full) kv_head(c, l, kind == 2u, local)[size_t(slot) * 128u + t] = f2bf(out);\n"
        "          else if (t < 64u) kv8_store(c, l, kind == 2u, local, slot, hb, t >> 5);   // kv8: simdgroup = group\n",
        1,
    ),
    (
        "      att6(QBv(c, m, l) + kvh * 8u * 128u, kv_head(c, l, false, kvh), kv_head(c, l, true, kvh), kb, ke,\n"
        "           PARTv(c, m) + size_t(kvh * Cm + ch) * 8u * PROW, c.lane);\n",
        "      att6q8_item(c, l, kvh, QBv(c, m, l) + kvh * 8u * 128u, kb, ke, PARTv(c, m) + size_t(kvh * Cm + ch) * 8u * PROW);\n",
        1,
    ),
)


def _edit(src: str, edits) -> str:
    for anchor, new, count in edits:
        got = src.count(anchor)
        if got != count:
            raise AssertionError(
                f"kv8 source edit: anchor {anchor[:60]!r} found {got} times, expected {count}"
            )
        src = src.replace(anchor, new)
    return src


def apply_m1(header: str) -> str:
    """HEADER + HEADER2 of the main source -> the kv8 profile's (M = 1 pass and the shared primitives)."""
    return _edit(header, _M1)


def apply_v(header_v: str) -> str:
    """HEADER_V of the main source -> the kv8 profile's verify pass (quantise-on-write per row, att6q8 per row item)."""
    return _edit(header_v, _V)
