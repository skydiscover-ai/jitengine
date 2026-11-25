"""kv8 profile: layout 6d-kv8's attention item with the dequantisation folded out of the stream (form fold_kh_vh).

The item attends one (kv head, chunk) of a full layer over the 8-bit history (codes c in [0, 255], per 64-dim group g of
each key a bf16 scale s_g and bias b_g, so a dequantised value is s_g c + b_g). Instead of dequantising every element
(bf16(fma(c, s, b)) per element), both products run on the RAW codes and the affine map is applied
once per key and group:

  K side (scores):  q . k = sum_g ( s_g sum_{d in g} c_d q_d + b_g Qsum_g ),   Qsum_g = sum_{d in g} q_d (once per item)
  V side (output):  o = sum_k p_k v_k = sum_g [ sum_k (p_k s_kg) c_k + sum_k p_k b_kg ]   (the p b term kept per head
                    and group as a running sum, rescaled with l when the running max moves)

Codes are unpacked to half by the byte trick (w & 0x00FF00FF) | 0x64006400 = 1024 + c (two codes per AND / OR pair),
and the 1024 offset is folded into the bias term (b_g - 1024 s_g); Q enters the K-side MMAs as half (bf16 -> half is
exact for |q| < 65504 and above half's subnormal range). Accumulation stays fp32.

Head-dim map ("group map"): tile t = 8 g + tt holds head dims 64 g + 8 i + tt (i = tile row / column index), so every
tile lies inside ONE 64-dim group and a lane still reads contiguous code bytes: K, key row k0 + fm, bytes
[64 g + 8 fn, + 16); V, key rows k0 + fn and k0 + fn + 1, bytes [64 g + 8 fm, + 8). Each lane loads both scale / bias
words of its key (K: row k0 + fm; V: the P row's key, k0 + fm). The partial row layout is att6's (m, l, -, -, o[128]
in NATURAL head-dim order), so the merge is unchanged.

Exactness: not bit-identical to the per-element dequantising item (no bf16 rounding of dequantised K / V, a different
summation); within 1 bf16 ulp of it and of an fp64 reference over the dequantised history at the attention output
(measured on the M4 Pro). The item depends only on (q row, history, kb, ke), so a verify row equals the M = 1 item at its position
(row-exact by construction) and reruns are bitwise.
"""

from __future__ import annotations

_C4 = "xyzw"


def _unpack_half(prefix: str, words: list) -> tuple:
    """code bytes of uint words -> (Metal lines, per-byte half expressions 1024 + c in byte order)"""
    lines, el = [], []
    for wi, w in enumerate(words):
        lines.append(
            f"    const half2 {prefix}{wi}a = as_type<half2>(({w} & 0x00FF00FFu) | 0x64006400u);   // bytes 0, 2\n"
            f"    const half2 {prefix}{wi}b = as_type<half2>((({w} >> 8) & 0x00FF00FFu) | 0x64006400u);   // 1, 3\n"
        )
        el += [f"{prefix}{wi}a.x", f"{prefix}{wi}b.x", f"{prefix}{wi}a.y", f"{prefix}{wi}b.y"]
    return "".join(lines), el


def _q_tiles() -> str:
    """Q^T on the group map as half tiles, plus Qsum_g of heads fn / fn + 1 (fp32 over the bf16 values)"""
    src = (
        "  simdgroup_half8x8 QT[16];\n"
        "  float qs00, qs01, qs10, qs11;            // Qsum_g of heads fn (x0) / fn + 1 (x1)\n  {\n"
    )
    for g in range(2):
        src += (
            f"    const uint4 qa{g} = *(const device uint4*)(q8 + fn * 128 + {64 * g} + 8 * fm);\n"
            f"    const uint4 qb{g} = *(const device uint4*)(q8 + (fn + 1) * 128 + {64 * g} + 8 * fm);\n"
        )
    for g in range(2):
        src += f"    float sa{g} = 0.0f, sb{g} = 0.0f;\n"
        for tt in range(8):
            comp = _C4[tt >> 1]
            sh = "<< 16" if tt % 2 == 0 else "& 0xffff0000u"
            src += (
                f"    {{ const float x = as_type<float>(qa{g}.{comp} {sh}), y = as_type<float>(qb{g}.{comp} {sh}); "
                f"sa{g} += x; sb{g} += y;\n"
                f"      QT[{8 * g + tt}].thread_elements()[0] = half(x); QT[{8 * g + tt}].thread_elements()[1] = half(y); }}\n"
            )
    src += (
        "    float4 qs = float4(sa0, sb0, sa1, sb1);\n"
        "    qs += simd_shuffle_xor(qs, ushort(2));\n    qs += simd_shuffle_xor(qs, ushort(4));\n"
        "    qs += simd_shuffle_xor(qs, ushort(16));\n"
        "    qs00 = qs.x; qs01 = qs.y; qs10 = qs.z; qs11 = qs.w;\n  }\n"
    )
    return src


def _scores() -> str:
    """S_g = sum over group g's tiles of C_t Q^T_t on the raw codes, then the per-key affine fold"""
    src = (
        "    simdgroup_float8x8 Sa = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n"
        "    simdgroup_float8x8 Sb = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n"
        "    const uint4 kg0 = *(const device uint4*)(Kq + size_t(rk) * 128 + 8 * fn);\n"
        "    const uint4 kg1 = *(const device uint4*)(Kq + size_t(rk) * 128 + 64 + 8 * fn);\n"
        "    const uint2 wk = *(const device uint2*)(Ksb + size_t(rk) * 2);\n"
        "    const float ks0 = lo16(wk.x), kb0 = hi16(wk.x), ks1 = lo16(wk.y), kb1 = hi16(wk.y);\n"
    )
    for g in range(2):
        acc = "Sa" if g == 0 else "Sb"
        code, el = _unpack_half(f"kq{g}_", [f"kg{g}.{c}" for c in _C4])
        src += code + "    { simdgroup_half8x8 Kt;\n"
        for tt in range(8):  # bytes 0..7 -> column fn of tile tt, bytes 8..15 -> column fn + 1
            src += (
                f"      Kt.thread_elements()[0] = {el[tt]}; Kt.thread_elements()[1] = {el[8 + tt]};\n"
                f"      simdgroup_multiply_accumulate({acc}, Kt, QT[{8 * g + tt}], {acc});\n"
            )
        src += "    }\n"
    src += (
        "    const float kc0 = kb0 - 1024.0f * ks0, kc1 = kb1 - 1024.0f * ks1;\n"
        "    const bool valid = (k0 + uint(fm)) < ke;\n"
        "    const float s0 = valid ? (ks0 * Sa.thread_elements()[0] + ks1 * Sb.thread_elements()[0] + kc0 * qs00 + kc1 * qs10) * QSCALE : -INFINITY;\n"
        "    const float s1 = valid ? (ks0 * Sa.thread_elements()[1] + ks1 * Sb.thread_elements()[1] + kc0 * qs01 + kc1 * qs11) * QSCALE : -INFINITY;\n"
    )
    return src


_SOFTMAX = r"""    float2 bm = float2(s0, s1);
    bm = max(bm, simd_shuffle_xor(bm, ushort(2)));
    bm = max(bm, simd_shuffle_xor(bm, ushort(4)));
    bm = max(bm, simd_shuffle_xor(bm, ushort(16)));
    const float n0 = max(m0, bm.x), n1 = max(m1, bm.y);
    if (simd_any(n0 > m0 || n1 > m1)) {
      const float a0 = (m0 == -INFINITY) ? 0.0f : exp2(m0 - n0);
      const float a1 = (m1 == -INFINITY) ? 0.0f : exp2(m1 - n1);
      l0 *= a0; l1 *= a1;
      bc00 *= a0; bc10 *= a0; bc01 *= a1; bc11 *= a1;
      const simdgroup_float8x8 Dg = diag_cols(a0, a1, fm, fn);
      for (int t = 0; t < 16; ++t) {
        simdgroup_float8x8 Tm;
        simdgroup_multiply(Tm, OT[t], Dg);
        OT[t] = Tm;
      }
      m0 = n0; m1 = n1;
    }
    const float p0 = (s0 == -INFINITY) ? 0.0f : exp2(s0 - m0);
    const float p1 = (s1 == -INFINITY) ? 0.0f : exp2(s1 - m1);
    l0 += p0; l1 += p1;
"""


def _pv() -> str:
    """O^T_t += C^T_t (s_g P)^T on the raw codes; the running sum of p (b_g - 1024 s_g) per head and group"""
    src = (
        "    const uint rv0 = k0 + uint(fn), rv1 = rv0 + 1u;\n"
        "    const uint2 vw00 = *(const device uint2*)(Vq + size_t(rv0) * 128 + 8 * fm);\n"
        "    const uint2 vw01 = *(const device uint2*)(Vq + size_t(rv1) * 128 + 8 * fm);\n"
        "    const uint2 vw10 = *(const device uint2*)(Vq + size_t(rv0) * 128 + 64 + 8 * fm);\n"
        "    const uint2 vw11 = *(const device uint2*)(Vq + size_t(rv1) * 128 + 64 + 8 * fm);\n"
        "    const uint2 wv = *(const device uint2*)(Vsb + size_t(rk) * 2);      // the P row's key: k0 + fm\n"
        "    const float vs0 = lo16(wv.x), vb0 = hi16(wv.x), vs1 = lo16(wv.y), vb1 = hi16(wv.y);\n"
        "    const float vc0 = vb0 - 1024.0f * vs0, vc1 = vb1 - 1024.0f * vs1;\n"
        "    bc00 += p0 * vc0; bc01 += p1 * vc0; bc10 += p0 * vc1; bc11 += p1 * vc1;\n"
        "    simdgroup_float8x8 PT0, PT1;\n"
        "    PT0.thread_elements()[0] = p0 * vs0; PT0.thread_elements()[1] = p1 * vs0;\n"
        "    PT1.thread_elements()[0] = p0 * vs1; PT1.thread_elements()[1] = p1 * vs1;\n"
    )
    for g in range(2):
        c0, e0 = _unpack_half(
            f"vq{g}0_", [f"vw{g}0.x", f"vw{g}0.y"]
        )  # key row k0 + fn: tiles tt 0..7
        c1, e1 = _unpack_half(f"vq{g}1_", [f"vw{g}1.x", f"vw{g}1.y"])  # key row k0 + fn + 1
        src += c0 + c1 + "    { simdgroup_half8x8 Vt;\n"
        for tt in range(8):
            t = 8 * g + tt
            src += (
                f"      Vt.thread_elements()[0] = {e0[tt]}; Vt.thread_elements()[1] = {e1[tt]};\n"
                f"      simdgroup_multiply_accumulate(OT[{t}], Vt, PT{g}, OT[{t}]);\n"
            )
        src += "    }\n"
    return src


def _store() -> str:
    """m, l and o (+ the reduced bias term) in att6's partial row layout, natural head-dim order"""
    src = r"""  float2 lv = float2(l0, l1);
  lv += simd_shuffle_xor(lv, ushort(2));
  lv += simd_shuffle_xor(lv, ushort(4));
  lv += simd_shuffle_xor(lv, ushort(16));
  device float* r0 = rows + fn * PROW;
  device float* r1 = rows + (fn + 1) * PROW;
  if (fm == 0) { r0[0] = m0; r0[1] = lv.x; r1[0] = m1; r1[1] = lv.y; }
  float4 bc = float4(bc00, bc01, bc10, bc11);
  bc += simd_shuffle_xor(bc, ushort(2));
  bc += simd_shuffle_xor(bc, ushort(4));
  bc += simd_shuffle_xor(bc, ushort(16));
"""
    for g in range(2):
        b0, b1 = ("bc.x", "bc.y") if g == 0 else ("bc.z", "bc.w")
        for h in range(2):  # tiles 8 g + 4 h .. + 3 = head dims 64 g + 8 fm + 4 h .. + 3
            ts = [8 * g + 4 * h + k for k in range(4)]
            src += (
                f"  *(device float4*)(r0 + 4 + {64 * g + 4 * h} + 8 * fm) = float4("
                + ", ".join(f"OT[{t}].thread_elements()[0] + {b0}" for t in ts)
                + ");\n"
            )
            src += (
                f"  *(device float4*)(r1 + 4 + {64 * g + 4 * h} + 8 * fm) = float4("
                + ", ".join(f"OT[{t}].thread_elements()[1] + {b1}" for t in ts)
                + ");\n"
            )
    return src


def att6q8_src(name: str = "att6q8") -> str:
    """Metal source of the folded kv8 item: inline void <name>(q8, Kq, Ksb, Vq, Vsb, kb, ke, rows, lane), the same
    signature and partial-row contract as layout 6d's other items (one simdgroup = one item; 8 q heads of one kv head)."""
    return (
        f"inline void {name}(const device ushort* q8, const device uchar* Kq, const device uint* Ksb, "
        "const device uchar* Vq, const device uint* Vsb, uint kb, uint ke, device float* rows, uint lane) {\n"
        "  const short fm = frag_fm(lane), fn = frag_fn(lane);\n"
        + _q_tiles()
        + "  simdgroup_float8x8 OT[16];\n"
        "  for (int t = 0; t < 16; ++t) OT[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n"
        "  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;\n"
        "  float bc00 = 0.0f, bc01 = 0.0f, bc10 = 0.0f, bc11 = 0.0f;   // V bias terms of group g, heads fn / fn + 1\n"
        "  for (uint k0 = kb; k0 < ke; k0 += 8) {\n    const uint rk = k0 + uint(fm);\n"
        + _scores()
        + _SOFTMAX
        + _pv()
        + "  }\n"
        + _store()
        + "}\n"
    )
