"""Model-free GPU checks of the kv8 profile's kernel primitives (no weights; seconds; < 1 GB).

    pytest tests/metal/kernels/test_kv8_src.py        # renders from tests/data/flint-int4's headers
    python tests/metal/kernels/test_kv8_src.py --out <json> --model DIR

The primitives are compiled from the kv8 megakernel's own rendered header (kv8_src.py inside metal_src.render), so the
functions under test are the production text:
  1. kv8_quant (quantise-on-write) vs mx.quantize(group_size=64, bits=8): codes, scales and biases bit for bit on varied
     rows (normal, all positive / negative, tiny, large, zero groups, repeated values, constant rows, wide spreads);
  2. the history's scale / bias words (lo16 / hi16, the affine terms att6q8 folds in) decode to mx.dequantize: the
     reference dequantisation bf16(fma(s, q, b)) vs mx.dequantize, bit for bit;
  3. att6q8 (the folded item, kv8_fold.py) vs an fp64 reference over the EXACT affine history s * c + b (the item's
     semantics: it never forms bf16 dequantised values), per item (m, l, o / l; tolerance 1e-4),
     over chunks of a request-fixed grid, including partial last blocks and an empty chunk. Reported, not judged: its
     distance to att6 over mx.dequantize's bf16 history (the per-element dequantising item's semantics, and what kv8
     prefill attends)
     and the distance between the two fp64 references (the bf16 rounding of the dequantised values).
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from jitengine_flint_m2.kernels.mathmode import METAL_COMPILE_OPTIONS

QUANT_SRC = r"""
  threadgroup float hb[128];
  const uint row = threadgroup_position_in_grid.x, tid = thread_index_in_threadgroup;
  hb[tid] = float(x[size_t(row) * 128u + tid]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 64u) kv8_quant((device uchar*)codes + size_t(row) * 128u, (device uint*)sb + size_t(row) * 2u, hb, tid >> 5,
                           thread_index_in_simdgroup);
"""
DEQ_SRC = r"""
  const uint i = thread_position_in_grid.x;              // 4 codes of one row
  if (i >= prm[0]) return;
  const uint w = ((const device uint*)codes)[i];
  const uint sbw = ((const device uint*)sb)[i / 16u];   // 16 words of codes per 64-dim group
  const float s = lo16(sbw), b = hi16(sbw);
  const bfloat4 d = bfloat4(fma(float4(as_type<uchar4>(w)), s, b));
  ((device bfloat4*)out)[i] = d;
"""
ATT8_SRC = r"""
  const uint item = threadgroup_position_in_grid.x;
  const uint kb = prm[2 * item], ke = prm[2 * item + 1];
  att6q8((const device ushort*)q, (const device uchar*)Kq, (const device uint*)Ksb, (const device uchar*)Vq,
         (const device uint*)Vsb, kb, ke, (device float*)rows + size_t(item) * 8u * PROW, thread_index_in_simdgroup);
"""
ATT16_SRC = r"""
  const uint item = threadgroup_position_in_grid.x;
  const uint kb = prm[2 * item], ke = prm[2 * item + 1];
  att6((const device ushort*)q, (const device ushort*)K, (const device ushort*)V, kb, ke,
       (device float*)rows + size_t(item) * 8u * PROW, thread_index_in_simdgroup);
"""


def header(model: str) -> str:
    from jitengine_flint_m2.kernels import metal_src as MS
    from jitengine_flint_m2.kernels import plan as P
    from jitengine_flint_m2.model.config import load_config, tensor_index
    from jitengine_flint_m2.weight_banks import BankPlan

    cfg = load_config(model)
    eo = P.expert_record_offsets(BankPlan(cfg, tensor_index(model, cfg)))
    h, _ = MS.render(cfg, eo, True, kv8=True)
    return h


def u16(a):
    import mlx.core as mx

    return np.array(a.reshape(-1).view(mx.uint16))


def check_quant(h: str) -> dict:
    import mlx.core as mx

    rng = np.random.default_rng(7)
    rows = []
    for kind in range(10):
        x = rng.normal(size=(2048, 128)).astype(np.float32)
        if kind == 1:
            x = np.abs(x)  # all positive
        if kind == 2:
            x = -np.abs(x)  # all negative
        if kind == 3:
            x = x * 1e-3  # tiny
        if kind == 4:
            x = x * 300.0  # large
        if kind == 5:
            x[:, :64] = 0.0  # zero groups
        if kind == 6:
            x = np.round(x * 4) / 4  # repeated values
        if kind == 7:
            x[::7] = x[::7] * 0 + 1.25  # constant rows
        if kind == 8:
            x = x * np.exp(rng.normal(size=(2048, 1)) * 3)  # wide per-row spreads
        if kind == 9:
            x[:, ::5] *= 40.0  # outlier dims (k/v-like)
        rows.append(x)
    X = mx.array(np.concatenate(rows)).astype(mx.bfloat16)
    n = X.shape[0]
    k = mx.fast.metal_kernel(
        name="kv8chk_quant",
        input_names=["x", "prm"],
        output_names=["codes", "sb"],
        header=h,
        source=QUANT_SRC,
        compile_options=METAL_COMPILE_OPTIONS,
    )
    codes, sb = k(
        inputs=[X.reshape(-1), mx.array([n], dtype=mx.uint32)],
        grid=(n * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(n * 128,), (n * 2,)],
        output_dtypes=[mx.uint8, mx.uint32],
    )
    wq, s, b = mx.quantize(X, group_size=64, bits=8)
    mx.eval(codes, sb, wq, s, b)
    sbn = np.array(sb)
    got_s, got_b = (sbn & 0xFFFF).astype(np.uint16), (sbn >> 16).astype(np.uint16)
    return {
        "rows": int(n),
        "code_mismatches": int((np.array(codes) != np.array(wq.view(mx.uint8).reshape(-1))).sum()),
        "scale_mismatches": int((got_s != u16(s)).sum()),
        "bias_mismatches": int((got_b != u16(b)).sum()),
    }


def check_dequant(h: str) -> dict:
    import mlx.core as mx

    out = {}
    k = mx.fast.metal_kernel(
        name="kv8chk_deq",
        input_names=["codes", "sb", "prm"],
        output_names=["out"],
        header=h,
        source=DEQ_SRC,
        compile_options=METAL_COMPILE_OPTIONS,
    )
    for scale in (0.02, 0.5, 3.0, 60.0, 900.0):
        X = (mx.random.normal((8192, 128), key=mx.random.key(int(scale * 100))) * scale).astype(
            mx.bfloat16
        )
        wq, s, b = mx.quantize(X, group_size=64, bits=8)
        sbw = (
            s.view(mx.uint16).astype(mx.uint32) | (b.view(mx.uint16).astype(mx.uint32) << 16)
        ).reshape(-1)
        nw = X.size // 4
        o = k(
            inputs=[wq.reshape(-1), sbw, mx.array([nw], dtype=mx.uint32)],
            grid=(nw, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(X.size,)],
            output_dtypes=[mx.bfloat16],
        )[0]
        ref = mx.dequantize(wq, s, b, group_size=64, bits=8)
        mx.eval(o, ref)
        out[f"scale{scale}"] = int((u16(o) != u16(ref)).sum())
    return {"mismatches": out}


def check_att(h: str, L: int = 3000, CH: int = 48) -> dict:  # noqa: PLR0915
    """att6q8 vs fp64 attention over the exact affine history (judged), and, for information, vs att6 over
    mx.dequantize's bf16 history and vs fp64 over that history, for one kv head."""
    import mlx.core as mx

    from jitengine_flint_m2.kernels import plan as P

    cap = -(-(L + 8) // 8) * 8
    rng = mx.random.key(3)
    K = (mx.random.normal((cap, 128), key=mx.random.split(rng, 3)[0]) * 1.5).astype(mx.bfloat16)
    V = (mx.random.normal((cap, 128), key=mx.random.split(rng, 3)[1]) * 0.7).astype(mx.bfloat16)
    q = (mx.random.normal((8, 128), key=mx.random.split(rng, 3)[2]) * 3.0).astype(mx.bfloat16)
    kq, ks, kbias = mx.quantize(K, group_size=64, bits=8)
    vq, vs, vbias = mx.quantize(V, group_size=64, bits=8)
    Kd = mx.dequantize(kq, ks, kbias, group_size=64, bits=8)
    Vd = mx.dequantize(vq, vs, vbias, group_size=64, bits=8)

    def sbw(s, b):
        return (
            s.view(mx.uint16).astype(mx.uint32) | (b.view(mx.uint16).astype(mx.uint32) << 16)
        ).reshape(-1)

    items = [(c * CH, min(c * CH + CH, L)) for c in range(-(-L // CH))] + [(16, 16), (L - 5, L)]
    prm = mx.array(np.array(items, dtype=np.uint32).reshape(-1))
    n = len(items)
    shape = [(n * 8 * P.PROW,)]
    k8 = mx.fast.metal_kernel(
        name="kv8chk_att8",
        input_names=["q", "Kq", "Ksb", "Vq", "Vsb", "prm"],
        output_names=["rows"],
        header=h,
        source=ATT8_SRC,
        compile_options=METAL_COMPILE_OPTIONS,
    )
    k16 = mx.fast.metal_kernel(
        name="kv8chk_att16",
        input_names=["q", "K", "V", "prm"],
        output_names=["rows"],
        header=h,
        source=ATT16_SRC,
        compile_options=METAL_COMPILE_OPTIONS,
    )
    r8 = k8(
        inputs=[q, kq.reshape(-1), sbw(ks, kbias), vq.reshape(-1), sbw(vs, vbias), prm],
        grid=(n * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=shape,
        output_dtypes=[mx.float32],
        init_value=0,
    )[0]
    r16 = k16(
        inputs=[q, Kd, Vd, prm],
        grid=(n * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=shape,
        output_dtypes=[mx.float32],
        init_value=0,
    )[0]
    mx.eval(r8, r16)
    a8 = np.array(r8).reshape(n, 8, P.PROW)
    a16 = np.array(r16).reshape(n, 8, P.PROW)
    qf = np.array(q.astype(mx.float32)).astype(np.float64)
    Kf = np.array(Kd.astype(mx.float32)).astype(np.float64)
    Vf = np.array(Vd.astype(mx.float32)).astype(np.float64)

    def affine(wq, s, b):  # exact s * c + b of the stored codes, bf16 scale and bias
        codes = np.array(wq).view(np.uint8).reshape(cap, 128).astype(np.float64)
        return codes * np.repeat(
            np.array(s.astype(mx.float32)).astype(np.float64), 64, axis=1
        ) + np.repeat(np.array(b.astype(mx.float32)).astype(np.float64), 64, axis=1)

    Ka, Va = affine(kq, ks, kbias), affine(vq, vs, vbias)
    qscale = (1.0 / math.sqrt(128)) * math.log2(math.e)
    worst = {
        "o_rel_vs_fp64": 0.0,
        "m_abs_vs_fp64": 0.0,
        "l_rel_vs_fp64": 0.0,  # judged (affine)
        "o_rel_vs_att6": 0.0,
        "o_rel_vs_fp64_bf16_history": 0.0,
        "m_abs_vs_fp64_bf16_history": 0.0,
        "ref_affine_vs_bf16_history_o_rel": 0.0,
        "att6_o_rel_vs_fp64": 0.0,
    }  # information
    empty_ok = True
    for i, (kb, ke) in enumerate(items):
        if ke <= kb:
            empty_ok = empty_ok and bool(
                np.all(a8[i, :, 0] == -np.inf)
                and np.all(a8[i, :, 1] == 0)
                and np.all(a8[i, :, 4:] == 0)
            )
            continue

        # ref() runs within this iteration, so it sees its kb and ke
        def ref(Kr, Vr):  # fp64 attention of the item, log2 domain
            S = (qf @ Kr[kb:ke].T) * qscale  # noqa: B023
            m = S.max(axis=1)
            p = np.exp2(S - m[:, None])
            lsum = p.sum(axis=1)
            return m, lsum, (p @ Vr[kb:ke]) / lsum[:, None]  # noqa: B023

        m, lsum, o = ref(Ka, Va)  # the item's semantics: exact affine history
        mb, _, ob = ref(Kf, Vf)  # mx.dequantize's bf16 history
        o8 = a8[i, :, 4:] / a8[i, :, 1:2]
        o16 = a16[i, :, 4:] / a16[i, :, 1:2]
        sc = np.abs(o).max() + 1e-30
        worst["o_rel_vs_fp64"] = max(worst["o_rel_vs_fp64"], float(np.abs(o8 - o).max() / sc))
        worst["m_abs_vs_fp64"] = max(worst["m_abs_vs_fp64"], float(np.abs(a8[i, :, 0] - m).max()))
        worst["l_rel_vs_fp64"] = max(
            worst["l_rel_vs_fp64"], float((np.abs(a8[i, :, 1] - lsum) / lsum).max())
        )
        worst["o_rel_vs_att6"] = max(worst["o_rel_vs_att6"], float(np.abs(o8 - o16).max() / sc))
        worst["o_rel_vs_fp64_bf16_history"] = max(
            worst["o_rel_vs_fp64_bf16_history"], float(np.abs(o8 - ob).max() / sc)
        )
        worst["m_abs_vs_fp64_bf16_history"] = max(
            worst["m_abs_vs_fp64_bf16_history"], float(np.abs(a8[i, :, 0] - mb).max())
        )
        worst["ref_affine_vs_bf16_history_o_rel"] = max(
            worst["ref_affine_vs_bf16_history_o_rel"], float(np.abs(o - ob).max() / sc)
        )
        worst["att6_o_rel_vs_fp64"] = max(
            worst["att6_o_rel_vs_fp64"], float(np.abs(o16 - ob).max() / sc)
        )
    # rerun: bit-identical
    r8b = k8(
        inputs=[q, kq.reshape(-1), sbw(ks, kbias), vq.reshape(-1), sbw(vs, vbias), prm],
        grid=(n * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=shape,
        output_dtypes=[mx.float32],
        init_value=0,
    )[0]
    mx.eval(r8b)
    return {
        "L": L,
        "chunk": CH,
        "items": n,
        **{k: float(v) for k, v in worst.items()},
        "empty_item_exact": empty_ok,
        "rerun_bit_identical": bool(np.array_equal(np.array(r8b), np.array(r8))),
    }


def quant_ok(q: dict) -> bool:
    return q["code_mismatches"] == 0 and q["scale_mismatches"] == 0 and q["bias_mismatches"] == 0


def dequant_ok(d: dict) -> bool:
    return all(v == 0 for v in d["mismatches"].values())


def att_ok(t: dict) -> bool:
    return (
        t["att6_o_rel_vs_fp64"] < 1e-4
        and t["o_rel_vs_fp64"] < 1e-4
        and t["m_abs_vs_fp64"] < 1e-4
        and t["l_rel_vs_fp64"] < 1e-4
        and t["empty_item_exact"]
        and t["rerun_bit_identical"]
    )


def test_kv8_quant_matches_mx_quantize(requires_metal, flint_headers):
    q = check_quant(header(flint_headers))
    assert quant_ok(q), q


def test_kv8_dequant_matches_mx_dequantize(requires_metal, flint_headers):
    d = check_dequant(header(flint_headers))
    assert dequant_ok(d), d


def test_att6q8_and_att6_match_fp64(requires_metal, flint_headers):
    t = check_att(header(flint_headers))
    assert att_ok(t), t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    h = header(a.model)
    R = {"quant": check_quant(h), "dequant": check_dequant(h), "att": check_att(h)}
    R["ok"] = quant_ok(R["quant"]) and dequant_ok(R["dequant"]) and att_ok(R["att"])
    with open(a.out, "w") as f:
        json.dump(R, f, indent=1)
    print(json.dumps(R), flush=True)
    return 0 if R["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
