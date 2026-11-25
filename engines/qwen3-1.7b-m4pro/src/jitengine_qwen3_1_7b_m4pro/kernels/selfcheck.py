"""Load-time check that the exact kernel and the fused projections still equal stock MLX here.

Another MLX version or GPU family may split K differently, so each shape is compared bit for bit
with stock on this device. A shape that differs, or raises, is switched off: stock runs it.
"""

from __future__ import annotations

import logging
import time

import mlx.core as mx

from jitengine_qwen3_1_7b_m4pro.engine import fusion
from jitengine_qwen3_1_7b_m4pro.engine.shapes import SHAPES
from jitengine_qwen3_1_7b_m4pro.kernels import qmm_exact

FUSED_ROWS = tuple(range(1, qmm_exact.MAX_ROWS + 1))  # every row count a fused decode can have


def _stock(x, w, s, b, segments):
    outs, a = [], 0
    for n in segments:
        outs.append(
            mx.quantized_matmul(
                x, w[a : a + n], s[a : a + n], b[a : a + n], transpose=True, group_size=64, bits=4
            )
        )
        a += n
    return mx.concatenate(outs, axis=-1)


def _same(a, b) -> bool:
    mx.eval(a, b)
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)))
    )


# Every draw takes its own key: the check leaves MLX's global generator, which sampling draws from,
# as the process started it.
def _random_linear(k, n, seed=0):
    w = (mx.random.normal((n, k), key=mx.random.key(seed)) * 0.05).astype(mx.bfloat16)
    return mx.quantize(w, group_size=64, bits=4)


def _inputs(rows: int, k: int, base: int):
    """Several decode inputs for one shape: three N(0,1) draws, one with a few large columns
    (activations of real models have outliers) and one of zeros."""
    out = []
    for i in range(3):
        key = mx.random.key(base + 97 * i + rows)
        out.append(mx.random.normal((rows, 1, k), key=key).astype(mx.bfloat16))
    x = mx.random.normal((rows, 1, k), key=mx.random.key(base + 500 + rows))
    spike = mx.where(mx.arange(k) % 211 == 0, 60.0, 1.0)
    out.append((x * spike).astype(mx.bfloat16))
    out.append(mx.zeros((rows, 1, k), dtype=mx.bfloat16))
    return out


def model_linears(model):
    """(name, k, segments, w, scales, biases) for each decode shape of layer 0 and the output head."""
    layer = model.model.layers[0]
    a, m = layer.self_attn, layer.mlp
    found = []
    for name, lin in (
        ("qkv", a.get("qkv_proj")),
        ("o", a.o_proj),
        ("gate_up", m.get("gate_up_proj")),
        ("down", m.down_proj),
    ):
        if lin is not None:
            found.append((name, lin))
    head = model.get("lm_head") or model.model.embed_tokens
    found.append(("lm_head", head))
    out = []
    for name, lin in found:
        n = lin.weight.shape[0]
        k = lin.weight.shape[1] * 32 // lin.bits
        out.append(
            (
                name,
                k,
                tuple(getattr(lin, fusion.SEGMENTS, (n,))),
                lin.weight,
                lin.scales,
                lin["biases"],
            )
        )
    return out


def synthetic_linears():
    return [(n, k, seg, *_random_linear(k, sum(seg))) for n, k, seg in SHAPES]


def run(model=None) -> dict:
    """Check every shape; disable what differs. Uses the model's own layer-0 weights when given."""
    t0 = time.perf_counter()
    linears = model_linears(model) if model is not None else synthetic_linears()
    checks, bad_rows, exact_ok = [], [], {}
    for name, k, segments, w, s, b in linears:
        for rows in range(qmm_exact.MIN_ROWS, qmm_exact.MAX_ROWS + 1):
            if qmm_exact.plan(rows, k, segments) is None:
                checks.append({"shape": name, "rows": rows, "path": "stock", "ok": True})
                continue
            try:
                ok = all(
                    _same(qmm_exact.qmm_exact(x, w, s, b, segments), _stock(x, w, s, b, segments))
                    for x in _inputs(rows, k, 1000)
                )
                err = None
            except Exception as e:  # noqa: BLE001 - a kernel that fails to build or run is off
                ok, err = False, f"{type(e).__name__}: {e}"
            if not ok:
                for seg in segments:
                    qmm_exact.disable(rows, k, seg)
            exact_ok[(name, rows)] = ok
            checks.append(
                {
                    "shape": name,
                    "rows": rows,
                    "path": "exact" if ok else "stock",
                    "ok": ok,
                    "error": err,
                }
            )
        if len(segments) > 1:
            for rows in FUSED_ROWS:
                if rows >= qmm_exact.MIN_ROWS and exact_ok.get((name, rows)):
                    continue  # the fused call runs the exact kernel, checked above
                # otherwise the fused call is stock on the joined weights: it must equal the
                # unfused stock calls (stock splits K differently for a joined N)
                try:
                    ok = all(
                        _same(
                            mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4),
                            _stock(x, w, s, b, segments),
                        )
                        for x in _inputs(rows, k, 2000)
                    )
                except Exception:  # noqa: BLE001
                    ok = False
                checks.append({"shape": name, "rows": rows, "path": "fused-stock", "ok": ok})
                if not ok:
                    bad_rows.append(rows)
    fusion_ok = not bad_rows
    # fused decode is used up to the first row count whose fused call is not exact
    fusion.FUSED_MAX_ROWS = min(bad_rows) - 1 if bad_rows else qmm_exact.MAX_ROWS
    return {
        "checks": checks,
        "exact_shapes_ok": all(c["ok"] for c in checks if c["path"] != "fused-stock"),
        "fusion_ok": fusion_ok,
        "fused_max_rows": fusion.FUSED_MAX_ROWS,
        "disabled": sorted(qmm_exact.DISABLED),
        "seconds": round(time.perf_counter() - t0, 2),
    }


def warnings(result: dict) -> list[str]:
    """What a user should be told when the check switched something off."""
    out = []
    if result["disabled"]:
        shapes = ", ".join(f"{rows} rows K={k} N={n}" for rows, k, n in result["disabled"])
        out.append(
            f"the exact kernel differs from stock MLX here for: {shapes}. Those shapes run on stock "
            "MLX, so the engine is slower than its published speed. Is the installed MLX the "
            "pinned version? See `doctor`."
        )
    if not result["fusion_ok"]:
        rows = result.get("fused_max_rows", 0)
        out.append(
            "fused projections differ from stock MLX here above "
            f"{rows} rows; decode with more rows runs unfused and slower."
        )
    return out


def run_for(model) -> dict:
    """Check with the model's own layer-0 weights; if they cannot be read (a LoRA adapter wraps the
    layers), check the same shapes on generated weights and say so."""
    try:
        return run(model)
    except Exception as e:  # noqa: BLE001 - the check must never stop the server from loading
        logging.warning(
            "startup self-check could not read the model's layers (%s); checking generated weights",
            e,
        )
        result = run()
        result["note"] = f"generated weights: {type(e).__name__}: {e}"
        return result
