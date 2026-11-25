"""Fused projections and the exact kernel against the unpatched model, bit for bit, on real weights.

Needs the pinned checkpoint: set JITENGINE_QWEN3_1_7B_M4PRO_E2E_CHECKPOINT to its directory."""

import os

import pytest

mx = pytest.importorskip("mlx.core")
mlx_lm = pytest.importorskip("mlx_lm")

CKPT = os.environ.get("JITENGINE_QWEN3_1_7B_M4PRO_E2E_CHECKPOINT")
ROWS = [1, 5, 12, 13, 14, 15, 16, 17, 32]
LAYERS = [0, 13, 27]


@pytest.mark.skipif(
    not CKPT, reason="set JITENGINE_QWEN3_1_7B_M4PRO_E2E_CHECKPOINT to the model directory"
)
def test_patched_layers_equal_stock_layers_at_every_row_count():
    from mlx_lm.utils import load

    stock, _ = load(CKPT)
    reference = {}
    for rows in ROWS:
        mx.random.seed(rows)
        x = mx.random.normal((rows, 1, 2048)).astype(mx.bfloat16)
        for i in LAYERS:
            layer = stock.model.layers[i]
            reference[(rows, i, "attn")] = layer.self_attn(x)
            reference[(rows, i, "mlp")] = layer.mlp(x)
        reference[(rows, "x")] = x
        reference[(rows, "head")] = stock.model.embed_tokens.as_linear(x)
    mx.eval(list(reference.values()))

    from jitengine_qwen3_1_7b_m4pro.engine import fusion, patches

    patches.install(selfcheck=False)
    patched, _ = load(CKPT)
    assert fusion.fuse_model(patched)
    for rows in ROWS:
        x = reference[(rows, "x")]
        for i in LAYERS:
            layer = patched.model.layers[i]
            for name, got in (("attn", layer.self_attn(x)), ("mlp", layer.mlp(x))):
                want = reference[(rows, i, name)]
                mx.eval(got)
                assert mx.array_equal(got.view(mx.uint16), want.view(mx.uint16)), (rows, i, name)
        got = patched.model.embed_tokens.as_linear(x)
        assert mx.array_equal(got.view(mx.uint16), reference[(rows, "head")].view(mx.uint16)), (
            rows,
            "head",
        )
