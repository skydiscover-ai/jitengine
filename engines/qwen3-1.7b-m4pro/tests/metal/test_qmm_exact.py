"""The exact kernel against stock `mx.quantized_matmul`, bit for bit, on Qwen3-1.7B's real shapes."""

import mlx.core as mx
import pytest

from jitengine_qwen3_1_7b_m4pro.kernels import qmm_exact as q

HIDDEN, INTER, VOCAB = 2048, 6144, 151936
# (name, K, output-row segments of the unfused projections)
SHAPES = [
    ("qkv", HIDDEN, (2048, 1024, 1024)),
    ("o", HIDDEN, (2048,)),
    ("gate_up", HIDDEN, (INTER, INTER)),
    ("down", INTER, (HIDDEN,)),
    ("lm_head", HIDDEN, (VOCAB,)),
]


def _weights(n, k, seed):
    mx.random.seed(seed)
    w = (mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    return wq, s, b


def _stock(x, wq, s, b, segments):
    outs, a = [], 0
    for n in segments:
        outs.append(
            mx.quantized_matmul(
                x, wq[a : a + n], s[a : a + n], b[a : a + n], transpose=True, group_size=64, bits=4
            )
        )
        a += n
    return mx.concatenate(outs, axis=-1)


@pytest.mark.parametrize("name,k,segments", SHAPES, ids=[s[0] for s in SHAPES])
@pytest.mark.parametrize("rows", [13, 14, 15, 16])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_stock(name, k, segments, rows, seed):
    n = sum(segments)
    wq, s, b = _weights(n, k, seed)
    mx.random.seed(100 + seed)
    x = mx.random.normal((rows, 1, k)).astype(mx.bfloat16)
    got = q.qmm_exact(x, wq, s, b, segments)
    assert got is not None, f"{name} at {rows} rows should be covered"
    want = _stock(x, wq, s, b, segments)
    mx.eval(got, want)
    assert got.dtype == want.dtype and got.shape == want.shape
    assert mx.array_equal(got.view(mx.uint16), want.view(mx.uint16))


@pytest.mark.parametrize("rows", [1, 4, 12, 17, 32])
def test_other_row_counts_are_left_to_stock(rows):
    wq, s, b = _weights(2048, 2048, 0)
    x = mx.zeros((rows, 1, 2048), dtype=mx.bfloat16)
    assert q.qmm_exact(x, wq, s, b) is None or 13 <= rows <= 16


def test_uncovered_k_is_left_to_stock():
    wq, s, b = _weights(2048, 4096, 0)
    x = mx.zeros((16, 1, 4096), dtype=mx.bfloat16)
    assert q.qmm_exact(x, wq, s, b) is None
