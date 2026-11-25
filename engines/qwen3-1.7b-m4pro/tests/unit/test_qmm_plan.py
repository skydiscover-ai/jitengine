import pytest

mx = pytest.importorskip("mlx.core")

from jitengine_qwen3_1_7b_m4pro.engine.shapes import SHAPES  # noqa: E402
from jitengine_qwen3_1_7b_m4pro.kernels import qmm_exact as q  # noqa: E402


@pytest.mark.parametrize(
    "n,k,rows,want",
    [
        (6144, 2048, 16, 2),
        (12288, 2048, 13, 1),
        (2048, 6144, 16, 8),
        (2048, 2048, 14, 8),
        (1024, 2048, 15, 16),
        (151936, 2048, 16, 1),
        (4096, 6144, 13, 4),
    ],
)
def test_split_k_table(n, k, rows, want):
    assert q.split_k(rows, k, n) == want


@pytest.mark.parametrize("rows", [1, 4, 12, 17, 32])
def test_other_row_counts_are_stock(rows):
    assert all(q.plan(rows, k, seg) is None for _, k, seg in SHAPES)


def test_uncovered_k_and_n_are_stock():
    assert q.split_k(16, 4096, 2048) is None
    assert q.split_k(16, 2048, 100) is None
    assert q.split_k(16, 2048, 512) is None


def test_every_real_shape_is_covered_at_13_to_16_rows():
    for _, k, seg in SHAPES:
        for rows in range(13, 17):
            assert q.plan(rows, k, seg) is not None, (k, seg, rows)


def test_disable_sends_a_shape_to_stock():
    try:
        q.disable(16, 2048, 2048)
        assert q.plan(16, 2048, (2048,)) is None
        assert q.plan(15, 2048, (2048,)) is not None
    finally:
        q.DISABLED.clear()
        q._plans.clear()


def test_eligibility_needs_bf16_4bit_decode():
    x = mx.zeros((16, 1, 2048), dtype=mx.bfloat16)
    s = mx.zeros((2048, 32), dtype=mx.bfloat16)
    b = mx.zeros((2048, 32), dtype=mx.bfloat16)
    assert q.eligible(x, s, b, 64, 4, "affine", 2048)
    assert not q.eligible(x.astype(mx.float16), s.astype(mx.float16), b, 64, 4, "affine", 2048)
    assert not q.eligible(x, s, b, 64, 8, "affine", 2048)
    assert not q.eligible(x, s, b, 32, 4, "affine", 2048)
    assert not q.eligible(x, s, None, 64, 4, "affine", 2048)
    assert not q.eligible(mx.zeros((12, 1, 2048), dtype=mx.bfloat16), s, b, 64, 4, "affine", 2048)
    assert not q.eligible(mx.zeros((16, 2, 2048), dtype=mx.bfloat16), s, b, 64, 4, "affine", 2048)
