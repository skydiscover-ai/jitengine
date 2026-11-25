"""The start-up check's generated weights and inputs leave MLX's global generator alone (sampling draws
from it), and come out the same each time."""

import pytest

mx = pytest.importorskip("mlx.core")

from jitengine_qwen3_1_7b_m4pro.kernels import selfcheck  # noqa: E402


def test_the_check_does_not_move_the_global_generator():
    mx.random.seed(7)
    want = mx.random.normal((4,))
    mx.random.seed(7)
    selfcheck._inputs(13, 2048, 1000)
    selfcheck._random_linear(2048, 64)
    assert mx.array_equal(mx.random.normal((4,)), want)


def test_the_inputs_are_the_same_each_time():
    a, b = selfcheck._inputs(16, 2048, 1000), selfcheck._inputs(16, 2048, 1000)
    assert all(mx.array_equal(x, y) for x, y in zip(a, b, strict=True))
