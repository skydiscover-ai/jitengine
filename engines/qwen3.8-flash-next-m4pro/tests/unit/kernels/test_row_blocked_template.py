"""The template list the dense and grouped row-blocked kernels are launched with."""

import pytest

mx = pytest.importorskip("mlx.core")

from jitengine_qwen3_8_flash_next_m4pro.kernels.qmv_row_blocked import (  # noqa: E402
    row_blocked_template,
)


def test_the_template_names_the_dtypes_and_shape_parameters_in_kernel_order():
    assert row_blocked_template(mx.float32, mx.bfloat16, 2, 8, 4, 2, 64) == [
        ("T", mx.float32),
        ("SB", mx.bfloat16),
        ("VECS", 2),
        ("KL", 8),
        ("NSG", 4),
        ("RB", 2),
        ("GS", 64),
    ]


def test_an_extra_parameter_goes_after_the_family_ones():
    template = [*row_blocked_template(mx.bfloat16, mx.bfloat16, 1, 16, 1, 1, 32), ("MAIN", 7)]
    assert [name for name, _ in template] == ["T", "SB", "VECS", "KL", "NSG", "RB", "GS", "MAIN"]
