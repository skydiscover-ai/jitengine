"""The batched GDN recurrence's input contract, checked before any launch."""

import pytest


def _operands(mx, rows: int, positions: int = 1, states: int | None = None):
    """q, k, v, g, beta and the per-row states, at Qwen3.6's GDN head shape (2 key heads,
    4 value heads, 128 by 128)."""
    q = k = mx.zeros((rows, positions, 2, 128), dtype=mx.bfloat16)
    v = mx.zeros((rows, positions, 4, 128), dtype=mx.bfloat16)
    g = beta = mx.zeros((rows, positions, 4), dtype=mx.bfloat16)
    st = [mx.zeros((1, 4, 128, 128))] * (rows if states is None else states)
    return q, k, v, g, beta, st


@pytest.mark.parametrize(
    "rows, positions, states",
    [(9, 1, None), (0, 1, None), (2, 2, None), (2, 1, 1)],
    ids=["too-many-rows", "no-rows", "two-positions", "a-state-short"],
)
def test_operands_outside_the_contract_raise(requires_mlx_any, rows, positions, states):
    """One position for 1 to ROWS_MAXB rows with one state each; anything else is the caller's bug
    and raises ValueError."""
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro.kernels import gdn_recurrence as recurrence

    assert recurrence.ROWS_MAXB == 8
    with pytest.raises(ValueError, match="gdn_recurrence_rows takes one position"):
        recurrence.gdn_recurrence_rows(*_operands(mx, rows, positions, states))


def test_the_rows_kernel_covers_two_to_eight_float32_states(requires_mlx_any):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.gdn_recurrence import supported_rows

    st = mx.zeros((1, 4, 128, 128))
    assert supported_rows([st] * 2) and supported_rows([st] * 8)
    assert not supported_rows([st]) and not supported_rows([st] * 9)
    assert not supported_rows([st, st.astype(mx.bfloat16)])
