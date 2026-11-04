"""The relaxed profile's dense rows: on only inside `relaxed_rows()`, and only on the thread in it."""

from __future__ import annotations

import threading


def _layer():
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro.linear import QLinear

    w = mx.random.normal((64, 512), key=mx.random.key(3)).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=64, bits=8)  # 8 bits: outside the block, rows run stock
    return QLinear(q, s, b, 64)


def test_rows_take_the_relaxed_kernel_only_inside_the_block(requires_mlx_any, monkeypatch):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro import linear
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels import relaxed_qmm

    calls = []
    monkeypatch.setattr(relaxed_qmm, "rows", lambda x2, *rest: calls.append(x2.shape))
    layer = _layer()
    x = mx.random.normal((3, 1, 512), key=mx.random.key(4)).astype(mx.bfloat16)
    layer.rows(x)
    assert calls == []
    with linear.relaxed_rows():
        y = layer.rows(x)  # the stand-in declines (None), so the stock op runs
    assert calls == [(3, 512)] and y.shape == (3, 1, 64)
    layer.rows(x)
    assert calls == [(3, 512)]


def test_the_switch_is_per_thread_and_reset_after_an_error(requires_mlx_any):
    from jitengine_qwen3_6_35b_a3b_m4pro import linear

    entered, seen = threading.Event(), []

    def other_engine():
        entered.wait()
        seen.append(linear._RELAX_ROWS.get())

    t = threading.Thread(target=other_engine)
    t.start()
    with linear.relaxed_rows():
        entered.set()
        t.join()
        seen.append(linear._RELAX_ROWS.get())
    assert seen == [False, True]
    try:
        with linear.relaxed_rows():
            raise RuntimeError("a verify pass that fails")
    except RuntimeError:
        pass
    assert linear._RELAX_ROWS.get() is False
