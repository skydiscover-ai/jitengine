"""The resident prefill returns MLX's buffer cache before a shorter last chunk and when it ends.

MLX serves an allocation only from a cached buffer of about its size, so the full-length chunks' transients cannot
serve a shorter last chunk, and decode reuses none of them. At full context they took the process past the auto
budget (31,816 MiB at a 31,764 MiB wired limit; 31,175 with the release). No chunk is computed here:
model.prefill_chunk and mx.clear_cache are recorded stand-ins.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from jitengine_flint_m2.engine.engine import Engine
from jitengine_flint_m2.engine.io_struct import Cancelled


def C(c0, m):
    return ("chunk", c0, m)


X = ("clear",)


@pytest.fixture
def recorded(monkeypatch):
    import mlx.core as mx

    from jitengine_flint_m2.model import model as M

    events = []

    def fake_chunk(W, hist, toks, start, rows):
        events.append(C(int(start), int(toks.shape[0])))
        return mx.zeros((len(rows), 4), dtype=mx.bfloat16)

    monkeypatch.setattr(M, "prefill_chunk", fake_chunk)
    monkeypatch.setattr(mx, "clear_cache", lambda: events.append(X))
    with mx.stream(mx.cpu):
        yield events


def engine(chunk, kv_bits, hist):
    return SimpleNamespace(
        hist=hist,
        W=None,
        prefill_chunk=chunk,
        kv_bits=kv_bits,
        memory=SimpleNamespace(mode="resident"),
    )


def run(events, n, chunk, kv_bits):
    events.clear()
    hist = SimpleNamespace(length=-1, valid=None, state_arrays=lambda: [])
    hist.invalidate = lambda: (setattr(hist, "length", 0), setattr(hist, "valid", False))
    last = Engine._prefill(engine(chunk, kv_bits, hist), list(range(n)))
    assert last.shape == (4,) and hist.length == n and hist.valid is True
    return list(events)


def test_prefill_releases_the_cache(recorded):
    # a shorter last chunk, then the boundary
    assert run(recorded, 5, 2, 16) == [C(0, 2), C(2, 2), X, C(4, 1), X]
    # every chunk full: the boundary only
    assert run(recorded, 6, 2, 16) == [C(0, 2), C(2, 2), C(4, 2), X]
    # one short chunk: nothing is cached before it
    assert run(recorded, 1, 2, 16) == [C(0, 1), X]
    assert run(recorded, 4100, 2048, 16) == [C(0, 2048), C(2048, 2048), X, C(4096, 4), X]
    # kv8 keeps its per-chunk release, plus the two
    assert run(recorded, 5, 2, 8) == [C(0, 2), X, C(2, 2), X, X, C(4, 1), X, X]


def test_a_cancelled_prefill_releases_nothing(recorded):
    cancel = threading.Event()
    cancel.set()
    hist = SimpleNamespace(length=0, valid=False, invalidate=lambda: None, state_arrays=lambda: [])
    with pytest.raises(Cancelled):
        Engine._prefill(engine(2, 16, hist), list(range(5)), cancel=cancel)
    assert recorded == []
