"""`rope_tables_at` (model/qsa_indexer.py) memoizes lazy arrays, and MLX's streams belong to the
thread that made them: an entry left unevaluated by one thread failed the next engine's worker on
its first decode with "There is no Stream(gpu, N) in current thread" (the
reproducer: a direct model call on the main thread, then an engine at the same position). Each
thread keeps its own entries, with the same values."""

import threading

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx

from jitengine_qwen3_8_flash_next_m4pro.model import qsa_indexer as q

DIMS, BASE = 8, 10_000_000.0


def _in_thread(fn):
    out = {}

    def run():
        try:
            out["value"] = fn()
        except Exception as exc:  # noqa: BLE001
            out["error"] = exc

    t = threading.Thread(target=run)
    t.start()
    t.join()
    if "error" in out:
        raise out["error"]
    return out["value"]


def _bits(a: mx.array) -> bytes:
    return np.array(a.astype(mx.float32)).tobytes()


@pytest.mark.parametrize("s0", [0, 7, 4093])
def test_an_entry_another_thread_left_unevaluated_does_not_reach_this_thread(s0):
    left = q.rope_tables_at(s0, DIMS, BASE)  # lazy, on the main thread's stream

    def worker():
        cos, sin = q.rope_tables_at(s0, DIMS, BASE)
        mx.eval(cos * 1.0, sin * 1.0)  # as a decode step consumes them
        fresh = q.rope_tables(mx.arange(s0, s0 + 1), DIMS, BASE)
        mx.eval(*fresh)
        return [_bits(x) for x in (cos, sin)], [_bits(x) for x in fresh]

    got, fresh = _in_thread(worker)
    mx.eval(*left)
    assert got == fresh == [_bits(x) for x in left]


def test_a_thread_reuses_its_own_entry_within_a_step():
    def worker():
        a = q.rope_tables_at(11, DIMS, BASE)
        b = q.rope_tables_at(11, DIMS, BASE)
        return a[0] is b[0] and a[1] is b[1]

    assert _in_thread(worker)
