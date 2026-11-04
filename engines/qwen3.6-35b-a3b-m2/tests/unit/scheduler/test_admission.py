"""Admission's view of the work still to come, on a plain queue (no MLX needed)."""

import queue
import threading
from types import SimpleNamespace

from jitengine_qwen3_6_35b_a3b_m2.scheduler.admission import AdmissionMixin


def _engine(queued_prompts):
    q = queue.Queue()
    for ids in queued_prompts:
        q.put(SimpleNamespace(req=SimpleNamespace(prompt_ids=ids)))
    return SimpleNamespace(_pending=[], _queue=q)


def test_a_queued_request_that_restores_the_prefix_counts_as_a_use():
    eng = _engine([[1, 2, 3, 4], [9, 9]])
    assert AdmissionMixin._pending_uses(eng, (1, 2, 3))
    assert not AdmissionMixin._pending_uses(eng, (9, 9))  # the prompt must go past the prefix
    assert not AdmissionMixin._pending_uses(eng, (5,))


def test_the_queue_is_read_under_its_own_lock():
    """A submitting thread appends under `Queue.mutex`; the read takes the same lock, so it never
    meets the deque mid-change."""
    eng = _engine([[1, 2, 3]])
    seen = []
    eng._queue.mutex.acquire()
    t = threading.Thread(target=lambda: seen.append(AdmissionMixin._pending_uses(eng, (1, 2))))
    t.start()
    t.join(0.2)
    assert t.is_alive() and not seen  # waits for the lock
    eng._queue.mutex.release()
    t.join(5)
    assert seen == [True]
