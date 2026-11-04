"""The model thread's loop (`SchedulingMixin._worker_loop`) on stand-ins: a fault that recurs every
round and that no step's failure clears (scheduler policy code, not a decode step) must not retry
forever while requests wait. After `STUCK_FAULT_ROUNDS` such rounds
the engine is marked dead and every waiter (pending, holding a slot, score, queued) gets an error;
a fault that a step's failure clears keeps the engine serving, as before."""

import queue
import threading
from types import SimpleNamespace

import pytest

from jitengine_qwen3_8_flash_next_m4pro.scheduler.batch import Job
from jitengine_qwen3_8_flash_next_m4pro.scheduler.retire import RetirementMixin
from jitengine_qwen3_8_flash_next_m4pro.scheduler.scheduler import SchedulingMixin

WAIT_S = 5.0


class Pool:
    def __init__(self):
        self.released = []

    def release(self, slot, kv):
        self.released.append(slot)

    def note_allocated(self, delta):
        pass


class Stats:
    def __init__(self):
        self.records = []

    def record(self, rec, log=True):
        self.records.append(dict(rec))


class Loop(SchedulingMixin, RetirementMixin):
    """Just the loop's state; `_schedule_once` is whatever the test makes it."""

    def __init__(self, schedule):
        self._queue = queue.Queue()
        self._pending, self._scores, self._rows = [], [], []
        self._batch, self._y, self._telem, self._reads = None, None, None, []
        self._dead = None
        self.scheduler_faults = 0
        self.rounds = 0
        self.pool, self.stats_, self.log_requests, self.expert_pool = Pool(), Stats(), False, None
        self._schedule = schedule

    def _schedule_once(self):
        self.rounds += 1
        self._schedule(self)


def _row(slot):
    job = Job("generate", SimpleNamespace(request_id=f"r{slot}"))
    rec = {"finish_reason": "length"}
    return SimpleNamespace(
        job=job,
        retired=False,
        ids=[],
        steps=0,
        rec=rec,
        slot=slot,
        kv_bytes=0,
        kv_alloc=0,
        state=None,
        task=None,
        pool0=None,
    )


def _start(loop):
    t = threading.Thread(target=loop._worker_loop, daemon=True)
    t.start()
    return t


def _stop(loop, t):
    """End a loop that is still running (the shipped one retries forever): make recovery raise."""

    def boom(exc):
        raise RuntimeError("test teardown")

    loop._abort_batch = boom
    t.join(timeout=WAIT_S)


def test_a_policy_fault_every_round_fails_every_waiter_and_marks_the_engine_dead():
    def schedule(loop):
        raise OSError(2, "No such file or directory", "mtp.safetensors")

    loop = Loop(schedule)
    pending, score, queued = Job("generate"), Job("score"), Job("generate")
    lone = _row(0)  # a lone MTP row: holds a slot but is in no decode batch
    loop._pending, loop._scores, loop._rows = [pending], [score], [lone]
    loop._queue.put(queued)
    t = _start(loop)
    try:
        for job in (pending, score, lone.job, queued):
            kind, exc = job.out.get(timeout=WAIT_S)
            assert kind == "error" and isinstance(exc, OSError)
        t.join(timeout=WAIT_S)
        assert not t.is_alive() and isinstance(loop._dead, OSError)
        assert loop.scheduler_faults == loop.rounds <= 2
        assert loop._pending == [] and loop._scores == [] and lone.retired
        assert loop.pool.released == [0] and loop.stats_.records[0]["finish_reason"] == "error"
    finally:
        if t.is_alive():
            _stop(loop, t)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")  # the teardown
def test_a_fault_that_a_step_failure_clears_keeps_the_engine_serving():
    """A step fault fails that step's rows; the next round runs, and the loop goes on."""

    def schedule(loop):
        if getattr(loop, "stop", False):
            raise RuntimeError("test teardown")
        if loop.rounds <= 3:
            loop._batch = [_row(loop.rounds)]  # a step in flight, which faults
            raise RuntimeError("Metal command buffer failed")
        if loop.rounds == 4:
            loop.ok = True
        loop._queue.get()  # idle, as `_drain(block=True)`

    loop = Loop(schedule)
    t = _start(loop)
    try:
        deadline = threading.Event()
        for _ in range(int(WAIT_S / 0.01)):
            if getattr(loop, "ok", False):
                break
            deadline.wait(0.01)
        assert getattr(loop, "ok", False) and loop._dead is None and t.is_alive()
        assert len(loop.stats_.records) == 3  # each faulting step failed its own row
    finally:
        loop.stop = True
        loop._queue.put(Job("generate"))  # wake the idle round, which then ends the loop
        _stop(loop, t)
