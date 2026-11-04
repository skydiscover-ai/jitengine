"""The model thread for `Engine`, as a mixin: its loop, one scheduling round, taking submitted
jobs off the queue (`_accept` checks each one), and failing everything if the thread dies.
"""

from __future__ import annotations

import logging
import queue
import time

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m2.engine.errors import ContextTooLargeError
from jitengine_qwen3_6_35b_a3b_m2.engine.io_struct import GenResult
from jitengine_qwen3_6_35b_a3b_m2.scheduler.batch import Job, new_record

logger = logging.getLogger(__name__)

CLEAR_CACHE_EVERY = 64  # scheduler rounds between allocator cache releases (never mid-step)


class _Stop(BaseException):
    """Internal: ends the worker loop (Engine.close)."""


class SchedulingMixin:
    """The worker loop, and what each round does before a row is admitted."""

    def _worker_loop(self) -> None:
        """The model thread; it must never die quietly.

        A fault inside a step errors that step's rows and the loop continues. If recovery itself
        raises, the engine is marked dead and every waiter is failed, so none blocks forever.
        """
        fault: BaseException | None = None
        try:
            while True:
                try:
                    self._schedule_once()
                except _Stop:
                    fault = RuntimeError("the engine was closed")
                    return
                except BaseException as exc:  # noqa: BLE001  a step fault is recoverable
                    try:
                        logger.error(f"[engine] scheduler fault: {exc!r}")
                        self._abort_batch(exc)
                    except BaseException as inner:  # recovery failed: stop honestly
                        fault = inner
                        raise
                    time.sleep(0.01)
        finally:
            self._fail_everything(fault or RuntimeError("the model worker thread exited"))

    def _fail_everything(self, exc: BaseException) -> None:
        """Mark the engine dead and error every job: in flight, holding a slot, pending, queued."""
        self._dead = exc
        if not self._closing:  # a close() is not news; anything else ends the engine unasked
            try:
                logger.error(
                    f"[engine] model thread is gone: {exc!r}; failing every pending request"
                )
            except BaseException:  # noqa: BLE001
                pass
        for row in list(self._batch or []) + list(self._rows):
            if not row.retired:
                try:
                    self._fail(row, exc)
                except BaseException:  # noqa: BLE001
                    row.retired = True
                    row.job.out.put(("error", exc))
        self._batch, self._y, self._telem = None, None, None
        for job in list(self._pending) + list(self._scores):
            job.out.put(("error", exc))
        self._pending, self._scores = [], []
        self._reap_queue(exc)

    def _reap_queue(self, exc: BaseException) -> None:
        """Error everything still on the submission queue."""
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                return
            job.out.put(("error", exc))

    def _schedule_once(self) -> None:
        idle = self._batch is None and not self._rows and not self._pending and not self._scores
        self._drain(block=idle)
        self._run_scores()
        self._admit()
        self._prompt_slice()
        self._pump()
        self._work += 1
        if self._batch is None and not self._rows and not self._pending:
            mx.clear_cache()
        elif self._work % CLEAR_CACHE_EVERY == 0:
            mx.clear_cache()

    def _drain(self, block: bool) -> None:
        """Move queued jobs into the scheduler's lists, blocking only when the engine is idle."""
        if block:
            self._take(self._queue.get())
        while True:
            try:
                self._take(self._queue.get_nowait())
            except queue.Empty:
                return

    def _take(self, job: Job) -> None:
        if job.kind == "score":
            self._scores.append(job)
        elif job.kind == "generate":
            self._accept(job)
        elif job.kind == "stop":
            raise _Stop()
        elif job.kind == "maintenance":
            # run inline, in queue order: earlier jobs are already in the scheduler's lists (and
            # count as outstanding work), later ones cannot start until the command finishes
            try:
                job.out.put(("done", self._run_maintenance(job.payload)))
            except BaseException as exc:  # noqa: BLE001
                job.out.put(("error", exc))
        else:
            job.out.put(("error", ValueError(f"unknown job {job.kind}")))

    def _accept(self, job: Job) -> None:
        """Validate a generate job and queue it; a bad request fails only itself, before a slot."""
        req = job.req
        try:
            prompt = [int(t) for t in req.prompt_ids]
            if not prompt:
                raise ValueError("empty prompt")
            for t in prompt:
                if t < 0 or t >= self.token_id_limit:
                    raise ValueError(f"token id {t} out of range")
            max_tokens = int(req.max_tokens)
            if max_tokens < 0:
                raise ValueError("max_tokens must be >= 0")
            # bound the whole generation, so a lone row always fits the KV budget at admission
            if len(prompt) + max_tokens > self.max_kv_positions:
                raise ContextTooLargeError(
                    f"prompt + max_tokens = {len(prompt) + max_tokens} tokens exceeds the context "
                    f"limit of {self.max_kv_positions} tokens"
                )
        except Exception as exc:  # noqa: BLE001
            job.out.put(("error", exc))
            return
        if max_tokens <= 0:
            rec = new_record(req, len(prompt), (time.monotonic() - job.submitted) * 1e3)
            job.out.put(
                (
                    "done",
                    GenResult(
                        token_ids=[], finish_reason="length", prompt_tokens=len(prompt), extra=rec
                    ),
                )
            )
            return
        job.req = req
        job.payload = (prompt, max_tokens)
        job.head = prompt[:-1]
        self._pending.append(job)
        self._pending_dirty = True
