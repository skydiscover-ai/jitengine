"""Retirement for `Engine`, as a mixin: finishing, cancelling and failing rows, and handing their
slots back.
"""

from __future__ import annotations

import time

from jitengine_qwen3_8_flash_next_m4pro.engine.io_struct import GenResult
from jitengine_qwen3_8_flash_next_m4pro.scheduler.batch import Req


class RetirementMixin:
    """Finishing, cancelling and failing rows."""

    def _note_expert_counters(self, row: Req) -> None:
        """Expert-cache counter deltas over the request (an interval shared by a batch)."""
        if self.expert_pool is None or row.pool0 is None:
            return
        now = self.expert_pool.counters()
        row.rec.update({k: now[k] - row.pool0[k] for k in now})

    def _close(self, row: Req) -> None:
        if row in self._rows:
            self._rows.remove(row)
        self.pool.release(row.slot, row.kv_bytes)
        self.pool.note_allocated(-row.kv_alloc)
        row.kv_alloc = 0
        row.retired = True
        row.state = None
        row.task = None

    def _retire(self, row: Req, finish: str, now: float | None = None) -> None:
        now = now or time.monotonic()
        row.rec.update(
            decode_steps=row.steps,
            decode_ms=(now - row.t_prefill) * 1e3 if row.steps else 0.0,
            first_token_ms=((row.t_first or now) - row.t_start) * 1e3,
            completion_tokens=len(row.ids),
            finish_reason=finish,
        )
        self._note_expert_counters(row)
        self.stats_.record(row.rec, log=self.log_requests)
        extra = dict(row.rec)
        if row.logprobs_k:
            extra["logprobs"] = list(row.trace)
        res = GenResult(
            token_ids=list(row.ids),
            finish_reason=finish,
            prompt_tokens=len(row.prompt),
            cached_prompt_tokens=row.rec["cached_tokens"],
            extra=extra,
        )
        self._close(row)
        row.job.out.put(("done", res))

    def _cancel(self, row: Req) -> None:
        row.rec.update(
            finish_reason="length",
            completion_tokens=len(row.ids),
            decode_steps=row.steps,
            cancelled=True,
        )
        self._note_expert_counters(row)
        self.stats_.record(row.rec, log=self.log_requests)
        extra = dict(row.rec)
        if row.logprobs_k:
            extra["logprobs"] = list(row.trace)
        res = GenResult(
            token_ids=list(row.ids),
            finish_reason="length",
            prompt_tokens=len(row.prompt),
            cached_prompt_tokens=row.rec["cached_tokens"],
            extra=extra,
        )
        self._close(row)
        row.job.out.put(("done", res))

    def _fail(self, row: Req, exc: BaseException) -> None:
        row.rec.update(
            finish_reason="error",
            error=type(exc).__name__,
            completion_tokens=len(row.ids),
            decode_steps=row.steps,
        )
        self.stats_.record(row.rec, log=self.log_requests)
        self._close(row)
        row.job.out.put(("error", exc))

    def _abort_batch(self, exc: BaseException) -> int:
        """Error every row of the faulting step; the engine itself keeps serving. Returns how many
        rows it failed (0: the fault was not a step's, see `_worker_loop`).

        The whole step fails, not just the faulting row: by then earlier layers have appended KV
        and overwritten the recurrent state in place, which cannot be undone cheaply. Bad input is
        refused at `_accept`, and a failing prompt pass fails only its own row.
        """
        rows = list(self._batch or [])
        self._batch, self._y, self._telem, self._reads = None, None, None, []
        failed = 0
        for r in rows:
            if not r.retired:
                self._fail(r, exc)
                failed += 1
        return failed
