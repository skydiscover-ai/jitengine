"""Retirement for `Engine`, as a mixin: finishing, cancelling and failing rows, and handing their
slots back.
"""

from __future__ import annotations

import logging
import time

from jitengine_qwen3_6_35b_a3b_m4pro.engine.io_struct import GenResult
from jitengine_qwen3_6_35b_a3b_m4pro.scheduler.batch import Req

logger = logging.getLogger(__name__)


class RetirementMixin:
    """Finishing, cancelling and failing rows."""

    def _note_expert_counters(self, row: Req) -> None:
        """Expert-cache counter deltas over this request's lifetime (shared with other rows)."""
        if self.expert_pool is None or row.pool0 is None:
            return
        now = self.expert_pool.counters()
        row.rec.update({k: now[k] - row.pool0[k] for k in now})

    def _close(self, row: Req) -> None:
        if row in self._rows:
            self._rows.remove(row)
        if self._prefilling is row:
            self._prefilling = None
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
        res = GenResult(
            token_ids=list(row.ids),
            finish_reason=finish,
            prompt_tokens=len(row.prompt),
            cached_prompt_tokens=row.rec["cached_tokens"],
            extra=dict(row.rec),
        )
        self._close(row)
        row.job.out.put(("done", res))

    def _cancel(self, row: Req) -> None:
        row.rec.update(
            finish_reason="length", completion_tokens=len(row.ids), decode_steps=row.steps
        )
        self._note_expert_counters(row)
        self.stats_.record(row.rec, log=self.log_requests)
        res = GenResult(
            token_ids=list(row.ids),
            finish_reason="length",
            prompt_tokens=len(row.prompt),
            cached_prompt_tokens=row.rec["cached_tokens"],
            extra=dict(row.rec),
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

    def _fail_rows(self, rows: list, exc: BaseException) -> None:
        """Fail rows whose decode step could not be built, so they are not re-issued forever."""
        logger.error(f"[engine] decode step build failed for {len(rows)} row(s): {exc!r}")
        for r in rows:
            if not r.retired:
                self._fail(r, exc)

    def _abort_batch(self, exc: BaseException) -> None:
        """Error every row of the faulting step; the engine itself keeps serving.

        The whole step fails, not just one row: by the time a fault fires, earlier layers have
        already updated every row's KV and DeltaNet state in place, and that cannot be undone.
        """
        rows = list(self._batch or [])
        self._batch, self._y, self._telem = None, None, None
        for r in rows:
            if not r.retired:
                self._fail(r, exc)
