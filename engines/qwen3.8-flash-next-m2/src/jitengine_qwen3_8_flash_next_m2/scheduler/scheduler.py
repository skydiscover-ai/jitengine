"""The model thread for `Engine`, as a mixin: its loop, one scheduling round, taking submitted
jobs off the queue (`_accept` checks each one), the bounded window that admits an arrival group
as one wave, the elastic extension pool's size (single-row, decode-in-turn and multi-row), the MTP
head's load and release with it, and failing everything if the thread dies.
"""

from __future__ import annotations

import logging
import queue
import time

import mlx.core as mx

from jitengine_qwen3_8_flash_next_m2.budget import GIB
from jitengine_qwen3_8_flash_next_m2.engine.io_struct import GenResult
from jitengine_qwen3_8_flash_next_m2.scheduler.batch import Job, kv_bytes_for

logger = logging.getLogger(__name__)

CLEAR_CACHE_EVERY = 64  # scheduler rounds between allocator cache releases (never mid-step)


class SchedulingMixin:
    """The worker loop, and what each round does before a row is admitted."""

    def _worker_loop(self) -> None:
        """The model thread; it must never die quietly.

        A fault inside a step errors that step's rows and the loop continues. If recovery itself
        raises, the engine is marked dead and every waiter is failed rather than left blocked.
        """
        fault: BaseException | None = None
        try:
            while True:
                try:
                    self._schedule_once()
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
        try:
            logger.error(f"[engine] model thread is gone: {exc!r}; failing every pending request")
        except BaseException:  # noqa: BLE001
            pass
        for row in list(self._batch or []) + list(self._rows):
            if not row.retired:
                try:
                    self._fail(row, exc)
                except BaseException:  # noqa: BLE001
                    row.retired = True
                    row.job.out.put(("error", exc))
        self._batch, self._y, self._telem, self._reads = None, None, None, []
        for job in list(self._pending) + list(self._scores):
            job.out.put(("error", exc))
        self._pending, self._scores = [], []
        self._reap_queue(exc)

    def _reap_queue(self, exc: BaseException) -> None:
        """Error everything still on the submission queue (and anything that lands after)."""
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                return
            job.out.put(("error", exc))

    def _schedule_once(self) -> None:
        idle = self._batch is None and not self._rows and not self._pending and not self._scores
        self._drain(block=idle)
        self._collect_arrivals()
        self._run_scores()
        self._ext_policy()  # release the extension before admission allocates any row state
        self._admit()
        self._prompt_slice()
        self._pump()
        self._work += 1
        if self._batch is None and not self._rows and not self._pending:
            mx.clear_cache()
        elif self._work % CLEAR_CACHE_EVERY == 0:
            mx.clear_cache()

    def _ext_slots_allowed(self, meta: dict) -> int:
        """Extension slot count, capped by the memory a single live request leaves idle."""
        raw = str(self.options.get("pool_ext_slots", "max")).strip().lower()
        plan = self.plan
        rec = int(self.plan.record_bytes)
        rows = max(1, int(plan.max_slots))
        idle = (
            (plan.state_bytes + plan.kv_bytes) * (rows - 1) / rows
            + plan.scratch_bytes
            + max(0.0, plan.budget_bytes - plan.planned_bytes)
        )
        cap_budget = int(idle // rec)
        self.ext_idle_slots = cap_budget  # the turn-mode operating point is carved from this
        # "max" takes the whole idle budget, so memory reserved for other rows and scratch returns
        # to the single-row operating point.
        want = cap_budget if raw == "max" else int(raw or 0)
        if want <= 0:
            return 0
        cap_records = max(
            0, int(meta["num_layers"]) * int(meta["num_experts"]) - int(plan.pool_slots)
        )
        allowed = max(0, min(want, cap_budget, cap_records))
        if getattr(self, "mtp_bytes", 0):
            # the MTP head lives in the single-row extension's memory
            head_slots = -(-int(self.mtp_bytes) // rec)
            if allowed < head_slots + 64:
                self.mtp_on, self.mtp_bytes = False, 0
                logger.warning(
                    f"[engine] MTP off: the single-row extension ({allowed} slots) cannot hold the head"
                )
            else:
                allowed -= head_slots
                logger.info(
                    f"[engine] MTP draft head: {self.mtp_bytes / GIB:.3f} GiB in the single-row extension "
                    f"({head_slots} slots' worth)"
                )
        # The part several decoding rows may keep: scratch and unallocated remainder only (their KV
        # and state are live), and never more than `allowed`.
        multi_cap = int(self.options.get("pool_ext_multi", "700"))
        # `multi_cap` is empirical: larger values let a multi-row decode peak above the MLX memory
        # limit, where an allocation raises.
        self.ext_slots_multi = max(
            0,
            min(
                allowed,
                multi_cap,
                int((plan.scratch_bytes + max(0.0, plan.budget_bytes - plan.planned_bytes)) // rec),
            ),
        )
        logger.info(
            f"[engine] extension: {allowed} slots single-row, {self.ext_slots_multi} while several rows decode"
        )
        if allowed < want:
            logger.debug(
                f"[engine] pool_ext_slots {want} clamped to {allowed} (idle budget {idle / GIB:.2f} GiB "
                f"-> {cap_budget} slots; records left {cap_records})"
            )
        return allowed

    def _scratch_needed(self) -> bool:
        """Whether prefill or score work above `ext_scratch_tokens` needs the plan's scratch."""
        th = self.ext_scratch_tokens
        for r in self._rows:
            if r.phase == "prefill" and not r.task.done and (r.task.n - r.task.pos) > th:
                return True
        for job in self._pending:
            if self._uncovered_prefill(job) > th:
                return True
        for job in self._scores:
            if len(job.payload[0]) + len(job.payload[1]) > th:
                return True
        return False

    def _ext_policy(self) -> None:
        """Size the elastic extension pool for the current operating point.

        One live row and no scratch need: the whole extension. Several rows: `ext_slots_multi`.
        Any prefill or score that needs scratch: none. Release happens before a newly admitted
        row's prefill runs, so each operating point sees the memory the plan reserved for it.
        """
        pool = getattr(self, "expert_pool", None)
        if pool is None or not getattr(pool, "ext_slots", 0):
            return
        live = len(self._rows) + len(self._pending)
        scratch = self._scratch_needed()
        # One row with no scratch need takes the whole extension; several rows all decoding take the
        # scratch part only; any prefill or score that needs scratch takes none.
        if self.decode_turn_q > 0 and live > 1:
            # decode in turns: one row decodes at b == 1, and only the parked rows' state is live
            want = 0 if (scratch or self._turn_scratch_needed()) else self._turn_ext_want()
        else:
            want = 0 if scratch else (pool.ext_slots if live <= 1 else self.ext_slots_multi)
        cur = pool.ext_live if pool.ext_active else 0
        single = want > 0 and want == pool.ext_slots and (self.decode_turn_q <= 0 or live <= 1)
        if self.mtp is not None and not single:
            self._mtp_release()  # the head's memory is the single-row extension's
        if want < cur:
            self.model.drop_compiled()  # compiled builds hold the extension arrays as constants
            pool.release_ext()
            mx.clear_cache()
            cur = 0
        if want > cur and not pool.ext_active:
            pool.regrow_ext(want)
            self.model.drop_compiled()  # ...and would otherwise keep using the dropped ones
        if self.mtp_on and self.mtp is None and single and pool.ext_active:
            self._mtp_load()

    def _turn_scratch_needed(self) -> bool:
        """`_scratch_needed` checks each prompt against `ext_scratch_tokens`; with the turn-mode
        extension holding the scratch, several covered rows prefilling in one wave must be bounded
        by their sum, since the wave's activations scale with the tokens in flight."""
        th = self.ext_scratch_tokens
        tot = sum(
            r.task.n - r.task.pos
            for r in self._rows
            if r.phase == "prefill" and r.task is not None and not r.task.done
        )
        tot += sum(self._uncovered_prefill(job) for job in self._admissible_pending())
        return tot > th

    def _admissible_pending(self) -> list:
        """The pending jobs `_admit` would admit right now (strict FIFO, the SlotPool's rule). A job
        that must wait for a slot (the 9th request) holds no state and costs the extension nothing."""
        out = []
        free, kv_live, live = self.pool.free_slots, self.pool.kv_live, self.pool.live_rows
        for job in self._pending:
            if job.cancelled:
                continue
            prompt, max_tokens = job.payload
            kv = kv_bytes_for(
                self.cfg, self._kv_reservation(len(prompt), max_tokens), self.act_bytes
            )
            if free <= 0 or (live > 0 and kv_live + kv > self.pool.kv_budget_bytes):
                break
            out.append((job, kv))
            free, kv_live, live = free - 1, kv_live + kv, live + 1
        return [j for j, _ in out]

    def _turn_ext_want(self) -> int:
        """The turn-mode extension's size, in records.

        The single-row extension is the plan's idle budget (other rows' state/KV reserve + scratch +
        slack); here the parked rows' real state is live, so it comes out: want = idle_slots -
        ceil(parked_bytes / record_bytes), clamped to the extension's arrays. A row's bytes are its
        recurrent+PLE state (`plan.state_per_row`) plus its KV reservation (fixed at admission, an
        exact upper bound on what it can allocate). `parked` is the sum over live rows minus the
        smallest, an upper bound on whichever rows are parked that does not depend on who owns the
        turn, so a rotation never changes the operating point or releases or regrows the extension.
        """
        per = [int(self.delta_bytes) + int(r.kv_bytes) for r in self._rows if not r.retired]
        for job in self._admissible_pending():
            prompt, max_tokens = job.payload
            per.append(
                int(self.delta_bytes)
                + kv_bytes_for(
                    self.cfg, self._kv_reservation(len(prompt), max_tokens), self.act_bytes
                )
            )
        pool = self.expert_pool
        if len(per) <= 1:
            want = int(pool.ext_slots)
        else:
            rec = int(self.plan.record_bytes)
            parked = sum(per) - min(per)
            idle = int(getattr(self, "ext_idle_slots", pool.ext_slots))
            want = max(0, min(int(pool.ext_slots), idle - (-(-parked // rec))))
        self.turn_ext_slots = want
        return want

    def _mtp_load(self) -> None:
        from jitengine_qwen3_8_flash_next_m2.linear import apply_dense_policy as _adp
        from jitengine_qwen3_8_flash_next_m2.model import mtp as _mtp

        t0 = time.monotonic()
        head = _mtp.MTPHead(self.cfg, self.compute_dtype, self.model.lm_head, self.model.embed)
        _adp(head.linears(), self.compute_dtype, self.dense_kernel)
        self.mtp = head
        self.mtp_gen += 1
        self.mtp_loads += 1
        logger.debug(
            f"[engine] MTP draft head loaded ({head.nbytes / GIB:.3f} GiB, {time.monotonic() - t0:.2f} s)"
        )

    def _mtp_release(self) -> None:
        self.mtp = None
        for r in self._rows:
            r.mtp = None
        mx.clear_cache()

    def _drain(self, block: bool) -> None:
        """Move queued jobs into the scheduler's lists; block only when there is nothing to do."""
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
        else:
            job.out.put(("error", ValueError(f"unknown job {job.kind}")))

    def _accept(self, job: Job) -> None:
        """Validate a generate job and queue it for admission; a bad request fails only itself."""
        req = job.req
        try:
            prompt = [int(t) for t in req.prompt_ids]
            if not prompt:
                raise ValueError("empty prompt")
            for t in prompt:
                if t < 0 or t >= self.token_id_limit:
                    raise ValueError(f"token id {t} out of range")
            max_tokens = int(req.max_tokens)
            # One row may not reserve more KV than the whole budget, which is what makes
            # SlotPool.can_admit's lone-row exemption safe.
            if len(prompt) + max_tokens > self.max_kv_positions:
                raise ValueError(
                    f"prompt + max_tokens = {len(prompt) + max_tokens} positions exceeds the KV "
                    f"budget of {self.max_kv_positions} positions"
                )
        except Exception as exc:  # noqa: BLE001
            job.out.put(("error", exc))
            return
        if max_tokens <= 0:
            rec = {
                "id": getattr(req, "request_id", None) or "",
                "kind": "generate",
                "prompt_tokens": len(prompt),
                "cached_tokens": 0,
                "prefill_tokens": 0,
                "prefill_ms": 0.0,
                "first_token_ms": 0.0,
                "decode_steps": 0,
                "decode_ms": 0.0,
                "completion_tokens": 0,
                "finish_reason": "length",
                "queue_ms": (time.monotonic() - job.submitted) * 1e3,
            }
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

    def _collect_arrivals(self) -> None:
        """Hold a short, bounded window open so an arrival group is admitted together.

        A wave can only include rows that already hold a slot, so without a window the first
        arrival would start a long pass alone. The window opens only when the engine is otherwise
        idle (nothing decoding or holding a slot, no score waiting), and only if some pending job
        has at least `admit_min_tokens` of uncovered prefill (hot: `admit_quiet_s` /
        `admit_window_s`) or every pending job is fully covered (warm: `admit_warm_quiet_s`). It
        closes when `wave_width` jobs are pending, on the quiet or total timeout, or when a score
        job arrives; a warm window that collects uncovered work turns hot.
        """
        if (
            self.admit_window_s <= 0
            or self.wave_width <= 1
            or self._dead is not None
            or self._scores
            or self._rows
            or self._batch is not None
            or not self._pending
        ):
            return
        if len(self._pending) >= self.wave_width:
            return
        hot = self._window_is_hot()
        if not hot and not (
            self.admit_warm_quiet_s > 0
            and all(self._uncovered_prefill(job) <= 0 for job in self._pending)
        ):
            return
        t0 = time.monotonic()
        warm_open = not hot
        before = len(self._pending)
        while len(self._pending) < self.wave_width:
            left = self._window_time_left(hot, t0)
            if left <= 0:
                break
            try:
                job = self._queue.get(timeout=left)
            except queue.Empty:
                break
            self._take(job)
            if self._scores:
                break
            hot = self._recheck_hot(hot)
        self._note_window(warm_open, t0, before)

    def _window_time_left(self, hot: bool, t0: float) -> float:
        """Seconds the window may still wait for an arrival: the quiet timeout, capped by the total one."""
        quiet = self.admit_quiet_s if hot else self.admit_warm_quiet_s
        total = self.admit_window_s if hot else self.admit_warm_quiet_s
        left = min(t0 + total - time.monotonic(), quiet)
        return left

    def _recheck_hot(self, hot: bool) -> bool:
        """A warm window turns hot once its pending jobs include uncovered prefill; returns the new state."""
        if not hot:
            hot = self._window_is_hot()
        return hot

    def _note_window(self, warm_open: bool, t0: float, before: int) -> None:
        """Count the window, its length and how many jobs it collected."""
        self.admit_windows += 1
        self.admit_warm_windows += 1 if warm_open else 0
        self.admit_window_seconds += time.monotonic() - t0
        self.admit_window_collected += len(self._pending) - before

    def _window_is_hot(self) -> bool:
        """Does some pending job face enough prefill of its own to be worth the long window?"""
        return any(self._uncovered_prefill(job) >= self.admit_min_tokens for job in self._pending)

    def _uncovered_prefill(self, job) -> int:
        """Tokens of `job`'s head the store cannot supply (by `PrefillTask`'s resumability rule).

        Read-only: counts no hit and moves no recency.
        """
        head = job.head
        n = len(head)
        return n - self.store.covered_len(head, lambda k: k % self.wave_chunk == 0 or k == n)
