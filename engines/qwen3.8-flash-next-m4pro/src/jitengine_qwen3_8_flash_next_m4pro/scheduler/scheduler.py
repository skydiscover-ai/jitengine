"""The model thread for `Engine`, as a mixin: its loop, one scheduling round, taking submitted
jobs off the queue (`_accept` checks each one), the bounded window that admits an arrival group
as one wave, the elastic extension pool's size, the MTP draft head's load and release with it (and
its subset readout, `model.draft_vocab`), and failing everything if the thread dies.
"""

from __future__ import annotations

import logging
import queue
import time

import mlx.core as mx

from jitengine_qwen3_8_flash_next_m4pro.budget import GIB
from jitengine_qwen3_8_flash_next_m4pro.engine.io_struct import GenResult
from jitengine_qwen3_8_flash_next_m4pro.scheduler.batch import Job

logger = logging.getLogger(__name__)

CLEAR_CACHE_EVERY = 64  # scheduler rounds between allocator cache releases (never mid-step)
# Consecutive scheduler rounds that may fault without failing any request before the engine stops:
# a fault that recurs every round and that no row's failure clears would otherwise retry forever,
# leaving pending, lone and score requests neither served nor failed while /health said ok.
STUCK_FAULT_ROUNDS = 2


class SchedulingMixin:
    """The worker loop, and what each round does before a row is admitted."""

    def _worker_loop(self) -> None:
        """The model thread; it must never die quietly.

        A fault inside a step errors that step's rows and the loop continues. If recovery itself
        raises, or the fault recurs for `STUCK_FAULT_ROUNDS` rounds without failing any row (it is
        not a step's, so nothing the loop can drop clears it), the engine is marked dead and every
        waiter is failed rather than left blocked; /health then reports the fault.
        """
        fault: BaseException | None = None
        stuck = 0  # consecutive faulting rounds that failed no row
        try:
            while True:
                try:
                    self._schedule_once()
                    stuck = 0
                except BaseException as exc:  # noqa: BLE001  a step fault is recoverable
                    try:
                        logger.error(f"[engine] scheduler fault: {exc!r}")
                        self.scheduler_faults += 1
                        failed = self._abort_batch(exc)
                    except BaseException as inner:  # recovery failed: stop honestly
                        fault = inner
                        raise
                    stuck = 0 if failed else stuck + 1
                    if stuck >= STUCK_FAULT_ROUNDS:
                        logger.error(
                            f"[engine] the same scheduler fault recurred in {stuck} rounds that no "
                            f"request's failure cleared; the engine stops serving"
                        )
                        fault = exc
                        return  # the `finally` fails every waiter with it
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
        self._settle_open_fetches()
        self._work += 1
        if self._batch is None and not self._rows and not self._pending:
            mx.clear_cache()
        elif self._work % CLEAR_CACHE_EVERY == 0:
            mx.clear_cache()

    def _settle_open_fetches(self) -> None:
        """No expert fetch stays open past a round: every forward has returned, so a split fetch
        still open was left by a step that raised between its begin and finish. Settling it here
        returns its landing slots to the region (the decode loops settle their own
        first, so this is the backstop)."""
        pool = getattr(self, "expert_pool", None)
        if pool is not None and getattr(pool, "open_fetches", lambda: 0)():
            n = pool.abandon_open()
            logger.warning(
                f"[engine] settled {n} landing slot(s) of an expert fetch its step left open"
            )

    def _ext_slots_allowed(self, meta: dict) -> int:
        """Extension slot count, capped by the memory a single live request leaves idle."""
        raw = str(self.options.get("pool_ext_slots", "max")).strip().lower()
        plan = self.plan
        rec = int(self.plan.slot_bytes)  # what one extension slot holds (pack16: the compact form)
        rows = max(1, int(plan.max_slots))
        idle = (
            (plan.state_bytes + plan.kv_bytes) * (rows - 1) / rows
            + plan.scratch_bytes
            + max(0.0, plan.budget_bytes - plan.planned_bytes)
        )
        cap_budget = int(idle // rec)
        # "max" takes the whole idle budget, so memory reserved for other rows and scratch returns
        # to the single-row operating point.
        want = cap_budget if raw == "max" else int(raw or 0)
        if want <= 0:
            return 0
        cap_records = max(
            0, int(meta["num_layers"]) * int(meta["num_experts"]) - int(plan.pool_slots)
        )
        allowed = max(0, min(want, cap_budget, cap_records))
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
        # the MTP head comes out of the single-row extension only; the multi-row part is sized above
        # (and stays inside the arrays, which are the single-row size)
        allowed = self._charge_mtp_head(allowed, rec)
        self.ext_slots_multi = min(self.ext_slots_multi, allowed)
        logger.info(
            f"[engine] extension: {allowed} slots single-row, {self.ext_slots_multi} while several rows decode"
        )
        if allowed < want:
            logger.debug(
                f"[engine] pool_ext_slots {want} clamped to {allowed} (idle budget {idle / GIB:.2f} GiB "
                f"-> {cap_budget} slots; records left {cap_records})"
            )
        return allowed

    def _charge_mtp_head(self, allowed: int, rec: int) -> int:
        """The single-row extension's slots once the MTP draft head is charged inside it.

        The head is on only when the single-row operating point holds at least `MTP_MIN_SLOTS`
        expert slots (option `mtp_min_slots`; main pool plus this extension, before the head's
        share), since below that the verify pass costs more misses than it saves; its bytes
        (`mtp_bytes`, the file plus the verify pass's transients) then come out of the extension,
        so the extension and the head together never exceed what the extension alone was allowed.
        With `draft_vocab` "subset" the draft's readout (`draft_vocab_plan`: lm_head rows copied at
        each head load, priced from the checkpoint headers) is held with the head and charged with it.
        """
        from jitengine_qwen3_8_flash_next_m4pro.model import draft_vocab as _dv
        from jitengine_qwen3_8_flash_next_m4pro.model.mtp import MTP_MIN_SLOTS

        _dv.mode_of(
            getattr(self, "options", None)
        )  # a bad `draft_vocab` is refused with or without a head
        self.draft_vocab_plan = None
        if not getattr(self, "mtp_bytes", 0):
            return allowed
        need = MTP_MIN_SLOTS
        if getattr(self, "options", {}).get("mtp_min_slots") is not None:
            need = max(0, int(self.options["mtp_min_slots"]))  # the `mtp_min_slots` option
        total = int(self.plan.pool_slots) + int(allowed)
        dv = _dv.plan_for(self)
        dv_bytes = dv.bytes if dv is not None else 0
        head_slots = -(-(int(self.mtp_bytes) + dv_bytes) // int(rec))
        if total < need or allowed < head_slots + 64:
            logger.info(
                f"[engine] MTP off: {total} single-row expert slots (need {need}) and an "
                f"extension of {allowed} (the head needs {head_slots} + 64)"
            )
            self.mtp_on, self.mtp_bytes = False, 0
            return allowed
        self.draft_vocab_plan = dv
        logger.info(
            f"[engine] MTP draft head: {self.mtp_bytes / GIB:.3f} GiB in the single-row extension "
            f"({head_slots} of its {allowed} slots; {total} single-row slots before the head)"
            + (
                f"; its subset readout {dv.rows} rows ({dv.ids.size} + {dv.rows - dv.ids.size} "
                f"prompt rows), {dv_bytes / 1e6:.1f} MB"
                if dv is not None
                else "; full readout"
            )
        )
        return allowed - head_slots

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
        want = 0 if scratch else (pool.ext_slots if live <= 1 else self.ext_slots_multi)
        cur = pool.ext_live if pool.ext_active else 0
        single = live <= 1 and want > 0  # the single-row operating point, which holds the head
        if self.mtp is not None and not single:
            self._mtp_release()  # the head's memory is the single-row extension's
        if want < cur:
            self.model.drop_compiled()  # compiled builds hold the extension arrays as constants
            pool.release_ext()
            mx.clear_cache()
            cur = 0
        if want > cur and not pool.ext_active and "extension" not in self._off():
            try:
                pool.regrow_ext(want)
            except Exception as exc:  # noqa: BLE001  an allocation failure: serve without it
                self._feature_off("extension", exc)
                self._mtp_feature_off("the extension that holds it is off")
            self.model.drop_compiled()  # ...and would otherwise keep using the dropped ones
        if self.mtp_on and self.mtp is None and single and pool.ext_active:
            try:
                self._mtp_load()
            except Exception as exc:  # noqa: BLE001  a head file gone, truncated or unreadable
                self._feature_off("mtp", exc)
                self._mtp_feature_off(None)
        self._annex_policy(single)

    def _feature_off(self, name: str, exc: BaseException) -> None:
        """Turn a memory-lending feature off for the engine's life after it failed: each is a pure
        speed-up (the same tokens and logits without it), so the engine serves on without it rather
        than retrying a fault every round. Listed on /stats and /health as `features_off`."""
        self._off(write=True)[name] = f"{type(exc).__name__}: {exc}"
        logger.error(
            f"[engine] {name} OFF for this process after a failure ({exc!r}); serving without it"
        )

    def _off(self, write: bool = False) -> dict:
        """`features_off`: {feature: why} for each one a failure turned off."""
        off = getattr(self, "features_off", None)
        if off is None:
            off = {}
            if write:
                self.features_off = off
        return off

    def _mtp_feature_off(self, why: str | None) -> None:
        """MTP off for the engine's life: no head, no row's draft state, its memory back."""
        if why is not None and self.mtp_on:
            self._off(write=True).setdefault("mtp", why)
        self.mtp_on = False
        self._mtp_release()

    def _annex_policy(self, single: bool) -> None:
        """Lend the prefix store's memory to the pool's annex while exactly one row decodes and
        nothing else waits; take it back (annex released, store read back) before anything else.

        One-row decode never reads the store, so its entries go to the spill file (written behind
        the decode: the annex waits for the writes, never the decode) and the bytes they held back
        the annex's slots, so the decode's footprint is the resident store's, slot for byte. Any
        arrival, prefill, score or second row first releases the annex (its own records only: the
        main pool and the extension keep theirs, so a short follow-up turn costs at most the
        annex's refill) and then restores the store, before `_admit` looks it up.
        """
        pool = getattr(self, "expert_pool", None)
        if pool is None or not getattr(pool, "ann_slots", 0) or self.store.spiller is None:
            return
        if "annex" in self._off():  # the store stays resident (restored when it failed)
            return
        lend = (
            single
            and pool.ext_active
            and not self._pending
            and not self._scores
            and len(self._rows) == 1
            and self._rows[0].phase == "decode"
            and not self._rows[0].retired
        )
        if lend:
            if not self.store.spilled:
                freed = self.store.spill()
                if freed is None:
                    return  # writes still behind the decode: the annex waits for a later step
                mx.clear_cache()  # the dropped entries' buffers back to the system first
                sp = self.store.spiller.stats()
                logger.info(
                    f"[store] spilled: {freed / 1e6:.1f} MB dropped, {self.store.spill_kept / 1e6:.1f} MB "
                    f"kept; {sp['spill_entries_written']} entries written in {sp['spill_write_seconds']} s "
                    f"behind the decode, file {sp['spill_file_bytes'] / 1e6:.1f} MB"
                )
            if not pool.ann_active:
                # The annex borrows what the store's entries hold (`bytes`, on file now), never the
                # cap they leave unfilled: one-row decode's transients and MLX's buffer cache live in
                # that slack. An annex at the cap put MLX's active + cache at its memory and wired
                # limits at 24 GB, which one-row decode paid as ~14 ms/token of host time.
                # An entry whose write failed keeps its arrays (`spill_kept`) and
                # lends nothing.
                free = int(self.store.bytes) - int(self.store.spill_kept)
                try:
                    n = pool.regrow_ann(min(int(pool.ann_slots), free // int(pool.slot_bytes)))
                except Exception as exc:  # noqa: BLE001  an allocation failure: the store comes back
                    self._feature_off("annex", exc)
                    self.store.restore()
                    return
                if n:
                    self.model.drop_compiled()  # compiled builds hold the pool's arrays as constants
                    logger.info(
                        f"[store] annex lent {n} slots x {int(pool.slot_bytes)} B of the {free} B the "
                        f"entries held (cap {int(self.store.cap_bytes)} B, {int(pool.ann_slots)} slots)"
                    )
            return
        if pool.ann_active:
            self.model.drop_compiled()
            pool.release_ann()
            mx.clear_cache()
        self.store.restore()

    def _mtp_load(self) -> None:
        """Load the MTP draft head into the single-row extension's memory (see `_charge_mtp_head`)."""
        from jitengine_qwen3_8_flash_next_m4pro.linear import apply_dense_policy
        from jitengine_qwen3_8_flash_next_m4pro.model import draft_vocab as _dv
        from jitengine_qwen3_8_flash_next_m4pro.model import mtp as _mtp

        t0 = time.monotonic()
        head = _mtp.MTPHead(self.cfg, self.compute_dtype, self.model.lm_head, self.model.embed)
        apply_dense_policy(head.linears(), self.compute_dtype, self.dense_kernel)
        dv = getattr(self, "draft_vocab_plan", None)
        if dv is not None:
            lm = self.model.lm_head
            held = _dv.buffer_bytes(dv.ids.size, lm.nbytes // lm.out_dims, lm.out_dims)
            if (
                held > dv.bytes
            ):  # the rows as loaded are wider than the headers priced: never over the charge
                logger.error(
                    f"[engine] MTP draft readout: {held} B as loaded > {dv.bytes} B charged; "
                    f"reading out in full"
                )
            else:
                head.readout = _dv.SubsetReadout(lm, dv.ids)
        ro = head.readout
        self.stats_.note_draft_vocab(
            "full" if ro is None else "subset", 0 if ro is None else ro.rows
        )
        self.mtp = head
        self.mtp_gen += 1
        self.mtp_loads += 1
        logger.info(
            f"[engine] MTP draft head loaded ({head.nbytes / GIB:.3f} GiB, "
            f"{time.monotonic() - t0:.2f} s; readout "
            + ("full" if ro is None else f"subset of {ro.rows} rows, {ro.nbytes / 1e6:.1f} MB")
            + ")"
        )

    def _mtp_release(self) -> None:
        """Drop the head and every row's draft state, and return their memory to the system."""
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
