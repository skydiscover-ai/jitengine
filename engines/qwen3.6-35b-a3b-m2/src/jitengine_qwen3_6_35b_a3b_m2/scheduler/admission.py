"""Admission for `Engine`, as a mixin: which pending requests get a slot, and the KV positions
each row is charged.
"""

from __future__ import annotations

import time

from jitengine_qwen3_6_35b_a3b_m2.engine.io_struct import GenResult
from jitengine_qwen3_6_35b_a3b_m2.kvcache.prefix_cache import group_prefix_points
from jitengine_qwen3_6_35b_a3b_m2.kvcache.state import RequestState
from jitengine_qwen3_6_35b_a3b_m2.sampler import Sampler
from jitengine_qwen3_6_35b_a3b_m2.scheduler.batch import Job, Req, kv_bytes_for


class AdmissionMixin:
    """Which pending requests get a slot, and what each row's KV is charged."""

    def _admit(self) -> None:
        if not self._pending:
            return
        self._refresh_group_points()
        n = 0
        for job in self._pending:
            if job.cancelled:
                job.out.put(
                    (
                        "done",
                        GenResult(
                            token_ids=[], finish_reason="length", prompt_tokens=len(job.payload[0])
                        ),
                    )
                )
                n += 1
                continue
            prompt, max_tokens = job.payload
            # charge the whole generation, not the lazily allocated first slice
            kv = kv_bytes_for(self.cfg, self._kv_reservation(len(prompt), max_tokens))
            if not self.pool.can_admit(kv):
                # strict FIFO: admitting past it could starve a large prompt behind small ones
                self.pool.deferred += 1
                break
            n += 1
            try:
                row = self._open_row(job, prompt, max_tokens, kv)
            except Exception as exc:  # noqa: BLE001
                job.out.put(("error", exc))
                continue
            self._rows.append(row)
        if n:
            self._pending = self._pending[n:]
            self._pending_dirty = True

    def _refresh_group_points(self) -> None:
        """When the pending list changed, recompute each pending job's shared-prefix point and clear the flag."""
        if self._pending_dirty:
            if len(self._pending) > 1:
                pts = group_prefix_points(
                    [j.head for j in self._pending], self.prefill_chunk, self.store.min_divergence
                )
            else:
                pts = [0] * len(self._pending)
            self._group_points = {id(j): p for j, p in zip(self._pending, pts)}
            self._pending_dirty = False

    def _pending_uses(self, key: tuple) -> bool:
        """Whether a pending or queued request would restore the stored prefix `key`."""
        n = len(key)
        for job in self._pending:
            if job.head is not None and len(job.head) >= n and tuple(job.head[:n]) == key:
                return True
        with self._queue.mutex:  # submitting threads append to the deque while this reads it
            queued = list(self._queue.queue)
        for job in queued:
            ids = getattr(getattr(job, "req", None), "prompt_ids", None)
            if ids is not None and len(ids) > n and tuple(int(t) for t in ids[:n]) == key:
                return True
        return False

    def _kv_reservation(self, prompt_len: int, max_new: int) -> int:
        """Positions admission charges: the whole generation, rounded up to KV_STEP like KVSlot."""
        from jitengine_qwen3_6_35b_a3b_m2.kvcache.kv_cache import KV_STEP

        raw = prompt_len + max(0, int(max_new))
        return max(KV_STEP, -(-raw // KV_STEP) * KV_STEP)

    def _note_kv_alloc(self, row: Req) -> None:
        """Re-read what this row's KV buffers have grown to and keep the pool's total in step."""
        now = int(row.state.kv_allocated_bytes)
        self.pool.note_allocated(now - row.kv_alloc)
        row.kv_alloc = now

    def _ensure_room(self, row: Req, n: int = 1) -> None:
        """Pre-size a row for `n` more positions; account for the growth if it happened."""
        if row.state.ensure_room(n):
            self._note_kv_alloc(row)

    def _open_row(self, job: Job, prompt: list[int], max_tokens: int, kv: int) -> Req:
        slot = self.pool.acquire(kv)
        try:
            state = RequestState(self.cfg, len(prompt), max_tokens, reserve=self.kv_reserve)
            stop_at_eos = bool(getattr(job.req, "stop_at_eos", True))
            sampler = Sampler(
                job.req.temperature,
                job.req.top_p,
                job.req.top_k,
                job.req.seed,
                self.vocab,
                radix=self.radix_sort,
            )
            row = Req(
                job,
                job.req,
                prompt,
                max_tokens,
                set(self.eos_ids) if stop_at_eos else set(),
                sampler,
                state,
                slot,
                kv,
                self._group_points.get(id(job), 0),
                time.monotonic(),
            )
            if self.expert_pool is not None:
                row.pool0 = self.expert_pool.counters()
            self._note_kv_alloc(row)
        except BaseException:
            self.pool.release(slot, kv)
            raise
        return row
