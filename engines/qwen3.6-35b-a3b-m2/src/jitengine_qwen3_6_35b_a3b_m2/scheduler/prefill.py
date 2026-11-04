"""Prompt processing as a resumable task, one absolute-grid chunk at a time.

Warm output is bit-identical to cold because the chunk partition depends only on positions: chunks
end at absolute multiples of `chunk`, and entries restore only at grid-aligned lengths or the whole
head. Every forward carries one row: `mx.quantized_matmul` switches kernel path with the row count,
so a batched pass would not be bit-identical to a solo one.
"""

from __future__ import annotations

import time

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m2.scheduler.batch import Req


class PrefillTask:
    """One row's prompt pass. `step()` computes one chunk; `done` reports completion."""

    __slots__ = (
        "model",
        "store",
        "chunk",
        "head",
        "state",
        "use_store",
        "n",
        "hit_len",
        "admit_at",
        "tail_at",
        "end_covered",
        "end_shared",
        "pos",
        "layer_major",
        "pending_uses",
    )

    def __init__(
        self,
        model,
        store,
        chunk: int,
        head: list[int],
        state,
        use_store: bool,
        group_point: int = 0,
        pending_uses=None,
    ):
        self.pending_uses = pending_uses or (lambda key: False)
        self.model = model
        self.store = store
        self.chunk = int(chunk)
        self.head = head
        self.state = state
        self.use_store = bool(use_store)
        self.n = len(head)
        self.hit_len = 0
        self.admit_at: set[int] = set()
        self.tail_at = 0
        self.end_covered = False
        self.end_shared = False  # the head's end is a planned shared boundary: store it whole
        if self.use_store and self.n > 0:
            self._restore_stored_prefix(head, store, state)
            if self.hit_len < self.n:
                self._plan_divergence_points(head, store)
                self._plan_group_point(group_point, store)
                self._plan_tail_point(store)
        self.pos = self.hit_len
        self.layer_major = False

    def _restore_stored_prefix(self, head: list[int], store, state) -> None:
        """Look up the longest stored prefix of `head` on the chunk grid and restore it into `state`."""
        entry = store.lookup(head, resumable=lambda k: k % self.chunk == 0 or k == self.n)
        if entry is not None:
            state.restore(entry)
            self.hit_len = entry.tokens

    def _plan_divergence_points(self, head: list[int], store) -> None:
        """Plan a store admission at each grid-floored divergence point past the restored prefix."""
        # floor divergence points to the grid; rounding up would overshoot the shared prefix
        for c in store.divergence_points(head, self.hit_len):
            a = (c // self.chunk) * self.chunk
            if a - self.hit_len >= store.min_divergence and self.hit_len < a:
                if a < self.n:
                    self.admit_at.add(a)
                elif a == self.n:
                    self.end_shared = True

    def _plan_group_point(self, group_point: int, store) -> None:
        """Plan a store admission at the boundary shared with this row's arrival group."""
        # grouped admission: the grid-floored boundary shared with this row's arrival group
        if self.hit_len < group_point <= self.n and (
            group_point - self.hit_len >= store.min_divergence
        ):
            if group_point < self.n:
                self.admit_at.add(group_point)
            else:
                self.end_shared = True

    def _plan_tail_point(self, store) -> None:
        """Plan the head's own snapshot at its last grid point, the one a longer prompt can restore.

        A lookup resumes only at grid-aligned lengths (or its own whole head), so a snapshot at the
        off-grid end of this head would never match the next turn of a conversation that appends to
        it. The last grid point inside the head is where that turn restores. It stands in for the
        end-of-head snapshot: at most one grid chunk of this head is left to recompute on a repeat.
        """
        g = (self.n // self.chunk) * self.chunk
        if not 0 < g < self.n or g < self.hit_len:
            return
        if g == self.hit_len or g in self.admit_at or tuple(self.head[:g]) in store:
            self.end_covered = True  # a grid snapshot at g exists or is planned: it stands in
        elif g - self.hit_len >= store.min_divergence:
            self.admit_at.add(g)
            self.tail_at = g
            self.end_covered = True

    @property
    def done(self) -> bool:
        return self.pos >= self.n

    @property
    def computed(self) -> int:
        return self.pos - self.hit_len

    def step(self) -> bool:
        """Compute the next chunk (one single-row forward). Returns True when the head is done."""
        if self.done:
            self._finish()
            return True
        end = min(((self.pos // self.chunk) + 1) * self.chunk, self.n)
        self.model.forward(
            mx.array(self.head[self.pos : end], dtype=mx.int32)[None], self.state, logits="none"
        )
        mx.eval(*self.state.arrays())
        self.pos = end
        if self.pos in self.admit_at:
            self._admit(self.pos, shared=self.pos != self.tail_at)
        if self.done:
            self._finish()
            return True
        return False

    def run(self, layer_major: bool = False) -> None:
        """Compute the whole head (used by `score`, which is never interleaved)."""
        if layer_major:
            self.run_layer_major()
            return
        while not self.step():
            pass

    def run_layer_major(self) -> bool:
        """Rest of the head in one layer-major pass, with the same calls and snapshots as `step()`.

        Not interleavable with decode steps; the engine uses it only when no row is decoding.
        """
        from jitengine_qwen3_6_35b_a3b_m2.kvcache.prefix_cache import Entry

        if self.done:
            self._finish()
            return True
        bounds, p = [], self.pos
        while p < self.n:
            e = min(((p // self.chunk) + 1) * self.chunk, self.n)
            bounds.append((p, e))
            p = e
        ends = {e for _, e in bounds}
        points = sorted(
            a
            for a in self.admit_at
            if a in ends and a < self.n and tuple(self.head[:a]) not in self.store
        )
        parts = self.model.forward_layer_major(self.head, bounds, self.state, capture=points)
        self.pos = self.n
        for a in points:
            entry = Entry(tuple(self.head[:a]), parts[a])
            mx.eval(*entry.arrays())
            self.store.admit(entry, shared=a != self.tail_at, pending_uses=self.pending_uses)
        del parts
        self._finish()
        return True

    def _layer_major_job(self):
        """(bounds, capture points) of the rest of the head, as run_layer_major computes them."""
        bounds, p = [], self.pos
        while p < self.n:
            e = min(((p // self.chunk) + 1) * self.chunk, self.n)
            bounds.append((p, e))
            p = e
        ends = {e for _, e in bounds}
        points = sorted(
            a
            for a in self.admit_at
            if a in ends and a < self.n and tuple(self.head[:a]) not in self.store
        )
        return bounds, points

    def _layer_major_done(self, points, parts) -> None:
        from jitengine_qwen3_6_35b_a3b_m2.kvcache.prefix_cache import Entry

        self.pos = self.n
        for a in points:
            entry = Entry(tuple(self.head[:a]), parts[a])
            mx.eval(*entry.arrays())
            self.store.admit(entry, shared=a != self.tail_at, pending_uses=self.pending_uses)
        self._finish()

    @staticmethod
    def run_group_layer_major(tasks: list) -> None:
        """run_layer_major for several heads, with one merged routed-expert gather per layer."""
        live = []
        for t in tasks:
            if t.done:
                t._finish()
            else:
                live.append(t)
        if not live:
            return
        jobs = [t._layer_major_job() for t in live]
        parts = live[0].model.forward_layer_major_multi(
            [(t.head, b, t.state, pts) for t, (b, pts) in zip(live, jobs)]
        )
        for t, (_, pts), pr in zip(live, jobs, parts):
            t._layer_major_done(pts, pr)

    def _admit(self, upto: int, shared: bool = True) -> None:
        key = tuple(self.head[:upto])
        if key not in self.store:
            self.store.admit(
                self.state.snapshot(key), shared=shared, pending_uses=self.pending_uses
            )

    def _finish(self) -> None:
        # the end-of-head snapshot is the only one not known to be shared with another request
        if self.use_store and self.n > self.hit_len and not self.end_covered:
            self._admit(self.n, shared=self.end_shared)


class PromptProcessingMixin:
    """Prefill between decode steps, then the start of decode."""

    def _prompt_slice(self) -> None:
        """Prompt work: at most one chunk while rows decode, else every pending prompt."""
        decoding = self._is_decoding()
        if self.group_prefill and self._prompt_group(decoding):
            return
        chunks = 0
        while True:
            row = self._prefilling
            if row is None:
                row = self._next_prefill_row()
                if row is None:
                    return
                if row.cancelled:
                    self._cancel(row)
                    continue
                try:
                    row.task = PrefillTask(
                        self.model,
                        self.store,
                        self.prefill_chunk,
                        row.job.head,
                        row.state,
                        use_store=True,
                        group_point=row.group_point,
                        pending_uses=self._pending_uses,
                    )
                except Exception as exc:  # noqa: BLE001
                    self._fail(row, exc)
                    continue
                self._prefilling = row
            if row.cancelled:
                self._cancel(row)
                self._prefilling = None
                continue
            try:
                finished = self._advance_prefill(row, decoding)
            except Exception as exc:  # noqa: BLE001
                self._prefilling = None
                self._fail(row, exc)
                continue
            chunks += 1
            self._finish_prefill_step(row, finished)
            if decoding and chunks >= 1:
                return

    def _is_decoding(self) -> bool:
        """Whether a batch or any row is in the decode phase."""
        decoding = self._batch is not None or any(r.phase == "decode" for r in self._rows)
        return decoding

    def _next_prefill_row(self) -> Req | None:
        """The first row waiting for prefill that has no task yet, or None."""
        row = next((r for r in self._rows if r.phase == "prefill" and r.task is None), None)
        return row

    def _advance_prefill(self, row: Req, decoding: bool) -> bool:
        """Run one layer-major pass or one chunk of `row`'s prefill; True when its head is done."""
        rem = row.task.n - row.task.pos
        if self.layer_major_prefill and (not decoding or rem <= self.merge_decoding_max):
            # one layer-major pass over the head: bit-identical, fewer expert reads
            finished = row.task.run_layer_major()
        else:
            finished = row.task.step()
        return finished

    def _finish_prefill_step(self, row: Req, finished: bool) -> None:
        """When the head is done, free the prefill slot and start decoding the row."""
        if finished:
            self._prefilling = None
            self._start_decode(row)

    def _prompt_group(self, decoding: bool) -> bool:
        """Prefill pending prompts as one layer-major group pass, or return False if it cannot.

        Tasks are made only once the group is taken: a PrefillTask restores its store hit when made,
        so a follower made before its leader stored the shared prefix would miss it.
        """
        if self._prefilling is not None:
            return False
        cand = [
            r for r in self._rows if r.phase == "prefill" and not r.cancelled and r.task is None
        ]
        if len(cand) < 2:
            return False
        chunk = self.prefill_chunk
        rem = []
        for r in cand:
            n = len(r.job.head)
            hit = self.store.peek_len(r.job.head, resumable=lambda k, n=n: k % chunk == 0 or k == n)
            if hit < r.group_point:
                return False  # a shared boundary not stored yet: leader first, followers hit it
            rem.append(n - hit)
        if sum(rem) > self.group_prefill_max or (decoding and max(rem) > self.merge_decoding_max):
            return False
        for r in cand:
            try:
                r.task = PrefillTask(
                    self.model,
                    self.store,
                    chunk,
                    r.job.head,
                    r.state,
                    use_store=True,
                    group_point=r.group_point,
                    pending_uses=self._pending_uses,
                )
            except Exception as exc:  # noqa: BLE001
                self._fail(r, exc)
        cand = [r for r in cand if r.task is not None and not r.retired]
        if not cand:
            return True
        try:
            PrefillTask.run_group_layer_major([r.task for r in cand])
        except Exception as exc:  # noqa: BLE001
            for r in cand:
                self._fail(r, exc)
            return True
        for r in cand:
            self._start_decode(r)
        return True

    def _start_decode(self, row: Req) -> None:
        row.t_prefill = time.monotonic()
        row.rec.update(
            cached_tokens=row.task.hit_len,
            prefill_tokens=row.task.computed,
            prefill_ms=(row.t_prefill - row.t_start) * 1e3,
        )
        self._ensure_room(row)  # and a restored prefix may have grown the buffers too
        self._note_kv_alloc(row)
        row.phase = "decode"
        row.next_tok = row.prompt[-1]
