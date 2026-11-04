"""Prompt processing as a resumable task, one absolute-grid chunk at a time.

Warm output is bit-identical to cold because a warm run issues exactly the forward calls a cold run
issues: chunks end at absolute multiples of `chunk` (or the end of the head), entries are restored
only at a grid multiple or the whole head, and each row is computed as its own one-row tensor, since
stock `mx.quantized_matmul` is not bit-identical between a batched row and the same row alone.
Nothing computed by a multi-row step may enter the prefix store.
"""

from __future__ import annotations

import time

import mlx.core as mx

from jitengine_qwen3_8_flash_next_m4pro.scheduler.batch import Req
from jitengine_qwen3_8_flash_next_m4pro.scheduler.faults import RowFaultError, StepGuard


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
        "pos",
        "snapshots",
        "meter",
        "boundary",
        "grid_point",
        "tail_at",
        "end_covered",
        "end_shared",
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
        meter=None,
        snap_grid: bool = False,
    ):
        self.model = model
        self.store = store
        self.meter = meter
        self.chunk = int(chunk)
        self.head = head
        self.state = state
        self.use_store = bool(use_store)
        self.n = len(head)
        self.hit_len = 0
        self.snapshots = 0
        self.grid_point = 0
        self.tail_at = 0
        self.end_covered = False
        self.end_shared = False  # the head's end is a planned shared boundary: store it whole
        self.admit_at: set[int] = set()
        if self.use_store and self.n > 0:
            entry = store.lookup(head, resumable=lambda k: k % self.chunk == 0 or k == self.n)
            if entry is not None:
                state.restore(entry)
                self.hit_len = entry.tokens
            if self.hit_len < self.n:
                self._admit_divergence_points(store, head)
                self._admit_group_point(store, group_point)
                self._admit_grid_point(store, snap_grid)
                self._admit_tail_point(store)
        self.pos = self.hit_len
        # chunks ending at or below this are the shared prefix; above it, this request's suffix
        self.boundary = max(self.hit_len, int(group_point))

    def _admit_divergence_points(self, store, head: list[int]) -> None:
        """Snapshot at the grid boundary at or below each divergence point the store knows for `head`."""
        # A divergence point c is admitted at the grid boundary at or below it (never above:
        # rounding up overshoots the shared prefix, so a follower could not use the entry).
        for c in store.divergence_points(head, self.hit_len):
            a = (c // self.chunk) * self.chunk
            if a - self.hit_len >= store.min_divergence and self.hit_len < a:
                if a < self.n:
                    self.admit_at.add(a)
                elif a == self.n:
                    self.end_shared = True

    def _admit_group_point(self, store, group_point: int) -> None:
        """Snapshot at the arrival group's shared boundary, when it is past the restored prefix."""
        # grouped admission: the group's shared boundary, already floored to the grid
        if self.hit_len < group_point <= self.n and (
            group_point - self.hit_len >= store.min_divergence
        ):
            if group_point < self.n:
                self.admit_at.add(group_point)
            else:
                self.end_shared = True

    def _admit_grid_point(self, store, snap_grid: bool) -> None:
        """A generate row also snapshots at the first interior grid boundary above what it restored."""
        # A generate row also snapshots at the first interior grid boundary above what it
        # restored, so a later arrival sharing its system prompt can resume there; one
        # point per row keeps the store's cap from thrashing.
        if snap_grid:
            self.grid_point = ((self.hit_len // self.chunk) + 1) * self.chunk
            if (
                self.hit_len < self.grid_point < self.n
                and self.grid_point - self.hit_len >= store.min_divergence
            ):
                self.admit_at.add(self.grid_point)

    def _admit_tail_point(self, store) -> None:
        """Snapshot at the last grid point inside the head, in place of the off-grid end-of-head entry.

        A lookup resumes only at a grid multiple or at its own whole head, so an entry at this
        head's off-grid end never matches the next turn of a conversation that appends to it. The
        last grid point inside the head does, and costs at most one chunk to recompute on a repeat.
        Like the end-of-head entry it replaces, it is unpinned.
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

    # ------------------------------------------------------------------ the chunk grid
    def chunk_end(self) -> int:
        """End of the next chunk: the next absolute multiple of `chunk`, or the end of the head."""
        return min(((self.pos // self.chunk) + 1) * self.chunk, self.n)

    def chunk_phase(self) -> str:
        """Prefix or suffix prefill for the next chunk (a chunk never straddles the boundary)."""
        return "prefix_prefill" if self.chunk_end() <= self.boundary else "suffix_prefill"

    def chunk_tokens(self):
        """The next chunk as a [1, S] int32 array: always one row."""
        return mx.array(self.head[self.pos : self.chunk_end()], dtype=mx.int32)[None]

    def complete_chunk(self, end: int) -> bool:
        """Book a computed chunk: advance, snapshot at an admission point, finish if done."""
        self.pos = end
        if self.pos in self.admit_at:
            self._admit(self.pos)
        if self.done:
            self._finish()
            return True
        return False

    def step(self) -> bool:
        """Compute the next chunk (one single-row forward). Returns True when the head is done."""
        if self.done:
            self._finish()
            return True
        end = self.chunk_end()
        phase = self.chunk_phase()
        snap = self.meter.snap() if self.meter is not None else None
        self.model.forward(
            mx.array(self.head[self.pos : end], dtype=mx.int32)[None], self.state, logits="none"
        )
        mx.eval(*self.state.arrays())
        if self.meter is not None:
            self.meter.note(phase, end - self.pos, snap)
        return self.complete_chunk(end)

    def run(self) -> None:
        """Compute the whole head (used by `score`, which is never interleaved)."""
        while not self.step():
            pass

    # ------------------------------------------------------------------ internals
    def _admit(self, upto: int) -> None:
        """Store the state after `upto` tokens of this head.

        Interior points (`upto < self.n`) are prefixes other rows will restore, so they are pinned;
        the end-of-head entry serves only this row and rotates through the unpinned part of the cap.
        """
        key = tuple(self.head[:upto])
        if key not in self.store:
            self.store.admit(
                self.state.snapshot(key),
                pin=(upto < self.n and upto != self.tail_at)
                or (upto == self.n and self.end_shared),
            )
            self.snapshots += 1

    def _finish(self) -> None:
        # Like every admission point, the end-of-head entry must gain at least `min_divergence`
        # tokens over what was restored; a shorter gain is recomputed rather than stored.
        if (
            self.use_store
            and not self.end_covered
            and self.n - self.hit_len >= self.store.min_divergence
        ):
            self._admit(self.n)


class WavePrefill:
    """One layer-synchronous chunk for N prefill rows; each layer's expert union is fetched once.

    Each row keeps its own `PrefillTask` bookkeeping (grid, phase, snapshots, finish rule); only the
    N `Qwen4ExpForCausalLM.forward` calls become one `Qwen4ExpForCausalLM.wave_forward`. Rows share
    `pool.ensure()`, never a tensor, so their chunk lengths and rope offsets may differ. All
    members' next chunks must be on the same side of their prefix boundary (`Engine._form_wave`
    picks the members).
    """

    __slots__ = ("model", "tasks", "meter")

    def __init__(self, model, tasks: list, meter=None):
        if len(tasks) < 2:
            raise ValueError("a wave needs at least two rows; one row uses PrefillTask.step")
        phases = {t.chunk_phase() for t in tasks}
        if len(phases) != 1:
            raise ValueError(f"a wave must be all-prefix or all-suffix, got {sorted(phases)}")
        self.model = model
        self.tasks = list(tasks)
        self.meter = meter

    def step(self) -> list:
        """Compute one chunk for every row. Returns the tasks that finished their head."""
        tasks = [t for t in self.tasks if not t.done]
        ends = [t.chunk_end() for t in tasks]
        phase = tasks[0].chunk_phase()
        tokens = [t.chunk_tokens() for t in tasks]
        snap = self.meter.snap() if self.meter is not None else None
        self.model.wave_forward(tokens, [t.state for t in tasks])
        arrays = []
        for t in tasks:
            arrays.extend(t.state.arrays())
        mx.eval(*arrays)
        if self.meter is not None:
            self.meter.note(phase, sum(e - t.pos for e, t in zip(ends, tasks)), snap)
        return [t for t, e in zip(tasks, ends) if t.complete_chunk(e)]


class PromptProcessingMixin:
    """Prefill between decode steps, then the start of decode."""

    def _open_task(self, row: Req) -> bool:
        """Give `row` its `PrefillTask`; returns False if the row failed and is gone."""
        if row.task is not None:
            return True
        try:
            row.task = PrefillTask(
                self.model,
                self.store,
                self.wave_chunk,
                row.job.head,
                row.state,
                use_store=True,
                group_point=row.group_point,
                meter=self.meter,
                snap_grid=True,
            )
        except Exception as exc:  # noqa: BLE001
            self._fail(row, exc)
            return False
        return True

    def _refresh_task(self, row: Req) -> bool:
        """Re-resolve a row's prefix lookup if it found nothing and the store now has a match.

        On a cold store the lead publishes the shared-prefix snapshot after its followers opened
        their tasks. Safe only while the row has computed and restored nothing, because
        `RequestState.restore` needs a fresh state. Returns False if the row failed and is gone.
        """
        t = row.task
        if t is None:
            return self._open_task(row)
        if t.hit_len or t.pos or t.n <= 0 or row.state.tokens:
            return True
        n = t.n
        if self.store.covered_len(t.head, lambda k: k % self.wave_chunk == 0 or k == n) <= 0:
            return True
        row.task = None
        return self._open_task(row)

    def _form_wave(self, allow_wave: bool = True) -> list:
        """The rows whose next prompt chunk this round computes, oldest prefilling row first.

        `allow_wave` is False while any row decodes, since a wave is non-preemptible. A lead inside
        its shared-prefix boundary runs alone so followers restore its snapshot rather than
        recompute it; followers join only on their suffixes, which keeps the phase meter exact.
        """
        rows, lead = self._pick_lead()
        if lead is None:
            return []
        if (
            not allow_wave
            or self.wave_width <= 1
            or lead.task.done
            or lead.task.chunk_phase() == "prefix_prefill"
        ):
            return [lead]
        if self.expert_pool is not None and not self.expert_pool.can_wave:
            return [lead]  # the pool cannot hold a layer's set: nothing to share
        wave, tokens = self._fill_wave(lead, rows)
        self.wave_tokens_max = max(self.wave_tokens_max, tokens if len(wave) > 1 else 0)
        return wave

    def _pick_lead(self) -> tuple:
        """The prefilling rows, oldest first, and the first that is not cancelled and holds a task (cancelled ones are dropped; else None)."""
        rows = [r for r in self._rows if r.phase == "prefill"]
        lead = None
        for r in rows:
            if r.cancelled:
                self._cancel(r)
                continue
            if not self._refresh_task(r):
                continue
            lead = r
            break
        return rows, lead

    def _fill_wave(self, lead: Req, rows: list[Req]) -> tuple:
        """The lead plus the followers whose suffix chunk fits the wave's token budget, in FIFO order;
        returns the wave and its token count."""
        wave = [lead]
        # The lead's chunk always runs whole: splitting it would change the absolute partition that
        # keeps warm runs bit-identical to cold ones. Followers fit the plan's wave token budget.
        tokens = lead.task.chunk_end() - lead.task.pos
        for r in rows:
            if r is lead or r.retired or r.cancelled or len(wave) >= self.wave_width:
                continue
            if not self._refresh_task(r):
                continue
            if r.task.done or r.task.chunk_phase() != "suffix_prefill":
                continue
            n = r.task.chunk_end() - r.task.pos
            if tokens + n > self.wave_tokens:
                break  # FIFO: the rows behind this one wait for the next wave
            wave.append(r)
            tokens += n
        return wave, tokens

    def _run_wave(self, wave: list) -> None:
        """Compute one chunk for every row of `wave`, then start decoding whoever finished.

        Runs inside `OffloadMoeCache.prefill_scope`, which makes prefill fetches first-choice
        eviction victims: a prefill pass never revisits a layer, so its records must not displace
        the seed.
        """
        if self.expert_pool is not None:
            with self.expert_pool.prefill_scope():
                self._run_wave_inner(wave)
        else:
            self._run_wave_inner(wave)

    def _run_wave_inner(self, wave: list) -> None:
        live = self._start_completed_rows(wave)
        while live:
            if len(live) == 1:
                row = live[0]
                self._set_owners([row.rid])
                try:
                    finished = row.task.step()  # the single-row path
                except Exception as exc:  # noqa: BLE001
                    self._fail(row, exc)
                    return
                if finished:
                    self._start_decode(row)
                return
            guard = self._begin_wave_step(live)
            try:
                done = self._step_wave(live)
            except RowFaultError as exc:
                live = self._fail_faulted_wave_rows(live, guard, exc)
                continue
            except Exception as exc:  # noqa: BLE001
                self._fail_wave_rows(live, exc)
                return
            finally:
                guard.release()
            finished = {id(t) for t in done}
            self._start_finished_rows(live, finished)
            return

    def _start_completed_rows(self, wave: list) -> list:
        """Start decoding the rows of `wave` whose prompt is already complete; returns the rows still prefilling."""
        live = [r for r in wave if not r.task.done]
        for r in wave:
            if r.task.done:
                r.task.step()  # already complete: books the finish, computes nothing
                self._start_decode(r)
        return live

    def _begin_wave_step(self, live: list) -> StepGuard:
        """Count the wave, set the fault owners and guard the rows' states; returns the guard."""
        self.waves_run += 1
        self.wave_rows_total += len(live)
        self.wave_rows_max = max(self.wave_rows_max, len(live))
        self._set_owners([r.rid for r in live])
        guard = StepGuard([r.state for r in live])
        return guard

    def _fail_faulted_wave_rows(self, live: list, guard: StepGuard, exc: RowFaultError) -> list:
        """Roll every row back to the wave's start, fail the rows that faulted; returns the rows left to re-run."""
        # A read failed on behalf of some rows (faults.py): roll every row's bookkeeping
        # back to the wave's start, fail exactly those rows, re-run the wave without them.
        guard.restore()
        bad = [r for i, r in enumerate(live) if i in exc.rows]
        for r in bad:
            self.row_faults += 1
            self._fail(r, exc)
        live = [r for r in live if r not in bad]
        return live

    def _fail_wave_rows(self, live: list, exc: Exception) -> None:
        """Fail every row of the wave with `exc`."""
        # Any other fault inside a wave is the whole wave's fault (state advanced in place).
        for r in live:
            self._fail(r, exc)

    def _step_wave(self, live: list) -> list:
        """Compute one chunk for every row of the wave; returns the tasks that finished their head."""
        done = WavePrefill(self.model, [r.task for r in live], meter=self.meter).step()
        return done

    def _start_finished_rows(self, live: list, finished: set[int]) -> None:
        """Start decoding the rows of the wave whose task finished its head."""
        for r in live:
            if id(r.task) in finished:
                self._start_decode(r)

    def _prompt_slice(self) -> None:
        """Run one wave of prompt work between decode steps, or all pending prompts when idle."""
        decoding = self._batch is not None or any(r.phase == "decode" for r in self._rows)
        while True:
            wave = self._form_wave(allow_wave=not decoding)
            if not wave:
                return
            self._run_wave(wave)
            if decoding:
                return

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
