"""Decoding for `Engine`, as a mixin: one round over every decoding row, issuing the next step
before reading this one's tokens; the decode-in-turn round that gives one row the device for a
quantum; and the MTP round that verifies two positions at once.
"""

from __future__ import annotations

import logging
import time

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m2.scheduler.batch import Req
from jitengine_qwen3_8_flash_next_m2.scheduler.faults import RowFaultError, StepGuard
from jitengine_qwen3_8_flash_next_m2.scheduler.score import decode_entry
from jitengine_qwen3_8_flash_next_m2.server.stats import MOE_SAMPLE_EVERY, StepTelemetry

logger = logging.getLogger(__name__)


class DecodingMixin:
    """One decode round over every decoding row, in turns, or speculative."""

    def _pump(self) -> None:
        """One decode round, phase-metered (see stats.PhaseMeter).

        One step stays in flight while the host consumes the previous one (mx.async_eval, as in
        mlx_lm's generate_step), so the meter spans the whole round, issue plus consume.
        """
        snap = self.meter.snap() if self._batch is not None else None
        rows_done = self._pump_round()
        if snap is not None and rows_done:
            self.meter.note("decode", rows_done, snap)

    def _pump_round(self) -> int:
        """The round itself; returns how many rows produced a token (0 if none did)."""
        if self.mtp is not None and self._batch is None:
            live = self._live_rows()
            if (
                len(live) == 1
                and live[0].phase == "decode"
                and live[0].sampler.greedy
                and not live[0].cancelled
            ):
                return self._mtp_round(live[0])
        if self.decode_turn_q > 0:
            return self._turn_round()
        if self._batch is None:
            rows = self._decoding_rows()
            if not rows:
                return 0
            self._open_pipeline(rows)
            return 0

        rows = self._batch
        cont = self._continuing_rows(rows)
        nxt = self._issue_next(cont, rows)

        toks = self._y.tolist()
        if self._telem is not None:
            self.stats_.record_moe(self._telem)
        self._consume(rows, toks if isinstance(toks, list) else [toks], self._reads)
        self._batch, self._y, self._telem, self._reads = nxt if cont else (None, None, None, [])
        return len(rows)

    def _live_rows(self) -> list[Req]:
        """The rows not yet retired, for the test of a solo MTP round."""
        live = [r for r in self._rows if not r.retired]
        return live

    def _decoding_rows(self) -> list[Req]:
        """The rows in the decode phase that have not retired."""
        rows = [r for r in self._rows if r.phase == "decode" and not r.retired]
        return rows

    def _open_pipeline(self, rows: list[Req]) -> None:
        """Make room for each row and issue the first step, which stays in flight into the next round."""
        for r in rows:
            self._ensure_room(r)
        y_in = mx.array([r.next_tok for r in rows], dtype=mx.int32)
        self._batch, self._y, self._telem, self._reads = self._issue(
            rows, y_in, [r.n for r in rows]
        )

    def _continuing_rows(self, rows: list[Req]) -> list[Req]:
        """The rows of the in-flight batch that take another step (none while a finished prompt joins)."""
        # Rows that may take another step. max_tokens is known without a sync; EOS is not, so a row
        # that stops on EOS wastes exactly one speculative step and is dropped next round.
        cont = [r for r in rows if not r.retired and not r.cancelled and r.n + 1 < r.max_tokens]
        # A row whose prompt just finished must be able to join: break the pipeline for one round.
        if any(r.phase == "decode" and not r.retired and r not in rows for r in self._rows):
            cont = []
        return cont

    def _issue_next(self, cont: list[Req], rows: list[Req]) -> tuple:
        """Issue the next step over `cont`, reusing the in-flight tokens; (None, None, None, []) if `cont` is empty."""
        nxt = (None, None, None, [])
        if cont:
            for r in cont:
                self._ensure_room(r)
            if len(cont) == len(rows):
                y_in = self._y
            else:
                pos = mx.array([rows.index(r) for r in cont], dtype=mx.int32)
                y_in = self._y[pos]
            nxt = self._issue(cont, y_in, [r.n + 1 for r in cont])
        return nxt

    def _turn_sync(self) -> None:
        """Keep the FIFO in step with the rows: drop retired / failed ones, append rows that just
        entered decode in `_rows` (admission) order."""
        order = [
            r for r in self._turn_order if not r.retired and r.phase == "decode" and r in self._rows
        ]
        for r in self._rows:
            if r.phase == "decode" and not r.retired and r not in order:
                order.append(r)
        self._turn_order = order
        if self._turn_owner is not None and self._turn_owner not in order:
            self._turn_owner = None

    def _turn_pick(self):
        """The row to issue next: the owner while its quantum lasts (or while nobody waits), else
        the head of the FIFO, with the old owner moved to the back."""
        order = self._turn_order
        if not order:
            return None
        o = self._turn_owner
        if o is not None and (self._turn_used < self.decode_turn_q or len(order) == 1):
            return o
        if o is not None:
            order.remove(o)
            order.append(o)
        self._turn_owner, self._turn_used = order[0], 0
        self.turns += 1
        return order[0]

    def _turn_round(self) -> int:
        """One decode round in turn mode: the batched round's pipeline over a batch of one row.

        The turn owner is the only row issued, with one step in flight while the host consumes the
        previous one. Parked rows are not touched (no state copy, recompute or recompile): every
        compiled solo build takes the row's conv/ssm/KV as inputs, so switching rows is switching
        arguments. Each row keeps its own sampler, whose draw is a function of (seed, the row's step
        index, logits), so the tokens do not depend on when the row runs.
        """
        q = self.decode_turn_q
        if self._batch is None:
            self._turn_sync()
            owner = self._turn_pick()
            if owner is None:
                return 0
            self.parked_rows_peak = max(self.parked_rows_peak, len(self._turn_order) - 1)
            self._ensure_room(owner)
            y_in = mx.array([owner.next_tok], dtype=mx.int32)
            self._batch, self._y, self._telem, self._reads = self._issue([owner], y_in, [owner.n])
            if self._batch:
                self._turn_used += 1
            return 0

        rows = self._batch
        owner = rows[0]
        self._turn_sync()
        alone = len(self._turn_order) <= 1
        cont = (
            not owner.retired
            and not owner.cancelled
            and owner.n + 1 < owner.max_tokens
            and owner is self._turn_owner
            and (self._turn_used < q or alone)
        )
        nxt = (None, None, None, [])
        if cont:
            self._ensure_room(owner)
            nxt = self._issue([owner], self._y, [owner.n + 1])
            if nxt[0]:
                self._turn_used += 1
        toks = self._y.tolist()
        if self._telem is not None:
            self.stats_.record_moe(self._telem)
        self._consume(rows, toks if isinstance(toks, list) else [toks], self._reads)
        self._batch, self._y, self._telem, self._reads = nxt if cont else (None, None, None, [])
        return len(rows)

    def _mtp_round(self, row: Req) -> int:  # noqa: PLR0915
        """One speculative round of a lone greedy row, returning how many tokens it emitted.

        The draft d (from the MTP head) for the position after the pending token t is verified with t
        in one pass (`Qwen4ExpForCausalLM.verify2`); both positions' logits are bit-identical to solo
        steps', so the emitted tokens are the solo path's. Accepted (t's argmax == d): d's argmax
        comes free (2 tokens). Rejected: position p + 1 is rolled back (1 token). A row's first round
        is a plain solo step that also yields the hidden state the head drafts from.
        """
        st, model, head = row.state, self.model, self.mtp
        t = int(row.next_tok)
        pool = self.expert_pool
        guard = StepGuard([st])
        self._set_owners([row.rid])
        pool.decay_tokens = getattr(pool, "decay_solo", pool.decay_tokens)
        telem = None
        self._telem_tokens += 2
        if self._telem_tokens >= MOE_SAMPLE_EVERY:
            self._telem_tokens = 0
            telem = StepTelemetry()
            telem.arm(1)
        try:
            self._ensure_room(row, 2)
            m = row.mtp
            if m is None or m["pos"] != st.tokens or m["t"] != t or m["gen"] != self.mtp_gen:
                logits, hp = model.forward_hidden(mx.array([[t]], dtype=mx.int32), st, telem)
                g = mx.argmax(logits, axis=-1).astype(mx.int32)
                mst = head.new_state(int(row.max_tokens) + 16)
                d = head.draft(mst, hp, [g[0]])
                mx.async_eval(g, d)
                toks, rows_lg = [int(np.asarray(g)[0])], [logits]
            else:
                mst = m["st"]
                dv = int(np.asarray(m["d"])[0])
                lg, hid, snaps = model.verify2(t, dv, st, telem)
                gd = mx.argmax(lg, axis=-1).astype(mx.int32)
                msn, n0 = model._snap(mst["slot"], None), mst["n"]
                d_acc = head.draft(mst, hid, [dv, gd[1]])  # assumes acceptance; replaced if not
                mx.async_eval(gd, d_acc)
                g = np.asarray(gd).reshape(-1)
                if int(g[0]) == dv:
                    toks, rows_lg, d = [dv, int(g[1])], [lg[0:1], lg[1:2]], d_acc
                    self.mtp_accepted += 1
                else:
                    model.rollback1(st, snaps)
                    model._restore(mst["slot"], None, msn)
                    mst["n"] = n0
                    toks, rows_lg = [int(g[0])], [lg[0:1]]
                    d = head.draft(mst, hid[0:1], [toks[0]])
                    mx.async_eval(d)
                self.mtp_verifies += 1
            row.mtp = {"st": mst, "d": d, "pos": st.tokens, "t": toks[-1], "gen": self.mtp_gen}
            reads = [
                decode_entry(rows_lg[i], row.logprobs_k, mx.array([toks[i]], dtype=mx.int32))
                if row.logprobs_k
                else None
                for i in range(len(toks))
            ]
        except RowFaultError as exc:
            guard.restore()
            row.mtp = None
            self.row_faults += 1
            self._fail(row, exc)
            return 0
        except Exception as exc:  # noqa: BLE001  the row fails; the engine keeps serving
            logger.error(f"[engine] MTP round failed for request {row.rid}: {exc!r}")
            row.mtp = None
            self._fail(row, exc)
            return 0
        finally:
            guard.release()
            pool.decay_tokens = getattr(pool, "decay_other", pool.decay_tokens)
        self._step_no += 1
        self.stats_.record_step(1)
        if telem is not None:
            mx.eval(*[a for a in telem.pending() if isinstance(a, mx.array)])
            self.stats_.record_moe(telem)
        n = 0
        for tok, ro in zip(toks, reads):
            if row.retired:
                break
            if ro is not None:
                mx.eval(*ro.arrays())
            self._consume([row], [tok], [ro])
            n += 1
        return n

    def _issue(self, rows: list[Req], y_in, steps: list[int]):
        """Build and launch one decode step over `rows` from their [B] input tokens `y_in`.

        Returns (rows, tokens, telemetry, readouts), or (None, None, None, []) if every row failed.
        A `RowFaultError` rolls back every row (`StepGuard`), fails the faulting rows and re-issues
        the step without them, so survivors decode exactly what they decode alone. Any other
        exception fails every row of the step, whose earlier layers may already have advanced its
        state.
        """
        rows = list(rows)
        while rows:
            b = len(rows)
            telem, reads = self._arm_step_telemetry(b)
            states, guard = self._guard_step(rows)
            try:
                _pool = self.expert_pool
                if b == 1:
                    # The solo path is the pre-batcher path, bit for bit; `telem` only observes.
                    _pool.decay_tokens = getattr(_pool, "decay_solo", _pool.decay_tokens)
                    _pool.cyc_tokens = getattr(_pool, "cyc_solo", 0)  # 0 = off
                    try:
                        logits = self.model.forward(
                            y_in.reshape(1, 1), states[0], logits="last", telem=telem
                        )
                    finally:
                        _pool.decay_tokens = getattr(_pool, "decay_other", _pool.decay_tokens)
                        _pool.cyc_tokens = 0
                else:
                    logits = self.model.decode_step(y_in.reshape(b, 1), states, telem)
                y, reads = self._draw_tokens(logits, rows, steps, b)
            except RowFaultError as exc:
                guard.restore()
                rows, steps, reads, y_in = self._fail_faulted_rows(rows, exc, steps, y_in)
                continue
            except Exception as exc:  # noqa: BLE001  the step's rows fail; the engine keeps serving
                logger.error(f"[engine] decode step failed for {len(rows)} row(s): {exc!r}")
                self._fail_rows(rows, exc)
                return None, None, None, []
            finally:
                guard.release()
            self._launch_step(y, telem, reads, b)
            return rows, y, telem, reads
        return None, None, None, []

    def _arm_step_telemetry(self, b: int) -> tuple:
        """The step's telemetry (armed every MOE_SAMPLE_EVERY row-tokens, else None) and its empty readouts."""
        telem = None
        reads: list = []  # per-row readout of the distribution drawn from
        # arm on row-tokens, not steps: a batch samples as often per token as a lone row
        self._telem_tokens += b
        if self._telem_tokens >= MOE_SAMPLE_EVERY:
            self._telem_tokens = 0
            telem = StepTelemetry()
            telem.arm(b)
        return telem, reads

    def _guard_step(self, rows: list[Req]) -> tuple:
        """The rows' states, with the fault owners set and a `StepGuard` over the states."""
        states = [r.state for r in rows]
        self._set_owners([r.rid for r in rows])
        guard = StepGuard(states)
        return states, guard

    def _draw_tokens(self, logits, rows: list[Req], steps: list[int], b: int) -> tuple:
        """Draw one token per row from `logits`; returns the [B] tokens and each row's readout (or None)."""
        drawn = [r.sampler(logits[i : i + 1], s) for i, (r, s) in enumerate(zip(rows, steps))]
        y = drawn[0] if b == 1 else mx.concatenate(drawn, axis=0)
        # The top-k of the distribution each token was drawn from, in `score`'s own
        # arithmetic, built lazily and evaluated with y.
        reads = [
            decode_entry(logits[i : i + 1], r.logprobs_k, drawn[i]) if r.logprobs_k else None
            for i, r in enumerate(rows)
        ]
        return y, reads

    def _fail_faulted_rows(
        self, rows: list[Req], exc: RowFaultError, steps: list[int], y_in
    ) -> tuple:
        """Fail the rows that faulted and drop them: the surviving rows, their steps and input tokens, and no readouts."""
        bad = [r for i, r in enumerate(rows) if i in exc.rows]
        for r in bad:
            self.row_faults += 1
            self._fail(r, exc)
        keep = [i for i, r in enumerate(rows) if r not in bad]
        rows = [rows[i] for i in keep]
        steps = [steps[i] for i in keep]
        reads = []
        y_in = y_in[mx.array(keep, dtype=mx.int32)] if rows else y_in
        return rows, steps, reads, y_in

    def _fail_rows(self, rows: list[Req], exc: Exception) -> None:
        """Fail every row of the step with `exc`."""
        for r in rows:
            self._fail(r, exc)

    def _launch_step(self, y, telem, reads: list, b: int) -> None:
        """Count the step, start evaluating its tokens, telemetry and readouts, and record it in the stats."""
        self._step_no += 1
        self.decode_rows_active_max = max(self.decode_rows_active_max, b)
        pend = [a for ro in reads if ro is not None for a in ro.arrays()]
        mx.async_eval(y, *(telem.pending() if telem is not None else []), *pend)
        self.stats_.record_step(b)

    def _consume(self, rows: list[Req], toks: list, reads: list | None = None) -> None:
        now = time.monotonic()
        reads = list(reads or [])
        for i, (r, tok) in enumerate(zip(rows, toks)):
            if r.retired:
                continue
            r.steps += 1
            if r.t_first is None:
                r.t_first = now
            tok = int(tok)
            # Record the entry for the step that drew the token, EOS included, so entry j always
            # describes the distribution token j came from.
            ro = reads[i] if i < len(reads) else None
            if ro is not None:
                r.trace.extend(ro.entries())
            if tok in r.eos:
                self._retire(r, "stop", now)
                continue
            r.ids.append(tok)
            r.n += 1
            r.next_tok = tok
            r.job.out.put(("token", tok))
            if r.n >= r.max_tokens:
                self._retire(r, "length", now)
            elif r.cancelled:
                self._retire(r, "length", now)
