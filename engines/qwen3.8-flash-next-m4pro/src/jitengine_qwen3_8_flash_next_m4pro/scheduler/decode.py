"""Decoding for `Engine`, as a mixin: one round over every decoding row, issuing the next step
before reading this one's tokens; and the MTP round of a lone greedy row, which verifies two
positions in one pass.
"""

from __future__ import annotations

import logging
import time

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m4pro import envs
from jitengine_qwen3_8_flash_next_m4pro.scheduler.batch import Req
from jitengine_qwen3_8_flash_next_m4pro.scheduler.faults import RowFaultError, StepGuard
from jitengine_qwen3_8_flash_next_m4pro.scheduler.score import decode_entry
from jitengine_qwen3_8_flash_next_m4pro.server.stats import MOE_SAMPLE_EVERY, StepTelemetry

logger = logging.getLogger(__name__)


class DecodingMixin:
    """One decode round over every decoding row, or the MTP round of a lone greedy row."""

    def _pump(self) -> None:
        """One decode round, phase-metered (see stats.PhaseMeter).

        One step stays in flight while the host consumes the previous one (mx.async_eval, as in
        mlx_lm's generate_step), so the meter spans the whole round, issue plus consume.
        """
        snap = self.meter.snap() if (self._batch is not None or self.mtp is not None) else None
        rows_done = self._pump_round()
        if snap is not None and rows_done:
            self.meter.note("decode", rows_done, snap)

    def _pump_round(self) -> int:
        """The round itself; returns how many rows produced a token (0 if none did)."""
        if self._batch is None and self.mtp is not None:
            row = self._mtp_row()
            if row is not None:
                return self._mtp_round(row)
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
        # A lone greedy row whose MTP head just loaded: drain the pipeline into the MTP round.
        if cont and self.mtp is not None and self._mtp_row() is rows[0]:
            cont = []
        return cont

    def _mtp_row(self) -> Req | None:
        """The row an MTP round may run: the only live row, decoding, greedy, not cancelled."""
        live = [r for r in self._rows if not r.retired]
        if (
            len(live) == 1
            and live[0].phase == "decode"
            and live[0].sampler.greedy
            and not live[0].cancelled
        ):
            return live[0]
        return None

    def _mtp_round(self, row: Req) -> int:  # noqa: PLR0915
        """One speculative round of a lone greedy row; returns how many tokens it emitted.

        With a draft d (from the MTP head) for the position after the pending token t, the pass
        verifies t and d together (`Qwen4ExpForCausalLM.verify2`). Both rows' logits are
        bit-identical to solo steps', and tokens are drawn from them by the row's own sampler at the
        solo step numbers, so the emitted tokens and the rows the digest hook logs are the solo
        path's. Accepted (t's token is d): two tokens. Rejected: position p + 1 is rolled back
        (`rollback1`) and one token is emitted; the second row was never drawn from and is not
        logged. Without a valid draft (the row's first round, a reloaded head) or with one token
        left, the round is a solo step (`forward_hidden`), which also yields the hidden state the
        head drafts from.
        """
        st, model, head = row.state, self.model, self.mtp
        t = int(row.next_tok)
        n0 = int(row.n)
        pool = self.expert_pool
        m = row.mtp
        verify = (
            m is not None
            and m["pos"] == st.tokens
            and m["t"] == t
            and m["gen"] == self.mtp_gen
            and int(row.max_tokens) - n0 >= 2
        )
        guard = StepGuard([st])
        self._set_owners([row.rid])
        pool.decay_tokens = getattr(pool, "decay_solo", pool.decay_tokens)
        pool.cyc_tokens = getattr(pool, "cyc_solo", 0)  # 0 = off
        try:
            telem, _ = self._arm_step_telemetry(2 if verify else 1)
            self._ensure_room(row, 2 if verify else 1)
            if not verify:
                logits, hp = model.forward_hidden(mx.array([[t]], dtype=mx.int32), st, telem)
                drawn = [row.sampler(logits, n0)]
                rows_lg = [logits]
                mst = d = None
                if int(row.max_tokens) - n0 - 1 >= 2:  # a later round can verify a draft
                    mst = head.new_state(int(row.max_tokens) + 16)
                    _draft_prompt(self.stats_, head, row)
                    d = head.draft(mst, hp, [drawn[0]])
                mx.async_eval(
                    drawn[0],
                    *([d] if d is not None else []),
                    *(telem.pending() if telem is not None else []),
                )
                toks = [int(np.asarray(drawn[0]).reshape(-1)[0])]
            else:
                mst = m["st"]
                dv = int(np.asarray(m["d"]).reshape(-1)[0])
                lg, hid, snaps = model.verify2(t, dv, st, telem)
                drawn = [row.sampler(lg[0:1], n0), row.sampler(lg[1:2], n0 + 1)]
                head_snap, head_n = model._snap(mst["slot"], None), mst["n"]
                # drafted as if accepted, replaced if not
                d = head.draft(mst, hid, [dv, drawn[1]])
                mx.async_eval(*drawn, d, *(telem.pending() if telem is not None else []))
                y = [int(np.asarray(a).reshape(-1)[0]) for a in drawn]
                self.mtp_verifies += 1
                if y[0] == dv:
                    self.mtp_accepted += 1
                    toks, rows_lg = y, [lg[0:1], lg[1:2]]
                else:
                    model.rollback1(st, snaps)
                    model._restore(mst["slot"], None, head_snap)
                    mst["n"] = head_n
                    toks, rows_lg, drawn = y[:1], [lg[0:1]], drawn[:1]
                    d = head.draft(mst, hid[0:1], [toks[0]])
                    mx.async_eval(d)
            digest = envs.LOGITS_DIGEST.get()
            if digest:
                _digest_rows(
                    digest, mx.concatenate(rows_lg, axis=0), list(range(n0, n0 + len(toks)))
                )
            row.mtp = (
                None
                if d is None
                else {"st": mst, "d": d, "pos": st.tokens, "t": toks[-1], "gen": self.mtp_gen}
            )
            reads = [
                decode_entry(rows_lg[i], row.logprobs_k, drawn[i]) if row.logprobs_k else None
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
            pool.cyc_tokens = 0
        self.stats_.record_step(1)
        if telem is not None:
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
        digest = envs.LOGITS_DIGEST.get()
        if digest:
            _digest_rows(digest, logits, steps)
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
        """Start evaluating its tokens, telemetry and readouts, and record it in the stats."""
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
                # "length" is the finish a cancelled row has always reported (callers read it);
                # the record says it was cancelled, so the waste shows on /stats
                r.rec["cancelled"] = True
                self._retire(r, "length", now)


def _draft_prompt(stats, head, row) -> None:
    """A new draft state: the row's prompt ids into the head's subset readout (`model.draft_vocab`),
    counted on /stats. The draft only decides how many positions a pass verifies, so no drawn row
    depends on it. A stand-in head without a `readout` attribute drafts on its own."""
    if not hasattr(head, "readout"):
        return
    ro = head.readout
    if ro is None:
        stats.note_draft_vocab("full", 0)
        return
    n, cut = ro.prepare(row.prompt)
    stats.record_draft_prompt(ro.rows, n, cut)


def _digest_rows(path: str, logits, steps: list[int]) -> None:
    """The identity check's hook: append the sha256 of each logits row (its fp32 bytes) with its step."""
    import hashlib

    import numpy as np

    rows = np.asarray(logits.astype(mx.float32))
    with open(path, "a") as f:
        for i, s in enumerate(steps):
            f.write(f"{s} {hashlib.sha256(rows[i].tobytes()).hexdigest()}\n")
