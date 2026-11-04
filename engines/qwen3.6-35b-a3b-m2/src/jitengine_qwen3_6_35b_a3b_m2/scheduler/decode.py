"""Decoding for `Engine`, as a mixin: one round over every decoding row, issuing the next step
before reading this one's tokens.
"""

from __future__ import annotations

import time

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m2.scheduler.batch import Req
from jitengine_qwen3_6_35b_a3b_m2.server.stats import MOE_SAMPLE_EVERY, StepTelemetry


class DecodingMixin:
    """One decode round over every decoding row."""

    def _pump(self) -> None:
        """One decode round, issuing the next step before reading this one's tokens."""
        if self._batch is None:
            rows = self._decoding_rows()
            if not rows:
                return
            self._issue_first_step(rows)
            return

        rows = self._batch
        cont = self._continuing_rows(rows)
        nxt, cont = self._issue_next_step(rows, cont)

        toks = self._y.tolist()
        if self._telem is not None:
            self.stats_.record_moe(self._telem)
        self._consume(rows, toks if isinstance(toks, list) else [toks])
        self._batch, self._y, self._telem = nxt if cont else (None, None, None)

    def _decoding_rows(self) -> list[Req]:
        """The rows in the decode phase that are not retired."""
        rows = [r for r in self._rows if r.phase == "decode" and not r.retired]
        return rows

    def _issue_first_step(self, rows: list[Req]) -> None:
        """Make room for each row and launch the first step; a step that cannot be built fails its rows."""
        for r in rows:
            self._ensure_room(r)
        y_in = mx.array([r.next_tok for r in rows], dtype=mx.int32)
        try:
            self._batch, self._y, self._telem = self._issue(rows, y_in, [r.n for r in rows])
        except Exception as exc:  # noqa: BLE001  a step that cannot even be built fails ITS rows
            self._fail_rows(rows, exc)

    def _continuing_rows(self, rows: list[Req]) -> list[Req]:
        """The rows that take another step, none while a row whose prompt just finished is waiting to join."""
        # rows that may take another step: max_tokens is known without a sync, EOS is not, so a row
        # that stops on EOS costs one wasted step
        cont = [r for r in rows if not r.retired and not r.cancelled and r.n + 1 < r.max_tokens]
        # a row whose prompt just finished must be able to join: break the pipeline for one round
        if any(r.phase == "decode" and not r.retired and r not in rows for r in self._rows):
            cont = []
        return cont

    def _issue_next_step(self, rows: list[Req], cont: list[Req]) -> tuple:
        """Launch the next step over `cont` before the step in flight is read; a failed build ends only its rows."""
        nxt = (None, None, None)
        if cont:
            for r in cont:
                self._ensure_room(r)
            if len(cont) == len(rows):
                y_in = self._y
            else:
                pos = mx.array([rows.index(r) for r in cont], dtype=mx.int32)
                y_in = self._y[pos]
            try:
                nxt = self._issue(cont, y_in, [r.n + 1 for r in cont])
            except Exception as exc:  # noqa: BLE001
                # the step in flight is still consumed below; only the rows of the failed build end
                self._fail_rows(cont, exc)
                cont = []
        return nxt, cont

    def _issue(self, rows: list[Req], y_in, steps: list[int]):
        """Build (lazily) and launch one decode step over `rows`; returns (rows, tokens, telemetry)."""
        b = len(rows)
        self._step_no += 1
        telem = None
        if self._step_no % MOE_SAMPLE_EVERY == 0:
            telem = StepTelemetry()
            telem.arm(b)
        states = [r.state for r in rows]
        if b == 1:
            # the solo path: bit-identical to unbatched decode
            logits = self.model.forward(y_in.reshape(1, 1), states[0], logits="last")
        else:
            logits = self.model.decode_step(y_in.reshape(b, 1), states, telem)
        y = self._draw(rows, logits, steps)
        mx.async_eval(y, *(telem.pending() if telem is not None else []))
        self.stats_.record_step(b)
        return rows, y, telem

    def _draw(self, rows: list, logits: mx.array, steps: list) -> mx.array:
        """One token per row; rows with equal sampler settings draw as a batch, each as if alone."""
        b = len(rows)
        if b == 1:
            return rows[0].sampler(logits[0:1], steps[0])
        from jitengine_qwen3_6_35b_a3b_m2.sampler import logprobs_from_logits, rows_key, sample_rows

        n = logits.shape[-1]
        groups: dict = {}
        out: list = [None] * b
        self._group_rows(rows, logits, steps, n, groups, out, rows_key)
        self._draw_groups(rows, logits, steps, groups, out, logprobs_from_logits, sample_rows)
        return mx.concatenate(out, axis=0)

    def _group_rows(
        self, rows: list, logits: mx.array, steps: list, n: int, groups: dict, out: list, rows_key
    ) -> None:
        """Draw alone the rows whose sampler cannot batch; bucket the others by sampler settings into `groups`."""
        for i, r in enumerate(rows):
            key = rows_key(r.sampler, n)
            if key is None:
                out[i] = r.sampler(logits[i : i + 1], steps[i]).reshape(1)
            else:
                groups.setdefault(key, []).append(i)

    def _draw_groups(
        self,
        rows: list,
        logits: mx.array,
        steps: list,
        groups: dict,
        out: list,
        logprobs_from_logits,
        sample_rows,
    ) -> None:
        """Draw each bucket of rows with equal sampler settings as one batch, each row as if alone."""
        for members in groups.values():
            if len(members) == 1:
                i = members[0]
                out[i] = rows[i].sampler(logits[i : i + 1], steps[i]).reshape(1)
                continue
            # log-probabilities per row exactly as Sampler.__call__ forms them ([1, V] each)
            lp = mx.concatenate([logprobs_from_logits(logits[i : i + 1]) for i in members], axis=0)
            tok = sample_rows(lp, [rows[i].sampler for i in members], [steps[i] for i in members])
            for j, i in enumerate(members):
                out[i] = tok[j : j + 1]

    def _consume(self, rows: list[Req], toks: list) -> None:
        now = time.monotonic()
        for r, tok in zip(rows, toks):
            if r.retired:
                continue
            r.steps += 1
            if r.t_first is None:
                r.t_first = now
            tok = int(tok)
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
