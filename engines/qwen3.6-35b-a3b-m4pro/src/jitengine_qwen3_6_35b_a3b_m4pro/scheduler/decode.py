"""Decoding for `Engine`, as a mixin: one round over every decoding row, issuing the next step
before reading this one's tokens, or one speculative round for a lone row.
"""

from __future__ import annotations

import time

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m4pro.scheduler.batch import Req
from jitengine_qwen3_6_35b_a3b_m4pro.server.stats import MOE_SAMPLE_EVERY, StepTelemetry
from jitengine_qwen3_6_35b_a3b_m4pro.speculative.copy_draft import ContextCopy

SPEC_PROF = None  # a list to collect per-round speculative timings into, or None


class DecodingMixin:
    """One decode round over every decoding row, or one speculative round for a lone row."""

    def _pump(self) -> None:
        """One decode round, issuing the next step before reading this one's tokens."""
        if self._batch is None:
            rows = self._decoding_rows()
            if not rows:
                return
            if self.mtp is not None and len(rows) == 1 and len(self._rows) == 1:
                self._run_spec_round(rows)
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

    def _run_spec_round(self, rows: list[Req]) -> None:
        """Run one speculative round for the lone decoding row; a round that raises fails the row."""
        try:
            self._spec_round(rows[0])
        except Exception as exc:  # noqa: BLE001
            self._fail_rows(rows, exc)

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
        """The rows that take another step, none while a row whose prompt just finished is waiting to join or a lone row returns to speculative rounds."""
        # rows that may take another step: max_tokens is known without a sync, EOS is not, so a row
        # that stops on EOS costs one wasted step
        cont = [r for r in rows if not r.retired and not r.cancelled and r.n + 1 < r.max_tokens]
        # a row whose prompt just finished must be able to join: break the pipeline for one round
        if any(r.phase == "decode" and not r.retired and r not in rows for r in self._rows):
            cont = []
        # a lone row with a drafter goes back to speculative rounds: break the pipeline
        if self.mtp is not None and len(cont) == 1 and len(self._rows) == 1:
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
            drawn = self._draw_rows_alone(rows, logits, steps)
            return drawn[0] if b == 1 else mx.concatenate(drawn, axis=0)
        from jitengine_qwen3_6_35b_a3b_m4pro.sampler import (
            logprobs_from_logits,
            rows_key,
            sample_rows,
        )

        n = logits.shape[-1]
        groups: dict = {}
        out: list = [None] * b
        self._group_rows(rows, logits, steps, n, groups, out, rows_key)
        self._draw_groups(rows, logits, steps, b, groups, out, logprobs_from_logits, sample_rows)
        return mx.concatenate(out, axis=0)

    def _draw_rows_alone(self, rows: list, logits: mx.array, steps: list) -> list:
        """Each row's token drawn by its own sampler, one row at a time."""
        drawn = [r.sampler(logits[i : i + 1], s) for i, (r, s) in enumerate(zip(rows, steps))]
        return drawn

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
        b: int,
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
            # log-probabilities per row exactly as Sampler.__call__ forms them: mlx's logsumexp runs
            # one threadgroup per row with the same launch shape at any row count, so one call over
            # the group's rows is bitwise the per-row [1, V] calls (tests/metal/test_sampler.py)
            sel = logits if members == list(range(b)) else logits[mx.array(members)]
            lp = logprobs_from_logits(sel)
            tok = sample_rows(lp, [rows[i].sampler for i in members], [steps[i] for i in members])
            for j, i in enumerate(members):
                out[i] = tok[j : j + 1]

    def _spec_round(self, r: Req) -> None:  # noqa: C901, PLR0915
        """One speculative round for a lone decode row (synchronous).

        Verifies [next_tok, d_1 .. d_j] in one pass (every row bit-identical to plain solo decode),
        draws each position's token with the sampler key plain decoding uses for it (step r.n + i),
        accepts drafts while they equal the drawn tokens, emits the drawn tokens up to and including
        the first mismatch, rolls the state back to the committed positions and drafts the next round.
        The emitted sequence is exactly the plain-decode sequence, greedy and seeded sampling alike,
        except under `spec_sampling="leviathan"`."""
        from jitengine_qwen3_6_35b_a3b_m4pro.speculative.verify import commit, verify_step

        prof = SPEC_PROF
        t0 = time.perf_counter()
        st = r.state
        remaining = r.max_tokens - r.n
        drafts = getattr(st, "spec_drafts", None)
        copy = []
        if remaining > 2:
            # prompt lookup over this request's own context; when it proposes, the MTP drafts are dropped
            cp = getattr(st, "copier", None)
            if cp is None:
                cp = st.copier = ContextCopy(
                    list(r.prompt) + list(r.ids), self.copy_max, True, self.copy_wide_max
                )
            nxt = getattr(st, "next_copy", None)
            copy = nxt if nxt is not None else cp.propose(remaining - 1)
            st.next_copy = None
            copy = copy[: max(remaining - 1, 0)]
            if len(copy) < 2:
                copy = []
            if copy:
                drafts = mx.array(copy, dtype=mx.int32)
        nd = 0 if drafts is None else min(int(drafts.shape[0]), remaining - 1)
        x = mx.array([r.next_tok], dtype=mx.int32)
        toks = x if nd <= 0 else mx.concatenate([x, drafts[:nd]])
        M = 1 + max(nd, 0)
        self._ensure_room(r, M)
        logits, hn, undo = verify_step(self.model, toks, st)
        from jitengine_qwen3_6_35b_a3b_m4pro.sampler import rows_key as _rows_key

        lev = self.spec_sampling == "leviathan" and not r.sampler.greedy and nd > 0
        if lev:
            # relaxed variant: probabilistic acceptance; the emitted sequence has the target's sampling
            # distribution, not the plain path's draws for the same keys
            from jitengine_qwen3_6_35b_a3b_m4pro.sampler import logprobs_from_logits as _lpl
            from jitengine_qwen3_6_35b_a3b_m4pro.speculative import spec_sample as _ss

            sm = r.sampler
            qq = (
                _ss.one_hot_q(drafts[:nd])
                if copy or getattr(st, "spec_q", None) is None
                else (st.spec_q[0][:nd], st.spec_q[1][:nd])
            )
            lev_out = _ss.accept(
                logits,
                drafts[:nd],
                qq[0],
                qq[1],
                sm.temperature,
                sm.top_p,
                sm.top_k,
                [sm.key(r.n + i) for i in range(M)],
            )
            drawn = None
        elif M > 1 and _rows_key(r.sampler, logits.shape[-1]) is not None:
            # the window's M draws as one sample_rows batch: per row the same draw as Sampler.__call__
            # (the per-row logsumexp is bitwise the [1, V] call at any row count)
            from jitengine_qwen3_6_35b_a3b_m4pro.sampler import logprobs_from_logits as _lpl
            from jitengine_qwen3_6_35b_a3b_m4pro.sampler import sample_rows as _sr

            drawn = _sr(_lpl(logits), [r.sampler] * M, [r.n + i for i in range(M)]).reshape(M)
        else:
            drawn = mx.concatenate([r.sampler(logits[i : i + 1], r.n + i) for i in range(M)])
        ahead = None
        t1 = time.perf_counter()
        if prof is not None and nd > 0:
            mx.eval(drafts)
            t1b = time.perf_counter()
        if lev:
            acc_, res_, bon_ = lev_out
            mx.eval(acc_, res_, bon_, drafts)
            d = drafts[:nd].tolist()
            av, rv, bv = acc_.tolist(), res_.tolist(), bon_.tolist()
            a = 0
            while a < nd and av[a]:
                a += 1
            s = d[:a] + [rv[a] if a < nd else bv[0]]
            drawn = mx.array(s, dtype=mx.int32)
            t2 = time.perf_counter()
        else:
            if nd > 0:
                mx.eval(drawn, drafts)
                d = drafts[:nd].tolist()
            else:
                mx.eval(drawn)
                d = []
            t2 = time.perf_counter()
            s = drawn.tolist()
            a = 0
            while a < nd and d[a] == s[a]:
                a += 1
        keep = a + 1
        if copy:
            st.copier.observe(a, nd)
        commit(st, undo, keep)
        self.stats_.record_step(1)
        self.stats_.spec = getattr(self.stats_, "spec", None) or {
            "rounds": 0,
            "drafted": 0,
            "accepted": 0,
        }
        self.stats_.spec["rounds"] += 1
        self.stats_.spec["drafted"] += nd
        self.stats_.spec["accepted"] += a
        emitted = s[:keep]
        st.spec_drafts = None
        if getattr(st, "copier", None) is not None:
            st.copier.extend(emitted)
        self._consume_many(r, emitted)
        t3 = time.perf_counter()
        if ahead is not None:
            nxt_copy = None
            cp_ = getattr(st, "copier", None)
            if cp_ is not None and r.max_tokens - r.n > 2 and not r.retired:
                nxt_copy = cp_.propose(r.max_tokens - r.n - 1)
                st.next_copy = nxt_copy
            if keep == M and not r.retired and not nxt_copy:
                st.spec_drafts = ahead[1]
                st.spec_q = (ahead[2], ahead[3]) if ahead[2] is not None else None
                st.mtp_pending = None
                ahead = "used"
            else:
                st.mtp.trim(st.mtp.offset - ahead[0])  # undo the optimistic head update
        if not r.retired and self.mtp is not None and ahead != "used":
            pend = getattr(st, "mtp_pending", None) or []
            pend.append((hn[:keep], drawn[:keep]))
            cp = getattr(st, "copier", None)
            if cp is not None and r.max_tokens - r.n > 2 and ahead is None:
                # decide the next round's copy now; if it copies, the MTP draft would be thrown away, so
                # defer the head's cache update (the committed pairs are replayed, in order, before its next draft)
                st.next_copy = cp.propose(r.max_tokens - r.n - 1)
            if getattr(st, "next_copy", None):
                st.mtp_pending = pend
            else:
                st.mtp_pending = None
                hs = pend[0][0] if len(pend) == 1 else mx.concatenate([p[0] for p in pend], axis=0)
                ts = pend[0][1] if len(pend) == 1 else mx.concatenate([p[1] for p in pend], axis=0)
                coupled = None
                if self.spec_sampling == "leviathan" and not r.sampler.greedy:
                    sm = r.sampler
                    dkeys = [mx.random.split(sm.key(r.n + j), 4)[3] for j in range(self.spec_k)]
                    dr, qi, qp = self.mtp.draft_sampled(
                        st.mtp, hs, ts, self.spec_k, sm.temperature, sm.top_p, sm.top_k, dkeys
                    )
                    st.spec_drafts, st.spec_q = dr, (qi, qp)
                    mx.async_eval(dr, qi, qp)
                else:
                    st.spec_q = None
                    st.spec_drafts = self.mtp.advance_and_draft(
                        st.mtp, hs, ts, self.spec_k, coupled
                    )
                    mx.async_eval(st.spec_drafts)
        if prof is not None:
            t4 = time.perf_counter()
            prof.append(
                {
                    "M": M,
                    "keep": keep,
                    "build": (t1 - t0) * 1e3,
                    "wait_drafts": ((t1b - t1) * 1e3) if nd > 0 else 0.0,
                    "eval": (t2 - (t1b if nd > 0 else t1)) * 1e3,
                    "post": (t3 - t2) * 1e3,
                    "draft_build": (t4 - t3) * 1e3,
                }
            )

    def _consume_many(self, r: Req, toks: list) -> None:
        for t in toks:
            if r.retired:
                return
            self._consume([r], [t])

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
