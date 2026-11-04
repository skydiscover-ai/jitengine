"""Scoring for `Engine`, as a mixin: teacher-forced logprobs, run exclusively and cold (never
through the prefix store or a decode batch), and the forward partition and logit readout that
`score` shares with the served generate path.

The partition decides which router near-ties flip, so `score_chunks` issues exactly a cold generate
row's forward calls: the head (`n - 1` tokens) in chunks ending on the absolute prefix grid, then
the last token alone. Readouts use the row-exact head with fp32 log-softmax and top-k, sorted by
(-logprob, token id), so a position's numbers never depend on how many rows were read at once.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from jitengine_qwen3_8_flash_next_m4pro.engine.errors import InvariantError
from jitengine_qwen3_8_flash_next_m4pro.engine.io_struct import ClientDisconnected
from jitengine_qwen3_8_flash_next_m4pro.kvcache.state import RequestState

# Rows read out at once (row-exact head + fp32 log-softmax + top-k): a memory knob that never
# changes the forward partition.
SCORE_READOUT_ROWS = 128


def score_chunks(n: int, grid: int) -> list[tuple[int, int]]:
    """(start, end) forward calls a cold generate row issues for an `n`-token prompt (n >= 1)."""
    n = int(n)
    grid = max(1, int(grid))
    if n < 1:
        raise ValueError("score_chunks needs at least one token")
    out: list[tuple[int, int]] = []
    pos, head = 0, n - 1
    while pos < head:
        end = min(((pos // grid) + 1) * grid, head)
        out.append((pos, end))
        pos = end
    out.append((n - 1, n))
    return out


class Readout:
    """Lazy fp32 log-softmax + top-k of a [R, V] logits block; `targets[j]` is entry j's token."""

    __slots__ = ("idx", "vals", "tgt", "rows")

    def __init__(self, logits: mx.array, k: int, targets: Sequence[int] | None = None):
        part = logits.astype(mx.float32)
        lp = part - mx.logsumexp(part, axis=-1, keepdims=True)
        self.rows = int(lp.shape[0])
        k = max(1, min(int(k), int(lp.shape[-1])))
        self.idx = mx.argpartition(-lp, kth=k - 1, axis=-1)[..., :k]
        self.vals = mx.take_along_axis(lp, self.idx, axis=-1)
        self.tgt = None
        if targets is not None:
            t = (
                targets
                if isinstance(targets, mx.array)
                else mx.array([int(v) for v in targets], dtype=mx.int32)
            )
            m = int(t.shape[0])
            if m:
                self.tgt = mx.take_along_axis(lp[:m], t.reshape(m, 1), axis=-1)[:, 0]

    def arrays(self) -> list:
        return [self.idx, self.vals] + ([self.tgt] if self.tgt is not None else [])

    def entries(self) -> list[dict]:
        """`score`-shaped entries: {"logprob": target's or None, "top": [[id, lp], ...]}."""
        idx_l, vals_l = self.idx.tolist(), self.vals.tolist()
        tgt_l = self.tgt.tolist() if self.tgt is not None else []
        out = []
        for j in range(self.rows):
            pairs = sorted(zip(idx_l[j], vals_l[j]), key=lambda p: (-p[1], p[0]))
            out.append(
                {
                    "logprob": float(tgt_l[j]) if j < len(tgt_l) else None,
                    "top": [[int(t), float(v)] for t, v in pairs],
                }
            )
        return out


def decode_entry(logits: mx.array, k: int, drawn: mx.array | int | None = None) -> Readout:
    """Readout of one decode step's logits [1, V] in `score`'s shape; `drawn` gives the logprob."""
    tgt = None
    if drawn is not None:
        tgt = (
            drawn.reshape(1)
            if isinstance(drawn, mx.array)
            else mx.array([int(drawn)], dtype=mx.int32)
        )
    return Readout(logits, k, tgt)


def readout_rows(
    model,
    h: mx.array,
    s0: int,
    seq: Sequence[int],
    k: int,
    rows_per_group: int = SCORE_READOUT_ROWS,
) -> list[dict]:
    """Row-exact entries for `h` [1, R, hc*hidden], whose row 0 is position `s0` of `seq`."""
    n = len(seq)
    R = int(h.shape[1])
    out: list[dict] = []
    for b0 in range(0, R, max(1, int(rows_per_group))):
        b1 = min(b0 + max(1, int(rows_per_group)), R)
        block = h[0, b0:b1]  # [r, hc*hidden]
        logits = model.head(block, rows=True)  # row-exact: groups of row_limit
        tgt = [seq[i + 1] for i in range(s0 + b0, s0 + b1) if i + 1 < n]
        r = Readout(logits, k, tgt or None)
        mx.eval(*r.arrays())
        out.extend(r.entries())
        del r, logits, block
    return out


def score_sequence(
    model,
    state,
    seq: Sequence[int],
    grid: int,
    k: int,
    rows_per_group: int = SCORE_READOUT_ROWS,
    on_chunk=None,
    before_chunk=None,
) -> list[dict]:
    """Teacher-forced entries for every position of `seq` over `score_chunks`, on one state row.

    `before_chunk(s0, e)` and `on_chunk(s0, e)` are bookkeeping hooks; they must not change the
    partition.
    """
    seq = [int(t) for t in seq]
    n = len(seq)
    out: list[dict] = []
    for s0, e in score_chunks(n, grid):
        if before_chunk is not None:
            before_chunk(s0, e)
        toks = mx.array(seq[s0:e], dtype=mx.int32)[None]
        h = model.forward(toks, state, logits="hidden")
        mx.eval(h, *state.arrays())
        out.extend(readout_rows(model, h, s0, seq, k, rows_per_group))
        del h, toks
        if on_chunk is not None:
            on_chunk(s0, e)
        mx.clear_cache()
    if len(out) != n:
        raise AssertionError(f"score walked {len(out)} positions for a {n}-token sequence")
    return out


def served_entry(model, state, token: int, k: int) -> list[dict]:
    """Entries of a single-token forward of `token`, as the first decode step computes it."""
    logits = model.forward(mx.array([[int(token)]], dtype=mx.int32), state, logits="last")
    r = Readout(logits, k, None)
    mx.eval(*r.arrays())
    return r.entries()


def chunk_calls(n: int, grid: int) -> list[tuple[int, int]]:
    """Alias kept for readability at call sites that mean "the forward calls", not "the chunks"."""
    return score_chunks(n, grid)


class ScoringMixin:
    """Score requests, run one at a time between rounds."""

    def _run_scores(self) -> None:
        while self._scores:
            self._ext_policy()  # a long score chunk needs the scratch the extension may hold
            job = self._scores.pop(0)
            try:
                if job.cancelled:  # its client left while it queued
                    raise ClientDisconnected("score cancelled before it started")
                out = self._run_score(*job.payload, cancelled=lambda job=job: job.cancelled)
                job.out.put(("done", out))
            except ClientDisconnected as exc:
                job.out.put(("error", exc))
            except Exception as exc:  # noqa: BLE001
                self.stats_.count_error(type(exc).__name__)
                job.out.put(("error", exc))

    def _run_score(self, prompt_ids, continuation_ids, top_k: int, cancelled=None) -> list[dict]:
        """Teacher-forced top-k logprobs at every position of `prompt_ids + continuation_ids`.

        Entry `i` describes the distribution after consuming `seq[:i+1]`, i.e. the prediction for
        `seq[i+1]` (not the OpenAI convention); `logprob` is that token's logprob (None at the last
        position) and `top` holds `min(top_k, vocab)` entries sorted descending, ties by ascending
        id. The forwards are call for call those of a cold generate row on the same grid, read out
        row-exactly, through the same expert pool. Score jobs run alone on the model thread and
        never join a decode batch, so results are batch-invariant. `cancelled()`, when given, is
        read before each chunk: a score whose client has gone stops there.
        """
        prompt = [int(t) for t in prompt_ids]
        cont = [int(t) for t in continuation_ids]
        seq = prompt + cont
        if not seq:
            raise ValueError("empty prompt")
        # These ids reach the embedding gather, which MLX does not bounds-check; an out-of-range
        # id is an out-of-bounds GPU read that takes every concurrent stream with it.
        for t in seq:
            if t < 0 or t >= self.token_id_limit:
                raise ValueError(f"token id {t} out of range")
        n = len(seq)
        k = max(1, min(int(top_k), self.vocab))
        state = RequestState(self.cfg, n, 0, reserve=0, dtype=self.compute_dtype)
        self._set_owners([None])
        pending: dict = {}

        def before_chunk(s0: int, e: int) -> None:
            if cancelled is not None and cancelled():
                raise ClientDisconnected(f"score cancelled at position {s0} of {n}")
            pending["snap"] = self.meter.snap()

        def after_chunk(s0: int, e: int) -> None:
            self.meter.note("score", e - s0, pending.pop("snap", None))

        try:
            out = score_sequence(
                self.model,
                state,
                seq,
                self.wave_chunk,
                k,
                rows_per_group=self.score_readout_rows,
                before_chunk=before_chunk,
                on_chunk=after_chunk,
            )
        finally:
            del state
        if len(out) != n:
            raise InvariantError(f"score returned {len(out)} entries for {n} positions")
        return out
