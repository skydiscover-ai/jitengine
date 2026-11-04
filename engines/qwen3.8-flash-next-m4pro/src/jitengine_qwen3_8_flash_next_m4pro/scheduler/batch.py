"""Continuous-batcher bookkeeping: one `Req` per admitted request, `SlotPool` for admission.

Rows are never padded into a shared tensor: each row's state lives in its own slot, so rows join
and leave without touching other rows' buffers. Admission needs a free slot (bounding the fixed
per-row DeltaNet state) and room in the KV budget, which charges a row its whole generation
(`prompt + max_tokens`) up front; requests beyond either cap queue.
"""

from __future__ import annotations

import queue
import time


def kv_bytes_for(cfg, capacity: int, dtype_bytes: int = 2) -> int:
    """Bytes of attention state for `capacity` positions: keys, values and the QSA indexer keys."""
    n_attn = sum(1 for i in range(cfg.num_hidden_layers) if cfg.is_attention(i))
    kv = 2 * n_attn * cfg.num_key_value_heads * capacity * cfg.head_dim * dtype_bytes
    idx = n_attn * capacity * cfg.indexer_head_dim * dtype_bytes
    return kv + idx


class SlotPool:
    """Free-list of `max_slots` state slots plus the live-KV byte budget."""

    def __init__(self, max_slots: int, kv_budget_bytes: int):
        self.max_slots = int(max_slots)
        self.kv_budget_bytes = int(kv_budget_bytes)
        self._free = list(range(self.max_slots))
        self.kv_live = 0  # reserved for live rows (each whole generation, at admission)
        self.kv_allocated = 0  # what those rows' KV buffers have actually grown to
        self.admitted = 0
        self.peak_rows = 0
        self.deferred = 0  # admissions postponed because a budget was full

    @property
    def live_rows(self) -> int:
        return self.max_slots - len(self._free)

    def can_admit(self, kv_bytes: int) -> bool:
        """A row fits when a slot is free and its reservation stays inside the KV budget.

        A row alone is always admitted, since refusing it would deadlock. This is safe only because
        `Engine._accept` and the server refuse any request whose `prompt + max_tokens` exceeds the
        budget on its own.
        """
        if not self._free:
            return False
        if self.live_rows == 0:
            return True
        return self.kv_live + int(kv_bytes) <= self.kv_budget_bytes

    def acquire(self, kv_bytes: int) -> int:
        idx = self._free.pop(0)
        self.kv_live += int(kv_bytes)
        self.admitted += 1
        self.peak_rows = max(self.peak_rows, self.live_rows)
        return idx

    def release(self, idx: int, kv_bytes: int) -> None:
        if idx is None:
            return
        self.kv_live = max(0, self.kv_live - int(kv_bytes))
        if idx not in self._free:
            self._free.append(idx)
            self._free.sort()

    def note_allocated(self, delta: int) -> None:
        """Track the KV actually allocated as rows grow lazily."""
        self.kv_allocated = max(0, self.kv_allocated + int(delta))

    def stats(self) -> dict:
        return {
            "max_slots": self.max_slots,
            "live_rows": self.live_rows,
            "peak_rows": self.peak_rows,
            "kv_live_bytes": self.kv_live,
            "kv_allocated_bytes": self.kv_allocated,
            "kv_budget_bytes": self.kv_budget_bytes,
            "admitted": self.admitted,
            "deferred": self.deferred,
        }


class Req:
    """One admitted request: its slot, state, sampler and decode progress.

    `phase` is "prefill" until the prompt pass finishes, then "decode". `n` counts emitted tokens
    and is the sampler step of the next token, so a seeded draw is independent of the other rows.
    """

    __slots__ = (
        "job",
        "req",
        "prompt",
        "max_tokens",
        "eos",
        "sampler",
        "state",
        "slot",
        "kv_bytes",
        "kv_alloc",
        "group_point",
        "task",
        "phase",
        "next_tok",
        "n",
        "steps",
        "ids",
        "rec",
        "retired",
        "t_start",
        "t_prefill",
        "t_first",
        "pool0",
        "rid",
        "logprobs_k",
        "trace",
        "mtp",
    )

    def __init__(
        self,
        job,
        req,
        prompt: list[int],
        max_tokens: int,
        eos: set,
        sampler,
        state,
        slot: int,
        kv_bytes: int,
        group_point: int,
        t_start: float,
    ):
        self.job = job
        self.req = req
        self.prompt = prompt
        self.max_tokens = int(max_tokens)
        self.eos = eos
        self.sampler = sampler
        self.state = state
        self.slot = slot
        self.kv_bytes = int(kv_bytes)  # reserved at admission for the whole generation
        self.kv_alloc = 0  # KV actually allocated so far (kept in step with SlotPool)
        self.group_point = int(group_point)
        self.task = None
        self.phase = "prefill"
        self.next_tok = prompt[-1]
        self.n = 0
        self.steps = 0
        self.ids: list[int] = []
        self.retired = False
        self.t_start = t_start
        self.t_prefill = t_start
        self.t_first = None
        self.pool0 = None  # expert-cache counters when this row was admitted
        # with logprobs requested, `trace` gets one entry per decode step: the top-k of the
        # distribution the token was drawn from, computed as `score` computes it
        self.logprobs_k = int(getattr(job, "logprobs", 0) or 0)
        self.mtp = None  # the MTP draft state of a lone greedy row (DecodingMixin._mtp_round)
        self.trace: list[dict] = []
        self.rid = getattr(req, "request_id", None)  # the client's request id (for the fault seam)
        self.rec = {
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
            "slot": slot,
            "queue_ms": (t_start - job.submitted) * 1e3,
        }

    @property
    def cancelled(self) -> bool:
        return bool(self.job.cancelled)


class Job:
    """A unit of work for the model thread.

    `out` receives ("token", id) events, then exactly one ("done", payload) or ("error", exc).
    """

    __slots__ = ("kind", "req", "payload", "out", "cancelled", "submitted", "head", "logprobs")

    def __init__(self, kind: str, req=None, payload=None, logprobs: int = 0):
        self.kind = kind
        self.req = req
        self.payload = payload
        # top-k logprobs to report per generated token (0 = not requested)
        self.logprobs = int(logprobs or 0)
        self.out: queue.Queue = queue.Queue()
        self.cancelled = False
        self.submitted = time.monotonic()
        self.head: list[int] | None = None  # prompt[:-1], filled in at admission time
