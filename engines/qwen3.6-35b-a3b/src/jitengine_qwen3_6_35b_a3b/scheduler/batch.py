"""Continuous-batcher bookkeeping: one `Req` per admitted request, `SlotPool` for admission.

Each row's state lives in its own slot; nothing is padded into a shared [B, T, ...] tensor. A row
is charged KV for its whole generation (`prompt + max_tokens`) at admission while its buffers grow
lazily, so `kv_allocated <= kv_live <= kv_budget` holds at every step. Excess requests queue.
"""

from __future__ import annotations

import queue
import time


def kv_bytes_for(cfg, capacity: int) -> int:
    """Bytes of attention KV a row of `capacity` positions holds (keys + values, bf16)."""
    n_attn = sum(1 for i in range(cfg.num_hidden_layers) if cfg.is_attention(i))
    return 2 * n_attn * cfg.num_key_value_heads * capacity * cfg.head_dim * 2


def delta_bytes_for(cfg) -> int:
    """Bytes of DeltaNet recurrent state one row holds (fixed, independent of prompt length)."""
    n_delta = sum(1 for i in range(cfg.num_hidden_layers) if not cfg.is_attention(i))
    ssm = cfg.linear_num_value_heads * cfg.linear_value_head_dim * cfg.linear_key_head_dim * 4
    conv = (cfg.linear_conv_kernel_dim - 1) * cfg.conv_dim * 2
    return n_delta * (ssm + conv)


def new_record(req, prompt_tokens: int, queue_ms: float, **fields) -> dict:
    """The per-request stats record of a generate request before any work: zero counters and
    a "length" finish. `fields` land just before `queue_ms`, which stays last."""
    return {
        "id": getattr(req, "request_id", None) or "",
        "kind": "generate",
        "prompt_tokens": prompt_tokens,
        "cached_tokens": 0,
        "prefill_tokens": 0,
        "prefill_ms": 0.0,
        "first_token_ms": 0.0,
        "decode_steps": 0,
        "decode_ms": 0.0,
        "completion_tokens": 0,
        "finish_reason": "length",
        **fields,
        "queue_ms": queue_ms,
    }


class SlotPool:
    """Free-list of `max_slots` state slots plus the live-KV byte budget."""

    def __init__(self, max_slots: int, kv_budget_bytes: int):
        self.max_slots = int(max_slots)
        self.kv_budget_bytes = int(kv_budget_bytes)
        self._free = list(range(self.max_slots))
        self.kv_live = 0  # reserved for live rows' whole generations at admission
        self.kv_allocated = 0  # what those rows' KV buffers have actually grown to
        self.admitted = 0
        self.peak_rows = 0
        self.deferred = 0  # admissions postponed because a budget was full

    @property
    def live_rows(self) -> int:
        return self.max_slots - len(self._free)

    def can_admit(self, kv_bytes: int) -> bool:
        """A row fits when a slot is free and its reservation stays inside the KV budget."""
        if not self._free:
            return False
        # no lone-row exemption: `Engine._accept` caps every row at a context the KV budget holds
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
        """Track the KV that exists, as rows allocate lazily and grow."""
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

    `n` counts emitted tokens and is the sampler's step index for the next token, so a seeded draw
    depends only on (seed, step, logits), never on which rows share the step.
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
        self.rec = new_record(req, len(prompt), (t_start - job.submitted) * 1e3, slot=slot)

    @property
    def cancelled(self) -> bool:
        return bool(self.job.cancelled)


class Job:
    """A unit of work for the model thread.

    `out` receives ("token", id) events, then exactly one ("done", payload) or ("error", exc).
    """

    __slots__ = ("kind", "req", "payload", "out", "cancelled", "submitted", "head", "counted")

    def __init__(self, kind: str, req=None, payload=None):
        self.kind = kind
        self.req = req
        self.payload = payload
        self.out: queue.Queue = queue.Queue()
        self.cancelled = False
        self.submitted = time.monotonic()
        self.head: list[int] | None = None  # prompt[:-1], filled in at admission time
        self.counted = False  # holds one of the engine's outstanding-request places
