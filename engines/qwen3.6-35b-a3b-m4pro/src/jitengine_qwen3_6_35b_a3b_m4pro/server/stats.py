"""Operator counters for GET /stats: recent per-request records, batcher counters and totals.

`expert_cache` stays zero here; under a memory budget `Engine.stats()` overlays the pool's counters.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque

logger = logging.getLogger(__name__)

RING = 64
MOE_SAMPLE_EVERY = 32


class StepTelemetry:
    """MoE dispatch counters of one decode step: `arm()` before it, `note()` from each MoE block.

    Distinct counts stay lazy (`pending()`) and are evaluated with the step's token sync.
    """

    __slots__ = ("collect", "rows", "layers", "pairs", "_distinct")

    def __init__(self):
        self.collect = False
        self.rows = 0
        self.layers = 0
        self.pairs = 0
        self._distinct: list = []

    def arm(self, rows: int) -> None:
        """Start collecting for a step of `rows` rows, with the counts reset."""
        self.collect = True
        self.rows = rows
        self.layers = 0
        self.pairs = 0
        self._distinct = []

    def note(self, pairs: int, distinct) -> None:
        """Count one MoE layer: `pairs` (row, expert) pairs, and `distinct`, how many different
        experts they name (an int, or a lazy array that `total_distinct` evaluates)."""
        self.layers += 1
        self.pairs += int(pairs)
        self._distinct.append(distinct)

    def pending(self) -> list:
        """The lazy per-layer distinct counts, to be evaluated with the step's token sync."""
        return list(self._distinct)

    def total_distinct(self) -> int:
        """The distinct experts summed over the noted layers; this evaluates the lazy counts, so
        call it after the step's token sync."""
        return int(sum(int(d) for d in self._distinct))


class Stats:
    """The engine's request counters for `/stats`: totals, the ring of recent per-request
    records, and the batching and MoE dispatch counts."""

    def __init__(self, ring: int = RING):
        self._lock = threading.Lock()
        self._recent: deque = deque(maxlen=ring)
        self.started = time.time()
        self.requests = 0
        self.errors = 0
        # every request the engine failed, completions and scores, by the exception's class
        self.error_types: Counter = Counter()
        self.prompt_tokens = 0
        self.prefill_tokens = 0
        self.cached_tokens = 0
        self.completion_tokens = 0
        self.decode_steps = 0
        self.prefill_ms = 0.0
        self.decode_ms = 0.0
        self.expert_cache = {"hits": 0, "misses": 0, "miss_bytes": 0}
        self.batched_steps = 0  # decode steps that carried >= 2 rows
        self.batched_rows = 0  # rows summed over those steps (mean batch = rows / steps)
        self.solo_steps = 0  # decode steps that carried exactly one row
        self.moe = {"steps": 0, "rows": 0, "layers": 0, "pairs": 0, "distinct": 0}

    def record_step(self, rows: int) -> None:
        """Count one decode step of `rows` rows as batched (more than one) or solo. Called by the
        worker thread alone; takes no lock."""
        if rows > 1:
            self.batched_steps += 1
            self.batched_rows += rows
        else:
            self.solo_steps += 1

    def record_moe(self, telem) -> None:
        """One sampled step's MoE dispatch counters (see StepTelemetry)."""
        with self._lock:
            m = self.moe
            m["steps"] += 1
            m["rows"] += int(telem.rows)
            m["layers"] += int(telem.layers)
            m["pairs"] += int(telem.pairs)
            m["distinct"] += telem.total_distinct()

    def record(self, rec: dict, log: bool = True) -> None:
        """Add one finished request's record to the totals and the recent ring and log it unless
        `log` is false; a `finish_reason` of "error" also counts under `error_types`."""
        with self._lock:
            self._recent.append(rec)
            self.requests += 1
            if rec.get("finish_reason") == "error":
                self.errors += 1
                self.error_types[rec.get("error", "unknown")] += 1
            self.prompt_tokens += int(rec.get("prompt_tokens", 0))
            self.prefill_tokens += int(rec.get("prefill_tokens", 0))
            self.cached_tokens += int(rec.get("cached_tokens", 0))
            self.completion_tokens += int(rec.get("completion_tokens", 0))
            self.decode_steps += int(rec.get("decode_steps", 0))
            self.prefill_ms += float(rec.get("prefill_ms", 0.0))
            self.decode_ms += float(rec.get("decode_ms", 0.0))
        if log:
            ds = rec.get("decode_steps") or 0
            dms = rec.get("decode_ms") or 0.0
            logger.info(
                "[req] id={id} kind={kind} prompt={prompt_tokens} cached={cached_tokens} prefill={prefill_tokens} "
                "prefill_ms={prefill_ms:.0f} first_token_ms={first_token_ms:.0f} decode_steps={decode_steps} "
                "decode_ms={decode_ms:.0f} step_ms={step:.1f} completion={completion_tokens} finish={finish_reason} "
                "queue_ms={queue_ms:.0f}".format(
                    step=(dms / ds if ds else 0.0),
                    **{
                        k: rec.get(k, 0)
                        for k in (
                            "id",
                            "kind",
                            "prompt_tokens",
                            "cached_tokens",
                            "prefill_tokens",
                            "prefill_ms",
                            "first_token_ms",
                            "decode_steps",
                            "decode_ms",
                            "completion_tokens",
                            "finish_reason",
                            "queue_ms",
                        )
                    },
                )
            )

    def count_error(self, kind: str) -> None:
        """A score request failed with an exception of class `kind` (`record` counts completions)."""
        with self._lock:
            self.error_types[kind] += 1

    def snapshot(self) -> dict:
        """The counters, their derived means and the recent records as a dict, read under the
        lock; `Engine.stats` merges it into `/stats`."""
        with self._lock:
            m = self.moe
            return {
                "uptime_s": time.time() - self.started,
                "requests": self.requests,
                "errors": self.errors,
                "error_types": dict(self.error_types),
                "prompt_tokens": self.prompt_tokens,
                "prefill_tokens": self.prefill_tokens,
                "cached_tokens": self.cached_tokens,
                "completion_tokens": self.completion_tokens,
                "decode_steps": self.decode_steps,
                "prefill_ms": self.prefill_ms,
                "decode_ms": self.decode_ms,
                "expert_cache_hits": self.expert_cache["hits"],
                "expert_cache_misses": self.expert_cache["misses"],
                "expert_cache_miss_bytes": self.expert_cache["miss_bytes"],
                "batched_decode_steps": self.batched_steps,
                "solo_decode_steps": self.solo_steps,
                "mean_batch_rows": (self.batched_rows / self.batched_steps)
                if self.batched_steps
                else 0.0,
                "moe_dispatch": {
                    "sampled_steps": m["steps"],
                    "mean_rows": (m["rows"] / m["steps"]) if m["steps"] else 0.0,
                    # per layer per step: (row, expert) pairs dispatched, distinct experts read
                    "mean_pairs_per_layer": (m["pairs"] / m["layers"]) if m["layers"] else 0.0,
                    "mean_distinct_experts_per_layer": (m["distinct"] / m["layers"])
                    if m["layers"]
                    else 0.0,
                    "dedup_ratio": (m["pairs"] / m["distinct"]) if m["distinct"] else 0.0,
                },
                "recent": list(self._recent),
            }
