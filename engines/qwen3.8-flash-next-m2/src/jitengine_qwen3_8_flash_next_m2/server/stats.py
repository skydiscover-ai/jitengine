"""Operator-facing counters behind `Engine.stats()` and GET /stats.

Holds recent per-request records, batcher counters and totals, per-phase burst accounting
(`PhaseMeter`) and sampled MoE dispatch counters (`StepTelemetry`).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque

logger = logging.getLogger(__name__)

RING = 64
MOE_SAMPLE_EVERY = 32  # row-tokens (not batched steps) between armed MoE telemetry steps

# Expert-cache counter names on GET /stats, listed once so callers cannot drift on spelling.
EXPERT_COUNTER_KEYS = (
    "expert_lookups",
    "expert_hits",
    "expert_misses",
    "expert_miss_bytes",
    "expert_dedup_saved",
    "expert_prefetched",
    "expert_prefetch_bytes",
    "expert_read_seconds",
    "exposed_stall_seconds",
    "pool_slots",
    "pool_bytes",
    "resident_experts",
)
PHASES = ("prefix_prefill", "suffix_prefill", "decode", "score")


class StepTelemetry:
    """MoE dispatch counters of one sampled decode step: (row, expert) pairs and distinct experts.

    `arm()` before the step, `note()` from each MoE block; the distinct counts stay lazy
    (`pending()`) until the step's token sync, then `total_distinct()` reads them.
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

    def disarm(self) -> None:
        """Stop collecting (the MoE blocks note only while `collect` is set) and drop the lazy
        distinct counts."""
        self.collect = False
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
    records, and the batching and MoE dispatch counts. `expert_cache` holds zeros that
    `Engine.stats()` overlays with the live pool's counters."""

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
        # Zeros with the right names and types; Engine.stats() overlays the live pool counters.
        self.expert_cache = {k: (0.0 if k.endswith("seconds") else 0) for k in EXPERT_COUNTER_KEYS}
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
                **self.expert_cache,
                "batched_decode_steps": self.batched_steps,
                "solo_decode_steps": self.solo_steps,
                "mean_batch_rows": (self.batched_rows / self.batched_steps)
                if self.batched_steps
                else 0.0,
                "moe_dispatch": {
                    # raw cumulative totals, so a window can be differenced; means beside them
                    "sampled_steps": m["steps"],
                    "layers": m["layers"],
                    "pairs": m["pairs"],
                    "distinct": m["distinct"],
                    "rows": m["rows"],
                    "mean_rows": (m["rows"] / m["steps"]) if m["steps"] else 0.0,
                    # per layer per step: (row, expert) pairs dispatched, distinct experts touched
                    "mean_pairs_per_layer": (m["pairs"] / m["layers"]) if m["layers"] else 0.0,
                    "mean_distinct_experts_per_layer": (m["distinct"] / m["layers"])
                    if m["layers"]
                    else 0.0,
                    "dedup_ratio": (m["pairs"] / m["distinct"]) if m["distinct"] else 0.0,
                },
                "recent": list(self._recent),
            }


class PhaseMeter:
    """Wall time, tokens and expert-cache traffic per phase of a burst.

    Prefix prefill covers the chunks below a prompt's shared-prefix boundary (the chunk-grid point
    the prefix store restored or snapshots), suffix prefill the chunks above it, then decode. A
    restored prefix runs no forward pass and counts as cached tokens, not prefix wall.
    """

    __slots__ = ("pool", "_lock", "phases")

    def __init__(self, pool=None):
        self.pool = pool
        self._lock = threading.Lock()
        self.phases = {
            p: {
                "calls": 0,
                "tokens": 0,
                "seconds": 0.0,
                "lookups": 0,
                "hits": 0,
                "misses": 0,
                "miss_bytes": 0,
                "dedup_saved": 0,
                "stall_seconds": 0.0,
            }
            for p in PHASES
        }

    def snap(self) -> tuple:
        """Open a measurement window: (t, counters, exposed_stall)."""
        if self.pool is None:
            return (time.monotonic(), None, 0.0)
        return (time.monotonic(), self.pool.counters(), float(self.pool.exposed_stall_seconds))

    def note(self, phase: str, tokens: int, snap: tuple) -> None:
        """Close the window opened by `snap` and attribute it to `phase`."""
        t0, c0, s0 = snap
        dt = time.monotonic() - t0
        with self._lock:
            d = self.phases.get(phase)
            if d is None:
                return
            d["calls"] += 1
            d["tokens"] += int(tokens)
            d["seconds"] += dt
            if c0 is not None and self.pool is not None:
                c1 = self.pool.counters()
                for k_src, k_dst in (
                    ("expert_lookups", "lookups"),
                    ("expert_hits", "hits"),
                    ("expert_misses", "misses"),
                    ("expert_miss_bytes", "miss_bytes"),
                    ("expert_dedup_saved", "dedup_saved"),
                ):
                    d[k_dst] += int(c1[k_src] - c0[k_src])
                d["stall_seconds"] += float(self.pool.exposed_stall_seconds) - s0

    def snapshot(self) -> dict:
        """Each phase's totals with its hit rate, miss GB, tokens per second, stall per token and
        share of the burst wall time; `burst_wall_seconds` sums the three serving phases, and
        `score` stays out of the split."""
        with self._lock:
            out = {}
            for p, d in self.phases.items():
                distinct = d["hits"] + d["misses"]
                out[p] = {
                    **d,
                    "hit_rate": (d["hits"] / distinct) if distinct else 0.0,
                    "miss_gb": d["miss_bytes"] / 1e9,
                    "tokens_per_s": (d["tokens"] / d["seconds"]) if d["seconds"] else 0.0,
                    "stall_per_token_ms": (1e3 * d["stall_seconds"] / d["tokens"])
                    if d["tokens"]
                    else 0.0,
                }
            # the burst wall is the three serving phases; `score` is kept out of the split
            total = sum(
                self.phases[p]["seconds"] for p in ("prefix_prefill", "suffix_prefill", "decode")
            )
            for p in out:
                out[p]["wall_fraction"] = (out[p]["seconds"] / total) if total else 0.0
            out["burst_wall_seconds"] = total
            out["wall_seconds"] = total
            return out
