"""The stats of `OffloadMoeCache`, as a mixin: residency, the serving counters and `stats()`."""

from __future__ import annotations

from jitengine_qwen3_8_flash_next_m2.moe.offload_cache.reads import CHAIN_W_EDGES


class PoolStatsMixin:
    """Residency, the counter snapshot and its identity, and `stats()`."""

    # ------------------------------------------------------------------ residency
    def resident(self) -> int:
        return int((self.slot_for_id >= 0).sum())

    def capacity_fraction(self) -> float:
        return self.slots / float(self.num_ids)

    # ------------------------------------------------------------------ observability
    def counters(self) -> dict:
        """The cumulative serving counters, as a snapshot that satisfies the identity on its own."""
        with self._lock:
            return {
                "expert_lookups": self.lookups,
                "expert_hits": self.hits,
                "expert_misses": self.misses,
                "expert_dedup_saved": self.dedup_saved,
                "expert_miss_bytes": self.miss_bytes,
            }

    def _io_stats(self) -> dict:
        st = self.store.stats() if hasattr(self.store, "stats") else {}
        sc = st.get("sched") or {}
        with self._lock:
            out = {
                "expert_spec_reads_skipped": self.spec_reads_skipped,
                "expert_spec_promoted": self.spec_promoted,
                "expert_chain_gate": {
                    "min_weight": 0.0,
                    "max_rank": 0,
                    "weight_edges": list(CHAIN_W_EDGES),
                },
                "expert_chain_by_rank": {k: list(v) for k, v in self.chain_by_rank.items()},
                "expert_chain_by_weight": {k: list(v) for k, v in self.chain_by_weight.items()},
            }
        out["expert_io_sched"] = sc.get("sched", "unknown")
        out["expert_io_max_inflight"] = sc.get("max_inflight", 0)
        out["expert_io_max_inflight_observed"] = sc.get("max_inflight_observed", 0)
        out["expert_io_demand_wait_ms_p50"] = sc.get("demand_wait_ms_p50", 0.0)
        out["expert_io_demand_wait_ms_p95"] = sc.get("demand_wait_ms_p95", 0.0)
        out["expert_io_demand_latency_ms_p50"] = sc.get("demand_latency_ms_p50", 0.0)
        out["expert_io_demand_latency_ms_p95"] = sc.get("demand_latency_ms_p95", 0.0)
        return out

    def identity_ok(self) -> bool:
        """hits + misses + dedup_saved == lookups."""
        with self._lock:
            return self.hits + self.misses + self.dedup_saved == self.lookups

    def stats(self) -> dict:
        c = self.counters()
        distinct = c["expert_hits"] + c["expert_misses"]
        return {
            **c,
            # hit rate over distinct record requests; dedup is reported separately
            "expert_hit_rate": (c["expert_hits"] / distinct) if distinct else 0.0,
            "expert_distinct_requests": distinct,
            "expert_counter_identity_ok": self.identity_ok(),
            "expert_counter_identity": "expert_hits + expert_misses + expert_dedup_saved == expert_lookups",
            "expert_miss_bytes_per_miss": (c["expert_miss_bytes"] // c["expert_misses"])
            if c["expert_misses"]
            else 0,
            "expert_prefetched": self.prefetched,
            "expert_prefetch_bytes": self.prefetch_bytes,
            "expert_prefetch_used": self.prefetch_used,
            "expert_prefetch_wasted": self.prefetch_wasted,
            "expert_prefetch_waited": self.prefetch_waited,
            # read discipline. demand_wait = submit -> start of a demand read
            # (time it queued in software); demand_latency = submit -> landed.
            **self._io_stats(),
            "expert_spec_redo_layers": self.spec_redo_layers,
            "expert_spec_stale_layers": self.spec_stale_layers,
            "expert_reserve_slots": self.reserve_n,
            "expert_pool_ext_slots": self.ext_slots,
            "expert_pool_ext_active": bool(self.ext_active),
            "expert_pool_ext_releases": self.ext_releases,
            "expert_pool_ext_regrows": self.ext_regrows,
            "expert_landing_slots": self.landing_n,
            "expert_landing_fetches": self.landing_fetches,
            "expert_landing_fallbacks": self.landing_fallbacks,
            "expert_landing_orphaned": self.landing_orphaned,
            "expert_read_seconds": self.fetch_seconds,
            "exposed_stall_seconds": self.exposed_stall_seconds,
            "pool_slots": self.slots,
            "pool_bytes": self.slots * self.record_bytes,
            "resident_experts": self.resident(),
            "expert_pool_slots": self.slots,
            "expert_pool_resident": self.resident(),
            "expert_pool_capacity_fraction": self.capacity_fraction(),
            "expert_pool_bytes": self.slots * self.record_bytes,
            "expert_pool_largest_array_bytes": self.max_array_bytes,
            "expert_pool_evictions": self.evictions,
            "expert_pool_evictions_harvest": self.evictions_harvest,
            "expert_harvest_dropped_disabled": self.harvest_dropped_disabled,
            "expert_pool_cyc_evict_solo": int(self.cyc_solo),  # N armed for b == 1 decode, 0 = off
            "expert_pool_evictions_cyc": self.evictions_cyc,
            "expert_pool_cyc_observations": self.cyc_observations,
            "expert_pool_victim_policy": (
                "lru" if not self.decay_tokens else f"decay-{self.decay_tokens}"
            )
            + ("+transient-prefill" if self.transient_prefill else "")
            + (f"; solo decode cyc-{self.cyc_solo}" if self.cyc_solo else ""),
            "expert_pool_transient_prefill": self.transient_prefill,
            "expert_seed_records": self.seed_records,
            "expert_seed_bytes": self.seed_bytes,
            "expert_seed_seconds": self.seed_seconds,
            "expert_pool_landing": self.landing,
            "expert_fetch_seconds": self.fetch_seconds,
            "expert_read_retries": self.read_retries,
            "expert_retry_bytes": self.retry_bytes,
            "expert_wave_calls": self.wave_calls,
            "expert_wave_multi_calls": self.wave_multi_calls,
            "expert_io": self.store.stats(),
        }
