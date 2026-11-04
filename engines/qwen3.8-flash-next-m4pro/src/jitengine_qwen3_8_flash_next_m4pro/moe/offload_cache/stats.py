"""The stats of `OffloadMoeCache`, as a mixin: residency, the serving counters and `stats()`."""

from __future__ import annotations


class PoolStatsMixin:
    """Residency, the counter snapshot and its identity, and `stats()`."""

    # ------------------------------------------------------------------ residency
    def resident(self) -> int:
        return int((self.slot_for_id >= 0).sum())

    def planned_slots(self) -> int:
        """The plan's pool: main plus extension slots. The annex's ids (`ann0` on) are borrowed from
        the prefix store and reported on their own (`expert_pool_ann_*`), never as pool slots."""
        return int(getattr(self, "ann0", self.slots))

    def capacity_fraction(self) -> float:
        return self.planned_slots() / float(self.num_ids)

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
            "expert_spec_redo_layers": self.spec_redo_layers,
            "expert_spec_stale_layers": self.spec_stale_layers,
            "expert_reserve_slots": self.reserve_n,
            "expert_pool_ext_slots": self.ext_slots,
            "expert_pool_ext_active": bool(self.ext_active),
            "expert_pool_ext_releases": self.ext_releases,
            "expert_pool_ext_regrows": self.ext_regrows,
            "expert_pool_ann_slots": self.ann_slots,  # the id range; memory only while lent
            "expert_pool_ann_active": bool(self.ann_active),
            "expert_pool_ann_live": int(self.ann_live),
            "expert_pool_ann_live_bytes": int(self.ann_live) * int(self.slot_bytes),
            "expert_pool_ann_releases": self.ann_releases,
            "expert_pool_ann_regrows": self.ann_regrows,
            "expert_pool_ann_dropped": self.ann_dropped,
            "expert_landing_slots": self.landing_n,
            # the region's live count: free now, or parked until its write ends. Between steps it
            # equals expert_landing_slots; a lower value there is a drained region
            "expert_landing_live": len(self._landing_free) + len(self._orphans),
            "expert_landing_free": len(self._landing_free),
            "expert_landing_parked": len(self._orphans),
            "expert_landing_abandoned": self.landing_abandoned,
            "expert_landing_fetches": self.landing_fetches,
            "expert_landing_fallbacks": self.landing_fallbacks,
            "expert_landing_orphaned": self.landing_orphaned,
            "expert_read_seconds": self.fetch_seconds,
            "exposed_stall_seconds": self.exposed_stall_seconds,
            "pool_slots": self.planned_slots(),
            "pool_bytes": self.planned_slots() * self.slot_bytes,
            "resident_experts": self.resident(),
            "expert_pool_slots": self.planned_slots(),
            "expert_pool_resident": self.resident(),
            "expert_pool_capacity_fraction": self.capacity_fraction(),
            "expert_pool_bytes": self.planned_slots() * self.slot_bytes,
            "expert_pool_largest_array_bytes": self.max_array_bytes,
            "expert_pool_evictions": self.evictions,
            "expert_pool_victim_policy": (
                "lru" if not self.decay_tokens else f"decay-{self.decay_tokens}"
            )
            + ("+transient-prefill" if self.transient_prefill else "")
            + (f"; solo decode cyc-{self.cyc_solo}" if self.cyc_solo else ""),
            "expert_pool_cyc_evict_solo": int(self.cyc_solo),  # N armed for b == 1 decode, 0 = off
            "expert_pool_evictions_cyc": self.evictions_cyc,
            "expert_pool_cyc_observations": self.cyc_observations,
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
            **self.codec_stats(),
            "expert_io": self.store.stats(),
        }
