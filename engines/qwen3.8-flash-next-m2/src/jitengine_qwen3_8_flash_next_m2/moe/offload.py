"""Expert offloading for `Engine`, as a mixin: the expert pack and the slot pool over it, the PLE
pack's streamed rows, and the fault attribution the two share (`_set_owners`).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class OffloadingMixin:
    """Opens the packs the model streams from in offload mode."""

    def _set_owners(self, owners: list) -> None:
        """Record the request ids of the rows the next unit of work carries (fault attribution)."""
        if self.expert_pool is not None:
            self.expert_pool.owners = owners
        if self.ple_store is not None:
            self.ple_store.owners = owners

    def _open_expert_pool(self, model_path: str) -> None:
        """Validate the expert pack, open the uncached reader, allocate and seed the pool.

        `expert_store_factory(store)`, if given, is called once with the store this engine would
        use, and the pool reads through whatever it returns (e.g. a proxy that injects a read
        fault). The engine behaves identically whether or not the option is present.
        """
        from jitengine_qwen3_8_flash_next_m2.checkpoint.expert_reader import open_pack
        from jitengine_qwen3_8_flash_next_m2.moe.expert_store import WORKERS, ExpertStore
        from jitengine_qwen3_8_flash_next_m2.moe.offload_cache import (
            CYC_EVICT_SERVED,
            DECAY_TOKENS_BATCHED,
            DECAY_TOKENS_SERVED,
            OffloadMoeCache,
        )

        # The engine never builds a pack: `build_expert_pack` is accepted only as False.
        if self.options.get("build_expert_pack", False):
            raise ValueError(
                "the engine does not build expert packs; build them with `jitengine-qwen3.8-flash-next-m2 pack` "
                "and pass the result to `serve --expert-pack-path`"
            )
        path, meta, scan = open_pack(
            model_path, build=False, path=self.options.get("expert_pack_path"), log=logger.info
        )
        store = ExpertStore.from_header(
            path,
            meta,
            workers=int(self.options.get("expert_io_workers", WORKERS)),
            nocache=bool(self.options.get("expert_io_nocache", True)),
        )
        factory = self.options.get("expert_store_factory")
        if factory is not None:
            store = factory(store)
        self.expert_store = store
        q = meta.get("quantization") or {}
        bits, gs = self.cfg.quant_of("model.layers.0.mlp.switch_mlp.gate_proj")
        if q and (int(q.get("bits", bits)), int(q.get("group_size", gs))) != (bits, gs):
            raise ValueError(f"pack quantization {q} != config.json's {(bits, gs)} for the experts")
        self.expert_pool = OffloadMoeCache(
            self.expert_store,
            meta["num_layers"],
            meta["num_experts"],
            self.plan.pool_slots,
            meta["fields"],
            gs,
            bits=bits,
            decay_tokens=int(self.options.get("expert_pool_decay_tokens", DECAY_TOKENS_BATCHED)),
            # off by default: it lowers the decode hit rate
            transient_prefill=bool(self.options.get("transient_prefill", False)),
            reserve_slots=self.prefetch_reserve,
            landing_slots=self.landing_slots,
            ext_slots=self._ext_slots_allowed(meta),
        )
        # Decay period by operating point: single-row decode uses `decay_solo`, everything else
        # `decay_other`; `_issue` swaps them per step.
        _pool = self.expert_pool
        _pool.decay_other = int(_pool.decay_tokens)
        _pool.decay_solo = (
            int(self.options.get("expert_pool_decay_tokens_solo", DECAY_TOKENS_SERVED))
            if _pool.decay_tokens
            else 0
        )
        # cycle-aware expected-next-use eviction, armed only around the b == 1 decode step
        # (`_issue`); prefill, waves, batched decode and the MTP draft/verify round keep their
        # decay period. expert_pool_cyc_evict is the EMA period N; 0 = off.
        _pool.cyc_solo = max(0, int(self.options.get("expert_pool_cyc_evict", CYC_EVICT_SERVED)))
        if self.fault_seam is not None and self.fault_seam.kind == "expert_record":
            layer, e = self.fault_seam.key
            if not (0 <= layer < meta["num_layers"] and 0 <= e < meta["num_experts"]):
                raise ValueError(
                    f"fault_inject expert_record {self.fault_seam.key} outside the pack"
                )
            self.expert_pool.seam = self.fault_seam  # documented seam (faults.py); inert otherwise
        seed = self.options.get("expert_pool_seed_per_layer")
        # a uniform share of every layer, read layer-major so the fill is sequential
        per_layer = self.plan.pool_slots // scan.num_layers if seed is None else int(seed)
        if per_layer > 0:
            self.expert_pool.prefill_seed(per_layer)

    def _open_ple_store(self, model_path: str):
        """Validate the PLE pack and open its streamed row reader (never resident)."""
        from jitengine_qwen3_8_flash_next_m2.model.ple_store import WORKERS as PLE_WORKERS
        from jitengine_qwen3_8_flash_next_m2.model.ple_store import PLEStore, open_ple_pack

        if not any(self.cfg.has_ple(i) for i in range(self.cfg.num_hidden_layers)):
            return None
        path, meta, _ple = open_ple_pack(
            model_path, path=self.options.get("ple_pack_path"), log=logger.info
        )
        seam = (
            self.fault_seam
            if (self.fault_seam is not None and self.fault_seam.kind == "ple_row")
            else None
        )
        store = PLEStore(
            path,
            meta,
            workers=int(self.options.get("ple_io_workers", PLE_WORKERS)),
            nocache=bool(self.options.get("ple_io_nocache", True)),
            seam=seam,
        )
        logger.debug(
            f"[engine] PLE tier: {store.rows_present} rows of {store.row_bytes} B streamed from {path}"
        )
        return store
