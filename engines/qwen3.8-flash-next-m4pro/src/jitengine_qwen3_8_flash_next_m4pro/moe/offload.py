"""Expert offloading for `Engine`, as a mixin: the expert pack and the slot pool over it, the PLE
pack's streamed rows, and the fault attribution the two share (`_set_owners`).

With pack16 compact slots, the records are read from the SKYP16X1 pack beside the expert pack's
directory when it validates (`_pack16_disk`), so each read lands the slot's final bytes; otherwise
from the expert pack, encoded at publish.
"""

from __future__ import annotations

import logging
import os

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
        from jitengine_qwen3_8_flash_next_m4pro.checkpoint.expert_reader import open_pack
        from jitengine_qwen3_8_flash_next_m4pro.moe.expert_store import WORKERS, ExpertStore
        from jitengine_qwen3_8_flash_next_m4pro.moe.offload_cache import (
            DECAY_TOKENS_BATCHED,
            DECAY_TOKENS_SERVED,
            OffloadMoeCache,
        )
        from jitengine_qwen3_8_flash_next_m4pro.moe.offload_cache.victims import evict_period

        # The engine never builds a pack: `build_expert_pack` is accepted only as False.
        if self.options.get("build_expert_pack", False):
            raise ValueError(
                "the engine does not build expert packs; build them with `jitengine-qwen3.8-flash-next-m4pro pack` "
                "and pass the result to `serve --expert-pack-path`"
            )
        path, meta, scan = open_pack(
            model_path, build=False, path=self.options.get("expert_pack_path"), log=logger.info
        )
        self.expert_pack_path = path  # the prefix-store spill sits on the pack's device
        store = ExpertStore.from_header(
            path,
            meta,
            workers=int(self.options.get("expert_io_workers", WORKERS)),
            nocache=bool(self.options.get("expert_io_nocache", True)),
        )
        q = meta.get("quantization") or {}
        bits, gs = self.cfg.quant_of("model.layers.0.mlp.switch_mlp.gate_proj")
        if q and (int(q.get("bits", bits)), int(q.get("group_size", gs))) != (bits, gs):
            raise ValueError(f"pack quantization {q} != config.json's {(bits, gs)} for the experts")
        # before the factory wraps the store: a load-time census is not serving traffic
        codec = self._pool_codec(meta, scan, bits, gs, store)
        disk16, self.pack16_disk_why = False, "bf16 slots (no pack16 codec)"
        if codec is not None:
            store, disk16 = self._pack16_disk(path, meta, scan, codec, store)
        factory = self.options.get("expert_store_factory")
        if factory is not None:
            store = factory(store)
        self.expert_store = store
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
            codec=codec,
            # the plan prices an annex only on compact slots (`budget._store_spill_slots`); a
            # bf16 pool never takes one, whatever a caller set
            ann_slots=int(getattr(self.plan, "store_spill_slots", 0) or 0)
            if codec is not None
            else 0,
            disk16=disk16,
        )
        self.expert_pool.disk16_why = self.pack16_disk_why
        # Decay period by operating point: single-row decode uses `decay_solo`, everything else
        # `decay_other`; `_issue` swaps them per step.
        _pool = self.expert_pool
        _pool.decay_other = int(_pool.decay_tokens)
        _pool.decay_solo = (
            int(self.options.get("expert_pool_decay_tokens_solo", DECAY_TOKENS_SERVED))
            if _pool.decay_tokens
            else 0
        )
        # Single-row decode's victim rule: the cycle-aware one with EMA period `cyc_solo`, which the
        # decode loop arms around each b == 1 step and MTP round as it does `decay_solo`; 0 = decay
        _pool.cyc_solo = evict_period(self.expert_pool_evict)
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

    def _pool_codec(self, meta: dict, scan, bits: int, gs: int, store):
        """The pool's slot codec as the plan priced it: None (bf16 slots) or pack16's (Geometry,
        escape table), the table shipped for this checkpoint or censused from a small pack."""
        if self.plan.pool_codec != "pack16":
            return None
        from jitengine_qwen3_8_flash_next_m4pro.kernels.pack16 import Geometry
        from jitengine_qwen3_8_flash_next_m4pro.moe.offload_cache.codec import census, known_escapes

        geom = Geometry(meta["fields"], gs, bits)
        if geom.slot_bytes != self.plan.slot_bytes or geom.record_bytes != self.plan.record_bytes:
            raise ValueError(
                f"pack16 geometry {geom.slot_bytes}/{geom.record_bytes} B != the plan's "
                f"{self.plan.slot_bytes}/{self.plan.record_bytes} B"
            )
        table = known_escapes(scan.ckpt.index_sha256)
        if table is None:
            table = census(store, int(meta["records"]), geom)
        logger.info(
            f"[engine] pack16 slots: {geom.slot_bytes} B per slot for {geom.record_bytes} B records, "
            f"{len(table)} escape(s) ({self.plan.pool_codec_basis})"
        )
        return geom, table

    def _pack16_disk(self, source: str, meta: dict, scan, codec, store):
        """(store, disk16): the SKYP16X1 pack's store when the `pack16_disk` option is "on" (the
        default) and the file at `p16_format.pack16_path_for` validates against the checkpoint, this
        source pack and the resident escape table; else `store` (the source's), unchanged."""
        from jitengine_qwen3_8_flash_next_m4pro.checkpoint import p16_format as pf
        from jitengine_qwen3_8_flash_next_m4pro.checkpoint.pack_format import read_header_block
        from jitengine_qwen3_8_flash_next_m4pro.moe.expert_store import WORKERS, ExpertStore

        opt = str(self.options.get("pack16_disk", "on")).strip().lower()
        if opt not in ("on", "off"):
            raise ValueError(f"pack16_disk {opt!r}: 'on' or 'off'")
        geom, table = codec
        p16 = pf.pack16_path_for(source, scan)
        if opt == "off":
            why = "pack16_disk=off"
        elif not os.path.exists(p16):
            why = f"{p16} absent"
        else:
            ok, why = pf.validate_p16(p16, scan, source, geom, table)
            if ok:
                m16 = read_header_block(p16, pf.MAGIC_P16)
                disk = ExpertStore.from_header(
                    p16,
                    m16,
                    workers=int(self.options.get("expert_io_workers", WORKERS)),
                    nocache=bool(self.options.get("expert_io_nocache", True)),
                )
                store.close()
                self.pack16_disk_why = f"{p16} header validated"
                logger.info(
                    f"[engine] pack16 disk: on, {p16} ({m16['record_bytes']} B records, stride "
                    f"{m16['record_stride']}, {len(table)} escape(s)): reads land in the compact slots, "
                    f"no publish encode; the plan's {self.plan.staging_bytes} B of staging are not allocated"
                )
                return disk, True
        self.pack16_disk_why = why
        logger.info(
            f"[engine] pack16 disk: off ({why}); {meta['record_bytes']} B records from {source}, "
            "encoded at publish"
        )
        return store, False

    def _open_ple_store(self, model_path: str):
        """Validate the PLE pack and open its streamed row reader (never resident)."""
        from jitengine_qwen3_8_flash_next_m4pro.model.ple_store import WORKERS as PLE_WORKERS
        from jitengine_qwen3_8_flash_next_m4pro.model.ple_store import PLEStore, open_ple_pack

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
