"""Expert offloading for `Engine`, as a mixin: opening the expert pack and building the slot pool
the model's MoE blocks read from.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class OffloadingMixin:
    """Opens the packs the model streams from in offload mode."""

    def _open_expert_pool(self, model_path: str) -> None:
        """Validate the expert pack, open the uncached reader, allocate and seed the pool.

        A missing or stale pack stops startup unless `build_expert_pack=True`.
        """
        from jitengine_qwen3_6_35b_a3b_m4pro.checkpoint.pack import open_pack
        from jitengine_qwen3_6_35b_a3b_m4pro.moe.expert_store import WORKERS, ExpertStore

        path, meta, scan = open_pack(
            model_path,
            build=bool(self.options.get("build_expert_pack", False)),
            path=self.options.get("expert_pack_path"),
            log=logger.info,
        )
        self.expert_pack_path = path
        store = ExpertStore(
            path,
            meta["data_offset"],
            meta["record_bytes"],
            scan.num_records,
            workers=int(self.options.get("expert_io_workers", WORKERS)),
            nocache=bool(self.options.get("expert_io_nocache", True)),
            record_stride=int(meta.get("record_stride", meta["record_bytes"])),
        )
        factory = self.options.get("expert_store_factory")
        if factory is not None:
            store = factory(store)
        self.expert_store = store
        self.expert_meta = meta
        self.expert_pool = self._new_pool(self.plan.pool_slots)

    def _new_pool(self, slots: int):
        from jitengine_qwen3_6_35b_a3b_m4pro.moe.offload_cache import OffloadMoeCache

        pool = OffloadMoeCache(
            self.expert_store,
            self.scan.num_layers,
            self.scan.num_experts,
            int(slots),
            self.expert_meta["fields"],
            self.cfg.q_group_size,
            max_buffer_bytes=self.max_buffer_bytes or None,
        )
        seed = self.options.get("expert_pool_seed_per_layer")
        per_layer = int(slots) // self.scan.num_layers if seed is None else int(seed)
        if per_layer > 0:
            pool.prefill_seed(per_layer)
        return pool
