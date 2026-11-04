"""The batched-wave path of `Qwen4ExpTextSparseMoeBlock`, as a mixin.

A wave runs N independent prefill rows through a layer with one expert fetch for the
union of their experts (`Qwen4ExpTextDecoderLayer.wave`, `OffloadMoeCache.segments_multi`); each row
then applies its experts from the slots that fetch resolved (`apply`), with the block's `route`
and `shared`.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m4pro.engine.errors import InvariantError


class BatchedWavesMixin:
    """One row of a wave, its slots already resolved by the wave's fetch: `apply`."""

    def apply(
        self,
        x: mx.array,
        inds: mx.array,
        scores: mx.array,
        slots: np.ndarray,
        telem=None,
        shared: mx.array | None = None,
    ) -> mx.array:
        """Expert half of `__call__` for one row, with slots already resolved by the wave's fetch.

        `shared` is this row's prebuilt shared expert (the same graph `self.shared` would build).
        """
        if x.shape[0] != 1:
            raise InvariantError("a wave carries exactly one row per call (BI-4)")
        xe = mx.expand_dims(x, (-2, -3)).flatten(0, -3)
        experts = (
            np.asarray(inds).astype(np.int64, copy=False).reshape(-1)
        )  # already synced by the wave
        y = (
            self._gather_slots(xe, self.top_k, slots, inds.shape, experts=experts).squeeze(-2)
            * scores[..., None]
        ).sum(axis=-2)
        if telem is not None and telem.collect:
            telem.note(int(slots.size), int(np.unique(slots).size))
        return y + (self.shared(x) if shared is None else shared)
