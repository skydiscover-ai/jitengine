"""A fault rollback leaves no state from the undone unit, including the QSA block-key cache."""

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")

from jitengine_qwen3_8_flash_next_m4pro.kvcache.kv_cache import KVSlot
from jitengine_qwen3_8_flash_next_m4pro.scheduler.faults import StepGuard


def test_restore_rewinds_offsets_and_empties_the_block_key_cache():
    slot = KVSlot(SimpleNamespace(num_key_value_heads=1, head_dim=4, indexer_head_dim=4), 256)
    slot.offset = 64
    slot.idx_bk[0] = (1, "block keys built before the unit")
    st = SimpleNamespace(layers=[slot], ple={}, tokens=[1, 2])
    guard = StepGuard([st])
    st.tokens, slot.offset = [1, 2, 3], 192  # the unit advances the row ...
    slot.idx_bk[0] = (3, "block keys of the undone unit")
    guard.restore()  # ... and a RowFaultError undoes it
    assert (st.tokens, slot.offset, slot.idx_bk) == ([1, 2], 64, {})
