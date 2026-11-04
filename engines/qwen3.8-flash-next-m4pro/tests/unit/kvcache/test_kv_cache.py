"""Per-request state: what a KV slot refuses."""

import pytest


def test_a_restore_into_a_used_slot_is_an_invariant_error(requires_mlx_any):
    """`KVSlot.load` restores a prefix into a fresh slot only; anything else is an engine bug."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.engine.errors import InvariantError
    from jitengine_qwen3_8_flash_next_m4pro.kvcache.kv_cache import KVSlot
    from jitengine_qwen3_8_flash_next_m4pro.model.config import Qwen4ExpTextConfig

    cfg = Qwen4ExpTextConfig()
    slot = KVSlot(cfg, 512)
    kv = mx.zeros((1, cfg.num_key_value_heads, 8, cfg.head_dim), dtype=mx.bfloat16)
    idx = mx.zeros((1, 1, 8, cfg.indexer_head_dim), dtype=mx.bfloat16)
    slot.load(kv, kv, idx)
    assert slot.offset == 8
    with pytest.raises(InvariantError, match="fresh slot"):
        slot.load(kv, kv, idx)
