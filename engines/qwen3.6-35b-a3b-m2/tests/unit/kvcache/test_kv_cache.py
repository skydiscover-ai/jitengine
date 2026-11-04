"""Per-request state: what a KV slot refuses."""

import pytest


def test_a_restore_into_a_used_slot_is_an_invariant_error(requires_mlx_any):
    """`KVSlot.load` restores a prefix into a fresh slot only; anything else is an engine bug."""
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m2.engine.errors import InvariantError
    from jitengine_qwen3_6_35b_a3b_m2.kvcache.kv_cache import KVSlot
    from jitengine_qwen3_6_35b_a3b_m2.model.config import Qwen3_5MoeConfig

    cfg = Qwen3_5MoeConfig()
    slot = KVSlot(cfg, 512)
    prefix = mx.zeros((1, cfg.num_key_value_heads, 8, cfg.head_dim), dtype=mx.bfloat16)
    slot.load(prefix, prefix)
    assert slot.offset == 8
    with pytest.raises(InvariantError, match="fresh slot"):
        slot.load(prefix, prefix)
