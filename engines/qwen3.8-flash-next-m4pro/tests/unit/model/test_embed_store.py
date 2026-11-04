"""The streamed token embedding (`model.embed_store`) on the tiny checkpoint, on MLX's CPU build or
Metal: its lookup against the resident `QEmbedding.lookup` byte for byte, through the row cache,
past evictions and with no cache; its read and page counters; the plan that leaves the table out of
the dense weights; and an engine that serves with it."""

# ruff: noqa: E402
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")  # the imports below need MLX, so E402 is waived for the file

from jitengine_qwen3_8_flash_next_m4pro.checkpoint.expert_reader import CheckpointScan
from jitengine_qwen3_8_flash_next_m4pro.checkpoint.loader import load_language_weights
from jitengine_qwen3_8_flash_next_m4pro.checkpoint.pack_format import PAGE
from jitengine_qwen3_8_flash_next_m4pro.linear import QEmbedding
from jitengine_qwen3_8_flash_next_m4pro.model import embed_store as es
from jitengine_qwen3_8_flash_next_m4pro.model.config import load_config

BITS, GS = 4, 32  # the tiny checkpoint's quantization of embed_tokens


def _bytes(a: mx.array) -> bytes:
    return np.asarray(a.astype(mx.float32)).tobytes()


@pytest.fixture
def tables(tiny_flashnext, requires_mlx_any):
    """(resident QEmbedding, scan, model path) of the tiny checkpoint."""
    model, _pack = tiny_flashnext()
    w = load_language_weights(model, skip_experts=True)
    bits, gs = load_config(model).quant_of("model.embed_tokens")
    assert (bits, gs) == (BITS, GS)
    res = QEmbedding(
        w["model.embed_tokens.weight"],
        w["model.embed_tokens.scales"],
        w.get("model.embed_tokens.biases"),
        gs,
        bits,
    )
    return res, CheckpointScan(model), model


def _store(scan, **kw):
    kw.setdefault("nocache", False)  # F_NOCACHE is a macOS flag; the bytes are the same
    return es.StreamedEmbedding(scan, BITS, GS, **kw)


ID_SETS = [
    [[0]],
    [[248319]],  # the last row
    [[5, 5, 17, 248000, 5]],  # repeats inside one call
    [[11], [12], [11]],  # [B, 1], as a batched decode step passes them
    np.random.default_rng(0).integers(0, 248320, size=(1, 300)).tolist(),
]


@pytest.mark.parametrize("cache_rows", [8192, 4, 0])
def test_a_streamed_lookup_is_the_resident_lookup_byte_for_byte(tables, cache_rows):
    """Every id set, twice (a cold call, then the cache's): equal shapes, dtypes and every bit, in
    float32 and in bfloat16. With 4 rows the 300-id call evicts what it inserts; with 0 every row is
    read from the shard."""
    res, scan, _ = tables
    st = _store(scan, cache_rows=cache_rows)
    for ids in ID_SETS * 2:
        a = mx.array(ids, dtype=mx.int32)
        want, got = res.lookup(a), st.lookup(a)
        assert got.shape == want.shape and got.dtype == want.dtype, (got.shape, want.shape)
        for dt in (mx.float32, mx.bfloat16):
            assert _bytes(got.astype(dt)) == _bytes(want.astype(dt)), (ids, dt)
    st.close()


def test_the_negative_control_is_caught(tables):
    """A lookup one row off differs: the comparison above can fail."""
    res, scan, _ = tables
    st = _store(scan)
    a = mx.array([[100, 200, 300]], dtype=mx.int32)
    assert _bytes(st.lookup(a + 1)) != _bytes(res.lookup(a))
    st.close()


def test_a_lazy_id_is_materialized_first(tables):
    """The solo pipeline passes the in-flight sampled token and the MTP draft the verified one."""
    res, scan, _ = tables
    st = _store(scan)
    lazy = mx.argmax(mx.array([[0.0, 3.0, 1.0]]), axis=-1).reshape(1, 1) + 4000
    assert _bytes(st.lookup(lazy)) == _bytes(res.lookup(mx.array([[4001]], dtype=mx.int32)))
    st.close()


def test_the_counters_name_the_reads_and_the_pages(tables):
    _res, scan, _ = tables
    st = _store(scan, cache_rows=16)
    st.lookup(mx.array([[3, 3, 9]], dtype=mx.int32))
    s = st.stats()
    assert (s["embed_rows_requested"], s["embed_rows_read"], s["embed_cache_hits"]) == (3, 2, 0)
    assert s["embed_bytes"] == 2 * st.row_bytes == 2 * s["embed_row_bytes"]
    assert s["embed_page_bytes"] % PAGE == 0
    assert 2 * len(st.fields) * PAGE <= s["embed_page_bytes"] <= 4 * len(st.fields) * PAGE
    st.lookup(mx.array([[9, 4]], dtype=mx.int32))
    s = st.stats()
    assert (s["embed_rows_read"], s["embed_cache_hits"], s["embed_cache_live"]) == (3, 1, 3)
    assert s["embed_table_bytes"] == es.embed_streamed_bytes(scan, {})
    st.close()


def test_an_id_outside_the_table_raises(tables):
    _res, scan, _ = tables
    st = _store(scan)
    with pytest.raises(IndexError, match="outside the embedding table"):
        st.lookup(mx.array([[248320]], dtype=mx.int32))
    st.close()


def test_the_plan_streams_the_table_unless_told_not_to(tables):
    """The plan leaves the table out of the dense weights and charges the row cache; "off" is the
    resident plan, and the pool gains what the table held less the cache."""
    from jitengine_qwen3_8_flash_next_m4pro.budget import plan_budget

    _res, scan, model = tables
    cfg = load_config(model)
    opts = {"compute_dtype": "float32", "wave_tokens": 256}
    on = plan_budget(2 * 10**9, scan, cfg, dict(opts))
    off = plan_budget(2 * 10**9, scan, cfg, dict(opts, embed_stream="off"))
    table = es.embed_streamed_bytes(scan, {})
    assert table == 248320 * es.embed_row_bytes(scan) > 0
    assert (on.embed_streamed_bytes, on.embed_cache_rows) == (table, es.CACHE_ROWS)
    assert on.embed_cache_bytes == es.CACHE_ROWS * (
        es.embed_row_bytes(scan) + es.CACHE_ROW_OVERHEAD
    )
    assert on.dense_bytes == off.dense_bytes - table
    assert (off.embed_streamed_bytes, off.embed_cache_bytes) == (0, 0)
    assert on.pool_slots >= off.pool_slots
    names = [n for n, _ in on.as_dict()["buckets"]]
    assert any(n.startswith("token-embedding row cache") for n in names)
    assert not any(n.startswith("token-embedding") for n, _ in off.as_dict()["buckets"])
    assert sum(b for _, b in on.buckets()) <= on.planned_bytes <= on.budget_bytes
    with pytest.raises(ValueError, match="embed_stream"):
        plan_budget(2 * 10**9, scan, cfg, dict(opts, embed_stream="maybe"))


def test_an_engine_serves_with_the_table_streamed(cpu_flashnext_engine):
    """Tokens equal to the resident table's engine, the table never loaded, the rows counted. The
    request runs twice (no prefix store): the first reads its rows from the shard, the second takes
    every prompt and decode row from the LRU."""
    from jitengine_qwen3_8_flash_next_m4pro.engine.io_struct import GenRequest

    req = GenRequest(prompt_ids=list(range(1000, 1040)), max_tokens=12, stop_at_eos=False)
    on = cpu_flashnext_engine(prefix_store_entries=0)
    off = cpu_flashnext_engine(prefix_store_entries=0, embed_stream="off")
    assert isinstance(on.model.embed, es.StreamedEmbedding)
    assert not isinstance(off.model.embed, es.StreamedEmbedding)
    want = off.generate(req).token_ids
    assert on.generate(req).token_ids == want
    read = on.stats()["embed_rows_read"]
    assert on.generate(req).token_ids == want
    s = on.stats()
    assert s["embed_rows_read"] == read and s["embed_cache_hits"] >= 40 + 11
    assert s["embed_streamed"] and s["embed_rows_requested"] >= 2 * (40 + 11)
    assert s["memory_plan"]["embed_streamed_bytes"] == s["embed_table_bytes"]
    assert off.stats()["embed_streamed"] is False
