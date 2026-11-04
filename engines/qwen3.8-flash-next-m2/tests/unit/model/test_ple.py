"""The per-layer hashed n-gram embedding (PLE), on MLX's CPU build: the n-gram index against an
independent integer reference, the streamed row store against the checkpoint's own tensors, the
layer against a float64 numpy transcription of its formula, and the engine serving a tiny
checkpoint that has one PLE layer (`tiny_flashnext_ple`; the default tiny checkpoint has none)."""

# ruff: noqa: E402
import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")  # the imports below need MLX, so E402 is waived for the file

from jitengine_qwen3_8_flash_next_m2.checkpoint.pack_format import PackError
from jitengine_qwen3_8_flash_next_m2.model import ple
from jitengine_qwen3_8_flash_next_m2.model.ple_store import PLEStore, open_ple_pack
from jitengine_qwen3_8_flash_next_m2.scheduler.faults import FaultSeam, RowFaultError

EOS = 248044
PLE_LAYER = 1  # the tiny checkpoint's PLE layer (`ple_layer_ids` are 1-based)
PREFIX = "language_model.model.layers.1.ple."


# ---------------------------------------------------------------- the hash, the primes, the multipliers
def test_splitmix64_is_the_reference_generator():
    assert ple.splitmix64(0) == 0xE220A8397B1DCDAF  # the first output of splitmix64 seeded 0
    assert ple.splitmix64(1) == 0x910A2DEC89025CC1
    assert all(0 <= ple.splitmix64(v) < 1 << 64 for v in (2**63, 2**64 - 1, 12345))


def test_nth_prime_after():
    assert ple.find_nth_prime_after(49, 1) == 53
    assert ple.find_nth_prime_after(49, 2) == 59
    assert ple.find_nth_prime_after(10, 0) == 10
    assert ple.find_nth_prime_after(1, 4) == 7
    assert [ple._is_prime(n) for n in (0, 1, 2, 3, 4, 9, 97)] == [
        False,
        False,
        True,
        True,
        False,
        False,
        True,
    ]


def test_layer_multipliers_are_odd_bounded_and_depend_on_the_layer_and_the_seed():
    m = ple.build_layer_multipliers(248320, 3, 0, 1234)
    assert len(m) == 3 and all(v % 2 == 1 and 0 < v <= (1 << 63) - 1 // 248320 for v in m)
    assert m == ple.build_layer_multipliers(248320, 3, 0, 1234)
    assert m != ple.build_layer_multipliers(248320, 3, 1, 1234)
    assert m != ple.build_layer_multipliers(248320, 3, 0, 1235)


def test_to_i64_folds_unsigned_words_into_the_signed_range():
    assert ple._to_i64([1, -2, (1 << 64) - 1, 1 << 63]) == [1, -2, -1, -(1 << 63)]
    assert ple._to_i64(np.array([5, 6], dtype=np.int64)) == [5, 6]


# ---------------------------------------------------------------- the n-gram index
def _index(**kw):
    args = dict(
        vocab_size=1000,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=50,
        make_divisible_by=128,
        seed=1234,
        eos_token_id=EOS,
    )
    args.update(kw)
    return ple.NGramIndex(**args)


def _wrap(x):
    return ((x + (1 << 63)) % (1 << 64)) - (1 << 63)


def _reference_rows(ix, history):
    """The row ids of every position after the first `context_len`, by plain Python integers: a
    position looks back `shift` tokens only within the stretch since the last EOS."""
    out = []
    for t in range(ix.context_len, len(history)):
        prev_eos = max((u for u in range(t) if history[u] == ix.eos_token_id), default=-1)
        in_segment = t - (prev_eos + 1)
        shifted = [
            history[t - s] if (in_segment >= s and t - s >= 0) else ix.eos_token_id
            for s in range(ix.ngram_size)
        ]
        row = []
        for n in range(2, ix.ngram_size + 1):
            mixed = _wrap(shifted[0] * ix.layer_multipliers[0])
            for k in range(1, n):
                mixed = mixed ^ _wrap(shifted[k] * ix.layer_multipliers[k])
            for h in range((n - 2) * ix.heads_per_ngram, (n - 1) * ix.heads_per_ngram):
                row.append(mixed % ix.head_vocab_sizes_l[h] + ix.head_offsets_l[h])
        out.append(row)
    return np.array(out, dtype=np.int64)


TOKENS = [5, 9, 5, 9, 17, EOS, 3, 3, 3, 40, EOS, EOS, 8]


def test_the_index_matches_the_integer_reference_on_the_mx_and_the_numpy_paths():
    ix = _index()
    ids = mx.array([TOKENS], dtype=mx.int32)
    got = np.array(ix(ids))[0]
    history = [EOS] * ix.context_len + TOKENS
    want = _reference_rows(ix, history)
    assert got.shape == (len(TOKENS), ix.ngram_heads) == want.shape
    assert (got == want).all()
    assert (ix.rows_np(np.array(history)) == want).all()
    assert (ix.rows_np(np.array(history[: ix.context_len + 1])) == want[:1]).all()


def test_every_head_stays_inside_its_own_slice_of_the_table():
    ix = _index()
    rows = np.array(ix(mx.array([TOKENS], dtype=mx.int32)))[0]
    for h in range(ix.ngram_heads):
        lo = ix.head_offsets_l[h]
        assert ((rows[:, h] >= lo) & (rows[:, h] < lo + ix.head_vocab_sizes_l[h])).all()
    assert ix.padded_vocab_size % 128 == 0 and ix.padded_vocab_size >= ix.total_vocab_size


def test_a_chunked_pass_with_the_carried_context_is_a_full_pass():
    ix = _index()
    ids = mx.array([TOKENS], dtype=mx.int32)
    whole = np.array(ix(ids))
    context, parts = None, []
    for a, b in ((0, 4), (4, 5), (5, len(TOKENS))):
        chunk = ids[:, a:b]
        parts.append(np.array(ix(chunk, context)))
        history = mx.concatenate(
            [ix.initial_context() if context is None else context, chunk.astype(mx.int64)], axis=-1
        )
        context = ple.NGramIndex.next_context(history, ix.context_len)
    assert (np.concatenate(parts, axis=1) == whole).all()


def test_batch_rows_are_independent():
    ix = _index()
    a, b = TOKENS[:6], TOKENS[6:12]
    both = np.array(ix(mx.array([a, b], dtype=mx.int32)))
    assert (both[0:1] == np.array(ix(mx.array([a], dtype=mx.int32)))).all()
    assert (both[1:2] == np.array(ix(mx.array([b], dtype=mx.int32)))).all()
    assert tuple(ix.initial_context(3).shape) == (3, 2)


def test_stored_buffers_are_checked_against_the_recomputation():
    ix = _index()
    good = dict(
        layer_multipliers=ix.computed_multipliers,
        head_offsets=ix.computed_offsets,
        head_vocab_sizes=ix.computed_sizes,
    )
    same = _index(**good)
    same.check_buffers()
    assert (same(mx.array([TOKENS])) == ix(mx.array([TOKENS]))).all().item()
    for key, bad in (
        ("layer_multipliers", [m + 2 for m in ix.computed_multipliers]),
        ("head_offsets", [o + 1 for o in ix.computed_offsets]),
        ("head_vocab_sizes", [s + 2 for s in ix.computed_sizes]),
    ):
        with pytest.raises(ValueError):
            _index(**{**good, key: bad}).check_buffers()
    with pytest.raises(ValueError, match="wrong length"):
        _index(layer_multipliers=[1, 3])


def _checkpoint_tensors(model_path):
    """Every tensor of the checkpoint under its name without the `language_model.` prefix."""
    out = {}
    for name in sorted(os.listdir(model_path)):
        if name.endswith(".safetensors"):
            out.update(mx.load(os.path.join(model_path, name)))
    return {k.removeprefix("language_model."): v for k, v in out.items()}


def test_the_index_is_built_from_a_config_and_the_checkpoints_buffers(tiny_flashnext_ple):
    from jitengine_qwen3_8_flash_next_m2.model.config import load_config

    model_path, _, _ = tiny_flashnext_ple()
    cfg = load_config(model_path)
    assert cfg.has_ple(PLE_LAYER) and not cfg.has_ple(0) and cfg.ngram_heads == 2
    from_cfg = ple.NGramIndex.from_config(cfg)
    assert from_cfg.total_vocab_size == 53 + 59 and from_cfg.padded_vocab_size == 128
    w = _checkpoint_tensors(model_path)
    stored = ple.NGramIndex.from_config(cfg, w, layer=PLE_LAYER)  # checks them on the way
    assert stored.layer_multipliers == from_cfg.layer_multipliers
    assert stored.head_offsets_l == from_cfg.head_offsets_l == [0, 53]
    assert stored.head_vocab_sizes_l == from_cfg.head_vocab_sizes_l == [53, 59]
    key = "model.layers.1.ple.ple_embedding.layer_multipliers"
    w[key] = w[key] + 2
    with pytest.raises(ValueError, match="layer_multipliers"):
        ple.NGramIndex.from_config(cfg, w, layer=PLE_LAYER)
    assert ple.NGramIndex.from_config(cfg, w, layer=PLE_LAYER, check=False).layer_multipliers != (
        from_cfg.layer_multipliers
    )


# ---------------------------------------------------------------- the row store
@pytest.fixture(scope="module")
def table(tiny_flashnext_ple):
    """The n-gram table as the checkpoint stores it, dequantised directly: [128, 32]."""
    w = _checkpoint_tensors(tiny_flashnext_ple()[0])
    shards = []
    for s in range(2):
        base = f"model.layers.1.ple.ple_embedding.ngram_embedding.shards.{s}."
        shards.append(
            mx.dequantize(
                w[base + "weight"], w[base + "scales"], w[base + "biases"], group_size=32, bits=4
            )
        )
    return np.array(mx.concatenate(shards, axis=0).astype(mx.float32))


@pytest.fixture
def store(tiny_flashnext_ple):
    model_path, _, ple_pack = tiny_flashnext_ple()
    path, meta, _scan = open_ple_pack(model_path, path=ple_pack, log=lambda *_: None)
    s = PLEStore(path, meta, workers=2, nocache=False)
    yield s
    s.close()


def test_the_store_returns_the_rows_a_direct_dequantisation_gives(store, table):
    ids = [0, 63, 64, 127, 5, 64, 0]  # both shards, both ends, repeats
    rows = store.fetch(ids)
    assert tuple(rows.shape) == (len(ids), store.values_per_row) == (7, 32)
    assert (np.array(rows.astype(mx.float32)) == table[ids]).all()


def test_the_stores_geometry_and_counters_follow_the_pack(store):
    assert (store.total_rows, store.rows_present, store.row_bytes) == (128, 128, 20)
    assert store.bits == 4 and store.group_size == 32 and store.nocache is False
    store.fetch([3, 3, 4])
    store.fetch([4])
    c = store.counters()
    assert (c["ple_calls"], c["ple_rows_requested"], c["ple_rows_read"]) == (
        2,
        4,
        3,
    )  # one call dedups
    assert c["ple_bytes"] == 3 * 20 and c["ple_page_bytes"] == 3 * 16384
    assert store.stats()["ple_values_per_row"] == 32 and store.stats()["ple_total_rows"] == 128


def test_raw_fetch_returns_the_checkpoints_words_scales_and_biases(store, tiny_flashnext_ple):
    w = _checkpoint_tensors(tiny_flashnext_ple()[0])
    base = "model.layers.1.ple.ple_embedding.ngram_embedding.shards.1."
    words, scales, biases = store.fetch_raw([64 + 7, 64 + 9])
    assert (words == np.array(w[base + "weight"])[[7, 9]]).all()
    assert (scales == np.array(w[base + "scales"].view(mx.uint16))[[7, 9]]).all()
    assert (biases == np.array(w[base + "biases"].view(mx.uint16))[[7, 9]]).all()


def test_a_row_outside_the_table_or_the_pack_is_refused(store):
    for bad in (-1, 128, 1000):
        with pytest.raises(PackError, match="out of range"):
            store.fetch([1, bad])
    store._seg_end = store._seg_end - 1  # a partial pack: the last row of the segment is absent
    with pytest.raises(PackError, match="is not in this pack"):
        store.offsets_of(np.array([127]))


def test_a_ple_row_fault_fires_before_anything_is_read_or_counted(store):
    store.seam = FaultSeam("ple_row", 70)
    store.owners = ["a", "b"]
    with pytest.raises(RowFaultError) as e:
        store.fetch([1, 70, 2], row_of=np.array([0, 1, 0]))
    assert e.value.rows == {1}
    assert store.counters()["ple_calls"] == 0 and store.counters()["ple_bytes"] == 0
    assert store.fetch([70]).shape == (1, 32)  # disarmed: the next read goes through
    assert store.counters()["ple_calls"] == 1


def test_a_missing_or_unusable_pack_is_refused(tiny_flashnext_ple, tmp_path):
    model_path, expert_pack, ple_pack = tiny_flashnext_ple()
    with pytest.raises(PackError, match="does not exist"):
        open_ple_pack(model_path, path=str(tmp_path / "absent.ple"))
    lines = []
    with pytest.raises(PackError, match="unusable"):
        open_ple_pack(model_path, path=expert_pack, log=lines.append)  # a pack of another kind
    assert lines and "rejecting" in lines[0]
    assert open_ple_pack(model_path, path=ple_pack, log=lines.append)[0] == ple_pack


def test_closing_the_store_closes_its_file(tiny_flashnext_ple):
    model_path, _, ple_pack = tiny_flashnext_ple()
    path, meta, _ = open_ple_pack(model_path, path=ple_pack)
    s = PLEStore(path, meta, workers=1, nocache=False)
    fd = s.fd
    s.close()
    assert s.fd == -1
    with pytest.raises(OSError):
        os.fstat(fd)
    s.close()  # a second close does nothing


# ---------------------------------------------------------------- the layer
@pytest.fixture(scope="module")
def engine(cpu_flashnext_engine):
    """The tiny engine with a PLE layer, on the stock ops; its PLE pack is read without F_NOCACHE."""
    return cpu_flashnext_engine(ple=True, prefix_store_entries=0)


@pytest.fixture
def layer(engine):
    return engine.model.layers[PLE_LAYER].ple


def _dense(lin):
    """A QLinear's dequantised weight as float64 numpy [out, in]."""
    w = mx.dequantize(lin.weight, lin.scales, lin.biases, group_size=lin.group_size, bits=lin.bits)
    return np.array(w.astype(mx.float32), dtype=np.float64)


def _h(seq, seed=0):
    return mx.random.normal((1, seq, 256), key=mx.random.key(seed))


def _gnorm(x, w1, hc, hid, eps):
    xg = x.reshape(*x.shape[:-1], hc, hid)
    xg = xg / np.sqrt((xg**2).mean(-1, keepdims=True) + eps)
    return (xg * w1.reshape(hc, hid)).reshape(x.shape)


def _reference_apply(layer, h, emb, state):
    """`Qwen4ExpTextPLELayer.apply` in float64 numpy from the dequantised weights."""
    hc, hid, eps = layer.hc, layer.hid, layer.eps
    h, emb, state = (np.array(a, dtype=np.float64) for a in (h, emb, state))
    w1 = {
        k: np.array(getattr(layer, f"norm_{k}_w1"), dtype=np.float64)
        for k in ("key", "query", "conv")
    }
    key = _gnorm(emb @ _dense(layer.key_proj).T, w1["key"], hc, hid, eps).reshape(
        *h.shape[:-1], hc, hid
    )
    value = emb @ _dense(layer.value_proj).T
    query = _gnorm(h, w1["query"], hc, hid, eps).reshape(*h.shape[:-1], hc, hid)
    gate = (key * query).sum(-1, keepdims=True) / np.sqrt(hid)
    gate = np.sqrt(np.maximum(np.abs(gate), 1e-6)) * np.sign(gate)
    gated = (1 / (1 + np.exp(-gate))) * value[..., None, :]
    flat = gated.reshape(*gated.shape[:-2], hc * hid)
    normed = _gnorm(flat, w1["conv"], hc, hid, eps)
    xt = np.concatenate([state, normed], axis=1)
    conv_w = np.array(layer.conv_w32, dtype=np.float64)
    acc = sum(
        xt[:, j * layer.dilation : j * layer.dilation + h.shape[1], :] * conv_w[:, j]
        for j in range(layer.conv_k)
    )
    return flat + acc / (1 + np.exp(-acc)), xt[:, -layer.state_len :, :]


def test_the_model_has_one_ple_layer_with_the_geometry_the_checkpoint_gives(engine, layer):
    assert engine.model.ple_layers == [PLE_LAYER]
    assert [bool(ly.ple) for ly in engine.model.layers] == [False, True, False, False]
    assert (layer.hc, layer.hid, layer.ple_embed, layer.values_per_row) == (4, 64, 64, 32)
    assert (layer.conv_k, layer.dilation, layer.state_len) == (4, 3, 9)
    state = layer.fresh_state()
    assert tuple(state.context.shape) == (1, 2) and tuple(state.conv.shape) == (1, 9, 256)
    assert (np.array(state.context) == EOS).all() and not np.array(state.conv).any()


def test_embed_gathers_the_rows_of_the_ngram_index_from_the_store(layer, table):
    ids = mx.array([[5, 9, 5, 17]], dtype=mx.int32)
    emb, row_ids = layer.embed(ids, layer.fresh_state().context)
    rows = np.array(row_ids)[0]  # [4, 2]
    assert tuple(emb.shape) == (1, 4, 64) and rows.shape == (4, 2)
    want = table[rows.reshape(-1)].reshape(1, 4, 64)
    assert np.array(emb) == pytest.approx(want, abs=0)
    assert (rows == layer.ngram.rows_np(np.array([EOS, EOS, 5, 9, 5, 17]))).all()


def test_apply_equals_the_float64_transcription_of_the_formula(layer):
    emb = mx.random.normal((1, 5, 64), key=mx.random.key(1))
    state = mx.random.normal((1, 9, 256), key=mx.random.key(2))
    h = _h(5, 3)
    out, new_state = layer.apply(h, emb, state)
    want_out, want_state = _reference_apply(layer, h, emb, state)
    assert np.array(out, dtype=np.float64) == pytest.approx(want_out, abs=1e-4, rel=1e-4)
    assert np.array(new_state, dtype=np.float64) == pytest.approx(want_state, abs=1e-6)
    assert float(np.abs(want_out).max()) > 0.1  # a real signal, not zeros agreeing


def test_a_multi_token_pass_and_token_by_token_calls_agree_and_end_in_one_state(layer):
    ids = mx.array([[5, 9, 5, 17, EOS, 3, 40]], dtype=mx.int32)
    h = _h(7, 4)
    whole_state = layer.fresh_state()
    whole = layer(h, ids, whole_state)
    step_state, outs = layer.fresh_state(), []
    for i in range(7):
        outs.append(layer(h[:, i : i + 1], ids[:, i : i + 1], step_state))
    assert np.array(mx.concatenate(outs, axis=1)) == pytest.approx(np.array(whole), abs=2e-4)
    assert (np.array(step_state.context) == np.array(whole_state.context)).all()
    assert np.array(step_state.conv) == pytest.approx(np.array(whole_state.conv), abs=2e-4)
    assert np.array(whole_state.context).tolist() == [[3, 40]]  # the last two ids


def test_a_trace_records_the_rows_and_the_embedding(layer):
    trace = {}
    ids = mx.array([[5, 9]], dtype=mx.int32)
    layer(_h(2), ids, layer.fresh_state(), trace=trace)
    assert tuple(trace["ple_row_ids"].shape) == (1, 2, 2) and tuple(trace["ple_emb"].shape) == (
        1,
        2,
        64,
    )


def test_the_single_token_path_reuses_the_rows_of_an_undone_build(layer):
    ids, h = mx.array([[17]], dtype=mx.int32), _h(1, 5)
    state = layer.fresh_state()
    first = layer(h, ids, state)
    assert state.cached is not None
    # undo the call as the speculative layer builds do, then rebuild: no second read
    state.context, state.conv = layer.fresh_state().context, layer.fresh_state().conv
    reads = layer.store.calls
    again = layer(h, ids, state)
    assert layer.store.calls == reads
    assert np.array(again) == pytest.approx(np.array(first), abs=0)


def test_step_gives_each_row_what_a_lone_call_gives(layer):
    ids = mx.array([[5], [9], [17]], dtype=mx.int32)
    h = mx.random.normal((3, 1, 256), key=mx.random.key(6))
    warm = [layer.fresh_state() for _ in range(3)]
    for i, st in enumerate(warm):  # give each row a different history first
        layer(_h(3, i), mx.array([[3 + i, 4 + i, 5 + i]], dtype=mx.int32), st)
    solo_states = [ple.PLEState(st.context, st.conv) for st in warm]
    batched = layer.step(h, ids, warm)
    for i in range(3):
        solo = layer(h[i : i + 1], ids[i : i + 1], solo_states[i])
        assert np.array(batched[i : i + 1]) == pytest.approx(np.array(solo), abs=1e-5)
        assert (np.array(warm[i].context) == np.array(solo_states[i].context)).all()
        assert np.array(warm[i].conv) == pytest.approx(np.array(solo_states[i].conv), abs=1e-5)


def test_a_layer_whose_rows_do_not_fill_the_embedding_is_refused(engine, tiny_flashnext_ple):
    class Narrow:
        values_per_row = 16

    model = engine.model
    with pytest.raises(ValueError, match="ple_embed_dim"):
        ple.Qwen4ExpTextPLELayer(
            model.cfg,
            _checkpoint_tensors(tiny_flashnext_ple()[0]),
            "model.layers.1.ple.",
            Narrow(),
            model.layers[1].ple.ngram,
        )


# ---------------------------------------------------------------- the engine
P = list(range(10, 34))


def _req(request_id, shift=0, n=12):
    from jitengine_qwen3_8_flash_next_m2.engine.io_struct import GenRequest

    return GenRequest(
        prompt_ids=[t + shift for t in P[:n]],
        max_tokens=3,
        request_id=request_id,
        stop_at_eos=False,
    )


def test_the_engine_opens_the_ple_tier_and_serves_through_it(engine):
    before = engine.stats()
    out = engine.generate(_req("a"))
    assert out.finish_reason == "length" and len(out.token_ids) == 3
    after = engine.stats()
    assert after["ple_rows_read"] > before["ple_rows_read"] and after["ple_total_rows"] == 128
    assert after["ple_bytes"] == after["ple_rows_read"] * after["ple_row_bytes"]
    assert after["ple_page_bytes"] == after["ple_rows_read"] * 16384
    assert after["ple_nocache"] is False and after["ple_path"].endswith("tiny.ple")
    assert engine.generate(_req("a")).token_ids == out.token_ids  # deterministic


def test_a_model_without_ple_has_no_ple_tier(cpu_flashnext_engine):
    assert cpu_flashnext_engine().ple_store is None


def test_the_ple_layer_changes_what_the_model_computes(engine, cpu_flashnext_engine):
    """The same prompt through the model with PLE and the model without: different tokens."""
    no_ple = cpu_flashnext_engine()
    assert no_ple.generate(_req("x")).token_ids != engine.generate(_req("x")).token_ids


def test_score_runs_through_the_ple_layer_and_matches_a_direct_forward(engine):
    from jitengine_qwen3_8_flash_next_m2.kvcache.state import RequestState

    seq = [3, 4, 5, 6, 7]
    entries = engine.score(seq[:3], seq[3:], top_k=3)
    state = RequestState(engine.cfg, len(seq), 0, reserve=0, dtype=engine.compute_dtype)
    hs = []
    for a in ((0, 4), (4, 5)):
        toks = mx.array(seq[a[0] : a[1]], dtype=mx.int32)[None]
        hs.append(engine.model.forward(toks, state, logits="hidden")[0])
    logits = engine.model.head(mx.concatenate(hs, axis=0)).astype(mx.float32)
    lp = np.array(logits - mx.logsumexp(logits, axis=-1, keepdims=True))
    assert [e["logprob"] for e in entries[:-1]] == pytest.approx(
        [lp[i, seq[i + 1]] for i in range(4)], abs=2e-3
    )
    assert set(state.ple) == {PLE_LAYER}


def _arm(engine, request_id, row):
    seam = FaultSeam("ple_row", row, request_id)
    engine.fault_seam, engine.ple_store.seam = seam, seam
    return seam


def _disarm(engine):
    engine.fault_seam, engine.ple_store.seam = None, None


def _row_of(engine, req, position):
    """A table row the request's prompt reads at `position`: that position's first head. The last
    position is not prefilled: it is the first decode step's token."""
    ids = mx.array([list(req.prompt_ids)], dtype=mx.int32)
    return int(np.array(engine.model.layers[PLE_LAYER].ple.ngram(ids))[0, position, 0])


@pytest.mark.parametrize(
    "position, phase",
    [(5, "the prefill wave"), (-1, "the first decode step")],
)
def test_a_ple_row_fault_fails_that_request_and_rewinds_the_ple_state_of_the_others(
    engine, position, phase
):
    """In a prefill wave the rows before the faulted one have advanced their PLE context and conv
    when it fails, and the wave is rewound and re-run without it; in a decode step nothing has
    advanced yet. Either way the survivors equal a wave of the two of them that never saw a fault."""
    bad = _req("bad", 1)
    seam = _arm(engine, "bad", _row_of(engine, bad, position))
    try:
        results = engine.generate_batch([_req("good"), bad, _req("other", 2)])
    finally:
        _disarm(engine)
    assert seam.fired == 1, phase
    assert results[1].finish_reason == "error" and "ple_row" in results[1].extra["error"]
    clean = engine.generate_batch([_req("good"), _req("other", 2)])
    assert [results[0].token_ids, results[2].token_ids] == [r.token_ids for r in clean]
    assert engine.stats()["row_faults"] >= 1 and engine.health()["status"] == "ok"


def test_a_ple_row_fault_on_a_lone_row_fails_it_and_the_engine_serves_on(engine):
    bad = _req("lone", 3)
    seam = _arm(engine, "lone", _row_of(engine, bad, -1))
    try:
        with pytest.raises(RowFaultError, match="ple_row"):
            engine.generate(bad)
    finally:
        _disarm(engine)
    assert seam.fired == 1 and engine.health()["status"] == "ok"
    assert len(engine.generate(bad).token_ids) == 3
