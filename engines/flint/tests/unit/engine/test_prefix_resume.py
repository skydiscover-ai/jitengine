"""The prefix cache resumes a prompt that starts with the last prompt's head.

An agent's next turn repeats the conversation so far, so its prompt starts with the last prompt. The engine keeps the
last prompt's rings at its last chunk boundary before the end (the head) and, for a prompt that starts with those
positions, restores them and prefills only from that boundary, running the chunks a cold prefill runs from there.
No chunk is computed here: model.prefill_chunk, model.prefill_layer_major and the history are recorded stand-ins;
tests/metal/engine/test_prefix_resume.py checks the resumed runs against cold ones on the real checkpoint.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from jitengine_flint.engine.engine import Engine
from jitengine_flint.kv_cache import History, Snapshot


@pytest.fixture
def chunks(monkeypatch):
    import mlx.core as mx

    from jitengine_flint.model import model as M

    events = []

    def fake_chunk(W, hist, toks, start, rows):
        events.append(("chunk", int(start), int(toks.shape[0])))
        return mx.zeros((len(rows), 4), dtype=mx.bfloat16)

    monkeypatch.setattr(M, "prefill_chunk", fake_chunk)
    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    with mx.stream(mx.cpu):
        yield events


def resident(events, length=0):
    hist = SimpleNamespace(length=length, valid=bool(length), state_arrays=lambda: [])
    hist.invalidate = lambda: (
        setattr(hist, "length", 0),
        setattr(hist, "valid", False),
        events.append(("inval",)),
    )
    hist.snapshot = lambda prompt, logits, length: (
        events.append(("head", length)),
        ("HEAD", len(prompt)),
    )[1]
    return SimpleNamespace(
        hist=hist, W=None, prefill_chunk=2048, kv_bits=16, memory=SimpleNamespace(mode="resident")
    )


def test_a_resumed_prefill_runs_the_cold_prefill_s_chunks_from_its_start(chunks):
    cold = resident(chunks)
    Engine._prefill(cold, list(range(6200)))
    assert chunks == [
        ("inval",),
        ("chunk", 0, 2048),
        ("chunk", 2048, 2048),
        ("chunk", 4096, 2048),
        ("chunk", 6144, 56),
    ]
    chunks.clear()
    warm = resident(chunks, length=4096)
    last, kept = Engine._prefill(warm, list(range(6200)), start=4096)
    # the history keeps positions 0..4095: nothing is invalidated, and the chunks are the cold run's from 4096
    assert chunks == [("chunk", 4096, 2048), ("chunk", 6144, 56)]
    assert (
        last.shape == (4,) and kept is None and warm.hist.length == 6200 and warm.hist.valid is True
    )


def test_the_head_is_the_rings_once_the_chunk_ending_at_head_at_is_done(chunks):
    eng = resident(chunks)
    _, kept = Engine._prefill(eng, list(range(6200)), head_at=6144)
    assert chunks[-2:] == [("head", 6144), ("chunk", 6144, 56)] and kept == ("HEAD", 6144)
    chunks.clear()
    _, kept = Engine._prefill(
        resident(chunks, length=4096), list(range(6200)), start=4096, head_at=6144
    )
    assert chunks == [("chunk", 4096, 2048), ("head", 6144), ("chunk", 6144, 56)] and kept == (
        "HEAD",
        6144,
    )


def test_offload_copies_each_sliding_ring_after_the_head_chunk_reaches_its_layer(monkeypatch):
    import mlx.core as mx

    from jitengine_flint.model import model as M

    events = []
    layers = [SimpleNamespace(index=i, sliding=i % 4 != 3) for i in range(8)]

    def fake_major(W, hist, tokens, chunks, check, after):
        events.append(("chunks", list(chunks), int(tokens.shape[0])))
        for lw in W.layers:
            for i in range(len(chunks)):
                events.append(("layer", lw.index, i))
                if after is not None:
                    after(lw, i)
        return [mx.zeros((1, c1 - c0, 4)) for c0, c1 in chunks]

    monkeypatch.setattr(M, "prefill_layer_major", fake_major)
    monkeypatch.setattr(M, "head", lambda W, h: mx.zeros((h.shape[0], 4), dtype=mx.bfloat16))
    hist = SimpleNamespace(length=4096, valid=True)
    hist.invalidate = lambda: events.append(("inval",))
    hist.ring_copy = lambda layer: (events.append(("ring", layer)), (f"k{layer}", f"v{layer}"))[1]
    hist.snapshot_rings = lambda prompt, rings: ("HEAD", len(prompt), dict(rings))
    eng = SimpleNamespace(
        hist=hist,
        W=SimpleNamespace(layers=layers),
        prefill_chunk=2048,
        memory=SimpleNamespace(mode="offload"),
        _annex_close=lambda: None,
        _annex_allowed=True,
    )
    with mx.stream(mx.cpu):
        last, kept = Engine._prefill_offload(
            eng, list(range(6200)), None, None, None, start=4096, head_at=6144
        )
    assert events[0] == (
        "chunks",
        [(4096, 6144), (6144, 6200)],
        6200,
    )  # the whole prompt, chunks from the start
    assert ("inval",) not in events
    # each sliding layer's ring is copied right after chunk 0 (ending at 6144) went through it, before chunk 1
    rings = [e for e in events if e[0] == "ring"]
    assert rings == [("ring", lw.index) for lw in layers if lw.sliding]
    for lw in layers:
        if lw.sliding:
            at = events.index(("ring", lw.index))
            assert events[at - 1] == ("layer", lw.index, 0) and events[at + 1] == (
                "layer",
                lw.index,
                1,
            )
    assert kept == (
        "HEAD",
        6144,
        {lw.index: (f"k{lw.index}", f"v{lw.index}") for lw in layers if lw.sliding},
    )
    assert last.shape == (4,) and hist.length == 6200 and hist.valid is True


class Cache:
    """An engine with the prefix cache and a recorded _prefill: what _prefill_or_restore restores and prefills."""

    def __init__(self, prefix_cache=True):
        self.prefix_cache = prefix_cache
        self.prefill_chunk = 2048
        self._snap = None
        self.calls = []
        self.hist = SimpleNamespace(rk=True)
        self.hist.restore = lambda snap: self.calls.append(("restore", snap.length))
        self.hist.snapshot = lambda prompt, logits: Snapshot(
            tuple(prompt), len(prompt), {}, {}, logits
        )

    def _prefill(self, prompt, cancel=None, start=0, head_at=0):
        self.calls.append(("prefill", len(prompt), start, head_at))
        head = Snapshot(tuple(prompt[:head_at]), head_at, {}, {}, None) if head_at else None
        return f"logits{len(prompt)}", head

    def run(self, prompt):
        self.calls.clear()
        return Engine._prefill_or_restore(self, prompt)


def test_an_agent_s_turns_resume_from_the_last_prompt_s_head():
    c = Cache()
    turn1 = list(range(4662))  # Claude Code's first request through jit-launch
    assert c.run(turn1) == ("logits4662", 0) and c.calls == [("prefill", 4662, 0, 4096)]
    assert c._snap.length == 4662 and c._snap.head.length == 4096
    turn2 = [*turn1, *range(9000, 9015)]  # the reply and the next message
    assert c.run(turn2) == ("logits4677", 4096)
    # the head is still the last boundary before the end: it is kept, and nothing new is copied
    assert (
        c.calls == [("restore", 4096), ("prefill", 4677, 4096, 0)] and c._snap.head.length == 4096
    )
    turn3 = [*turn2, *range(20000, 22000)]
    assert c.run(turn3) == ("logits6677", 4096)
    assert (
        c.calls == [("restore", 4096), ("prefill", 6677, 4096, 6144)]
        and c._snap.head.length == 6144
    )
    assert c.run(turn3) == ("logits6677", 6677) and c.calls == [
        ("restore", 6677)
    ]  # the same prompt again


def test_a_prompt_that_leaves_the_head_prefills_from_the_start():
    c = Cache()
    first = list(range(5000))
    c.run(first)
    changed = [*first[:4095], -1, *first[4096:], 7]  # differs inside the head
    assert c.run(changed) == ("logits5001", 0) and c.calls == [("prefill", 5001, 0, 4096)]
    assert c.run(list(range(4096))) == (
        "logits4096",
        0,
    )  # the head itself: no logits kept for its end
    assert c.calls == [("prefill", 4096, 0, 2048)]
    c.run(list(range(100)))  # one chunk: no head
    assert c.calls == [("prefill", 100, 0, 0)] and c._snap.head is None


def test_without_the_prefix_cache_nothing_is_kept():
    c = Cache(prefix_cache=False)
    assert c.run(list(range(5000))) == ("logits5000", 0) and c.calls == [("prefill", 5000, 0, 0)]
    assert c._snap is None


def test_history_ring_copies_make_the_same_snapshot_as_the_rings():
    import mlx.core as mx

    cfg = SimpleNamespace(
        sliding_window=8, n_kv_heads=4, head_dim=4, sliding_layers=[0, 2], full_layers=[1], bits=4
    )
    with mx.stream(mx.cpu):
        h = History(cfg, 64)
        h.reserve(32)
        rng = np.random.default_rng(0)
        for layer in cfg.sliding_layers:
            k = mx.array(rng.standard_normal((1, 4, 20, 4)), dtype=mx.bfloat16)
            h.write_ring(layer, 0, k, k + 1)
        h.length = 20
        whole = h.snapshot(list(range(20)), None)
        rings = {layer: h.ring_copy(layer) for layer in cfg.sliding_layers}
        for layer in cfg.sliding_layers:  # later writes do not reach the copies
            h.write_ring(
                layer,
                20,
                mx.ones((1, 4, 8, 4), dtype=mx.bfloat16),
                mx.ones((1, 4, 8, 4), dtype=mx.bfloat16),
            )
        built = h.snapshot_rings(list(range(20)), rings)
        assert (
            built.length == whole.length == 20
            and built.prompt == whole.prompt
            and built.logits is None
        )
        for layer in cfg.sliding_layers:
            assert mx.array_equal(built.rings_k[layer], whole.rings_k[layer]).item()
            assert mx.array_equal(built.rings_v[layer], whole.rings_v[layer]).item()
        h.restore(built)
        assert h.length == 20 and h.valid
        for layer in cfg.sliding_layers:
            assert mx.array_equal(h.rk[layer], whole.rings_k[layer]).item()


def test_an_engine_s_next_turn_reports_the_head_as_cached(fake_gpu, monkeypatch):
    """Through the engine on the fake GPU (the stub model's logits depend on the position only; stock MLX decode, as
    the fake megakernel serves no requests): the next turn's prompt starts with the last one's, so it restores the
    head and its usage counts the head's tokens as cached."""
    from jitengine_flint import kv_cache
    from jitengine_flint.engine.io_struct import GenRequest

    def snapshot(self, prompt, logits, length=None):
        return Snapshot(tuple(prompt), self.length if length is None else length, {}, {}, logits)

    def restore(self, snap):
        self.length, self.valid = snap.length, True

    def ring_copy(self, layer):
        return None, None

    for name in ("History", "MKHistory", "MKHistory8"):
        cls = getattr(kv_cache, name)
        monkeypatch.setattr(
            kv_cache,
            name,
            type(
                name,
                (cls,),
                {"rk": True, "snapshot": snapshot, "restore": restore, "ring_copy": ring_copy},
            ),
        )
    eng = fake_gpu.engine([], prefix_cache=True, prefill_chunk=256, decode_backend="graph")

    def turn(prompt):
        r = eng.generate(
            GenRequest(prompt_ids=prompt, max_tokens=3, temperature=0.0, ignore_eos=True)
        )
        assert r.token_ids == fake_gpu.stream_of(len(prompt), 3), r
        return r.cached_prompt_tokens

    first = list(range(1, 601))
    assert turn(first) == 0
    assert turn([*first, *range(700, 750)]) == 512  # 650 tokens: the head at 512 is kept
    assert turn([*first, *range(700, 750), *range(800, 1000)]) == 512  # 850: the next head, at 768
    assert turn([*first, *range(700, 750), *range(800, 1000), 5]) == 768
    assert turn([*first[:100], 9, *first[101:]]) == 0  # an earlier change: from the start
