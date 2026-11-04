"""A fake GPU for the model-free engine tests: the real Engine, its load check and doctor, on stubs.

`fake_gpu` builds engines from tests/data's checkpoint headers (no tokenizer, no weights) on a stand-in megakernel and a
stub model, all on the CPU. The device is the tuned 48 GB M4 Pro unless a test simulates another (the engine's own sim
hook): the real Mac's facts never reach these tests.

The stub model's logits at position p select token 100 + 37 p mod 5000 (bf16 8.0, every other logit 0). The fake
megakernel (FakeMK) answers every pass with that distribution's result words (top-1, top-20 and log-probabilities, as
loadcheck.ref_rows computes them), records each kernel object built (`built`) and each probe-stage dispatch
(`dispatched`), refuses the grids in `refuse` at its residency probe and fails warm_verify when `verify_fails`. A
descriptor table other than the one built, or a history whose full-layer rows were zeroed (its `stale` flag), moves the
distribution to position p + 7: the fake's stand-in for wrong weights or a stale history.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

M4_PRO = {  # the measured Mac's GPU, its wired limit not raised
    "architecture": "applegpu_g16s",
    "device_name": "Apple M4 Pro",
    "memory_size": 48 * 2**30,
    "max_recommended_working_set_size": 40_200_000_000,
    "max_buffer_length": 30_150_672_384,
}


def stream_of(n_prompt: int, k: int) -> list:
    """The stub model's greedy stream after a prompt of n_prompt tokens."""
    return [100 + (37 * (n_prompt - 1 + j)) % 5000 for j in range(k)]


class StubTokenizer:
    """One id per character (ids 256..4351), enough for the load check's fixed text and doctor's smoke prompt."""

    def __init__(self, model_path=None):
        pass

    def encode(self, text: str) -> list:
        return [256 + ord(c) % 4096 for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(32 + int(t) % 95) for t in ids)


class FakeHist:
    """The History surface the graph backend's worker, prefill and decode loop touch (no arrays)."""

    rk = None

    def __init__(self, *a):
        self.length, self.valid, self.reserved = 0, False, 0

    def reserve(self, n):
        self.reserved = max(self.reserved, n)

    def invalidate(self):
        self.valid = False

    def state_arrays(self):
        return []

    def ring_mask(self, p):
        return None

    def release(self):
        pass


class FakeMKHist(FakeHist):
    """The megakernel history's load-check surface: its arrays are itself (so the fake kernel sees `stale`, set when
    full-layer rows are zeroed and cleared by invalidate), ring snapshots are tokens."""

    capacity = 4096
    released = 0

    def __init__(self, cfg=None, limit=4096):
        super().__init__()
        self.limit, self.stale = limit, False

    @property
    def arrays(self):
        return [self]

    def invalidate(self):
        self.valid, self.stale = False, False

    def write_full(self, layer, start, k, v):
        self.stale = True

    def snapshot(self, prompt, logits):
        return ("snap", tuple(prompt))

    def restore(self, snap):
        assert snap[0] == "snap"
        self.valid = True

    def release(self):
        FakeMKHist.released += 1


def stub_row(p: int, vocab: int):
    """The stub model's logits at position p, as float32."""
    import numpy as np

    row = np.zeros(vocab, dtype=np.float32)
    row[100 + (37 * p) % 5000] = 8.0
    return row


class FakeMK:
    """runner.MegaKernel's load-check surface on a fake GPU (see the module docstring)."""

    refuse: set = set()  # noqa: RUF012  (class-level state the tests reset)
    verify_fails = False
    built: list = []  # noqa: RUF012
    dispatched: list = []  # noqa: RUF012
    _rows: dict = {}  # noqa: RUF012

    def __init__(self, W, G=40, log=print, kv8=False, **k):
        import mlx.core as mx
        import numpy as np

        from jitengine_flint_m4pro.kernels import plan as P

        self.G, self.kv8, self.vrows, self.nbar = G, kv8, (), 150
        self.cfg = W.cfg
        self.desc = mx.array(P.pack_descriptors(W.plan))
        self._desc0 = np.array(self.desc)
        self.scratch = None
        FakeMK.built.append(G)

    def _words(self, p: int, token: int, hist_arrays):
        import numpy as np

        from jitengine_flint_m4pro import loadcheck
        from jitengine_flint_m4pro.kernels import plan as P

        stale = bool(hist_arrays) and getattr(hist_arrays[0], "stale", False)
        q = p + 7 if (stale or not np.array_equal(np.array(self.desc), self._desc0)) else p
        if q not in FakeMK._rows:
            row = stub_row(q, self.cfg.vocab)
            FakeMK._rows[q] = (row, loadcheck.ref_rows(row[None])[0])
        row, ref = FakeMK._rows[q]
        w = np.zeros(P.RES_WORDS, dtype=np.uint32)
        w[P.R_TOKEN] = w[P.R_ARGMAX] = ref["argmax"]
        w[P.R_FINITE] = w[P.R_DONE] = 1
        w[P.R_TOPI : P.R_TOPI + P.TOPN] = ref["ids"]
        w[P.R_TOPV : P.R_TOPV + P.TOPN] = ref["lp"].astype(np.float32).view(np.uint32)
        w[48] = token
        return w, row

    def _write_scratch(self, row, p: int, token: int) -> None:
        import mlx.core as mx
        import numpy as np

        from jitengine_flint_m4pro.kernels import plan as P

        lg = mx.array(row).astype(mx.bfloat16).view(mx.uint8)
        o = P.OFF["LOGITS"]
        self.scratch[o : o + lg.size] = lg
        h = P.OFF["HS"]
        self.scratch[h : h + 8] = mx.array(np.array([p, token], dtype=np.uint32).view(np.uint8))

    def run_pass(self, hist_arrays, prm, prev=None, dep=None):
        import mlx.core as mx

        from jitengine_flint_m4pro.kernels import plan as P

        if int(prm[P.P_PROBE]):
            FakeMK.dispatched.append(self.G)
            return mx.zeros((1,))
        p, token = int(prm[P.P_POS]), int(prm[P.P_TOKEN])
        w, row = self._words(p, token, hist_arrays)
        self._write_scratch(row, p, token)
        return mx.array(w)

    run_pass_chain = run_pass

    def run_vpass(self, hist_arrays, prm, prev=None):
        import mlx.core as mx
        import numpy as np

        from jitengine_flint_m4pro.kernels import plan as P

        M, p = int(prm[P.P_ROWS]), int(prm[P.P_POS])
        out = np.zeros(P.VRES_WORDS, dtype=np.uint32)
        for m in range(M):
            w = self._words(p + m, int(prm[P.P_RTOK + m]), hist_arrays)[0]
            out[m * P.VROW_WORDS : (m + 1) * P.VROW_WORDS] = w[: P.VROW_WORDS]
        return mx.array(out)

    def probe(self, hist_arrays, att, cap):
        from jitengine_flint_m4pro.kernels.runner import BarrierProbeError

        if self.G in FakeMK.refuse:
            raise BarrierProbeError(f"megakernel residency probe failed at G={self.G}: (fake GPU)")
        return {"bar_ms": 0.2, "ring_ms": 0.3, "real_ms": 7.0}

    def warm_verify(self, hist_arrays, att, cap):
        from jitengine_flint_m4pro.kernels.runner import BarrierProbeError

        if FakeMK.verify_fails:
            raise BarrierProbeError("verify pass M=2: (fake GPU)")
        self.vrows = tuple(range(2, 9))
        return dict.fromkeys(self.vrows, 10.0)

    @classmethod
    def reset(cls, refuse=(), verify_fails=False):
        cls.refuse, cls.verify_fails, cls.built, cls.dispatched = set(refuse), verify_fails, [], []
        FakeMKHist.released = 0


class FakePool:
    """Offload mode's expert pool, without arrays (the stub model never gathers an expert)."""

    def __init__(self, cfg, index, slots, log=print):
        self.slots, self.experts = slots, ()

    def selfcheck(self):
        pass

    def stats(self):
        return {"slots": self.slots, "hit_rate": None}

    def close(self):
        self.experts = None


@pytest.fixture
def fake_gpu(flint_headers, monkeypatch, tmp_path):  # noqa: PLR0915  (one stub per seam)
    """Stubs the GPU, the model, the tokenizer and the checkpoint's digest check (see the module docstring). Returns a
    namespace: engine(logs, sim=None, **options) builds and starts an Engine and waits for its load; MK, Hist; model
    (the headers directory); grid_cache (a fresh FLINT_GRID_CACHE path); stream_of. Every engine built is closed at
    teardown."""
    import mlx.core as mx

    from jitengine_flint_m4pro import device, expert_pool, kv_cache, loadcheck, loader, tokenizer
    from jitengine_flint_m4pro.engine import engine as E
    from jitengine_flint_m4pro.kernels import runner
    from jitengine_flint_m4pro.model import model as graph
    from jitengine_flint_m4pro.model.config import load_config
    from jitengine_flint_m4pro.weight_banks import BankPlan

    cfg = load_config(flint_headers)
    with open(os.path.join(flint_headers, "flint-int4.json")) as f:
        manifest = json.load(f)
    ids = mx.arange(cfg.vocab, stream=mx.cpu)
    mx.eval(ids)  # concrete here: a lazy array cannot be evaluated by the worker

    def logits(p):
        return mx.where(ids == 100 + (37 * p) % 5000, 8.0, 0.0).astype(mx.bfloat16)

    def decode_step(W, hist, token, p, ring, trace=None):
        return logits(p)

    def prefill_chunk(W, hist, toks, start, rows=None):
        return mx.stack([logits(start + r) for r in rows])

    def load_weights(cfg, index, log=None, experts=True):
        return SimpleNamespace(cfg=cfg, plan=BankPlan(cfg, index), experts=experts, pool=None)

    monkeypatch.setattr(expert_pool, "ExpertPool", FakePool)

    monkeypatch.setattr(device, "mlx_device_info", lambda: dict(M4_PRO))
    # The engine and doctor wire the fake GPU's working set (M4_PRO's 40.2 GB). MLX on a Mac whose real working set is
    # smaller (a 24 GB M2) refuses that, so the call goes to a stub: the fake GPU's limit never reaches the real one.
    monkeypatch.setattr(mx, "set_wired_limit", lambda limit: 0)
    monkeypatch.setattr(device, "gpu_core_count", lambda: 20)
    monkeypatch.setattr(device, "sysctl_wired_limit_mb", lambda: 0)
    monkeypatch.setattr(E, "check_checkpoint", lambda path, verify_weights=False: manifest)
    monkeypatch.setattr(E, "Tokenizer", StubTokenizer)
    monkeypatch.setattr(tokenizer, "Tokenizer", StubTokenizer)
    monkeypatch.setattr(loader, "load_weights", load_weights)
    monkeypatch.setattr(kv_cache, "History", FakeHist)
    monkeypatch.setattr(kv_cache, "MKHistory", FakeMKHist)
    monkeypatch.setattr(kv_cache, "MKHistory8", FakeMKHist)
    monkeypatch.setattr(graph, "decode_step", decode_step)
    monkeypatch.setattr(graph, "prefill_chunk", prefill_chunk)
    monkeypatch.setattr(runner, "MegaKernel", FakeMK)
    # the load check unbounded in time: here its reference rows run on the CPU, 12 s for the doctor's
    # on an idle host and over its 120 s bound on a loaded one (test_bound sets a bound of its own)
    monkeypatch.setattr(loadcheck, "BOUND_S", 1e6)
    grid_cache = str(tmp_path / "grid-v1.json")
    monkeypatch.setenv("FLINT_GRID_CACHE", grid_cache)
    FakeMK.reset()
    FakeMK._rows = {}
    default = mx.default_device()
    mx.set_default_device(mx.cpu)  # every array of the fakes and the stub model: CPU
    engines: list = []

    def engine(logs, sim=None, **options):
        kw = {"context_limit": 4096, "prefix_cache": False, "log": logs.append, "wait": False}
        kw.update(options)
        if sim is not None:
            kw.update(sim=sim, allow_fault_injection=True)
        eng = E.Engine(flint_headers, **kw)
        engines.append(eng)
        # deadlines only a hang reaches: on a loaded host the fake load check took over 60 s, and an engine
        # left loading past teardown went on into the next test (its history release counted there)
        assert eng._ready.wait(600)
        return eng

    yield SimpleNamespace(
        engine=engine,
        MK=FakeMK,
        Hist=FakeMKHist,
        model=flint_headers,
        grid_cache=grid_cache,
        stream_of=stream_of,
    )
    for eng in engines:
        eng.close(timeout=600)
    FakeMK._rows = {}
    mx.set_default_device(default)
