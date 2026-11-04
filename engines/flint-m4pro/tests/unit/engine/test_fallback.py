"""The megakernel's bring-up on the fake GPU: the grid search per device, and what a failed load-check stage means.

"auto" + main: one WARNING line, /health decode_backend "graph" and mk_unavailable {stage, reason}, the kernels and the
megakernel history freed, requests decoded with stock MLX ops (here the stub model). kv8: refused, never served as
main. decode_backend "mk" or a pinned grid: refused, never searched or replaced. No weights, no kernel runs."""

from jitengine_flint_m4pro.engine.io_struct import GenRequest
from jitengine_flint_m4pro.model.config import StartupError

WARN = "megakernel unavailable on this Mac ("
FALLBACK = "decoding with stock MLX ops, about 0.56x the tuned speed at 1K, no speculation"
M4_32 = {"architecture": "applegpu_g16s", "device_name": "Apple M4 Pro", "gpu_cores": 32}


def expect_raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return
    raise AssertionError(f"{fn.__name__}{a} did not raise {exc.__name__}")


def _greedy(n):
    return GenRequest(prompt_ids=list(range(1, 33)), max_tokens=n, temperature=0.0, ignore_eos=True)


def _decodes_with_stock_mlx(fake, eng, logs, stage, grids):
    h = eng.health()
    assert h["status"] == "ready" and h["decode_backend"] == "graph" and h["mk"] is None, h
    mu = h["mk_unavailable"]
    assert mu["stage"] == stage and not h["multi_token"] and eng.mk is None, h
    assert mu["grid"] == eng.grid.reason and [t["G"] for t in mu["tried"]] == grids, mu
    lines = [m for m in logs if WARN in m]
    assert len(lines) == 1 and lines[0].startswith(f"[engine] WARNING: {WARN}{stage}: "), lines
    assert lines[0].endswith(FALLBACK)
    assert fake.Hist.released == 1  # the megakernel history was freed before the graph backend's
    r = eng.generate(_greedy(6))
    assert r.finish_reason == "length" and r.token_ids == fake.stream_of(32, 6), r
    expect_raises(ValueError, eng._validate, GenRequest(prompt_ids=[1], fault="kernel_stall@3"))


def _refused(eng, *needles):
    h = eng.health()
    assert h["status"] == "unhealthy" and h["decode_backend"] != "graph", h
    assert "mk_unavailable" not in h and all(n in h["fatal"] for n in needles), h["fatal"]
    expect_raises(Exception, eng.generate, GenRequest(prompt_ids=[1, 2], max_tokens=2))


def test_tuned_device_probes_g40_only(fake_gpu):
    """The tuned M4 Pro: G 40, no search, even when the grids above it would be refused."""
    fake_gpu.MK.reset(refuse={64, 60, 56, 52, 48, 44})
    logs = []
    e = fake_gpu.engine(logs)
    h = e.health()
    tried = h["mk"]["tried"]
    assert [(t["G"], t["ok"]) for t in tried] == [(40, True)] and 0 <= tried[0]["s"] < 5, tried
    assert h["decode_backend"] == "mk"
    assert {k: h["mk"][k] for k in ("G", "G_source", "G0")} == {
        "G": 40,
        "G_source": "tuned",
        "G0": 40,
    }
    assert h["mk"]["reason"].startswith(
        "tuned: Apple M4 Pro (applegpu_g16s, 20 GPU cores) runs G 40"
    )
    assert fake_gpu.MK.built == [40] and fake_gpu.MK.dispatched == [40] and e.mk_grid == 40
    assert e.multi_token() and h["device"]["tuned"] and h["decode_backend_requested"] == "auto"
    assert not [m for m in logs if "WARNING" in m]


def test_search_down_from_g0(fake_gpu):
    """Another device searches down from G0 to the first co-resident grid; the kernel text is compiled once."""
    fake_gpu.MK.reset(refuse={64})
    logs = []
    h = fake_gpu.engine(logs, sim={"device": M4_32}).health()
    assert (h["mk"]["G"], h["mk"]["G_source"], h["mk"]["G0"]) == (60, "search", 64), h["mk"]
    assert [(t["G"], t["ok"]) for t in h["mk"]["tried"]] == [(64, False), (60, True)]
    assert fake_gpu.MK.built == [64, 60] and fake_gpu.MK.dispatched == [64]
    assert h["decode_backend"] == "mk" and h["simulated"]["device"] == M4_32
    assert any("refused by the residency probe" in m for m in logs)
    fake_gpu.MK.reset()
    e = fake_gpu.engine([], sim={"device": M4_32, "fail_probe_G": [64, 60]})
    assert e.health()["mk"]["G"] == 56 and fake_gpu.MK.built == [64, 60, 56]


def test_failed_stage_falls_back_to_stock_mlx(fake_gpu):
    """ "auto" + main: a failed compile stage, every grid refused, or the tuned grid refused -> stock MLX ops."""
    fake_gpu.MK.reset()
    logs = []
    e = fake_gpu.engine(logs, sim={"fail_compile": True})
    _decodes_with_stock_mlx(fake_gpu, e, logs, "compile", [])
    assert fake_gpu.MK.built == [40] and fake_gpu.MK.dispatched == []
    assert "simulated Metal library build failure" in e.health()["mk_unavailable"]["reason"]
    fake_gpu.MK.reset()
    logs = []
    e = fake_gpu.engine(logs, sim={"device": M4_32, "fail_probe_G": "all"})
    _decodes_with_stock_mlx(fake_gpu, e, logs, "probe", list(range(64, 15, -4)))
    fake_gpu.MK.reset(refuse={40})
    logs = []
    e = fake_gpu.engine(logs)
    _decodes_with_stock_mlx(fake_gpu, e, logs, "probe", [40])  # no search below the tuned grid
    assert fake_gpu.MK.built == [40]


def test_kv8_mk_and_a_pinned_grid_refuse(fake_gpu):
    """kv8 is refused rather than served as main; a pinned grid is probed at that grid only and refused when it is not
    co-resident; decode_backend "mk" searches on another device but never falls back."""
    for sim in ({"fail_compile": True}, {"fail_probe_G": "all"}):
        fake_gpu.MK.reset()
        e = fake_gpu.engine([], sim=sim, profile="kv8")
        _refused(
            e, "profile kv8 needs the megakernel, unavailable on this Mac (", "run --profile main"
        )
    fake_gpu.MK.reset(refuse={64})
    e = fake_gpu.engine([], mk_grid=64, speculation=False)
    _refused(e, "mk_grid=64 is pinned", "residency probe failed at G=64")
    assert fake_gpu.MK.built == [64]
    fake_gpu.MK.reset(refuse={64})
    h = fake_gpu.engine([], mk_grid=48).health()
    assert (h["mk"]["G"], h["mk"]["G_source"]) == (48, "pinned") and fake_gpu.MK.built == [48]
    fake_gpu.MK.reset()
    e = fake_gpu.engine([], sim={"device": M4_32, "fail_probe_G": "all"}, decode_backend="mk")
    _refused(e, "decode_backend 'mk' was requested")
    assert fake_gpu.MK.built == list(range(64, 15, -4))
    fake_gpu.MK.reset()
    _refused(fake_gpu.engine([], sim={"fail_compile": True}, decode_backend="mk"), "(compile: ")


def test_graph_backend_and_constructor_refusals(fake_gpu):
    """decode_backend "graph" builds no kernel; every refusal of a backend, a grid, a context limit or a simulation
    happens in the constructor, before any weight."""
    from jitengine_flint_m4pro.engine.engine import Engine

    fake_gpu.MK.reset()
    h = fake_gpu.engine([], decode_backend="graph").health()
    assert h["decode_backend"] == "graph" and "mk_unavailable" not in h and fake_gpu.MK.built == []
    for bad in (
        dict(mk_grid=62),
        dict(mk_grid=68),
        dict(mk_grid=64, decode_backend="graph"),
        dict(decode_backend="graph", profile="kv8"),
        dict(decode_backend="bogus"),
        dict(context_limit="bogus"),
        dict(context_limit=262_145),
        dict(sim={"fail_compile": True}),  # the hook needs allow_fault_injection
        dict(sim={"bogus": 1}, allow_fault_injection=True),
    ):
        kw = {"context_limit": 4096, **bad}
        expect_raises(StartupError, Engine, fake_gpu.model, start=False, **kw)
