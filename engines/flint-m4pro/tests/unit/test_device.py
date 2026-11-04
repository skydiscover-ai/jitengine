"""Admission by capability, the grid per device and its cache, and the simulation hook, on device facts given here."""

import json

from jitengine_flint_m4pro import device as D
from jitengine_flint_m4pro.kernels import plan as P
from jitengine_flint_m4pro.kernels import runner as RN
from jitengine_flint_m4pro.model.config import StartupError

GIB = 2**30
BANKS = [10_870_000_000, 3_900_000_000]  # the checkpoint's largest bank is 10.87 GB
TUNED = D.Facts("applegpu_g16s", "Apple M4 Pro", 20, 48 * GIB, 40_200_000_000, 30_150_672_384, 0)
VERSIONS = {"kernel_source_sha256": "0123456789abcdef", "mlx": "0.32.2", "macos": "26.6"}


def expect_raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return str(e)
    raise AssertionError(f"{fn.__name__}{a} did not raise {exc.__name__}")


def facts(name="Apple M4 Pro", arch="applegpu_g16s", cores=20, mbl=30_150_672_384, metal=True):
    return D.Facts(arch, name, cores, 48 * GIB, 40_200_000_000, mbl, 0, metal)


def test_g0_rule_and_grid_domain():
    """Two threadgroups per core, a multiple of 4 in 16..64, 40 when the core count is unknown; check_grid accepts
    exactly the grids MegaKernel's constructor accepts."""
    cores = (8, 10, 14, 16, 18, 19, 20, 30, 32, 38, 76)
    assert [D.g0_rule(c) for c in cores] == [16, 20, 28, 32, 36, 36, 40, 60, 64, 64, 64]
    assert D.g0_rule(None) == D.g0_rule(0) == D.g0_rule(True) == 40
    assert P.MAX_SG // 8 == D.G_MAX
    for c in range(1, 200):
        g = D.g0_rule(c)
        assert g % 4 == 0 and 16 <= g <= 64 and (g <= 2 * c or g == 16), (c, g)
    for g in range(80):
        try:
            RN.MegaKernel(None, G=g)
            runner_ok = True
        except ValueError:
            runner_ok = False
        except AttributeError:  # past the grid check (no weights)
            runner_ok = True
        try:
            D.check_grid(g)
            ok = True
        except StartupError:
            ok = False
        assert ok == runner_ok, g
    for bad in (True, 40.0, "40"):
        expect_raises(StartupError, D.check_grid, bad)


def test_plan_grid():
    """The plan: pinned > tuned > a `doctor --tune` cache entry > the core-count rule; each says why."""

    def plan(*a, **k):
        p = D.plan_grid(*a, **k)
        assert p.reason.startswith(p.source + ": "), p
        return p.candidates, p.source, p.G0

    assert D.TUNED == {("applegpu_g16s", "Apple M4 Pro", 20): 40}
    assert plan(TUNED) == ((40,), "tuned", 40)
    binned = facts(cores=16)
    assert plan(binned) == ((32, 28, 24, 20, 16), "rule", 32)
    assert "2 x 16 GPU cores" in D.plan_grid(binned).reason
    blind = facts(cores=None)
    assert plan(blind) == ((40, 36, 32, 28, 24, 20, 16), "rule", 40)
    assert "unknown" in D.plan_grid(blind).reason
    assert plan(TUNED, pinned=64) == ((64,), "pinned", 64)
    expect_raises(StartupError, D.plan_grid, TUNED, pinned=62)
    m32 = facts(cores=32)
    assert plan(m32) == (tuple(range(64, 15, -4)), "rule", 64)
    cands, source, g0 = plan(m32, cached={"G": 48, "when": "t"})
    assert (source, g0) == ("cache", 48) and cands == (
        48,
        64,
        60,
        56,
        52,
        44,
        40,
        36,
        32,
        28,
        24,
        20,
        16,
    )
    assert D.plan_grid(m32, pinned=44, cached={"G": 48}).candidates == (44,)
    assert D.plan_grid(TUNED, cached={"G": 48}).candidates == (40,)


def test_admission():
    """Refused only for no Metal, a GPU that is not Apple silicon, or a weight bank larger than the largest buffer;
    any other GPU is admitted with a note, and an unknown or unverified generation with one more."""
    assert D.admit(TUNED, BANKS) == [
        "device Apple M4 Pro (applegpu_g16s, 20 GPU cores): the tuned device"
    ]
    assert "no Metal GPU" in expect_raises(StartupError, D.admit, facts(metal=False), BANKS)
    msg = expect_raises(StartupError, D.admit, facts("AMD Radeon Pro W6800", "gfx1030"), BANKS)
    assert "not Apple silicon" in msg
    assert "largest buffer" in expect_raises(StartupError, D.admit, facts(mbl=8 * 10**9), BANKS)
    notes = D.admit(facts("Apple M1 Pro", "applegpu_g13s", 16), BANKS)
    assert len(notes) == 2 and "not the tuned device" in notes[0] and "stock MLX ops" in notes[0]
    assert "generation 13" in notes[1] and "not yet verified" in notes[1]
    # a verified generation: one note
    assert len(D.admit(facts("Apple M2 Max", "applegpu_g14c", 38), BANKS)) == 1
    notes = D.admit(facts("Apple M6 Pro", "applegpu_g18s", 20), BANKS)
    assert "generation 18 is outside" in notes[1]
    assert "unknown number of" in D.admit(facts(cores=None), BANKS)[0]
    # the wired limit: iogpu.wired_limit_mb when set, else the working set
    assert TUNED.wired_bytes == 40_200_000_000
    assert D.Facts(*TUNED.key, 1, 2, 3, 26_624).wired_bytes == 26_624 * 2**20
    s = TUNED.summary()
    assert s["tuned"] and s["wired_bytes"] == 40_200_000_000 and s["gpu_cores"] == 20


def test_read_facts_with_simulation(monkeypatch):
    """A simulation replaces the device facts and the wired limit; Metal is the real GPU's."""
    monkeypatch.setattr(
        D,
        "mlx_device_info",
        lambda: {"architecture": "applegpu_g16s", "device_name": "Apple M4 Pro"},
    )
    monkeypatch.setattr(D, "gpu_core_count", lambda: 20)
    monkeypatch.setattr(D, "sysctl_wired_limit_mb", lambda: 46_080)
    real = D.read_facts()
    assert real.tuned_grid == 40 and real.wired_bytes == 46_080 * 2**20 and real.metal
    sim = D.parse_sim({"device": {"gpu_cores": 32}, "wired_limit_mb": 0}, True)
    f = D.read_facts(sim)
    assert f.gpu_cores == 32 and f.tuned_grid is None and f.wired_limit_mb == 0
    monkeypatch.setattr(D, "mlx_device_info", dict)
    assert not D.read_facts().metal and D.read_facts().gpu_cores is None


def test_simulation_hook():
    """The hook needs allow_fault_injection, refuses unknown keys and bad values, and echoes what it was given."""
    assert D.parse_sim(None, False) is None
    expect_raises(StartupError, D.parse_sim, {"fail_compile": True}, False)
    for bad in (
        [],
        {"bogus": 1},
        {"device": {"cores": 8}},
        {"device": {"gpu_cores": 0}},
        {"device": {"architecture": 16}},
        {"device": []},
        {"wired_limit_mb": -1},
        {"fail_compile": 1},
        {"fail_probe_G": [64.0]},
        {"fail_probe_G": "some"},
        {"check_mutant": "bogus"},
    ):
        expect_raises(StartupError, D.parse_sim, bad, True)
    s = D.parse_sim({"fail_probe_G": [64, 60], "check_mutant": "pos_shift"}, True)
    assert (
        s.probe_fails(64)
        and not s.probe_fails(56)
        and D.parse_sim({"fail_probe_G": "all"}, True).probe_fails(16)
    )
    assert s.as_dict() == {
        "device": None,
        "wired_limit_mb": None,
        "fail_compile": False,
        "fail_probe_G": [64, 60],
        "check_mutant": "pos_shift",
    }
    assert set(D.CHECK_MUTANTS) == {
        "wrong_expert",
        "pos_shift",
        "stale_history",
        "chain_bitflip",
        "verify_row",
    }


def test_grid_cache(tmp_path, monkeypatch):
    """The `doctor --tune` cache: keyed by the device, the kernel sources, the profile's history width, MLX and macOS;
    advisory, so a missing, malformed or out-of-domain entry is ignored; a grid outside the domain is never written."""
    path = str(tmp_path / "grid.json")
    monkeypatch.setenv(D.GRID_CACHE_ENV, path)
    m32 = facts(cores=32)
    key = D.grid_cache_key(m32, VERSIONS, 16)
    assert key == "applegpu_g16s|Apple M4 Pro|32|ksrc 0123456789abcdef|kv16|mlx 0.32.2|macOS 26.6"
    assert D.grid_cache_path() == path
    monkeypatch.delenv(D.GRID_CACHE_ENV)
    assert D.grid_cache_path().endswith("/.cache/flint-engine/grid-v1.json")
    monkeypatch.setenv(D.GRID_CACHE_ENV, path)
    assert D.read_grid_cache(m32, VERSIONS, 16) is None
    assert D.write_grid_cache(m32, VERSIONS, 16, {"G": 48, "ms": {"48": 7.1}}) == path
    e = D.read_grid_cache(m32, VERSIONS, 16)
    assert e["G"] == 48 and e["ms"] == {"48": 7.1} and e["when"]
    assert D.read_grid_cache(m32, VERSIONS, 8) is None
    for other in ({"kernel_source_sha256": "f" * 16}, {"mlx": "0.0.1"}, {"macos": "15.0"}):
        assert D.read_grid_cache(m32, {**VERSIONS, **other}, 16) is None
    assert D.read_grid_cache(facts(cores=16), VERSIONS, 16) is None
    D.write_grid_cache(facts(cores=16), VERSIONS, 16, {"G": 32})  # other entries are kept
    assert D.read_grid_cache(m32, VERSIONS, 16)["G"] == 48
    expect_raises(StartupError, D.write_grid_cache, m32, VERSIONS, 16, {"G": 62})
    raw = json.load(open(path))
    raw["entries"][key]["G"] = 68
    json.dump(raw, open(path, "w"))
    assert D.read_grid_cache(m32, VERSIONS, 16) is None
    open(path, "w").write("{not json")
    assert D.read_grid_cache(m32, VERSIONS, 16) is None
    json.dump({"version": 2, "entries": {key: {"G": 48}}}, open(path, "w"))
    assert D.read_grid_cache(m32, VERSIONS, 16) is None


def _simdev(name, arch, cores, mem_gib, ws_gib, mbl_gb=30.15):
    return {
        "architecture": arch,
        "device_name": name,
        "gpu_cores": cores,
        "memory_size": int(mem_gib * GIB),
        "max_recommended_working_set_size": int(ws_gib * GIB),
        "max_buffer_length": int(mbl_gb * 1e9),
    }


def test_simulated_macs(fake_gpu):
    """Simulated Macs through the engine's constructor (the fake GPU, no weights): admitted or refused before any
    weight, the grid plan from the core count, the context limit from the wired limit (main and kv8)."""
    from jitengine_flint_m4pro.engine.engine import Engine

    def build(dev, wired=None, **kw):
        sim = {"device": dev}
        if wired is not None:
            sim["wired_limit_mb"] = wired
        kw = {"start": False, "allow_fault_injection": True, "log": lambda m: None, **kw}
        return Engine(fake_gpu.model, sim=sim, **kw)

    m1pro = _simdev("Apple M1 Pro", "applegpu_g13s", 16, 32, 24)
    cases = [  # device, wired limit (MiB, 0: the working set) -> G0, main context, kv8 context
        (m1pro, 26 * 1024, 32, 73_728, 139_264),
        (_simdev("Apple M3 Pro", "applegpu_g15s", 18, 36, 27), 0, 36, 111_616, 209_920),
        (_simdev("Apple M4", "applegpu_g16g", 10, 32, 21.3), 28 * 1024, 20, 148_480, 262_144),
        (_simdev("Apple M2 Max", "applegpu_g14c", 38, 96, 72), 0, 64, 262_144, 262_144),
        (_simdev("Apple M2 Ultra", "applegpu_g14d", 76, 192, 144), 0, 64, 262_144, 262_144),
        (_simdev("Apple M6 Pro", "applegpu_g18s", 20, 48, 37.4), 0, 40, 262_144, 262_144),
        (_simdev("Apple M4 Pro", "applegpu_g16s", 16, 48, 37.4), 0, 32, 262_144, 262_144),
    ]
    for dev, wired, g0, ctx, ctx8 in cases:
        e = build(dev, wired)
        h = e.health()
        assert e.grid.source == "rule" and e.grid.G0 == g0 and e.grid.candidates[-1] == 16, dev
        assert (e.context_limit, e.context_limit_source) == (ctx, "auto"), (dev, e.context_limit)
        assert build(dev, wired, profile="kv8").context_limit == ctx8, dev
        assert h["simulated"]["device"] == dev and not h["device"]["tuned"]
        assert h["device"]["gpu_cores"] == dev["gpu_cores"] and h["context_limit"] == ctx
        assert h["decode_backend"] == "auto" and h["mk"] is None
        assert any("not the tuned device" in n for n in e.device_notes)
    assert any("generation 18" in n for n in build(cases[5][0]).device_notes)
    assert build(_simdev("Apple M4 Pro", "applegpu_g16s", 20, 48, 37.4)).grid.candidates == (40,)
    # explicit limits: as given when they fit, else refused with what fits and the sysctl
    e = build(m1pro, 26 * 1024, context_limit=73_728)
    assert (e.context_limit, e.context_limit_source) == (73_728, "explicit")
    assert e.memory.mode == "resident" and e.memory.pool_slots == 0
    # no simulation: the fake GPU is the tuned M4 Pro, whose default working set holds 262,144 tokens
    e = Engine(fake_gpu.model, start=False, log=lambda m: None)
    assert (e.grid.candidates, e.grid.source) == ((40,), "tuned") and e.context_limit == 262_144
    assert e.context_limit_source == "auto" and e.decode_backend == "auto" and e.speculation


def test_simulated_macs_offload(fake_gpu):
    """Simulated Macs that do not hold the whole model: profile main offloads the routed experts (memory plan
    "offload", the graph backend, no speculation); kv8, the megakernel or a pinned grid keep the resident refusal; a
    budget picks resident or offload, and one too small for a layer of experts is refused before any weight."""
    from jitengine_flint_m4pro.engine.engine import Engine

    def build(dev, wired=None, **kw):
        sim = {"device": dev}
        if wired is not None:
            sim["wired_limit_mb"] = wired
        kw = {"start": False, "allow_fault_injection": True, "log": lambda m: None, **kw}
        return Engine(fake_gpu.model, sim=sim, **kw)

    m1pro = _simdev("Apple M1 Pro", "applegpu_g13s", 16, 32, 24)
    e = build(m1pro, 26 * 1024, context_limit=73_728)
    assert e.memory.mode == "resident" and e.memory.pool_slots == 0
    # an explicit limit the wired limit cannot hold beside the whole model: offload keeps it; the megakernel refuses
    e = build(m1pro, 26 * 1024, context_limit=262_144)
    assert (e.memory.mode, e.context_limit, e.decode_backend) == ("offload", 262_144, "graph")
    assert not e.speculation and e.grid is None and 0 < e.memory.pool_slots < e.memory.total_records
    msg = expect_raises(
        StartupError, build, m1pro, 26 * 1024, context_limit=262_144, decode_backend="mk"
    )
    assert "context limit 262,144 needs" in msg and "the most that fits is 73,728 tokens" in msg, (
        msg
    )
    # the 32 GB M1 Pro's 24 GiB working set does not hold the whole model: offload within it, 32,768 tokens of history
    e = build(m1pro)
    assert (e.memory.mode, e.memory.budget_source, e.context_limit) == (
        "offload",
        "default",
        32_768,
    )
    assert e.memory.budget_bytes == 24 * 2**30 and e.memory.planned_bytes <= e.memory.budget_bytes
    # kv8 needs every expert resident: refused as before, with the sysctl that would hold 4,096 tokens
    msg = expect_raises(StartupError, build, m1pro, profile="kv8")
    assert "Flint needs 24,655 MiB for 4,096 tokens" in msg, msg
    assert "sudo sysctl iogpu.wired_limit_mb=24655" in msg, msg
    msg = expect_raises(StartupError, build, m1pro, decode_backend="mk")
    assert "Flint needs 24,708 MiB for 4,096 tokens" in msg, msg
    assert "sudo sysctl iogpu.wired_limit_mb=24708" in msg, msg
    m2 = _simdev("Apple M2", "applegpu_g14g", 10, 24, 16)
    assert build(m2).memory.mode == "offload"
    assert "cannot hold it" in expect_raises(StartupError, build, m2, profile="kv8")
    # a budget: resident when it holds the whole model, offload below, refused below one layer of experts
    assert build(m1pro, 26 * 1024, memory_budget_bytes=25 * 2**30).memory.mode == "resident"
    e = build(m1pro, 26 * 1024, memory_budget_bytes=12 * 2**30)
    assert (e.memory.mode, e.memory.budget_bytes, e.memory.budget_source) == (
        "offload",
        12 * 2**30,
        "flag",
    )
    msg = expect_raises(StartupError, build, m1pro, 26 * 1024, memory_budget_bytes=2 * 2**30)
    assert "too small for Flint" in msg and "declare at least" in msg, msg
    assert "not Apple silicon" in expect_raises(
        StartupError, build, _simdev("AMD Radeon Pro W6800", "gfx1030", 60, 64, 48)
    )
    # an expert bank larger than the GPU's largest buffer: offload (pool arrays are smaller); kv8 is refused
    small_buffers = _simdev("Apple M1", "applegpu_g13g", 8, 64, 48, mbl_gb=8.0)
    assert build(small_buffers).memory.mode == "offload"
    assert "largest buffer" in expect_raises(StartupError, build, small_buffers, profile="kv8")
    # --mode: offload on a Mac that holds the whole model, with the default budget; resident refuses where auto
    # would offload; offload with kv8 or the megakernel is refused
    e = build(m1pro, 26 * 1024, memory_mode="offload")
    assert (e.memory.mode, e.decode_backend, e.memory.budget_source) == (
        "offload",
        "graph",
        "default",
    )
    assert e.health()["memory_mode_requested"] == "offload"
    msg = expect_raises(StartupError, build, m1pro, memory_mode="resident")
    assert "Flint needs" in msg and "sudo sysctl iogpu.wired_limit_mb=" in msg, msg
    for kw in ({"profile": "kv8"}, {"decode_backend": "mk"}):
        msg = expect_raises(StartupError, build, m1pro, 26 * 1024, memory_mode="offload", **kw)
        assert "memory mode 'offload' streams the experts" in msg, msg
