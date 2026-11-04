"""`doctor` on the fake GPU: the JSON report, its exit status, simulated Macs, and `--tune` with the grid cache."""

import json
import os

from jitengine_flint_m4pro import device as D
from jitengine_flint_m4pro.cli import main

M1_PRO = {  # a 32 GB M1 Pro, its wired limit raised to 26 GiB below
    "architecture": "applegpu_g13s",
    "device_name": "Apple M1 Pro",
    "gpu_cores": 16,
    "memory_size": 32 * 2**30,
    "max_recommended_working_set_size": 24 * 2**30,
    "max_buffer_length": 30_150_000_000,
}
M4_32 = {"architecture": "applegpu_g16s", "device_name": "Apple M4 Pro", "gpu_cores": 32}


def doctor(fake, tmp_path, *extra, smoke=0, context="4096"):
    path = str(tmp_path / f"doctor{len(os.listdir(tmp_path))}.json")
    argv = ["doctor", "--model-path", fake.model, "--json", path, "--smoke-tokens", str(smoke)]
    code = main([*argv, "--context-limit", context, *extra])
    with open(path) as f:
        rep = json.load(f)
    assert rep["exit"] == code and rep["doctor"] == 1 and rep["model"] == fake.model, rep
    return code, rep


def test_report_and_exit_status(fake_gpu, tmp_path, capsys):
    """Exit 0: the megakernel with speculation; 1: speculation off, or stock MLX ops; 2: refused. The report carries
    the device, the notes, /health (backend, grid, context, load check) and the smoke decode (here through the
    stock-MLX fallback: the fake megakernel answers the load check, not a decode)."""
    code, rep = doctor(fake_gpu, tmp_path)  # no smoke tokens: the fake megakernel cannot decode
    assert code == 0 and rep["verdict"] == "mk" and "reason" not in rep, rep
    assert set(rep) >= {"requested", "device", "notes", "health", "wall_s"} and "smoke" not in rep
    assert rep["requested"] == {
        "context_limit": 4096,
        "memory_budget_gib": None,
        "memory_budget_gb": None,
        "mode": "auto",
        "decode_backend": "auto",
        "mk_grid": None,
        "simulated": None,
    }
    assert rep["device"]["device_name"] == "Apple M4 Pro" and rep["device"]["tuned"]
    h = rep["health"]
    assert (
        h["mk"]["G"] == 40
        and h["mk"]["G_source"] == "tuned"
        and h["context_limit_source"] == "explicit"
    )
    assert h["mk"]["loadcheck"]["stock"]["ok"] and h["multi_token"]
    out = capsys.readouterr().out
    assert "flint doctor: mk (exit 0)" in out and "grid G 40 (tuned)" in out and "check " in out
    fake_gpu.MK.reset(verify_fails=True)
    code, rep = doctor(fake_gpu, tmp_path)
    assert code == 1 and rep["verdict"] == "mk" and "speculation off" in rep["reason"], rep
    fake_gpu.MK.reset()
    code, rep = doctor(fake_gpu, tmp_path, "--simulate", json.dumps({"check_mutant": "verify_row"}))
    assert code == 1 and "the load check's verify rows differ" in rep["reason"], rep
    fake_gpu.MK.reset()
    code, rep = doctor(
        fake_gpu, tmp_path, "--simulate", json.dumps({"fail_compile": True}), smoke=4
    )
    assert (
        code == 1
        and rep["verdict"] == "graph"
        and rep["health"]["mk_unavailable"]["stage"] == "compile"
    )
    assert len(rep["smoke"]["token_ids"]) == 4 and rep["smoke"]["error"] is None, rep["smoke"]
    assert rep["health"]["simulated"]["fail_compile"] is True
    fake_gpu.MK.reset()
    sim = json.dumps({"fail_probe_G": "all"})
    code, rep = doctor(fake_gpu, tmp_path, "--profile", "kv8", "--simulate", sim)
    assert (
        code == 2
        and rep["verdict"] == "refused"
        and "profile kv8 needs the megakernel" in rep["reason"]
    )
    assert rep["health"]["status"] == "unhealthy"
    for extra, needle in (
        (["--simulate", json.dumps({"bogus": 1})], "unknown sim keys"),
        (["--mk-grid", "62"], "mk_grid"),
        (["--context-limit", "262145"], "context limit"),
    ):
        code, rep = doctor(fake_gpu, tmp_path, *extra)
        assert code == 2 and needle in rep["reason"] and "health" not in rep, rep
    capsys.readouterr()
    code = main(["doctor", "--model-path", fake_gpu.model, "--json", "--smoke-tokens", "0"])
    # --json with no file: the report on stdout, the summary on stderr
    captured = capsys.readouterr()
    assert code == 0 and json.loads(captured.out)["verdict"] == "mk"
    assert "flint doctor: mk (exit 0)" in captured.err


def test_simulated_macs(fake_gpu, tmp_path):
    """The report another Mac would send, simulated: an M1 Pro with its wired limit raised to 26 GiB gets G0 32 and
    73,728 tokens of context; with its default limit the whole model does not fit, so it offloads experts (exit 1),
    while profile kv8, which needs every expert resident, is refused with the sysctl that would hold 4,096 tokens; a
    GPU that is not Apple silicon is refused before the engine is built."""
    sim = {"device": M1_PRO, "wired_limit_mb": 26 * 1024}
    code, rep = doctor(fake_gpu, tmp_path, "--simulate", json.dumps(sim), context="auto")
    h = rep["health"]
    assert (
        code == 0 and rep["device"]["device_name"] == "Apple M1 Pro" and not rep["device"]["tuned"]
    )
    assert (h["mk"]["G"], h["mk"]["G_source"]) == (32, "rule") and h["context_limit"] == 73_728
    assert h["context_limit_source"] == "auto" and any("generation 13" in n for n in rep["notes"])
    code, rep = doctor(
        fake_gpu, tmp_path, "--simulate", json.dumps({"device": M1_PRO}), context="auto"
    )
    mem = rep["health"]["memory"]
    assert code == 1 and rep["verdict"] == "offload" and mem["mode"] == "offload", rep
    assert mem["budget_source"] == "default" and rep["health"]["decode_backend"] == "graph"
    assert rep["reason"].startswith("the whole model does not fit"), rep["reason"]
    code, rep = doctor(fake_gpu, tmp_path, "--simulate", json.dumps(sim), "--mode", "offload")
    assert code == 1 and rep["health"]["memory"]["mode"] == "offload", rep
    assert rep["requested"]["mode"] == "offload" and rep["reason"].startswith("--mode offload: "), (
        rep
    )
    code, rep = doctor(
        fake_gpu,
        tmp_path,
        "--profile",
        "kv8",
        "--simulate",
        json.dumps({"device": M1_PRO}),
        context="auto",
    )
    assert code == 2 and "iogpu.wired_limit_mb=" in rep["reason"], rep
    amd = {**M1_PRO, "architecture": "gfx1030", "device_name": "AMD"}
    code, rep = doctor(fake_gpu, tmp_path, "--simulate", json.dumps({"device": amd}))
    assert code == 2 and "not Apple silicon" in rep["reason"] and "health" not in rep, rep


def test_tune_and_the_grid_cache(fake_gpu, tmp_path, monkeypatch):
    """`doctor --tune` times the grids (stubbed here), caches the fastest under the device the engine sees, and serves
    on it; later engines on that device try the cached grid first, then the rule's search without it. On the tuned
    M4 Pro the tuned table wins and the cache is never read."""
    from jitengine_flint_m4pro import tune
    from jitengine_flint_m4pro.engine.engine import toolchain_versions

    calls = []

    def fake_tune(W, kv8, tokenizer, log=print, **k):
        calls.append((W.cfg.n_layers, kv8))
        return {
            "G": 52,
            "ms": {"52": 7.0, "60": 7.2},
            "per_position": {},
            "refused": {"64": "x"},
            "positions": [1536, 4608],
            "T": 16,
            "rounds": 3,
            "s": 0.1,
        }

    monkeypatch.setattr(tune, "tune", fake_tune)
    sim32 = {"device": M4_32}
    fake_gpu.MK.reset(refuse={64})
    code, rep = doctor(fake_gpu, tmp_path, "--tune", "--simulate", json.dumps(sim32))
    t = rep["tune"]
    assert (
        code == 0 and t["G"] == 52 and t["cache"] == fake_gpu.grid_cache and calls == [(49, False)]
    )
    assert not t["tuned_table_wins"] and t["key"].startswith("applegpu_g16s|Apple M4 Pro|32|ksrc ")
    assert (rep["health"]["mk"]["G"], rep["health"]["mk"]["G_source"]) == (52, "cache")
    v = toolchain_versions()
    f32 = D.read_facts(D.parse_sim(sim32, True))
    assert D.read_grid_cache(f32, v, 16)["G"] == 52 and D.read_grid_cache(f32, v, 8) is None
    fake_gpu.MK.reset(refuse={52, 64})  # the cached grid refused: the rule's search, without it
    h = fake_gpu.engine([], sim=sim32).health()
    assert (h["mk"]["G"], h["mk"]["G_source"]) == (60, "search") and fake_gpu.MK.built == [
        52,
        64,
        60,
    ]
    assert h["mk"]["reason"].startswith("cache: `doctor --tune` timed")
    os.remove(fake_gpu.grid_cache)
    fake_gpu.MK.reset(refuse={64})
    e = fake_gpu.engine([], sim=sim32)
    assert e.grid_cache is None and (e.health()["mk"]["G"], e.health()["mk"]["G_source"]) == (
        60,
        "search",
    )
    # the tuned M4 Pro: --tune runs, the tuned table wins; the cache is never read
    fake_gpu.MK.reset()
    code, rep = doctor(fake_gpu, tmp_path, "--tune")
    assert code == 0 and rep["tune"]["tuned_table_wins"] and rep["health"]["mk"]["G"] == 40
    assert rep["health"]["mk"]["G_source"] == "tuned" and fake_gpu.MK.built == [40]
    e = fake_gpu.engine([])
    assert e.grid_cache is None and e.health()["mk"]["G"] == 40
    # a refusal comes before any tuning
    calls.clear()
    code, rep = doctor(fake_gpu, tmp_path, "--tune", "--mk-grid", "62")
    assert code == 2 and "tune" not in rep and calls == []


def inspect(fake, tmp_path, *extra, context="4096"):
    path = str(tmp_path / f"inspect{len(os.listdir(tmp_path))}.json")
    code = main(
        ["inspect", "--model-path", fake.model, "--json", path, "--context-limit", context, *extra]
    )
    with open(path) as f:
        rep = json.load(f)
    assert rep["exit"] == code and rep["inspect"] == 1 and rep["model"] == fake.model, rep
    return code, rep


def test_inspect_plans_without_loading(fake_gpu, tmp_path, monkeypatch, capsys):
    """inspect reports what serve would plan and loads nothing: resident on the tuned M4 Pro with its grid, offload on
    a simulated 32 GB M1 Pro or with --mode offload, and exit 2 with serve's reason where serve refuses."""
    from jitengine_flint_m4pro import loader

    def no_weights(*a, **k):
        raise AssertionError("inspect read the weights")

    monkeypatch.setattr(loader, "load_weights", no_weights)
    code, rep = inspect(fake_gpu, tmp_path)
    assert code == 0 and rep["verdict"] == "resident" and rep["memory"]["mode"] == "resident", rep
    assert rep["grid"] == {"G0": 40, "source": "tuned", "reason": rep["grid"]["reason"]}
    assert (rep["context_limit"], rep["decode_backend"], rep["speculation"]) == (4096, "auto", True)
    out = capsys.readouterr().out
    assert out.startswith("flint inspect: resident (exit 0)") and "  memory   resident, " in out, (
        out
    )
    sim = json.dumps({"device": M1_PRO})
    code, rep = inspect(fake_gpu, tmp_path, "--simulate", sim, context="auto")
    assert code == 0 and rep["verdict"] == "offload" and rep["memory"]["budget_source"] == "default"
    assert (rep["decode_backend"], rep["grid"], rep["speculation"]) == ("graph", None, False)
    assert rep["context_limit"] == 32_768 and any("not the tuned device" in n for n in rep["notes"])
    code, rep = inspect(fake_gpu, tmp_path, "--mode", "offload")
    assert (
        code == 0 and rep["memory"]["mode"] == "offload" and rep["requested"]["mode"] == "offload"
    )
    code, rep = inspect(fake_gpu, tmp_path, "--memory-budget-gib", "2")
    assert code == 2 and rep["verdict"] == "refused" and "budget of 2.00 GiB" in rep["reason"], rep
    assert fake_gpu.MK.built == []
