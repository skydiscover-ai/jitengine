"""Packaging: CLI without MLX, profiles, wheel and sdist contents, and an installed wheel."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "src")


def _run(args, env_extra=None, cwd=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd or tempfile.gettempdir(),
        timeout=120,
    )


def test_help_version_and_print_config_do_not_import_mlx():
    code = (
        "import sys, runpy; sys.argv=['jitengine-qwen3.6-35b-a3b-m4pro','serve','--print-config','--profile','24gb-static'];"
        "from jitengine_qwen3_6_35b_a3b_m4pro.cli import main; rc=main(sys.argv[1:]);"
        "print('MLX_IMPORTED' if 'mlx.core' in sys.modules else 'NO_MLX')"
    )
    r = _run(["-c", code])
    assert r.returncode == 0, r.stderr
    assert "NO_MLX" in r.stdout and '"memory_budget_bytes": 19327352832' in r.stdout
    r = _run(["-m", "jitengine_qwen3_6_35b_a3b_m4pro", "--version"])
    assert r.returncode == 0 and "0.1.0a1" in r.stdout
    r = _run(["-m", "jitengine_qwen3_6_35b_a3b_m4pro", "--help"])
    assert r.returncode == 0 and "serve" in r.stdout and "doctor" in r.stdout
    r = _run(["-m", "jitengine_qwen3_6_35b_a3b_m4pro", "serve", "--help"])
    assert "--memory-budget-gib" in r.stdout and "decimal GB" in r.stdout


def test_profiles_resolve_to_documented_bytes():
    from jitengine_qwen3_6_35b_a3b_m4pro.engine.args import engine_options, profile_names, resolve

    assert profile_names() == ["24gb-adaptive", "24gb-static", "relaxed"]
    s, origin = resolve("24gb-static")
    assert s["memory_budget_bytes"] == 18 << 30 and s["system_reserve_bytes"] == 6 << 30
    assert s["mode"] == "auto" and s["memory_policy"] == "static" and s["max_slots"] == 8
    assert s["max_context_tokens"] == 8192 and s["prefill_chunk_tokens"] == 256
    assert s["prefix_cache_bytes"] == 768 << 20 and s["host"] == "127.0.0.1"
    assert origin["memory_budget_bytes"] == "profile:24gb-static"
    a, _ = resolve("24gb-adaptive")
    assert (a["mode"], a["memory_policy"]) == ("offload", "adaptive")
    diff = {k for k in s if s[k] != a[k]}
    assert diff == {"mode", "memory_policy"}
    opts = engine_options(s)
    assert opts["process_budget_bytes"] == 18 << 30 and opts["prefill_chunk"] == 256
    r, _ = resolve("relaxed")
    assert (r["verify_relaxed"], r["spec_sampling"], r["spec_k"]) == (True, "leviathan", 2)


def test_precedence_and_rejections(tmp_path):
    from jitengine_qwen3_6_35b_a3b_m4pro.engine.args import ConfigError, resolve

    cfgp = tmp_path / "c.json"
    cfgp.write_text(
        json.dumps({"schema_version": 1, "max_slots": 2, "expert_pack_path": "packs/x.experts"})
    )
    s, o = resolve("24gb-static", str(cfgp), {"max_slots": 3})
    assert s["max_slots"] == 3 and o["max_slots"] == "flag"
    assert s["expert_pack_path"] == str(tmp_path / "packs" / "x.experts")  # relative to the file
    for bad in (
        {"schema_version": 2},
        {"schema_version": 1, "nope": 1},
        {"schema_version": 1, "max_slots": 1.5},
        {"schema_version": 1, "mode": "fast"},
    ):
        cfgp.write_text(json.dumps(bad))
        try:
            resolve(None, str(cfgp))
        except ConfigError:
            continue
        raise AssertionError(f"accepted {bad}")
    cfgp.write_text('{"schema_version": 1, "memory_budget_bytes": NaN}')
    try:
        resolve(None, str(cfgp))
    except ConfigError:
        pass
    else:
        raise AssertionError("accepted NaN")


def test_gb_and_gib_flags_are_distinct_and_exclusive():
    r = _run(
        [
            "-m",
            "jitengine_qwen3_6_35b_a3b_m4pro",
            "serve",
            "--print-config",
            "--memory-budget-gb",
            "18",
        ]
    )
    assert '"memory_budget_bytes": 18000000000' in r.stdout
    r = _run(
        [
            "-m",
            "jitengine_qwen3_6_35b_a3b_m4pro",
            "serve",
            "--print-config",
            "--memory-budget-gib",
            "18",
        ]
    )
    assert '"memory_budget_bytes": 19327352832' in r.stdout
    r = _run(
        [
            "-m",
            "jitengine_qwen3_6_35b_a3b_m4pro",
            "serve",
            "--print-config",
            "--memory-budget-gib",
            "18",
            "--memory-budget-gb",
            "18",
        ]
    )
    assert r.returncode != 0 and "not both" in r.stderr


def _copy_project(dst) -> str:
    """Copy what a build reads, so setuptools' build/ and egg-info stay out of the checkout."""
    proj = os.path.join(str(dst), "project")
    shutil.copytree(
        os.path.join(ROOT, "src"),
        os.path.join(proj, "src"),
        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
    )
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(os.path.join(ROOT, name), proj)
    return proj


def _build(kind: str, out: str, root: str) -> str:
    """Build with the declared backend, setuptools, in-process and offline."""
    pytest.importorskip("setuptools", minversion="77")
    code = f"from setuptools import build_meta as b; print(b.build_{kind}({out!r}))"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stderr[-2000:]
    return os.path.join(out, r.stdout.strip().splitlines()[-1])


def test_wheel_contents_are_exactly_the_package(tmp_path):
    whl = _build("wheel", str(tmp_path / "dist"), _copy_project(tmp_path))
    names = zipfile.ZipFile(whl).namelist()
    assert (
        "jitengine_qwen3_6_35b_a3b_m4pro/cli.py" in names
        and "jitengine_qwen3_6_35b_a3b_m4pro/kernels/selfcheck.py" in names
    )
    assert "jitengine_qwen3_6_35b_a3b_m4pro/profiles/24gb-static.json" in names
    dist = [n for n in names if ".dist-info/" in n]
    assert any(n.endswith("licenses/LICENSE") for n in dist) and any(
        n.endswith("entry_points.txt") for n in dist
    )
    bad = [
        n
        for n in names
        if not (n.startswith("jitengine_qwen3_6_35b_a3b_m4pro/") or ".dist-info/" in n)
    ]
    assert not bad, bad
    assert not any(
        n.endswith((".safetensors", ".experts", ".pyc")) or "skydiscover" in n for n in names
    )
    meta = zipfile.ZipFile(whl).read(next(n for n in dist if n.endswith("METADATA"))).decode()
    deps = [line for line in meta.splitlines() if line.startswith("Requires-Dist:")]
    assert any(d.startswith("Requires-Dist: mlx") for d in deps)
    assert not any("skydiscover" in d.lower() for d in deps), deps
    assert "Provides-Extra: benchmark" in meta


def test_sdist_rebuilds_an_identical_wheel(tmp_path):
    proj = _copy_project(tmp_path)
    sd = _build("sdist", str(tmp_path / "sd"), proj)
    names = tarfile.open(sd).getnames()
    base = os.path.basename(sd)[: -len(".tar.gz")]
    assert f"{base}/pyproject.toml" in names
    assert f"{base}/src/jitengine_qwen3_6_35b_a3b_m4pro/profiles/24gb-static.json" in names
    tarfile.open(sd).extractall(tmp_path / "x", filter="data")
    w1 = _build("wheel", str(tmp_path / "w1"), proj)
    w2 = _build("wheel", str(tmp_path / "w2"), str(tmp_path / "x" / base))
    z1, z2 = zipfile.ZipFile(w1), zipfile.ZipFile(w2)
    assert sorted(z1.namelist()) == sorted(z2.namelist())
    assert all(z1.read(n) == z2.read(n) for n in z1.namelist())


@pytest.mark.skipif(sys.version_info < (3, 14), reason="the package requires Python 3.14")
def test_installed_wheel_runs_outside_the_checkout(tmp_path):
    """The wheel installs (no deps) into a fresh venv and its script runs outside the checkout."""
    if importlib.util.find_spec("pip") is None:
        pytest.skip("no pip here (a uv venv, for example)")
    whl = _build("wheel", str(tmp_path / "dist"), _copy_project(tmp_path))
    venv = tmp_path / "venv"
    r = subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    py = str(venv / "bin" / "python")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "--python", py, "install", "--no-deps", "--no-index", whl],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:  # e.g. an older pip without --python
        raise AssertionError(r.stderr[-2000:])
    exe = str(venv / "bin" / "jitengine-qwen3.6-35b-a3b-m4pro")
    r = subprocess.run([exe, "--version"], capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 0 and "0.1.0a1" in r.stdout, r.stderr
    r = subprocess.run(
        [
            py,
            "-c",
            "import jitengine_qwen3_6_35b_a3b_m4pro, sys; print(jitengine_qwen3_6_35b_a3b_m4pro.__file__)",
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert str(venv) in r.stdout and SRC not in r.stdout
    r = subprocess.run(
        [exe, "serve", "--print-config", "--profile", "24gb-adaptive"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert r.returncode == 0 and '"memory_policy": "adaptive"' in r.stdout


def test_planner_modules_are_mlx_free():
    """The modules that plan memory without Metal (`inspect`, the policy, resizing) import no MLX:
    the check names what they define, so it follows a definition to whichever module holds it."""
    from jitengine_qwen3_6_35b_a3b_m4pro.checkpoint.inventory import inventory
    from jitengine_qwen3_6_35b_a3b_m4pro.engine.args import resolve, resolve_options
    from jitengine_qwen3_6_35b_a3b_m4pro.memory.budget import capacity_for, limits_for, make_plan
    from jitengine_qwen3_6_35b_a3b_m4pro.memory.pressure import PressurePolicy
    from jitengine_qwen3_6_35b_a3b_m4pro.memory.resize import run_resize

    planning = (inventory, resolve, resolve_options, make_plan, capacity_for, limits_for)
    mods = sorted({f.__module__ for f in (*planning, PressurePolicy, run_resize)})
    code = (
        f"import sys; import {', '.join(mods)};"
        "print('MLX' if 'mlx.core' in sys.modules else 'CLEAN')"
    )
    r = _run(["-c", code])
    assert r.returncode == 0 and "CLEAN" in r.stdout, r.stdout + r.stderr


def test_serve_host_default_is_loopback_and_does_not_override_a_config(tmp_path):
    from jitengine_qwen3_6_35b_a3b_m4pro.cli import build_parser

    serve = next(
        a
        for a in build_parser()._subparsers._group_actions[0].choices.values()
        if a.prog.endswith("serve")
    )
    host = next(x for x in serve._actions if x.dest == "host")
    assert host.default == "127.0.0.1"
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"schema_version": 1, "host": "127.0.0.2", "port": 9999}))
    r = _run(
        ["-m", "jitengine_qwen3_6_35b_a3b_m4pro", "serve", "--print-config", "--config", str(cfg)]
    )
    assert '"host": "127.0.0.2"' in r.stdout and '"port": 9999' in r.stdout
    r = _run(
        [
            "-m",
            "jitengine_qwen3_6_35b_a3b_m4pro",
            "serve",
            "--print-config",
            "--config",
            str(cfg),
            "--port",
            "7000",
        ]
    )
    assert '"port": 7000' in r.stdout and '"host": "127.0.0.2"' in r.stdout


def test_preflight_says_how_much_memory_is_missing(monkeypatch):
    from jitengine_qwen3_6_35b_a3b_m4pro import cli
    from jitengine_qwen3_6_35b_a3b_m4pro.memory import monitor as memory_monitor

    gib = 2**30
    monkeypatch.setattr(
        memory_monitor,
        "system_memory",
        lambda: {"available_bytes_estimate": 16.83 * gib, "swap_used_bytes": 0},
    )
    ok, why = cli._preflight(int(17.96 * gib), 0)
    assert not ok and "short by 1.13 GiB" in why
    ok, why = cli._preflight(int(16 * gib), 0)
    assert ok and "short by" not in why


def test_the_benchmark_ships_beside_the_package():
    bench = os.path.join(ROOT, "benchmarks")
    for name in ("benchmark.py", "accuracy.py"):
        with open(os.path.join(bench, name), encoding="utf-8") as f:
            compile(f.read(), name, "exec")
    for name in ("workload.json", "accuracy_prompts.json"):
        with open(os.path.join(bench, name), encoding="utf-8") as f:
            json.load(f)
