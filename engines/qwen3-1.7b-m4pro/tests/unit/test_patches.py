import subprocess
import sys
import textwrap
import types

import pytest

from jitengine_qwen3_1_7b_m4pro.engine import patches


def test_listen_backlog_is_deep():
    assert patches._Backlog.request_queue_size == 1024 > 5
    assert patches._Backlog.daemon_threads is True


def test_pinned_edits_match_installed_mlx_lm():
    pytest.importorskip("mlx_lm")
    out = _run("""
        import importlib, inspect
        from jitengine_qwen3_1_7b_m4pro.engine import _step_edits as E, patches, source_patch
        generate = importlib.import_module("mlx_lm.generate")
        server = importlib.import_module("mlx_lm.server")
        step = source_patch.source_sha256(generate.GenerationBatch._step)
        gen = source_patch.source_sha256(server.ResponseGenerator._generate)
        if (step, gen) != (patches.STEP_SHA256, patches.GENERATE_SHA256):
            print("SKIP"); raise SystemExit(0)
        s1 = inspect.getsource(generate.GenerationBatch._step)
        s2 = inspect.getsource(server.ResponseGenerator._generate)
        assert s1.count(E.STEP_OLD_1) == 1 and s1.count(E.STEP_OLD_2) == 1
        assert all(s2.count(old) == 1 for old, _ in patches.GENERATE_EDITS)
        print("OK")
    """)
    if out == "SKIP":
        pytest.skip("not the pinned mlx-lm; the engine would run stock here")
    assert out == "OK"


def _run(code: str) -> str:
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr[-800:]
    return out.stdout.strip()


def test_install_applies_everything_and_refuses_other_models():
    pytest.importorskip("mlx_lm")
    out = _run("""
        from jitengine_qwen3_1_7b_m4pro.engine import patches
        from jitengine_qwen3_1_7b_m4pro.engine.state import STATE
        import mlx_lm.server as s
        STATE["served_names"] = {"a/b"}; STATE["served_primary"] = "a/b"
        r = patches.install(selfcheck=False)
        print("applied", " ".join(sorted(r.applied)))
        for model, draft in (("other/model", None), ("a/b", "some/draft")):
            try:
                s.ModelProvider.load(object(), model, None, draft)
            except ValueError as e:
                print("refused", e)
    """)
    applied = next(line for line in out.splitlines() if line.startswith("applied")).split()[1:]
    assert {"lazy_logprobs", "exact_qmm", "listen_backlog"} <= set(applied), out
    assert "other/model" in out and "some/draft" in out and out.count("refused") == 2, out


def test_an_adapter_keeps_the_projections_unfused():
    pytest.importorskip("mlx_lm")
    out = _run("""
        import types
        import mlx_lm.server as s
        s.load = lambda *a, **k: (types.SimpleNamespace(model_type="other"), None, None)
        from jitengine_qwen3_1_7b_m4pro.engine import patches, fusion
        calls = []
        fusion.fuse_model = lambda m: calls.append(m)
        patches.install(selfcheck=False)
        s.load("x", adapter_path="adapters")
        s.load("x", adapter_path=None)
        print(len(calls))
    """)
    assert out.splitlines()[-1] == "1"


def test_a_server_that_cannot_be_patched_keeps_the_batch_step_stock():
    pytest.importorskip("mlx_lm")
    out = _run("""
        import importlib
        from jitengine_qwen3_1_7b_m4pro.engine import patches
        patches.GENERATE_SHA256 = "0" * 64
        r = patches.install(selfcheck=False)
        gen = importlib.import_module("mlx_lm.generate")
        print("lazy_logprobs" in r.skipped, "lazy_logprobs" in r.applied,
              hasattr(gen.GenerationBatch._step, "_jit_source_sha256"))
    """)
    assert out.splitlines()[-1] == "True False False"


def test_the_http_server_is_built_with_the_deep_backlog():
    pytest.importorskip("mlx_lm")
    out = _run("""
        import importlib
        from jitengine_qwen3_1_7b_m4pro.engine import patches
        patches.install(selfcheck=False)
        s = importlib.import_module("mlx_lm.server")
        print(s._run_http_server.keywords["server_class"] is patches._Backlog)
    """)
    assert out.splitlines()[-1] == "True"


def test_fusion_is_refused_when_the_replaced_methods_are_not_the_pinned_source(monkeypatch):
    pytest.importorskip("mlx_lm")
    from jitengine_qwen3_1_7b_m4pro.engine import fusion

    assert fusion.guard() is None
    monkeypatch.setattr(fusion, "ATTENTION_SHA256", "0" * 64)
    assert "qwen3.Attention.__call__" in fusion.guard()
    assert fusion.fuse_model(types.SimpleNamespace(model_type="qwen3")) is False
