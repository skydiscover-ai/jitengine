"""The tiny checkpoint's per-machine cache (tests/conftest.py) never hands a test an incomplete root:
an interrupted build leaves no empty `model/` behind, so the next build's rename onto the root
succeeds and engine tests do not fail with FileNotFoundError."""

import importlib.util
import os
import tempfile

import pytest


def _conftest():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "conftest.py")
    spec = importlib.util.spec_from_file_location("_tests_conftest_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_fixture_rebuilds_a_husk_left_by_an_interrupted_build(
    tiny_flashnext, monkeypatch, tmp_path, requires_mlx_any
):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    husk = tmp_path / "jitengine-qwen3.8-flash-next-m4pro-tiny-v1"
    (husk / "model").mkdir(parents=True)  # what the interrupted build left
    model, pack = tiny_flashnext()
    assert os.path.isfile(os.path.join(model, "config.json")) and os.path.isfile(pack)
    assert model == str(husk / "model")
    assert sorted(p.name for p in tmp_path.iterdir()) == [husk.name]  # nothing left beside it


def test_a_complete_root_is_used_as_it_is_and_a_husk_is_replaced(tmp_path):
    cb = _conftest().cached_build
    built = []

    def build(d):
        built.append(d)
        os.makedirs(os.path.join(d, "model"))
        for f in ("model/config.json", "pack"):
            open(os.path.join(d, f), "w").write("x")

    root = str(tmp_path / "root")
    os.makedirs(os.path.join(root, "model"))  # a husk
    assert cb(root, ("model/config.json", "pack"), build) == root and len(built) == 1
    assert os.path.isfile(os.path.join(root, "pack"))
    assert cb(root, ("model/config.json", "pack"), build) == root and len(built) == 1  # cached


def test_a_root_that_cannot_be_completed_raises(tmp_path, monkeypatch):
    mod = _conftest()

    def no_rename(a, b):
        raise OSError("rename refused")

    monkeypatch.setattr(mod.os, "rename", no_rename)
    root = str(tmp_path / "root")
    os.makedirs(os.path.join(root, "model"))
    with pytest.raises(RuntimeError, match="cannot put a complete test checkpoint"):
        mod.cached_build(
            root, ("model/config.json",), lambda d: os.makedirs(os.path.join(d, "model"))
        )
