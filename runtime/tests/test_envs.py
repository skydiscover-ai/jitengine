"""jitengine_runtime.envs: what an engine's environment variables read as."""

import pytest

from jitengine_runtime.envs import EnvPath, EnvSwitch


def test_a_path_expands_the_home_directory_and_unset_or_empty_is_none(monkeypatch):
    v = EnvPath("JITENGINE_TEST_PATH")
    monkeypatch.delenv(v.name, raising=False)
    assert v.get() is None
    monkeypatch.setenv(v.name, "")
    assert v.get() is None
    monkeypatch.setenv(v.name, "~/packs")
    assert v.get() is not None and not v.get().startswith("~") and v.get().endswith("/packs")
    monkeypatch.setenv(v.name, "/abs/p")
    assert v.get() == "/abs/p"


def test_a_switch_reads_on_and_off_and_refuses_anything_else(monkeypatch):
    v = EnvSwitch("JITENGINE_TEST_SWITCH")
    monkeypatch.delenv(v.name, raising=False)
    assert v.get() is None
    monkeypatch.setenv(v.name, "on")
    assert v.get() is True
    monkeypatch.setenv(v.name, "off")
    assert v.get() is False
    monkeypatch.setenv(v.name, "yes")
    with pytest.raises(ValueError, match="JITENGINE_TEST_SWITCH='yes': expected on or off"):
        v.get()
