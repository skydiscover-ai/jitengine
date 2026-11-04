"""The environment variables the engine reads go through envs.py, under the package's prefix."""

from types import SimpleNamespace

import pytest

from jitengine_qwen3_8_flash_next_m4pro.checkpoint.pack_format import (
    PackError,
    default_pack_path,
    default_ple_pack_path,
)

SCAN = SimpleNamespace(
    model_path="/m/Qwen3.8-Flash-Next-4bit/",
    ckpt=SimpleNamespace(index_sha256="0123456789abcdef"),
    num_layers=48,
    total_rows=1000,
)
NAMES = (
    "JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_PACK_DIR",
    "JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_EXPERT_PACK",
    "JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_PLE_PACK",
)


def test_the_pack_dir_names_both_packs_after_their_checkpoint(monkeypatch):
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_PACK_DIR", "~/packs")
    tag = "Qwen3.8-Flash-Next-4bit-0123456789ab"
    assert default_pack_path(SCAN).endswith(f"/packs/{tag}.experts")
    assert default_ple_pack_path(SCAN).endswith(f"/packs/{tag}.ple")
    assert "~" not in default_pack_path(SCAN)


def test_a_pack_file_overrides_the_dir(monkeypatch):
    monkeypatch.setenv("JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_PACK_DIR", "/packs")
    monkeypatch.setenv("JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_EXPERT_PACK", "/x.experts")
    monkeypatch.setenv("JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_PLE_PACK", "/y.ple")
    assert (default_pack_path(SCAN), default_ple_pack_path(SCAN)) == ("/x.experts", "/y.ple")


def test_without_a_location_the_error_names_the_variables(monkeypatch):
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(
        PackError,
        match=r"JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_PACK_DIR.*JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_EXPERT_PACK",
    ):
        default_pack_path(SCAN)
    with pytest.raises(
        PackError,
        match=r"JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_PACK_DIR.*JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_PLE_PACK",
    ):
        default_ple_pack_path(SCAN)


def test_an_on_off_switch_reads_on_or_off_and_refuses_anything_else(monkeypatch):
    from jitengine_qwen3_8_flash_next_m4pro import envs

    sw = envs.PLE_PREFETCH
    assert sw.name == "JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_PLE_PREFETCH"
    assert envs.PLE_HOST_CONTEXT.name == "JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_PLE_HOST_CONTEXT"
    monkeypatch.delenv(sw.name, raising=False)
    assert sw.get() is None
    for value, want in (("on", True), ("off", False), ("", None)):
        monkeypatch.setenv(sw.name, value)
        assert sw.get() is want
    monkeypatch.setenv(sw.name, "true")
    with pytest.raises(ValueError, match="expected on or off"):
        sw.get()
