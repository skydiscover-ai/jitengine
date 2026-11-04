"""The environment variables the engine reads go through envs.py, under the package's prefix."""

from types import SimpleNamespace

import pytest

from jitengine_qwen3_8_flash_next_m2.checkpoint.pack_format import (
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
    "JITENGINE_QWEN3_8_FLASH_NEXT_M2_PACK_DIR",
    "JITENGINE_QWEN3_8_FLASH_NEXT_M2_EXPERT_PACK",
    "JITENGINE_QWEN3_8_FLASH_NEXT_M2_PLE_PACK",
)


def test_the_pack_dir_names_both_packs_after_their_checkpoint(monkeypatch):
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JITENGINE_QWEN3_8_FLASH_NEXT_M2_PACK_DIR", "~/packs")
    tag = "Qwen3.8-Flash-Next-4bit-0123456789ab"
    assert default_pack_path(SCAN).endswith(f"/packs/{tag}.experts")
    assert default_ple_pack_path(SCAN).endswith(f"/packs/{tag}.ple")
    assert "~" not in default_pack_path(SCAN)


def test_a_pack_file_overrides_the_dir(monkeypatch):
    monkeypatch.setenv("JITENGINE_QWEN3_8_FLASH_NEXT_M2_PACK_DIR", "/packs")
    monkeypatch.setenv("JITENGINE_QWEN3_8_FLASH_NEXT_M2_EXPERT_PACK", "/x.experts")
    monkeypatch.setenv("JITENGINE_QWEN3_8_FLASH_NEXT_M2_PLE_PACK", "/y.ple")
    assert (default_pack_path(SCAN), default_ple_pack_path(SCAN)) == ("/x.experts", "/y.ple")


def test_without_a_location_the_error_names_the_variables(monkeypatch):
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(
        PackError,
        match=r"JITENGINE_QWEN3_8_FLASH_NEXT_M2_PACK_DIR.*JITENGINE_QWEN3_8_FLASH_NEXT_M2_EXPERT_PACK",
    ):
        default_pack_path(SCAN)
    with pytest.raises(
        PackError,
        match=r"JITENGINE_QWEN3_8_FLASH_NEXT_M2_PACK_DIR.*JITENGINE_QWEN3_8_FLASH_NEXT_M2_PLE_PACK",
    ):
        default_ple_pack_path(SCAN)
