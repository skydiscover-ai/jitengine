"""The expert and PLE packs are named from their checkpoint: the directory's name as always, `org--name` for a folder of
the Hugging Face cache (whose own name is a revision hash)."""

from types import SimpleNamespace

import pytest

from jitengine_qwen3_8_flash_next_m2 import envs
from jitengine_qwen3_8_flash_next_m2.checkpoint import pack_paths

SHA = "0123456789abcdef" * 4


@pytest.fixture(autouse=True)
def _packs_in_one_directory(monkeypatch):
    monkeypatch.setenv(envs.PACK_DIR.name, "/packs")
    monkeypatch.delenv(envs.EXPERT_PACK.name, raising=False)
    monkeypatch.delenv(envs.PLE_PACK.name, raising=False)


def packs_of(model_path: str) -> list[str]:
    scan = SimpleNamespace(
        model_path=model_path, ckpt=SimpleNamespace(index_sha256=SHA), num_layers=4, total_rows=8
    )
    return [pack_paths.default_pack_path(scan), pack_paths.default_ple_pack_path(scan)]


def test_a_directory_checkpoint_keeps_its_directory_names_packs():
    assert packs_of("/Users/x/models/Some-Model-4bit") == [
        "/packs/Some-Model-4bit-0123456789ab.experts",
        "/packs/Some-Model-4bit-0123456789ab.ple",
    ]
    assert packs_of("/Users/x/models/Some-Model-4bit/") == packs_of(
        "/Users/x/models/Some-Model-4bit"
    )


def test_a_checkpoint_of_the_hugging_face_cache_is_named_by_its_repository_not_its_revision():
    snapshot = "/h/hub/models--some-org--Some-Model-4bit/snapshots/" + "7" * 40
    want = [
        "/packs/some-org--Some-Model-4bit-0123456789ab.experts",
        "/packs/some-org--Some-Model-4bit-0123456789ab.ple",
    ]
    assert packs_of(snapshot) == want
    assert packs_of(snapshot.replace("7" * 40, "8" * 40)) == want
