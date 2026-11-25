"""A pack is named from its checkpoint: the directory's name as always, `org--name` for a folder of the Hugging Face cache
(whose own name is a revision hash)."""

from types import SimpleNamespace

import pytest

from jitengine_qwen3_6_35b_a3b.checkpoint import pack
from jitengine_qwen3_6_35b_a3b_m2 import envs
from jitengine_qwen3_6_35b_a3b_m2.checkpoint import pack_paths

SHA = "0123456789abcdef" * 4
SUFFIX = f"-0123456789ab.v{pack.VERSION}.experts"


@pytest.fixture(autouse=True)
def _no_pack_override(monkeypatch):
    monkeypatch.delenv(envs.EXPERT_PACK.name, raising=False)


def name_of(model_path: str) -> str:
    scan = SimpleNamespace(model_path=model_path, index_sha256=SHA)
    return pack_paths.default_pack_path(scan).rsplit("/", 1)[1]


def test_a_directory_checkpoint_keeps_its_directory_names_pack():
    assert name_of("/Users/x/models/Some-Model-4bit") == "Some-Model-4bit" + SUFFIX
    assert name_of("/Users/x/models/Some-Model-4bit/") == "Some-Model-4bit" + SUFFIX


def test_a_checkpoint_of_the_hugging_face_cache_is_named_by_its_repository_not_its_revision():
    snapshot = "/h/hub/models--some-org--Some-Model-4bit/snapshots/" + "7" * 40
    assert name_of(snapshot) == "some-org--Some-Model-4bit" + SUFFIX
    assert name_of(snapshot.replace("7" * 40, "8" * 40)) == "some-org--Some-Model-4bit" + SUFFIX
