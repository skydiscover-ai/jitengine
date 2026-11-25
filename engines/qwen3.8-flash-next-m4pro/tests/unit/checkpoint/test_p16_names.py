"""The pack16 file is named from its checkpoint as the expert pack is: the directory's name as always, `org--name` for a
folder of the Hugging Face cache."""

from types import SimpleNamespace

from jitengine_qwen3_8_flash_next_m4pro.checkpoint import p16_format

SHA = "0123456789abcdef" * 4


def tag(model_path: str) -> str:
    return p16_format.tag_of(
        SimpleNamespace(model_path=model_path, ckpt=SimpleNamespace(index_sha256=SHA))
    )


def test_the_name_stem_is_the_directory_name_or_the_cache_repository():
    assert tag("/Users/x/models/Some-Model-4bit") == "Some-Model-4bit-0123456789ab"
    assert tag("/h/hub/models--some-org--Some-Model-4bit/snapshots/" + "7" * 40) == (
        "some-org--Some-Model-4bit-0123456789ab"
    )
