"""The tensor prefix read from the checkpoint's index: absent or unreadable gives None."""

import json

from jitengine_qwen3_8_flash_next_m2.model.config import _derive_prefix


def test_an_index_that_cannot_be_read_gives_no_prefix(tmp_path):
    idx = tmp_path / "model.safetensors.index.json"
    assert _derive_prefix(str(tmp_path)) is None  # no index
    idx.write_text("{not json")
    assert _derive_prefix(str(tmp_path)) is None
    idx.write_text(json.dumps({"metadata": {}}))  # no weight_map
    assert _derive_prefix(str(tmp_path)) is None


def test_the_prefix_comes_from_the_weight_map(tmp_path):
    names = {"language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": "a.safetensors"}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": names}))
    assert _derive_prefix(str(tmp_path)) == "language_model."
