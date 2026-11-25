import json

from jitengine_qwen3_1_7b_m4pro.engine import config
from jitengine_qwen3_1_7b_m4pro.provenance import MODEL_REVISION

GOOD = {**config.EXPECTED, "quantization": {"bits": 4, "group_size": 64}}


def _dir(tmp_path, cfg, weights=True, name="model"):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg))
    if weights:
        (d / "model.safetensors").write_bytes(b"")
    return str(d)


def test_good_directory_is_ok_with_a_revision_warning(tmp_path):
    r = config.validate(_dir(tmp_path, GOOD))
    assert r["ok"] and r["problems"] == [] and r["revision"] is None and r["warnings"]


def test_measured_revision_in_the_path_has_no_warning(tmp_path):
    snap = tmp_path / "snapshots"
    snap.mkdir()
    r = config.validate(_dir(snap, GOOD, name=MODEL_REVISION))
    assert r["ok"] and r["revision"] == MODEL_REVISION and r["warnings"] == []


def test_other_revision_warns(tmp_path):
    r = config.validate(_dir(tmp_path, GOOD, name="a" * 40))
    assert r["ok"] and any("differs" in w for w in r["warnings"])


def test_wrong_bits_and_size_are_refused(tmp_path):
    bad = {**GOOD, "hidden_size": 4096, "quantization": {"bits": 8, "group_size": 64}}
    r = config.validate(_dir(tmp_path, bad))
    assert not r["ok"] and len(r["problems"]) == 2


def test_missing_weights_and_config(tmp_path):
    assert not config.validate(_dir(tmp_path, GOOD, weights=False))["ok"]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not config.validate(str(empty))["ok"]
