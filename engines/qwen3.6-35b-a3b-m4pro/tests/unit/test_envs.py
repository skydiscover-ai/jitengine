"""The environment variables the engine reads go through envs.py, under the package's prefix."""

from types import SimpleNamespace

from jitengine_qwen3_6_35b_a3b_m4pro.checkpoint.pack import VERSION, default_pack_path

SCAN = SimpleNamespace(model_path="/m/Qwen3.6-35B-A3B-4bit/", index_sha256="0123456789abcdef")


def test_the_pack_dir_names_the_pack_after_its_checkpoint(monkeypatch):
    monkeypatch.delenv("JITENGINE_QWEN3_6_35B_A3B_M4PRO_EXPERT_PACK", raising=False)
    monkeypatch.setenv("JITENGINE_QWEN3_6_35B_A3B_M4PRO_PACK_DIR", "/packs")
    want = f"/packs/Qwen3.6-35B-A3B-4bit-0123456789ab.v{VERSION}.experts"
    assert default_pack_path(SCAN) == want


def test_an_expert_pack_file_overrides_the_dir(monkeypatch):
    monkeypatch.setenv("JITENGINE_QWEN3_6_35B_A3B_M4PRO_PACK_DIR", "/packs")
    monkeypatch.setenv("JITENGINE_QWEN3_6_35B_A3B_M4PRO_EXPERT_PACK", "~/x.experts")
    path = default_pack_path(SCAN)
    assert path.endswith("/x.experts") and "~" not in path
