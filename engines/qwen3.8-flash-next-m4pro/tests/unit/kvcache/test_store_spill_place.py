"""Where the prefix-store spill goes (`open_store_spill`): on the pack's volume but never under the
pack directory, which the engine only reads, and off (not a startup failure) where it cannot be
written (a read-only pack location must not stop the server after the full load)."""

import os
import stat
import sys

import pytest

pytest.importorskip("mlx.core")

from jitengine_qwen3_8_flash_next_m4pro.kvcache.store_spill import open_store_spill

NOCACHE = sys.platform == "darwin"  # F_NOCACHE is macOS's; elsewhere the spill opens without it


@pytest.fixture
def packs(tmp_path):
    """tmp/packs/ro/x.experts with ro read-only; the model beside packs. Permissions come back."""
    ro = tmp_path / "packs" / "ro"
    ro.mkdir(parents=True)
    (ro / "x.experts").write_bytes(b"pack")
    (tmp_path / "model").mkdir()
    ro.chmod(
        stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    )
    yield tmp_path, ro
    for d in (tmp_path / "packs", ro):
        d.chmod(0o755)


def _listing(d):
    return sorted(os.listdir(d)), os.stat(d).st_mtime_ns


def test_the_spill_goes_beside_the_pack_directory_not_in_it(packs):
    root, ro = packs
    before = _listing(ro)
    sp, why = open_store_spill(str(ro / "x.experts"), str(root / "model"), nocache=NOCACHE)
    try:
        assert why is None and sp.dir == str(root / "packs" / "jitengine-spill")
        assert _listing(ro) == before  # nothing opened, made or touched under the pack directory
    finally:
        sp.close()
    assert not (root / "packs" / "jitengine-spill").exists()  # the directory it made is gone


def test_an_unwritable_place_turns_the_spill_off_and_leaves_nothing(packs):
    root, ro = packs
    (root / "packs").chmod(0o555)
    sp, why = open_store_spill(str(ro / "x.experts"), str(root / "model"))
    assert sp is None and "jitengine-spill" in why
    assert sorted(os.listdir(root / "packs")) == ["ro"]
    sp, why = open_store_spill(str(ro / "x.experts"), str(root / "model"), str(ro / "spill"))
    assert sp is None and "only reads" in why  # under the pack directory: never written
    sp, why = open_store_spill(
        str(ro / "x.experts"), str(root / "model"), str(root / "model" / "s")
    )
    assert sp is None and "only reads" in why  # nor under the checkpoint


def test_a_named_writable_place_is_used(packs, tmp_path):
    root, ro = packs
    sp, why = open_store_spill(
        str(ro / "x.experts"), str(root / "model"), str(tmp_path / "elsewhere"), nocache=NOCACHE
    )
    try:
        assert why is None and sp.dir == str(tmp_path / "elsewhere")
    finally:
        sp.close()


def test_a_default_place_on_another_device_turns_the_spill_off(packs, monkeypatch):
    from jitengine_qwen3_8_flash_next_m4pro.kvcache import store_spill

    root, ro = packs
    real = store_spill._st_dev
    monkeypatch.setattr(
        store_spill, "_st_dev", lambda p: real(p) + (1 if "jitengine-spill" in p else 0)
    )
    sp, why = open_store_spill(str(ro / "x.experts"), str(root / "model"))
    assert sp is None and "device" in why
