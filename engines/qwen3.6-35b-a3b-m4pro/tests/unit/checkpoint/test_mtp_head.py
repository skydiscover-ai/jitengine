"""Where the MTP head is: beside a checkpoint directory by name (as before), or the pinned repository's file when the
checkpoint is a folder of the Hugging Face cache. The download is replaced; nothing here needs the network or MLX."""

from __future__ import annotations

import pytest

from jitengine_qwen3_6_35b_a3b_m4pro import pinned
from jitengine_qwen3_6_35b_a3b_m4pro.checkpoint import mtp_head
from jitengine_runtime import hub


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"head")
    return str(path)


# ---------------------------------------------------------------- a directory layout: unchanged
def test_the_head_is_the_checkpoints_mtp_neighbour(tmp_path):
    ckpt = tmp_path / "Qwen3.6-35B-A3B-4bit"
    ckpt.mkdir()
    assert mtp_head.local_head(str(ckpt)) is None
    head = touch(tmp_path / "Qwen3.6-35B-A3B-MTP-4bit" / "model.safetensors")
    assert mtp_head.local_head(str(ckpt)) == head
    assert mtp_head.local_head(str(ckpt) + "/") == head  # a trailing slash changes nothing


def test_a_checkpoint_without_a_bit_suffix_finds_name_plus_mtp_4bit(tmp_path):
    ckpt = tmp_path / "Qwen3.6"
    ckpt.mkdir()
    head = touch(tmp_path / "Qwen3.6-MTP-4bit" / "model.safetensors")
    assert mtp_head.local_head(str(ckpt)) == head


def test_a_head_inside_the_checkpoint_is_the_last_resort(tmp_path):
    ckpt = tmp_path / "Qwen3.6-35B-A3B-4bit"
    inside = touch(ckpt / "mtp.safetensors")
    assert mtp_head.local_head(str(ckpt)) == inside
    beside = touch(tmp_path / "Qwen3.6-35B-A3B-MTP-4bit" / "model.safetensors")
    assert mtp_head.local_head(str(ckpt)) == beside


# ---------------------------------------------------------------- a folder of the Hugging Face cache
def snapshot_of(repo, rev="a" * 40):
    return f"/h/hub/models--{repo.replace('/', '--')}/snapshots/{rev}"


class Fetches(list):
    """The calls the replaced download got, as (repo, revision, allow_patterns); `empty` makes it return a folder with
    no head file in it."""

    empty = False


@pytest.fixture
def fetched(monkeypatch, tmp_path):
    calls = Fetches()
    calls.folder = tmp_path / "mtp-snapshot"

    def fake(repo, rev, allow_patterns):
        calls.append((repo, rev, allow_patterns))
        calls.folder.mkdir(exist_ok=True)
        if not calls.empty:
            touch(calls.folder / "model.safetensors")
        return str(calls.folder)

    monkeypatch.setattr(hub, "_snapshot_download", fake)
    return calls


def test_the_pinned_checkpoint_of_the_cache_gets_the_pinned_head_fetched(fetched):
    log, warn = [], []
    head = mtp_head.hub_head(
        snapshot_of(pinned.CHECKPOINT, pinned.REVISION), log.append, warn.append
    )
    assert fetched == [(pinned.MTP_CHECKPOINT, pinned.MTP_REVISION, ["model.safetensors"])]
    assert head == str(fetched.folder / "model.safetensors") and warn == []
    assert log[0].startswith(
        f"[hub] fetching {pinned.MTP_CHECKPOINT} at revision {pinned.MTP_REVISION[:12]}"
    )


def test_the_pinned_checkpoint_is_matched_ignoring_case(fetched):
    assert mtp_head.hub_head(snapshot_of(pinned.CHECKPOINT.lower()), print, print) is not None


def test_nothing_is_fetched_for_a_directory_or_another_repository(fetched):
    assert mtp_head.hub_head("/Users/x/models/Qwen3.6-35B-A3B-4bit", print, print) is None
    assert mtp_head.hub_head(snapshot_of("org/other-4bit"), print, print) is None
    assert fetched == []


def test_a_head_that_cannot_be_fetched_is_one_warning_and_no_head(monkeypatch):
    def fail(repo, rev, allow_patterns):
        raise OSError("Network is unreachable")

    monkeypatch.setattr(hub, "_snapshot_download", fail)
    warn = []
    assert mtp_head.hub_head(snapshot_of(pinned.CHECKPOINT), print, warn.append) is None
    assert warn == [
        f'[engine] no MTP head, so one token is decoded per pass: Downloading "{pinned.MTP_CHECKPOINT}" from '
        "Hugging Face failed: Network is unreachable."
    ]


def test_a_snapshot_without_the_head_file_is_no_head(fetched):
    fetched.empty = True
    assert mtp_head.hub_head(snapshot_of(pinned.CHECKPOINT), print, print) is None
