"""jitengine_runtime.hub: a `--model-path` that is a directory, or a Hugging Face model id fetched into the cache.

No test touches the network: `hub._snapshot_download` (or `huggingface_hub.snapshot_download` under it) is replaced."""

import errno
import fnmatch
import json
import os
from types import SimpleNamespace

import huggingface_hub
import pytest
from huggingface_hub import constants, errors

from jitengine_runtime import hub

SHA = "3b93f9053b530488405ae9fca51d4900d6ce21f0"
PINNED = {"RecursiveSI/flint-int4": SHA}


@pytest.fixture
def fetches(monkeypatch):
    """Replace the download: each call is recorded as (repo, revision, allow_patterns) and returns a folder name."""
    calls = []

    def fake(repo, rev, allow_patterns):
        calls.append((repo, rev, allow_patterns))
        return f"/cache/models--{repo.replace('/', '--')}/snapshots/{rev}"

    monkeypatch.setattr(hub, "_snapshot_download", fake)
    return calls


def lines():
    out = []
    return out, out.append


# ---------------------------------------------------------------- what a value is
@pytest.mark.parametrize(
    "value, want",
    [
        ("org/name", ("org/name", None)),
        ("org/name@abc123", ("org/name", "abc123")),
        ("RecursiveSI/flint-int4", ("RecursiveSI/flint-int4", None)),
        (
            "mlx-community/Qwen3.6-35B-A3B-4bit@refs/pr/3",
            ("mlx-community/Qwen3.6-35B-A3B-4bit", "refs/pr/3"),
        ),
        ("a.b/c_d-e", ("a.b/c_d-e", None)),
    ],
)
def test_a_model_id_is_org_name_with_an_optional_revision(value, want):
    ref = hub.parse_model_ref(value)
    assert (ref.repo, ref.revision) == want


@pytest.mark.parametrize(
    "value",
    [
        "",
        "name",
        "/abs/path",
        "./rel/path",
        "../x/y",
        "~/models/x",
        "org/name@",
        "org/na me",
        "a/b/c",
        "org/",
        "a--b/c",
        "a/b..c",
    ],
)
def test_anything_else_is_not_a_model_id(value):
    assert hub.parse_model_ref(value) is None


def test_a_directory_is_used_as_given_with_no_download_and_no_log_line(tmp_path, monkeypatch):
    def boom(*a):
        raise AssertionError("a directory must not reach the Hub")

    monkeypatch.setattr(hub, "_snapshot_download", boom)
    out, log = lines()
    assert hub.resolve_model_path(str(tmp_path), pinned=PINNED, log=log) == str(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "org" / "name").mkdir(parents=True)
    assert (
        hub.resolve_model_path("org/name", pinned=PINNED, log=log) == "org/name"
    )  # shaped like an id, but a directory
    monkeypatch.setenv("HOME", str(tmp_path))
    assert hub.resolve_model_path("~/org/name", log=log) == "~/org/name"
    assert out == []


def test_a_value_that_is_neither_a_directory_nor_an_id_is_one_plain_sentence(tmp_path):
    with pytest.raises(hub.ModelPathError) as e:
        hub.resolve_model_path(str(tmp_path / "missing"), log=lines()[1])
    assert str(e.value) == (
        f'"{tmp_path}/missing" is not a directory and not a Hugging Face model id (org/name or org/name@revision).'
    )


# ---------------------------------------------------------------- which revision
def test_the_revision_is_the_explicit_one_else_the_pin_else_main(fetches):
    log = lines()[1]
    pinned = {"RecursiveSI/flint-int4": SHA}
    assert hub.resolve_model_path("RecursiveSI/flint-int4@dev", pinned=pinned, log=log).endswith(
        "/snapshots/dev"
    )
    assert hub.resolve_model_path("RecursiveSI/flint-int4", pinned=pinned, log=log).endswith(
        f"/snapshots/{SHA}"
    )
    assert hub.resolve_model_path("org/other", pinned=pinned, log=log).endswith("/snapshots/main")
    assert hub.resolve_model_path("org/other", log=log).endswith("/snapshots/main")
    assert hub.resolve_model_path("org/other", revision="v2", pinned=pinned, log=log).endswith(
        "/snapshots/v2"
    )
    assert hub.resolve_model_path("org/other@v1", revision="v2", log=log).endswith("/snapshots/v1")
    assert [c[1] for c in fetches] == ["dev", SHA, "main", "main", "v2", "v1"]


def test_the_pin_matches_the_repository_ignoring_case_and_keeps_the_builds_spelling(fetches):
    hub.resolve_model_path("recursivesi/FLINT-int4", pinned=PINNED, log=lines()[1])
    assert fetches == [("RecursiveSI/flint-int4", SHA, None)]


def test_a_pinned_head_is_fetched_with_only_the_files_asked_for(fetches):
    hub.resolve_model_path("a/b", allow_patterns=["model.safetensors"], log=lines()[1])
    assert fetches == [("a/b", "main", ["model.safetensors"])]


def test_the_download_skips_the_weights_no_engine_reads(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda repo, **kw: seen.update(kw, repo=repo) or "/x"
    )
    hub._snapshot_download("a/b", "main", None)
    assert seen["revision"] == "main" and seen["allow_patterns"] is None
    for pattern in ("*.gguf", "*.bin", "*.pt", "*.pth", "original/*"):
        assert pattern in seen["ignore_patterns"]
    for needed in (
        "*.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "tokenizer.json",
    ):
        assert not any(fnmatch.fnmatch(needed, p) for p in seen["ignore_patterns"]), needed


# ---------------------------------------------------------------- what the user sees
def test_the_log_says_what_is_fetched_and_where_it_is_used(fetches, monkeypatch):
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", "/hub")
    out, log = lines()
    path = hub.resolve_model_path("RecursiveSI/flint-int4", pinned=PINNED, log=log)
    assert out == [
        "[hub] fetching RecursiveSI/flint-int4 at revision 3b93f9053b53 into the Hugging Face cache /hub "
        "(files already there are reused).",
        f"[hub] using {path}.",
    ]


def test_offline_the_log_says_it_only_looks_in_the_cache(fetches, monkeypatch):
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", "/hub")
    out, log = lines()
    hub.resolve_model_path("org/name", log=log)
    assert (
        out[0]
        == "[hub] looking for org/name at revision main in the Hugging Face cache /hub (offline)."
    )


def response():
    return SimpleNamespace(headers={}, request=None, status_code=404)


@pytest.mark.parametrize(
    "exc, offline, want",
    [
        (
            errors.RepositoryNotFoundError("404", response=response()),
            False,
            '"org/name" is not a directory, and Hugging Face has no such model or it is private '
            "(set HF_TOKEN to a token that can read it).",
        ),
        (
            errors.GatedRepoError("403", response=response()),
            False,
            '"org/name" is gated on Hugging Face: accept its terms on the model page, then set HF_TOKEN to a '
            "token of an account that has.",
        ),
        (
            errors.RevisionNotFoundError("404", response=response()),
            False,
            '"org/name" on Hugging Face has no revision "main".',
        ),
        (
            errors.LocalEntryNotFoundError("nothing cached"),
            True,
            'HF_HUB_OFFLINE is set and "org/name" at revision main is not in the Hugging Face cache.',
        ),
        (
            errors.LocalEntryNotFoundError("connection error"),
            False,
            'Hugging Face cannot be reached and "org/name" at revision main is not in the Hugging Face cache.',
        ),
        (
            errors.OfflineModeIsEnabled("offline"),
            True,
            'HF_HUB_OFFLINE is set and "org/name" at revision main is not in the Hugging Face cache.',
        ),
        (
            OSError(errno.ENOSPC, "No space left on device"),
            False,
            'Downloading "org/name" from Hugging Face failed: [Errno 28] No space left on device.',
        ),
    ],
)
def test_every_failure_is_one_plain_sentence_that_names_the_model(monkeypatch, exc, offline, want):
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", offline)

    def fail(*a):
        raise exc

    monkeypatch.setattr(hub, "_snapshot_download", fail)
    with pytest.raises(hub.ModelPathError) as e:
        hub.resolve_model_path("org/name", log=lines()[1])
    assert str(e.value) == want and "\n" not in want


def test_a_value_with_a_revision_is_quoted_and_the_model_named_apart(monkeypatch):
    def fail(*a):
        raise errors.RepositoryNotFoundError("404", response=response())

    monkeypatch.setattr(hub, "_snapshot_download", fail)
    with pytest.raises(hub.ModelPathError) as e:
        hub.resolve_model_path("org/nme@v1", log=lines()[1])
    assert str(e.value) == (
        '"org/nme@v1" is not a directory, and Hugging Face has no model "org/nme" or it is private '
        "(set HF_TOKEN to a token that can read it)."
    )


@pytest.mark.skipif(
    not hasattr(errors, "IncompleteSnapshotError"),
    reason="huggingface_hub before 2.0 has no such error",
)
def test_a_cut_off_snapshot_found_offline_says_so(monkeypatch):
    def fail(*a):
        raise errors.IncompleteSnapshotError("3 files are missing", snapshot_path="/x")

    monkeypatch.setattr(hub, "_snapshot_download", fail)
    with pytest.raises(
        hub.ModelPathError, match=r"holds only part of \"org/name\" at revision main"
    ):
        hub.resolve_model_path("org/name", log=lines()[1])


def test_a_snapshot_with_a_shard_missing_is_refused_even_when_the_client_could_not_tell(
    tmp_path, monkeypatch
):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}})
    )
    (tmp_path / "model-1.safetensors").write_bytes(b"x")
    monkeypatch.setattr(hub, "_snapshot_download", lambda *a: str(tmp_path))
    with pytest.raises(hub.ModelPathError) as e:
        hub.resolve_model_path("org/name", log=lines()[1])
    assert str(e.value) == (
        'The Hugging Face cache holds only part of "org/name" at revision main (1 weight files are missing); '
        "connect to Hugging Face and run again to finish the download."
    )
    (tmp_path / "model-2.safetensors").write_bytes(b"x")
    assert hub.resolve_model_path("org/name", log=lines()[1]) == str(tmp_path)


# ---------------------------------------------------------------- names
def test_a_configuration_value_is_absolute_unless_it_is_a_model_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert hub.normalize_model_ref("org/name") == "org/name"
    assert hub.normalize_model_ref("org/name@v1", str(tmp_path)) == "org/name@v1"
    assert hub.normalize_model_ref("./local") == str(tmp_path / "local")
    assert hub.normalize_model_ref("/abs/dir") == "/abs/dir"
    assert hub.normalize_model_ref("~/m").startswith("/") and not hub.normalize_model_ref(
        "~/m"
    ).startswith("~")
    (tmp_path / "org" / "name").mkdir(parents=True)
    assert hub.normalize_model_ref("org/name") == str(
        tmp_path / "org" / "name"
    )  # a directory wins, as always
    (tmp_path / "cfg" / "m" / "x").mkdir(parents=True)
    assert hub.normalize_model_ref("m/x", str(tmp_path / "cfg")) == str(
        tmp_path / "cfg" / "m" / "x"
    )


def test_a_checkpoints_name_is_its_directory_name_or_its_repository_for_a_cache_snapshot():
    assert (
        hub.checkpoint_name("/Users/x/models/Qwen3.8-Flash-Next-4bit") == "Qwen3.8-Flash-Next-4bit"
    )
    assert (
        hub.checkpoint_name("/Users/x/models/Qwen3.8-Flash-Next-4bit/") == "Qwen3.8-Flash-Next-4bit"
    )
    assert hub.checkpoint_name("~/models/flint-int4") == "flint-int4"
    snap = "/h/hub/models--mlx-community--Qwen3.8-Flash-Next-4bit/snapshots/07b5dc6c54600a359b87f1e53e7adf6351c72a2c"
    assert hub.checkpoint_name(snap) == "mlx-community--Qwen3.8-Flash-Next-4bit"
    assert hub.checkpoint_name(snap + "/") == "mlx-community--Qwen3.8-Flash-Next-4bit"
    assert hub.checkpoint_name("/x/snapshots/abc") == "abc"  # not in a cache folder
    assert hub.snapshot_repo(snap) == "mlx-community/Qwen3.8-Flash-Next-4bit"
    assert hub.snapshot_repo("/Users/x/models/Qwen3.8-Flash-Next-4bit") is None


# ---------------------------------------------------------------- install.sh: reuse what the cache holds
REV = "a" * 40


def make_cache(root, files, repo="org/name"):
    """A Hugging Face cache folder with a snapshot: {path: (blob name, bytes)}; Xet files sit in a shared store."""
    model = root / ("models--" + repo.replace("/", "--"))
    snap = model / "snapshots" / REV
    for path, (blob, data) in files.items():
        (model / "blobs").mkdir(parents=True, exist_ok=True)
        if path.endswith(
            ".safetensors"
        ):  # the Hub client's Xet layout: a link in blobs/ to a shared store
            store = root / "blobs" / blob[:2]
            store.mkdir(parents=True, exist_ok=True)
            (store / blob).write_bytes(data)
            os.symlink(os.path.relpath(store / blob, model / "blobs"), model / "blobs" / blob)
        else:
            (model / "blobs" / blob).write_bytes(data)
        (snap / path).parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.relpath(model / "blobs" / blob, (snap / path).parent), snap / path)
    return snap


def link(tmp_path, log=None, repo="org/name", rev=REV, target="dir"):
    return hub.link_cached_snapshot(
        repo, rev, str(tmp_path / target), cache_dir=str(tmp_path / "hub"), log=log or lines()[1]
    )


def test_the_large_files_the_cache_holds_are_hard_linked_and_the_small_ones_left_alone(tmp_path):
    big, small = "b" * 64, "c" * 40
    make_cache(
        tmp_path / "hub",
        {
            "model-00001.safetensors": (big, b"weights"),
            "sub/tokenizer.json": ("d" * 64, b"tok"),
            "config.json": (small, b"{}"),
        },
    )
    out, log = lines()
    assert link(tmp_path, log, target="models/name") == (2, 2)
    target = tmp_path / "models" / "name"
    assert (target / "model-00001.safetensors").read_bytes() == b"weights"
    assert os.path.samefile(
        target / "model-00001.safetensors", tmp_path / "hub" / "blobs" / big[:2] / big
    )
    assert os.stat(target / "model-00001.safetensors").st_nlink == 2
    assert (target / "sub" / "tokenizer.json").read_bytes() == b"tok"
    assert not (target / "config.json").exists()
    assert out == ["Reusing 2 files (0.0 GB) from the Hugging Face cache without copying them."]
    again, log2 = lines()
    assert link(tmp_path, log2, target="models/name") == (
        0,
        2,
    )  # linked before: in place, nothing new, no line
    assert again == []


def test_a_file_already_in_the_folder_is_left_alone_and_is_not_the_caches(tmp_path):
    make_cache(tmp_path / "hub", {"m.safetensors": ("b" * 64, b"cached")})
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "m.safetensors").write_bytes(b"partial")
    assert link(tmp_path) == (0, 0)
    assert (tmp_path / "dir" / "m.safetensors").read_bytes() == b"partial"


def test_a_cache_on_another_volume_links_nothing_and_says_nothing(tmp_path, monkeypatch):
    make_cache(
        tmp_path / "hub", {"a.safetensors": ("b" * 64, b"1"), "b.safetensors": ("c" * 64, b"2")}
    )

    def cross_volume(src, dst):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(os, "link", cross_volume)
    out, log = lines()
    assert link(tmp_path, log) == (0, 0)
    assert out == [] and not any((tmp_path / "dir").glob("*.safetensors"))


def test_a_revision_or_a_repository_the_cache_lacks_links_nothing(tmp_path):
    make_cache(tmp_path / "hub", {"a.safetensors": ("b" * 64, b"1")})
    assert link(tmp_path, repo="org/other") == (0, 0)
    assert link(tmp_path, rev="f" * 40) == (0, 0)


def test_the_cache_is_the_clients_own_unless_one_is_given(tmp_path, monkeypatch):
    make_cache(tmp_path / "hub", {"a.safetensors": ("b" * 64, b"1")})
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    assert hub.link_cached_snapshot("org/name", REV, str(tmp_path / "dir"), log=lines()[1]) == (
        1,
        1,
    )


def test_the_link_command_prints_what_it_reused_and_exits_0_only_when_the_folder_holds_cache_files(
    tmp_path, monkeypatch, capsys
):
    make_cache(tmp_path / "hub", {"a.safetensors": ("b" * 64, b"1")})
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    assert hub.main(["link", "org/name", REV, str(tmp_path / "dir")]) == 0
    assert capsys.readouterr().out.startswith("Reusing 1 files")
    assert (
        hub.main(["link", "org/name", REV, str(tmp_path / "dir")]) == 0
    )  # again: in place, silent
    assert capsys.readouterr().out == ""
    assert hub.main(["link", "org/other", REV, str(tmp_path / "other")]) == 1  # nothing cached
    monkeypatch.setattr(
        hub, "link_cached_snapshot", lambda *a, **k: (_ for _ in ()).throw(PermissionError("no"))
    )
    assert (
        hub.main(["link", "org/name", REV, str(tmp_path / "dir")]) == 1
    )  # an error: the download does it all


def test_a_model_id_is_served_by_its_name_and_a_directory_by_its_path(tmp_path):
    assert (
        hub.served_name("mlx-community/Qwen3.6-35B-A3B-4bit")
        == "mlx-community/Qwen3.6-35B-A3B-4bit"
    )
    assert hub.served_name("org/name@0123abc") == "org/name"
    assert hub.served_name(str(tmp_path)) is None
    assert hub.served_name("not an id") is None
