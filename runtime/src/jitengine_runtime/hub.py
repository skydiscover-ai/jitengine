"""Hugging Face model ids for `--model-path`.

Every engine's `--model-path` takes a checkpoint directory, or a Hugging Face model id: `org/name`, or
`org/name@revision`. A directory is used as it is, with no network and no log line. Any other value that is a model id is
downloaded into the standard Hugging Face cache (`HF_HOME` or `HF_HUB_CACHE`, `HF_TOKEN` and `HF_HUB_OFFLINE` apply;
files already in the cache are reused) and the engine runs on the snapshot folder the download returns. The revision is
the explicit `@revision`, else the one a build pins for its own checkpoint (`pinned`), else `main`.

A checkpoint in the cache lives in a folder named by its revision hash, so `checkpoint_name` gives the name the packs
and caches made from it carry: the directory's own name for a directory (unchanged), `org--name` for a cache snapshot.

`python -m jitengine_runtime.hub link REPO REVISION DIR` is install.sh's: it hard-links the large files the cache already
holds for a revision into DIR, so the installer's `hf download --local-dir` verifies them in place instead of copying.

`huggingface_hub` is imported only when a model id is fetched or linked, so the rest of the runtime does not need it.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

DEFAULT_REVISION = "main"

# What no engine reads: other formats of the weights, and the unconverted weights. An engine reads config.json, the
# tokenizer files, the safetensors shards and their index (Flint also flint-int4.json), so those are fetched.
IGNORE_PATTERNS = (
    "*.gguf",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.h5",
    "*.msgpack",
    "*.onnx",
    "*.ot",
    "original/*",
)

_NAME = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_MODEL_ID = re.compile(rf"({_NAME}/{_NAME})(?:@([^\s@]+))?")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CACHE_PREFIX = "models--"

Log = Callable[[str], None]


class ModelPathError(ValueError):
    """A `--model-path` that cannot be used; the message is one plain sentence that names the value."""


@dataclass(frozen=True)
class ModelRef:
    """A model id as written: the repository, and the revision when `@revision` followed it."""

    repo: str
    revision: str | None = None


def log_to_stderr(line: str) -> None:
    """The default `log`: one line on stderr, beside the download's own progress."""
    print(line, file=sys.stderr, flush=True)


def parse_model_ref(value: str) -> ModelRef | None:
    """`org/name` or `org/name@revision` as a `ModelRef`; None for any other value."""
    match = _MODEL_ID.fullmatch(value)
    if match is None or ".." in match.group(1) or "--" in match.group(1):
        return None
    return ModelRef(match.group(1), match.group(2))


def served_name(value: str) -> str | None:
    """The name an engine serves a model by when `--model-path` was a Hugging Face model id: the repository as written
    (without `@revision`), which `/v1/models` then lists instead of the cache's snapshot folder; None for a directory."""
    if is_directory(value):
        return None
    ref = parse_model_ref(value)
    return ref.repo if ref else None


def is_directory(value: str) -> bool:
    """Whether `value` names an existing directory (a leading `~` is the home directory, as in the engines)."""
    return os.path.isdir(os.path.expanduser(value))


def normalize_model_ref(value: str, base_dir: str | None = None) -> str:
    """A `model_path` from a configuration layer: the absolute path as before, unless the value is a model id.

    A value that names a directory (relative to `base_dir`, the folder of the configuration file, or to the working
    directory) or is not shaped like a model id is made absolute, as it always was. Any other value is the id itself,
    which `resolve_model_path` fetches."""
    path = os.path.expanduser(value)
    absolute = os.path.abspath(os.path.join(base_dir, path) if base_dir else path)
    if os.path.isdir(absolute) or parse_model_ref(value) is None:
        return absolute
    return value


def _cache_folder(model_path: str) -> str | None:
    """`org--name` for a folder of the Hugging Face cache (`.../models--org--name/snapshots/<revision hash>`), else None."""
    parts = os.path.normpath(os.path.expanduser(model_path.rstrip("/"))).split(os.sep)
    if len(parts) >= 3 and parts[-2] == "snapshots" and parts[-3].startswith(_CACHE_PREFIX):
        return parts[-3][len(_CACHE_PREFIX) :]
    return None


def checkpoint_name(model_path: str) -> str:
    """The name the packs and caches made from a checkpoint carry, before the index hash.

    The directory's own name, as always, except for a folder of the Hugging Face cache, which is named `org--name` and
    not by its revision hash: the name then stays the same when the revision changes, and says what the checkpoint is."""
    return _cache_folder(model_path) or os.path.basename(model_path.rstrip("/"))


def snapshot_repo(model_path: str) -> str | None:
    """`org/name` for a folder of the Hugging Face cache, None for any other directory."""
    folder = _cache_folder(model_path)
    return folder.replace("--", "/", 1) if folder else None


def _pin(repo: str, pinned: Mapping[str, str] | None) -> tuple[str, str | None]:
    """The repository as the build spells it, and the revision it pins; the match ignores case."""
    for known, revision in (pinned or {}).items():
        if known.lower() == repo.lower():
            return known, revision
    return repo, None


def resolve_model_path(
    value: str,
    *,
    pinned: Mapping[str, str] | None = None,
    revision: str | None = None,
    allow_patterns: Sequence[str] | None = None,
    log: Log = log_to_stderr,
) -> str:
    """The checkpoint directory for a `--model-path` value.

    A directory comes back as given. A model id is downloaded into the Hugging Face cache, or found there, and the
    snapshot folder comes back. Its revision is `@revision` in the value, else `revision`, else the one `pinned`
    (repository to revision) has for it, else `main`. `allow_patterns` limits the download to those files. Raises
    `ModelPathError` when the value is neither, or the repository cannot be fetched."""
    if is_directory(value):
        return value
    ref = parse_model_ref(value)
    if ref is None:
        raise ModelPathError(
            f'"{value}" is not a directory and not a Hugging Face model id (org/name or org/name@revision).'
        )
    repo, pinned_revision = _pin(ref.repo, pinned)
    rev = ref.revision or revision or pinned_revision or DEFAULT_REVISION
    return fetch_snapshot(repo, rev, allow_patterns=allow_patterns, log=log, value=value)


def fetch_snapshot(
    repo: str,
    revision: str,
    *,
    allow_patterns: Sequence[str] | None = None,
    log: Log = log_to_stderr,
    value: str | None = None,
) -> str:
    """Download `repo` at `revision` into the Hugging Face cache (what is cached is reused) and return its snapshot folder.

    This is `resolve_model_path` for a repository that is never a directory, such as an engine's pinned MTP head;
    `value` is what the user wrote, when the errors should quote that. Raises `ModelPathError`."""
    from huggingface_hub import constants

    rev, value = revision, value or repo
    short = rev[:12] if re.fullmatch(r"[0-9a-f]{40}", rev) else rev
    if constants.HF_HUB_OFFLINE:
        log(
            f"[hub] looking for {repo} at revision {short} in the Hugging Face cache {constants.HF_HUB_CACHE} (offline)."
        )
    else:
        log(
            f"[hub] fetching {repo} at revision {short} into the Hugging Face cache {constants.HF_HUB_CACHE} "
            "(files already there are reused)."
        )
    try:
        path = _snapshot_download(repo, rev, allow_patterns)
    except Exception as exc:  # whatever the download raises becomes one sentence naming the model
        raise ModelPathError(_explain(exc, repo, rev, value)) from exc
    missing = _missing_files(path)
    if missing:
        raise ModelPathError(_incomplete(repo, rev, len(missing)))
    log(f"[hub] using {path}.")
    return path


def _snapshot_download(repo: str, rev: str, allow_patterns: Sequence[str] | None) -> str:
    """`huggingface_hub.snapshot_download` into the standard cache, without the weights no engine reads."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo,
        revision=rev,
        allow_patterns=list(allow_patterns) if allow_patterns else None,
        ignore_patterns=list(IGNORE_PATTERNS),
    )


def _missing_files(snapshot: str) -> list[str]:
    """The shards a snapshot's index names that are not in the folder: a download that was cut off, found offline.

    The Hub client checks this against the file list it cached, when it has one; the index is there for any cache."""
    import json

    index = os.path.join(snapshot, "model.safetensors.index.json")
    if not os.path.isfile(index):
        return []
    try:
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    except OSError, ValueError, KeyError, AttributeError:
        return []
    return [s for s in shards if not os.path.exists(os.path.join(snapshot, s))]


def _incomplete(repo: str, rev: str, count: int) -> str:
    return (
        f'The Hugging Face cache holds only part of "{repo}" at revision {rev[:12]} ({count} weight files are missing); '
        "connect to Hugging Face and run again to finish the download."
    )


def _explain(exc: Exception, repo: str, rev: str, value: str) -> str:
    """One sentence for a failed download: what is wrong with the model, and what to do about it."""
    from huggingface_hub import constants, errors

    if isinstance(exc, errors.GatedRepoError):
        return (
            f'"{repo}" is gated on Hugging Face: accept its terms on the model page, then set HF_TOKEN '
            "to a token of an account that has."
        )
    if isinstance(exc, errors.RepositoryNotFoundError):
        model = "no such model" if value == repo else f'no model "{repo}"'
        return (
            f'"{value}" is not a directory, and Hugging Face has {model} or it is private '
            "(set HF_TOKEN to a token that can read it)."
        )
    if isinstance(exc, errors.RevisionNotFoundError):
        return f'"{repo}" on Hugging Face has no revision "{rev}".'
    if isinstance(exc, getattr(errors, "IncompleteSnapshotError", ())):
        return (
            f'The Hugging Face cache holds only part of "{repo}" at revision {rev[:12]} and Hugging Face '
            "cannot be reached; connect and run again to finish the download."
        )
    if isinstance(exc, (errors.LocalEntryNotFoundError, errors.OfflineModeIsEnabled)):
        why = (
            "HF_HUB_OFFLINE is set"
            if constants.HF_HUB_OFFLINE
            else "Hugging Face cannot be reached"
        )
        return f'{why} and "{repo}" at revision {rev[:12]} is not in the Hugging Face cache.'
    detail = " ".join(str(exc).split()) or type(exc).__name__
    return f'Downloading "{repo}" from Hugging Face failed: {detail[:200]}.'


# ---------------------------------------------------------------- install.sh: reuse what the cache holds
class Linked(NamedTuple):
    """What `link_cached_snapshot` did: the files it linked now, and the files of the folder that are the cache's own."""

    new: int
    in_place: int


def link_cached_snapshot(
    repo: str,
    revision: str,
    local_dir: str,
    *,
    cache_dir: str | None = None,
    log: Log = print,
) -> Linked:
    """Hard-link the large files the Hugging Face cache holds for `repo` at `revision` into `local_dir`.

    A large file's blob is named by its sha256, and `hf download --local-dir` hashes a local file that has no
    metadata and keeps it when the hash is the file's, so a linked file is verified and never copied or fetched. A file
    that is already in `local_dir` is left alone, and the small files stay for `hf download` to copy. A cache on another
    volume, or a volume without hard links, links nothing: `hf download` then copies from the cache as it always did."""
    if cache_dir is None:
        from huggingface_hub import constants

        cache_dir = constants.HF_HUB_CACHE
    snapshot = os.path.join(
        cache_dir, _CACHE_PREFIX + repo.replace("/", "--"), "snapshots", revision
    )
    new = in_place = size = 0
    for root, _dirs, files in os.walk(snapshot):
        for name in files:
            source = os.path.join(root, name)
            target = os.path.join(local_dir, os.path.relpath(source, snapshot))
            if not _is_large_blob(source):
                continue
            if os.path.lexists(target):
                in_place += _same_file(source, target)
                continue
            try:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                os.link(os.path.realpath(source), target)
            except OSError:
                return _report(Linked(new, in_place + new), size, log)
            new += 1
            size += os.path.getsize(source)
    return _report(Linked(new, in_place + new), size, log)


def _same_file(a: str, b: str) -> bool:
    """Whether two paths are one file (a hard link, or a link to the same blob)."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _is_large_blob(source: str) -> bool:
    """Whether a snapshot's file is a link to a blob named by its sha256: the cache's name for a large (LFS or Xet)
    file; a small file's blob is named by its git hash, and a file that is not a link was put there by hand."""
    try:
        return _SHA256.fullmatch(os.path.basename(os.readlink(source))) is not None
    except OSError:
        return False


def _report(linked: Linked, size: int, log: Log) -> Linked:
    if linked.new:
        log(
            f"Reusing {linked.new} files ({size / 1e9:.1f} GB) from the Hugging Face cache without copying them."
        )
    return linked


def main(argv: Sequence[str] | None = None) -> int:
    """`link REPO REVISION DIR`: exit 0 when DIR holds files of the cache (linked now or before), else 1."""
    parser = argparse.ArgumentParser(prog="python -m jitengine_runtime.hub")
    sub = parser.add_subparsers(dest="cmd", required=True)
    link = sub.add_parser(
        "link", help="hard-link the large files the Hugging Face cache holds into a folder"
    )
    link.add_argument("repo")
    link.add_argument("revision")
    link.add_argument("local_dir")
    args = parser.parse_args(argv)
    try:
        linked = link_cached_snapshot(args.repo, args.revision, args.local_dir)
    except OSError, ImportError:
        return 1  # nothing reused: the download that follows does it all
    return 0 if linked.in_place else 1


if __name__ == "__main__":
    raise SystemExit(main())
