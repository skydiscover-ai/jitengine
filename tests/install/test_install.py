"""install.sh on Linux, with fake `uname` and `sysctl` standing in for a Mac: the build it picks from
engines/builds.tsv for every chip and memory size (against the rule it had before the table), what it refuses, the
download watchdog, and the `jitengine` command it writes."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INSTALL = ROOT / "install.sh"

CHIPS = [
    "M1",
    "M1 Pro",
    "M1 Max",
    "M1 Ultra",
    "M2",
    "M2 Pro",
    "M2 Max",
    "M2 Ultra",
    "M3",
    "M3 Pro",
    "M3 Max",
    "M3 Ultra",
    "M4",
    "M4 Pro",
    "M4 Max",
    "M5",
    "M5 Pro",
    "M5 Max",
]
MEMORY = [8, 16, 18, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512]


def fake_mac(tmp_path: Path) -> Path:
    """A bin folder whose uname and sysctl answer like an Apple silicon Mac with FAKE_MEM_GB and FAKE_CHIP."""
    bin_ = tmp_path / "fakebin"
    bin_.mkdir(exist_ok=True)
    (bin_ / "uname").write_text(
        '#!/bin/bash\ncase "$1" in -s) echo Darwin ;; -m) echo arm64 ;; *) echo Darwin ;; esac\n'
    )
    (bin_ / "sysctl").write_text(
        textwrap.dedent("""\
        #!/bin/bash
        case "$2" in
          hw.memsize) echo $((FAKE_MEM_GB * 1073741824)) ;;
          machdep.cpu.brand_string) [ -n "$FAKE_CHIP" ] && echo "Apple $FAKE_CHIP" || exit 1 ;;
          *) exit 1 ;;
        esac
        """)
    )
    for f in bin_.iterdir():
        f.chmod(0o755)
    return bin_


def dry_run(tmp_path: Path, model: str, mem: int, chip: str) -> subprocess.CompletedProcess:
    env = dict(
        os.environ,
        PATH=f"{fake_mac(tmp_path)}:{os.environ['PATH']}",
        FAKE_MEM_GB=str(mem),
        FAKE_CHIP=chip,
        JITENGINE_DRY_RUN="1",
        JITENGINE_HOME=str(tmp_path / "home"),
        HOME=str(tmp_path),
    )
    return subprocess.run(
        ["bash", str(INSTALL), model], env=env, capture_output=True, text=True, timeout=60
    )


def rule_before_the_table(model: str, mem: int, chip: str) -> str | None:
    """install.sh's case statement before engines/builds.tsv, plus the two refusals the table adds or keeps: Qwen3.6
    below 16 GB (its engine refuses an 8 GB Mac's budget) and Flash-Next below 24 GB; and Qwen3-1.7B's one build, from
    24 GB, which came after the table."""
    if model == "qwen3-1.7b":
        return None if mem < 24 else "qwen3-1.7b-m4pro"
    if model == "qwen3.6":
        return None if mem < 16 else "qwen3.6-35b-a3b-" + ("m4pro" if mem >= 32 else "m2")
    if model == "flash-next":
        return (
            None
            if mem < 24
            else "qwen3.8-flash-next-" + ("m2" if chip in ("M1", "M2") else "m4pro")
        )
    return "flint-" + ("m4pro" if mem >= 40 else "m2")


@pytest.mark.parametrize("model", ["qwen3.6", "flash-next", "flint", "qwen3-1.7b"])
def test_every_chip_and_memory_size(tmp_path, model):
    for chip in CHIPS + [""]:
        for mem in MEMORY:
            out = dry_run(tmp_path, model, mem, chip)
            want = rule_before_the_table(model, mem, chip)
            if want is None:
                assert out.returncode == 1 and "need a Mac with" in out.stderr, (
                    chip,
                    mem,
                    out.stdout,
                    out.stderr,
                )
            else:
                got = re.search(r"gets the (\S+) engine", out.stdout)
                assert out.returncode == 0 and got and got.group(1) == want, (
                    chip,
                    mem,
                    want,
                    out.stdout,
                    out.stderr,
                )


def test_refusals_name_the_memory_needed(tmp_path):
    out = dry_run(tmp_path, "qwen3.6", 8, "M2")
    assert (
        out.returncode == 1
        and "the qwen3.6 engines need a Mac with 16 GB of memory or more (this one has 8 GB)"
        in out.stderr
    )
    out = dry_run(tmp_path, "flash-next", 16, "M4")
    assert out.returncode == 1 and "need a Mac with 24 GB" in out.stderr
    out = dry_run(tmp_path, "llama", 64, "M4 Max")
    assert (
        out.returncode == 1
        and "unknown model 'llama': use one of qwen3.6 flash-next flint qwen3-1.7b" in out.stderr
    )


def test_aliases_and_default(tmp_path):
    assert "qwen3.8-flash-next-m2" in dry_run(tmp_path, "flashnext", 24, "M2").stdout
    env_default = dry_run(tmp_path, "", 24, "M2")  # bash -s -- with no model: flint
    assert "gets the flint-m2 engine" in env_default.stdout


def test_needs_apple_silicon(tmp_path):
    env = dict(os.environ, JITENGINE_DRY_RUN="1")
    out = subprocess.run(
        ["bash", str(INSTALL), "flint"], env=env, capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 1 and "Apple silicon" in out.stderr


def function_text(name: str) -> str:
    """A shell function of install.sh, from `name() {` to the `}` that closes it at the start of a line."""
    src = INSTALL.read_text()
    start = src.index(f"\n{name}() {{\n") + 1
    return src[start : src.index("\n}\n", start) + 3]


def test_download_restarts_a_stalled_download_and_removes_its_partial_files(tmp_path):
    home, models = tmp_path / "home", tmp_path / "models"
    (home / "venv/bin").mkdir(parents=True)
    # the first attempt writes a partial file and hangs; the second completes
    (home / "venv/bin/hf").write_text(
        textwrap.dedent(f"""\
        #!/bin/bash
        n=$(cat {tmp_path}/attempts 2>/dev/null || echo 0); n=$((n + 1)); echo $n > {tmp_path}/attempts
        dir="$6"; mkdir -p "$dir/.cache/huggingface/download"
        if [ $n -eq 1 ]; then echo partial > "$dir/.cache/huggingface/download/a.1234.incomplete"; exec sleep 600; fi
        echo weights > "$dir/model.safetensors"
        """)
    )
    (home / "venv/bin/hf").chmod(0o755)
    script = "\n".join(
        [
            "set -euo pipefail",
            'say() { echo "==> $*"; }',
            'die() { echo "die: $*" >&2; exit 1; }',
            f"HOME_DIR={home}; VENV={home}/venv; MODELS={models}; STALL=6; REPORT=60",
            function_text("reuse_cache"),
            function_text("download"),
            "download org/model abc123 model",
            "echo returned",
        ]
    )
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "returned" in out.stdout, (out.stdout, out.stderr)
    assert "has not moved for 6 s; starting it again (1 of 5)" in out.stdout
    assert (tmp_path / "attempts").read_text().strip() == "2"
    assert (models / "model/model.safetensors").exists()
    assert not list((models / "model/.cache/huggingface/download").glob("*.incomplete"))


def test_download_gives_up_after_five_failures(tmp_path):
    home = tmp_path / "home"
    (home / "venv/bin").mkdir(parents=True)
    (home / "venv/bin/hf").write_text("#!/bin/bash\nexit 1\n")
    (home / "venv/bin/hf").chmod(0o755)
    script = "\n".join(
        [
            "set -euo pipefail",
            "say() { :; }",
            'die() { echo "die: $*" >&2; exit 1; }',
            f"HOME_DIR={home}; VENV={home}/venv; MODELS={tmp_path}/models; STALL=60; REPORT=60",
            function_text("reuse_cache"),
            function_text("download"),
            "download org/model abc123 model",
        ]
    )
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=120)
    assert out.returncode == 1 and "die: could not download org/model" in out.stderr


REV = "a" * 40
BLOB = "b" * 64  # a large file's blob in the Hugging Face cache is named by its sha256


def fake_cache(root: Path) -> Path:
    """A Hugging Face cache holding org/model at REV: a large file (a link to a blob, as the Hub client lays it out) and a
    small one."""
    model = root / "models--org--model"
    (model / "blobs").mkdir(parents=True)
    (model / "snapshots" / REV).mkdir(parents=True)
    (model / "blobs" / BLOB).write_bytes(b"weights")
    (model / "blobs" / ("c" * 40)).write_bytes(b"{}")
    os.symlink(f"../../blobs/{BLOB}", model / "snapshots" / REV / "model.safetensors")
    os.symlink(f"../../blobs/{'c' * 40}", model / "snapshots" / REV / "config.json")
    return model / "blobs" / BLOB


def download_run(
    tmp_path, *, python_exit=None, cache=None, report=60, hf_sleep=0, extra_env=None, repo_rev=REV
):
    """`download org/model REV model` from install.sh with a fake `hf` that logs what it was given. The venv's python is
    the test's own with the runtime importable (or a stub that exits `python_exit`); `cache` is the Hugging Face cache."""
    home, models = tmp_path / "home", tmp_path / "models"
    (home / "venv/bin").mkdir(parents=True)
    runtime_src = ROOT / "runtime" / "src"
    python = home / "venv/bin/python"
    if python_exit is None:
        python.write_text(f'#!/bin/bash\nPYTHONPATH={runtime_src} exec {sys.executable} "$@"\n')
    else:
        python.write_text(f"#!/bin/bash\nexit {python_exit}\n")
    python.chmod(0o755)
    log = tmp_path / "hf.log"
    (home / "venv/bin/hf").write_text(
        textwrap.dedent(f"""\
        #!/bin/bash
        dir="$6"; mkdir -p "$dir"
        echo "args: $*" >> {log}
        echo "cache: ${{HF_HUB_CACHE-unset}} update_check: ${{HF_HUB_DISABLE_UPDATE_CHECK-unset}}" >> {log}
        ls "$dir" >> {log}
        for i in $(seq 1 {hf_sleep}); do sleep 1; head -c 1048576 /dev/zero > "$dir/part$i"; done
        echo "{{}}" > "$dir/config.json"
        """)
    )
    (home / "venv/bin/hf").chmod(0o755)
    script = "\n".join(
        [
            "set -euo pipefail",
            'say() { echo "==> $*"; }',
            'die() { echo "die: $*" >&2; exit 1; }',
            f"HOME_DIR={home}; VENV={home}/venv; MODELS={models}; STALL=60; REPORT={report}",
            function_text("reuse_cache"),
            function_text("download"),
            f"download org/model {repo_rev} model",
        ]
    )
    env = dict(os.environ, HF_HUB_CACHE=str(cache or tmp_path / "no-cache"), **(extra_env or {}))
    out = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, timeout=120
    )
    return out, models / "model", log.read_text() if log.exists() else ""


def test_download_hard_links_what_the_hugging_face_cache_holds_and_fetches_the_rest_afresh(
    tmp_path,
):
    blob = fake_cache(tmp_path / "hub")
    out, model, log = download_run(tmp_path, cache=tmp_path / "hub")
    assert out.returncode == 0, (out.stdout, out.stderr)
    assert (
        "Reusing 1 files (0.0 GB) from the Hugging Face cache without copying them." in out.stdout
    )
    # the large file was linked before hf ran, so hf finds it in place; the small one is left for hf
    assert os.path.samefile(model / "model.safetensors", blob) and os.stat(blob).st_nlink == 2
    assert "model.safetensors" in log
    assert f"args: download org/model --revision {REV} --local-dir {model}" in log
    # hf looks in an empty cache of its own for this download, so a linked file that fails its check is fetched afresh
    assert f"cache: {model}/.cache/huggingface/no-hub-cache update_check: 1" in log
    assert (model / ".cache/huggingface/no-hub-cache").is_dir()


def test_download_without_a_cached_snapshot_is_the_plain_hf_download(tmp_path):
    fake_cache(tmp_path / "hub")
    out, _, log = download_run(tmp_path, cache=tmp_path / "hub", repo_rev="f" * 40)
    assert out.returncode == 0, (out.stdout, out.stderr)
    assert "Reusing" not in out.stdout
    assert os.stat(tmp_path / "hub/models--org--model/blobs" / BLOB).st_nlink == 1
    assert (
        f"cache: {tmp_path / 'hub'} update_check: 1" in log
    )  # the cache hf was given is its own: left as set


def test_download_falls_back_to_the_plain_hf_download_when_nothing_can_be_linked(tmp_path):
    """A cache on another volume (the helper exits 1): hf is not told to look elsewhere, so it copies from the cache."""
    fake_cache(tmp_path / "hub")
    out, model, log = download_run(tmp_path, cache=tmp_path / "hub", python_exit=1)
    assert out.returncode == 0, (out.stdout, out.stderr)
    assert "Reusing" not in out.stdout and not (model / "model.safetensors").exists()
    assert f"cache: {tmp_path / 'hub'} update_check: 1" in log


def test_download_says_how_much_is_there_when_hf_draws_no_byte_progress(tmp_path):
    """stderr is a pipe here, not a terminal, where hf draws no bars of bytes: every REPORT seconds the installer says how
    far the download is."""
    out, model, _ = download_run(tmp_path, report=5, hf_sleep=7)
    assert out.returncode == 0, (out.stdout, out.stderr)
    assert re.search(
        rf"  \d+\.\d GiB of org/model is in {re.escape(str(model))} so far", out.stdout
    ), out.stdout


def test_the_hf_download_runs_with_its_update_and_skill_hints_off(tmp_path):
    _, _, log = download_run(tmp_path)
    assert "update_check: 1" in log


def pinned_in_the_installer() -> dict[str, tuple[str, str, str]]:
    """install.sh's pinned checkpoints: build case -> (repository, revision, directory); the MTP head of the M4 Pro
    Qwen3.6 build under its own key."""
    src = INSTALL.read_text()
    block = src[src.index('case "$MODEL:$MAC" in') : src.index('say "This Mac (')]
    labels = list(re.finditer(r"(?m)^  ([\w.:*-]+)\)$", block))
    found = {}
    for label, nxt in zip(labels, labels[1:] + [None]):
        body = block[label.end() : nxt.start() if nxt else len(block)]
        m = re.search(r"REPO_ID=(\S+); REV=(\w+); DIR=(\S+)", body)
        if m:
            found[label.group(1)] = m.groups()
        extra = re.search(r'EXTRA_REPO="(\S+) (\w+) (\S+)"', body)
        if extra:
            found[label.group(1) + ":mtp"] = extra.groups()
    return found


def load_pinned(path: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location("pinned_under_test", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_installers_pinned_checkpoints_are_the_ones_the_engines_pin():
    """The engines pin their checkpoint in a pinned.py each (what `serve` with no --model-path fetches); install.sh
    downloads the same revision into ~/models, from a table of its own."""
    found = pinned_in_the_installer()
    assert set(found) == {
        "flint:*",
        "qwen3.6:m2",
        "qwen3.6:m4pro",
        "qwen3.6:m4pro:mtp",
        "flash-next:*",
        "qwen3-1.7b:*",
    }, found
    flint = load_pinned("engines/flint/src/jitengine_flint/pinned.py")
    m2 = load_pinned("engines/qwen3.6-35b-a3b-m2/src/jitengine_qwen3_6_35b_a3b_m2/pinned.py")
    m4 = load_pinned("engines/qwen3.6-35b-a3b-m4pro/src/jitengine_qwen3_6_35b_a3b_m4pro/pinned.py")
    flash = load_pinned("engines/qwen3.8-flash-next/src/jitengine_qwen3_8_flash_next/pinned.py")
    small = load_pinned("engines/qwen3-1.7b-m4pro/src/jitengine_qwen3_1_7b_m4pro/provenance.py")
    assert found["flint:*"][:2] == (flint.CHECKPOINT, flint.REVISION)
    assert found["qwen3.6:m2"][:2] == (m2.CHECKPOINT, m2.REVISION)
    assert found["qwen3.6:m4pro"][:2] == (m4.CHECKPOINT, m4.REVISION)
    assert found["qwen3.6:m4pro:mtp"][:2] == (m4.MTP_CHECKPOINT, m4.MTP_REVISION)
    assert found["flash-next:*"][:2] == (flash.CHECKPOINT, flash.REVISION)
    assert found["qwen3-1.7b:*"][:2] == (small.MODEL_REPO, small.MODEL_REVISION)
    for module in (flint, m2, m4, flash):
        assert module.PINNED[module.CHECKPOINT] == module.REVISION
    assert m4.PINNED == {m4.CHECKPOINT: m4.REVISION, m4.MTP_CHECKPOINT: m4.MTP_REVISION}


def test_jitengine_remembers_where_it_was_installed(tmp_path):
    """The command install.sh writes finds its engine without JITENGINE_HOME, even in a folder with spaces and quotes."""
    home = tmp_path / 'my "odd" home $x'
    (home / "bin").mkdir(parents=True)
    (home / "engine.env").write_text(
        'ENGINE_CMD=jitengine-flint-m2\nMODEL_PATH=/m\nSERVE_ARGS=""\nWIRED_MB=""\n'
        'NEED_GIB=18\nRESIDENT_GIB=25\nBUDGET_ARGS=""\n'
    )
    src = INSTALL.read_text()
    block = src[
        src.index("{ cat <<'EOF'\n#!/bin/bash\n# jitengine:") : src.index(
            '} > "$HOME_DIR/bin/jitengine"\n'
        )
    ]
    script = f"HOME_DIR={shlex.quote(str(home))}\n" + block + '} > "$HOME_DIR/bin/jitengine"\n'
    subprocess.run(["bash", "-c", script], check=True, timeout=30)
    cmd = home / "bin/jitengine"
    env = {k: v for k, v in os.environ.items() if k != "JITENGINE_HOME"}
    help_ = subprocess.run(
        ["bash", str(cmd), "--help"], env=env, capture_output=True, text=True, timeout=30
    )
    assert (
        help_.returncode == 0
        and "jitengine bench [--report]" in help_.stdout
        and "JITENGINE_BUDGET_GIB" in help_.stdout
    )
    status = subprocess.run(
        ["bash", str(cmd), "status"], env=env, capture_output=True, text=True, timeout=30
    )
    assert status.returncode == 0 and status.stdout.strip() == "Not running", (
        status.stdout,
        status.stderr,
    )


def write_jitengine(home: Path, engine_env: str) -> Path:
    """The `jitengine` command install.sh writes, with this engine.env, in home/bin."""
    (home / "bin").mkdir(parents=True, exist_ok=True)
    (home / "engine.env").write_text(engine_env)
    src = INSTALL.read_text()
    block = src[
        src.index("{ cat <<'EOF'\n#!/bin/bash\n# jitengine:") : src.index(
            '} > "$HOME_DIR/bin/jitengine"\n'
        )
    ]
    script = f"HOME_DIR={shlex.quote(str(home))}\n" + block + '} > "$HOME_DIR/bin/jitengine"\n'
    subprocess.run(["bash", "-c", script], check=True, timeout=30)
    return home / "bin/jitengine"


def serve_args(tmp_path: Path, engine: str, need_gib: int, budget_args: str, **env) -> str:
    """The command line `jitengine serve` runs: a stand-in engine in the build's environment prints it."""
    home = tmp_path / "home"
    fake = home / "venvs" / engine / "bin" / f"jitengine-{engine}"
    fake.parent.mkdir(parents=True)
    fake.write_text('#!/bin/bash\necho "ARGS: $*"\n')
    fake.chmod(0o755)
    cmd = write_jitengine(
        home,
        f'ENGINE_CMD=jitengine-{engine}\nMODEL_PATH=/m\nSERVE_ARGS=""\nWIRED_MB=""\n'
        f'NEED_GIB={need_gib}\nRESIDENT_GIB={need_gib}\nBUDGET_ARGS="{budget_args}"\n',
    )
    out = subprocess.run(
        ["bash", str(cmd), "serve"],
        env=dict(os.environ, **env),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0, (out.stdout, out.stderr)
    return out.stdout.strip()


def test_jitengine_runs_the_engine_of_the_builds_own_environment(tmp_path):
    assert serve_args(tmp_path, "flint-m2", 0, "") == "ARGS: serve --model-path /m"


def test_an_engine_with_no_memory_budget_is_never_given_one(tmp_path):
    """Qwen3-1.7B has no --memory-budget-gib: NEED_GIB=0 keeps the budget out of its command line, even when asked."""
    assert (
        serve_args(tmp_path, "qwen3-1.7b-m4pro", 0, "", JITENGINE_BUDGET_GIB="12")
        == "ARGS: serve --model-path /m"
    )
    assert serve_args(tmp_path / "b", "qwen3.6-35b-a3b-m2", 18, "", JITENGINE_BUDGET_GIB="12") == (
        "ARGS: serve --model-path /m --memory-budget-gib 12"
    )


def shared_packages(engine: str) -> list[str]:
    """The shared packages install.sh installs with a build, from its loop over engines/, run on this checkout."""
    src = INSTALL.read_text()
    start = src.index('SHARED=("$SRC/runtime")')
    block = src[start : src.index("\ndone\n", start) + 6]
    script = (
        f"SRC={shlex.quote(str(ROOT))}; ENGINE={engine}\n{block}\nprintf '%s\\n' \"${{SHARED[@]}}\""
    )
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=30, check=True
    ).stdout
    return [str(Path(p).relative_to(ROOT)) for p in out.split()]


@pytest.mark.parametrize(
    "engine, want",
    [
        ("flint-m2", ["runtime", "engines/flint"]),
        ("flint-m4pro", ["runtime", "engines/flint"]),
        ("qwen3.6-35b-a3b-m2", ["runtime", "engines/qwen", "engines/qwen3.6-35b-a3b"]),
        ("qwen3.6-35b-a3b-m4pro", ["runtime", "engines/qwen", "engines/qwen3.6-35b-a3b"]),
        ("qwen3.8-flash-next-m2", ["runtime", "engines/qwen", "engines/qwen3.8-flash-next"]),
        ("qwen3.8-flash-next-m4pro", ["runtime", "engines/qwen", "engines/qwen3.8-flash-next"]),
        ("qwen3-1.7b-m4pro", ["runtime"]),
    ],
)
def test_a_build_installs_the_shared_packages_it_names(engine, want):
    assert shared_packages(engine) == want
