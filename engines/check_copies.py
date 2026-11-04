"""Files that several engine builds share stay the same in each build.

copies.txt lists them, under the group of engines that share each one. Copies may differ only in
the engine's own names: its package (`src/<pkg>`), the package's upper-case form (the prefix of its
environment variables), its distribution name and its directory name.

    python engines/check_copies.py          # exit 1, with each difference, when a copy has drifted
    python engines/check_copies.py --new    # files the same in two or more builds and not listed

A change to a shared file goes to every copy in the same commit. When two builds should differ in
it from then on, the file comes off its group in that commit. pre-commit runs this check, and so
does CI's lint job.
"""

from __future__ import annotations

import difflib
import subprocess
import sys
from pathlib import Path

ENGINES = Path(__file__).resolve().parent
LIST = ENGINES / "copies.txt"


def package(engine: str) -> str:
    return "jitengine_" + engine.replace("-", "_").replace(".", "_")


def path(engine: str, rel: str) -> Path:
    return ENGINES / engine / rel.replace("<pkg>", package(engine))


def normalized(engine: str, rel: str) -> list[str]:
    """The copy's lines with the engine's own names replaced by placeholders."""
    pkg = package(engine)
    text = path(engine, rel).read_text(encoding="utf-8")
    for name, mark in (
        (pkg.upper(), "<PKG>"),
        (pkg, "<pkg>"),
        ("jitengine-" + engine, "<dist>"),
        (engine, "<engine>"),
    ):
        text = text.replace(name, mark)
    return text.splitlines()


def listed() -> list[tuple[tuple[str, ...], str]]:
    """(engines, file) for every file in copies.txt."""
    group, out = None, []
    for n, raw in enumerate(LIST.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            group = tuple(line[1:-1].split())
            if len(group) < 2 or not all((ENGINES / e / "pyproject.toml").is_file() for e in group):
                sys.exit(f"{LIST.name}:{n}: a group names two or more engine directories: {line}")
        elif group is None:
            sys.exit(f"{LIST.name}:{n}: a file before the first [engines] group: {line}")
        else:
            out.append((group, line))
    return out


def check() -> int:
    bad = []
    for group, rel in listed():
        missing = [e for e in group if not path(e, rel).is_file()]
        if missing:
            bad.append(f"{rel}: not in {', '.join(missing)}")
            continue
        first = normalized(group[0], rel)
        for other in group[1:]:
            lines = normalized(other, rel)
            if lines != first:
                diff = difflib.unified_diff(
                    first, lines, f"{group[0]}/{rel}", f"{other}/{rel}", lineterm="", n=1
                )
                shown = "\n".join("    " + d for d in list(diff)[:40])
                bad.append(f"{rel}: {group[0]} and {other} differ\n{shown}")
    if bad:
        print(f"Files listed in engines/{LIST.name} differ between builds:", *bad, sep="\n")
        print(
            "Make the change in every copy, or, if the builds should differ in this file from now"
            " on, take it off its group."
        )
        return 1
    return 0


def new() -> int:
    """Print `[engines] file` for each file the same in two or more builds and not listed for them."""
    files = subprocess.run(
        ["git", "ls-files"], cwd=ENGINES, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    same: dict[str, dict[str, list[str]]] = {}
    for f in files:
        engine, _, rel = f.partition("/")
        if not rel or not (ENGINES / engine / "pyproject.toml").is_file():
            continue
        if (ENGINES / f).stat().st_size == 0:
            continue
        rel = rel.replace(package(engine), "<pkg>")
        try:
            text = "\n".join(normalized(engine, rel))
        except UnicodeDecodeError:
            continue
        same.setdefault(rel, {}).setdefault(text, []).append(engine)
    covered: dict[str, set[str]] = {}
    for group, rel in listed():
        covered.setdefault(rel, set()).update(group)
    for rel in sorted(same):
        for group in same[rel].values():
            if len(group) > 1 and not set(group) <= covered.get(rel, set()):
                print(f"[{' '.join(sorted(group))}] {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(new() if sys.argv[1:] == ["--new"] else check())
