"""--log-level: the CLI takes it and hands it to the runtime's logs for this engine's package and every shared
package it imports, whose modules log too (jitengine_runtime.logs has the handler's own tests)."""

import ast
from pathlib import Path

from jitengine_qwen3_6_35b_a3b_m4pro import cli
from jitengine_runtime import logs


def test_serve_takes_a_log_level():
    a = cli.build_parser().parse_args(["serve", "--model-path", "m", "--log-level", "debug"])
    assert a.log_level == "debug"
    assert cli.build_parser().parse_args(["serve", "--model-path", "m"]).log_level == "info"


def test_serve_configures_the_log_packages(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(logs, "configure", lambda level, *packages: seen.append((level, packages)))
    try:
        cli.main(["serve", "--model-path", str(tmp_path / "no-such-model"), "--log-level", "debug"])
    except SystemExit:
        pass
    assert seen[:1] == [("debug", cli.LOG_PACKAGES)]


def test_the_log_packages_cover_every_shared_package_the_engine_imports():
    """A module that moves into a shared package keeps its log lines only if the CLI names that package."""
    src = Path(cli.__file__).parent
    imported = set()
    for path in src.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            names += [a.name for a in node.names] if isinstance(node, ast.Import) else []
            imported |= {n.split(".")[0] for n in names if n.startswith("jitengine_")}
    assert imported - {"jitengine_runtime"} <= set(cli.LOG_PACKAGES), imported
    assert cli.LOG_PACKAGES[0] == cli.__package__
