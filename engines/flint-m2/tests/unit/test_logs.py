"""serve --log-level: a warning or an error is a str that carries its level, so plain-string logs take it unchanged,
and the leveled stderr log prints the lines at or above its level."""

import pytest

from jitengine_flint_m2 import logs

LINES = (
    "[engine] memory plan: resident",
    logs.warning("[engine] WARNING: x"),
    logs.error("[engine] y"),
)


def test_lines_stay_strings():
    plain = []
    for line in LINES:
        plain.append(line)
    assert plain == ["[engine] memory plan: resident", "[engine] WARNING: x", "[engine] y"]
    assert [logs.level_of(m) for m in plain] == ["info", "warning", "error"]
    assert logs.level_of(LINES[1] + " more") == "info"  # a new string carries no level


@pytest.mark.parametrize(
    ("level", "printed"),
    [("debug", LINES), ("info", LINES), ("warning", LINES[1:]), ("error", LINES[2:])],
)
def test_leveled_prints_at_or_above(level, printed, capsys):
    log = logs.leveled(level)
    for line in LINES:
        log(line)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.splitlines() == list(printed)
