"""Log levels for `serve --log-level`. The engine logs one line per event through a `log(line)` callable. A warning
or an error is a `str` that carries its level (`warning(...)`, `error(...)`), so a callable that takes plain strings
(print, list.append) takes it unchanged, and `leveled(name)` prints to stderr only the lines at or above that level.
The engine logs nothing below info, so debug prints what info prints."""

from __future__ import annotations

import sys

LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40}


class _Warning(str):
    level = "warning"


class _Error(str):
    level = "error"


def warning(line: str) -> str:
    """The log line at level warning."""
    return _Warning(line)


def error(line: str) -> str:
    """The log line at level error."""
    return _Error(line)


def level_of(line: str) -> str:
    """A line's level: warning or error when made by those, else info."""
    return getattr(line, "level", "info")


def leveled(name: str):
    """The stderr log that prints the lines at or above level `name` (one of LEVELS)."""
    floor = LEVELS[name]

    def log(line: str) -> None:
        if LEVELS[level_of(line)] >= floor:
            print(line, file=sys.stderr, flush=True)

    return log
