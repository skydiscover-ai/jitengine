"""An engine's logs: each module logs to `logging.getLogger(__name__)`, under its package's logger.

`configure(level, *packages)`, which an engine's CLI calls for `--log-level`, gives each package's logger one stderr
handler and writes each record as its message alone, so the lines read as before. An engine passes its own package and
the shared packages its modules log from (jitengine_qwen3_6_35b_a3b's scheduler, for one), so their lines reach the
same stderr at the same level. A closed terminal makes a write raise `BrokenPipeError`, which must never reach the
model thread, so the handler drops every record after its first failed write. An application that embeds the engine
skips `configure` and routes the packages' loggers as it routes any other.
"""

from __future__ import annotations

import logging
import sys

LEVELS = ("debug", "info", "warning", "error")


class PipeSafeHandler(logging.Handler):
    """One line per record on stderr; after a write fails, nothing more."""

    def __init__(self) -> None:
        super().__init__()
        self.broken = False

    def emit(self, record: logging.LogRecord) -> None:
        if self.broken:
            return
        try:
            sys.stderr.write(self.format(record) + "\n")
            sys.stderr.flush()
        except BrokenPipeError, OSError, ValueError:
            self.broken = True


def configure(level: str, *packages: str) -> list[logging.Logger]:
    """Send each package's records at `level` and above to stderr through one `PipeSafeHandler` per package."""
    if not packages:
        raise ValueError("configure needs the packages whose records it sends")
    loggers = []
    for package in packages:
        logger = logging.getLogger(package)
        for handler in [h for h in logger.handlers if isinstance(h, PipeSafeHandler)]:
            logger.removeHandler(handler)
        logger.addHandler(PipeSafeHandler())
        logger.setLevel(level.upper())
        logger.propagate = False
        loggers.append(logger)
    return loggers
