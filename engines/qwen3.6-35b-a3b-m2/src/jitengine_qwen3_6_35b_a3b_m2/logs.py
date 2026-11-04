"""The engine's logs: each module logs to `logging.getLogger(__name__)`, under this package's logger.

`configure(level)`, which the CLI's `--log-level` sets, gives the package's logger one stderr
handler and writes each record as its message alone, so the lines read as before. A closed
terminal makes a write raise `BrokenPipeError`, which must never reach the model thread, so the
handler drops every record after its first failed write. An application that embeds the engine
skips `configure` and routes the package's logger as it routes any other.
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


def configure(level: str = "info") -> logging.Logger:
    """Send this package's records at `level` and above to stderr through one `PipeSafeHandler`."""
    logger = logging.getLogger(__package__)
    for handler in [h for h in logger.handlers if isinstance(h, PipeSafeHandler)]:
        logger.removeHandler(handler)
    logger.addHandler(PipeSafeHandler())
    logger.setLevel(level.upper())
    logger.propagate = False
    return logger
