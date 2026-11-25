"""An engine's logs: levels from `--log-level`, lines that read as the message alone, every package an engine names, and
a handler that a closed terminal turns off rather than breaks."""

import logging

import pytest

from jitengine_runtime import logs

PACKAGES = ("jitengine_demo_engine", "jitengine_demo_shared")


@pytest.fixture
def loggers():
    """The two packages' loggers, restored after the test."""
    saved = []
    for name in PACKAGES:
        logger = logging.getLogger(name)
        saved.append((logger, logger.level, logger.propagate, list(logger.handlers)))
    yield [s[0] for s in saved]
    for logger, level, propagate, handlers in saved:
        logger.setLevel(level)
        logger.propagate = propagate
        logger.handlers[:] = handlers


def test_the_level_filters_and_each_line_is_the_message(loggers, capsys):
    logs.configure("warning", *PACKAGES)
    logging.getLogger("jitengine_demo_engine.engine").info("[engine] ready in 1.0s")
    logging.getLogger("jitengine_demo_engine.engine").warning("[engine] WARNING: a kernel is off")
    logging.getLogger("jitengine_demo_shared.scheduler").warning("[scheduler] WARNING: queue full")
    assert (
        capsys.readouterr().err
        == "[engine] WARNING: a kernel is off\n[scheduler] WARNING: queue full\n"
    )


def test_configuring_twice_keeps_one_handler_per_package(loggers):
    logs.configure("info", *PACKAGES)
    logs.configure("debug", *PACKAGES)
    for logger in loggers:
        assert sum(isinstance(h, logs.PipeSafeHandler) for h in logger.handlers) == 1
        assert logger.level == logging.DEBUG


def test_a_package_is_required():
    with pytest.raises(ValueError):
        logs.configure("info")


def test_a_closed_terminal_turns_the_handler_off_instead_of_raising(loggers, monkeypatch):
    class Closed:
        def write(self, _):
            raise BrokenPipeError

        def flush(self):
            pass

    logs.configure("info", *PACKAGES)
    (handler,) = [h for h in loggers[0].handlers if isinstance(h, logs.PipeSafeHandler)]
    monkeypatch.setattr("sys.stderr", Closed())
    module = logging.getLogger("jitengine_demo_engine.scheduler")
    module.info("[engine] first")  # must not raise into the model thread
    module.info("[engine] second")
    assert handler.broken
