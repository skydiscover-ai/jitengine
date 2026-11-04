"""The engine's logs: levels from `--log-level`, lines that read as the message alone, and a
handler that a closed terminal turns off rather than breaks."""

import logging

import pytest

from jitengine_qwen3_6_35b_a3b_m2 import cli, logs


@pytest.fixture
def package():
    """The package's logger, restored after the test."""
    logger = logging.getLogger(logs.__package__)
    saved = (logger.level, logger.propagate, list(logger.handlers))
    yield logger
    logger.setLevel(saved[0])
    logger.propagate = saved[1]
    logger.handlers[:] = saved[2]


def test_the_level_filters_and_each_line_is_the_message(package, capsys):
    logs.configure("warning")
    module = logging.getLogger(f"{logs.__package__}.engine.engine")
    module.info("[engine] ready in 1.0s")
    module.warning("[engine] WARNING: a kernel is off")
    assert capsys.readouterr().err == "[engine] WARNING: a kernel is off\n"


def test_configuring_twice_keeps_one_handler(package):
    logs.configure("info")
    logs.configure("debug")
    assert sum(isinstance(h, logs.PipeSafeHandler) for h in package.handlers) == 1
    assert package.level == logging.DEBUG


def test_a_closed_terminal_turns_the_handler_off_instead_of_raising(package, monkeypatch):
    class Closed:
        def write(self, _):
            raise BrokenPipeError

        def flush(self):
            pass

    logs.configure("info")
    (handler,) = [h for h in package.handlers if isinstance(h, logs.PipeSafeHandler)]
    monkeypatch.setattr("sys.stderr", Closed())
    module = logging.getLogger(f"{logs.__package__}.scheduler.scheduler")
    module.info("[engine] first")  # must not raise into the model thread
    module.info("[engine] second")
    assert handler.broken


def test_serve_takes_a_log_level():
    a = cli.build_parser().parse_args(["serve", "--model-path", "m", "--log-level", "debug"])
    assert a.log_level == "debug"
    assert cli.build_parser().parse_args(["serve", "--model-path", "m"]).log_level == "info"
