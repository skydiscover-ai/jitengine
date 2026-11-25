"""Flint's 24 GB M2 build: the engine in jitengine_flint, run as `jitengine-flint-m2`. This package holds only the
build's name; its results, end-to-end check and tuned Mac are in engines/flint-m2."""

from jitengine_flint import __version__, create_engine  # noqa: F401

PROG = "jitengine-flint-m2"
