"""Fixtures for this engine's tests: Flint's config and safetensors headers, and the device skip markers.

tests/data/flint-int4 holds the checkpoint's config.json and flint-int4.json and the headers of its
five safetensors files, with no tensor data (RecursiveSI/flint-int4 at 3b93f905, Apache-2.0): the
startup checks and the weight-bank layout read only those. Tests that build an Engine on the real
checkpoint ($JITENGINE_FLINT_M2_TEST_CHECKPOINT, default ~/models/flint-int4) need it and the M4
Pro this build is tuned on.
"""

from __future__ import annotations

import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_ENV = "JITENGINE_FLINT_M2_TEST_CHECKPOINT"


def _metal() -> bool:
    try:
        import mlx.core as mx

        return mx.metal.is_available()
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture
def flint_headers():
    """A checkpoint directory with Flint's config, its manifest and its safetensors headers only."""
    return os.path.join(HERE, "data", "flint-int4")


@pytest.fixture
def requires_metal():
    """Skips the test unless MLX runs on Metal."""
    if not _metal():
        pytest.skip("no MLX/Metal")


@pytest.fixture
def requires_m4pro():
    """Skips the test off the M4 Pro the build is tuned on (device.TUNED): these tests check its measured
    configuration, the megakernel at G = 40 on the 20-core GPU, and its recorded outputs."""
    if not _metal():
        pytest.skip("no MLX/Metal")
    from jitengine_flint_m2 import device

    facts = device.read_facts()
    if facts.tuned_grid is None:
        pytest.skip(
            f"these tests check the build tuned on {sorted(device.TUNED)}; this GPU is {facts.key}"
        )


@pytest.fixture
def m4pro_checkpoint(requires_m4pro):
    """The real checkpoint on the M4 Pro ($JITENGINE_FLINT_M2_TEST_CHECKPOINT, default ~/models/flint-int4)."""
    path = os.path.expanduser(os.environ.get(CHECKPOINT_ENV) or "~/models/flint-int4")
    if not os.path.isfile(os.path.join(path, "flint-int4.json")):
        pytest.skip(f"no flint-int4 checkpoint at {path} (set {CHECKPOINT_ENV})")
    return path
