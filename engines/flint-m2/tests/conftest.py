"""The device check this build's end-to-end test needs (the engine's own fixtures are in ../flint/tests)."""

from __future__ import annotations

import pytest


@pytest.fixture
def requires_metal():
    """Skips the test unless MLX runs on Metal."""
    try:
        import mlx.core as mx

        ok = mx.metal.is_available()
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        pytest.skip("no MLX/Metal")
