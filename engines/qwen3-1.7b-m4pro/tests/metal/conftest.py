import pytest


@pytest.fixture(scope="session", autouse=True)
def _need_metal():
    try:
        import mlx.core as mx

        ok = mx.metal.is_available()
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        pytest.skip("the Metal tests need a Mac with Metal", allow_module_level=True)
