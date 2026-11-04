"""The load-time kernel checks on this Mac: every check the model runs passes, at the tiny
checkpoint's geometry and at Qwen3.8-Flash-Next's (`Qwen4ExpTextConfig()`, its config values; the
checkpoint is not needed): at float32, the served compute dtype, with either routed-expert variant,
and at bfloat16, the other compute dtype the engine takes. With an MTP head (`mtp=True`), so the three
verify-pass kernels are checked too."""

import pytest


def _checks(cfg, dtype="float32", variant="split"):
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.model.kernel_checks import run_kernel_checks

    return run_kernel_checks(cfg, getattr(mx, dtype), "rb", "grouped", variant, mtp=True)


@pytest.mark.parametrize(
    "dtype, variant", [("float32", "split"), ("float32", "fused"), ("bfloat16", "split")]
)
def test_every_check_passes_at_the_real_geometry(requires_mlx, dtype, variant):
    from jitengine_qwen3_8_flash_next_m4pro.model.config import Qwen4ExpTextConfig
    from jitengine_qwen3_8_flash_next_m4pro.model.kernel_checks import NAMES

    checks = _checks(Qwen4ExpTextConfig(), dtype, variant)
    assert checks.errors == {}
    assert checks.results == dict.fromkeys(NAMES, True)


def test_every_check_passes_at_the_tiny_geometry(requires_mlx, tiny_flashnext):
    """The dense kernel's table serves only the real shapes, so it has nothing to check here."""
    from jitengine_qwen3_8_flash_next_m4pro.model.config import load_config
    from jitengine_qwen3_8_flash_next_m4pro.model.kernel_checks import NAMES

    checks = _checks(load_config(tiny_flashnext()[0]))
    assert checks.errors == {}
    assert checks.results == dict.fromkeys(set(NAMES) - {"dense_kernel"}, True)
