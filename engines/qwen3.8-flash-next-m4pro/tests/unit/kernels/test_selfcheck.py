"""The load-time check plumbing: what `Checks` records, the warning `report` writes, and the
tolerance `no_worse_than_stock` holds a kernel to, on MLX's CPU build."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from jitengine_qwen3_8_flash_next_m4pro.kernels import selfcheck  # noqa: E402


def _boom():
    raise RuntimeError("no such kernel")


def test_run_records_each_result_and_the_error_of_a_raise():
    checks = selfcheck.Checks()
    assert checks.run("a", lambda: True) is True
    assert checks.run("b", lambda x: x > 1, 0) is False
    assert checks.run("c", _boom) is False
    assert checks.results == {"a": True, "b": False, "c": False}
    assert checks.errors == {"c": "RuntimeError: no such kernel"}


def test_run_many_records_every_name_and_a_raise_fails_them_all():
    checks = selfcheck.Checks()
    assert checks.run_many(("x", "y"), lambda: {"x": True, "y": False, "z": True}) == {
        "x": True,
        "y": False,
    }
    assert checks.run_many(("p", "q"), _boom) == {"p": False, "q": False}
    assert checks.results == {"x": True, "y": False, "p": False, "q": False}
    assert checks.errors == {
        "p": "RuntimeError: no such kernel",
        "q": "RuntimeError: no such kernel",
    }


def test_report_names_every_kernel_that_is_off_and_why():
    assert selfcheck.report({"a": True, "b": True}, "0.32.2", "applegpu_g14g") is None
    msg = selfcheck.report(
        {"a": True, "b": False, "c": False},
        "0.32.4",
        "applegpu_g13s",
        {"c": "RuntimeError: no such kernel"},
    )
    # a, which passed, is not named
    assert msg.startswith("b: disagree with stock MLX on this Mac; ")
    assert "c (RuntimeError: no such kernel): could not run on this Mac" in msg
    assert "(mlx 0.32.4, applegpu_g13s)" in msg


def test_no_worse_than_stock_allows_another_summation_order_only():
    exact = np.linspace(-1.0, 1.0, 64)
    stock = mx.array((exact + 1e-6).astype(np.float32))
    assert selfcheck.no_worse_than_stock(mx.array((exact - 5e-6).astype(np.float32)), stock, exact)
    assert not selfcheck.no_worse_than_stock(
        mx.array((exact + 1e-4).astype(np.float32)), stock, exact
    )
    with pytest.raises(ValueError, match="stock is"):  # a wrong reference is not the kernel's
        selfcheck.no_worse_than_stock(stock, mx.array((exact + 0.1).astype(np.float32)), exact)


def test_a_16_bit_stock_may_be_two_ulps_off():
    """2^-9 of the scale for float16 (1e-3 for float32), 2^-6 for bfloat16."""
    exact = np.arange(-1024, 1024) / 1024.0  # float16 holds these and the shifted values
    for dtype, near, far in ((mx.float16, 3 * 2.0**-11, 2.0**-8), (mx.bfloat16, 2.0**-7, 2.0**-5)):
        got = mx.array(exact.astype(np.float32)).astype(dtype)
        stock = mx.array((exact + near).astype(np.float32)).astype(dtype)
        assert selfcheck.no_worse_than_stock(got, stock, exact)
        with pytest.raises(ValueError, match="stock is"):
            selfcheck.no_worse_than_stock(got, (stock + far).astype(dtype), exact)


def test_exact_matmul_reads_the_stored_values():
    lin = selfcheck.Linear(48, 64, seed=3, group_size=32)
    x = selfcheck.normal((2, 64), seed=4)
    s, b = lin.scales.astype(mx.float32), lin.biases.astype(mx.float32)
    w = selfcheck.f64(mx.dequantize(lin.weight, s, b, group_size=32, bits=4))
    want = selfcheck.f64(x) @ w.T
    assert np.allclose(lin.exact(selfcheck.f64(x)), want, rtol=0, atol=1e-5)
    assert lin.exact(selfcheck.f64(x)).shape == (2, 48)


def test_quantized_codes_are_uniform_and_entries_scale_with_1_over_k():
    k = 512
    w, s, b = selfcheck.quantized((256, k), seed=5, group_size=32)
    ones, zeros = mx.ones(s.shape), mx.zeros(s.shape)
    codes = selfcheck.f64(mx.dequantize(w, ones, zeros, group_size=32, bits=4))
    counts = np.bincount(codes.astype(np.int64).ravel(), minlength=16)
    assert counts.size == 16 and counts.min() > 0.9 * codes.size / 16
    f32 = mx.float32
    dq = selfcheck.f64(mx.dequantize(w, s.astype(f32), b.astype(f32), group_size=32, bits=4))
    assert abs(dq.mean()) < 0.05 * k**-0.5 and 0.8 < dq.var() * k < 1.25


def test_sample_spreads_over_the_range():
    assert np.array_equal(selfcheck.sample(10), np.arange(10))
    picked = selfcheck.sample(1000, 64)
    assert picked[0] == 0 and picked[-1] == 999 and picked.size == 64
    assert np.all(np.diff(picked) > 0)


def test_another_gpu_family_is_told_what_ran_without_a_warning():
    assert selfcheck.tuned_note("Apple M4 Pro", selfcheck.TUNED_GPU, checks_passed=True) is None
    assert selfcheck.tuned_note("Apple M1 Pro", "applegpu_g13s", checks_passed=True) == (
        "This build was tuned on a 48 GB M4 Pro (applegpu_g16s). It runs on this M1 Pro "
        "(applegpu_g13s): every kernel check passed. Speed will differ from the published numbers."
    )
    off = selfcheck.tuned_note("Apple M2", "applegpu_g14g", checks_passed=False)
    assert "the kernels whose checks failed are off" in off
    assert "checks every kernel at load" in selfcheck.tuned_note("Apple M2", "applegpu_g14g")
