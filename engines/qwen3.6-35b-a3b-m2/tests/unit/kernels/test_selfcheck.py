"""A kernel whose self-check fails or raises is switched off and named in the warning."""


def test_a_kernel_that_fails_its_check_is_named_in_words(requires_mlx_any):
    from jitengine_qwen3_6_35b_a3b_m2.kernels.selfcheck import report

    assert report({"dense_v4": True, "exact_sdpa": None}, "0.32.2", "applegpu_g14g") is None
    msg = report(
        {"dense_v4": False, "moe_v4": False, "exact_sdpa": True}, "0.32.4", "applegpu_g13s"
    )
    assert (
        msg.startswith("dense_v4, moe_v4: not bit-identical") and "mlx 0.32.4, applegpu_g13s" in msg
    )
    assert "the ones built on them are off" in msg and "slower than the published numbers" in msg


def test_a_check_that_raises_turns_its_kernel_off_and_says_why(requires_mlx_any):
    from jitengine_qwen3_6_35b_a3b_m2.kernels.selfcheck import Checks, report

    def broken(*_):
        raise RuntimeError("no such kernel")

    checks = Checks()
    assert checks.run("exact_sdpa", lambda: True) is True
    assert checks.run("gdn_rows", broken) is False
    assert checks.run_many(("moe_v4", "rows_v4"), broken) == {"moe_v4": False, "rows_v4": False}
    assert checks.results == {
        "exact_sdpa": True,
        "gdn_rows": False,
        "moe_v4": False,
        "rows_v4": False,
    }
    msg = report(checks.results, "0.32.2", "applegpu_g14g", checks.errors)
    assert "gdn_rows (RuntimeError: no such kernel)" in msg and "could not run on this Mac" in msg
    assert "not bit-identical" not in msg


def test_one_check_of_several_kernels_gives_each_its_result(requires_mlx_any):
    from jitengine_qwen3_6_35b_a3b_m2.kernels.selfcheck import Checks

    checks = Checks()
    got = checks.run_many(("moe_v4", "rows_v4"), lambda: {"moe_v4": 1, "rows_v4": 0})
    assert got == checks.results == {"moe_v4": True, "rows_v4": False} and not checks.errors


def test_another_gpu_family_is_told_what_ran_without_a_warning(requires_mlx_any):
    from jitengine_qwen3_6_35b_a3b_m2.kernels.selfcheck import TUNED_GPU, tuned_note

    assert tuned_note("Apple M2", TUNED_GPU, checks_passed=True) is None
    assert tuned_note("Apple M1 Pro", "applegpu_g13s", checks_passed=True) == (
        "This build was tuned on a 24 GB M2 Mac (applegpu_g14g). It runs on this M1 Pro "
        "(applegpu_g13s): every kernel check passed. Speed will differ from the published numbers."
    )
    assert "the kernels whose checks failed are off" in tuned_note(
        "Apple M1 Pro", "applegpu_g13s", checks_passed=False
    )
    assert "checks every kernel at load" in tuned_note("Apple M1 Pro", "applegpu_g13s")
