"""Stop strings and the fault-injection seam."""

from jitengine_flint_m4pro.engine import faults
from jitengine_flint_m4pro.tokenizer import StopMatcher


def expect_raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return
    raise AssertionError(f"{fn.__name__}{a} did not raise {exc.__name__}")


def test_stops_and_faults():
    m = StopMatcher(["</end>", "STOP"])
    assert m.find("abc STOP x </end>") == 4 and m.find("abc") is None
    assert m.safe_prefix("abcdef") == 1
    assert faults.parse("nonfinite@5") == faults.Fault("nonfinite", 5)
    for bad in (
        "nonfinite",
        "bogus@1",
        "nonfinite@0",
        "nonfinite@x",
        3,
        "nonfinite@5:persist",
        "kernel_stall@0:persist",
        "kernel_stall@:persist",
    ):
        expect_raises(ValueError, faults.parse, bad)
    # kernel_stall: in-kernel; :persist = every pass from the step on
    f, fp = faults.parse("kernel_stall@5"), faults.parse("kernel_stall@5:persist")
    assert f == faults.Fault("kernel_stall", 5) and fp == faults.Fault("kernel_stall", 5, True)
    assert [faults.stalls(f, j) for j in (4, 5, 6)] == [False, True, False]
    assert [faults.stalls(fp, j) for j in (4, 5, 6, 99)] == [False, True, True, True]
    assert not faults.stalls(faults.parse("nonfinite@5"), 5) and not faults.stalls(None, 5)
    faults.fire_host(f, 5)  # in-kernel only: nothing raised on the host
