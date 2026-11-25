"""Neighbouring workloads do not regress: c=1, 4, 8 and 32 against stock in the same sitting.

Throughput depends on the state of the Mac, so this runs only when asked and compares the engine
with stock measured alongside it, with a tolerance for the noise of one run each."""

import importlib.util
import os
import sys

import pytest

BENCH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmarks"
)
SLOW = pytest.mark.skipif(
    not os.environ.get("JITENGINE_QWEN3_1_7B_M4PRO_E2E_SLOW"),
    reason="set JITENGINE_QWEN3_1_7B_M4PRO_E2E_SLOW=1",
)
TOLERANCE = 0.90  # the engine must reach this fraction of stock's speed at a guard concurrency


def _runner():
    sys.path.insert(0, BENCH)
    spec = importlib.util.spec_from_file_location("bench_run", os.path.join(BENCH, "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@SLOW
@pytest.mark.parametrize("concurrency", [1, 4, 8, 32])
def test_engine_is_not_slower_than_stock(checkpoint, tmp_path, concurrency):
    run = _runner()
    out = tmp_path / f"guard{concurrency}.json"
    run.main(
        [
            "--model-path",
            checkpoint,
            "--concurrency",
            str(concurrency),
            "--repeats",
            "2",
            "--settle",
            "3",
            "--out",
            str(out),
        ]
    )
    import json

    summary = json.load(open(out))["summary"]
    assert summary["engine"]["all_within_bounds"] and summary["stock"]["all_within_bounds"]
    assert (
        summary["engine"]["tokens_per_sec_median"]
        >= TOLERANCE * summary["stock"]["tokens_per_sec_median"]
    ), summary
