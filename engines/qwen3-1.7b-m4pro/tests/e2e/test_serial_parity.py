"""Serial greedy answers equal stock mlx-lm's: whole, streamed and with logprobs."""

import importlib.util
import os
import sys

BENCH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmarks"
)


def _accuracy():
    sys.path.insert(0, BENCH)
    spec = importlib.util.spec_from_file_location(
        "bench_accuracy", os.path.join(BENCH, "accuracy.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_engine_answers_equal_stock_answers(checkpoint, tmp_path):
    accuracy = _accuracy()
    stock = accuracy.collect("stock", checkpoint, str(tmp_path))
    engine = accuracy.collect("engine", checkpoint, str(tmp_path))
    differing = [k for k in stock if stock[k] != engine[k]]
    assert len(stock) == 36 and not differing, differing
