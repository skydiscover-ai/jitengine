import threading
import time

import pytest

from jitengine_qwen3_1_7b_m4pro.engine.admission import MODEL_MAX_POSITIONS, TokenBudget, TooLong


def test_a_request_over_the_context_is_refused():
    b = TokenBudget(1000, 2000)
    with pytest.raises(TooLong) as e:
        b.acquire(1001)
    assert "prompt is too long" in str(e.value) and "1000" in str(e.value)


def test_limits_are_validated():
    with pytest.raises(ValueError):
        TokenBudget(3000, 2000)
    with pytest.raises(ValueError):
        TokenBudget(MODEL_MAX_POSITIONS + 1, MODEL_MAX_POSITIONS + 1)


def test_requests_that_fit_run_together_and_the_rest_wait():
    b = TokenBudget(1000, 2000)
    assert b.acquire(1000) and b.acquire(1000)
    started = threading.Event()

    def third():
        b.acquire(500)
        started.set()

    t = threading.Thread(target=third)
    t.start()
    time.sleep(0.2)
    assert not started.is_set() and b.snapshot()["waiting"] == 1
    b.release(1000)
    assert started.wait(5)
    t.join()
    assert b.snapshot()["inflight_tokens"] == 1500


def test_waiters_are_served_in_arrival_order_even_when_a_later_one_would_fit():
    b = TokenBudget(1000, 1000)
    b.acquire(900)
    order = []

    def want(name, need):
        b.acquire(need)
        order.append(name)
        b.release(need)

    big = threading.Thread(target=want, args=("big", 800))
    big.start()
    time.sleep(0.2)
    small = threading.Thread(target=want, args=("small", 50))  # fits now, but arrived later
    small.start()
    time.sleep(0.2)
    assert order == []
    b.release(900)
    big.join(5)
    small.join(5)
    assert order == ["big", "small"]


def test_a_client_that_leaves_while_waiting_gives_up_its_place():
    b = TokenBudget(1000, 1000)
    b.acquire(1000)
    gone = threading.Event()
    result = []
    t = threading.Thread(target=lambda: result.append(b.acquire(500, gone=gone.is_set)))
    t.start()
    time.sleep(0.2)
    gone.set()
    t.join(5)
    assert result == [False] and b.snapshot()["waiting"] == 0
    b.release(1000)
    assert b.acquire(1000)


def test_the_peak_and_counts_are_reported():
    b = TokenBudget(1000, 3000)
    b.acquire(700)
    b.acquire(800)
    b.release(700)
    snap = b.snapshot()
    assert (
        snap["peak_padded_tokens"] == 1600
        and snap["admitted"] == 2
        and snap["inflight_tokens"] == 800
    )


def test_one_long_request_and_several_short_ones_are_charged_the_padded_cache():
    # the running batch is one left-padded tensor: requests x the longest, not the sum
    b = TokenBudget(20_000, 32_768)
    b.acquire(20_000)
    # the sum (20,000 + 1,000) would fit, but the padded cache of 2 x 20,000 does not
    admitted = []
    t = threading.Thread(target=lambda: admitted.append(b.acquire(1_000)))
    t.start()
    time.sleep(0.2)
    assert admitted == [] and b.snapshot()["waiting"] == 1
    b.release(20_000)
    t.join(5)
    assert admitted == [True]
    assert b.snapshot()["padded_tokens"] == 1_000


def test_a_short_request_runs_beside_a_long_one_when_the_padded_cache_fits():
    b = TokenBudget(16_000, 32_768)
    assert b.acquire(16_000) and b.acquire(500)
    snap = b.snapshot()
    assert snap["padded_tokens"] == 32_000 and snap["inflight_tokens"] == 16_500


def test_requests_of_one_length_are_charged_their_sum():
    b = TokenBudget(1_000, 5_000)
    for _ in range(5):
        assert b.acquire(1_000)
    assert b.snapshot()["padded_tokens"] == 5_000
