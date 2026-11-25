"""Context limit, token budget and queueing on the real checkpoint."""

import json
import os
import threading
import time
import urllib.error
import urllib.request

import pytest

from .conftest import Server

MODEL = "mlx-community/Qwen3-1.7B-4bit"
WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november".split()
)
SLOW = pytest.mark.skipif(
    not os.environ.get("JITENGINE_QWEN3_1_7B_M4PRO_E2E_SLOW"),
    reason="set JITENGINE_QWEN3_1_7B_M4PRO_E2E_SLOW=1",
)


def text(n_words: int, seed: int) -> str:
    return " ".join(WORDS[(seed + i * 7) % len(WORDS)] + str(seed * 31 + i) for i in range(n_words))


def post(server, path, body, timeout=600):
    req = urllib.request.Request(
        f"{server.url}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def chat(server, n_words, seed, max_tokens=4):
    return post(
        server,
        "/v1/chat/completions",
        {
            "model": MODEL,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [{"role": "user", "content": text(n_words, seed)}],
        },
    )


def get(server, path):
    return json.load(urllib.request.urlopen(f"{server.url}{path}", timeout=10))


@pytest.fixture(scope="module")
def small(checkpoint):
    srv = Server(checkpoint, "--max-context", "3000", "--max-inflight-tokens", "3000")
    yield srv
    srv.stop()


def test_models_lists_the_context(small):
    entry = get(small, "/v1/models")["data"][0]
    assert entry["id"] == MODEL and entry["max_model_len"] == 3000


def test_a_prompt_over_the_context_is_refused_with_the_overflow_code(small):
    status, body = chat(small, 3000, 1)  # far over 3000 tokens
    assert status == 400 and body["error"]["code"] == "context_length_exceeded"
    assert "prompt is too long" in body["error"]["message"]
    assert get(small, "/health")["status"] == "ok"


def test_max_tokens_counts_toward_the_context(small):
    status, body = chat(small, 300, 2, max_tokens=5000)
    assert status == 400 and "context" in body["error"]["message"]


def test_the_messages_api_reports_it_as_prompt_too_long(small):
    status, body = post(
        small,
        "/v1/messages",
        {"model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": text(3000, 3)}]},
    )
    assert status == 400 and body["error"]["message"].startswith("prompt is too long")


def test_requests_over_the_budget_wait_in_arrival_order_and_all_succeed(small):
    done = []
    results = [None] * 5

    def go(i):
        results[i] = chat(small, 420, 10 + i)[0]  # about 1,850 tokens each; no two fit together
        done.append(i)

    threads = []
    for i in range(5):
        t = threading.Thread(target=go, args=(i,))
        t.start()
        threads.append(t)
        time.sleep(0.4)
    [t.join() for t in threads]
    assert results == [200] * 5
    assert done == sorted(done), f"served out of arrival order: {done}"
    adm = get(small, "/stats")["admission"]
    assert adm["peak_padded_tokens"] <= 3000 and adm["padded_tokens"] == 0 and adm["waiting"] == 0


@SLOW
def test_many_long_prompts_queue_instead_of_crashing_the_server(checkpoint):
    """Four prompts of about 23k tokens crash stock mlx-lm's generation thread for good."""
    srv = Server(checkpoint)
    try:
        results = [None] * 4
        threads = [
            threading.Thread(target=lambda i=i: results.__setitem__(i, chat(srv, 5000, 20 + i)[0]))
            for i in range(4)
        ]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert results == [200] * 4
        assert get(srv, "/health")["status"] == "ok"
        assert get(srv, "/stats")["admission"]["peak_padded_tokens"] <= 32768
    finally:
        srv.stop()


def test_the_prompt_cache_stays_near_its_cap(checkpoint):
    """Stock keeps 10 finished requests' KV caches with no byte limit; an agent session filled 9.6 GB.

    mlx-lm trims the cache to its cap when the next request starts, so right after a request it
    may hold the cap plus the entry that request just left."""
    cap_gib, entry_gib = 1.0, 1.5  # a request of about 9k tokens leaves about 1.3 GiB
    srv = Server(checkpoint, "--prompt-cache-gib", str(cap_gib))
    try:
        for i in range(6):  # all kept, they would hold about 7.7 GiB
            assert chat(srv, 2000, 40 + i)[0] == 200
            held = get(srv, "/stats")["prompt_cache"]["bytes"]
            assert held <= (cap_gib + entry_gib) * 2**30, (
                f"cache holds {held / 2**30:.2f} GiB after request {i}"
            )
        assert get(srv, "/stats")["prompt_cache"]["sequences"] >= 1
    finally:
        srv.stop()


def _mixed(srv, first_words, then_words, count, gap=0.3):
    """One request of `first_words`, then `count` of `then_words`; the status of each."""
    results = {}

    def go(name, words, seed):
        results[name] = chat(srv, words, seed)[0]

    threads = [threading.Thread(target=go, args=("first", first_words, 70))]
    threads += [
        threading.Thread(target=go, args=(f"then{i}", then_words, 80 + i)) for i in range(count)
    ]
    for t in threads:
        t.start()
        time.sleep(gap)
    [t.join() for t in threads]
    return results


@SLOW
@pytest.mark.parametrize("order", ["long_first", "short_first"])
def test_mixed_lengths_are_charged_the_padded_cache(checkpoint, order):
    """One 20k-token request with eight 1k-token ones: the running batch is one left-padded tensor,
    so a sum of lengths (about 28k) would admit them together and the padded cache (9 x 20k tokens,
    about 19 GiB) would run the Mac out of memory."""
    srv = Server(checkpoint)
    try:
        long_words, short_words = 4300, 220
        if order == "long_first":
            results = _mixed(srv, long_words, short_words, 8)
        else:
            results = _mixed(srv, short_words, long_words, 1)  # a short one, then the long one
            more = _mixed(srv, short_words, short_words, 7)
            results.update({f"more{k}": v for k, v in more.items()})
        assert set(results.values()) == {200}, results
        assert get(srv, "/health")["status"] == "ok"
        stats = get(srv, "/stats")
        assert stats["admission"]["peak_padded_tokens"] <= 32768
        assert stats["memory"]["peak_bytes"] < 14 * 2**30, stats["memory"]
    finally:
        srv.stop()
