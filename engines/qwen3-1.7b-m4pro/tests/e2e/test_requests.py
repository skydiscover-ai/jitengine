"""Request handling on the real checkpoint: streaming, logprobs, identity, bad input, disconnects."""

import json
import threading
import time

from ._http import (
    MODEL,
    chat_body,
    get,
    post,
    short_latency,
    start_stream_then_leave,
    stream_chunks,
)


def test_a_stream_sends_text_as_it_is_generated_and_ends_cleanly(shared):
    chunks, done = stream_chunks(
        shared,
        chat_body(
            "Count from one to twenty.",
            max_tokens=64,
            stream=True,
            stream_options={"include_usage": True},
        ),
    )
    content = [c for c in chunks if c.get("choices") and c["choices"][0]["delta"].get("content")]
    usage = [c["usage"] for c in chunks if c.get("usage")]
    assert done and usage, "the stream must end with [DONE] and report usage"
    # mlx-lm sends text as the detokenizer completes a segment, so a few tokens add no chunk; a
    # server that batched its output would send far fewer
    assert len(content) >= 0.6 * usage[-1]["completion_tokens"], (len(content), usage[-1])
    assert [
        c["choices"][0]["finish_reason"]
        for c in chunks
        if c.get("choices") and c["choices"][0]["finish_reason"]
    ]


def test_a_draft_model_named_in_a_request_is_refused_and_the_server_keeps_serving(shared):
    status, body = post(
        shared, "/v1/chat/completions", chat_body("Hi", max_tokens=8, draft_model="some/draft")
    )
    assert status >= 400 and "some/draft" in json.dumps(body), (status, body)
    status, body = post(shared, "/v1/chat/completions", chat_body("Hi", max_tokens=8))
    assert status == 200 and body["choices"][0]["message"]["content"], body


def test_a_whole_reply_has_the_openai_shape(shared):
    status, body = post(shared, "/v1/chat/completions", chat_body("Say hi.", max_tokens=16))
    assert status == 200 and body["object"] == "chat.completion"
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant" and choice["finish_reason"] in (
        "stop",
        "length",
    )
    assert body["usage"]["completion_tokens"] <= 16 and body["usage"]["prompt_tokens"] > 0


def test_logprobs_and_top_logprobs_are_returned_when_asked_and_only_then(shared):
    status, body = post(
        shared,
        "/v1/chat/completions",
        chat_body("Say hi.", max_tokens=6, logprobs=True, top_logprobs=3),
    )
    assert status == 200
    for item in body["choices"][0]["logprobs"]["content"]:
        assert item["logprob"] <= 0 and len(item["top_logprobs"]) == 3
        assert item["top_logprobs"][0]["logprob"] >= item["top_logprobs"][-1]["logprob"]
    status, plain = post(shared, "/v1/chat/completions", chat_body("Say hi.", max_tokens=6))
    assert status == 200 and not plain["choices"][0].get("logprobs")


def test_concurrent_requests_each_get_their_own_answer(shared):
    codes = [f"ZQ{i:02d}-{(i * 7919) % 9973:04d}" for i in range(16)]
    replies = [None] * 16

    def go(i):
        body = chat_body(
            f"Reply with exactly this code and nothing else: {codes[i]}", max_tokens=24
        )
        replies[i] = post(shared, "/v1/chat/completions", body)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    texts = [r[1]["choices"][0]["message"]["content"] for r in replies]
    assert all(r[0] == 200 for r in replies)
    foreign = [
        (i, c) for i, t in enumerate(texts) for j, c in enumerate(codes) if j != i and c in t
    ]
    assert not foreign, f"an answer holds another request's code: {foreign}"
    assert sum(codes[i] in t for i, t in enumerate(texts)) >= 14, texts


def test_bad_requests_do_not_disturb_healthy_streams_or_the_server(shared):
    healthy = [None] * 6
    long_body = chat_body("Write a story about a lighthouse keeper.", max_tokens=96, stream=True)

    def run(i):
        chunks, done = stream_chunks(shared, long_body)
        healthy[i] = (
            len(
                [c for c in chunks if c.get("choices") and c["choices"][0]["delta"].get("content")]
            ),
            done,
        )

    threads = [threading.Thread(target=run, args=(i,)) for i in range(6)]
    [t.start() for t in threads]
    time.sleep(0.5)
    bad = {
        "not json": b"{not json",
        "no messages": json.dumps({"model": MODEL}).encode(),
        "messages not a list": json.dumps({"model": MODEL, "messages": "hello"}).encode(),
        "negative max_tokens": json.dumps(chat_body("hi", max_tokens=-5)).encode(),
        "bad temperature": json.dumps(chat_body("hi", temperature="hot", max_tokens=4)).encode(),
        "empty body": b"",
        "array body": b"[1, 2, 3]",
    }
    statuses = {name: post(shared, "/v1/chat/completions", raw)[0] for name, raw in bad.items()}
    [t.join() for t in threads]
    assert (
        statuses["not json"] == 400
        and statuses["no messages"] == 400
        and statuses["negative max_tokens"] == 400
    ), statuses
    assert all(code < 600 for code in statuses.values()), statuses
    assert all(h[1] and h[0] >= 60 for h in healthy), healthy
    assert get(shared, "/health")["status"] == "ok"
    assert post(shared, "/v1/chat/completions", chat_body("Say hi.", max_tokens=4))[0] == 200


def test_a_client_that_leaves_mid_decode_frees_the_server(shared):
    baseline = min(short_latency(shared) for _ in range(3))
    for path, body in (
        (
            "/v1/chat/completions",
            chat_body("Write a very long story about a dragon.", max_tokens=4000, stream=True),
        ),
        (
            "/v1/messages",
            {
                "model": MODEL,
                "max_tokens": 4000,
                "stream": True,
                "messages": [
                    {"role": "user", "content": "Write a very long story about a dragon."}
                ],
            },
        ),
    ):
        assert start_stream_then_leave(shared, path, body) > 0
        time.sleep(2)
        assert short_latency(shared) <= baseline * 3 + 0.3, (
            f"{path}: the server kept generating after the client left"
        )
