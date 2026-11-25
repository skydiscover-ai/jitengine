"""Token counting and request validation in front of the admission budget."""

import io
import types

import pytest

from jitengine_qwen3_1_7b_m4pro.engine.admission import TokenBudget
from jitengine_qwen3_1_7b_m4pro.server import limits


class Tok:
    """Joins message text; fails on a content-part list, as a plain template would."""

    def apply_chat_template(self, messages, tools=None, tokenize=False, **kw):
        return " ".join(m["content"] for m in messages)

    def encode(self, text, add_special_tokens=False):
        return text.split()


def _handler(max_tokens=8):
    h = types.SimpleNamespace()
    h.response_generator = types.SimpleNamespace(
        model_provider=types.SimpleNamespace(tokenizer=Tok()),
        cli_args=types.SimpleNamespace(chat_template_args={}),
    )
    h.max_tokens = max_tokens
    h.sent = {}
    h.wfile = io.BytesIO()
    h._set_completion_headers = lambda status: h.sent.setdefault("status", status)
    h.send_header = lambda *a: None
    h.end_headers = lambda: None
    return h


def _request(messages):
    return types.SimpleNamespace(
        request_type="chat", messages=messages, tools=None, prompt=None, role_mapping=None
    )


def test_content_part_lists_are_counted_as_their_text():
    pytest.importorskip("mlx_lm")
    parts = [{"type": "text", "text": "one two"}, {"type": "text", "text": " three four five"}]
    request = _request([{"role": "user", "content": parts}])
    assert limits.prompt_tokens(_handler(), request) == 5
    assert request.messages[0]["content"] == parts  # the request itself is left as it was


def test_a_message_without_content_counts_as_empty_not_as_a_failure():
    pytest.importorskip("mlx_lm")
    request = _request([{"role": "assistant", "content": None}, {"role": "user", "content": "a b"}])
    assert limits.prompt_tokens(_handler(), request) == 2


@pytest.mark.parametrize("value", [0, -1, "100", 5.5, True, None, [3]])
def test_a_max_tokens_that_is_not_a_positive_integer_is_a_400(value):
    cls = types.SimpleNamespace(handle_completion=lambda self, request, stop: "ran")
    limits.install(cls, TokenBudget(1000, 2000))
    h = _handler(value)
    assert cls.handle_completion(h, _request([{"role": "user", "content": "a"}]), []) is None
    assert h.sent["status"] == 400
    assert b"invalid_max_tokens" in h.wfile.getvalue()


def test_a_valid_max_tokens_runs_and_is_charged():
    pytest.importorskip("mlx_lm")
    budget = TokenBudget(1000, 2000)
    cls = types.SimpleNamespace(handle_completion=lambda self, request, stop: "ran")
    limits.install(cls, budget)
    h = _handler(8)
    h.connection = None
    assert cls.handle_completion(h, _request([{"role": "user", "content": "a b c"}]), []) == "ran"
    snap = budget.snapshot()
    assert snap["admitted"] == 1 and snap["peak_padded_tokens"] == 11
