"""The chat control-token backstop (`Tokenizer.render_chat`): the placeholder render it compares
against is cached under the whole placeholder input, so whether a request is accepted never
depends on which requests came before it; the cache is bounded; and a refused request leaves
nothing in it. The key is the whole placeholder input, not the role sequence alone."""

from types import SimpleNamespace

import pytest

TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}"
    "{% if m.tool_calls %}<tool_call>{% for c in m.tool_calls %}{{ c.function.name }}{% endfor %}"
    "</tool_call>{% endif %}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
SPECIALS = ["<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>"]


@pytest.fixture
def checkpoint(tmp_path):
    """A word-level tokenizer whose template emits <tool_call> markers for an assistant's
    `tool_calls`, as the checkpoint's does."""
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["[UNK]", "user", "assistant", "hello", ".", "hi", "ok", "next", "f"]
    core = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]")
    tok.add_special_tokens({"additional_special_tokens": SPECIALS})
    tok.chat_template = TEMPLATE
    tok.save_pretrained(tmp_path)
    return str(tmp_path)


def _fresh(path):
    from jitengine_qwen3_8_flash_next_m4pro.tokenizer import Tokenizer

    return Tokenizer(path)


CALL = [{"type": "function", "function": {"name": "f", "arguments": {}}}]
X = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "ok", "tool_calls": CALL},
    {"role": "user", "content": "next"},
]
Y = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "ok"},
    {"role": "user", "content": "next"},
]


def test_acceptance_does_not_depend_on_earlier_requests(checkpoint):
    ids_x = _fresh(checkpoint).render_chat(X)
    ids_y = _fresh(checkpoint).render_chat(Y)
    call_id = _fresh(checkpoint).tok.convert_tokens_to_ids("<tool_call>")
    assert call_id in ids_x and call_id not in ids_y
    tok = _fresh(checkpoint)
    assert tok.render_chat(Y) == ids_y
    assert tok.render_chat(X) == ids_x  # the same roles as Y, with a tool call
    tok = _fresh(checkpoint)
    assert tok.render_chat(X) == ids_x
    assert tok.render_chat(Y) == ids_y
    assert tok.render_chat(X) == ids_x  # and back, from the cache


def test_the_cache_is_bounded(checkpoint):
    tok = _fresh(checkpoint)
    for n in range(1, 201):
        tok.render_chat([{"role": "user", "content": "hello"}] * n)
    assert len(tok._blank_cache) <= 64


def test_a_request_refused_past_the_prompt_bound_leaves_nothing_cached(checkpoint):
    from jitengine_qwen3_8_flash_next_m4pro.server.protocol import (
        MAX_PROMPT_TOKENS,
        InvalidRequestError,
        parse_request,
    )

    tok = _fresh(checkpoint)
    engine = SimpleNamespace(tokenizer=tok, vocab=64, max_kv_positions=200_000, model_name="m")
    long_msg = [{"role": "user", "content": "hello " * (MAX_PROMPT_TOKENS + 10)}]
    for i in range(3):
        body = {"messages": [*long_msg] + [{"role": "user", "content": "hi"}] * i}
        with pytest.raises(InvalidRequestError) as e:
            parse_request(engine, body, chat=True)
        assert e.value.code == "context_length_exceeded"
    assert len(tok._blank_cache) == 0
    req, _ = parse_request(engine, {"messages": Y, "max_tokens": 4}, chat=True)
    assert req.prompt_ids == _fresh(checkpoint).render_chat(Y) and len(tok._blank_cache) == 1
