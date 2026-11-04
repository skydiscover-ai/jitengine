"""A /v1/completions prompt is bounded before it is encoded or converted: text past
`MAX_PROMPT_CHARS` is refused as chat's rendered text is, and an id list past `MAX_PROMPT_TOKENS` is
refused before any element is read, so a refused request costs work in proportion to the bound, not
to the 32 MiB body limit. Model-free: a recording stand-in tokenizer."""

import json
import os
from types import SimpleNamespace

import pytest

from jitengine_qwen3_8_flash_next_m4pro.server import protocol
from jitengine_qwen3_8_flash_next_m4pro.server.protocol import (
    MAX_PROMPT_TOKENS,
    InvalidRequestError,
    parse_request,
)
from jitengine_qwen3_8_flash_next_m4pro.tokenizer import MAX_RENDERED_CHARS


class RecordingTokenizer:
    def __init__(self):
        self.encoded: list[int] = []  # the length of every text encoded

    def encode(self, text: str) -> list[int]:
        self.encoded.append(len(text))
        return [5] * 10


class CountedIds(list):
    """An id list that counts the elements read from it."""

    read = 0

    def __iter__(self):
        for x in super().__iter__():
            self.read += 1
            yield x


def _engine():
    return SimpleNamespace(
        tokenizer=RecordingTokenizer(), vocab=248320, max_kv_positions=40000, model_name="m"
    )


def test_the_completions_cap_is_the_chat_cap():
    assert MAX_RENDERED_CHARS == 4 * 2**20
    assert getattr(protocol, "MAX_PROMPT_CHARS", None) == MAX_RENDERED_CHARS


def test_an_oversized_text_prompt_is_refused_before_encoding():
    e = _engine()
    text = ("lorem ipsum dolor sit amet " * 1_200_000)[:32_000_000]
    with pytest.raises(InvalidRequestError) as exc:
        parse_request(e, {"prompt": text}, chat=False)
    assert exc.value.param == "prompt" and "32000000 characters" in str(exc.value)
    assert e.tokenizer.encoded == []
    with pytest.raises(InvalidRequestError):  # a batch of one prompt is that prompt
        parse_request(e, {"prompt": [text]}, chat=False)
    assert e.tokenizer.encoded == []


def test_a_prompt_at_the_cap_is_still_encoded():
    e = _engine()
    req, _ = parse_request(e, {"prompt": "a" * MAX_RENDERED_CHARS, "max_tokens": 1}, chat=False)
    assert e.tokenizer.encoded == [MAX_RENDERED_CHARS] and req.prompt_ids == [5] * 10
    with pytest.raises(InvalidRequestError):
        parse_request(e, {"prompt": "a" * (MAX_RENDERED_CHARS + 1)}, chat=False)
    assert e.tokenizer.encoded == [MAX_RENDERED_CHARS]


def test_an_oversized_id_prompt_is_refused_before_it_is_read():
    ids = CountedIds([100000] * 4_000_000)
    with pytest.raises(InvalidRequestError) as exc:
        parse_request(_engine(), {"prompt": ids}, chat=False)
    assert exc.value.code == "context_length_exceeded" and exc.value.param == "prompt"
    assert "4000000 tokens" in str(exc.value)
    assert ids.read <= 1  # the batch-of-prompts check looks at the first element only
    # at the bound it is read and served as before
    ok = [7] * MAX_PROMPT_TOKENS
    req, _ = parse_request(_engine(), {"prompt": ok, "max_tokens": 1}, chat=False)
    assert req.prompt_ids == ok
    with pytest.raises(InvalidRequestError) as exc:
        parse_request(_engine(), {"prompt": ok + [7]}, chat=False)
    assert exc.value.code == "context_length_exceeded"


CHECKPOINT = os.environ.get("JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_E2E_CHECKPOINT")


@pytest.mark.skipif(
    not CHECKPOINT, reason="needs JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_E2E_CHECKPOINT"
)
def test_the_cap_refuses_no_prompt_that_fits_the_token_bound():
    """Every token covers at most its surface's UTF-8 bytes of text, and a character is at least a
    byte, so text longer than MAX_PROMPT_TOKENS x the longest surface cannot fit the bound."""
    tokenizers = pytest.importorskip("tokenizers")
    path = os.path.join(os.path.expanduser(CHECKPOINT), "tokenizer.json")
    tk = tokenizers.Tokenizer.from_file(path)
    longest = max(
        len(tk.decode([i], skip_special_tokens=False).encode()) for i in tk.get_vocab().values()
    )
    assert json.load(open(path)).get("normalizer", {}).get("type") in (None, "NFC")
    assert protocol.MAX_PROMPT_CHARS >= MAX_PROMPT_TOKENS * longest, longest
