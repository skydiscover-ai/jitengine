"""Request and result dataclasses."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class GenRequest:
    """One generation request: the prompt as token ids, `max_tokens`, and how to sample.

    `temperature` 0 is greedy; `top_k` 0 and `top_p` 1.0 filter nothing; a sampled request with
    no `seed` draws one. `stop_at_eos` ends the row at an EOS id; `request_id` labels its record
    and response.
    """

    prompt_ids: Sequence[int]
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    stop_at_eos: bool = True
    request_id: str | None = None


@dataclass
class GenResult:
    """What a request produced: the new `token_ids`, why it stopped, and the prompt's size.

    `finish_reason` is "stop" (an EOS id), "length" (`max_tokens`, or cancelled) or "error";
    `cached_prompt_tokens` came from the prefix store; `extra` is the request's record, or for an
    error its `error` text.
    """

    token_ids: list[int]
    finish_reason: str
    prompt_tokens: int
    cached_prompt_tokens: int = 0
    extra: dict = field(default_factory=dict)
