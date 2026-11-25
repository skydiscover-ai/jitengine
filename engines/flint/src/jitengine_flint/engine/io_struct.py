"""Request/result types. The engine also accepts any object carrying these attribute names (duck typing), so a caller may
build requests from its own types."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20
TOP_LOGPROBS_MAX = 20
FORCED_TOP = 20


@dataclass
class GenRequest:
    prompt_ids: Sequence[int]
    max_tokens: int | None = None
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    seed: int | None = None
    ignore_eos: bool = False
    stop: Sequence[str] = ()
    logprobs: int | None = None
    forced_tokens: Sequence[int] | None = None
    one_token_per_pass: bool | None = None
    fault: str | None = None
    request_id: str | None = None


@dataclass
class StepOutput:
    index: int
    token_id: int | None
    logprob: float | None = None
    top: list = field(default_factory=list)
    t_release: float = 0.0


@dataclass
class GenResult:
    token_ids: list[int]
    finish_reason: str
    prompt_tokens: int
    cached_prompt_tokens: int = 0
    steps: list[StepOutput] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    error: str | None = None
    text: str | None = None  # decoded completion, cut at a matched stop string


@dataclass
class ScoredPosition:
    position: int
    token_id: int
    logprob: float
    top: list


class QueueFull(RuntimeError):
    """More requests are waiting than the engine admits (HTTP 429)."""


class ContextTooLarge(ValueError):
    """The prompt (or prompt + forced stream) does not fit the context limit (HTTP 400)."""


class EngineUnavailable(RuntimeError):
    """The engine is loading, stopping or unhealthy (HTTP 503)."""


class Cancelled(RuntimeError):
    """The request was cancelled before it produced output (its client went away); nothing is answered."""
