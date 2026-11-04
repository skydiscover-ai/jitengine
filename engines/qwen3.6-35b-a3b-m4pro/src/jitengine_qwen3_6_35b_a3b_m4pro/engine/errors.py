"""What the engine refuses or stops a request with; the HTTP server maps each to its status and
OpenAI error object by class name (`server.openai_api.error_response`)."""

from __future__ import annotations


class QueueFullError(RuntimeError):
    """More requests are outstanding than the configured queue admits. Retryable."""

    retryable = True


class ContextTooLargeError(ValueError):
    """prompt + max_tokens exceeds the configured context bound."""


class MemoryPressureError(RuntimeError):
    """A running request was stopped because the process would exceed its memory ceiling."""

    retryable = True


class AdmissionPausedError(RuntimeError):
    """New work is temporarily refused (memory pressure or a resize in progress). Retryable."""

    retryable = True


class InvariantError(RuntimeError):
    """An engine invariant does not hold: a bug in the engine, not a bad request. The step that hit
    it fails its rows (HTTP 500), the engine keeps serving, and `/stats` counts it under
    `error_types`."""
