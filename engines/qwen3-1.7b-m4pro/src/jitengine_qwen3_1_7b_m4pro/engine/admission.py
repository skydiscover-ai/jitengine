"""Admission control: a request longer than the context is refused, and requests wait in arrival
order while the padded KV cache of the running requests would pass the budget.

mlx-lm keeps a running batch as one left-padded tensor, so its KV memory is the number of running
requests times the longest of them (112 KiB per token for Qwen3-1.7B), not the sum of their lengths.
The budget therefore charges `running requests x the longest request`, each request counted as
prompt plus the output it may produce. For requests of one length that equals their sum.

Stock mlx-lm admits everything at once. Many long prompts together exhaust Metal memory, which
kills the generation thread for good. The limits are the ones measured on a 24 GB Mac."""

from __future__ import annotations

import collections
import threading

DEFAULT_MAX_CONTEXT = 32768  # tokens one request may hold, prompt and output together
DEFAULT_MAX_INFLIGHT = 32768  # padded tokens of the running requests: their number x the longest
# Stock mlx-lm keeps up to 10 finished requests' KV caches with no limit on their size. An agent
# session of 15k to 30k-token prompts filled 9.6 GB with them and ran the Mac out of memory.
DEFAULT_PROMPT_CACHE_GIB = 6.0
# MLX keeps freed buffers for reuse with a limit as large as the Mac's memory; in an agent session
# that cache reached 10 GiB on top of the live arrays and the Mac ran out of Metal memory.
DEFAULT_MLX_CACHE_GIB = 2.0
MODEL_MAX_POSITIONS = 40960  # Qwen3-1.7B's trained context
POLL_SECONDS = 0.5


class TooLong(Exception):
    def __init__(self, need: int, limit: int):
        super().__init__(
            f"prompt is too long: {need} tokens (prompt plus max_tokens) exceed the context of {limit}"
        )
        self.need = need
        self.limit = limit


class TokenBudget:
    def __init__(
        self, max_context: int = DEFAULT_MAX_CONTEXT, max_inflight: int = DEFAULT_MAX_INFLIGHT
    ):
        if not 0 < max_context <= max_inflight:
            raise ValueError("max_context must be positive and at most max_inflight")
        if max_context > MODEL_MAX_POSITIONS:
            raise ValueError(f"max_context is over the model's {MODEL_MAX_POSITIONS} positions")
        self.max_context = max_context
        self.max_inflight = max_inflight
        self._cond = threading.Condition()
        self._running: list[int] = []
        self._queue: collections.deque = collections.deque()
        self._served = 0
        self._peak = 0

    @staticmethod
    def _padded(needs) -> int:
        return len(needs) * max(needs, default=0)

    def acquire(self, need: int, gone=lambda: False) -> bool:
        """Wait until the padded cache with `need` added fits, first come first served. False if `gone()` says the client left."""
        if need > self.max_context:
            raise TooLong(need, self.max_context)
        ticket = object()
        with self._cond:
            self._queue.append(ticket)
            try:
                while not (
                    self._queue[0] is ticket
                    and self._padded([*self._running, need]) <= self.max_inflight
                ):
                    self._cond.wait(POLL_SECONDS)
                    if gone():
                        return False
                self._running.append(need)
                self._peak = max(self._peak, self._padded(self._running))
                self._served += 1
                return True
            finally:
                self._queue.remove(ticket)
                self._cond.notify_all()

    def release(self, need: int) -> None:
        with self._cond:
            self._running.remove(need)
            self._cond.notify_all()

    def snapshot(self) -> dict:
        with self._cond:
            return {
                "max_context": self.max_context,
                "max_inflight_tokens": self.max_inflight,
                "running": len(self._running),
                "inflight_tokens": sum(self._running),
                "padded_tokens": self._padded(self._running),
                "waiting": len(self._queue),
                "peak_padded_tokens": self._peak,
                "admitted": self._served,
            }
