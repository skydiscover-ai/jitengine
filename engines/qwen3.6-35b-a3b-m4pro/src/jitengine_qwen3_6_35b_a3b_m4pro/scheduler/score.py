"""Scoring for `Engine`, as a mixin: teacher-forced logprobs, run exclusively and cold (never
through the prefix store or a decode batch).
"""

from __future__ import annotations

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m4pro.engine.errors import ContextTooLargeError
from jitengine_qwen3_6_35b_a3b_m4pro.kvcache.state import RequestState
from jitengine_qwen3_6_35b_a3b_m4pro.sampler import logprobs_from_logits
from jitengine_qwen3_6_35b_a3b_m4pro.scheduler.prefill import PrefillTask


class ScoringMixin:
    """Score requests, run one at a time between rounds."""

    def _run_scores(self) -> None:
        while self._scores:
            job = self._scores.pop(0)
            try:
                job.out.put(("done", self._run_score(*job.payload)))
            except Exception as exc:  # noqa: BLE001
                self.stats_.count_error(type(exc).__name__)
                job.out.put(("error", exc))

    def _run_score(self, prompt_ids, continuation_ids, top_k: int) -> list[dict]:
        prompt = [int(t) for t in prompt_ids]
        cont = [int(t) for t in continuation_ids]
        if not prompt:
            raise ValueError("empty prompt")
        # MLX's embedding gather does not bounds-check, so an out-of-range id would be an
        # out-of-bounds GPU read; same bound as `_accept`, before any forward pass
        for t in prompt + cont:
            if t < 0 or t >= self.token_id_limit:
                raise ValueError(f"token id {t} out of range")
        if not cont:
            return []
        if len(prompt) + len(cont) > self.max_kv_positions:
            raise ContextTooLargeError(
                f"score of {len(prompt) + len(cont)} positions exceeds the context "
                f"limit of {self.max_kv_positions} tokens"
            )
        state = RequestState(self.cfg, len(prompt), len(cont), reserve=len(cont))
        model = self.model
        k = max(1, min(int(top_k), self.vocab))
        out: list[dict] = []
        try:
            # score never touches the store and never joins a decode batch: cold-exact, single row.
            PrefillTask(
                model, self.store, self.prefill_chunk, prompt[:-1], state, use_store=False
            ).run(layer_major=self.layer_major_prefill)
            block = [prompt[-1]] + cont[:-1]
            for s in range(0, len(block), self.prefill_chunk):
                chunk = block[s : s + self.prefill_chunk]
                logits = model.forward(mx.array(chunk, dtype=mx.int32)[None], state, logits="all")[
                    0
                ]
                lp = logprobs_from_logits(logits)
                idx = mx.argpartition(-lp, kth=k - 1, axis=-1)[..., :k]
                vals = mx.take_along_axis(lp, idx, axis=-1)
                tgt = mx.array(cont[s : s + len(chunk)])
                tgt_lp = mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]
                mx.eval(idx, vals, tgt_lp)
                idx_l, vals_l, tgt_l = idx.tolist(), vals.tolist(), tgt_lp.tolist()
                for i in range(len(chunk)):
                    pairs = sorted(zip(idx_l[i], vals_l[i]), key=lambda p: -p[1])
                    out.append(
                        {"logprob": float(tgt_l[i]), "top": [[int(t), float(v)] for t, v in pairs]}
                    )
                mx.clear_cache()
        finally:
            del state
        return out
