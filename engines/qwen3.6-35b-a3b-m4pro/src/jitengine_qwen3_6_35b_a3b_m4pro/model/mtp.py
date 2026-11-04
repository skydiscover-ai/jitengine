"""The checkpoint's multi-token-prediction head, used as a speculative drafter.

It only proposes tokens and the target's verify pass decides every emitted one, so inexactness here
(stock mlx-lm modules, a vocabulary-prefix lm_head) can lower acceptance but never change output.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class MTPHead:
    """The MTP head loaded from `path`, sharing the target's embedding and a vocabulary-prefix
    `lm_head`; it drafts from the target's last hidden state and holds its own KV cache."""

    def __init__(self, path: str, model, model_path: str, draft_vocab: int = 65536):
        from mlx_lm.models.base import create_attention_mask
        from mlx_lm.models.cache import KVCache
        from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs

        self._mask = create_attention_mask
        self._KVCache = KVCache
        import json
        import os

        conf = json.load(open(os.path.join(model_path, "config.json")))
        args = TextModelArgs.from_dict(conf.get("text_config", conf))
        D = args.hidden_size

        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.pre_fc_norm_embedding = nn.RMSNorm(D, eps=args.rms_norm_eps)
                self.pre_fc_norm_hidden = nn.RMSNorm(D, eps=args.rms_norm_eps)
                self.fc = nn.Linear(2 * D, D, bias=False)
                self.layers = [DecoderLayer(args, layer_idx=args.full_attention_interval - 1)]
                self.norm = nn.RMSNorm(D, eps=args.rms_norm_eps)

        m = _M()
        w = mx.load(path)
        meta = mx.load(path, return_metadata=True)[1]
        bits = int(meta.get("bits", 4))

        def pred(p, _):
            if f"{p}.scales" not in w:
                return False
            # bits per layer from the stored shapes (in = scales_cols * 64; bits = packed_cols * 32 / in),
            # since a head may mix bit widths (for example 8-bit routers)
            in_features = w[f"{p}.scales"].shape[-1] * 64
            return {"group_size": 64, "bits": w[f"{p}.weight"].shape[-1] * 32 // in_features}

        nn.quantize(m, group_size=64, bits=bits, class_predicate=pred)
        m.load_weights(list(w.items()), strict=True)
        mx.eval(m.parameters())
        self.m = m
        self.model = model
        head = model.lm_head
        V = min(draft_vocab or head.out_dims, head.out_dims)
        self.V = V
        self._hw = (
            head.weight[:V],
            head.scales[:V],
            head.biases[:V] if head.biases is not None else None,
            head.group_size,
            head.bits,
        )
        mx.eval(*[t for t in self._hw[:3] if t is not None])

    def nbytes(self) -> int:
        """The bytes the head's own parameters take."""
        from mlx.utils import tree_flatten

        return sum(v.nbytes for _, v in tree_flatten(self.m.parameters()))

    def new_cache(self):
        """An empty KV cache for one request's draft stream."""
        return self._KVCache()

    def _logits(self, u):
        w, s, b, gs, bits = self._hw
        return mx.quantized_matmul(
            u, w, scales=s, biases=b, transpose=True, group_size=gs, bits=bits
        )

    def _run(self, cache, h, toks):
        """h [L, D] target hiddens (post-norm), toks [L] next tokens -> (u [1, L, D], normed)."""
        emb = self.model.embed.lookup(toks.reshape(1, -1))
        x = mx.concatenate(
            [self.m.pre_fc_norm_embedding(emb), self.m.pre_fc_norm_hidden(h[None])], axis=-1
        )
        x = self.m.fc(x)
        x = self.m.layers[0](x, mask=self._mask(x, cache), cache=cache)
        return x, self.m.norm(x)

    def extend(self, cache, h: mx.array, next_toks: mx.array, chunk: int = 512) -> None:
        """Append the pairs (h_i, x_{i+1}) (h [L, D], next_toks [L]) to `cache` (prompt chunks)."""
        L = h.shape[0]
        for s0 in range(0, L, chunk):
            self._run(cache, h[s0 : s0 + chunk], next_toks[s0 : s0 + chunk])

    @staticmethod
    def _coupled(
        lg: mx.array, temperature: float, top_p: float, top_k: int, u01: mx.array
    ) -> mx.array:
        """Draw the draft for a sampled step the way the target will, over the head's logits.

        Uses mlx_lm's top_p / top_k cut and an inverse-CDF draw in vocabulary order, with the same
        uniform the target uses (`Sampler.uniform(step)`). Only the proposal changes: the verify
        still draws every emitted token itself, so outputs are unchanged.
        """
        lp = (lg - mx.logsumexp(lg, axis=-1, keepdims=True)).astype(mx.float32).reshape(-1)
        kk = top_k if 0 < top_k <= 256 else 64
        cand = mx.argpartition(-lp, kth=kk - 1)[:kk]
        cv = lp[cand]
        if 0.0 < top_p < 1.0:
            o = mx.argsort(-cv)  # descending
            above = mx.cumsum(mx.exp(cv[o]), inclusive=False)  # mass ranked strictly above
            keep_sorted = above < top_p
            keep = mx.put_along_axis(mx.zeros((kk,), dtype=mx.bool_), o, keep_sorted, axis=0)
        else:
            keep = mx.ones((kk,), dtype=mx.bool_)
        x = mx.where(keep, cv * (1.0 / temperature), -float("inf"))
        w = mx.exp(x - mx.max(x))
        ov = mx.argsort(cand)  # vocabulary order
        ws, cs = w[ov], cand[ov]
        cdf = mx.cumsum(ws, inclusive=False)
        i = mx.maximum(mx.sum(cdf <= u01 * mx.sum(ws)) - 1, 0)
        return cs[i].reshape(1).astype(mx.int32)

    def _pick(self, lg: mx.array, coupled, j: int) -> mx.array:
        if coupled is None:
            return mx.argmax(lg, axis=-1).astype(mx.int32)
        temperature, top_p, top_k, us = coupled
        return self._coupled(lg, temperature, top_p, top_k, us[j])

    def draft_sampled(
        self,
        cache,
        h: mx.array,
        next_toks: mx.array,
        k: int,
        temperature: float,
        top_p: float,
        top_k: int,
        keys: list,
    ):
        """Draft k tokens, each sampled from the head's filtered distribution q (for probabilistic acceptance).

        Returns (drafts [k] int32, qids [k, kk] int32, qprob [k, kk] float32), q being the request's
        top_p / top_k / temperature applied over the draft vocabulary.
        """
        from jitengine_qwen3_6_35b_a3b_m4pro.speculative.spec_sample import (
            draw_sparse,
            filtered_sparse,
        )

        u, un = self._run(cache, h, next_toks)
        drafts, qi, qp = [], [], []
        uc = u[:, -1:, :]
        d = None
        for j in range(k):
            if j > 0:
                uc, un = self._run(cache, uc[0], d)
            lg = self._logits(un[:, -1, :])
            ids, pr = filtered_sparse(lg, temperature, top_p, top_k, logits=True)
            d = draw_sparse(ids, pr, keys[j]).astype(mx.int32)
            drafts.append(d)
            qi.append(ids)
            qp.append(pr)
        if k > 1:
            cache.trim(k - 1)
        return mx.concatenate(drafts), mx.concatenate(qi), mx.concatenate(qp)

    def advance_and_draft(self, cache, h: mx.array, next_toks: mx.array, k: int, coupled=None):
        """Append the committed pairs (h_j, x_{j+1}) (h [L, D], toks [L]), then draft k tokens (lazy [k] int32).

        The k-1 chained speculative entries are trimmed from the cache before returning, so it holds
        exactly the committed pairs. `coupled` = (temperature, top_p, top_k, [uniform of each drafted
        step]) draws sampled drafts (see `_coupled`).
        """
        u, un = self._run(cache, h, next_toks)
        d = self._pick(self._logits(un[:, -1, :]), coupled, 0)  # [1]
        drafts = [d]
        uc = u[:, -1:, :]
        for j in range(1, k):
            uc, un = self._run(cache, uc[0], d)
            d = self._pick(self._logits(un[:, -1, :]), coupled, j)
            drafts.append(d)
        if k > 1:
            cache.trim(k - 1)
        return mx.concatenate(drafts)
