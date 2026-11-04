"""The QSA indexer (`Qwen4ExpTextQSAIndexer`), which picks the cached positions each attention query
may attend to, and the rotary tables that rope its queries and block keys.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m2.engine.errors import InvariantError
from jitengine_qwen3_8_flash_next_m2.kernels import indexer_prefill as _ixp
from jitengine_qwen3_8_flash_next_m2.kernels import qsa_decode_mask as _af
from jitengine_qwen3_8_flash_next_m2.kernels import qsa_indexer_decode as _ixf
from jitengine_qwen3_8_flash_next_m2.linear import qlinear_from
from jitengine_qwen3_8_flash_next_m2.model.config import Qwen4ExpTextConfig


# ----------------------------------------------------------------------------- rotary tables
def rope_tables(positions: mx.array, dims: int, base: float) -> tuple[mx.array, mx.array]:
    """cos/sin [P, dims] fp32 tables of the partial rope at integer `positions`.

    Used by the QSA indexer for block keys roped at positions `cr * j`. Attention q/k use
    `mx.fast.rope`, which is not bit-exact against these tables.
    """
    inv = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
    f = positions.astype(mx.float32)[:, None] * inv[None, :]
    return mx.concatenate([mx.cos(f), mx.cos(f)], axis=-1), mx.concatenate(
        [mx.sin(f), mx.sin(f)], axis=-1
    )


# The decode indexer's rope tables depend only on the position, so the attention layers of a step
# share one cached copy.
_ROPE_POS: dict = {}


def rope_tables_at(s0: int, dims: int, base: float) -> tuple[mx.array, mx.array]:
    """`rope_tables(mx.arange(s0, s0 + 1), dims, base)`, cached by position."""
    key = (int(s0), int(dims), float(base))
    hit = _ROPE_POS.get(key)
    if hit is None:
        if len(_ROPE_POS) > 8:
            _ROPE_POS.clear()
        hit = _ROPE_POS[key] = rope_tables(mx.arange(s0, s0 + 1), dims, base)
    return hit


def apply_rope_tables(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """x [..., P, D]; cos/sin [P, R], R <= D: rotate the first R dims (rotate_half), pass the rest."""
    r = cos.shape[-1]
    h = r // 2
    xr, xn = x[..., :r], x[..., r:]
    rot = mx.concatenate([-xr[..., h:], xr[..., :h]], axis=-1)
    xr = xr * cos + rot * sin
    return mx.concatenate([xr, xn], axis=-1) if xn.shape[-1] else xr


class Qwen4ExpTextQSAIndexer:
    """`Qwen4ExpTextQSAIndexer`: which cached positions each query may attend to.

    `index_qk_proj` gives per position `nh` query heads and one raw (pre-norm, pre-rope) key, which
    is appended to the row's `KVSlot.idx` cache with the KV. Selection, all in fp32:
      * blocks are absolute: block j covers positions [cr*j, cr*j + cr); its key is the mean of its
        raw keys, `(1 + w)` normed and roped at its first position, so chunked == single pass;
      * scores = relu(q . block key) summed over heads / sqrt(hd), q `(1 + w)` normed and roped;
      * a query at p sees nb_p = (p + 1) // cr complete blocks; if nb_p <= budget // cr it sees
        every causal token, else the top budget // cr blocks by a stable descending sort (equal
        scores keep ascending block order, as in the reference) plus the ragged tail.
    No query prunes while the chunk's last position is below `prune_from`, and attention then uses
    the plain causal path without computing scores.
    """

    def __init__(
        self,
        cfg: Qwen4ExpTextConfig,
        w: dict,
        p: str,
        compute_dtype=mx.float32,
        kernels_off: frozenset = frozenset(),
    ):
        self.dtype = compute_dtype
        self.indexer_fused_ok = "indexer_fused" not in kernels_off
        self.attn_fused_ok = "attn_fused" not in kernels_off
        self.indexer_prefill_ok = "indexer_prefill" not in kernels_off
        self.qk_proj = qlinear_from(w, p + "index_qk_proj", cfg)
        self.q_norm_w1 = 1.0 + w[p + "q_layernorm.weight"].astype(mx.float32)
        self.k_norm_w1 = 1.0 + w[p + "k_layernorm.weight"].astype(mx.float32)
        mx.eval(self.q_norm_w1, self.k_norm_w1)  # derived weights are evaluated here
        self.nh = int(cfg.indexer_n_heads)
        self.hd = int(cfg.indexer_head_dim)
        self.budget = int(cfg.indexer_budget)
        self.cr = int(cfg.indexer_compress_ratio)
        self.block_topk = self.budget // self.cr
        self.prune_from = self.cr * (self.block_topk + 1) - 1
        self.rope_dims = int(cfg.rope_dims)
        self.rope_base = float(cfg.rope_theta)
        self.eps = float(cfg.rms_norm_eps)
        self.scale = self.hd**-0.5

    def linears(self) -> list:
        return [self.qk_proj]

    def project(self, x: mx.array, rows: bool = False) -> tuple[mx.array, mx.array]:
        """x [B, L, hidden] -> (q [B, L, nh, hd], raw_k [B, 1, L, hd]) in the compute dtype."""
        B, L, _ = x.shape
        qk = self.qk_proj.rows(x) if rows else self.qk_proj(x)
        nq = self.nh * self.hd
        q = qk[..., :nq].reshape(B, L, self.nh, self.hd)
        raw_k = qk[..., nq:].reshape(B, 1, L, self.hd)
        return q, raw_k

    def _norm(self, x: mx.array, w1: mx.array) -> mx.array:
        """`(1 + w)` rms norm in fp32, written out as the reference does it."""
        x = x.astype(mx.float32)
        return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps) * w1

    def _block_keys(self, idx_keys: mx.array, nb: int, cache: dict | None) -> mx.array:
        """Pooled, normed, roped block keys [nb, hd], from the slot's cache when unchanged."""
        bk = None
        j0 = 0  # first block row this call must build
        if cache is not None:
            hit = cache.get(id(self))
            if hit is not None and hit[0] == nb:
                bk = hit[1]  # nothing completed since the last call: the same array
            elif hit is not None and 0 < hit[0] < nb:
                j0, bk = hit[0], hit[1]  # only the blocks that completed since are new
        if bk is None or j0:
            seg = idx_keys[0, 0, j0 * self.cr : nb * self.cr, :]
            pooled = seg.astype(mx.float32).reshape(nb - j0, self.cr, self.hd).mean(axis=1)
            pooled = self._norm(pooled, self.k_norm_w1)
            cosb, sinb = rope_tables(mx.arange(j0, nb) * self.cr, self.rope_dims, self.rope_base)
            new = apply_rope_tables(pooled, cosb, sinb)  # [nb - j0, hd]
            bk = new if not j0 else mx.concatenate([bk, new], axis=0)  # [nb, hd]
            if cache is not None:
                cache[id(self)] = (nb, bk)
        return bk

    def scores(
        self, q: mx.array, idx_keys: mx.array, s0: int, cache: dict | None = None
    ) -> tuple[mx.array, int]:
        """(q [1, L, nh, hd], idx_keys [1, 1, P, hd]) -> (scores [L, nb] fp32, nb); P = s0 + L.

        Inputs are raw (pre-norm, pre-rope). Block keys depend only on idx_keys[:nb * cr], which is
        never rewritten, so they are cached in `cache` (the slot's dict) until a block completes.
        """
        L = q.shape[1]
        P = idx_keys.shape[2]
        nb = P // self.cr
        qf = self._norm(q, self.q_norm_w1).transpose(0, 2, 1, 3)  # [1, nh, L, hd]
        cos, sin = rope_tables(mx.arange(s0, s0 + L), self.rope_dims, self.rope_base)
        qf = apply_rope_tables(qf, cos, sin)[0]  # [nh, L, hd]
        bk = self._block_keys(idx_keys, nb, cache)
        sc = mx.maximum(qf @ bk.T, 0.0).sum(axis=0) * self.scale  # [L, nb]
        return sc, nb

    def select(
        self, q: mx.array, idx_keys: mx.array, s0: int, dispatch=(), cache: dict | None = None
    ) -> np.ndarray | None:
        """Numpy bool mask [L, P] of the tokens each query may see, or None when none prunes.

        `dispatch` (lazy arrays independent of the mask) is submitted after the score sync, since a
        stream runs in submission order and earlier submission would make the sync wait for it, and
        before the numpy mask build, so the GPU works while the host does.
        """
        L = q.shape[1]
        P = idx_keys.shape[2]
        if P != s0 + L:
            raise InvariantError(f"idx cache holds {P} positions, chunk ends at {s0 + L}")
        if P - 1 < self.prune_from:
            return None
        sc, nb = self.scores(q, idx_keys, s0, cache=cache)
        scn = np.array(sc)  # the host sync
        if dispatch:
            mx.async_eval(*dispatch)
        return self.mask_from_scores(scn, nb, s0, L, P, shared=cache)

    def _mask_base(self, s0: int, L: int, P: int, nb: int, shared: dict | None):
        """(pos, col, causal, nb_i, prune), cached per (s0, L, P) in the slot's `shared` dict."""
        key = (s0, L, P)
        if shared is not None:
            hit = shared.get("mask_base")
            if hit is not None and hit[0] == key:
                return hit[1]
        pos = np.arange(s0, s0 + L)
        nb_i = (pos + 1) // self.cr
        col = np.arange(P)[None, :]
        causal = col <= pos[:, None]
        prune = nb_i > self.block_topk
        base = (pos, col, causal, nb_i, prune)
        if shared is not None:
            shared["mask_base"] = (key, base)
        return base

    def device_prefill_ok(self, P: int) -> bool:
        return self.indexer_prefill_ok and _ixp.supports(
            self.nh, self.hd, self.rope_dims, P // self.cr
        )

    def select_prefill_device(
        self, q: mx.array, idx_keys: mx.array, s0: int, cache: dict | None = None
    ):
        """Device `select` for L > 1: a [1, 1, L, P] bool mask, or None when no query prunes."""
        L = q.shape[1]
        P = idx_keys.shape[2]
        if P != s0 + L:
            raise InvariantError(f"idx cache holds {P} positions, chunk ends at {s0 + L}")
        if P - 1 < self.prune_from:
            return None
        nb = P // self.cr
        bk = self._block_keys(idx_keys, nb, cache)
        cos, sin = rope_tables(mx.arange(s0, s0 + L), self.rope_dims, self.rope_base)
        return _ixp.prefill_mask(
            q.reshape(L, self.nh, self.hd),
            self.q_norm_w1,
            bk,
            cos,
            sin,
            s0,
            L,
            P,
            self.cr,
            self.block_topk,
            self.eps,
            self.rope_base,
            self.rope_dims,
            self.hd,
        )

    def select_device(
        self, q: mx.array, idx_keys: mx.array, s0: int, cache: dict | None = None
    ) -> mx.array | None:
        """Decode-only (L == 1) mask built on the GPU: [1, 1, 1, P] bool, or None if not pruning.

        MLX's sort is not stable, so each score is mapped to an order-preserving int and packed with
        its block index, `(monotonic(neg) << 21) | j`: all keys are distinct and any correct sort
        reproduces the stable tie order. At L == 1 every complete block is visible, so the mask is
        `repeat(keep, cr)` plus an all-true tail for the partial block.
        """
        P = idx_keys.shape[2]
        if P - 1 < self.prune_from:
            return None
        topk, cr = self.block_topk, self.cr
        nb0 = P // cr
        if self.indexer_fused_ok and _ixf.supports(self.nh, self.hd, self.rope_dims, nb0):
            bk = self._block_keys(idx_keys, nb0, cache)
            cos, sin = rope_tables_at(s0, self.rope_dims, self.rope_base)  # the stock tables
            mask, _ = _ixf.indexer_mask(
                q.reshape(self.nh, self.hd),
                self.q_norm_w1,
                bk,
                cos,
                sin,
                s0,
                nb0,
                cr,
                topk,
                P,
                self.eps,
                self.rope_base,
                self.rope_dims,
                self.hd,
            )
            return mask
        sc, nb = self.scores(q, idx_keys, s0, cache=cache)  # [1, nb] fp32, stays on device
        if self.attn_fused_ok and _af.supports(nb):
            return _af.qsa_mask(sc[0], nb, cr, topk, P)
        neg = -sc[0]  # [nb]; every block is valid here
        u = neg.view(mx.int32)
        k = mx.where(u < 0, mx.array(np.int32(-2147483648), dtype=mx.int32) - u, u).astype(mx.int64)
        key = (k << 21) | mx.arange(nb, dtype=mx.int64)
        rank = mx.argsort(mx.argsort(key))  # rank of each block
        keep = rank < topk  # [nb] bool
        full = mx.repeat(keep, cr)  # [nb * cr]
        tail = P - nb * cr
        if tail:
            full = mx.concatenate([full, mx.ones((tail,), dtype=mx.bool_)])
        return full[None, None, None, :]  # [1, 1, 1, P]

    def mask_from_scores(
        self, scn: np.ndarray, nb: int, s0: int, L: int, P: int, shared: dict | None = None
    ) -> np.ndarray:
        cr, topk = self.cr, self.block_topk
        _pos, _col, causal, nb_i, prune = self._mask_base(s0, L, P, nb, shared)
        if not prune.any():
            return causal.copy()
        # the mask is `causal` minus the unselected blocks: a selected block j < nb_i ends at
        # (j + 1) * cr <= pos + 1, so it never extends past `causal`
        mask = causal.copy()
        rows = np.nonzero(prune)[0]
        valid = np.arange(nb)[None, :] < nb_i[rows][:, None]
        # ascending sort of -score; blocks past the query's horizon sort last (+inf); -0.0 == 0.0
        # so equal (zero) scores keep ascending block order under the stable sort
        neg = np.where(valid, -scn[rows], np.inf).astype(np.float32)
        order = np.argsort(neg, axis=1, kind="stable")  # [R, nb]
        for r, row in enumerate(rows):
            drop = order[r, topk : nb_i[row]]  # unselected blocks
            if drop.size:
                tok = (drop[:, None] * cr + np.arange(cr)[None, :]).reshape(-1)
                mask[row, tok] = False
        return mask
