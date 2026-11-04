"""Fixtures the Flash-Next kernel tests share: random 4-bit weights and exact float64 references.

Where a kernel is not claimed bit-identical to stock MLX it still does stock's float32 arithmetic in
a different summation order. `accuracy.no_worse_than_stock` checks that against an exact float64
result computed with numpy: the kernel's largest error may be at most `factor` times stock's largest
error on the same inputs, plus a few float32 ulps of the result for when stock happens to be exact.
Errors of two summation orders of the same products are of the same size, and a wrong index, group
or formula is orders of magnitude larger. Stock itself must be within 1e-3 of the reference's scale,
or 2^-6 of it for a bfloat16 result, which checks the reference. A bfloat16 ulp is at most 2^-7 of
the value, but stock's bfloat16 quantized matmul rounds partial sums to bfloat16: on the M2 it came
to 1.02 x 2^-7 of the scale (a 320x10240 matvec), where the kernels round once.

`q4`'s weights, and every array a compiled stock path captures, are evaluated before use, as the
model's weights are. `mx.compile` recomputes a lazy array it captures inside the compiled graph,
where fused arithmetic can round it differently, so a compiled stock path over lazy weights runs on
other values than the reference reads (on the M2, 1e-3 of the scale for the hyper-connection's
mixed output).
"""

from __future__ import annotations

import numpy as np
import pytest

U32 = 2.0**-24  # float32 unit roundoff


class Accuracy:
    @staticmethod
    def np64(a) -> np.ndarray:
        """An MLX or numpy array as float64 numpy (bfloat16 goes through float32, exactly)."""
        import mlx.core as mx

        if isinstance(a, mx.array):
            a = a if a.dtype in (mx.float32, mx.float64) else a.astype(mx.float32)
        return np.array(a, dtype=np.float64)

    def weights(self, w, scales, biases, group_size: int = 32, bits: int = 4) -> np.ndarray:
        """The exact float64 values of an affine-quantized matrix: scale * q + bias per group."""
        import mlx.core as mx

        q = mx.dequantize(
            w,
            mx.ones_like(scales),
            mx.zeros_like(biases),
            group_size=group_size,
            bits=bits,
        )
        q = self.np64(q)
        groups = q.reshape(*q.shape[:-1], -1, group_size)
        w = groups * self.np64(scales)[..., None] + self.np64(biases)[..., None]
        return w.reshape(q.shape)

    def no_worse_than_stock(
        self, got, stock, exact, factor: float = 8.0, what: str = "", slack: float = 0.0
    ) -> None:
        """`slack` adds a documented error bound the kernel has and stock does not."""
        import mlx.core as mx

        bf16 = getattr(stock, "dtype", None) == mx.bfloat16
        got, stock, exact = self.np64(got), self.np64(stock), np.asarray(exact, np.float64)
        assert got.shape == stock.shape == exact.shape, (what, got.shape, stock.shape, exact.shape)
        e_got = float(np.abs(got - exact).max())
        e_stock = float(np.abs(stock - exact).max())
        scale = float(np.abs(exact).max())
        floor = 4 * U32 * scale
        # a check on the test's own reference, which a wrong formula misses by the result's size
        ref = (2.0**-6 if bf16 else 1e-3) * scale
        assert e_stock <= ref, f"{what}: stock is {e_stock:.3g} from the exact reference"
        assert e_got <= factor * e_stock + floor + slack, (
            f"{what}: kernel error {e_got:.3g} vs stock error {e_stock:.3g} (allowed "
            f"{factor} x stock + {floor:.3g} + {slack:.3g})"
        )


@pytest.fixture(scope="session")
def accuracy() -> Accuracy:
    return Accuracy()


@pytest.fixture(scope="session")
def q4():
    """`q4(shape, seed)`: (weight, scales, biases) of a random matrix quantized as Flash-Next's
    checkpoint is: 4-bit affine, group size 32, bfloat16 scales and biases, evaluated. Values are
    N(0, 1/K) for K input columns, so a matvec of unit-variance activations stays O(1)."""

    def make(shape, seed: int, group_size: int = 32):
        import mlx.core as mx

        w = mx.random.normal(shape, key=mx.random.key(seed)) * shape[-1] ** -0.5
        q, s, b = mx.quantize(w.astype(mx.bfloat16), group_size=group_size, bits=4)
        s, b = s.astype(mx.bfloat16), b.astype(mx.bfloat16)
        mx.eval(q, s, b)
        return q, s, b

    return make


class QSA:
    def __init__(self):
        import mlx.core as mx

        from jitengine_qwen3_8_flash_next_m2.model.config import Qwen4ExpTextConfig
        from jitengine_qwen3_8_flash_next_m2.model.qsa_indexer import Qwen4ExpTextQSAIndexer

        cfg = Qwen4ExpTextConfig()
        nh, hd, h = cfg.indexer_n_heads, cfg.indexer_head_dim, cfg.hidden_size
        key = lambda i: mx.random.key(900 + i)  # noqa: E731
        w = mx.random.normal(((nh + cfg.indexer_kv_heads) * hd, h), key=key(0)) * h**-0.5
        q, s, b = mx.quantize(w.astype(mx.bfloat16), group_size=32, bits=4)
        p = "indexer."
        weights = {
            p + "index_qk_proj.weight": q,
            p + "index_qk_proj.scales": s,
            p + "index_qk_proj.biases": b,
            p + "q_layernorm.weight": (0.1 * mx.random.normal((hd,), key=key(1))).astype(
                mx.bfloat16
            ),
            p + "k_layernorm.weight": (0.1 * mx.random.normal((hd,), key=key(2))).astype(
                mx.bfloat16
            ),
        }
        self.cfg = cfg
        self.ix = Qwen4ExpTextQSAIndexer(cfg, weights, p, mx.float32)

    def inputs(self, P: int, L: int, seed: int):
        """Raw (pre-norm, pre-rope) float32 queries [1, L, nh, hd] at positions P - L .. P - 1 and
        the raw indexer keys of all P positions [1, 1, P, hd], as the slot holds them."""
        import mlx.core as mx

        ix = self.ix
        q = mx.random.normal((1, L, ix.nh, ix.hd), key=mx.random.key(seed))
        keys = mx.random.normal((1, 1, P, ix.hd), key=mx.random.key(seed + 1))
        return q, keys

    def exact_scores(self, q, keys, s0: int) -> np.ndarray:
        """`Qwen4ExpTextQSAIndexer.scores` in float64 [L, nb], on the same rope tables and block
        keys (inputs to both the kernels and stock)."""
        import mlx.core as mx

        from jitengine_qwen3_8_flash_next_m2.model.qsa_indexer import rope_tables

        ix = self.ix
        L, P = q.shape[1], keys.shape[2]
        nb = P // ix.cr
        bk = np.array(ix._block_keys(keys, nb, None), dtype=np.float64)  # [nb, hd]
        cos, sin = (
            np.array(t, np.float64)
            for t in rope_tables(mx.arange(s0, s0 + L), ix.rope_dims, ix.rope_base)
        )
        x = np.array(q, np.float64)[0].transpose(1, 0, 2)  # [nh, L, hd]
        w1 = np.array(ix.q_norm_w1, np.float64)
        x = x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + ix.eps) * w1
        r, half = ix.rope_dims, ix.rope_dims // 2
        xr = x[..., :r]
        rot = np.concatenate([-xr[..., half:], xr[..., :half]], axis=-1)
        x = np.concatenate([xr * cos + rot * sin, x[..., r:]], axis=-1)
        return np.maximum(x @ bk.T, 0.0).sum(axis=0) * ix.scale  # [L, nb]

    def mask_from_scores(self, sc: np.ndarray, P: int) -> np.ndarray:
        """The decode mask the tie rule defines: the top `topk` blocks by (score descending, block
        ascending), each covering `cr` positions, plus the visible ragged tail: bool [P]."""
        ix = self.ix
        keep = np.zeros(sc.size, dtype=bool)
        keep[np.argsort(-sc, kind="stable")[: ix.block_topk]] = True
        return np.concatenate([np.repeat(keep, ix.cr), np.ones(P - sc.size * ix.cr, dtype=bool)])


@pytest.fixture
def qsa(requires_mlx):
    """`QSA`: Flash-Next's QSA indexer (4 heads of 128, compress ratio 4, budget 2048, so 512 blocks
    and pruning from position 2051) on random weights. Needs Metal, so it skips without it."""
    return QSA()
