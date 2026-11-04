"""qsa_decode_mask, the QSA decode mask in one launch from the block scores
`Qwen4ExpTextQSAIndexer.scores` returns.

qsa_decode_mask.py claims block j is kept iff fewer than `topk` blocks precede it in (score descending,
block index ascending) order, the packed-key argsort's tie rule, with the ragged tail visible. It
only compares scores, so on the same scores it must give exactly the mask of `select_device`'s
stock packed-key path, ties included: random scores (where relu leaves blocks at exactly 0) and
scores built to tie, up to the 4096 blocks the kernel takes.
"""

import numpy as np


def _masks(qsa, monkeypatch, P, scores=None, seed=0):
    """(kernel, stock) masks from `select_device` with the one-launch indexer off, so the scores
    are stock's (or `scores`, when given) and only the mask construction differs."""
    from jitengine_qwen3_8_flash_next_m4pro.kernels import qsa_decode_mask as attn_fused
    from jitengine_qwen3_8_flash_next_m4pro.kernels import qsa_indexer_decode as indexer_fused

    ix = qsa.ix
    q, keys = qsa.inputs(P, 1, seed=seed)
    with monkeypatch.context() as m:
        m.setattr(indexer_fused, "supports", lambda *a: False)
        if scores is not None:
            m.setattr(ix, "scores", lambda *a, **kw: (scores[None], scores.shape[0]))
        kernel = ix.select_device(q, keys, P - 1)
        m.setattr(attn_fused, "supports", lambda nb: False)
        stock = ix.select_device(q, keys, P - 1)
    return kernel, stock


def test_equals_the_packed_key_argsort_on_stock_scores(qsa, monkeypatch):
    import mlx.core as mx

    for P in (2052, 2053, 2600, 4099, 9000):
        kernel, stock = _masks(qsa, monkeypatch, P, seed=P)
        assert kernel.shape == stock.shape == (1, 1, 1, P)
        assert mx.array_equal(kernel, stock).item(), P
        sc, _ = qsa.ix.scores(*qsa.inputs(P, 1, seed=P), P - 1)
        assert np.array_equal(
            np.array(kernel).reshape(-1), qsa.mask_from_scores(np.array(sc[0]), P)
        )


def test_equals_the_packed_key_argsort_on_tied_scores(qsa, monkeypatch):
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels.qsa_decode_mask import NBMAX

    cr = qsa.ix.cr
    for t, nb in enumerate((513, 700, 2048, NBMAX)):
        for kind in ("levels", "zeros", "mostly zero"):
            if kind == "levels":  # four values: hundreds of exact ties at the boundary
                sc = mx.random.randint(0, 4, (nb,), key=mx.random.key(t)).astype(mx.float32)
            elif kind == "zeros":
                sc = mx.zeros((nb,), dtype=mx.float32)
            else:
                r = mx.random.normal((nb,), key=mx.random.key(t))
                sc = mx.maximum(r - 1.5, 0.0)  # about 7% positive
            P = nb * cr + t % cr  # ragged tails of 0 to 3 positions
            kernel, stock = _masks(qsa, monkeypatch, P, scores=sc, seed=t)
            assert mx.array_equal(kernel, stock).item(), (nb, kind)
            want = qsa.mask_from_scores(np.array(sc), P)
            assert np.array_equal(np.array(kernel).reshape(-1), want), (nb, kind)
