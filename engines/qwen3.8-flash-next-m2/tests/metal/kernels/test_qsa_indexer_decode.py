"""qsa_indexer_decode, the decode QSA indexer in one launch: query norm and rope, block scores, mask.

The module claims "the same arithmetic as `Qwen4ExpTextQSAIndexer.scores` + `select_device` in a
fixed order",
with ties ranked as the packed-key argsort ranks them. Its mask must be exactly the tie rule's mask
of its own scores (its second output), the scores must be as accurate as stock `scores` against an
exact float64 computation on the same rope tables and block keys, and `select_device` must return
the kernel's mask. About one block in 16 scores exactly 0 (relu of four negative heads), so at
2052 positions (513 blocks) fewer than 512 blocks score above 0 and the tie rule picks among zeros.
"""

import numpy as np

POSITIONS = (2052, 2053, 3000, 8193)


def _kernel(qsa, P, seed):
    from jitengine_qwen3_8_flash_next_m2.kernels.qsa_indexer_decode import indexer_mask, supports
    from jitengine_qwen3_8_flash_next_m2.model.qsa_indexer import rope_tables_at

    ix = qsa.ix
    q, keys = qsa.inputs(P, 1, seed=seed)
    s0, nb = P - 1, P // ix.cr
    assert supports(ix.nh, ix.hd, ix.rope_dims, nb)
    bk = ix._block_keys(keys, nb, None)
    cos, sin = rope_tables_at(s0, ix.rope_dims, ix.rope_base)
    mask, sc = indexer_mask(
        q.reshape(ix.nh, ix.hd),
        ix.q_norm_w1,
        bk,
        cos,
        sin,
        s0,
        nb,
        ix.cr,
        ix.block_topk,
        P,
        ix.eps,
        ix.rope_base,
        ix.rope_dims,
        ix.hd,
    )
    return q, keys, mask, sc


def test_mask_is_the_tie_rule_on_its_own_scores(qsa):
    import mlx.core as mx

    for P in POSITIONS:
        q, keys, mask, sc = _kernel(qsa, P, seed=P)
        assert mask.shape == (1, 1, 1, P) and sc.shape == (P // qsa.ix.cr,)
        scores = np.array(sc)
        if P == POSITIONS[0]:  # the boundary falls among tied zeros
            assert 0 < (scores > 0).sum() < qsa.ix.block_topk < scores.size
        want = qsa.mask_from_scores(scores, P)
        assert np.array_equal(np.array(mask).reshape(-1), want), P
        model = qsa.ix.select_device(q, keys, P - 1)  # the attention block's call
        assert mx.array_equal(model, mask).item(), P


def test_scores_as_accurate_as_stock(qsa, accuracy):
    for P in POSITIONS:
        q, keys, _, sc = _kernel(qsa, P, seed=P)
        stock, nb = qsa.ix.scores(q, keys, P - 1)
        exact = qsa.exact_scores(q, keys, P - 1)
        accuracy.no_worse_than_stock(sc.reshape(1, nb), stock, exact, what=f"P={P}")
