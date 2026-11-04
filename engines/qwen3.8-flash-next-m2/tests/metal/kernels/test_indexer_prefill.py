"""indexer_prefill, a prefill chunk's QSA mask built on the device.

The module claims row i (position p) is plain causal while it has at most `topk` complete blocks,
else keeps the top `topk` blocks by (score descending, block ascending), the numpy path's stable
argsort tie rule, plus the ragged tail [nb * cr, p]. Against `Qwen4ExpTextQSAIndexer.select` (stock
scores, the
numpy mask) causal rows and tails must match exactly. The kernel computes the scores itself, in
another order, so a pruned row may differ only where rounding can reorder two blocks: at a near tie
of the exact scores at the top-k boundary, within the kernel's and stock's score error (stock's
measured here, the kernel's at most 8 times it, as the decode kernel's scores are checked).
"""

import numpy as np

CHUNKS = ((2000, 128), (3000, 64), (5000, 256))  # (first position, length): rows on both sides


def test_matches_the_numpy_mask_up_to_rounding_ties(qsa):
    ix = qsa.ix
    for s0, L in CHUNKS:
        P = s0 + L
        q, keys = qsa.inputs(P, L, seed=s0)
        assert ix.device_prefill_ok(P)
        got = np.array(ix.select_prefill_device(q, keys, s0))
        assert got.shape == (1, 1, L, P)
        got = got[0, 0]
        want = ix.select(q, keys, s0)
        stock, _ = ix.scores(q, keys, s0)
        exact = qsa.exact_scores(q, keys, s0)
        floor = 4 * 2.0**-24 * np.abs(exact).max()
        tol = 2 * (9 * np.abs(np.array(stock, np.float64) - exact).max() + floor)
        for i in np.nonzero((got != want).any(axis=1))[0]:
            p = s0 + i
            nbi = (p + 1) // ix.cr
            assert nbi > ix.block_topk, f"causal row {i} (position {p}) differs"
            cols = np.nonzero(got[i] != want[i])[0]
            assert cols.max() < nbi * ix.cr, f"row {i}: the tail differs"
            row = np.sort(exact[i, :nbi])[::-1]
            gap = row[ix.block_topk - 1] - row[ix.block_topk]
            assert gap <= tol, f"row {i}: differs with no near tie at the boundary ({gap} > {tol})"
            edge = 0.5 * (row[ix.block_topk - 1] + row[ix.block_topk])
            for j in np.unique(cols // ix.cr):
                assert abs(exact[i, j] - edge) <= tol, f"row {i}: block {j} is not at the boundary"
