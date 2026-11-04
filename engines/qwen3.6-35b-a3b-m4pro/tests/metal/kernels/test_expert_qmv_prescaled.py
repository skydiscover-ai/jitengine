"""The prescaled routed-expert kernels, and one-expert views of their stacks: the same bits."""

from __future__ import annotations


def _qz(mx, shape, bits, seed):
    w = (mx.random.normal(shape, key=mx.random.key(seed)) * (shape[-1] ** -0.5)).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=64, bits=bits)
    return q, s.astype(mx.bfloat16), b.astype(mx.bfloat16)


def test_expert_v4_solo_and_rows_bitwise_vs_expert_exact(requires_mlx):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro.kernels import expert_solo as ee
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.expert_qmv_prescaled import (
        expert_gate_up_prescaled,
        expert_gate_up_prescaled_rows,
    )
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.expert_rows import experts_rows_exact
    from jitengine_qwen3_6_35b_a3b_m4pro.linear import QLinear

    E, D, H, k = 24, 2048, 512, 8
    gu = QLinear(*_qz(mx, (E, 2 * H, D), 3, 1), 64)
    dn = QLinear(*_qz(mx, (E, D, H), 3, 2), 64)
    for M in (1, 2, 3, 4):
        x = mx.random.normal((M, D), key=mx.random.key(30 + M)).astype(mx.bfloat16)
        logits = mx.random.normal((M, E), key=mx.random.key(40 + M))
        inds = mx.argpartition(-logits, kth=k - 1, axis=-1)[:, :k].astype(mx.uint32)
        scores = mx.softmax(mx.take_along_axis(logits, inds, axis=-1), axis=-1).astype(mx.bfloat16)
        want_h = mx.stack([ee.expert_gate_up_exact(x[r], gu, inds[r]) for r in range(M)])
        want_y = mx.stack(
            [ee.expert_down_exact(want_h[r], dn, inds[r], scores[r]) for r in range(M)]
        )
        if M == 1:
            assert mx.array_equal(expert_gate_up_prescaled(x, gu, inds[0]), want_h[0]).item()
            assert mx.array_equal(
                ee.expert_down_exact_par(want_h[0], dn, inds[0], scores[0], rps=8), want_y[0]
            ).item()
        h = expert_gate_up_prescaled_rows(x, gu, inds).reshape(M, k, H)
        assert mx.array_equal(h, want_h).item(), M
        y = ee.expert_down_exact_par(
            h.reshape(M * k, H), dn, inds.reshape(-1), scores.reshape(-1), rows=M, rps=8
        )
        assert mx.array_equal(y.reshape(M, D), want_y).item(), M
        assert mx.array_equal(experts_rows_exact(x, gu, dn, inds, scores), want_y).item(), M


def test_one_expert_views_change_no_bits(requires_mlx):
    """Expert stacks reach the kernels as one-expert views of their buffers
    (`QLinear.expert_views`); passing the whole stacks instead gives identical outputs."""
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro import linear as Q
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels import expert_solo as ee
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.expert_qmv_prescaled import (
        expert_gate_up_prescaled,
        expert_gate_up_prescaled_rows,
    )
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.gather_qmm_prefill import seg_gather_qmm

    class Whole(Q.QLinear):
        """The same stack, handed to the kernels whole."""

        __slots__ = ()

        def expert_views(self):
            return self.weight, self.scales, self.biases

    E, D, H, k = 24, 2048, 512, 8
    gu_arrays, dn_arrays = _qz(mx, (E, 2 * H, D), 3, 21), _qz(mx, (E, D, H), 3, 22)
    x = mx.random.normal((4, D), key=mx.random.key(23)).astype(mx.bfloat16)
    inds = mx.argpartition(-mx.random.normal((4, E), key=mx.random.key(24)), kth=k - 1, axis=-1)[
        :, :k
    ].astype(mx.uint32)
    sc = mx.random.uniform(shape=(4 * k,), key=mx.random.key(25)).astype(mx.bfloat16)
    flat = inds.reshape(-1)
    order = mx.argsort(flat)
    xs = mx.repeat(x, k, axis=0)[order]

    def run(cls):
        gu, dn = cls(*gu_arrays, 64), cls(*dn_arrays, 64)
        h1 = expert_gate_up_prescaled(x[:1], gu, inds[0])
        h4 = expert_gate_up_prescaled_rows(x, gu, inds)
        y4 = ee.expert_down_exact_par(h4, dn, flat, sc, rows=4, rps=8)
        g = seg_gather_qmm(xs, gu, flat[order])
        mx.eval(h1, h4, y4, g)
        return h1, h4, y4, g

    for a, b in zip(run(Q.QLinear), run(Whole)):
        assert mx.array_equal(a, b).item()
