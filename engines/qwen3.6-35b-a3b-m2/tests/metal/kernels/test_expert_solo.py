"""expert_solo's fused single-token expert kernels return the same bits as the stock ops."""


def _qz(mx, shape, bits, seed):
    w = (mx.random.normal(shape, key=mx.random.key(seed)) * (shape[-1] ** -0.5)).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=64, bits=bits)
    return q, s.astype(mx.bfloat16), b.astype(mx.bfloat16)


def test_expert_exact_bitwise_vs_stock(requires_mlx):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m2.kernels import expert_solo as ee
    from jitengine_qwen3_6_35b_a3b_m2.kernels.elementwise import swiglu
    from jitengine_qwen3_6_35b_a3b_m2.linear import QLinear

    E, D, H, k = 12, 2048, 512, 8
    for bits in (3, 4, 8):
        gu = QLinear(*_qz(mx, (E, 2 * H, D), bits, 1), 64)
        dn = QLinear(*_qz(mx, (E, D, H), bits, 2), 64)
        assert ee.supported(gu, dn, mx.bfloat16)
        for c in range(6):
            x = mx.random.normal((1, 1, D), key=mx.random.key(10 + c)).astype(mx.bfloat16)
            logits = mx.random.normal((E,), key=mx.random.key(20 + c))
            inds = mx.argpartition(-logits, kth=k - 1)[:k].astype(mx.uint32)
            scores = mx.softmax(logits[inds]).astype(mx.bfloat16)
            ii = inds.reshape(1, 1, k)
            g, u = mx.split(gu.gather(mx.expand_dims(x, (-2, -3)), ii), 2, axis=-1)
            hh = swiglu(g, u)
            e = dn.gather(hh, ii).squeeze(-2)
            want = (e * scores.reshape(1, 1, k)[..., None]).sum(axis=-2)
            h = ee.expert_gate_up_exact(x.reshape(-1), gu, inds)
            assert mx.array_equal(h, hh.reshape(k, H)).item(), (bits, c, "gate_up")
            got = ee.expert_down_exact(h, dn, inds, scores).reshape(want.shape)
            assert mx.array_equal(got, want).item(), (bits, c, "down")
