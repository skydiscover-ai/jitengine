"""sdpa_decode_masked, Flash-Next's decode attention over the QSA-selected keys.

The module says results "match MLX's unfused SDPA path (its fallback at head dim 256) up to
summation order" and states no tolerance, so against stock `mx.fast.scaled_dot_product_attention`
with the same boolean mask (what the batched step runs) the kernel must be as accurate as stock
against an exact float64 attention. Inputs are what Qwen4ExpTextAttention passes: q [1, 24, 1, 256],
the slot's whole K/V buffers [1, 2, capacity, 256] with the first P valid, a mask of whole
4-position blocks (512 kept) plus the visible ragged tail, float32. Past P the buffers hold NaN and
the mask the kernel gets runs on true, so a kernel that read past P would return NaN.
"""

import numpy as np
import pytest

H, HKV, D, CR, TOPK = 24, 2, 256, 4, 512


def _mask(P, seed, structured=False):
    nb = P // CR
    keep = np.zeros(nb, dtype=bool)
    if structured:  # whole chunks of keys masked out between the kept ends
        keep[: TOPK // 2] = keep[nb - TOPK // 2 :] = True
    else:
        keep[np.random.default_rng(seed).permutation(nb)[:TOPK]] = True
    return np.concatenate([np.repeat(keep, CR), np.ones(P - nb * CR, dtype=bool)])


def _exact(q, k, v, mask, scale):
    """[H, D] float64: softmax over the unmasked keys of each head's KV head."""
    q, k, v = (np.array(a, dtype=np.float64) for a in (q, k, v))
    out = np.empty((H, D))
    for h in range(H):
        kv = h // (H // HKV)
        s = (k[0, kv][mask] @ q[0, h, 0]) * scale
        p = np.exp(s - s.max())
        out[h] = (p / p.sum()) @ v[0, kv][mask]
    return out


def test_as_accurate_as_stock_sdpa_with_the_same_mask(requires_mlx, accuracy):
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels.sdpa_decode_masked import (
        decode_masked,
        supports,
    )
    from jitengine_qwen3_8_flash_next_m4pro.model.attention import (
        MASKED_SDPA_CHUNK,
        MASKED_SDPA_HEADS_PER_SIMDGROUP,
    )

    scale = D**-0.5
    assert supports(H, HKV, D, MASKED_SDPA_HEADS_PER_SIMDGROUP)
    cases = [(2052, False), (2053, False), (2600, False), (4099, False), (8191, True)]
    for t, (P, structured) in enumerate(cases):
        cap = -(-(P + 1) // 256) * 256 + 256
        q = mx.random.normal((1, H, 1, D), key=mx.random.key(t))
        kv = [mx.random.normal((1, HKV, P, D), key=mx.random.key(10 + t + i)) for i in (0, 1)]
        pad = mx.full((1, HKV, cap - P, D), float("nan"))
        kbuf, vbuf = (mx.concatenate([a, pad], axis=2) for a in kv)
        m = _mask(P, seed=t, structured=structured)
        mask = mx.array(m)[None, None, None, :]
        # the kernel's mask runs on, all true, over the NaN rows: reading past P shows as NaN
        padded = mx.concatenate([mask, mx.ones((1, 1, 1, cap - P), dtype=mx.bool_)], axis=-1)
        got = decode_masked(
            q,
            kbuf,
            vbuf,
            P,
            scale,
            padded,
            chunk=MASKED_SDPA_CHUNK,
            hps=MASKED_SDPA_HEADS_PER_SIMDGROUP,
        )
        stock = mx.fast.scaled_dot_product_attention(
            q, kbuf[:, :, :P], vbuf[:, :, :P], scale=scale, mask=mask
        )
        assert got.shape == stock.shape == (1, H, 1, D)
        exact = _exact(q, *kv, m, scale)
        accuracy.no_worse_than_stock(got.reshape(H, D), stock.reshape(H, D), exact, what=f"P={P}")


def _padded_mask(P, cap, seed):
    import mlx.core as mx

    m = np.concatenate([_mask(P, seed), np.ones(cap - P, dtype=bool)])
    return mx.array(m)[None, None, None, :]


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_two_rows_in_one_launch_are_each_the_row_alone(requires_mlx, dtype):
    """`decode_masked2`, the verify pass's launch: row 0 over P keys and row 1 over P + 1, each
    under its own mask, from one K/V buffer that runs past P + 1 with NaN. Each row must equal
    `decode_masked` on that row alone, bit for bit, including where P + 1 opens one more chunk."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels.sdpa_decode_masked import (
        decode_masked,
        decode_masked2,
    )
    from jitengine_qwen3_8_flash_next_m4pro.model.attention import (
        MASKED_SDPA_CHUNK,
        MASKED_SDPA_HEADS_PER_SIMDGROUP,
    )

    dt, scale, ch, hps = (
        getattr(mx, dtype),
        D**-0.5,
        MASKED_SDPA_CHUNK,
        MASKED_SDPA_HEADS_PER_SIMDGROUP,
    )
    for t, P0 in enumerate((2052, 17 * ch, 3001)):
        P1 = P0 + 1
        cap = -(-(P1 + 1) // 256) * 256 + 256
        q = mx.random.normal((3, H, 1, D), key=mx.random.key(40 + t)).astype(dt)
        kv = [
            mx.random.normal((1, HKV, P1, D), key=mx.random.key(50 + t + i)).astype(dt)
            for i in (0, 1)
        ]
        pad = mx.full((1, HKV, cap - P1, D), float("nan"), dtype=dt)
        kbuf, vbuf = (mx.concatenate([a, pad], axis=2) for a in kv)
        m0, m1 = _padded_mask(P0, cap, 2 * t), _padded_mask(P1, cap, 2 * t + 1)
        both = decode_masked2(q[1:2], q[2:3], kbuf, vbuf, P0, P1, scale, m0, m1, chunk=ch, hps=hps)
        for r, (P, m) in enumerate(((P0, m0), (P1, m1))):
            alone = decode_masked(q[1 + r : 2 + r], kbuf, vbuf, P, scale, m, chunk=ch, hps=hps)
            assert both[r].shape == alone.shape == (1, H, 1, D)
            assert not mx.any(mx.isnan(both[r])).item()
            assert mx.array_equal(both[r], alone).item(), (P0, r)


def test_the_load_time_check_catches_a_reordered_reduction(requires_mlx, monkeypatch):
    """`check_decode_masked2` passes the shipped kernel and fails one whose reduce adds the chunks
    in reverse order (the same values in another summation order)."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels import sdpa_decode_masked as sd
    from jitengine_qwen3_8_flash_next_m4pro.model.attention import (
        MASKED_SDPA_CHUNK,
        MASKED_SDPA_HEADS_PER_SIMDGROUP,
    )

    args = (H, HKV, D, mx.float32, MASKED_SDPA_CHUNK, MASKED_SDPA_HEADS_PER_SIMDGROUP, CR, TOPK)
    assert sd.check_decode_masked2(*args)
    reordered = sd.REDUCE2_SOURCE.replace(
        "for (int c = 0; c < NCH; c++) {\n      const ulong pi",
        "for (int c = NCH - 1; c >= 0; c--) {\n      const ulong pi",
    )
    assert reordered != sd.REDUCE2_SOURCE
    neg = mx.fast.metal_kernel(
        name="test_sdpa_decode2_reduce_rows2_reordered",
        input_names=["pm", "pl", "pacc", "dims"],
        output_names=["o"],
        header=sd.HEADER,
        source=reordered,
        ensure_row_contiguous=True,
    )
    real = sd._kernel
    monkeypatch.setattr(sd, "_kernel", lambda which: neg if which == "reduce2" else real(which))
    assert not sd.check_decode_masked2(*args)
