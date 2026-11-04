"""Where the exact decode SDPA runs its own kernel and where it leaves the shape to stock."""

import pytest


def test_exact_sdpa_stops_where_stock_switches_to_two_passes(requires_mlx_any, monkeypatch):
    """MLX 0.32.2: two passes from 1024 keys on Pro, Max and Ultra GPUs, from 4096 with GQA."""
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m2.kernels import sdpa_decode as sdpa_exact

    for gpu, gqa, mha in (
        ("g", (True, True, True, False), (True, True, True, True)),
        ("s", (True, False, False, False), (True, False, False, False)),
        ("d", (True, False, False, False), (True, False, False, False)),
    ):
        monkeypatch.setattr(sdpa_exact, "_gpu_family", lambda gpu=gpu: gpu)
        assert tuple(sdpa_exact.single_pass(n, 16, 2) for n in (1023, 1024, 4095, 4096)) == gqa, gpu
        assert tuple(sdpa_exact.single_pass(n, 16, 16) for n in (1023, 1024, 4095, 4096)) == mha, (
            gpu
        )
    # and only for shapes stock's vector kernel takes: its head dims, at most 32 query heads per KV head
    monkeypatch.setattr(sdpa_exact, "_gpu_family", lambda: "g")
    q = lambda h, d: mx.zeros((1, h, 1, d), dtype=mx.bfloat16)  # noqa: E731
    kv = lambda h, d: mx.zeros((1, h, 64, d), dtype=mx.bfloat16)  # noqa: E731
    assert sdpa_exact.supported(q(16, 256), kv(2, 256), 50)
    assert not sdpa_exact.supported(q(16, 160), kv(2, 160), 50)
    assert not sdpa_exact.supported(q(64, 128), kv(1, 128), 50)


def test_rows_outside_the_contract_raise(requires_mlx_any):
    """`sdpa_decode_rows` takes 1 to MAXB rows; any other count is the caller's bug, refused
    before a launch."""
    import mlx.core as mx
    import pytest

    from jitengine_qwen3_6_35b_a3b_m2.kernels import sdpa_decode as sdpa_exact

    kv = mx.zeros((1, 2, 64, 256), dtype=mx.bfloat16)
    for rows in (0, sdpa_exact.MAXB + 1):
        qs = mx.zeros((rows, 16, 1, 256), dtype=mx.bfloat16)
        with pytest.raises(ValueError, match=f"got {rows}"):
            sdpa_exact.sdpa_decode_rows(qs, [kv] * rows, [kv] * rows, [8] * rows, 1.0)


KV = (1, 2, 64, 256)  # [1, Hkv, cap, D]
# operand shapes (q, keys, values) and a length that break `decode`'s contract, and the f32 query
CONTRACT_BREAKS = {
    "a rank-3 query": ((16, 1, 256), KV, KV, 8),
    "two queries": ((1, 16, 2, 256), KV, KV, 8),
    "another head dim": ((1, 16, 1, 256), (1, 2, 64, 128), (1, 2, 64, 128), 8),
    "values unlike keys": ((1, 16, 1, 256), KV, (1, 2, 32, 256), 8),
    "heads that do not share kv heads": ((1, 15, 1, 256), KV, KV, 8),
    "a float32 query": ((1, 16, 1, 256), KV, KV, 8),
    "no keys": ((1, 16, 1, 256), KV, KV, 0),
    "more keys than the buffer": ((1, 16, 1, 256), KV, KV, 65),
}


@pytest.mark.parametrize("case", list(CONTRACT_BREAKS), ids=list(CONTRACT_BREAKS))
def test_operands_outside_the_contract_raise(requires_mlx_any, case):
    """`decode` refuses what neither its kernel nor stock's would honour, before any launch."""
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m2.kernels import sdpa_decode as sdpa_exact

    q, k, v, n = CONTRACT_BREAKS[case]
    dtype = mx.float32 if case == "a float32 query" else mx.bfloat16
    with pytest.raises(ValueError):
        sdpa_exact.decode(
            mx.zeros(q, dtype=dtype), mx.zeros(k, mx.bfloat16), mx.zeros(v, mx.bfloat16), n, 0.0625
        )


def test_decode_rows_keeps_the_contract_in_every_row(requires_mlx_any):
    """`decode_rows` keeps `decode`'s contract in every row, and takes one of each operand per row."""
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m2.kernels import sdpa_decode as sdpa_exact

    qs, kv = mx.zeros((2, 16, 1, 256), mx.bfloat16), mx.zeros(KV, mx.bfloat16)
    with pytest.raises(ValueError, match="65"):
        sdpa_exact.decode_rows(qs, [kv, kv], [kv, kv], [8, 65], 0.0625)
    with pytest.raises(ValueError, match="per row"):
        sdpa_exact.decode_rows(qs, [kv], [kv], [8], 0.0625)
