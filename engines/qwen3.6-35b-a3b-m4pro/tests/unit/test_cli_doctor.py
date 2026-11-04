"""doctor probes MLX's quantized ops at the width this build serves."""

from __future__ import annotations


def test_doctor_probes_the_served_width(requires_mlx_any, monkeypatch):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro import cli

    widths = []
    real = mx.quantize
    monkeypatch.setattr(mx, "quantize", lambda w, **kw: widths.append(kw["bits"]) or real(w, **kw))
    ops = cli._probe_quant_ops(cli.SERVED_BITS)
    assert cli.SERVED_BITS == 4 and widths == [4]
    assert {k: ops[k] for k in ("quantized_matmul", "gather_qmm", "dequantize")} == {
        "quantized_matmul": True,
        "gather_qmm": True,
        "dequantize": True,
    }
    assert isinstance(
        ops["gather_equals_matmul"], bool
    )  # rounding of the two kernels varies by GPU
