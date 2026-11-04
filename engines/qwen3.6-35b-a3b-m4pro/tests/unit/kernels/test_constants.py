"""The kernels' constant inputs are built once per value."""


def test_a_constant_is_built_once_per_value(requires_mlx_any):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro.kernels import constants

    assert constants.f32(0.0625) is constants.f32(0.0625)
    assert constants.i32(512) is constants.i32(512) and constants.i32(512) is not constants.i32(768)
    assert constants.f32(0.0625).dtype == mx.float32 and constants.i32(512).dtype == mx.int32
    assert constants.i32(4096, 2048).tolist() == [4096, 2048]
