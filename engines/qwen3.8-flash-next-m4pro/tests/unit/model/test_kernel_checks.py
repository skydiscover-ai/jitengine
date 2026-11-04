"""Which load-time kernel checks `run_kernel_checks` runs, and with what geometry, on MLX's CPU build.
The checks themselves need Metal (tests/metal/model/test_kernel_checks.py runs them), so
here each is replaced by a recorder of its arguments."""

import pytest

mx = pytest.importorskip("mlx.core")

from jitengine_qwen3_8_flash_next_m4pro.model import kernel_checks as kc  # noqa: E402

CHECKS = {
    "dense_kernel": (kc.dense_kernel, "check_dense"),
    "moe_kernel": (kc.moe_kernel, "check_grouped"),
    "moe_fused": (kc.moe_fused, "check_router_combine"),
    "hc_kernel": (kc.hc_kernel, "check_hc"),
    "gdn_fused": (kc.gdn_fused, "check_glue"),
    "gdn_kernel": (kc.gdn_kernel, "check_recurrence"),
    "sdpa_decode_masked": (kc.sdpa_decode_masked, "check_decode_masked"),
    "attn_fused": (kc.attn_fused, "check_qsa_mask"),
    "indexer_fused": (kc.indexer_fused, "check_indexer"),
    "indexer_prefill": (kc.indexer_prefill, "check_prefill_mask"),
    "qmv_multi_v": (kc.gdn_fused, "check_multi_v"),
    "grouped_qmv_u2": (kc.moe_kernel, "check_grouped_u2"),
    "sdpa_decode_masked2": (kc.sdpa_decode_masked, "check_decode_masked2"),
    "gdn_pair": (kc.gdn_kernel, "check_pair"),
}
# the MTP verify pass's kernels (`gdn_pair` under `gdn.GDN_LAZY`, on by default)
VERIFY = {"qmv_multi_v", "grouped_qmv_u2", "sdpa_decode_masked2", "gdn_pair"}


def _recorders(monkeypatch, verdicts=None) -> dict:
    """Every check replaced by one that records its arguments and answers its verdict (default
    True; an exception is raised)."""
    calls = {}
    for name, (module, fn) in CHECKS.items():

        def check(*args, name=name):
            calls[name] = args
            verdict = (verdicts or {}).get(name, True)
            if isinstance(verdict, Exception):
                raise verdict
            return verdict

        monkeypatch.setattr(module, fn, check)
    return calls


@pytest.fixture
def tiny_cfg(tiny_flashnext):
    from jitengine_qwen3_8_flash_next_m4pro.model.config import load_config

    return load_config(tiny_flashnext()[0])


def test_every_kernel_has_a_check():
    assert set(CHECKS) == set(kc.NAMES)


def test_the_real_geometry_checks_every_kernel(monkeypatch):
    from jitengine_qwen3_8_flash_next_m4pro.model.config import Qwen4ExpTextConfig

    calls = _recorders(monkeypatch)
    checks = kc.run_kernel_checks(
        Qwen4ExpTextConfig(), mx.float32, "rb", "grouped", "split", mtp=True
    )
    assert checks.results == dict.fromkeys(kc.NAMES, True) and checks.errors == {}
    assert set(calls) == set(kc.NAMES)
    assert calls["moe_kernel"] == (2560, 640, 4, 32, mx.float32, 10, "split")
    assert calls["grouped_qmv_u2"] == (2560, 640, 4, 32, mx.float32, 10)
    assert calls["qmv_multi_v"] == calls["gdn_fused"][:2]  # the solo path's weight sets
    assert calls["sdpa_decode_masked2"] == calls["sdpa_decode_masked"]  # the solo launch's geometry
    assert calls["moe_fused"] == (512, 10, True, 4, 2560, mx.float32)


def test_the_tiny_geometry_has_no_dense_shape_to_check(monkeypatch, tiny_cfg):
    calls = _recorders(monkeypatch)
    checks = kc.run_kernel_checks(tiny_cfg, mx.float32, "rb", "grouped", "split", mtp=True)
    assert set(calls) == set(checks.results) == set(kc.NAMES) - {"dense_kernel"}


def test_a_kernel_the_options_turn_off_is_not_checked(monkeypatch):
    from jitengine_qwen3_8_flash_next_m4pro.model.config import Qwen4ExpTextConfig

    calls = _recorders(monkeypatch)
    kc.run_kernel_checks(Qwen4ExpTextConfig(), mx.float32, "stock", "stock", "split")
    assert set(calls) == set(kc.NAMES) - {"dense_kernel", "moe_kernel"} - VERIFY


def test_the_verify_kernels_are_checked_only_with_an_mtp_head(monkeypatch):
    from jitengine_qwen3_8_flash_next_m4pro.model.config import Qwen4ExpTextConfig

    calls = _recorders(monkeypatch)
    checks = kc.run_kernel_checks(Qwen4ExpTextConfig(), mx.float32, "rb", "grouped", "split")
    assert set(calls) == set(checks.results) == set(kc.NAMES) - VERIFY


def test_the_verify_kernels_are_checked_whatever_the_solo_kernels_and_options_are(monkeypatch):
    """The verify pass launches both either way, so a failed `gdn_fused` or `moe_kernel`, or the
    option `moe_kernel="stock"`, does not skip them."""
    from jitengine_qwen3_8_flash_next_m4pro.model.config import Qwen4ExpTextConfig

    cfg = Qwen4ExpTextConfig()
    for failed in ("gdn_fused", "moe_kernel"):
        calls = _recorders(monkeypatch, {failed: False})
        kc.run_kernel_checks(cfg, mx.float32, "rb", "grouped", "split", mtp=True)
        assert VERIFY <= set(calls)
    calls = _recorders(monkeypatch)
    kc.run_kernel_checks(cfg, mx.float32, "stock", "stock", "split", mtp=True)
    assert set(calls) == set(kc.NAMES) - {"dense_kernel", "moe_kernel"}


def test_failed_and_raising_verify_checks_are_recorded(monkeypatch):
    from jitengine_qwen3_8_flash_next_m4pro.model.config import Qwen4ExpTextConfig

    _recorders(
        monkeypatch,
        {
            "qmv_multi_v": False,
            "grouped_qmv_u2": RuntimeError("no library"),
            "sdpa_decode_masked2": False,
            "gdn_pair": False,
        },
    )
    checks = kc.run_kernel_checks(
        Qwen4ExpTextConfig(), mx.float32, "rb", "grouped", "split", mtp=True
    )
    assert {name for name, ok in checks.results.items() if not ok} == VERIFY
    assert checks.errors == {"grouped_qmv_u2": "RuntimeError: no library"}


def test_the_pair_launch_is_checked_only_under_gdn_lazy(monkeypatch):
    """`gdn_recurrence_pair` runs only in the lazy pair build, so with `GDN_LAZY` off it is not
    checked, and with it on it is checked at the geometry of the recurrence's own check."""
    from jitengine_qwen3_8_flash_next_m4pro.model import gdn
    from jitengine_qwen3_8_flash_next_m4pro.model.config import Qwen4ExpTextConfig

    assert gdn.GDN_LAZY
    calls = _recorders(monkeypatch)
    kc.run_kernel_checks(Qwen4ExpTextConfig(), mx.float32, "rb", "grouped", "split", mtp=True)
    assert calls["gdn_pair"] == calls["gdn_kernel"][:5] == (16, 48, 128, 128, mx.float32)
    monkeypatch.setattr(gdn, "GDN_LAZY", False)
    calls = _recorders(monkeypatch)
    checks = kc.run_kernel_checks(
        Qwen4ExpTextConfig(), mx.float32, "rb", "grouped", "split", mtp=True
    )
    assert set(calls) == set(checks.results) == set(kc.NAMES) - {"gdn_pair"}


def test_failed_and_raising_checks_are_recorded(monkeypatch):
    from jitengine_qwen3_8_flash_next_m4pro.model.config import Qwen4ExpTextConfig

    _recorders(monkeypatch, {"attn_fused": False, "gdn_kernel": RuntimeError("no library")})
    checks = kc.run_kernel_checks(Qwen4ExpTextConfig(), mx.float32, "rb", "grouped", "split")
    off = {name for name, ok in checks.results.items() if not ok}
    assert off == {"attn_fused", "gdn_kernel"}
    assert checks.errors == {"gdn_kernel": "RuntimeError: no library"}


def _glue_metadata(calls) -> dict:
    """{(out, in): metadata dtype} of the weight sets the glue check was given."""
    sets, _launch, _swiglu = zip(*calls["gdn_fused"][0], strict=True)
    return {(n, k): meta for group in sets for n, k, _bits, _gs, meta in group}


def test_a_failed_dense_check_leaves_every_linear_with_float32_metadata(monkeypatch):
    """As `apply_dense_policy` does: a served shape keeps the checkpoint's bfloat16 metadata, so
    the glue and hyper-connection checks read the dtypes the model will hold."""
    from jitengine_qwen3_8_flash_next_m4pro.model.config import Qwen4ExpTextConfig

    calls = _recorders(monkeypatch)
    kc.run_kernel_checks(Qwen4ExpTextConfig(), mx.float32, "rb", "grouped", "split")
    served = _glue_metadata(calls)
    assert served[(10240, 2560)] == mx.bfloat16 and served[(48, 2560)] == mx.float32
    assert calls["hc_kernel"][7] == [mx.bfloat16] * 3
    calls = _recorders(monkeypatch, {"dense_kernel": False})
    kc.run_kernel_checks(Qwen4ExpTextConfig(), mx.float32, "rb", "grouped", "split")
    assert set(_glue_metadata(calls).values()) == {mx.float32}
    assert calls["hc_kernel"][7] == [mx.float32] * 3


@pytest.mark.parametrize("serve", [None, (96, 64)])
def test_shapes_and_metadata_are_the_built_model_s(monkeypatch, tiny_cfg, tiny_flashnext, serve):
    """`dense_shapes` and the metadata dtypes the checks assume, against the tiny model built on
    the CPU: with the real table (no tiny shape served) and with one tiny shape served."""
    from jitengine_qwen3_8_flash_next_m4pro.checkpoint.loader import load_language_weights
    from jitengine_qwen3_8_flash_next_m4pro.linear import QEmbedding
    from jitengine_qwen3_8_flash_next_m4pro.model.model import Qwen4ExpForCausalLM

    if serve is not None:
        monkeypatch.setitem(kc.dense_kernel.CONFIGS, (*serve, "float32"), (16, 2, 2))
    weights = load_language_weights(tiny_flashnext()[0])
    model = Qwen4ExpForCausalLM(tiny_cfg, weights, calibrate=False)
    lins = [lin for lin in model.dense_linears() if not isinstance(lin, QEmbedding)]
    built = {(lin.out_dims, lin.in_dims, lin.bits, lin.group_size) for lin in lins}
    assert built == set(kc.dense_shapes(tiny_cfg))
    predict = kc._Linears(tiny_cfg, mx.float32, served=True)
    for name, lin in _named_linears(model).items():
        assert predict(name, lin.out_dims, lin.in_dims).meta == lin.scales.dtype, name


def _named_linears(model) -> dict:
    """{module path: QLinear} for the first layer of each kind, as the checks name them."""
    out = {}
    for i, layer in enumerate(model.layers):
        p = f"model.layers.{i}."
        hc = layer.hc_attn
        out[p + "attn_hyper_connection.input_mix_weight_down"] = hc.mix_down
        out[p + "attn_hyper_connection.input_mix_weight_up"] = hc.mix_up
        out[p + "attn_hyper_connection.block_inject_weight"] = hc.block_inject
        moe = layer.moe
        out[p + "mlp.shared_expert.gate_proj"] = moe.shared_gate_proj
        out[p + "mlp.shared_expert.down_proj"] = moe.shared_down_proj
        blk = layer.block
        if layer.is_attention:
            out[p + "self_attn.q_proj"] = blk.q_proj
            out[p + "self_attn.k_proj"] = blk.k_proj
            out[p + "self_attn.indexer.index_qk_proj"] = blk.indexer.qk_proj
        else:
            out[p + "linear_attn.in_proj_qkv"] = blk.in_proj_qkv
            out[p + "linear_attn.in_proj_b"] = blk.in_proj_b
    return out
