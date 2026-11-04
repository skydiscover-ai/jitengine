"""`Qwen4ExpForCausalLM(kernels_off=...)`: a custom kernel whose load-time check failed is never
launched, and its
path runs the stock ops instead, on MLX's CPU build. Every custom kernel launch raises here, so the
tiny model's prefill, solo decode and batched decode with every kernel off show that no path
reaches one, and switching each back on alone shows that the path it guards does."""

# ruff: noqa: E402
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")  # the imports below need MLX, so E402 is waived for the file

from jitengine_qwen3_8_flash_next_m2.kernels import gdn_decode as gdn_fused
from jitengine_qwen3_8_flash_next_m2.kernels import gdn_recurrence as gdn_kernel
from jitengine_qwen3_8_flash_next_m2.kernels import grouped_expert_qmv as moe_kernel
from jitengine_qwen3_8_flash_next_m2.kernels import hc_decode as hc_kernel
from jitengine_qwen3_8_flash_next_m2.kernels import indexer_prefill, sdpa_decode_masked
from jitengine_qwen3_8_flash_next_m2.kernels import qmv_row_blocked as dense_kernel
from jitengine_qwen3_8_flash_next_m2.kernels import qsa_decode_mask as attn_fused
from jitengine_qwen3_8_flash_next_m2.kernels import qsa_indexer_decode as indexer_fused
from jitengine_qwen3_8_flash_next_m2.kernels import router_combine as moe_fused
from jitengine_qwen3_8_flash_next_m2.model.kernel_checks import NAMES

KERNEL_MODULES = (
    attn_fused,
    indexer_fused,
    indexer_prefill,
    sdpa_decode_masked,
    gdn_fused,
    gdn_kernel,
    dense_kernel,
    hc_kernel,
    moe_fused,
    moe_kernel,
)
# the kernels the tiny model's paths reach without an expert pool, and the launch each one's path
# makes first
GUARDED = {
    "hc_kernel": "moe_engine_hc_pre",
    "gdn_fused": "moe_engine_qmv_multi",
    "gdn_kernel": "moe_engine_gdn_recurrence",
    "sdpa_decode_masked": "moe_engine_sdpa_decode2_part",
    "attn_fused": "moe_engine_qsa_mask",
    "indexer_fused": "moe_engine_indexer_fused",
    "indexer_prefill": "moe_engine_indexer_prefill_mask",
}


class Launched(Exception):
    """A custom kernel was launched."""


@pytest.fixture
def no_kernels(monkeypatch):
    """`mx.fast.metal_kernel` builds kernels whose launch raises `Launched`, with the kernel
    factories' caches emptied before and after."""
    caches = [
        getattr(m, n)
        for m in KERNEL_MODULES
        for n in dir(m)
        if hasattr(getattr(m, n), "cache_clear")
    ]

    def metal_kernel(name, **_):
        def launch(*args, **kwargs):
            raise Launched(name)

        return launch

    for cache in caches:
        cache.cache_clear()
    monkeypatch.setattr(mx.fast, "metal_kernel", metal_kernel)
    yield
    for cache in caches:
        cache.cache_clear()


@pytest.fixture
def tiny(tiny_flashnext):
    """(config, weights) of the tiny checkpoint."""
    from jitengine_qwen3_8_flash_next_m2.checkpoint.loader import load_language_weights
    from jitengine_qwen3_8_flash_next_m2.model.config import load_config

    path = tiny_flashnext()[0]
    return load_config(path), load_language_weights(path)


def _model(tiny, off, pool=None):
    from jitengine_qwen3_8_flash_next_m2.model.model import Qwen4ExpForCausalLM

    cfg, weights = tiny
    stock = dict(dense_kernel="stock", moe_kernel="stock")
    return Qwen4ExpForCausalLM(
        cfg, weights, pool=pool, calibrate=False, kernels_off=frozenset(off), **stock
    )


def _serve(model, cfg) -> None:
    """Prefill two rows past the indexer's first pruned position, decode one alone, then both."""
    from jitengine_qwen3_8_flash_next_m2.kvcache.state import RequestState

    states = [RequestState(cfg, 24, 4, dtype=mx.float32) for _ in range(2)]
    for i, state in enumerate(states):
        mx.eval(model.forward(mx.arange(3 + i, 27 + i).reshape(1, -1), state))
    mx.eval(model.forward(mx.array([[7]]), states[0]))
    mx.eval(model.decode_step(mx.array([[7], [9]]), states))


def test_with_every_kernel_off_no_path_launches_one(no_kernels, tiny):
    _serve(_model(tiny, NAMES), tiny[0])


@pytest.mark.parametrize("name", GUARDED)
def test_a_kernel_on_alone_is_launched_by_the_path_it_guards(no_kernels, tiny, name):
    with pytest.raises(Launched, match=GUARDED[name]):
        _serve(_model(tiny, set(NAMES) - {name}), tiny[0])


def test_the_fused_moe_paths_follow_their_kernels(tiny):
    """Without a pool the fused MoE paths are off anyway, so their flags are read on a model given
    one: the router/combine kernels gate the fused and speculative paths, and the shared expert's
    fused launch needs the glue too."""
    from jitengine_qwen3_8_flash_next_m2.linear import qlinear_from

    cfg, weights = tiny
    p = "model.layers.0.mlp.switch_mlp."
    names = ("gate_proj", "up_proj", "down_proj")
    pool = SimpleNamespace(**{n: qlinear_from(weights, p + n, cfg) for n in names})

    def flags(off):
        moe = [layer.moe for layer in _model(tiny, off, pool).layers]
        return {(m.fused_ok, m.spec_ok, m.shared_fused_ok) for m in moe}

    assert flags(()) == {(True, True, True)}
    assert flags({"moe_fused"}) == {(False, False, False)}
    assert flags({"gdn_fused"}) == {(True, True, False)}
