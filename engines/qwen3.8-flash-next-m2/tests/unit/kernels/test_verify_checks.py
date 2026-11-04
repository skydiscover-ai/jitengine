"""The load-time checks of the MTP verify pass's two kernels, `gdn_decode.check_multi_v` and
`grouped_expert_qmv.check_grouped_u2`, on MLX's CPU build.

The kernels need Metal (tests/metal/kernels/gdn and moe run the checks on them), so here each
kernel is replaced by a CPU stand-in that computes the same thing row by row, and the checks' own
logic is what is tested: that a stand-in that matches the solo kernel passes, and one that returns
the rows in the wrong order, scales a row, or reads the wrong layer's slots fails. A set the
kernel does not take is refused before anything runs.
"""

# ruff: noqa: E402
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")  # the imports below need MLX, so E402 is waived for the file

from jitengine_qwen3_8_flash_next_m2.kernels import gdn_decode as gdn_fused
from jitengine_qwen3_8_flash_next_m2.kernels import grouped_expert_qmv as moe_kernel

BF16 = mx.bfloat16


def _qmv_multi(x, lins, cfg=(16, 4, 2), swiglu=False):
    """`gdn_decode.qmv_multi` in stock ops: the sets' outputs side by side, SwiGLU taken first."""
    if swiglu:
        k = x.shape[1] // 2
        x = x[:, :k] * mx.sigmoid(x[:, :k]) * x[:, k:]
    return mx.concatenate([lin.stock(x) for lin in lins], axis=1)


def _qmv_multi_v(x, lins, cfg=(16, 4, 2), swiglu=False, vecs=2):
    """The verify launch as `_qmv_multi` row by row (its rows do not depend on each other)."""
    return mx.concatenate([_qmv_multi(x[r : r + 1], lins, cfg, swiglu) for r in range(len(x))])


# (weight sets, launch config, swiglu) as `kernel_checks._glue_calls` lists them, at a small shape
CALLS = [
    ([(32, 64, 4, 32, BF16), (8, 64, 4, 32, mx.float32)], (16, 4, 2), False),
    ([(16, 64, 4, 32, BF16)], (16, 4, 2), False),
    ([(64, 32, 4, 32, BF16)], (16, 4, 2), True),
]


@pytest.fixture
def stand_in_multi(monkeypatch):
    monkeypatch.setattr(gdn_fused, "qmv_multi", _qmv_multi)
    monkeypatch.setattr(gdn_fused, "qmv_multi_v", _qmv_multi_v)


@pytest.mark.usefixtures("stand_in_multi")
def test_check_multi_v_passes_a_launch_whose_rows_are_the_solo_rows():
    assert gdn_fused.check_multi_v(CALLS, mx.float32) is True


@pytest.mark.usefixtures("stand_in_multi")
@pytest.mark.parametrize(
    "wrong",
    [
        lambda out: out[::-1],  # rows in the wrong order
        lambda out: mx.concatenate([out[:1], out[1:] * 1.01]),  # row 1 not the solo row
    ],
    ids=["swapped rows", "scaled row"],
)
def test_check_multi_v_fails_rows_that_differ_from_the_solo_rows(monkeypatch, wrong):
    def launch(*args, **kwargs):
        return wrong(_qmv_multi_v(*args, **kwargs))

    monkeypatch.setattr(gdn_fused, "qmv_multi_v", launch)
    assert gdn_fused.check_multi_v(CALLS, mx.float32) is False


@pytest.mark.usefixtures("stand_in_multi")
def test_check_multi_v_launches_each_call_with_two_rows_and_the_calls_launch(monkeypatch):
    seen = []

    def launch(x, lins, cfg, swiglu, vecs):
        seen.append((tuple(x.shape), len(lins), cfg, swiglu, vecs))
        return _qmv_multi_v(x, lins, cfg, swiglu, vecs)

    monkeypatch.setattr(gdn_fused, "qmv_multi_v", launch)
    gdn_fused.check_multi_v(CALLS, mx.float32)
    assert seen == [
        ((2, 64), 2, (16, 4, 2), False, 2),
        ((2, 64), 1, (16, 4, 2), False, 2),
        ((2, 64), 1, (16, 4, 2), True, 2),
    ]


def test_check_multi_v_refuses_weight_sets_the_kernel_does_not_take():
    with pytest.raises(ValueError, match="does not take the weight sets"):
        gdn_fused.check_multi_v([([(32, 64, 8, 32, BF16)], (16, 4, 2), False)], mx.float32)


# ------------------------------------------------------------------------------ grouped_qmv_u2
def _pair_rows(w, s, b, ext, main, slots, rows, x, group_size):
    """Each pair's row of `x` times its slot's expert (a negative slot reads slot 0, as the kernel
    clamps it), in float32."""
    if ext is None:
        ext, main = (w, s, b), 1 << 30
    out = []
    for slot, row in zip(np.asarray(slots), np.asarray(rows), strict=True):
        slot = max(int(slot), 0)
        mats = (w, s, b) if slot < main else ext
        i = slot if slot < main else slot - main
        deq = mx.dequantize(mats[0][i], mats[1][i], mats[2][i], group_size=group_size, bits=4)
        out.append((x[int(row)].astype(mx.float32) @ deq.astype(mx.float32).T).astype(x.dtype))
    return mx.stack(out)


def _grouped_qmv_x(
    x, w, s, b, ext, main, seg_slot, seg_start, seg_cnt, pair_row, n_pairs, *rest, **kw
):
    return _pair_rows(w, s, b, ext, main, seg_slot, pair_row, x, kw["group_size"])


def _grouped_qmv_u2(x, w, s, b, ext, main, inds2, tab, off, kp, xpair, *rest, group_size, **kw):
    slots = mx.take(tab, inds2 + off[0])
    rows = np.arange(2 * kp) if xpair else np.arange(2 * kp) // kp
    return _pair_rows(w, s, b, ext, main, slots, rows, x, group_size)


@pytest.fixture
def stand_in_grouped(monkeypatch):
    monkeypatch.setattr(moe_kernel, "grouped_qmv_x", _grouped_qmv_x)
    monkeypatch.setattr(moe_kernel, "grouped_qmv_u2", _grouped_qmv_u2)


@pytest.mark.usefixtures("stand_in_grouped")
@pytest.mark.parametrize("top_k", [1, 2, 3, 10])
def test_check_grouped_u2_passes_pairs_that_are_the_solo_pairs(top_k):
    assert moe_kernel.check_grouped_u2(64, 32, 4, 32, mx.float32, top_k) is True


@pytest.mark.parametrize("top_k", [2, 3, 10])
def test_the_two_rows_share_experts_and_leave_some_unresident(top_k):
    route = moe_kernel._Routing2(top_k, seed=21)
    row0, row1 = route.inds[:top_k], route.inds[top_k:]
    assert len(set(row0)) == len(set(row1)) == top_k  # distinct experts per row
    shared = set(row0) & set(row1)
    assert shared and set(row0) - shared and set(row1) - shared
    resident = route.resident
    assert resident[:top_k].any() and resident[top_k:].any() and not resident.all()
    assert all(resident[i] for i, e in enumerate(route.inds) if e in shared)  # a shared one is read
    assert 1 <= route.main < route.slots  # some slots in the extension set


@pytest.mark.usefixtures("stand_in_grouped")
@pytest.mark.parametrize(
    "wrong",
    [
        lambda args, kwargs: _grouped_qmv_u2(*args, **kwargs)[::-1],  # rows in the wrong order
        lambda args, kwargs: _grouped_qmv_u2(*args[:8], args[8] - 1, *args[9:], **kwargs),
    ],
    ids=["swapped rows", "wrong layer offset"],
)
def test_check_grouped_u2_fails_pairs_that_differ_from_the_solo_pairs(monkeypatch, wrong):
    def launch(*args, **kwargs):
        return wrong(args, kwargs)

    monkeypatch.setattr(moe_kernel, "grouped_qmv_u2", launch)
    assert moe_kernel.check_grouped_u2(64, 32, 4, 32, mx.float32, 10) is False


def test_check_grouped_u2_refuses_a_geometry_the_kernel_does_not_take():
    with pytest.raises(ValueError, match="no grouped kernel"):
        moe_kernel.check_grouped_u2(64, 32, 8, 32, mx.float32, 10)
