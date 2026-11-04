"""The verify pass's MoE block per row count (kernels/vmoe_src.py), and HEADER_V as the single source
of every verify render."""

import pytest

from jitengine_flint_m2.kernels import metal_src as MS
from jitengine_flint_m2.kernels import plan as P
from jitengine_flint_m2.kernels import vmoe_src as V
from jitengine_flint_m2.model.config import load_config, tensor_index
from jitengine_flint_m2.weight_banks import BankPlan


def _cfg_eo(flint_headers):
    cfg = load_config(flint_headers)
    return cfg, P.expert_record_offsets(BankPlan(cfg, tensor_index(flint_headers, cfg)))


def test_verify_moe_variants(flint_headers):
    """Each verify render's MoE block is HEADER_V with vmoe_src's anchored edits: the snake DOWN deal
    at every M, S1s (vmoe_s1) from S1S_MIN_ROWS rows, in both profiles; M = 1 renders carry neither.
    An anchor that is missing or repeated fails loudly."""
    cfg, eo = _cfg_eo(flint_headers)
    snake_deal = "for (uint p = c.G - 1u - c.w; p < np; p += c.G) {"
    fused = "inline bool vmoe_fused(thread Ctx& c, thread VX& vx, uint l) {"
    assert 2 < V.S1S_MIN_ROWS <= P.VMAX
    for M in range(1, P.VMAX + 1):
        for kv8 in (False, True):
            h, _ = MS.render(cfg, eo, False, mr=M, kv8=kv8)
            s1 = M >= V.S1S_MIN_ROWS
            assert h.count(snake_deal) == (M > 1) and h.count(fused) == (M > 1)
            assert h.count("return vmoe_s1(c, vx, l, dr, dg, dd);") == s1
            assert h.count("inline void vgu_piece_s1(") == s1
    for edit in (V.snake, V.s1s):
        with pytest.raises(AssertionError):
            edit("no anchors here")
        with pytest.raises(AssertionError):
            edit(MS.HEADER_V + MS.HEADER_V)


def test_header_v_live(flint_headers, monkeypatch):
    """A marker planted in metal_src.HEADER_V reaches every verify render, M = 2 .. VMAX of both
    profiles: the verify headers are derived from HEADER_V's current text at render time, never from
    a copy of it."""
    cfg, eo = _cfg_eo(flint_headers)
    mark = "// PROBE-HEADER_V-LIVE-7f3a\n"
    monkeypatch.setattr(MS, "HEADER_V", mark + MS.HEADER_V)
    keys = [(kv8, M) for kv8 in (False, True) for M in range(2, P.VMAX + 1)]
    live = [k for k in keys if mark in MS.render(cfg, eo, False, mr=k[1], kv8=k[0])[0]]
    assert len(live) == len(keys) == 2 * (P.VMAX - 1), (
        f"{len(live)}/{len(keys)} verify renders carry a HEADER_V edit"
    )
