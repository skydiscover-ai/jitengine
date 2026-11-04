"""The M = 1 render's phase-mask specialization, and the host's refusals on it."""

from jitengine_flint_m4pro.model.config import (
    load_config,
    tensor_index,
)
from jitengine_flint_m4pro.weight_banks import BankPlan


def expect_raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return
    raise AssertionError(f"{fn.__name__}{a} did not raise {exc.__name__}")


def test_pm1_render(flint_headers):
    """The persistent M = 1 render's default (pmask 0, PM1) has no per-phase mask
    test and runs a mask-0 pass as the pass's grid barriers alone; pmask 1 (the general kernel) keeps them, and every
    chain, verify and diag render is the general text whatever pmask says (pmask 0 is refused there). The host refuses
    a partial mask or a timing ablation on a PM1 kernel before any dispatch."""
    from jitengine_flint_m4pro.kernels import metal_src as MS
    from jitengine_flint_m4pro.kernels import plan as P
    from jitengine_flint_m4pro.kernels import runner as RN

    cfg = load_config(flint_headers)
    eo = P.expert_record_offsets(BankPlan(cfg, tensor_index(flint_headers, cfg)))
    for kv8 in (False, True):
        h0, b0 = MS.render(cfg, eo, False, kv8=kv8)
        h1, b1 = MS.render(cfg, eo, False, kv8=kv8, pmask=1)
        assert (h0, b0) == MS.render(cfg, eo, False, kv8=kv8, pmask=0) and (h0, b0) != (h1, b1)
        assert (
            "PK(c, P_MASK) >>" not in h0 + b0
            and "MOE_BITS) !=" not in b0
            and "if (PK(c, P_MASK) == 0u) {" in b0
        )
        assert (
            "PK(c, P_MASK) >> ph" in h1
            and "MOE_BITS) !=" in b1
            and "if (PK(c, P_MASK) == 0u) {" not in b1
        )
        for chain, M in ((True, 1), (False, 2), (False, 4), (True, 4)):
            assert MS.render(cfg, eo, chain, mr=M, kv8=kv8) == MS.render(
                cfg, eo, chain, mr=M, kv8=kv8, pmask=1
            )
            expect_raises(AssertionError, MS.render, cfg, eo, chain, mr=M, kv8=kv8, pmask=0)
    assert MS.render(cfg, eo, False, diag=1) == MS.render(cfg, eo, False, diag=1, pmask=1)
    expect_raises(AssertionError, MS.render, cfg, eo, False, diag=1, pmask=0)
    k = object.__new__(RN.MegaKernel)
    prm = P.make_params(
        pos=600,
        token=None,
        forced_next=None,
        sel_mode=0,
        top_k=1,
        top_p=1.0,
        temperature=0.0,
        u=0.0,
        poison=False,
        s_f=32,
        s_s=16,
        cap=4096,
        theta=1e4,
    )
    partial = (
        P.MASK_ALL & ~(1 << P.PH["ATT"]),
        1 << P.PH["QKV"],
        P.MASK_ALL | (8 << 16),
        P.MASK_ALL | (1 << 16),
    )
    for pm, mask in [(0, P.MASK_ALL), (0, 0), (1, P.MASK_ALL), (1, 0)] + [(1, m) for m in partial]:
        k.pmask, prm[P.P_MASK] = pm, mask
        k._check_mask(prm)
    for mask in partial:
        k.pmask, prm[P.P_MASK] = 0, mask
        expect_raises(ValueError, k._check_mask, prm)
