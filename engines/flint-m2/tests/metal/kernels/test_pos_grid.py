"""The rendered Metal pos_grid against plan.pos_grid for every position below 262,144."""

from jitengine_flint_m2.kernels.mathmode import METAL_COMPILE_OPTIONS
from jitengine_flint_m2.model.config import (
    load_config,
    tensor_index,
)
from jitengine_flint_m2.weight_banks import BankPlan


def test_pos_grid_metal(requires_metal, flint_headers):
    """The kernel's pos_grid (the exact text the megakernel renders) equals plan.pos_grid at every q < 262,144 for every
    policy's T, and the pass-entry / stall / probe / error-marker code is rendered into the persistent M = 1 and
    verify kernels."""
    import re

    import mlx.core as mx
    import numpy as np

    from jitengine_flint_m2.kernels import metal_src as MS
    from jitengine_flint_m2.kernels import plan as P

    fn = re.search(r"inline uint pos_grid\(uint q, uint T\) \{.*?\n\}\n", MS.HEADER2, re.S).group(0)
    k = mx.fast.metal_kernel(
        name="flint_test_pos_grid",
        input_names=["tv"],
        output_names=["out"],
        header=fn,
        source="uint i = thread_position_in_grid.x; out[i] = pos_grid(i, tv[0]);",
        compile_options=METAL_COMPILE_OPTIONS,
    )
    q = np.arange(262144)
    for T in sorted(set(P.SIXD_T.values())):
        got = np.array(
            k(
                inputs=[mx.array([T], dtype=mx.uint32)],
                grid=(262144, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(262144,)],
                output_dtypes=[mx.uint32],
            )[0]
        )
        n = q + 1
        want = 8 * -(-n // np.where(n <= T, P.L6_SPAN, P.L6_SPAN_FINE))
        assert np.array_equal(got, want), T
        assert all(P.pos_grid(int(x), T) == int(want[x]) for x in range(0, 262144, 4093))
    cfg = load_config(flint_headers)
    eo = P.expert_record_offsets(BankPlan(cfg, tensor_index(flint_headers, cfg)))
    for M in (1, 2, 8):
        for kv8 in (False, True):
            h, b = MS.render(cfg, eo, False, mr=M, kv8=kv8)
            assert b.count("if (pass_entry(c, ") == 1 and "P_6CH" not in h and "ch6" not in h
            assert h.count("set_err(c, ") >= 5 and "probe_ring(c, c.prm[P_PROBE])" in h
            bc = MS.render(cfg, eo, True, mr=M, kv8=kv8)[
                1
            ]  # the chain reference: the gate is in the #else branch
            assert (
                bc.index("#if CHAIN")
                < bc.index("#else")
                < bc.index("if (pass_entry(c, ")
                < bc.rindex("#endif")
            )
