"""State that several Flash-Next blocks and model.py share: the speculative layer build's flag
(`_IN_SPEC_BUILD`) and dispatch hold (`_DEFER`, `_dispatch`), and the launch config
(`GDN_FUSED_MULTI`) of the multi-weight projections in the GDN block and the MoE shared expert.
"""

from __future__ import annotations

import mlx.core as mx

# The GDN block's solo-decode glue as three launches around the recurrence kernel and out_proj.
GDN_FUSED_MULTI = (16, 4, 2)  # measured (KL, NSG, RB) of the multi-weight projection launches


_IN_SPEC_BUILD = [False]  # set by Qwen4ExpTextDecoderLayer.build_spec (model thread only)


# After a miss, build the corrected combine and the next layer while the read is in flight (their
# graphs depend on the landing slots, not the bytes), holding dispatches in `_DEFER` until then.
_DEFER: list = [None]


def _dispatch(*arrs) -> None:
    """`mx.async_eval`, or held back in order while a prebuild is open."""
    if _DEFER[0] is not None:
        _DEFER[0].append(arrs)
    else:
        mx.async_eval(*arrs)
