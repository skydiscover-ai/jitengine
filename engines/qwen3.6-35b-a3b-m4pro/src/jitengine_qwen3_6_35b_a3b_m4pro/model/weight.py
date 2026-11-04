"""How the Qwen3.6 blocks load their quantized projections from the checkpoint's weights: one by
one (`_qlin`) or several fused into one launch (`_fuse`, `FUSIONS`)."""

from __future__ import annotations

from jitengine_qwen3_6_35b_a3b_m4pro.linear import FusedLinear, QLinear


def _qlin(w: dict, name: str, group_size: int) -> QLinear:
    return QLinear(w[name + ".weight"], w[name + ".scales"], w.get(name + ".biases"), group_size)


def _fuse(w: dict, names: list, group_size: int):
    """A FusedLinear over `names` (popped from `w` to free the originals), else None."""
    parts = [_qlin(w, n, group_size) for n in names]
    if not FusedLinear.compatible(parts):
        return None
    fused = FusedLinear(parts)
    del parts
    for n in names:
        for suffix in (".weight", ".scales", ".biases"):
            w.pop(n + suffix, None)
    return fused


# the same-input projections a block may fuse into one launch (`fuse` below)
FUSIONS = frozenset({"gdn", "attn", "router", "shared", "experts"})
