"""MoE routing and combine, in the vendor-export arithmetic the reference implementation names.

Router: logits = x @ W_router^T with bf16 inputs (fp32 accumulation, rounded to bf16), scores =
sigmoid(fp32(logits)). Top-6 by score; an exact tie is broken toward the LOWER expert id and the six
are ranked (score desc, id asc) -- the rule of the routing in rec, Recursive's own inference engine for Flint, and of mlx-lm's kernel,
which the trusted reference declares.
The selection key packs the fp32 score bits above (511 - id) in one uint64, so the order is total and
deterministic on the GPU. Weights: S = ((((p0+p1)+p2)+p3)+p4)+p5 + 1e-20 in fp32, rank order;
w_r = bf16(p_r / S).

Combine: each expert output (bf16) times its bf16 weight, rounded to bf16, summed in rank order (best
score first) with a bf16 rounding after every addition, then the shared expert added in bf16.
"""

from __future__ import annotations

TIE_TOWARD_HIGHER_ID = False
NORM_EPS = 1e-20


def route(x, router_w, top_k: int):
    """x [..., hidden] bf16 -> (ids uint32 [..., k] best first, weights bf16 [..., k])."""
    import mlx.core as mx

    logits = mx.matmul(x, router_w.T)  # bf16 out, fp32 accumulate
    scores = mx.sigmoid(logits.astype(mx.float32))
    n = scores.shape[-1]
    bits = scores.view(mx.uint32).astype(mx.uint64)
    ids = mx.arange(n, dtype=mx.uint64)
    if not TIE_TOWARD_HIGHER_ID:
        ids = mx.array(511, dtype=mx.uint64) - ids
    key = mx.left_shift(bits, mx.array(9, dtype=mx.uint64)) | ids
    top = mx.sort(mx.topk(key, top_k, axis=-1), axis=-1)[..., ::-1]  # best first
    low = top & mx.array(511, dtype=mx.uint64)
    if not TIE_TOWARD_HIGHER_ID:
        low = mx.array(511, dtype=mx.uint64) - low
    sel = low.astype(mx.uint32)
    s = mx.right_shift(top, mx.array(9, dtype=mx.uint64)).astype(mx.uint32).view(mx.float32)
    total = s[..., 0:1]
    for j in range(1, top_k):
        total = total + s[..., j : j + 1]  # sequential, rank order
    w = s / (total + NORM_EPS)
    return sel, w.astype(mx.bfloat16)


def combine(expert_out, weights):
    """expert_out [..., k, hidden] bf16, weights [..., k] bf16 -> [..., hidden] bf16 (rank order)."""
    import mlx.core as mx

    prod = expert_out * mx.expand_dims(weights, -1)  # bf16 * bf16 -> rounded to bf16
    acc = prod[..., 0, :]
    for j in range(1, prod.shape[-2]):
        acc = acc + prod[..., j, :]  # bf16 rounding after every addition
    return acc
