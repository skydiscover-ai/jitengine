"""`doctor --tune`: time every co-resident megakernel grid on this Mac and report the fastest (opt-in).

The grid G (threadgroups of 256 threads) is the one megakernel quantity that differs per GPU. The tuned table
(device.TUNED) holds the M4 Pro's measured optimum; any other device starts from the core-count rule (two threadgroups
per core) and searches down for residency. On the M4 Pro the optimum is sharp (G +- 4 costs 6-23% at M = 1) and G = 40
matches the kernel's 40-way splits, so the rule's grid is not known to be the fastest elsewhere. This module measures
instead: on the device at hand, every grid of the domain (16..64, step 4) is built and proved co-resident by the
engine's own residency probe (runner.MegaKernel.probe, on a small separate history), and every co-resident grid is timed
on the M = 1 pass (the raw decode pass, PM1) in bursts of T dependent passes at fixed positions of a history prefilled
with stock MLX ops from the load-check text, over several rounds with the grid order rotated. The score is the
geometric mean over positions of the median ms per pass; the lowest wins.

The caller (doctor.py) stores the result with device.write_grid_cache; device.plan_grid tries it first on a device
outside the tuned table. Tokens, logits and top-20 ids do not depend on G (measured on the M4 Pro: only the head's
log-sum-exp does, by <= 2.8e-6 nat), so a tuned grid changes speed, not outputs.
"""

from __future__ import annotations

import math
import statistics
import time

POSITIONS = (1536, 4608)  # where the M = 1 pass is timed (1K and 4K of context, plus 512)
T = 16  # dependent passes per burst
ROUNDS = 3  # rounds, the grid order rotated each round


def _prefilled(W, hcls, ids: list, n: int, kv8: bool):
    """A history of n + 64 positions with the first n ids prefilled by stock MLX ops."""
    import mlx.core as mx

    from jitengine_flint_m4pro.model.model import prefill_chunk

    hist = hcls(W.cfg, n + 64)
    hist.reserve(n + 16)
    for c0 in range(0, n, 2048):
        rows = [min(n, c0 + 2048) - c0 - 1]
        lg = prefill_chunk(W, hist, mx.array(ids[c0 : c0 + 2048], dtype=mx.int32), c0, rows)
        mx.eval(lg, hist.state_arrays())
        del lg
        if kv8:
            mx.clear_cache()
    return hist


def _co_resident(W, kv8: bool, grids: list, att: dict, log) -> tuple:
    """({G: kernel} for every grid the residency probe admits, {G: reason} for the others)."""
    import mlx.core as mx

    from jitengine_flint_m4pro.kernels import runner as RN
    from jitengine_flint_m4pro.kv_cache import MKHistory, MKHistory8

    small = (MKHistory8 if kv8 else MKHistory)(W.cfg, 64)
    small.reserve(8)
    kern, refused = {}, {}
    for G in grids:
        mk = RN.MegaKernel(W, G=G, log=lambda m: None, kv8=kv8)
        try:
            mk.probe(small.arrays, att, small.capacity)
            kern[G] = mk
        except RN.BarrierProbeError as exc:
            refused[G] = str(exc)[:200]
            del mk
            mx.clear_cache()
    small.release()
    if not kern:
        raise RuntimeError(f"no grid of {grids} ran co-resident on this GPU")
    log(f"[tune] co-resident grids {sorted(kern)}; refused {sorted(refused)}")
    return kern, refused


def tune(
    W,
    kv8: bool,
    tokenizer,
    log=print,
    grids=None,
    positions=POSITIONS,
    rounds: int = ROUNDS,
    t_burst: int = T,
) -> dict:
    """Time the co-resident grids. Returns {"G" (fastest), "ms" {G: score}, "per_position" {G: {pos: median ms}},
    "refused" {G: reason}, "positions", "T", "rounds", "s"}."""
    import mlx.core as mx

    from jitengine_flint_m4pro import loadcheck
    from jitengine_flint_m4pro.device import G_MAX, G_MIN, G_STEP
    from jitengine_flint_m4pro.kernels import plan as P
    from jitengine_flint_m4pro.kv_cache import MKHistory, MKHistory8

    t_start = time.perf_counter()
    cfg = W.cfg
    grids = list(grids) if grids is not None else list(range(G_MIN, G_MAX + 1, G_STEP))
    att = P.attn_params(P.default_policy(8 if kv8 else 16))
    n = max(positions)
    ids = loadcheck.text_ids(tokenizer, n)
    hist = _prefilled(W, MKHistory8 if kv8 else MKHistory, ids, n, kv8)
    kern, refused = _co_resident(W, kv8, grids, att, log)
    tok = ids[n // 2]

    def burst(mk, pos):
        prm = P.make_params(
            pos=pos,
            token=tok,
            forced_next=None,
            sel_mode=0,
            top_k=1,
            top_p=1.0,
            temperature=0.0,
            u=0.0,
            poison=False,
            cap=hist.capacity,
            theta=cfg.rope_theta,
            **att,
        )
        t0 = time.perf_counter()
        prev = None
        for _ in range(t_burst):
            prev = mk.run_pass(hist.arrays, prm, prev)
        mx.eval(prev)
        ms = (time.perf_counter() - t0) * 1e3 / t_burst
        err = int(prev[P.R_ERR].item())
        if err:
            raise RuntimeError(f"G={mk.G}: timed pass failed in the kernel: {P.describe_err(err)}")
        return ms

    arms = sorted(kern)
    for G in arms:  # warm every arm at every position once
        for pos in positions:
            burst(kern[G], pos)
    times = {G: {pos: [] for pos in positions} for G in arms}
    for r in range(rounds):
        order = arms[r % len(arms) :] + arms[: r % len(arms)]
        if r % 2:
            order = order[::-1]
        for pos in positions if r % 2 == 0 else positions[::-1]:
            for G in order:
                times[G][pos].append(burst(kern[G], pos))
    per = {G: {pos: round(statistics.median(v), 4) for pos, v in times[G].items()} for G in arms}
    score = {
        G: round(math.exp(sum(math.log(per[G][p]) for p in positions) / len(positions)), 4)
        for G in arms
    }
    best = min(arms, key=lambda g: (score[g], g))
    del kern
    hist.release()
    mx.clear_cache()
    log(
        f"[tune] fastest G {best}: "
        + ", ".join(f"G{g} {score[g]:.3f}" for g in arms)
        + " ms per M = 1 pass"
    )
    return {
        "G": best,
        "ms": {str(g): score[g] for g in arms},
        "per_position": {str(g): per[g] for g in arms},
        "refused": {str(g): v for g, v in refused.items()},
        "positions": list(positions),
        "T": t_burst,
        "rounds": rounds,
        "s": round(time.perf_counter() - t_start, 1),
    }
