"""The megakernel's load check on the GPU with the checkpoint (two weight loads, ~25 GB; run alone on the box).

On the tuned M4 Pro an engine with the default options brings the megakernel up at G 40 with no search and passes
every stage: the persistent pass equals its one-dispatch-per-phase chain and a repeat of itself, its top-1 agrees with
stock MLX on the fixed text, and the verify rows equal the one-token passes. A defect injected into the chain stage
(one LOGITS bit) makes profile main fall back to stock MLX ops, with the megakernel and its history freed.
"""

from jitengine_flint_m2.engine.io_struct import GenRequest

STAGES = ("compile", "probe", "chain", "stock", "verify")


def test_every_stage_passes(m4pro_checkpoint):
    from jitengine_flint_m2 import loadcheck as LC
    from jitengine_flint_m2.engine.engine import Engine

    logs = []
    eng = Engine(m4pro_checkpoint, context_limit=4096, prefix_cache=False, log=logs.append)
    try:
        h = eng.health()
        mk = h["mk"]
        lc = mk["loadcheck"]
        assert h["status"] == "ready" and h["decode_backend"] == "mk" and h["multi_token"], h
        assert (mk["G"], mk["G_source"], [t["G"] for t in mk["tried"]]) == (40, "tuned", [40]), mk
        assert all(lc[s]["ok"] for s in STAGES), lc
        assert lc["chain"]["chain_equal"] and lc["chain"]["repeat_equal"], lc["chain"]
        st = lc["stock"]
        assert st["n"] == LC.FORCED_TOKENS and st["agree"] >= LC.STOCK_A, st
        # deterministic on one Mac: measured median d20 of 0.43-0.59 nat on this text
        assert st["d20_median"] < 1.0, st
        assert lc["verify"]["rows"] == list(range(2, 9)) and lc["text"]["version"] == "p0-v1"
        assert lc["s"] < LC.BOUND_S and not [m for m in logs if "WARNING" in m]
        print(
            f"\n[loadcheck] {lc['s']} s; stock agreement {st['agree']:.3f}, median d20 "
            f"{st['d20_median']:.3f} nat"
        )
    finally:
        eng.close()


def test_chain_defect_falls_back_to_stock_mlx(m4pro_checkpoint):
    from jitengine_flint_m2.engine.engine import Engine

    logs = []
    eng = Engine(
        m4pro_checkpoint,
        context_limit=4096,
        prefix_cache=False,
        log=logs.append,
        allow_fault_injection=True,
        sim={"check_mutant": "chain_bitflip"},
    )
    try:
        h = eng.health()
        mu = h["mk_unavailable"]
        assert h["status"] == "ready" and h["decode_backend"] == "graph" and h["mk"] is None, h
        assert mu["stage"] == "chain" and "LOGITS" in mu["reason"] and not h["multi_token"], mu
        assert mu["loadcheck"]["compile"]["ok"] and mu["loadcheck"]["probe"]["ok"]
        warnings = [m for m in logs if "WARNING" in m]
        assert len(warnings) == 1 and "megakernel unavailable on this Mac (chain: " in warnings[0]
        assert eng.mk is None
        prompt = eng.encode("def fibonacci(n):\n")
        r = eng.generate(
            GenRequest(prompt_ids=prompt, max_tokens=8, temperature=0.0, ignore_eos=True)
        )
        assert r.finish_reason == "length" and len(r.token_ids) == 8 and r.error is None, r
    finally:
        eng.close()
