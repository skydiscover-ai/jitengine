"""`jitengine-flint-m2 doctor --model-path DIR [--json [FILE]]`: what this Mac runs, as one report.

The evidence format for a Mac other than the tuned M4 Pro. It loads the engine with the options `serve` takes and lets
it run what a server start runs: admission by capability (device.py), the context limit for this GPU's wired limit
(config.resolve_context_limit), and the megakernel load check (loadcheck.py) with its fallback. Then it decodes a few
greedy tokens from a fixed prompt and reports the device facts, the admission notes, the context limit and its source,
the decode backend (and why the megakernel is unavailable, if it is), the grid and how it was chosen, the verify passes,
load time, toolchain versions and the smoke decode.

With --tune (opt-in), before the engine loads, every co-resident megakernel grid is timed on this Mac (tune.py) and the
fastest is stored in the grid cache (device.write_grid_cache; FLINT_GRID_CACHE or ~/.cache/flint-engine/grid-v1.json)
under this device and kernel source. Engines started afterwards on a device outside the tuned table try that grid
first; on the tuned M4 Pro the tuned table still wins.

Exit status:
  0  the megakernel serves, with speculation (verify passes compiled and probed, verify rows exact);
  1  the engine serves, but with stock MLX ops (the graph backend) or without speculation;
  2  the engine refused (before or during its load), failed the smoke decode, or --tune failed.

`jitengine-flint-m2 inspect --model-path DIR [--json [FILE]]` is the same report up to the load: what `serve` with
these options would plan on this Mac (admission, the memory plan, the context limit, the decode backend and grid it
would try), from the checkpoint headers and the device facts, without reading a weight or compiling a kernel. Exit 0:
serve would start; 2: it would refuse, with the reason.
"""

from __future__ import annotations

import json
import sys
import time

from jitengine_flint_m2 import device
from jitengine_flint_m2.budget import flag_budget
from jitengine_flint_m2.engine.io_struct import EngineUnavailable, GenRequest
from jitengine_flint_m2.model.config import StartupError

SMOKE_PROMPT = 'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n'
REPORT_VERSION = 1


def _verdict(rep: dict, verdict: str, code: int, reason: str | None = None) -> int:
    rep.update(verdict=verdict, exit=code)
    if reason:
        rep["reason"] = reason
    return code


def _smoke(eng, n: int) -> dict:
    ids = eng.encode(SMOKE_PROMPT)
    t0 = time.perf_counter()
    r = eng.generate(
        GenRequest(prompt_ids=ids, max_tokens=n, temperature=0.0, ignore_eos=True, seed=0)
    )
    return {
        "prompt_tokens": len(ids),
        "token_ids": list(r.token_ids),
        "text": eng.decode(r.token_ids),
        "finish_reason": r.finish_reason,
        "error": r.error,
        "wall_s": round(time.perf_counter() - t0, 3),
        "decode_tok_s": (r.metrics or {}).get("decode_tok_s"),
    }


def _tune(a, sim, log) -> dict:
    """Time the co-resident grids on this Mac with the weights loaded once (then freed), and cache the fastest under
    the device the engine will see (the simulated one, with --simulate)."""
    import gc

    import mlx.core as mx

    from jitengine_flint_m2 import tune
    from jitengine_flint_m2.engine.engine import toolchain_versions
    from jitengine_flint_m2.loader import load_weights
    from jitengine_flint_m2.model.config import KV_BITS, load_config, tensor_index
    from jitengine_flint_m2.tokenizer import Tokenizer

    facts = device.read_facts(device.parse_sim(sim, allow_fault_injection=sim is not None))
    kv_bits = KV_BITS[a.profile]
    info = device.mlx_device_info()
    mx.set_wired_limit(int(info["max_recommended_working_set_size"]))
    cfg = load_config(a.model)
    W = load_weights(cfg, tensor_index(a.model, cfg), log=lambda m: None)
    try:
        res = tune.tune(W, kv_bits == 8, Tokenizer(a.model), log=log)
    finally:
        W = None
        gc.collect()
        mx.clear_cache()
    versions = toolchain_versions()
    entry = {"G": res["G"], "ms": res["ms"], "refused": sorted(res["refused"])}
    res["cache"] = device.write_grid_cache(facts, versions, kv_bits, entry)
    res["key"] = device.grid_cache_key(facts, versions, kv_bits)
    res["tuned_table_wins"] = facts.tuned_grid is not None
    return res


def _served(rep: dict, smoke_tokens: int) -> int:
    """The verdict on an engine that loaded (rep["health"]) and its smoke decode."""
    sm = rep.get("smoke")
    if sm is not None and (sm["error"] or len(sm["token_ids"]) != smoke_tokens):
        return _verdict(
            rep, "failed", 2, f"smoke decode failed: {sm['error'] or sm['finish_reason']}"
        )
    h = rep["health"]
    mem = h.get("memory") or {}
    if mem.get("mode") == "offload":
        why = (
            "--mode offload"
            if h.get("memory_mode_requested") == "offload"
            else "the whole model does not fit the memory budget or the GPU wired limit"
        )
        return _verdict(
            rep,
            "offload",
            1,
            f"{why}: {mem['pool_slots']:,} of {mem['total_records']:,} experts in memory, the rest read from the "
            f"checkpoint, decoding with stock MLX ops",
        )
    if h["decode_backend"] != "mk":
        mu = h.get("mk_unavailable") or {}
        return _verdict(
            rep,
            "graph",
            1,
            f"megakernel unavailable ({mu.get('stage')}: {mu.get('reason')}); decoding with stock MLX ops",
        )
    if not h["multi_token"]:
        vf = ((h.get("mk") or {}).get("loadcheck") or {}).get("verify") or {}
        why = (
            f"the load check's verify rows differ ({vf.get('reason')})"
            if vf.get("ok") is False
            else "the verify passes were refused on this GPU"
        )
        return _verdict(rep, "mk", 1, f"speculation off: {why}")
    return _verdict(rep, "mk", 0)


def _report(kind: str, a) -> dict:
    """The report's head: what it is, the checkpoint, and the options it was asked with."""
    return {
        kind: REPORT_VERSION,
        "model": a.model,
        "profile": a.profile,
        "requested": {
            "context_limit": a.context_limit,
            "memory_budget_gib": getattr(a, "memory_budget_gib", None),
            "memory_budget_gb": getattr(a, "memory_budget_gb", None),
            "mode": getattr(a, "mode", "auto"),
            "decode_backend": a.decode_backend,
            "mk_grid": a.mk_grid,
            "simulated": a.simulate,
        },
    }


def _engine_options(a, rep: dict, log) -> dict:
    """The Engine options `serve` would pass for these arguments; records the device facts in the report. Raises
    StartupError or ValueError for a simulation or a budget that is refused."""
    sim = a.simulate
    parsed = device.parse_sim(sim, allow_fault_injection=sim is not None)
    rep["device"] = device.read_facts(parsed).summary()
    budget = flag_budget(
        getattr(a, "memory_budget_gib", None), getattr(a, "memory_budget_gb", None)
    )
    return dict(
        profile=a.profile,
        context_limit=a.context_limit,
        decode_backend=a.decode_backend,
        mk_grid=a.mk_grid,
        sim=sim,
        allow_fault_injection=sim is not None,
        log=log,
        prefix_cache=False,  # one smoke request: nothing to reuse
        memory_budget_bytes=budget,
        memory_mode=getattr(a, "mode", "auto"),
    )


def inspect_plan(a, log) -> tuple:
    """(exit status, report) for parsed `inspect` arguments: the engine is built up to its load (every refusal
    `serve` makes before a weight) and its plan reported; nothing is loaded, compiled or decoded."""
    from jitengine_flint_m2.engine.engine import Engine

    rep = _report("inspect", a)
    try:
        eng = Engine(a.model, start=False, **_engine_options(a, rep, log))
    except (StartupError, OSError, ValueError) as exc:
        return _verdict(rep, "refused", 2, str(exc)), rep
    grid = eng.grid
    rep.update(
        notes=list(eng.device_notes),
        memory=eng.memory.to_dict(),
        memory_line=eng.memory.line(),
        context_limit=eng.context_limit,
        context_limit_source=eng.context_limit_source,
        decode_backend=eng.decode_backend,
        speculation=eng.speculation,
        grid=None
        if grid is None
        else {"G0": grid.G0, "source": grid.source, "reason": grid.reason},
    )
    return _verdict(rep, eng.memory.mode, 0), rep


def diagnose(a, log) -> tuple:
    """(exit status, report) for parsed `doctor` arguments; log(line) receives the engine's log."""
    from jitengine_flint_m2.engine.engine import Engine

    sim = a.simulate
    rep = _report("doctor", a)
    t0 = time.perf_counter()
    try:
        opts = _engine_options(a, rep, log)
    except (StartupError, ValueError) as exc:
        return _verdict(rep, "refused", 2, str(exc)), rep
    if getattr(a, "tune", False):
        try:
            Engine(a.model, start=False, **opts)  # every refusal before weights, before tuning
            rep["tune"] = _tune(a, sim, log)
        except (StartupError, OSError, ValueError) as exc:
            rep["wall_s"] = round(time.perf_counter() - t0, 2)
            return _verdict(rep, "refused", 2, f"refused before loading weights: {exc}"), rep
        except Exception as exc:  # noqa: BLE001  (reported: doctor exits 2)
            rep["wall_s"] = round(time.perf_counter() - t0, 2)
            return _verdict(rep, "failed", 2, f"--tune failed: {type(exc).__name__}: {exc}"), rep
    try:
        eng = Engine(a.model, wait=False, **opts)
    except (StartupError, OSError, ValueError) as exc:
        rep["wall_s"] = round(time.perf_counter() - t0, 2)
        return _verdict(rep, "refused", 2, f"refused before loading weights: {exc}"), rep
    try:
        rep["notes"] = list(eng.device_notes)
        try:
            eng.wait_ready()
        except EngineUnavailable as exc:
            rep["health"] = eng.health()
            return _verdict(rep, "refused", 2, str(exc)), rep
        if a.smoke_tokens > 0:
            rep["smoke"] = _smoke(eng, a.smoke_tokens)
        rep["health"] = eng.health()
    finally:
        eng.close()
        rep["wall_s"] = round(time.perf_counter() - t0, 2)
    return _served(rep, a.smoke_tokens), rep


def _head(rep: dict, kind: str) -> list:
    """The summary's first lines: the verdict and the device."""
    d = rep.get("device") or {}
    why = f": {rep['reason']}" if rep.get("reason") else ""
    return [
        f"flint {kind}: {rep['verdict']} (exit {rep['exit']}){why}",
        f"  device   {d.get('device_name')} {d.get('architecture')}, {d.get('gpu_cores')} GPU cores, "
        f"memory {d.get('memory_size', 0) / 1e9:.1f} GB, wired limit {d.get('wired_bytes', 0) / 1e9:.1f} GB"
        f"{', tuned' if d.get('tuned') else ''}",
    ]


def inspect_summary(rep: dict) -> str:
    lines = _head(rep, "inspect")
    if rep["exit"] == 0:
        g = rep["grid"]
        grid = f"G0 {g['G0']} ({g['source']})" if g else "none"
        lines += [
            f"  memory   {rep['memory_line']}",
            f"  engine   backend {rep['decode_backend']}, grid {grid}, context {rep['context_limit']:,} "
            f"({rep['context_limit_source']}), speculation {rep['speculation']}",
        ]
    return "\n".join(lines + [f"  note     {n}" for n in rep.get("notes", [])])


def summary(rep: dict) -> str:
    h = rep.get("health") or {}
    lines = _head(rep, "doctor")
    if h:
        mk = h.get("mk") or {}
        grid = f"G {mk.get('G')} ({mk.get('G_source')})" if mk else "none"
        lines.append(
            f"  engine   backend {h.get('decode_backend')}, grid {grid}, context "
            f"{h.get('context_limit')} ({h.get('context_limit_source')}), speculation {h.get('multi_token')}, "
            f"load {h.get('load_s')} s"
        )
        mem = h.get("memory") or {}
        if mem:
            lines.append(
                f"  memory   {mem['mode']}, planned {mem['planned_bytes'] / 2**30:.2f} GiB"
                + (
                    f" of a {mem['budget_bytes'] / 2**30:.2f} GiB budget ({mem['budget_source']})"
                    if mem.get("budget_bytes")
                    else ""
                )
                + (
                    f", {mem['pool_slots']:,} of {mem['total_records']:,} experts in memory"
                    if mem["mode"] == "offload"
                    else ""
                )
            )
        lc = mk.get("loadcheck") or (h.get("mk_unavailable") or {}).get("loadcheck")
        if lc:
            st = lc.get("stock") or {}
            stages = ", ".join(
                f"{s} {'ok' if (lc.get(s) or {}).get('ok') else 'FAILED'}"
                for s in ("compile", "probe", "chain", "stock", "verify")
                if s in lc
            )
            agreement = (
                f"; stock MLX agreement {st['agree']:.3f}, median d20 {st['d20_median']:.3f} nat"
                if "agree" in st
                else ""
            )
            lines.append(f"  check    {lc.get('s')} s: {stages}{agreement}")
    if rep.get("tune"):
        t = rep["tune"]
        wins = "; the tuned table wins on this device" if t.get("tuned_table_wins") else ""
        lines.append(
            f"  tune     fastest G {t['G']} of {sorted(int(g) for g in t['ms'])} ({t['s']} s), cached in "
            f"{t['cache']}{wins}"
        )
    if rep.get("smoke"):
        s = rep["smoke"]
        lines.append(
            f"  smoke    {len(s['token_ids'])} tokens, {s['decode_tok_s']} tok/s: {s['text']!r}"
        )
    return "\n".join(lines)


def _log(m):
    print(m, file=sys.stderr, flush=True)


def _emit(a, rep: dict, text: str) -> None:
    """The report as JSON to --json's FILE or stdout, and the summary beside it."""
    report = json.dumps(rep, indent=1, default=str)
    if a.json == "-":
        print(report, flush=True)
    elif a.json:
        with open(a.json, "w") as f:
            f.write(report + "\n")
    print(text, file=sys.stderr if a.json == "-" else sys.stdout, flush=True)


def run(a) -> int:
    code, rep = diagnose(a, _log)
    _emit(a, rep, summary(rep))
    return code


def run_inspect(a) -> int:
    code, rep = inspect_plan(
        a, lambda m: None
    )  # the plan is the report; the engine's lines repeat it
    _emit(a, rep, inspect_summary(rep))
    return code
