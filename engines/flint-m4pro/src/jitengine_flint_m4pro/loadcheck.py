"""Load-time check of the decode megakernel on this GPU: five stages, each a fallback point.

A failure raises MKUnavailable(stage, reason, tried, report) and the engine decides what it means
(Engine._bring_up_mk): decode_backend "auto" decodes profile main with stock MLX ops (the graph backend), profile kv8 is
refused (it has no graph form), and decode_backend "mk" or a pinned grid raises. The verify stage is the exception: a
failure there turns speculation off and the megakernel keeps serving one token per pass.

  compile  the profile's persistent M = 1 kernel (PM1) builds and completes one minimal pass: the residency probe's
           own first dispatch (a one-lap flag ring, no phase). Its text does not depend on G (the kernel reads
           threadgroups_per_grid), so it is built once, at the first candidate grid.
  probe    the residency probe (runner.MegaKernel.probe: chained barrier-only passes at a short spin bound, the flag
           ring, real passes) at each candidate grid in order (device.GridPlan: the tuned grid alone, a pinned grid
           alone, a `doctor --tune` cache entry first, or G0 down to 16 in steps of 4), each on a fresh kernel object
           and scratch, until one passes. A BarrierProbeError moves on to the next grid (a refused grid costs about
           0.1 s on the M4 Pro); any other error ends the stage.
  chain    the persistent pass against its race-free reference, the same pass as one dispatch per phase
           (runner.run_pass_chain; its render is compiled here, its only user in the engine): CHAIN_PASSES chained
           passes on a history prefilled with the first PROMPT_TOKENS of a fixed text (loadcheck_text), every result
           word, the LOGITS and the residual trace HS of every pass bit-identical; then a second persistent run,
           bit-identical to the first (repeat determinism). Each run starts from the same state: the sliding rings
           restored from a copy and a zeroed scratch (the full-layer rows past the prompt are rewritten by each pass
           before any later pass reads them). Catches races and fences that do not hold on this GPU.
  stock    agreement with stock MLX (the graph backend, model/model.py) on the fixed text: FORCED_TOKENS megakernel
           passes fed the text's next tokens after the PROMPT_TOKENS prefill, against stock MLX's echo of the whole
           text at the same positions. Per position, d20 = max over the reference's top-20 ids of |lp_mk - lp_ref|
           (an id missing from the kernel's top-20 is bounded by its 20th log-prob) and agreement = kernel argmax == reference argmax. Passes iff agreement >= STOCK_A (and the
           median d20 <= STOCK_D once a bound is calibrated). The kernel is not bit-identical to stock MLX (neither
           is mx.quantized_matmul correctly rounded), so this check is distributional. Measured on the M4 Pro: the
           real kernel sits at median d20 0.43-0.86 nat on three natural texts (main and kv8), and the three subtle
           mutants below at 1.55-4.13, every mutant above every legitimate run but short of a 3x margin, and the
           real kernel's d20 depends on the text. So STOCK_D is withheld (None: the median d20 is reported, not
           gated), and STOCK_A = 0.6875 (22 of 32) is a coarse gate that refuses gross numeric failure (garbage
           logits, a broken head or layer stack), not the subtle mutants (their agreement is 0.72-0.94).
  verify   one verify pass of M rows per compiled row count (2..8) at the text's positions: every row's 64 result words
           equal the M = 1 pass's at that position (the stock stage's passes). Else speculation off.

The whole check is bounded (BOUND_S from the compile stage on; a stage that ends over it fails). It leaves the history
invalid, the kernel's descriptor table as built, and a zeroed scratch.

Simulation (device.Sim, tests only). fail_compile fails the compile stage before the real build; fail_probe_G fails the
probe at the listed grids (or "all") without running it, as a BarrierProbeError would. check_mutant injects one defect
through the real kernel (no Metal text change) into every pass of the stages after the probe:
  wrong_expert   one routed layer's descriptor expert base points at the next routed layer's experts  -> stock (d20)
  pos_shift      the megakernel passes run at p + 1                                                  -> stock (d20)
  stale_history  the last STALE_ROWS full-layer rows of the prompt are zeroed before the passes       -> stock (d20)
  chain_bitflip  one LOGITS bit of a persistent pass is flipped in the scratch                       -> chain
  verify_row     the last row of the largest verify pass is fed another token                        -> verify
On the real kernel the three stock mutants raise the reported median d20 (1.55-4.13 nat against 0.43-0.86) but pass
the agreement gate: with STOCK_D withheld they are reported, not refused. The kernel's descriptor table is restored
after the check, so a mutant never reaches a served request.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from jitengine_flint_m4pro.kernels import plan as P
from jitengine_flint_m4pro.kernels import runner as RN
from jitengine_flint_m4pro.loadcheck_text import P0_TEXT, P0_VERSION

BOUND_S = 120.0  # the whole check, compile stage on; about 2.7 s on the M4 Pro
CHAIN_PASSES = 8  # chained persistent passes compared with the chain render (and repeated)
PROMPT_TOKENS = 512  # the fixed text's prefix, prefilled with stock MLX ops
FORCED_TOKENS = 32  # the text's next tokens, one megakernel pass each (stock stage)
STOCK_A = 0.6875  # stock stage: top-1 agreement at least this (22 of 32) ...
STOCK_D = None  # ... and median d20 (nat) at most this: withheld, reported only
TOPN_REF = 20  # d20 is taken over the reference's top-20
STALE_ROWS = 64  # mutant stale_history: full-layer rows zeroed at the end of the prompt
FLIP_PASS = 4  # mutant chain_bitflip: the persistent pass whose LOGITS bit is flipped
# a context limit too small for this prefix + FORCED_TOKENS + VMAX skips chain / stock / verify
MIN_PROMPT = 64
_SIZE = dict(P._SIZES)  # scratch region bytes (HS: the residual trace, LOGITS: the bf16 logits)


class MKUnavailable(RuntimeError):
    """A load-check stage failed: the megakernel cannot serve on this GPU."""

    def __init__(self, stage: str, reason: str, tried: list, report: dict | None = None):
        super().__init__(f"{stage}: {reason}")
        self.stage, self.reason, self.tried = stage, reason, list(tried)
        self.report = report if report is not None else {}


# ------------------------------------------------------------------------------------------------ compile + probe
def _first_pass_params(mk, att: dict, cap: int):
    return P.make_params(
        pos=0,
        token=0,
        forced_next=None,
        sel_mode=0,
        top_k=1,
        top_p=1.0,
        temperature=0.0,
        u=0.0,
        poison=False,
        cap=cap,
        theta=mk.cfg.rope_theta,
        mask=0,
        spin=P.SPIN_PROBE,
        probe=1,
        **att,
    )


def compile_stage(mk, hist, att: dict, sim=None) -> None:
    """Build the persistent M = 1 kernel and complete one minimal pass (the probe's first dispatch)."""
    import mlx.core as mx

    if sim is not None and sim.fail_compile:
        raise RuntimeError("simulated Metal library build failure (sim fail_compile)")
    mx.eval(mk.run_pass(hist.arrays, _first_pass_params(mk, att, hist.capacity)))


def bring_up(W, hist, att: dict, grid, kv8: bool, sim=None, log=print, report: dict | None = None):
    """Run the compile and probe stages over grid.candidates. Returns (kernel, probe result, tried), where tried lists
    every grid probed as {"G", "ok", "s" (seconds, kernel build to verdict), "reason"?}; raises MKUnavailable at the
    first stage that fails. `report` (optional) receives {"compile": {ok, s}, "probe": {ok, s, G}}."""
    rep = report if report is not None else {}
    tried: list = []
    cands = list(grid.candidates)
    t_probe = 0.0
    for i, G in enumerate(cands):
        t0 = time.perf_counter()
        mk = RN.MegaKernel(W, G=G, log=log, kv8=kv8)
        if i == 0:
            try:
                compile_stage(mk, hist, att, sim)
            except Exception as exc:  # a Metal build or dispatch failure on this GPU
                rep["compile"] = {"ok": False, "s": round(time.perf_counter() - t0, 3)}
                raise MKUnavailable("compile", f"{type(exc).__name__}: {exc}", tried, rep) from exc
            rep["compile"] = {"ok": True, "s": round(time.perf_counter() - t0, 3)}
        t1 = time.perf_counter()
        try:
            if sim is not None and sim.probe_fails(G):
                raise RN.BarrierProbeError(
                    f"simulated residency probe failure at G={G} (sim fail_probe_G)"
                )
            pr = mk.probe(hist.arrays, att, hist.capacity)
        except RN.BarrierProbeError as exc:
            t_probe += time.perf_counter() - t1
            tried.append(
                {
                    "G": G,
                    "ok": False,
                    "s": round(time.perf_counter() - t0, 3),
                    "reason": str(exc)[:240],
                }
            )
            nxt = f"; trying G={cands[i + 1]}" if i + 1 < len(cands) else ""
            log(f"[engine] megakernel G={G} refused by the residency probe: {str(exc)[:160]}{nxt}")
            del mk
            continue
        except Exception as exc:
            tried.append(
                {
                    "G": G,
                    "ok": False,
                    "s": round(time.perf_counter() - t0, 3),
                    "reason": f"{type(exc).__name__}: {exc}"[:240],
                }
            )
            rep["probe"] = {"ok": False, "s": round(t_probe + time.perf_counter() - t1, 3)}
            raise MKUnavailable("probe", f"G={G}: {type(exc).__name__}: {exc}", tried, rep) from exc
        tried.append({"G": G, "ok": True, "s": round(time.perf_counter() - t0, 3)})
        rep["probe"] = {"ok": True, "s": round(t_probe + time.perf_counter() - t1, 3), "G": G}
        return mk, pr, tried
    grids = ", ".join(str(g) for g in cands)
    rep["probe"] = {"ok": False, "s": round(t_probe, 3)}
    raise MKUnavailable(
        "probe",
        f"no grid of {grids} threadgroups ran co-resident ({tried[-1]['reason'][:160]})"
        if len(cands) > 1
        else tried[-1]["reason"],
        tried,
        rep,
    )


# ------------------------------------------------------------------------------------------------ stock statistics
def ref_rows(logits: np.ndarray) -> list:
    """Stock MLX's rows: per position its argmax over the bf16 logits (lowest id on a tie) and its top-TOPN_REF ids
    with fp64 log-probabilities, best first (ties: lowest id first). logits: [K, vocab] float32."""
    out = []
    for lf in np.asarray(logits, dtype=np.float32):
        x = lf.astype(np.float64)
        m = float(x.max())
        lse = m + float(np.log(np.exp(x - m).sum()))
        thr = np.partition(x, x.size - TOPN_REF)[x.size - TOPN_REF]
        idx = np.nonzero(x >= thr)[0]
        idx = idx[np.lexsort((idx, -x[idx]))][:TOPN_REF]
        out.append({"argmax": int(np.argmax(lf)), "ids": idx.astype(np.int64), "lp": x[idx] - lse})
    return out


def position_d20(cand_ids, cand_lps, ref: dict) -> float:
    """max over the reference's top-20 of |lp_cand - lp_ref|; an id missing from the kernel's top-20 counts
    max(0, lp_ref - the kernel's 20th log-prob)."""
    cmap = {int(t): float(v) for t, v in zip(cand_ids, cand_lps)}
    lowest = min(cmap.values())
    d = 0.0
    for t, lr in zip(ref["ids"].tolist(), ref["lp"].tolist()):
        d = max(d, abs(cmap[t] - lr) if t in cmap else max(0.0, lr - lowest))
    return d


def stock_stats(words: list, logits: np.ndarray) -> dict:
    """Per position (one megakernel pass result per reference row): agreement and d20; their aggregate."""
    refs = ref_rows(logits)
    d20, agree = [], []
    for w, ref in zip(words, refs):
        w = np.asarray(w, dtype=np.uint32)
        ids = w[P.R_TOPI : P.R_TOPI + P.TOPN]
        lps = w[P.R_TOPV : P.R_TOPV + P.TOPN].view(np.float32).astype(np.float64)
        d20.append(position_d20(ids, lps, ref) if np.all(np.isfinite(lps)) else float("inf"))
        agree.append(int(w[P.R_ARGMAX]) == ref["argmax"])
    d = np.array(d20)
    return {
        "n": len(d20),
        "agree": float(np.mean(agree)),
        "d20_median": float(np.median(d)),
        "d20_max": float(d.max()),
        "d20": [round(float(x), 4) for x in d20],
        "agree_each": [int(a) for a in agree],
    }


_UNSET = object()


def stock_verdict(st: dict, A=_UNSET, D=_UNSET) -> tuple:
    """(ok, reason): agreement >= A and, when D is a bound (not None), median d20 <= D. Defaults: STOCK_A, STOCK_D."""
    A = STOCK_A if A is _UNSET else A
    D = STOCK_D if D is _UNSET else D
    ok = st["agree"] >= A and (D is None or st["d20_median"] <= D)
    bound = "reported, no bound" if D is None else f"needs <= {D}"
    why = (
        ""
        if ok
        else (
            f"{st['n']} forced positions of the fixed text vs stock MLX: top-1 agreement "
            f"{st['agree']:.3f} (needs >= {A}), median d20 {st['d20_median']:.3f} nat ({bound})"
        )
    )
    return ok, why


# ------------------------------------------------------------------------------------------------ mutants (sim)
def wrong_expert_table(plan) -> tuple:
    """Mutant wrong_expert: the descriptor table with one routed layer's expert bank and base pointing at the next
    routed layer's experts (same record layout, wrong weights). Returns (table, layer, next layer)."""
    tab = P.pack_descriptors(plan)
    routed = sorted(plan.expert_home)
    i = len(routed) // 2
    layer, nxt = routed[i], routed[i + 1]
    for f in (P.D_EBANK, P.D_EBASE):
        tab[layer * P.NF + f] = tab[nxt * P.NF + f]
    P.check_tg_descriptors(tab)
    return tab, layer, nxt


def flip_logits_bit(mk) -> None:
    """Mutant chain_bitflip: flip the low bit of LOGITS byte 0 in the kernel's scratch (after a pass, before it is
    read), as a race that left one wrong logit would."""
    import mlx.core as mx

    o = P.OFF["LOGITS"]
    s = mk.scratch
    s[o : o + 1] = mx.bitwise_xor(s[o : o + 1], mx.array(1, dtype=mx.uint8))
    mx.eval(s)


def zero_full_rows(hist, cfg, start: int, rows: int) -> None:
    """Mutant stale_history: full-layer K / V rows start .. start + rows - 1 zeroed (in place, every full layer)."""
    import mlx.core as mx

    z = mx.zeros((1, cfg.n_kv_heads, rows, cfg.head_dim), dtype=mx.bfloat16)
    for layer in cfg.full_layers:
        hist.write_full(layer, start, z, z)


def text_ids(tokenizer, n: int) -> list:
    """The first n ids of the fixed text (repeated if a tokenizer ever yields fewer)."""
    ids = list(tokenizer.encode(P0_TEXT))
    if not ids:
        raise ValueError("the tokenizer returned no ids for the load-check text")
    while len(ids) < n:
        ids = ids + ids
    return ids[:n]


def text_info(n_prompt: int, n_forced: int) -> dict:
    return {
        "version": P0_VERSION,
        "sha256": hashlib.sha256(P0_TEXT.encode()).hexdigest()[:16],
        "prompt_tokens": n_prompt,
        "forced_tokens": n_forced,
    }


# ------------------------------------------------------------------------------------------------ stages 3-5
class _Check:
    """The state stages chain, stock and verify share: the fixed text prefilled with stock MLX ops, a copy of the
    sliding rings after it, the mutant in force."""

    def __init__(self, W, mk, hist, att: dict, ids: list, n: int, mutant: str | None, log):
        self.W, self.mk, self.hist, self.att, self.log = W, mk, hist, att, log
        self.cfg = W.cfg
        self.n, self.K = n, FORCED_TOKENS
        self.prompt, self.forced = ids[:n], ids[n : n + FORCED_TOKENS]
        self.mutant = mutant
        self.shift = 1 if mutant == "pos_shift" else 0
        self.snap = None
        self.desc0 = None

    # ---- state
    def setup(self) -> dict:
        import mlx.core as mx

        from jitengine_flint_m4pro.model.model import prefill_chunk

        t0 = time.perf_counter()
        info = {}
        self.hist.reserve(self.n + self.K + P.VMAX + 2)
        if self.mutant == "wrong_expert":
            tab, layer, nxt = wrong_expert_table(self.W.plan)
            self.desc0 = self.mk.desc
            self.mk.desc = mx.array(tab)
            mx.eval(self.mk.desc)
            info["mutant"] = f"wrong_expert: layer {layer}'s experts read from layer {nxt}'s"
        self.hist.invalidate()
        lg = prefill_chunk(
            self.W, self.hist, mx.array(self.prompt, dtype=mx.int32), 0, [self.n - 1]
        )
        mx.eval(lg, self.hist.state_arrays())
        if self.mutant == "stale_history":
            zero_full_rows(self.hist, self.cfg, self.n - STALE_ROWS, STALE_ROWS)
            info["mutant"] = (
                f"stale_history: full-layer rows {self.n - STALE_ROWS}..{self.n - 1} zeroed"
            )
        self.snap = self.hist.snapshot(self.prompt, lg[0])
        del lg
        mx.clear_cache()
        info["s"] = round(time.perf_counter() - t0, 3)
        return info

    def restore(self) -> None:
        """Back to the state after the prefill: the sliding rings from their copy, a zeroed scratch (the full-layer rows
        at and past the prompt are rewritten by each pass before a later pass reads them)."""
        import mlx.core as mx

        self.hist.restore(self.snap)
        self.mk.scratch = mx.zeros((P.MK_SCRATCH_BYTES,), dtype=mx.uint8)
        mx.eval(self.mk.scratch)

    def close(self) -> None:
        import mlx.core as mx

        if self.desc0 is not None:
            self.mk.desc = self.desc0
            self.desc0 = None
        self.snap = None
        self.hist.invalidate()
        mx.clear_cache()

    # ---- passes
    def prm(self, k: int, token: int):
        return P.make_params(
            pos=self.n + k - 1 + self.shift,
            token=token,
            forced_next=None,
            sel_mode=0,
            top_k=P.TOPN,
            top_p=1.0,
            temperature=0.0,
            u=0.0,
            poison=False,
            cap=self.hist.capacity,
            theta=self.cfg.rope_theta,
            **self.att,
        )

    def region(self, name: str) -> np.ndarray:
        o = P.OFF[name]
        return np.array(self.mk.scratch[o : o + _SIZE[name]])

    def run_m1(
        self, passes: int, chain: bool = False, capture: bool = False, flip: int | None = None
    ) -> list:
        """passes chained M = 1 passes from the restored state, fed the text's next tokens. Returns per pass (result
        words, LOGITS bytes, HS bytes) (the regions only with capture, which syncs after every pass)."""
        import mlx.core as mx

        self.restore()
        prev, out = None, []
        for k in range(1, passes + 1):
            prm = self.prm(k, self.forced[k - 1])
            r = (self.mk.run_pass_chain if chain else self.mk.run_pass)(self.hist.arrays, prm, prev)
            if capture:
                mx.eval(r)
                if flip == k:
                    flip_logits_bit(self.mk)
                out.append((np.array(r), self.region("LOGITS"), self.region("HS")))
            else:
                out.append(r)
            prev = r
        if not capture:
            mx.eval(out)
            out = [(np.array(r), None, None) for r in out]
        return out

    @staticmethod
    def kernel_errors(run: list, what: str) -> str | None:
        for k, (w, _, _) in enumerate(run, 1):
            if int(w[P.R_ERR]):
                return f"{what} pass {k} failed in the kernel: {P.describe_err(int(w[P.R_ERR]))}"
            if not int(w[P.R_DONE]):
                return f"{what} pass {k} did not complete"
        return None

    @staticmethod
    def first_diff(x: list, y: list) -> str | None:
        for k, (a, b) in enumerate(zip(x, y), 1):
            pairs = (
                ("result words", a[0][: P.R_ARRIVE], b[0][: P.R_ARRIVE]),
                ("LOGITS", a[1], b[1]),
                ("HS", a[2], b[2]),
            )
            bad = [nm for nm, u, v in pairs if not np.array_equal(u, v)]
            if bad:
                return f"pass {k}: {', '.join(bad)}"
        return None

    # ---- stage 3
    def chain_stage(self) -> dict:
        flip = FLIP_PASS if self.mutant == "chain_bitflip" else None
        a = self.run_m1(CHAIN_PASSES, capture=True, flip=flip)
        b = self.run_m1(CHAIN_PASSES, chain=True, capture=True)
        c = self.run_m1(CHAIN_PASSES, capture=True)
        err = (
            self.kernel_errors(a, "persistent")
            or self.kernel_errors(b, "chain")
            or self.kernel_errors(c, "persistent (repeat)")
        )
        d_ab, d_ac = self.first_diff(a, b), self.first_diff(a, c)
        rec = {
            "passes": CHAIN_PASSES,
            "positions": [self.n + self.shift, self.n + self.shift + CHAIN_PASSES - 1],
            "chain_equal": d_ab is None,
            "repeat_equal": d_ac is None,
        }
        if err:
            rec.update(ok=False, reason=err)
        elif d_ab is not None:
            rec.update(
                ok=False,
                reason=f"the persistent pass differs from its one-dispatch-per-phase chain at {d_ab} "
                f"(a race or a fence that does not hold on this GPU)",
            )
        elif d_ac is not None:
            rec.update(
                ok=False,
                reason=f"two persistent runs from the same state differ at {d_ac} (not deterministic)",
            )
        else:
            rec["ok"] = True
        return rec

    # ---- stage 4, first half: the megakernel passes (their words also feed stage 5)
    def stock_candidate(self) -> list:
        run = self.run_m1(self.K)
        err = self.kernel_errors(run, "forced")
        if err:
            raise _StageFail(err)
        return [w for w, _, _ in run]

    # ---- stage 5
    def verify_stage(self, words: list, rows) -> dict:
        import mlx.core as mx

        cases, bad = [], []
        top = max(rows)
        for M in rows:
            self.restore()
            toks = list(self.forced[:M])
            if self.mutant == "verify_row" and M == top:
                toks[-1] = (toks[-1] + 1) % self.cfg.vocab
            vp = P.make_vparams(
                pos=self.n + self.shift,
                tokens=toks,
                sel_mode=0,
                top_k=P.TOPN,
                top_p=1.0,
                temperature=0.0,
                us=[0.0] * M,
                poison_rows=0,
                cap=self.hist.capacity,
                theta=self.cfg.rope_theta,
                **self.att,
            )
            r = self.mk.run_vpass(self.hist.arrays, vp)
            mx.eval(r)
            a = np.array(r)
            err = int(a[P.VR_ERR])
            rows_bad = [
                m
                for m in range(M)
                if not np.array_equal(
                    a[m * P.VROW_WORDS : (m + 1) * P.VROW_WORDS], words[m][: P.VROW_WORDS]
                )
            ]
            cases.append({"M": M, "err": err, "rows_bad": rows_bad})
            if err or rows_bad:
                what = (
                    P.describe_err(err) if err else f"rows {rows_bad} differ from the M = 1 passes"
                )
                bad.append(f"M={M}: {what}")
        rec = {"rows": list(rows), "cases": cases, "ok": not bad}
        if bad:
            rec["reason"] = (
                "verify rows differ from the M = 1 passes at their positions ("
                + "; ".join(bad)[:300]
                + ")"
            )
        return rec

    # ---- stage 4, second half: stock MLX's echo of the whole text (rewrites the history: runs last)
    def stock_reference(self) -> np.ndarray:
        import mlx.core as mx

        from jitengine_flint_m4pro.model.model import prefill_chunk

        self.snap = None
        self.hist.invalidate()
        ids = mx.array(self.prompt + self.forced, dtype=mx.int32)
        lg = prefill_chunk(self.W, self.hist, ids, 0, list(range(self.n, self.n + self.K)))
        lf = lg.astype(mx.float32)
        mx.eval(lf)
        out = np.array(lf)
        del lg, lf
        return out


class _StageFail(Exception):
    pass


class _Clock:
    """The check's bound and its failures: an MKUnavailable carrying the report so far."""

    def __init__(self, rep: dict, t0: float, tried):
        self.rep, self.t0, self.tried = rep, t0, tried

    def fail(self, stage: str, reason: str) -> MKUnavailable:
        self.rep["s"] = round(time.perf_counter() - self.t0, 3)
        return MKUnavailable(stage, reason, self.tried, self.rep)

    def over(self, stage: str) -> None:
        total = time.perf_counter() - self.t0
        if total > BOUND_S:
            raise self.fail(
                stage, f"the load check took {total:.0f} s, over its {BOUND_S:.0f} s bound"
            )


def _chain(c: _Check, rep: dict, clock: _Clock) -> None:
    t = time.perf_counter()
    try:
        rec = c.chain_stage()
    except Exception as exc:  # the chain render's build or a dispatch failed on this GPU
        rep["chain"] = {"ok": False, "s": round(time.perf_counter() - t, 3)}
        raise clock.fail("chain", f"{type(exc).__name__}: {exc}") from exc
    rec["s"] = round(time.perf_counter() - t, 3)
    rep["chain"] = rec
    if not rec["ok"]:
        raise clock.fail("chain", rec["reason"])
    clock.over("chain")


def _stock_and_verify(c: _Check, mk, rep: dict, clock: _Clock) -> None:
    """Stock (candidate passes), verify (rows against them), stock (reference and verdict)."""
    t = time.perf_counter()
    try:
        words = c.stock_candidate()
    except Exception as exc:
        rep["stock"] = {"ok": False, "s": round(time.perf_counter() - t, 3)}
        why = str(exc) if isinstance(exc, _StageFail) else f"{type(exc).__name__}: {exc}"
        raise clock.fail("stock", why) from exc
    t_cand = time.perf_counter() - t
    rows = tuple(sorted(getattr(mk, "vrows", ()) or ()))
    t = time.perf_counter()
    if rows:
        try:
            vrec = c.verify_stage(words, rows)
        except Exception as exc:  # noqa: BLE001  (a verify failure turns speculation off; it is reported)
            vrec = {"rows": list(rows), "ok": False, "reason": f"{type(exc).__name__}: {exc}"}
    else:
        vrec = {"rows": [], "ok": True, "skipped": "no verify passes compiled (speculation off)"}
    vrec["s"] = round(time.perf_counter() - t, 3)
    t = time.perf_counter()
    try:
        ref = c.stock_reference()
    except Exception as exc:
        rep["stock"] = {"ok": False, "s": round(t_cand + time.perf_counter() - t, 3)}
        raise clock.fail(
            "stock", f"stock MLX echo of the fixed text: {type(exc).__name__}: {exc}"
        ) from exc
    st = stock_stats(words, ref)
    ok, why = stock_verdict(st)
    st.update(ok=ok, A=STOCK_A, D=STOCK_D, s=round(t_cand + time.perf_counter() - t, 3))
    if not ok:
        st["reason"] = why
    rep["stock"] = st
    if not ok:
        raise clock.fail("stock", why)
    clock.over("stock")
    rep["verify"] = vrec


def check(
    W,
    mk,
    hist,
    att: dict,
    tokenizer,
    sim=None,
    log=print,
    t0: float | None = None,
    tried=(),
    report: dict | None = None,
) -> dict:
    """Stages chain, stock and verify on `mk` (through compile and probe; its verify rows, if any, compiled and
    probed). Returns the load-check report ({stage: {ok, s, ...}}, "s", "bound_s", "text"); raises MKUnavailable at a
    failed chain or stock stage. A failed verify stage is reported (report["verify"]["ok"] False) for the caller to turn
    speculation off. t0: when the check began (the compile stage), for the bound."""
    t0 = time.perf_counter() if t0 is None else t0
    rep = report if report is not None else {}
    mutant = getattr(sim, "check_mutant", None) if sim is not None else None
    n = min(PROMPT_TOKENS, hist.limit - FORCED_TOKENS - P.VMAX - 2)
    rep["bound_s"] = BOUND_S
    rep["text"] = text_info(max(n, 0), FORCED_TOKENS)
    if mutant:
        rep["mutant"] = mutant
    if n < MIN_PROMPT:
        why = f"skipped: the context limit {hist.limit} is too small for the {MIN_PROMPT + FORCED_TOKENS}-token text"
        for s in ("chain", "stock", "verify"):
            rep[s] = {"ok": True, "skipped": why, "s": 0.0}
        rep["s"] = round(time.perf_counter() - t0, 3)
        return rep
    ids = text_ids(tokenizer, n + FORCED_TOKENS)
    c = _Check(W, mk, hist, att, ids, n, mutant, log)
    clock = _Clock(rep, t0, tried)
    try:
        try:
            rep["prefill"] = c.setup()
        except Exception as exc:
            raise clock.fail(
                "chain", f"prefill of the fixed text: {type(exc).__name__}: {exc}"
            ) from exc
        _chain(c, rep, clock)
        _stock_and_verify(c, mk, rep, clock)
        rep["s"] = round(time.perf_counter() - t0, 3)
        vrec = rep["verify"]
        if rep["s"] > BOUND_S and vrec["ok"]:
            vrec.update(
                ok=False,
                reason=f"the load check took {rep['s']:.0f} s, over its {BOUND_S:.0f} s bound",
            )
        return rep
    finally:
        c.close()


def summary(rep: dict) -> str:
    """One log line for a passed check."""
    parts = []
    ch, st, vf = rep.get("chain", {}), rep.get("stock", {}), rep.get("verify", {})
    if ch.get("skipped"):
        return f"load check: chain / stock / verify {ch['skipped']}"
    parts.append(
        f"chain ok ({ch.get('passes')} passes = the per-phase chain, repeat identical; {ch.get('s')} s)"
    )
    bound = "reported" if st.get("D") is None else f"D {st.get('D')}"
    parts.append(
        f"stock MLX agreement {st.get('agree', 0):.3f} (A {st.get('A')}), median d20 "
        f"{st.get('d20_median', 0):.3f} nat ({bound}; {st.get('s')} s)"
    )
    if vf.get("skipped"):
        parts.append(f"verify {vf['skipped']}")
    else:
        rows = ",".join(str(m) for m in vf.get("rows", []))
        parts.append(
            f"verify rows M={rows} {'exact' if vf.get('ok') else 'DIFFER'} ({vf.get('s')} s)"
        )
    return f"load check {rep.get('s')} s: " + "; ".join(parts)
