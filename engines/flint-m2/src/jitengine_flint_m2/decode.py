"""The engine's generate loop on the megakernel backend.

Step 0 comes from the prefill logits on the host (as on the graph backend). Decode passes 1..m are encoded in
release groups of <= 8 back-to-back persistent passes, each taking the previous pass's result as an input (it
carries the chosen token and orders the in-place history writes). The next group is encoded before the current
group is read, so the host's sync, release and bookkeeping overlap the GPU. One host sync per group.

Token choice inside the kernel: greedy (argmax, lowest id on a tie) and seeded sampling with 1 <= top_k <= 20
(T, top-k, top-p, inverse CDF at sampler.uniform(seed, j)); forced streams take the token from the parameters.
top_k = 0 or > 20 falls back to sampler.sample on the pass's bf16 logits with one sync per pass (correct,
slower). Nothing at or after EOS, a stop string, a fault or a cancel is released; a pass already in flight is
settled before the request returns (a pass whose previous pass failed returns at once in the kernel: pass_entry).

The attention partition is the per-position grid: every pass's parameters carry the policy's span switch T
only, never a request quantity, so the stream does not depend on max_tokens or the context limit.
"""

from __future__ import annotations

import time

import numpy as np

from jitengine_flint_m2.engine import faults
from jitengine_flint_m2.engine.io_struct import GenResult, StepOutput
from jitengine_flint_m2.kernels import plan as P
from jitengine_flint_m2.sampler import uniform
from jitengine_flint_m2.tokenizer import IncrementalDecoder

GROUP = 8


class _Passes:
    def __init__(self, eng, plan, first_token: int):
        self.mk, self.hist, self.plan = eng.mk, eng.hist, plan
        self.n = len(plan.prompt)
        self.greedy = plan.forced is not None or plan.temperature <= 0 or plan.top_k == 1
        self.in_kernel = self.greedy or 1 <= plan.top_k <= P.TOPN
        self.att = P.attn_params(eng.mk_chunks)  # the per-position grid: no N_max, no max_tokens
        self.theta = eng.cfg.rope_theta
        self.first_token = first_token
        self.next_pass = 1
        self.last = None

    def params(self, k: int, host_token: int | None):
        plan = self.plan
        if plan.forced is not None:
            token = plan.forced[k - 1]
            forced_next = plan.forced[k] if k < len(plan.forced) else None
        else:
            token = host_token
            forced_next = None
        return P.make_params(
            pos=self.n + k - 1,
            token=token,
            forced_next=forced_next,
            sel_mode=0 if self.greedy else 1,
            top_k=plan.top_k,
            top_p=plan.top_p,
            temperature=plan.temperature,
            u=uniform(plan.seed, k) if not self.greedy else 0.0,
            poison=faults.poisons_logits(plan.fault, k),
            cap=self.hist.capacity,
            theta=self.theta,
            stall=faults.stalls(plan.fault, k),
            **self.att,
        )

    def encode(self, g: int, host_token: int | None = None) -> list:
        import mlx.core as mx

        out = []
        for k in range(self.next_pass, self.next_pass + g):
            tok = self.first_token if k == 1 else host_token if k == self.next_pass else None
            r = self.mk.run_pass(self.hist.arrays, self.params(k, tok), self.last)
            out.append((k, r))
            self.last = r
        self.next_pass += g
        mx.async_eval([r for _, r in out])
        return out


def _step_from_result(k: int, a: np.ndarray, plan) -> StepOutput:
    return _step_from_parsed(k, P.parse_result(a), plan)


def _step_from_parsed(k: int, r: dict, plan) -> StepOutput:
    if r["err"]:
        raise faults.GPUSyncError(
            f"megakernel decode pass {k} failed in the kernel: {P.describe_err(r['err'])}"
        )
    if not r["done"]:
        raise faults.GPUSyncError(f"megakernel pass {k} did not complete")
    if not r["finite"]:
        raise faults.NonFiniteLogits(f"nonfinite logits at completion index {k}")
    top = list(zip(r["top_ids"], r["top_lps"]))
    if plan.forced is not None:
        tok = plan.forced[k] if k < len(plan.forced) else None
        return StepOutput(
            k, tok, r["forced_lp"] if tok is not None else None, top, time.perf_counter()
        )
    lp = r["chosen_lp"] if plan.logprobs is not None else None
    return StepOutput(
        k, r["token"], lp, top[: plan.logprobs] if plan.logprobs else [], time.perf_counter()
    )


def generate(eng, job) -> None:  # noqa: C901, PLR0915
    import mlx.core as mx

    plan = job.plan
    n = len(plan.prompt)
    t0 = time.perf_counter()
    mx.reset_peak_memory()
    result = GenResult([], "length", n)
    if plan.n_out == 0:
        result.metrics = eng._metrics(plan, result, t0, t0, None, None, 0)
        job.out.put(result)
        return
    eng.hist.reserve(n + plan.passes)
    logits0, cached = eng._prefill_or_restore(plan.prompt, cancel=job.cancel)
    result.cached_prompt_tokens = cached
    t_prefill = time.perf_counter()
    _tok0, outs = eng._step_graph(logits0, 0, plan)
    mx.async_eval(eng._arrays(outs))
    step = eng._sync(0, outs, plan)
    dec = IncrementalDecoder(eng.tokenizer) if plan.stop else None
    t_first = step.t_release
    t_last = t_first
    finish = None
    passes = _Passes(eng, plan, step.token_id if step.token_id is not None else 0)
    pending: list = []
    mem0 = mx.get_active_memory()

    def release(s: StepOutput) -> bool:
        nonlocal finish, t_last
        job.out.put(s)
        result.steps.append(s)
        t_last = s.t_release
        if s.token_id is not None:
            result.token_ids.append(s.token_id)
        if plan.forced is None and not plan.ignore_eos and s.token_id in eng.eos:
            finish = "stop"
        if dec is not None and s.token_id is not None:
            dec.push(s.token_id)
            cut = plan.stop.find(dec.text)
            if cut is not None:
                finish = "stop"
                result.text = dec.text[:cut]
        return bool(finish) or len(result.steps) >= plan.n_out or job.cancel.is_set()

    spec = _spec_enabled(eng, plan, passes)
    if spec:
        base = eng.draft_base(plan.prompt)  # built before step 0 is released (outside the window)
    stats = None
    try:
        done = release(step)
        if not passes.in_kernel:
            done = done or _fallback_loop(eng, plan, passes, release)
        elif spec:
            if not done:
                stats = _spec_loop(eng, plan, passes, release, base, mem0)
        else:
            if not done:
                pending = passes.encode(min(GROUP, plan.passes))
            while pending and not done:
                remaining = plan.passes - (passes.next_pass - 1)
                ahead = passes.encode(min(GROUP, remaining)) if remaining > 0 else []
                for k, r in pending:
                    a = np.array(r)
                    eng.mk.check_memory_once(mem0)
                    faults.fire_host(plan.fault, k)
                    if release(_step_from_result(k, a, plan)):
                        done = True
                        break
                pending = ahead
    finally:
        if passes.last is not None:
            mx.eval(passes.last)  # settle any pass still in flight (history is then concrete)
    eng.hist.length = n + len(result.steps) - 1
    eng._finish(job, result, finish)  # stop / length, or "error" when a cancel cut it short
    if dec is not None and result.text is None:
        result.text = dec.text
    result.metrics = eng._metrics(plan, result, t0, t_prefill, t_first, t_last, cached)
    if result.finish_reason == "error":
        result.metrics["error_type"] = eng._cancel_reason(job)[0]
    if stats is not None:
        result.metrics.update(stats.summary())
        eng.log(f"[engine] speculation: {stats.line()}")
    elif passes.in_kernel and plan.forced is None:
        result.metrics.update(one_token_per_pass=True, decode_passes=max(0, len(result.steps) - 1))
    job.out.put(result)


# ------------------------------------------------------------------ speculation (mode `code`)
def _spec_enabled(eng, plan, passes) -> bool:
    """Verify passes run for in-kernel greedy / seeded requests, unless one token per pass is requested. Forced (raw /
    teacher-forced) requests never use them. Full layers are layout 6d at every N_max (the default "l6d"
    partition) for M = 1 and verify rows alike, so speculation runs at every context; under the legacy "strided"
    partition the long layout-2 stream (N_max > LONG_FULL) has no multi-row form and every pass stays M = 1."""
    otp = plan.one_token_per_pass if plan.one_token_per_pass is not None else eng.one_token_per_pass
    return (
        eng.speculation
        and bool(eng.mk.vrows)
        and plan.forced is None
        and passes.in_kernel
        and not otp
        and P.verify_capable(passes.att)
        and plan.passes >= 2
    )


class _SpecStats:
    def __init__(self, vmax: int):
        self.passes = 0
        self.verify = 0
        self.tokens = 0
        self.rows = [0] * (vmax + 1)  # passes by row count M
        self.accepted = [0] * vmax  # verify passes by accepted drafts (0 .. M - 1)
        self.drafted = 0
        self.drains = 0  # passes encoded into an empty queue (the GPU idled for the host)
        self.backoffs = 0  # depth reductions made by the online acceptance guard
        self.host_s = 0.0

    def summary(self) -> dict:
        return {
            "one_token_per_pass": False,
            "decode_passes": self.passes,
            "verify_passes": self.verify,
            "tokens_per_pass": round(self.tokens / self.passes, 4) if self.passes else None,
            "passes_by_rows": self.rows,
            "accepted_hist": self.accepted,
            "drafted_tokens": self.drafted,
            "queue_drains": self.drains,
            "guard_backoffs": self.backoffs,
            "host_ms_per_pass": round(self.host_s * 1e3 / self.passes, 4) if self.passes else None,
        }

    def line(self) -> str:
        sm = self.summary()
        return (
            f"{self.tokens} tokens in {self.passes} passes ({sm['tokens_per_pass']} per pass; {self.verify} "
            f"verify, rows {self.rows[1:]}, accepted {self.accepted}, drafted {self.drafted}, drains "
            f"{self.drains}, guard backoffs {self.backoffs}, host {sm['host_ms_per_pass']} ms/pass)"
        )


QDEPTH = 3  # units (M = 1 groups or one verify pass) in flight: one running, two queued


def _group(certain_left: int) -> int:
    """M = 1 passes in the next unit when certain_left passes beyond those in flight are certain to draft nothing: half
    of them (1..GROUP), so the certainty horizon is spread over several units and one is still queued when the next
    lookup comes (a unit of all of them would leave nothing to queue behind it: the queue drains although no decision
    needs the in-flight tokens; replay: 273 -> 175 drains on the 32K greedy stream, 90 -> 61 at 1K)."""
    return min(GROUP, max(1, certain_left // 2))


class _Unit:
    """One encoded unit of the speculation loop: a group of M = 1 passes (one command buffer) or one verify pass."""

    __slots__ = ("j0", "rs", "drafts", "n", "k")

    def __init__(self, j0: int, rs: list, drafts=None, n: int = 0, k: int = 0):
        self.j0, self.rs, self.drafts, self.n, self.k = j0, rs, drafts, n, k


def _spec_loop(eng, plan, passes: _Passes, release, base, mem0) -> _SpecStats:  # noqa: C901, PLR0915
    """Commit the longest prefix the target itself selects. A verify pass of M = k + 1 rows feeds the last committed
    token and k drafts; row m's selection t_m (greedy argmax, or the seeded draw at u(seed, j + m)) is committed
    while every earlier draft equalled the target's own selection. Every row is bit-identical to the M = 1 pass at
    its position, so the committed stream is the one-token-per-pass stream whatever the policy decides.

    The loop is pipelined. M = 1 passes are queued on the device, each taking its token from the previous
    pass's own selection (the argmax or the (seed, j) draw in the result buffer), whenever every token they depend on
    is certain to be one an M = 1 pass commits: the next pass is certain to draft nothing even if the tokens still in
    flight extend the current match (copy_draft.certain_no_draft). So the GPU never waits for the host's accept / draft
    decision on those passes; up to QDEPTH units are in flight (_group spreads the certain passes over them) and
    each unit is read while the next ones run. The queue drains (one host round trip with the GPU idle) only where
    the decision needs the in-flight tokens: before a pass that may draft, and after a verify pass (its acceptance
    sets the next position). Each unit is read, the next one encoded, and only then are its steps released."""
    import mlx.core as mx

    from jitengine_flint_m2 import copy_draft as D

    mk, hist = eng.mk, eng.hist
    n = len(plan.prompt)
    mode = "greedy" if passes.greedy else "sampled"
    sel = 0 if passes.greedy else 1
    vrows = set(mk.vrows)
    vmax = max(vrows)
    dx = D.DraftIndex(base)
    dx.append(passes.first_token)
    pol = D.DepthPolicy(mode)
    st = _SpecStats(P.VMAX)
    N = plan.passes
    S = {
        "known": 0,
        "last": passes.first_token,
        "j_enc": 1,
        "prev": None,
    }  # host-committed index / token
    q: list = []

    def encode_m1(g: int, host_token):
        j0 = S["j_enc"]
        rs = []
        for c in range(g):
            r = mk.run_pass(
                hist.arrays, passes.params(j0 + c, host_token if c == 0 else None), S["prev"]
            )
            S["prev"] = passes.last = r  # generate()'s finally settles whatever is still in flight
            rs.append(r)
        mx.async_eval(rs)
        q.append(_Unit(j0, rs))
        S["j_enc"] = j0 + g

    def top_up():
        """Encode units while the queue has room and the next pass's inputs are certain (or the queue is empty)."""
        while S["j_enc"] <= N and len(q) < QDEPTH:
            j = S["j_enc"]
            nm, e = dx.lookup()  # the match of the host's history (through token `known`)
            tab = pol.table(n + j)
            u = j - 1 - S["known"]  # tokens pass j depends on that are still on the device
            rem = N - j + 1
            if u == 0:  # the queue is empty: an exact decision
                if st.passes:
                    st.drains += 1
                k = min(tab[D.bucket(nm)], rem - 1, vmax - 1)
                while k > 0 and (k + 1) not in vrows:
                    k -= 1
                if k > 0:
                    d = dx.draft(e, k)
                    M = k + 1
                    us = [uniform(plan.seed, j + m) if sel else 0.0 for m in range(M)]
                    poison = sum(
                        1 << m for m in range(M) if faults.poisons_logits(plan.fault, j + m)
                    )
                    stall = any(faults.stalls(plan.fault, j + m) for m in range(M))
                    prm = P.make_vparams(
                        pos=n + j - 1,
                        tokens=[S["last"]] + d,
                        sel_mode=sel,
                        top_k=plan.top_k,
                        top_p=plan.top_p,
                        temperature=plan.temperature,
                        us=us,
                        poison_rows=poison,
                        cap=hist.capacity,
                        theta=passes.theta,
                        stall=stall,
                        **passes.att,
                    )
                    r = mk.run_vpass(hist.arrays, prm, S["prev"])
                    S["prev"] = passes.last = r
                    mx.async_eval(r)
                    q.append(_Unit(j, [r], d, nm, k))
                    return  # the position after a verify pass is known once it is read
                encode_m1(min(_group(D.certain_no_draft(nm, tab)), rem), S["last"])
            else:
                if q[-1].drafts is not None:
                    return
                left = D.certain_no_draft(nm, tab) - u
                if left < 1:
                    return
                encode_m1(min(_group(left), rem), None)

    def read(unit: _Unit) -> list:
        """Parse a unit's results and commit its tokens on the host (index, policy); returns the steps to release."""
        steps = []
        if unit.drafts is None:
            for c, r in enumerate(unit.rs):
                a = np.array(r)
                mk.check_memory_once(mem0)
                faults.fire_host(plan.fault, unit.j0 + c)
                s = _step_from_result(unit.j0 + c, a, plan)
                st.passes += 1
                st.rows[1] += 1
                st.tokens += 1
                dx.append(s.token_id)
                S["last"] = s.token_id
                S["known"] = unit.j0 + c
                steps.append(s)
            return steps
        a = np.array(unit.rs[0])
        mk.check_memory_once(mem0)
        M = unit.k + 1
        rows = P.parse_vresult(a, M)
        st.passes += 1
        st.verify += 1
        st.rows[M] += 1
        st.drafted += unit.k
        acc = 0
        for m in range(M):
            faults.fire_host(plan.fault, unit.j0 + m)
            s = _step_from_parsed(unit.j0 + m, rows[m], plan)
            st.tokens += 1
            dx.append(s.token_id)
            S["last"] = s.token_id
            steps.append(s)
            if m == M - 1 or s.token_id != unit.drafts[m]:
                break
            acc += 1
        st.accepted[acc] += 1
        pol.observe(unit.n, unit.k, acc)
        S["known"] = unit.j0 + acc
        S["j_enc"] = S["known"] + 1
        return steps

    def may_end(steps: list) -> bool:
        """Could releasing these steps end the request (stop strings, an EOS, the last pass)? Then they are released
        before anything more is encoded. A cancel is seen by release(); a unit encoded just before it is settled by
        generate()'s finally."""
        if plan.stop or S["known"] >= N:
            return True
        return not plan.ignore_eos and any(s.token_id in eng.eos for s in steps)

    th = time.perf_counter()
    top_up()
    st.host_s += time.perf_counter() - th
    while q:
        unit = q.pop(0)
        steps = read(unit)  # blocks until the unit is done; the next unit is running
        th = time.perf_counter()
        early = not may_end(steps)
        if early:
            top_up()  # encode before releasing: the host work hides behind the GPU
        st.host_s += time.perf_counter() - th
        for s in steps:
            if release(s):
                st.backoffs = pol.backoffs
                return st
        if not early:
            th = time.perf_counter()
            top_up()
            st.host_s += time.perf_counter() - th
    st.backoffs = pol.backoffs
    return st


def _fallback_loop(eng, plan, passes: _Passes, release) -> bool:
    """Sampled with top_k = 0 or > 20: one pass per sync, the draw on the host (sampler.sample)."""
    import mlx.core as mx

    from jitengine_flint_m2.sampler import logprob_outputs, sample

    token = passes.first_token
    for k in range(1, plan.passes + 1):
        ((_, r),) = passes.encode(1, host_token=token)
        a = np.array(r)
        faults.fire_host(plan.fault, k)
        res = P.parse_result(a)
        if res["err"]:
            raise faults.GPUSyncError(
                f"megakernel decode pass {k} failed in the kernel: {P.describe_err(res['err'])}"
            )
        if not res["finite"]:
            raise faults.NonFiniteLogits(f"nonfinite logits at completion index {k}")
        logits = eng.mk.logits_view()
        tarr = sample(logits, plan.temperature, plan.top_k, plan.top_p, uniform(plan.seed, k))
        lp = ids = vals = None
        if plan.logprobs is not None:
            lp, ids, vals = logprob_outputs(logits, tarr, plan.logprobs)
            mx.eval(tarr, lp, *(x for x in (ids, vals) if x is not None))
        else:
            mx.eval(tarr)
        token = int(tarr.item())
        top = list(zip(ids.tolist(), vals.tolist())) if ids is not None else []
        s = StepOutput(
            k, token, float(lp.item()) if lp is not None else None, top, time.perf_counter()
        )
        if release(s):
            return True
    return True
