"""Host side of the decode megakernel: compile, bind, encode passes, probe, and the chain reference mode.

Binding: the three weight banks (base arrays: a sliced input would be copied by ensure_row_contiguous on every
call), the packed descriptor table, the six consolidated history arrays, the persistent scratch (written in
place by the kernel), the per-pass parameters, and the previous pass's result. MLX cannot see the in-place
writes, so the previous result is what orders passes: every pass takes it as an input, in every mode.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from jitengine_flint_m4pro.kernels import metal_src as MS
from jitengine_flint_m4pro.kernels import plan as P
from jitengine_flint_m4pro.kernels.mathmode import METAL_COMPILE_OPTIONS


class BarrierProbeError(RuntimeError):
    """The persistent grid could not synchronise (threadgroups not co-resident): refuse, never hang."""


# Residency probe (M4 Pro, G x 256 threads):
#   G = 40 (production): barrier-only pass 0.75-0.93 ms, flag ring of 8 laps 0.62-0.79 ms (64 laps 1.6-1.9 ms), real
#            passes 7.0 ms each with no error at spin 2^21 or 2^17;
#   G = 64 (not co-resident): barrier-only pass 16.5 ms and ring 16.4 ms (8 or 64 laps), both WITHOUT an error
#            word (the late threadgroups only run after a ~16 ms GPU time slice), and the first real pass times out
#            (barrier word) after 0.4-0.6 s at spin 2^21.
#   with the M = 1 pass constants read from a threadgroup copy (metal_src PK, measured on the M4 Pro):
#            G = 64 real passes COMPLETE with no error word (~24 ms each), and the flag ring at G = 64 measured 1.3-42 ms
#            (under its bound on some tries), so neither check refuses G = 64. Host-timed barrier-only passes do not
#            separate the grids either: at SPIN_PROBE a G = 64 dispatch took 12-19 ms in one session and 0.9-1.4 ms (no
#            error word, 9 of 20) in another, while G = 40 took 0.27-0.31 ms warm but 0.8-1.2 ms on a cold GPU. What
#            separates them is the arrival skew, which a short spin bound measures in the kernel: barrier-only passes at
#            spin 2^11 (~0.3 ms) never timed out at G = 40 (540 passes, 81,000 barrier waits) and always at G = 64 (90 of
#            90; 88 of 90 at 2^13). A G = 40 barrier waits ~1 us (and the dispatch ramp tens of us).
BAR_PASSES = 8  # chained barrier-only passes (mask 0: every grid barrier of the pass, no phase) ...
SPIN_BAR = (
    1 << 11
)  # ... every barrier wait bounded at this spin (~0.3 ms; a G = 40 wait is ~1 us) ...
BAR_MS_MAX = (
    4.0  # ... and the chain within this bound per pass (G = 40 <= 0.4; a paused, time-sliced
)
# grid accrues no spins but >= 12 ms of wall time per dispatch)
RING_LAPS = 8  # laps of the flag ring through every threadgroup ...
RING_MS_MAX = 8.0  # ... within this bound (10x G = 40; half of one time slice)
REAL_PASSES = 4  # chained real decode passes at positions 0.. after the ring ...
REAL_MS_MAX = 100.0  # ... within this bound per pass (14x G = 40) ...
SPIN_REAL = 1 << 21  # ... at this bounded spin per wait (~0.4-0.8 s), a quarter of SPIN_DEFAULT
PROBE_TRIES = (
    3  # each check passes on its first good try: a transient GPU co-tenant (e.g. one display
)
# frame) does not refuse a resident grid; a non-resident grid fails every try
PROBE_BACKOFF_S = (
    0.05  # pause after a failed try, so the next one does not land in the same transient
)
VERIFY_ROWS = tuple(range(2, P.VMAX + 1))  # the row counts M of the verify variants


class MegaKernel:
    def __init__(
        self,
        W,
        G: int = P.G_DEFAULT,
        log=print,
        kb: int = 1,
        hg: int = 4,
        sabl: int = 0,
        diag: int = 0,
        vxs: int = P.VXS,
        kv8: bool = False,
        pmask: int = 0,
    ):
        import mlx.core as mx

        if G % 4 or not 16 <= G <= P.MAX_SG // 8:
            raise ValueError(
                f"grid of {G} threadgroups: must be a multiple of 4 in 16..{P.MAX_SG // 8} (the MoE "
                f"block's shared team is 10 threadgroups and its router runs on the others)"
            )
        self.W = W
        self.cfg = W.cfg
        self.G = G
        self.kb = int(kb)  # strided attention: keys per load batch (compile-time)
        self.hg = int(hg)  # strided attention: q heads per task (compile-time)
        # sabl, diag: benchmarks/kernels/megakernel.py only (0 = production)
        self.sabl = int(sabl)  # full-layer stream timing ablation
        self.diag = int(diag)  # cold/warm phase diagnostic
        self.vxs = int(
            vxs
        )  # verify passes: bf16 elements of x staged per threadgroup (compile-time)
        self.kv8 = bool(
            kv8
        )  # kv8 profile (kv8_src.py): bind kv_cache.MKHistory8's arrays; layout 6d required
        # persistent M = 1 kernel: 0 = PM1 (production; no per-phase mask test), 1 = the general kernel (phase-mask
        # arms and timing ablations: benchmarks/kernels/megakernel.py). diag and sabl kernels are general.
        self.pmask = 1 if (self.diag or self.sabl) else int(pmask)
        assert self.pmask in (0, 1)
        self.log = log
        self.banks = list(W.banks)
        assert len(self.banks) == 3  # base arrays from load_weights (never slices)
        self.eo = P.expert_record_offsets(W.plan)
        self.desc = mx.array(P.pack_descriptors(W.plan))
        self.scratch = mx.zeros((P.MK_SCRATCH_BYTES,), dtype=mx.uint8)
        self.dummy = mx.zeros((P.RES_WORDS,), dtype=mx.uint32)
        self.ph0 = mx.zeros((2,), dtype=mx.uint32)
        mx.eval(self.desc, self.scratch, self.dummy, self.ph0)
        self._kernels: dict = {}
        self._mem_checked = False
        self.schedule = P.phase_schedule(self.cfg.dense_layers)
        self.nbar = P.n_barriers(self.cfg.dense_layers)  # grid barriers of a persistent pass
        self.vscratch = None  # verify passes: VMAX row regions of the M = 1 scratch layout (lazy)
        self.vrows: tuple = ()  # verify row counts compiled and probed (warm_verify)
        self.probe_trace: list = []  # every probe try: (check, ms per pass, error word) (probe / warm_verify)

    # ------------------------------------------------------------------ compile
    def kernel(self, chain: bool, mr: int = 1):
        """The M = 1 pass (mr = 1; persistent: PM1 unless pmask = 1) or the mr-row verify pass."""
        import mlx.core as mx

        k = self._kernels.get((chain, mr))
        if k is None:
            if mr == 1 and not self.kv8:
                h, s = MS.render(
                    self.cfg,
                    self.eo,
                    chain,
                    self.kb,
                    self.hg,
                    self.sabl,
                    self.diag,
                    pmask=None if chain else self.pmask,
                )
            elif mr == 1:
                h, s = MS.render(
                    self.cfg,
                    self.eo,
                    chain,
                    self.kb,
                    self.hg,
                    self.sabl,
                    self.diag,
                    kv8=True,
                    pmask=None if chain else self.pmask,
                )
            else:
                h, s = MS.render(
                    self.cfg,
                    self.eo,
                    chain,
                    self.kb,
                    self.hg,
                    mr=mr,
                    G=self.G,
                    xs=self.vxs,
                    kv8=self.kv8,
                )
            tag = hashlib.sha256((h + s).encode()).hexdigest()[:12]
            kind = (
                ("chain" if chain else "persist")
                + ("" if mr == 1 else f"_v{mr}")
                + ("_kv8" if self.kv8 else "")
            )
            k = mx.fast.metal_kernel(
                name=f"flint_mk_{kind}_{tag}",
                input_names=MS.INPUTS,
                output_names=["res"],
                header=h,
                source=s,
                ensure_row_contiguous=True,
                compile_options=METAL_COMPILE_OPTIONS,
            )
            self._kernels[(chain, mr)] = k
        return k

    # ------------------------------------------------------------------ passes
    def _inputs(self, hist_arrays, prm, ph, prev, dep):
        return [*self.banks, self.desc, *hist_arrays, self.scratch, prm, ph, prev, dep]

    def _check_kv8(self, prm_np: np.ndarray) -> None:
        """kv8: every pass that runs attention must use layout 6d on the full layers (the only full-layer path that
        reads the 8-bit history; the strided / layout-2 full paths would read it as bf16)."""
        if (
            self.kv8
            and (int(prm_np[P.P_MASK]) >> P.PH["ATT"]) & 1
            and (prm_np[P.P_6T] == 0 or prm_np[P.P_PF] != 0)
        ):
            raise ValueError(
                "kv8 profile: full-layer attention needs layout 6d (six_t > 0, no strided full parts)"
            )

    def _check_mask(self, prm_np: np.ndarray) -> None:
        """The PM1 kernel (pmask 0) has no per-phase mask test: it runs every phase (MASK_ALL) or, with mask 0, only
        the pass's grid barriers. Any other mask (a phase arm or a timing ablation) needs the general kernel, so it is
        refused here instead of silently running the whole pass."""
        m = int(prm_np[P.P_MASK])
        if not self.pmask and m not in (P.MASK_ALL, 0):
            raise ValueError(
                f"phase mask {m:#x} on the production M = 1 kernel (PM1 runs every phase or, with mask 0, "
                f"only the grid barriers): build MegaKernel(pmask=1) for phase-mask and ablation arms"
            )

    def run_pass(self, hist_arrays, prm_np: np.ndarray, prev=None, dep=None):
        """Encode one persistent pass (lazy); returns its result array [RES_WORDS] uint32.
        dep: the diagnostic table of benchmarks/kernels/megakernel.py (diag kernels only); production passes the zero
        dummy."""
        import mlx.core as mx

        self._check_mask(prm_np)
        self._check_kv8(prm_np)
        prm = mx.array(prm_np)
        prev = self.dummy if prev is None else prev
        dep = self.dummy if dep is None else dep
        return self.kernel(False)(
            inputs=self._inputs(hist_arrays, prm, self.ph0, prev, dep),
            grid=(self.G * P.TPG, 1, 1),
            threadgroup=(P.TPG, 1, 1),
            output_shapes=[(P.RES_WORDS,)],
            output_dtypes=[mx.uint32],
            init_value=0,
        )[0]

    def run_pass_chain(self, hist_arrays, prm_np: np.ndarray, prev=None, after_phase=None):
        """The same pass as one dispatch per phase (race-free reference). Each dispatch takes the previous
        dispatch's output as a dependency input (the scratch writes are invisible to MLX).
        after_phase(phase, layer, out) is called after each dispatch is ENCODED (it may evaluate it)."""
        import mlx.core as mx

        self._check_kv8(prm_np)
        prm = mx.array(prm_np)
        prev = self.dummy if prev is None else prev
        k = self.kernel(True)
        dep = prev
        out = None
        for ph, layer in self.schedule:
            pha = mx.array([ph, layer], dtype=mx.uint32)
            out = k(
                inputs=self._inputs(hist_arrays, prm, pha, prev, dep),
                grid=(self.G * P.TPG, 1, 1),
                threadgroup=(P.TPG, 1, 1),
                output_shapes=[(P.RES_WORDS,)],
                output_dtypes=[mx.uint32],
                init_value=0,
            )[0]
            dep = out
            if after_phase is not None:
                after_phase(ph, layer, out)
        return out

    # ------------------------------------------------------------------ verify passes (mode `code`)
    def ensure_vscratch(self):
        import mlx.core as mx

        if self.vscratch is None:
            self.vscratch = mx.zeros((P.VSCRATCH_BYTES,), dtype=mx.uint8)
            mx.eval(self.vscratch)
        return self.vscratch

    def _vinputs(self, hist_arrays, prm, ph, prev, dep):
        return [*self.banks, self.desc, *hist_arrays, self.ensure_vscratch(), prm, ph, prev, dep]

    def run_vpass(self, hist_arrays, prm_np: np.ndarray, prev=None):
        """Encode one verify pass of M = prm[P_ROWS] rows (lazy); returns its result array [VRES_WORDS]."""
        import mlx.core as mx

        M = int(prm_np[P.P_ROWS])
        self._check_kv8(prm_np)
        prm = mx.array(prm_np)
        prev = self.dummy if prev is None else prev
        return self.kernel(False, M)(
            inputs=self._vinputs(hist_arrays, prm, self.ph0, prev, self.dummy),
            grid=(self.G * P.TPG, 1, 1),
            threadgroup=(P.TPG, 1, 1),
            output_shapes=[(P.VRES_WORDS,)],
            output_dtypes=[mx.uint32],
            init_value=0,
        )[0]

    def run_vpass_chain(self, hist_arrays, prm_np: np.ndarray, prev=None):
        """The verify pass as one dispatch per phase (race-free reference of run_vpass)."""
        import mlx.core as mx

        M = int(prm_np[P.P_ROWS])
        self._check_kv8(prm_np)
        prm = mx.array(prm_np)
        prev = self.dummy if prev is None else prev
        k = self.kernel(True, M)
        dep = prev
        out = None
        for ph, layer in self.schedule:
            pha = mx.array([ph, layer], dtype=mx.uint32)
            out = k(
                inputs=self._vinputs(hist_arrays, prm, pha, prev, dep),
                grid=(self.G * P.TPG, 1, 1),
                threadgroup=(P.TPG, 1, 1),
                output_shapes=[(P.VRES_WORDS,)],
                output_dtypes=[mx.uint32],
                init_value=0,
            )[0]
            dep = out
        return out

    def warm_verify(self, hist_arrays, att: dict, cap: int, rows=VERIFY_ROWS) -> dict:
        """Compile every verify variant and prove its grid co-resident the way probe() proves the M = 1 pass's (the
        chained barrier-only passes, the flag ring, then one real verify pass of M rows at positions 0 .. M - 1 at
        SPIN_REAL, each against its bound), so no request pays a compile and a variant that cannot run co-resident is
        never used. Returns {M: ms of the real pass}."""
        import mlx.core as mx

        out = {}
        for M in rows:
            kw = dict(
                pos=0,
                tokens=[0] * M,
                sel_mode=0,
                top_k=1,
                top_p=1.0,
                temperature=0.0,
                us=[0.0] * M,
                poison_rows=0,
                cap=cap,
                theta=self.cfg.rope_theta,
                **att,
            )
            mx.eval(
                self.run_vpass(
                    hist_arrays, P.make_vparams(mask=0, spin=P.SPIN_PROBE, probe=1, **kw)
                )
            )  # compile

            # bar(), ring() and real() run within this iteration, so they see its kw
            def bar():
                r = None
                for _ in range(BAR_PASSES):
                    r = self.run_vpass(
                        hist_arrays,
                        P.make_vparams(mask=0, spin=SPIN_BAR, **kw),  # noqa: B023
                        r,
                    )
                return r

            def ring():
                return self.run_vpass(
                    hist_arrays,
                    P.make_vparams(mask=0, spin=P.SPIN_PROBE, probe=RING_LAPS, **kw),  # noqa: B023
                )

            def real():
                return self.run_vpass(hist_arrays, P.make_vparams(spin=SPIN_REAL, **kw))  # noqa: B023

            self._check(
                bar,
                BAR_PASSES,
                P.VR_ERR,
                BAR_MS_MAX,
                f"verify pass M={M}: {BAR_PASSES} barrier-only passes",
            )
            self._check(
                ring, 1, P.VR_ERR, RING_MS_MAX, f"verify pass M={M}: flag ring of {RING_LAPS} laps"
            )
            out[M] = round(
                self._check(real, 1, P.VR_ERR, REAL_MS_MAX, f"verify pass M={M}: one real pass"), 1
            )
        self.vrows = tuple(sorted(set(self.vrows) | set(out)))
        mx.eval(self.ensure_vscratch())
        return out

    def vlogits_view(self, m: int):
        """bf16 [vocab] logits of row m of the last evaluated verify pass."""
        import mlx.core as mx

        o = m * P.SCRATCH_BYTES + P.OFF["LOGITS"]
        return self.vscratch[o : o + P.VOCAB * 2].view(mx.bfloat16)

    # ------------------------------------------------------------------ checks
    def _check(self, encode, n: int, err_word: int, bound_ms: float, what: str) -> float:
        """Run `encode()` (n dependent passes per call, the last one's result returned) up to PROBE_TRIES times; pass on
        the first try with no error word and n passes within n x bound_ms. Returns its ms per pass, or refuses."""
        tries = []
        for _ in range(PROBE_TRIES):
            t0 = time.perf_counter()
            a = np.array(encode())
            ms = (time.perf_counter() - t0) * 1e3 / n
            err = int(a[err_word])
            self.probe_trace.append((what, round(ms, 3), err))
            if not err and ms <= bound_ms:
                return ms
            tries.append(
                P.describe_err(err) if err else f"{ms:.1f} ms, over its {bound_ms:.0f} ms bound"
            )
            if len(tries) < PROBE_TRIES:
                time.sleep(PROBE_BACKOFF_S)
        raise BarrierProbeError(
            f"megakernel residency probe failed at G={self.G}: {what}: {'; '.join(tries)}. The "
            f"{self.G} threadgroups do not run co-resident on this GPU; refusing the persistent decode "
            f"path (it would only complete as the GPU time-slices threadgroups, ~750x slower)"
        )

    def probe(self, hist_arrays, att: dict, cap: int) -> dict:
        """Prove the G threadgroups co-resident before any request, or refuse (BarrierProbeError). No error word
        proves it (measured on the M4 Pro): at G = 64 a barrier-only pass and the flag ring complete with none, and so
        do real passes once the M = 1 pass reads its constants from threadgroup memory; such a grid only completes as
        the GPU time-slices its threadgroups. So the
        refusal rests on a short bounded wait and on time, both measured on this GPU. Three checks on the persistent
        kernel itself (its resources, its waits), each passing on its first good try of PROBE_TRIES (no error word,
        within its bound; PROBE_BACKOFF_S between tries):
          bar   BAR_PASSES chained barrier-only passes (mask 0: every grid barrier of the pass, no phase), every
                barrier wait bounded at SPIN_BAR (~0.3 ms: a threadgroup that is not resident with the others arrives
                later, and its barrier times out) and the chain within BAR_MS_MAX per pass (a time-sliced grid);
          ring  a flag ring of RING_LAPS laps through all G threadgroups (pass_entry -> probe_ring; SPIN_PROBE):
                G dependent hand-offs per lap, within RING_MS_MAX;
          real  REAL_PASSES chained real decode passes at positions 0.. (every phase and every head-flag, task-counter,
                team and barrier wait) at SPIN_REAL, no error word, within REAL_MS_MAX per pass.
        Every try is recorded in probe_trace. The history rows the real passes write are left invalid (the caller's
        history is empty at startup). Returns {"bar_ms", "ring_ms", "real_ms"}."""
        import mlx.core as mx

        def prm(pos, token, **k):
            return P.make_params(
                pos=pos,
                token=token,
                forced_next=None,
                sel_mode=0,
                top_k=1,
                top_p=1.0,
                temperature=0.0,
                u=0.0,
                poison=False,
                cap=cap,
                theta=self.cfg.rope_theta,
                **att,
                **k,
            )

        def bar():
            r = None
            for _ in range(BAR_PASSES):
                r = self.run_pass(hist_arrays, prm(0, 0, mask=0, spin=SPIN_BAR), r)
            return r

        def ring():
            return self.run_pass(hist_arrays, prm(0, 0, mask=0, spin=P.SPIN_PROBE, probe=RING_LAPS))

        def real():
            r = None
            for k in range(REAL_PASSES):
                r = self.run_pass(hist_arrays, prm(k, 0 if k == 0 else None, spin=SPIN_REAL), r)
            return r

        mx.eval(
            self.run_pass(hist_arrays, prm(0, 0, mask=0, spin=P.SPIN_PROBE, probe=1))
        )  # compile + first run
        bar_ms = self._check(
            bar,
            BAR_PASSES,
            P.R_ERR,
            BAR_MS_MAX,
            f"{BAR_PASSES} barrier-only passes ({self.nbar} grid barriers each)",
        )
        ring_ms = self._check(ring, 1, P.R_ERR, RING_MS_MAX, f"flag ring of {RING_LAPS} laps")
        real_ms = self._check(real, REAL_PASSES, P.R_ERR, REAL_MS_MAX, f"{REAL_PASSES} real passes")
        return {
            "bar_ms": round(bar_ms, 3),
            "ring_ms": round(ring_ms, 3),
            "real_ms": round(real_ms, 2),
        }

    def check_memory_once(self, before: int) -> None:
        """After the first real pass: active memory must not have grown by a bank-sized copy."""
        import mlx.core as mx

        if self._mem_checked:
            return
        self._mem_checked = True
        grew = mx.get_active_memory() - before
        if grew > 256 * 2**20:
            raise RuntimeError(
                f"active memory grew by {grew / 1e9:.2f} GB over the first megakernel pass: an input "
                f"was copied (bind base arrays only)"
            )

    def logits_view(self):
        """bf16 [vocab] logits of the last evaluated pass (a view of the scratch; read only after a sync)."""
        import mlx.core as mx

        o = P.OFF["LOGITS"]
        return self.scratch[o : o + P.VOCAB * 2].view(mx.bfloat16)
