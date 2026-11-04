"""Kernel-level timing of the decode megakernel on real weights with a random history at length L.

    python benchmarks/kernels/megakernel.py --model DIR --L 1536 33280 66048 --out <json> [--chunks grid,c4] [--G 40]

Arms are phase masks of the SAME persistent kernel (the grid barriers always run; on a sparse layer RSG =
the router rows, EGU = every gate/up piece, DOWN = every DOWN piece of the fused MoE block): full pass, no attention
(ATT+MERGE off), attention only (ATT+MERGE), barriers only, full pass without the full-layer attention items
(nofull: full - nofull = the in-kernel full-layer attention time), and every phase alone. T dependent passes per
evaluation (each takes the previous result), median of reps; position L-1, so attention reads L keys per full
layer. Per-phase GB/s uses that phase's weight (or KV) bytes over (phase alone - barriers only).

--twice (cold/warm diagnostic, at the first L only): a second kernel compiled with diag=1 runs, in interleaved
rounds, the full pass, the full pass without phase X ("once" = full - minus_X), the full pass with X run twice
back to back in its slot ("warm" = twice_X - full: code and data warm), and with the second run on the next layer
of the same kind ("warm_next" = twnext_X - full: code warm, that layer's weights not yet touched this pass).
cold excess per phase instance = once - warm(_next); reported in us per instance.

Every arm runs the general M = 1 kernel (MegaKernel(pmask=1), which has per-phase mask tests), so "full" is that
kernel's full pass; the production kernel has no mask test and refuses a partial mask. --twice's
prod_full arm is the production kernel.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

from jitengine_flint_m4pro.kernels import plan as P
MB = 1e6


def phase_bytes(L: int, kv_row: int = 2048) -> dict:
    """Bytes each phase reads per pass (weights at 1152 B per 2048-wide int4 row incl. scales/biases; kv_row = full-layer
    K + V bytes per key per layer: 2048 bf16, 1088 kv8)."""
    row = lambda k: k // 2 + k // 16  # noqa: E731  bytes of one int4 g64 row (weights + bf16 scales + biases)
    full_kv = 14 * L * kv_row
    ring_kv = 35 * min(L, P.RING) * 2048
    return {
        "QKV": 49 * 5120 * row(2048),
        "ATT": full_kv + ring_kv,
        "MERGE": 0,
        "O": 49 * 2048 * row(4096),
        "RSG": 47 * 256 * 4096,
        "EGU": 47 * (6 * 1024 + 2048) * row(2048),
        "DOWN": 47 * (6 * 2048 * row(512) + 2048 * row(1024)),
        "DGU": 2 * 12288 * row(2048),
        "DDOWN": 2 * 2048 * row(6144),
        "HEAD": P.VOCAB * row(2048),
        "SELECT": 0,
    }


def main(argv=None) -> int:  # noqa: C901, PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, nargs="+", default=[1536])
    ap.add_argument("--G", type=int, default=P.G_DEFAULT)
    ap.add_argument(
        "--chunks",
        default=None,
        help="comma list of partition policies (first = main arms; default: "
        "the profile's, plan.default_policy); "
        "L<C>x<NSUB>a<ASG>f<FLAGS> = the long full-layer stream with those parameters",
    )
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="full / noATT / attonly / barriers only")
    ap.add_argument(
        "--full-abl", action="store_true", help="add full-layer-only attention ablation arms (f_*)"
    )
    ap.add_argument("--kb", type=int, default=1, help="strided attention key batch (compile-time)")
    ap.add_argument(
        "--hg", type=int, default=4, help="strided attention q heads per task (compile-time)"
    )
    ap.add_argument(
        "--sabl",
        type=int,
        default=0,
        help="full-layer stream timing ablation (compile-time): "
        "1 = 16 L1-resident K/V rows (no DRAM), 2 = loads only (no MMA / softmax)",
    )
    ap.add_argument(
        "--twice", action="store_true", help="cold/warm diagnostic per phase at the first L"
    )
    ap.add_argument(
        "--rounds", type=int, default=7, help="interleaved rounds of the --twice diagnostic"
    )
    ap.add_argument(
        "--verify",
        default="",
        help="comma list of row counts M: verify-pass arms (M = 1 is the raw "
        "kernel), fed consecutive ids of a real code prompt (realistic routing overlap)",
    )
    ap.add_argument("--verify-only", action="store_true", help="skip the M = 1 phase-mask arms")
    ap.add_argument(
        "--verify-phases", default="", help="comma list of M: every phase alone (only_X arms) per M"
    )
    ap.add_argument(
        "--vxs", type=int, default=P.VXS, help="verify: bf16 elements of x staged per threadgroup"
    )
    ap.add_argument(
        "--profile",
        default="main",
        choices=("main", "kv8"),
        help="kv8: the 8-bit full-layer history kernels over a random quantised history (layout 6d only)",
    )
    a = ap.parse_args(argv)
    kv8 = a.profile == "kv8"
    if a.chunks is None:
        a.chunks = P.default_policy(8 if kv8 else 16)
    if kv8 and (not all(P.is_6d(x) for x in a.chunks.split(",")) or a.twice or a.sabl):
        raise SystemExit(
            "--profile kv8 times layout 6d only (l6dk / l6d2 / l6d; no --twice / --sabl)"
        )
    import mlx.core as mx

    from jitengine_flint_m4pro.kernels.runner import MegaKernel
    from jitengine_flint_m4pro.kv_cache import MKHistory, MKHistory8
    from jitengine_flint_m4pro.loader import load_weights
    from jitengine_flint_m4pro.model.config import load_config, tensor_index

    mx.set_wired_limit(int(mx.device_info()["max_recommended_working_set_size"]))
    cfg = load_config(a.model)
    W = load_weights(cfg, tensor_index(a.model, cfg), log=lambda m: None)
    mk = MegaKernel(
        W, G=a.G, kb=a.kb, hg=a.hg, sabl=a.sabl, vxs=a.vxs, kv8=kv8, pmask=1
    )  # the general kernel
    hist = (MKHistory8 if kv8 else MKHistory)(cfg, 262144)
    policies = a.chunks.split(",")
    R = {
        "argv": sys.argv,
        "G": a.G,
        "T": a.T,
        "reps": a.reps,
        "kb": a.kb,
        "hg": a.hg,
        "sabl": a.sabl,
        "profile": a.profile,
        "results": {},
    }
    kv_row = 1088 if kv8 else 2048  # full-layer K + V bytes per key per layer (4 kv heads)

    def timed(L, mask, policy, kern=None, diag=0, reps=None, dep=None):
        kern = kern or mk
        prm = P.make_params(
            pos=L - 1,
            token=None,
            forced_next=None,
            sel_mode=0,
            top_k=1,
            top_p=1.0,
            temperature=0.0,
            u=0.0,
            poison=False,
            cap=hist.capacity,
            theta=cfg.rope_theta,
            mask=mask,
            **P.attn_params(policy, L + 1),
        )
        prm[P.P_TOKMODE] = 0
        prm[P.P_TOKEN] = 1000
        prm[P.P_DIAG] = diag
        arrs = hist.arrays

        def burst():
            prev = None
            for _ in range(a.T):
                prev = kern.run_pass(arrs, prm, prev, dep)
            return prev

        r = burst()
        mx.eval(r)
        if np.array(r)[P.R_ERR]:
            raise RuntimeError("barrier timeout during timing")
        xs = []
        for _ in range(a.reps if reps is None else reps):
            t0 = time.perf_counter()
            mx.eval(burst())
            xs.append((time.perf_counter() - t0) * 1e3 / a.T)
        return statistics.median(xs), xs

    vtoks = []
    if a.verify:
        pf = os.path.join(
            os.path.expanduser(os.environ.get("FLINT_BENCH_DIR") or "~/flint-bench"), "bench", "prompts", "code", "1024.json"
        )
        vtoks = (
            json.load(open(pf))["ids"][400 : 400 + 64]
            if os.path.exists(pf)
            else list(range(1000, 1064))
        )

    def vtimed(L, M, mask, reps=None):
        """ms per verify pass of M rows at positions L-1 .. L+M-2 (T dependent passes per evaluation)."""
        att = P.attn_params(policies[0], L + P.VMAX + 1)
        if M == 1:
            prm = P.make_params(
                pos=L - 1,
                token=vtoks[0],
                forced_next=None,
                sel_mode=0,
                top_k=1,
                top_p=1.0,
                temperature=0.0,
                u=0.0,
                poison=False,
                cap=hist.capacity,
                theta=cfg.rope_theta,
                mask=mask,
                **att,
            )
            run = lambda prev: mk.run_pass(hist.arrays, prm, prev)  # noqa: E731
            errw = P.R_ERR
        else:
            prm = P.make_vparams(
                pos=L - 1,
                tokens=vtoks[:M],
                sel_mode=0,
                top_k=1,
                top_p=1.0,
                temperature=0.0,
                us=[0.0] * M,
                poison_rows=0,
                cap=hist.capacity,
                theta=cfg.rope_theta,
                mask=mask,
                **att,
            )
            run = lambda prev: mk.run_vpass(hist.arrays, prm, prev)  # noqa: E731
            errw = P.VR_ERR

        def burst():
            prev = None
            for _ in range(a.T):
                prev = run(prev)
            return prev

        r = burst()
        mx.eval(r)
        if np.array(r)[errw]:
            raise RuntimeError("barrier timeout during verify timing")
        xs = []
        for _ in range(a.reps if reps is None else reps):
            t0 = time.perf_counter()
            mx.eval(burst())
            xs.append((time.perf_counter() - t0) * 1e3 / a.T)
        return statistics.median(xs), xs

    def twice_diag(L, policy):
        """Per-phase cold/warm split (see the module docstring); interleaved rounds, median per arm."""
        mkd = MegaKernel(W, G=a.G, kb=a.kb, hg=a.hg, sabl=a.sabl, diag=1)
        mkp = MegaKernel(W, G=a.G, kb=a.kb, hg=a.hg, sabl=a.sabl)  # production (PM1 unless sabl)
        # next layer of the same attention type and MLP kind (cyclic), for the twnext arms
        kind = [(layer in cfg.full_layers, layer in cfg.dense_layers) for layer in range(P.NL)]
        nxt = np.zeros(P.RES_WORDS, dtype=np.uint32)
        for layer in range(P.NL):
            nxt[layer] = next(
                (layer + k) % P.NL
                for k in range(1, P.NL + 1)
                if kind[(layer + k) % P.NL] == kind[layer]
            )
        nxt[P.NL] = P.NL
        dtab = mx.array(nxt)
        per_pass = {
            "QKV": 49,
            "ATT": 49,
            "MERGE": 49,
            "O": 49,
            "RSG": 47,
            "EGU": 47,
            "DOWN": 47,
            "DGU": 2,
            "DDOWN": 2,
            "HEAD": 1,
            "SELECT": 1,
        }
        arms = {
            "prod_full": (mkp, P.MASK_ALL, 0),
            "full": (mkd, P.MASK_ALL, 0),
            "barriers": (mkd, 0, 0),
        }
        for n in P.PHASES:
            b = 1 << P.PH[n]
            arms[f"minus_{n}"] = (mkd, P.MASK_ALL & ~b, 0)
            arms[f"twice_{n}"] = (mkd, P.MASK_ALL, b)
            arms[f"twnext_{n}"] = (mkd, P.MASK_ALL, b | (1 << 16))
        rounds = {k: [] for k in arms}
        for r in range(a.rounds):
            for k, (kern, mask, dg) in arms.items():
                rounds[k].append(
                    timed(L, mask, policy, kern, dg, reps=1, dep=dtab if kern is mkd else None)[0]
                )
            print(
                f"[bench_mk] twice round {r}: full {rounds['full'][-1]:.3f} prod {rounds['prod_full'][-1]:.3f}",
                flush=True,
            )
        med = {k: statistics.median(v) for k, v in rounds.items()}
        out = {
            "next_layer": [int(x) for x in nxt[: P.NL]],
            "arms_ms": {k: round(v, 4) for k, v in med.items()},
            "rounds_ms": {k: [round(x, 4) for x in v] for k, v in rounds.items()},
            "per_phase_us": {},
        }
        for n in P.PHASES:
            cnt = per_pass[n]
            once = (med["full"] - med[f"minus_{n}"]) * 1e3
            warm = (med[f"twice_{n}"] - med["full"]) * 1e3
            warm_n = (med[f"twnext_{n}"] - med["full"]) * 1e3
            out["per_phase_us"][n] = {
                "instances": cnt,
                "once_per_pass": round(once, 1),
                "once": round(once / cnt, 2),
                "warm_same": round(warm / cnt, 2),
                "warm_next": round(warm_n / cnt, 2),
                "cold_excess_same": round((once - warm) / cnt, 2),
                "cold_excess_next": round((once - warm_n) / cnt, 2),
            }
        return out

    probe_done = False
    for L in a.L:
        hist.release()
        hist.reserve(L + 8)
        key = mx.random.key(L)
        for i in range(4):
            sh, ki = hist.full[i].shape, mx.random.split(key, 6)[i]
            if not kv8:
                hist.full[i] = (mx.random.normal(sh, key=ki) * 0.5).astype(mx.bfloat16)
            elif i % 2 == 0:  # kv8 codes: random bytes
                hist.full[i] = mx.random.randint(0, 2**31 - 1, sh, key=ki).astype(mx.uint32) * 3
            else:  # kv8 (scale, bias) pairs: |scale| ~ 0.004, bias ~ -0.5
                hist.full[i] = (
                    mx.random.uniform(-0.5, 0.5, sh, key=ki) * mx.array([0.01, 1.0, 0.01, 1.0])
                ).astype(mx.bfloat16)
        hist.kr = (mx.random.normal(hist.kr.shape, key=mx.random.split(key, 6)[4]) * 0.5).astype(
            mx.bfloat16
        )
        hist.vr = (mx.random.normal(hist.vr.shape, key=mx.random.split(key, 6)[5]) * 0.5).astype(
            mx.bfloat16
        )
        mx.eval(hist.arrays)
        if not probe_done:
            R["probe_ms"] = mk.probe(hist.arrays, P.attn_params(policies[0], L + 1), hist.capacity)
            probe_done = True
        res = {}
        if a.verify:
            Ms = [int(x) for x in a.verify.split(",")]
            noatt = P.MASK_ALL & ~((1 << P.PH["ATT"]) | (1 << P.PH["MERGE"]))
            for M in Ms:
                for name, mask in (("full", P.MASK_ALL), ("noATT", noatt), ("barriers", 0)):
                    med, xs = vtimed(L, M, mask)
                    res[f"verify/M{M}/{name}"] = {
                        "ms": round(med, 4),
                        "reps": [round(x, 4) for x in xs],
                    }
                    print(f"[bench_mk] L={L} verify M={M} {name}: {med:.3f} ms", flush=True)
            for M in [int(x) for x in a.verify_phases.split(",") if x]:
                base = vtimed(L, M, 0)[0]
                for i, ph in enumerate(P.PHASES):
                    med, xs = vtimed(L, M, 1 << i)
                    res[f"verify/M{M}/only_{ph}"] = {
                        "ms": round(med, 4),
                        "above_barriers": round(med - base, 4),
                    }
                    print(
                        f"[bench_mk] L={L} verify M={M} only_{ph}: {med - base:.3f} ms above barriers",
                        flush=True,
                    )
            f1 = res.get("verify/M1/full", {}).get("ms")
            if f1:
                res["verify_ratio"] = {M: round(res[f"verify/M{M}/full"]["ms"] / f1, 4) for M in Ms}
                print(f"[bench_mk] L={L} verify T(M)/T(1): {res['verify_ratio']}", flush=True)
            if a.verify_only:
                R["results"][str(L)] = res
                with open(a.out, "w") as f:
                    json.dump(R, f, indent=1)
                continue
        pb = phase_bytes(L, kv_row)
        arms = {
            "full": P.MASK_ALL,
            "noATT": P.MASK_ALL & ~((1 << P.PH["ATT"]) | (1 << P.PH["MERGE"])),
            "attonly": (1 << P.PH["ATT"]) | (1 << P.PH["MERGE"]),
            "barriers": 0,
            "nofull": P.MASK_ALL | (8 << 16),
        }
        if a.full_abl:
            arms.update(
                {
                    "f_noblk": P.MASK_ALL | ((2 | 16) << 16),
                    "f_noq": P.MASK_ALL | ((1 | 16) << 16),
                    "f_nostore": P.MASK_ALL | ((4 | 16) << 16),
                    "f_skel": P.MASK_ALL | ((7 | 16) << 16),
                }
            )
        if not a.quick:
            arms.update({f"only_{n}": 1 << i for i, n in enumerate(P.PHASES)})
            att = arms["attonly"]
            arms.update(
                {
                    "att_noq": att | (1 << 16),
                    "att_noblk": att | (2 << 16),
                    "att_nostore": att | (4 << 16),
                    "att_skeleton": att | (7 << 16),
                }
            )
        for pol in policies:
            for name, mask in arms.items():
                if (
                    pol != policies[0]
                    and name not in ("full", "attonly", "only_ATT", "only_MERGE", "nofull")
                    and not name.startswith("f_")
                ):
                    continue
                if name.startswith("att_") and L != a.L[0]:
                    continue
                med, xs = timed(L, mask, pol)
                res[f"{pol}/{name}"] = {"ms": round(med, 4), "reps": [round(x, 4) for x in xs]}
                print(f"[bench_mk] L={L} {pol}/{name}: {med:.3f} ms", flush=True)
        fb = 14 * L * kv_row  # full-layer K/V bytes of the pass
        for pol in policies:
            f, nf = res.get(f"{pol}/full"), res.get(f"{pol}/nofull")
            if f and nf and f["ms"] > nf["ms"]:
                res[f"{pol}/full_layer_att"] = {
                    "ms": round(f["ms"] - nf["ms"], 4),
                    "GB_s": round(fb / ((f["ms"] - nf["ms"]) * 1e-3) / 1e9, 1),
                }
        base = res[f"{policies[0]}/barriers"]["ms"]
        if not a.quick:
            per = {}
            for n in P.PHASES:
                k = f"{policies[0]}/only_{n}"
                ms = res[k]["ms"] - base
                per[n] = {
                    "ms": round(ms, 4),
                    "GB_s": round(pb[n] / (ms * 1e-3) / 1e9, 1) if pb[n] and ms > 0 else None,
                    "MB": round(pb[n] / MB, 1),
                }
            res["per_phase_minus_barriers"] = per
            res["sum_phases_plus_barriers"] = round(sum(v["ms"] for v in per.values()) + base, 3)
        if a.twice and L == a.L[0]:
            res["twice"] = twice_diag(L, policies[0])
            print(
                f"[bench_mk] L={L} twice: " + json.dumps(res["twice"]["per_phase_us"]), flush=True
            )
        R["results"][str(L)] = res
        with open(a.out, "w") as f:
            json.dump(R, f, indent=1)
    Ls = sorted(a.L)
    if len(Ls) >= 2 and not a.verify_only:
        lo, hi = Ls[-2], Ls[-1]
        d = (
            R["results"][str(hi)][f"{policies[0]}/full"]["ms"]
            - R["results"][str(lo)][f"{policies[0]}/full"]["ms"]
        )
        R["full_layer_marginal_GB_s"] = (
            round(14 * (hi - lo) * kv_row / (d * 1e-3) / 1e9, 1) if d > 0 else None
        )
    with open(a.out, "w") as f:
        json.dump(R, f, indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != "results"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
