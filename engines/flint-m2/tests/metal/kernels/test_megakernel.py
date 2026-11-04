"""Correctness checks of the decode megakernel on real weights (run alone on the box; ~25 GB).

    JITENGINE_FLINT_M2_TEST_CHECKPOINT=~/models/flint-int4 pytest tests/metal/kernels/test_megakernel.py -s
    python tests/metal/kernels/test_megakernel.py --out <json> --model DIR [--ctx 1000] [--passes 64] [--steps 1024]

1. phase oracle: one pass in chain mode; after every dispatch its outputs are compared with stock MLX ops
   (mx.quantized_matmul, fast.rms_norm, fast.sdpa) applied to the kernel's OWN inputs read back from the scratch
   (errors never compound across phases), in bf16 ulps, plus an explicit fp32 attention reference;
2. h trace: the kernel's per-layer residual at the first decode position vs the graph backend's decode_step
   (model.py) on a history prefilled by the same MLX prefill (the bf16 noise floor);
3. persistent == chain over N passes, every result word, every history byte and the whole scratch row region
   (the MoE team counters after it are synchronisation state, as the barrier words are);
4. greedy 1,024 steps, then the same stream forced twice: forced == greedy, and the two forced runs are
   bit-identical (repeat determinism).

--profile kv8 runs the same checks on the kv8 kernels and history (kv_cache.MKHistory8): the oracle's attention is
referenced to the dequantised 8-bit history and its full-layer K / V rows are checked by
6. the quantise-on-write cross-check: at every full layer of a decode pass, the MAIN kernel's QKV dispatch is replayed on
   the kv8 chain's own scratch and its bf16 K / V rows are quantised by mx.quantize; the kv8 kernel's stored codes,
   scales and biases must equal them bit for bit (and the replay must leave the queries unchanged).
The h trace (2) compares against the bf16 graph decode, so it is skipped for kv8.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

import jitengine_flint_m2
from jitengine_flint_m2.kernels import plan as P

PACKAGE = os.path.dirname(os.path.abspath(jitengine_flint_m2.__file__))


def log(msg: str) -> None:
    print(f"[selftest {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- bf16 helpers
def bits(a) -> np.ndarray:
    """uint16 bit patterns of an MLX bf16 array (flattened)."""
    import mlx.core as mx

    return np.array(a.astype(mx.bfloat16).reshape(-1).view(mx.uint16)).astype(np.uint16)


def f32_of_bits(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << 16).view(np.float32)


def ulps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    def key(x):
        x = x.astype(np.int32)
        return np.where(x & 0x8000, -(x & 0x7FFF), x)

    return np.abs(key(a) - key(b))


def cmp(got: np.ndarray, ref: np.ndarray) -> dict:
    u = ulps(got, ref)
    g, r = f32_of_bits(got), f32_of_bits(ref)
    scale = float(np.max(np.abs(r))) or 1.0
    return {
        "n": int(u.size),
        "max_ulp": int(u.max()),
        "frac_gt1": float(np.mean(u > 1)),
        "frac_ne": float(np.mean(u > 0)),
        "max_abs_rel": float(np.max(np.abs(g - r)) / scale),
        "nan": int(np.isnan(g).sum()),
    }


def bf16_round_bits(x: np.ndarray) -> np.ndarray:
    """fp64 -> bf16 bits, round to nearest even (via fp32)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF).astype(np.uint16)


def dequant_rows(bank0: np.ndarray, plan, prefix: str, rows: int, K: int) -> np.ndarray:
    """fp64 W [rows, K] of an int4 g64 tensor straight from the bank bytes (W = s * q + b)."""

    def raw(name):
        sl = plan.slots[name]
        return bank0[sl.offset : sl.offset + sl.nbytes]

    w = raw(prefix + ".weight").reshape(rows, K // 2)
    sc = f32_of_bits(raw(prefix + ".scales").view(np.uint16).reshape(rows, K // 64)).astype(
        np.float64
    )
    bi = f32_of_bits(raw(prefix + ".biases").view(np.uint16).reshape(rows, K // 64)).astype(
        np.float64
    )
    q = np.stack([w & 15, w >> 4], -1).reshape(rows, K).astype(np.float64)
    return q * np.repeat(sc, 64, 1) + np.repeat(bi, 64, 1)


def mxbf(b: np.ndarray):
    import mlx.core as mx

    return mx.array(b.astype(np.uint16)).view(mx.bfloat16)


# ---------------------------------------------------------------- harness
class T:
    def __init__(self, a):
        import mlx.core as mx

        from jitengine_flint_m2.kernels.runner import MegaKernel
        from jitengine_flint_m2.kv_cache import History, MKHistory
        from jitengine_flint_m2.loader import load_weights
        from jitengine_flint_m2.model.config import load_config, tensor_index
        from jitengine_flint_m2.tokenizer import Tokenizer

        mx.set_wired_limit(int(mx.device_info()["max_recommended_working_set_size"]))
        self.a = a
        self.kv8 = a.profile == "kv8"
        self.cfg = load_config(a.model)
        self.W = load_weights(self.cfg, tensor_index(a.model, self.cfg), log=log)
        tok = Tokenizer(a.model)
        text = "".join(open(f).read() for f in sorted(glob.glob(os.path.join(PACKAGE, "*.py"))))
        ids = tok.encode(text)
        self.prompt = (ids * -(-a.ctx // len(ids)))[
            : a.ctx
        ]  # the source repeated up to --ctx (long-path checks)
        self.n = len(self.prompt)
        from jitengine_flint_m2.kv_cache import MKHistory8

        self.hist = (MKHistory8 if self.kv8 else MKHistory)(self.cfg, 262144)
        self.hist.reserve(self.n + a.steps + 16)
        self.mk = MegaKernel(self.W, G=a.G, log=log, kv8=self.kv8)
        self.names = (
            ["Kq", "Ksb", "Vq", "Vsb"] if self.kv8 else ["K_lo", "K_hi", "V_lo", "V_hi"]
        ) + ["K_ring", "V_ring"]
        self.History = History
        self.att = P.attn_params(a.chunks, self.n + a.steps + 1)
        log(f"prompt {self.n} tokens, capacity {self.hist.capacity}, attention {self.att}, G {a.G}")

    def prefill(self, hist):
        import mlx.core as mx

        from jitengine_flint_m2.model.model import prefill_chunk

        hist.invalidate()
        lg = None
        for c0 in range(0, self.n, 2048):
            c1 = min(self.n, c0 + 2048)
            lg = prefill_chunk(
                self.W, hist, mx.array(self.prompt[c0:c1], dtype=mx.int32), c0, [c1 - c0 - 1]
            )
            mx.eval(lg, hist.state_arrays())
        hist.length, hist.valid = self.n, True
        return lg[0]

    def prm(self, k: int, token, *, forced_next=None, mask=P.MASK_ALL):
        return P.make_params(
            pos=self.n + k - 1,
            token=token,
            forced_next=forced_next,
            sel_mode=0,
            top_k=1,
            top_p=1.0,
            temperature=0.0,
            u=0.0,
            poison=False,
            cap=self.hist.capacity,
            theta=self.cfg.rope_theta,
            mask=mask,
            **self.att,
        )

    def sbits(self, name: str, off_elems: int, count: int, size: int = 2) -> np.ndarray:
        o = P.OFF[name] + off_elems * size
        raw = np.array(self.mk.scratch[o : o + count * size])
        return raw.view(np.uint16 if size == 2 else np.uint32).copy()

    def save_state(self):
        import mlx.core as mx

        from jitengine_flint_m2.kv_cache import _copy

        s = [_copy(a) for a in self.hist.arrays]
        mx.eval(s)
        return s

    def restore_state(self, s):
        import mlx.core as mx

        from jitengine_flint_m2.kv_cache import _copy

        new = [_copy(a) for a in s]
        mx.eval(new)
        self.hist.full = new[:4]
        self.hist.kr, self.hist.vr = new[4], new[5]
        self.mk.scratch = mx.zeros((P.MK_SCRATCH_BYTES,), dtype=mx.uint8)
        mx.eval(self.mk.scratch)


# ---------------------------------------------------------------- 1. phase oracle
def phase_oracle(t: T, token: int) -> dict:  # noqa: C901, PLR0915
    import mlx.core as mx

    from jitengine_flint_m2.model.model import embed, qmm, swiglu

    W, cfg, hist = t.W, t.cfg, t.hist
    p = t.n
    eps = cfg.rms_eps
    out: dict = {
        k: []
        for k in (
            "EMB",
            "Q",
            "K",
            "V",
            "ATT_sdpa",
            "ATT_fp32",
            "O",
            "O_vs_exact_kernel",
            "O_vs_exact_mlx",
            "ROUTER",
            "SHARED_ACT",
            "TOPK",
            "EXPERT_ACT",
            "DOWN_H",
            "DENSE_ACT",
            "DDOWN_H",
            "LOGITS",
        )
    }
    exact: dict = {}
    pending: dict = {}  # layer -> expected h bits (the MoE output is stored by the next prologue)
    cs, sn = P.rope_table(p, cfg.rope_theta)
    cs, sn = mx.array(cs), mx.array(sn)

    def rope_ref(x):  # x [H, 128] bf16 -> R's exact-mode rotation
        xf = x.astype(mx.float32)
        x1, x2 = xf[:, :64], xf[:, 64:]
        return mx.concatenate([x1 * cs - x2 * sn, x2 * cs + x1 * sn], axis=-1).astype(mx.bfloat16)

    def hist_row(layer, isv, kvh):
        if layer in cfg.full_layers:
            f = cfg.full_layers.index(layer)
            a = hist.full[(2 if isv else 0) + f // 7]
            return bits(a[f % 7, kvh, p, :])
        s = cfg.sliding_layers.index(layer)
        return bits((hist.vr if isv else hist.kr)[s, kvh, p % P.RING, :])

    def h_of(layer):
        return t.sbits("HS", layer * 2048, 2048)

    def mid(layer):
        hm = mxbf(h_of(layer)).astype(mx.float32) + mxbf(t.sbits("O", layer * 2048, 2048)).astype(
            mx.float32
        )
        return hm.astype(mx.bfloat16)

    def topk_np(probs):
        ids = np.arange(256)
        order = np.lexsort((ids, -probs))
        sel = order[:6]
        ps = probs[sel].astype(np.float32)
        S = np.float32(0.0)
        for r in range(6):
            S = np.float32(S + ps[r])
        S = np.float32(S + np.float32(1e-20))
        return sel, (ps / S).astype(np.float32)

    def after(ph, layer, res):  # noqa: C901, PLR0915
        mx.eval(res)
        name = P.PHASES[ph]
        lw = W.layers[layer] if layer < cfg.n_layers else None
        if (
            name in ("QKV", "HEAD") and layer in pending
        ):  # h_l of a sparse layer l - 1, combined by this prologue
            out["DOWN_H"].append(cmp(h_of(layer), pending.pop(layer)))
        if name == "QKV":
            if layer == 0:
                out["EMB"].append(cmp(h_of(0), bits(embed(W, mx.array([[token]], dtype=mx.int32)))))
            x = mx.fast.rms_norm(mxbf(h_of(layer))[None], lw.ln1, eps)
            q = mx.fast.rms_norm(qmm(x, lw.q).reshape(32, 128), lw.q_norm, eps)
            k = mx.fast.rms_norm(qmm(x, lw.k).reshape(4, 128), lw.k_norm, eps)
            v = qmm(x, lw.v).reshape(4, 128)
            if lw.sliding:
                q, k = rope_ref(q), rope_ref(k)
            out["Q"].append(cmp(t.sbits("Q", layer * 4096, 4096), bits(q)))
            if not (
                t.kv8 and layer in cfg.full_layers
            ):  # kv8 full-layer rows: quant_xcheck (exact) instead
                out["K"].append(
                    cmp(np.concatenate([hist_row(layer, False, h) for h in range(4)]), bits(k))
                )
                out["V"].append(
                    cmp(np.concatenate([hist_row(layer, True, h) for h in range(4)]), bits(v))
                )
        elif name == "MERGE":
            q = mxbf(t.sbits("Q", layer * 4096, 4096)).reshape(1, 32, 1, 128)
            if lw.sliding:
                s = cfg.sliding_layers.index(layer)
                ns = min(p + 1, P.RING)
                K, V = hist.kr[s : s + 1, :, :ns, :], hist.vr[s : s + 1, :, :ns, :]
                keep = ((p - np.arange(ns)) % P.RING) < P.WINDOW
            else:
                K, V = hist.full_kv(layer, p + 1)
                keep = np.ones(p + 1, dtype=bool)
            mask = None if keep.all() else mx.array(keep.reshape(1, 1, 1, -1))
            ref = mx.fast.scaled_dot_product_attention(
                q, K, V, scale=1.0 / math.sqrt(128), mask=mask, sinks=lw.sinks
            )
            got = t.sbits("ATTN", layer * 4096, 4096)
            out["ATT_sdpa"].append(cmp(got, bits(ref)))
            # explicit fp32 reference (R's form): s = (scale q) . k, one softmax with the sink
            qf = q.reshape(4, 8, 128).astype(mx.float32) * (1.0 / math.sqrt(128))
            Kf, Vf = K[0].astype(mx.float32), V[0].astype(mx.float32)
            S = qf @ mx.swapaxes(Kf, 1, 2)
            S = mx.where(mx.array(keep)[None, None, :], S, -mx.inf)
            sink = lw.sinks.astype(mx.float32).reshape(4, 8, 1)
            m = mx.maximum(mx.max(S, axis=-1, keepdims=True), sink)
            Pm = mx.exp(S - m)
            o = (Pm @ Vf) / (mx.sum(Pm, axis=-1, keepdims=True) + mx.exp(sink - m))
            out["ATT_fp32"].append(cmp(got, bits(o.reshape(-1))))
        elif name == "O":
            xb = t.sbits("ATTN", layer * 4096, 4096)
            yk = t.sbits("O", layer * 2048, 2048)
            ym = bits(qmm(mxbf(xb)[None], lw.o))
            out["O"].append(cmp(yk, ym))
            if (
                layer % 4 == 0
            ):  # exact reference: fp64 GEMV from the bank bytes, correctly rounded to bf16
                if "bank0" not in exact:
                    exact["bank0"] = np.array(W.banks[0])
                Wd = dequant_rows(
                    exact["bank0"], W.plan, f"model.layers.{layer}.self_attn.o_proj", 2048, 4096
                )
                ye = bf16_round_bits(Wd @ f32_of_bits(xb).astype(np.float64))
                out["O_vs_exact_kernel"].append(cmp(yk, ye))
                out["O_vs_exact_mlx"].append(cmp(ym, ye))
        elif name == "RSG":
            xn = mx.fast.rms_norm(mid(layer)[None], lw.ln2, eps)
            lg = mx.matmul(xn, lw.router.T).astype(mx.float32)
            lgn = np.array(lg).reshape(-1)
            ref = np.where(
                lgn < 0, 1 / (1 + np.exp(np.abs(lgn))), 1 - 1 / (1 + np.exp(np.abs(lgn)))
            ).astype(np.float32)
            got = t.sbits("SC", layer * 256, 256, 4).view(np.float32)
            out["ROUTER"].append(
                {"max_abs": float(np.max(np.abs(got - ref))), "n_ne": int(np.sum(got != ref))}
            )
            sg, su, _ = lw.shared
            out["SHARED_ACT"].append(
                cmp(
                    t.sbits("ACT", layer * 6144 + 3072, 1024),
                    bits(swiglu(qmm(xn, sg), qmm(xn, su))),
                )
            )
        elif name == "EGU":
            xn = mx.fast.rms_norm(mid(layer)[None], lw.ln2, eps)
            probs = t.sbits("SC", layer * 256, 256, 4).view(np.float32)
            sel, _ = topk_np(probs)
            eg, eu, _ = lw.experts
            acts = []
            for e in sel:
                g = mx.quantized_matmul(
                    xn,
                    eg.w[int(e)],
                    eg.s[int(e)],
                    eg.b[int(e)],
                    transpose=True,
                    group_size=64,
                    bits=4,
                )
                u = mx.quantized_matmul(
                    xn,
                    eu.w[int(e)],
                    eu.s[int(e)],
                    eu.b[int(e)],
                    transpose=True,
                    group_size=64,
                    bits=4,
                )
                acts.append(swiglu(g, u).reshape(-1))
            out["EXPERT_ACT"].append(
                cmp(t.sbits("ACT", layer * 6144, 3072), bits(mx.concatenate(acts)))
            )
            out["TOPK"].append([int(x) for x in sel])
        elif name == "DOWN":
            probs = t.sbits("SC", layer * 256, 256, 4).view(np.float32)
            sel, w = topk_np(probs)
            _, _, ed = lw.experts
            act = t.sbits("ACT", layer * 6144, 4096)
            w16 = mx.array(w).astype(mx.bfloat16)
            acc = None
            for r, e in enumerate(sel):
                y = mx.quantized_matmul(
                    mxbf(act[r * 512 : (r + 1) * 512])[None],
                    ed.w[int(e)],
                    ed.s[int(e)],
                    ed.b[int(e)],
                    transpose=True,
                    group_size=64,
                    bits=4,
                )
                prod = y * w16[r]
                acc = prod if acc is None else acc + prod
            z = qmm(mxbf(act[3072:4096])[None], lw.shared[2])
            h1 = mid(layer)[None] + (acc + z)
            pending[layer + 1] = bits(h1)
        elif name == "DGU":
            xn = mx.fast.rms_norm(mid(layer)[None], lw.ln2, eps)
            g, u, _ = lw.dense
            out["DENSE_ACT"].append(
                cmp(t.sbits("ACT", layer * 6144, 6144), bits(swiglu(qmm(xn, g), qmm(xn, u))))
            )
        elif name == "DDOWN":
            act = mxbf(t.sbits("ACT", layer * 6144, 6144))[None]
            h1 = mid(layer)[None] + qmm(act, lw.dense[2])
            out["DDOWN_H"].append(cmp(h_of(layer + 1), bits(h1)))
        elif name == "HEAD":
            hn = mx.fast.rms_norm(mxbf(h_of(cfg.n_layers))[None], W.norm, eps)
            out["LOGITS"].append(cmp(t.sbits("LOGITS", 0, P.VOCAB), bits(qmm(hn, W.lm_head))))

    res = t.mk.run_pass_chain(hist.arrays, t.prm(1, token), None, after_phase=after)
    mx.eval(res)
    r = P.parse_result(np.array(res))
    # SELECT vs a host recomputation from the kernel's own logits
    lg = f32_of_bits(t.sbits("LOGITS", 0, P.VOCAB)).astype(np.float64)
    lse = float(np.log(np.sum(np.exp(lg - lg.max()))) + lg.max())
    order = np.lexsort((np.arange(P.VOCAB), -lg))[:20]
    sel = {
        "argmax_ok": r["argmax"] == int(order[0]),
        "top20_ids_ok": r["top_ids"] == [int(x) for x in order],
        "lse_abs": abs(r["lse"] - lse),
        "top_lp_max_abs": float(np.max(np.abs(np.array(r["top_lps"]) - (lg[order] - lse)))),
        "finite": r["finite"],
    }
    summ = {}
    for k, v in out.items():
        if not v:
            continue
        if k == "TOPK":
            summ[k] = {"layers": len(v)}
        elif k == "ROUTER":
            summ[k] = {
                "layers": len(v),
                "max_abs": max(x["max_abs"] for x in v),
                "n_ne_total": sum(x["n_ne"] for x in v),
            }
        else:
            summ[k] = {
                "layers": len(v),
                "max_ulp": max(x["max_ulp"] for x in v),
                "frac_gt1_max": max(x["frac_gt1"] for x in v),
                "frac_gt1_mean": float(np.mean([x["frac_gt1"] for x in v])),
                "frac_ne_mean": float(np.mean([x["frac_ne"] for x in v])),
                "max_abs_rel": max(x["max_abs_rel"] for x in v),
                "nan": sum(x["nan"] for x in v),
                "worst_layer": int(np.argmax([x["frac_gt1"] for x in v])),
            }
    return {
        "summary": summ,
        "select": sel,
        "result": {k: r[k] for k in ("token", "argmax", "lse", "finite")},
        "per_layer": {
            k: v
            for k, v in out.items()
            if k in ("Q", "ATT_fp32", "O", "DOWN_H", "O_vs_exact_kernel", "O_vs_exact_mlx")
        },
    }


# ---------------------------------------------------------------- 2. h trace vs the graph backend
def h_trace(t: T, token: int) -> dict:
    import mlx.core as mx

    from jitengine_flint_m2.model.model import decode_step

    hg = t.History(t.cfg, 262144)
    hg.reserve(t.n + 8)
    lg0 = t.prefill(hg)
    p = t.n
    trace: list = []
    logits = decode_step(
        t.W, hg, mx.array([token], dtype=mx.int32), p, hg.ring_mask(p), trace=trace
    )
    mx.eval(logits, trace)
    per = []
    for layer in range(t.cfg.n_layers + 1):
        got = t.sbits("HS", layer * 2048, 2048)
        per.append(cmp(got, bits(trace[layer])))
    # history rows written by prefill must be byte-identical between the two layouts (same MLX prefill)
    same_hist = True
    for layer in (0, 2, 5, 47, 48):
        if layer in t.cfg.full_layers:
            f = t.cfg.full_layers.index(layer)
            a = np.array(t.hist.full[f // 7][f % 7, :, :p, :].reshape(-1).view(mx.uint16))
            b = np.array(hg.k[layer][0, :, :p, :].reshape(-1).view(mx.uint16))
        else:
            s = t.cfg.sliding_layers.index(layer)
            a = np.array(t.hist.kr[s].reshape(-1).view(mx.uint16))
            b = np.array(hg.rk[layer][0].reshape(-1).view(mx.uint16))
            sl = [(q % P.RING) for q in range(max(0, p - 512), p)]
            a, b = a.reshape(4, P.RING, 128)[:, sl], b.reshape(4, P.RING, 128)[:, sl]
        same_hist = same_hist and bool(np.array_equal(a, b))
    kl = f32_of_bits(t.sbits("LOGITS", 0, P.VOCAB))
    gl = np.array(logits.astype(mx.float32))
    top_k = set(np.argsort(-kl)[:20].tolist()) & set(np.argsort(-gl)[:20].tolist())
    return {
        "per_layer_max_abs_rel": [round(x["max_abs_rel"], 5) for x in per],
        "per_layer_frac_gt1": [round(x["frac_gt1"], 4) for x in per],
        "logits_max_abs": float(np.max(np.abs(kl - gl))),
        "top1_agree": int(np.argmax(kl)) == int(np.argmax(gl)),
        "top20_overlap": len(top_k),
        "prefill_history_identical": same_hist,
        "graph_prefill_top1": int(np.argmax(np.array(lg0.astype(mx.float32)))),
    }


# ---------------------------------------------------------------- 3/4. sequences
def run_seq(t: T, state, n_pass: int, token: int, *, chain=False, forced=None, keep_hist=False):
    import mlx.core as mx

    t.restore_state(state)
    prev, rs = None, []
    for k in range(1, n_pass + 1):
        if forced is not None:
            prm = t.prm(k, forced[k - 1], forced_next=forced[k] if k < len(forced) else None)
        else:
            prm = t.prm(k, token if k == 1 else None)
        r = (
            t.mk.run_pass_chain(t.hist.arrays, prm, prev)
            if chain
            else t.mk.run_pass(t.hist.arrays, prm, prev)
        )
        rs.append(r)
        prev = r
        if chain or k % 8 == 0:
            mx.eval(r)
    mx.eval(rs)
    out = np.stack([np.array(r) for r in rs])
    hb = None
    if keep_hist:
        hb = [np.array(a.reshape(-1).view(mx.uint16)) for a in t.hist.arrays] + [
            np.array(t.mk.scratch[: P.SCRATCH_BYTES])
        ]
    return out, hb


# ---------------------------------------------------------------- 5. verify rows == M = 1 passes
def verify_rows(t: T, state, token: int, rows, starts, S: int = 24, seed: int = 20261023) -> dict:  # noqa: PLR0915
    """Row-exactness of the verify pass: a pass of M rows at positions p .. p + M - 1 (fed tokens of a stream) must
    give, in every row, the M = 1 pass's result words (token, argmax, lse, top-20 ids and log-probabilities, chosen
    lp), its bf16 logits and its history writes BIT for bit; greedy and seeded-sampled (u = uniform(seed, j))."""
    import mlx.core as mx

    from jitengine_flint_m2.sampler import uniform

    def u16(a):
        return np.array(a.reshape(-1).view(mx.uint16)).copy()

    def pm(k, tokv, sel):
        return P.make_params(
            pos=t.n + k - 1,
            token=tokv,
            forced_next=None,
            sel_mode=sel,
            top_k=20,
            top_p=0.95,
            temperature=1.0,
            u=uniform(seed, k) if sel else 0.0,
            poison=False,
            cap=t.hist.capacity,
            theta=t.cfg.rope_theta,
            **t.att,
        )

    g, _ = run_seq(t, state, S + max(rows), token)
    toks = [token] + g[:, P.R_TOKEN].tolist()
    out = {"cases": [], "ok": True}
    nw = 48  # result words compared per row (R_TOKEN .. top-20 lps)
    for sel in (0, 1):
        t.restore_state(state)
        ref_res, ref_lg, prev = [None], [None], None
        for k in range(1, S + max(rows) + 1):
            r = t.mk.run_pass(t.hist.arrays, pm(k, toks[k - 1], sel), prev)
            mx.eval(r)
            prev = r
            ref_res.append(np.array(r)[:nw].copy())
            ref_lg.append(u16(t.mk.logits_view()))
        ref_hist = [u16(a) for a in t.hist.arrays]
        for M in rows:
            for k0 in starts:
                t.restore_state(state)
                prev = None
                for k in range(1, k0):
                    prev = t.mk.run_pass(t.hist.arrays, pm(k, toks[k - 1], sel), prev)
                    mx.eval(prev)
                vp = P.make_vparams(
                    pos=t.n + k0 - 1,
                    tokens=toks[k0 - 1 : k0 - 1 + M],
                    sel_mode=sel,
                    top_k=20,
                    top_p=0.95,
                    temperature=1.0,
                    us=[uniform(seed, k0 + m) if sel else 0.0 for m in range(M)],
                    poison_rows=0,
                    cap=t.hist.capacity,
                    theta=t.cfg.rope_theta,
                    **t.att,
                )
                r = t.mk.run_vpass(t.hist.arrays, vp, prev)
                mx.eval(r)
                a = np.array(r)
                bad_res = [
                    m
                    for m in range(M)
                    if not np.array_equal(
                        a[m * P.VROW_WORDS : m * P.VROW_WORDS + nw], ref_res[k0 + m]
                    )
                ]
                bad_lg = [
                    m
                    for m in range(M)
                    if not np.array_equal(u16(t.mk.vlogits_view(m)), ref_lg[k0 + m])
                ]
                # history rows written at positions p .. p + M - 1 (full layers) and their ring slots
                hist_ok = True
                pos = [t.n + k0 - 1 + m for m in range(M)]
                for ai, arr in enumerate(t.hist.arrays):
                    sh = (
                        arr.shape[0],
                        arr.shape[1],
                        arr.shape[2],
                        -1,
                    )  # [layers][4][rows][row elements]
                    cur = u16(arr).reshape(sh)
                    idx = pos if ai < 4 else [q % P.RING for q in pos]
                    v = cur[:, :, idx]
                    w = ref_hist[ai].reshape(sh)[:, :, idx]
                    hist_ok = hist_ok and bool(np.array_equal(v, w))
                ok = not bad_res and not bad_lg and hist_ok and a[P.VR_ERR] == 0
                out["ok"] = out["ok"] and ok
                case = {
                    "sel": "sampled" if sel else "greedy",
                    "M": M,
                    "start": k0,
                    "ok": ok,
                    "rows_bad_result": bad_res,
                    "rows_bad_logits": bad_lg,
                    "history_ok": hist_ok,
                    "err": int(a[P.VR_ERR]),
                    "tokens": [int(a[m * P.VROW_WORDS]) for m in range(M)],
                }
                if bad_res:
                    m = bad_res[0]
                    case["first_bad"] = {
                        "row": m,
                        "got": a[m * P.VROW_WORDS : m * P.VROW_WORDS + 8].tolist(),
                        "want": ref_res[k0 + m][:8].tolist(),
                    }
                out["cases"].append(case)
                log(f"verify rows {case}")
    # persistent == chain for the verify pass (every result word, all history arrays, the verify scratch)
    M = max(rows)
    vp = P.make_vparams(
        pos=t.n,
        tokens=toks[:M],
        sel_mode=0,
        top_k=20,
        top_p=0.95,
        temperature=1.0,
        us=[0.0] * M,
        poison_rows=0,
        cap=t.hist.capacity,
        theta=t.cfg.rope_theta,
        **t.att,
    )
    snaps = []
    for chain in (False, True):
        t.restore_state(state)
        t.mk.vscratch = None
        r = (t.mk.run_vpass_chain if chain else t.mk.run_vpass)(t.hist.arrays, vp)
        mx.eval(r)
        snaps.append(
            [np.array(r)[: P.VR_ARRIVE].copy()]
            + [u16(a) for a in t.hist.arrays]
            + [np.array(t.mk.vscratch[: P.VMAX * P.SCRATCH_BYTES])]
        )
    eq = [bool(np.array_equal(x, y)) for x, y in zip(*snaps)]
    out["persistent_eq_chain"] = {"M": M, "results": eq[0], "history": eq[1:7], "scratch": eq[7]}
    out["ok"] = out["ok"] and all(eq)
    return out


# ---------------------------------------------------------------- 6. kv8 quantise-on-write == mx.quantize
def quant_xcheck(t: T, state, token: int) -> dict:
    """At every full layer of decode pass 1, replay the MAIN kernel's QKV dispatch on the kv8 chain's own scratch (the
    QKV phase reads only the scratch rows its own layer's prologue reads, which the kv8 QKV leaves unchanged), take its
    bf16 K / V rows at the pass's position, quantise them with mx.quantize, and require the kv8 kernel's stored codes,
    scales and biases to be bit-identical. The replay must also reproduce the queries the kv8 dispatch wrote."""
    import mlx.core as mx

    from jitengine_flint_m2.kernels.runner import MegaKernel
    from jitengine_flint_m2.kv_cache import MKHistory

    cfg = t.cfg
    t.restore_state(state)
    h16 = MKHistory(cfg, 262144)
    h16.reserve(t.hist.capacity)
    assert h16.capacity == t.hist.capacity
    mk16 = MegaKernel(t.W, G=t.a.G, log=log)
    mk16.scratch = t.mk.scratch  # the SAME scratch array: the replay reads the kv8 chain's state
    k16 = mk16.kernel(True)
    p = t.n
    prm = t.prm(1, token)
    cases, q_same = [], True

    def u8rows(arr, f, pos):  # [4][row bytes] of full layer f at position pos
        a = np.array(arr[f].reshape(-1).view(mx.uint8)).reshape(arr.shape[1], arr.shape[2], -1)
        return a[:, pos]

    def after(ph, layer, out):
        nonlocal q_same
        if ph != P.PH["QKV"] or layer not in cfg.full_layers:
            return
        mx.eval(out)
        q0 = t.sbits("Q", layer * 4096, 4096)
        r = k16(
            inputs=mk16._inputs(
                h16.arrays, mx.array(prm), mx.array([ph, layer], dtype=mx.uint32), mk16.dummy, out
            ),
            grid=(mk16.G * P.TPG, 1, 1),
            threadgroup=(P.TPG, 1, 1),
            output_shapes=[(P.RES_WORDS,)],
            output_dtypes=[mx.uint32],
            init_value=0,
        )[0]
        mx.eval(r)
        q_same = q_same and bool(np.array_equal(q0, t.sbits("Q", layer * 4096, 4096)))
        f = cfg.full_layers.index(layer)
        for isv in (False, True):
            k_bf = h16.full_kv(layer, p + 1)[1 if isv else 0][0, :, p, :]  # [4, 128] bf16 (main)
            wq, sc, bi = mx.quantize(k_bf, group_size=64, bits=8)
            mx.eval(wq, sc, bi)
            want_codes = np.array(wq.reshape(-1).view(mx.uint8)).reshape(4, 128)
            want_sb = np.stack([bits(sc).reshape(4, 2), bits(bi).reshape(4, 2)], -1).reshape(4, 4)
            got_codes = u8rows(t.hist.full[2 if isv else 0], f, p)
            got_sb = u8rows(t.hist.full[3 if isv else 1], f, p).view(np.uint16).reshape(4, 4)
            cases.append(
                {
                    "layer": layer,
                    "v": isv,
                    "codes_ne": int((got_codes != want_codes).sum()),
                    "sb_ne": int((got_sb != want_sb).sum()),
                }
            )

    res = t.mk.run_pass_chain(t.hist.arrays, prm, None, after_phase=after)
    mx.eval(res)
    ok = (
        q_same
        and len(cases) == 2 * len(cfg.full_layers)
        and all(c["codes_ne"] == 0 and c["sb_ne"] == 0 for c in cases)
    )
    return {
        "ok": ok,
        "position": p,
        "rows_checked": len(cases),
        "queries_unchanged": q_same,
        "codes_ne_total": sum(c["codes_ne"] for c in cases),
        "sb_ne_total": sum(c["sb_ne"] for c in cases),
        "cases": cases,
    }


def main(argv=None) -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, default=1000)
    ap.add_argument("--passes", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--G", type=int, default=P.G_DEFAULT)
    ap.add_argument(
        "--chunks",
        default=None,
        help="partition policy (default: the profile's, plan.default_policy)",
    )
    ap.add_argument("--skip", default="", help="comma list of: oracle,trace,chain,seq,verify,quant")
    ap.add_argument(
        "--vrows", default="2,4,8", help="verify row counts checked against M = 1 passes"
    )
    ap.add_argument("--vstarts", default="1,6", help="first decode pass of each verify check")
    ap.add_argument(
        "--profile",
        default="main",
        choices=("main", "kv8"),
        help="kv8: the 8-bit full-layer history kernels",
    )
    a = ap.parse_args(argv)
    if a.chunks is None:
        a.chunks = P.default_policy(8 if a.profile == "kv8" else 16)
    if a.profile == "kv8" and not P.is_6d(a.chunks):
        raise SystemExit(
            "--profile kv8 needs a layout-6d partition (--chunks l6dk, the default, l6d2 or l6d)"
        )
    import mlx.core as mx

    skip = set(x for x in a.skip.split(",") if x)
    R: dict = {"argv": sys.argv, "ok": {}}
    t0 = time.time()
    t = T(a)
    R["probe_ms"] = t.mk.probe(t.hist.arrays, t.att, t.hist.capacity)
    log(f"residency probe ok {R['probe_ms']}")
    lg0 = t.prefill(t.hist)
    token = int(np.argmax(np.array(lg0.astype(mx.float32))))
    R["prompt_tokens"], R["first_token"] = t.n, token
    state = t.save_state()
    log(f"prefill done; step-0 token {token}")

    def dump():
        with open(a.out, "w") as f:
            json.dump(R, f, indent=1, default=str)

    if "oracle" not in skip:
        t.restore_state(state)
        R["oracle"] = phase_oracle(t, token)
        log("oracle: " + json.dumps(R["oracle"]["summary"])[:3000])
        log("select: " + json.dumps(R["oracle"]["select"]))
        dump()
    if t.kv8 and "quant" not in skip:
        R["quant_xcheck"] = quant_xcheck(t, state, token)
        R["ok"]["quant_on_write_eq_mx_quantize"] = R["quant_xcheck"]["ok"]
        log(
            "quant xcheck: "
            + json.dumps({k: v for k, v in R["quant_xcheck"].items() if k != "cases"})
        )
        dump()
    if "trace" not in skip and not t.kv8:
        R["h_trace"] = h_trace(t, token)
        log("h trace: " + json.dumps(R["h_trace"])[:2000])
        dump()
    if "chain" not in skip:
        pers, hp = run_seq(t, state, a.passes, token, keep_hist=True)
        chn, hc = run_seq(t, state, a.passes, token, chain=True, keep_hist=True)
        res_eq = bool(np.array_equal(pers[:, : P.R_ARRIVE], chn[:, : P.R_ARRIVE]))
        names = t.names + ["scratch"]
        mem_eq = {n: bool(np.array_equal(x, y)) for n, x, y in zip(names, hp, hc)}
        R["persistent_vs_chain"] = {
            "passes": a.passes,
            "results_equal": res_eq,
            "buffers_equal": mem_eq,
            "first_diff_pass": int(
                np.argmax(np.any(pers[:, : P.R_ARRIVE] != chn[:, : P.R_ARRIVE], axis=1))
            )
            if not res_eq
            else None,
            "barrier_err": int(pers[:, P.R_ERR].max()),
            "tokens": pers[:16, 0].tolist(),
        }
        R["ok"]["persistent_eq_chain"] = res_eq and all(mem_eq.values())
        log("persistent vs chain: " + json.dumps(R["persistent_vs_chain"]))
        del hp, hc
        dump()
    if "seq" not in skip:
        g, _ = run_seq(t, state, a.steps, token)
        toks = [token] + g[:, P.R_TOKEN].tolist()
        f1, _ = run_seq(t, state, a.steps, token, forced=toks[: a.steps + 1])
        f2, _ = run_seq(t, state, a.steps, token, forced=toks[: a.steps + 1])
        cmpw = [w for w in range(P.R_ARRIVE) if w != P.R_FORCED_LP]
        fg = bool(np.array_equal(g[:, cmpw], f1[:, cmpw]))
        lp_match = bool(np.array_equal(f1[:-1, P.R_FORCED_LP], g[:-1, P.R_CHOSEN_LP]))
        rep = bool(np.array_equal(f1, f2))
        R["sequences"] = {
            "steps": a.steps,
            "forced_eq_greedy": fg,
            "forced_lp_eq_chosen_lp": lp_match,
            "forced_repeat_bit_identical": rep,
            "distinct_tokens": len(set(toks)),
            "finite_all": bool(g[:, P.R_FINITE].all()),
            "barrier_err": int(g[:, P.R_ERR].max()),
        }
        R["ok"].update(forced_eq_greedy=fg and lp_match, repeat_identical=rep)
        log("sequences: " + json.dumps(R["sequences"]))
    if "verify" not in skip and P.verify_capable(t.att):
        R["verify_rows"] = verify_rows(
            t,
            state,
            token,
            [int(x) for x in a.vrows.split(",")],
            [int(x) for x in a.vstarts.split(",")],
        )
        R["ok"]["verify_rows_exact"] = R["verify_rows"]["ok"]
        log(
            "verify: ok "
            + str(R["verify_rows"]["ok"])
            + " "
            + json.dumps(R["verify_rows"]["persistent_eq_chain"])
        )
        dump()
    R["wall_s"] = round(time.time() - t0, 1)
    dump()
    log(f"done in {R['wall_s']} s: {R['ok']}")
    return 0


def _checks(checkpoint: str, out, *profile: str) -> dict:
    import gc

    import mlx.core as mx

    try:
        assert main(["--model", checkpoint, "--out", str(out), *profile]) == 0
    finally:  # the next test in this process loads its own 23 GB of weights
        gc.collect()
        mx.clear_cache()
    with open(out) as f:
        return json.load(f)["ok"]


def test_megakernel_checks(m4pro_checkpoint, tmp_path):
    ok = _checks(m4pro_checkpoint, tmp_path / "main.json")
    assert ok and all(ok.values()), ok


def test_megakernel_checks_kv8(m4pro_checkpoint, tmp_path):
    ok = _checks(m4pro_checkpoint, tmp_path / "kv8.json", "--profile", "kv8")
    assert ok and all(ok.values()), ok


if __name__ == "__main__":
    raise SystemExit(main())
