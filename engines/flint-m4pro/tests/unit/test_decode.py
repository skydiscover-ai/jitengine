"""The pipelined speculation loop on a fake megakernel whose model is next = f(token, position)."""

from jitengine_flint_m4pro.tokenizer import StopMatcher


def test_spec_loop_pipelined():
    """The pipelined speculation loop commits exactly the one-token-per-pass stream, whatever the policy
    drafts, on a fake megakernel whose 'model' is next = f(token, position); M = 1 passes fed from the device take the
    previous pass's own result, as the kernel does (P_TOKMODE)."""
    import types

    import mlx.core as mx
    import numpy as np

    from jitengine_flint_m4pro import copy_draft as D
    from jitengine_flint_m4pro import decode as DEC
    from jitengine_flint_m4pro.kernels import plan as P

    V = 50

    def f(tok, pos):  # copy-friendly with periodic breaks (drafts accepted / rejected)
        return (tok * 3 + 1) % 17 + 20 if pos % 23 else (tok * 7 + pos) % V

    class FakeMK:
        vrows = (2, 3, 4, 5, 6, 7, 8)

        def __init__(self):
            self.passes = 0

        def check_memory_once(self, _):
            pass

        def _row(self, tok, pos):
            w = np.zeros(P.RES_WORDS, dtype=np.uint32)
            w[P.R_TOKEN] = w[P.R_ARGMAX] = f(int(tok), int(pos))
            w[P.R_FINITE] = w[P.R_DONE] = 1
            return w

        def run_pass(self, arrays, prm, prev=None):
            self.passes += 1
            tok = int(np.array(prev)[P.R_TOKEN]) if prm[P.P_TOKMODE] else int(prm[P.P_TOKEN])
            return mx.array(self._row(tok, prm[P.P_POS]))

        def run_vpass(self, arrays, prm, prev=None):
            self.passes += 1
            M = int(prm[P.P_ROWS])
            out = np.zeros(P.VRES_WORDS, dtype=np.uint32)
            for m in range(M):
                out[m * P.VROW_WORDS : (m + 1) * P.VROW_WORDS] = self._row(
                    prm[P.P_RTOK + m], prm[P.P_POS] + m
                )[: P.VROW_WORDS]
            return mx.array(out)

    device = mx.default_device()
    mx.set_default_device(mx.cpu)  # no GPU work in the model-free suite
    for prompt_len, steps, mode, eos in (
        (40, 300, "greedy", None),
        (700, 257, "sampled", None),
        (30, 9, "greedy", None),
        (40, 300, "greedy", 20),
    ):
        prompt = [(i * 5) % 13 + 20 for i in range(prompt_len)]
        hist = types.SimpleNamespace(arrays=[], capacity=4096)
        eng = types.SimpleNamespace(
            mk=FakeMK(),
            hist=hist,
            cfg=types.SimpleNamespace(rope_theta=1e4),
            mk_chunks="l6d",
            eos=set() if eos is None else {eos},
        )
        plan = types.SimpleNamespace(
            prompt=prompt,
            passes=steps,
            forced=None,
            temperature=0.0 if mode == "greedy" else 1.0,
            top_k=0 if mode == "greedy" else 20,
            top_p=1.0,
            seed=3,
            fault=None,
            logprobs=None,
            ignore_eos=eos is None,
            stop=StopMatcher(None),
            n_out=steps + 1,
        )
        first = 7
        passes = DEC._Passes(eng, plan, first)
        out = []

        # the loop runs release() before the next iteration rebinds out, steps and eos
        def release(s):
            out.append(s.token_id)  # noqa: B023
            return len(out) >= steps or s.token_id == eos  # noqa: B023

        st = DEC._spec_loop(eng, plan, passes, release, D.PromptIndex(prompt), 0)
        ref, t = [], first
        for j in range(1, steps + 1):
            t = f(t, prompt_len + j - 1)
            ref.append(t)
        if eos is not None:  # nothing past the EOS is released
            ref = ref[: ref.index(eos) + 1]
            assert out == ref and len(out) < steps
            continue
        assert out == ref, (mode, prompt_len)
        assert st.tokens == steps and st.passes == eng.mk.passes and sum(st.rows) == st.passes
        assert sum(m * c for m, c in enumerate(st.rows)) >= st.passes
        if steps > 100:
            assert st.verify > 0 and st.tokens > st.passes and st.drains < st.passes
    mx.set_default_device(device)
