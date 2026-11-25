"""Process-local changes to the pinned mlx-lm. Nothing is written to the installed package.

`install()` returns a report of what it applied and why anything was left as stock, so `doctor`
and `/stats` can show it.
"""

from __future__ import annotations

import functools
import http.server
import logging
from dataclasses import dataclass, field

from jitengine_qwen3_1_7b_m4pro.engine import _step_edits as E
from jitengine_qwen3_1_7b_m4pro.engine import source_patch
from jitengine_qwen3_1_7b_m4pro.engine.admission import TokenBudget
from jitengine_qwen3_1_7b_m4pro.engine.state import STATE

# sha256 of the stock function source at the pinned mlx-lm commit
STEP_SHA256 = "d7513c5031029194ca3820ae81d8966b792e66a90c39c1d7ad61d30bc9e8c5ad"
GENERATE_SHA256 = "3a4259353eae53417f6cc455c0c1263639b514cb2b62e21be6cb4e72cb49cd55"
LISTEN_BACKLOG = 1024

GENERATE_EDITS = [
    (
        '"top_logprobs": args.top_logprobs,\n',
        '"top_logprobs": args.top_logprobs,\n'
        '                        "want_logprobs": bool(args.logprobs) or args.top_logprobs > 0,\n',
    ),
    (
        "r.logprobs[r.token].item(),",
        '(_jit_materialize(r.logprobs)[r.token].item() if result["want_logprobs"] else 0.0),',
    ),
    (
        '                                    r.logprobs,\n                                    result["top_logprobs"],',
        '                                    _jit_materialize(r.logprobs) if result["top_logprobs"] > 0 else None,\n'
        '                                    result["top_logprobs"],',
    ),
]


class _Backlog(http.server.ThreadingHTTPServer):
    # The default listen backlog of 5 resets connections in a burst of 16 clients.
    request_queue_size = LISTEN_BACKLOG
    daemon_threads = True


@dataclass
class Report:
    applied: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"applied": list(self.applied), "skipped": dict(self.skipped)}


def _lazy_logprobs(mx):
    class JitLazyLogprobs:
        """Log-probabilities of one row of a logits matrix, computed when first read."""

        __slots__ = ("logits", "row")

        def __init__(self, logits, row):
            self.logits = logits
            self.row = row

        def materialize(self):
            x = self.logits[self.row]
            return x - mx.logsumexp(x, axis=-1, keepdims=True)

        def __getitem__(self, idx):
            return self.materialize()[idx]

    return JitLazyLogprobs


def _install_lazy_logprobs(report, generate, server) -> None:
    """Skip the full-vocabulary log-probabilities when no request asked for them."""
    import mlx.core as mx

    lazy = _lazy_logprobs(mx)
    generate._JitLazyLogprobs = lazy
    server._jit_materialize = lambda lp: lp.materialize() if isinstance(lp, lazy) else lp
    # The server edit goes first and works with or without lazy values; the batch edit hands the
    # server lazy values, which stock server code cannot read as top log-probabilities.
    why = source_patch.apply(server.ResponseGenerator, "_generate", GENERATE_SHA256, GENERATE_EDITS)
    if why is None:
        why = source_patch.apply(
            generate.GenerationBatch,
            "_step",
            STEP_SHA256,
            [(E.STEP_OLD_1, E.STEP_NEW_1), (E.STEP_OLD_2, E.STEP_NEW_2)],
        )
    if why is None:
        report.applied.append("lazy_logprobs")
    else:  # a patched server alone still runs on stock log-probabilities
        report.skipped["lazy_logprobs"] = why


def _install_exact_qmm() -> None:
    """Exact decode matmul for 13 to 16 rows, for every quantized linear and the tied head."""
    import mlx.nn as nn

    from jitengine_qwen3_1_7b_m4pro.engine import fusion
    from jitengine_qwen3_1_7b_m4pro.kernels import qmm_exact

    if getattr(nn.QuantizedLinear, "_jit_exact", False):
        return
    orig_call = nn.QuantizedLinear.__call__
    orig_as_linear = nn.QuantizedEmbedding.as_linear

    def covered(m, x):
        return qmm_exact.eligible(
            x, m.scales, m.get("biases"), m.group_size, m.bits, m.mode, m.weight.shape[0]
        )

    def call(self, x):
        if covered(self, x):
            segments = getattr(self, fusion.SEGMENTS, None)
            y = qmm_exact.qmm_exact(x, self.weight, self.scales, self["biases"], segments)
            if y is not None:
                return y + self["bias"] if "bias" in self else y
        return orig_call(self, x)

    def as_linear(self, x):
        if covered(self, x):
            y = qmm_exact.qmm_exact(x, self.weight, self.scales, self["biases"])
            if y is not None:
                return y
        return orig_as_linear(self, x)

    nn.QuantizedLinear.__call__ = call
    nn.QuantizedEmbedding.as_linear = as_linear
    nn.QuantizedLinear._jit_exact = True


def _install_load_hook(server, *, fuse: bool, selfcheck: bool) -> None:
    """After each model load: fuse projections (not with an adapter), then check the kernels."""
    from jitengine_qwen3_1_7b_m4pro.engine import fusion

    orig_load = server.load

    def load(*args, **kwargs):
        out = orig_load(*args, **kwargs)
        model = out[0]
        if fuse and kwargs.get("adapter_path") is None:
            fusion.fuse_model(model)
        if selfcheck and getattr(model, "model_type", None) == "qwen3":
            from jitengine_qwen3_1_7b_m4pro.kernels import selfcheck as sc

            STATE["selfcheck"] = result = sc.run_for(model)
            for text in sc.warnings(result):
                logging.warning("startup self-check: %s", text)
        return out

    server.load = load


def _install_single_model(server) -> None:
    """Any request naming the served model uses the loaded one; any other name, and any draft model a
    request names, is refused, where stock mlx-lm would load it (from the Hugging Face Hub if need be)
    and reload the served model with it."""
    orig_provider_load = server.ModelProvider.load

    def provider_load(self, model_path, adapter_path=None, draft_model_path=None):
        served = STATE.get("served_names")
        if served:
            if draft_model_path not in (None, "default_model"):
                logging.warning("refused request for draft model %r", draft_model_path)
                raise ValueError(
                    f"this server takes no draft model from a request, not {draft_model_path!r}"
                )
            if model_path in served:
                model_path = "default_model"
            elif model_path != "default_model":
                logging.warning(
                    "refused request for model %r; serving %r",
                    model_path,
                    STATE.get("served_primary"),
                )
                raise ValueError(
                    f"this server serves only {STATE.get('served_primary')!r}, not {model_path!r}"
                )
        return orig_provider_load(self, model_path, adapter_path, draft_model_path)

    server.ModelProvider.load = provider_load


def install(
    *, fuse: bool = True, selfcheck: bool = True, budget: TokenBudget | None = None
) -> Report:
    """Apply every change; each one falls back to stock on its own."""
    import importlib

    from jitengine_qwen3_1_7b_m4pro.server import hosts, limits, routes, stats

    if STATE.get("installed") is not None:
        return STATE["installed"]
    # `mlx_lm.generate` is also a function there, so import the modules by name
    generate = importlib.import_module("mlx_lm.generate")
    server = importlib.import_module("mlx_lm.server")
    report = Report()

    _install_lazy_logprobs(report, generate, server)
    # `run` ignores its own server_class argument and calls `_run_http_server` with the default.
    server._run_http_server = functools.partial(server._run_http_server, server_class=_Backlog)
    report.applied.append("listen_backlog")
    _install_exact_qmm()
    report.applied.append("exact_qmm")
    if fuse:
        from jitengine_qwen3_1_7b_m4pro.engine import fusion

        why = fusion.guard()
        if why is not None:
            fuse = False
            report.skipped["fusion"] = why
    _install_load_hook(server, fuse=fuse, selfcheck=selfcheck)
    if fuse:
        report.applied.append("fusion")
    _install_single_model(server)
    report.applied.append("single_model")
    stats.install(server.APIHandler)
    report.applied.append("stats_endpoint")
    if budget is not None:
        limits.install(server.APIHandler, budget)
        STATE["budget"] = budget
        report.applied.append("admission")
    routes.install(server.APIHandler)
    report.applied.append("anthropic_responses_routes")
    if STATE.get("host") is not None:  # last, so the check runs before every other handler
        hosts.install(
            server.APIHandler, hosts.served_names(STATE["host"], STATE.get("allowed_hosts") or ())
        )
        report.applied.append("host_check")
    STATE["patches"] = report.as_dict()
    STATE["installed"] = report
    return report
