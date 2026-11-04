"""Flint INT4 single-stream decode engine: tuned on the 20-core Apple M4 Pro,
admitted on any Apple silicon Mac by capability.

Modules by concern: model/ (config and startup checks, the context limit, routing, the MLX-graph
forward pass), device (device facts, admission, the megakernel grid per device and its cache),
weight_banks + loader (the weight banks), kv_cache (attention state), kernels/ (the decode
megakernel's Metal sources, plan and runner), loadcheck + loadcheck_text (the megakernel's load
check and its fixed text), decode + copy_draft (the release-group loop and its exact speculation),
sampler (token choice, log-probabilities), engine/ (session, worker, prefix cache, faults, the
megakernel's bring-up and fallback, the request and result types), server + cli (HTTP surface),
doctor + tune (what this Mac runs; the opt-in grid timing), tokenizer (with the stop strings).
"""

from jitengine_flint_m4pro.engine.io_struct import (  # noqa: F401
    GenRequest,
    GenResult,
    ScoredPosition,
    StepOutput,
)

__version__ = "0.1.0a1"


def create_engine(model_path: str, **options):
    from jitengine_flint_m4pro.engine.engine import Engine

    return Engine(model_path, **options)
