"""Engine construction on the M4 Pro with a real checkpoint (no weights load): request validation, profiles,
limits and refusals."""

from jitengine_flint_m2.engine.io_struct import ContextTooLarge, GenRequest
from jitengine_flint_m2.model.config import (
    StartupError,
    load_config,
    tensor_index,
)
from jitengine_flint_m2.weight_banks import BankPlan


def expect_raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return
    raise AssertionError(f"{fn.__name__}{a} did not raise {exc.__name__}")


def test_request_validation(m4pro_checkpoint):
    from jitengine_flint_m2.engine.engine import Engine

    eng = Engine(m4pro_checkpoint, start=False, context_limit=4096)
    p = eng._validate(GenRequest(prompt_ids=[1, 2, 3]))
    assert p.n_out == 4096 - 3 and p.clamped
    assert p.temperature == 1.0 and p.top_p == 0.95 and p.top_k == 20
    p = eng._validate(GenRequest(prompt_ids=[1, 2, 3], max_tokens=10, forced_tokens=[5, 6]))
    assert p.n_out == 3 and p.passes == 2
    p = eng._validate(GenRequest(prompt_ids=[1, 2, 3], max_tokens=1025, temperature=0.0))
    assert p.n_out == 1025 and p.passes == 1024
    expect_raises(ContextTooLarge, eng._validate, GenRequest(prompt_ids=[1] * 4096))
    expect_raises(
        ContextTooLarge, eng._validate, GenRequest(prompt_ids=[1] * 4000, forced_tokens=[1] * 97)
    )
    for bad in (
        dict(prompt_ids=[150144]),
        dict(prompt_ids=[-1]),
        dict(prompt_ids=[1.5]),
        dict(prompt_ids=[]),
        dict(prompt_ids=[1], temperature=-1),
        dict(prompt_ids=[1], top_p=0),
        dict(prompt_ids=[1], top_k=-1),
        dict(prompt_ids=[1], max_tokens=-1),
        dict(prompt_ids=[1], logprobs=21),
        dict(prompt_ids=[1], seed=True),
        dict(prompt_ids=[1], fault="nonfinite@3"),
    ):
        expect_raises(ValueError, eng._validate, GenRequest(**bad))
    h = eng.health()
    assert h["profile"] == "main" and h["kv_bits"] == 16 and h["status"] == "loading"
    assert h["stop_token_ids"] == [150001, 150003, 150005, 150018]
    # decode_backend "auto" (the default) resolves at load; the context limit was explicit here
    assert h["decode_backend"] == h["decode_backend_requested"] == "auto" and h["mk"] is None
    assert h["context_limit_source"] == "explicit" and "mk_unavailable" not in h
    assert "simulated" not in h
    import platform
    from importlib.metadata import version

    assert (
        h["versions"]["mlx"] == version("mlx") and h["versions"]["macos"] == platform.mac_ver()[0]
    )
    assert len(h["versions"]["kernel_source_sha256"]) == 16
    expect_raises(StartupError, Engine, m4pro_checkpoint, start=False, decode_backend="bogus")
    expect_raises(ValueError, Engine, m4pro_checkpoint, start=False, mk_chunks="bogus")
    expect_raises(
        StartupError, Engine, m4pro_checkpoint, start=False, mk_chunks="strided"
    )  # request-fixed partitions: kernel timing only
    fe = Engine(m4pro_checkpoint, start=False, context_limit=4096, allow_fault_injection=True)
    assert fe._validate(GenRequest(prompt_ids=[1], fault="kernel_stall@3:persist")).fault.persist
    ge = Engine(
        m4pro_checkpoint,
        start=False,
        context_limit=4096,
        allow_fault_injection=True,
        decode_backend="graph",
    )
    expect_raises(ValueError, ge._validate, GenRequest(prompt_ids=[1], fault="kernel_stall@3"))
    # cancellation and stop_all (no worker started): a stopped engine refuses new jobs; queued jobs are cancelled
    import threading

    ev = threading.Event()
    it = fe.stream(GenRequest(prompt_ids=[1, 2], max_tokens=4), cancel=ev)
    job = next(iter(fe._pending))
    assert job.cancel is ev and fe._waiting == 1
    fe.stop_all("test stop")
    assert ev.is_set() and job.reason == "test stop"
    expect_raises(Exception, fe.stream, GenRequest(prompt_ids=[1, 2], max_tokens=4))
    fe._answer_cancelled(job)
    r = next(it)
    assert (
        r.finish_reason == "error"
        and r.metrics["error_type"] == "EngineUnavailable"
        and "test stop" in r.error
    )


def test_kv8_profile(m4pro_checkpoint):
    """The kv8 variant: explicit flag only, labelled everywhere, 8-bit g64 full-layer history, its own kernel source
    derived by anchored edits while the main source stays untouched."""
    from jitengine_flint_m2.engine.engine import Engine
    from jitengine_flint_m2.kernels import kv8_fold, kv8_src
    from jitengine_flint_m2.kernels import metal_src as MS
    from jitengine_flint_m2.kernels import plan as P

    cfg = load_config(m4pro_checkpoint)
    assert cfg.kv_variant == (8, 8, 64)
    assert cfg.history_bytes_per_token_bits(16) == cfg.history_bytes_per_token == 28_672
    assert cfg.history_bytes_per_token_bits(8) == 14 * 2 * 4 * (128 + 8) == 15_232
    eng = Engine(m4pro_checkpoint, start=False, context_limit=4096, profile="kv8")
    h = eng.health()
    # the megakernel, or refused at load
    assert h["profile"] == "kv8" and h["kv_bits"] == 8 and h["decode_backend"] == "auto"
    assert Engine(m4pro_checkpoint, start=False, context_limit=4096).health()["kv_bits"] == 16
    expect_raises(
        StartupError, Engine, m4pro_checkpoint, start=False, profile="kv8", decode_backend="graph"
    )
    expect_raises(
        StartupError, Engine, m4pro_checkpoint, start=False, profile="kv8", mk_chunks="strided"
    )
    assert (
        eng.mk_chunks == "l6dk"
        and Engine(m4pro_checkpoint, start=False, context_limit=4096).mk_chunks == "l6d"
    )
    assert (
        Engine(
            m4pro_checkpoint, start=False, context_limit=4096, profile="kv8", mk_chunks="l6d"
        ).mk_chunks
        == "l6d"
    )
    eo = P.expert_record_offsets(BankPlan(cfg, tensor_index(m4pro_checkpoint, cfg)))
    for chain in (False, True):
        for M in (1, 2, 8):
            hm, bm = MS.render(cfg, eo, chain, mr=M)
            hk, bk = MS.render(cfg, eo, chain, mr=M, kv8=True)
            assert "att6q8" not in hm and "kv8_store" not in hm and "#define KV8" not in hm
            assert bk == bm and "#define KV8 1" in hk and hk.count("kv8_store(c, l,") == 1 + (M > 1)
            assert hk.count("att6q8_item(c, l, kvh,") == 1 + (M > 1)
            # the folded item, once; no per-element dequantisation in the stream
            assert hk.count(kv8_fold.att6q8_src("att6q8")) == 1 and "bfloat4(fma(" not in hk
            assert hk.count("att6(Q") == 0 and hm.count("att6(Q") == 1 + (M > 1)
            # every layout-6d item loop is dealt threadgroup-first, in both profiles
            for src in (hm, hk):
                assert "for (uint it = c.w * 8u + c.sg;" not in src
                assert (
                    src.count("for (uint it = gsg(c); it < 4u * C; it += nsg(c)) {") == 2
                )  # 6d + legacy layout 2
                assert src.count("for (uint it = gsg(c); it < 4u * CL * MR; it += nsg(c)) {") == (
                    M > 1
                )
    expect_raises(AssertionError, kv8_src.apply_m1, "no anchors here")
    expect_raises(AssertionError, MS.render, cfg, eo, False, diag=1, kv8=True)


def test_request_limits(m4pro_checkpoint):
    """The optional output cap clamps max_tokens (off: unchanged), bad
    limits are refused at construction, and /health and /metrics never carry a request's seed or request_id."""
    from jitengine_flint_m2.engine.engine import Engine

    off = Engine(m4pro_checkpoint, start=False, context_limit=4096)
    assert off.request_deadline_s is None and off.max_output_tokens is None
    assert off._validate(GenRequest(prompt_ids=[1, 2, 3])).n_out == 4093
    assert off._validate(GenRequest(prompt_ids=[1, 2, 3], max_tokens=3000)).n_out == 3000
    cap = Engine(
        m4pro_checkpoint,
        start=False,
        context_limit=4096,
        max_output_tokens=100,
        request_deadline_s=2,
    )
    assert cap.request_deadline_s == 2.0 and cap.max_output_tokens == 100
    p = cap._validate(GenRequest(prompt_ids=[1, 2, 3]))
    assert p.n_out == 100 and p.passes == 99 and p.clamped
    p = cap._validate(GenRequest(prompt_ids=[1, 2, 3], max_tokens=50))
    assert p.n_out == 50 and not p.clamped
    assert cap._validate(GenRequest(prompt_ids=[1, 2, 3], max_tokens=5000)).n_out == 100
    assert (
        cap._validate(GenRequest(prompt_ids=[1] * 4090)).n_out == 6
    )  # the room is still the bound
    assert (
        cap._validate(GenRequest(prompt_ids=[1], forced_tokens=[2] * 300)).n_out == 301
    )  # forced: not capped
    for bad in (
        dict(request_deadline_s=0),
        dict(request_deadline_s=-1),
        dict(request_deadline_s=float("inf")),
        dict(request_deadline_s=float("nan")),
        dict(request_deadline_s=True),
        dict(request_deadline_s="5"),
        dict(max_output_tokens=0),
        dict(max_output_tokens=-3),
        dict(max_output_tokens=1.5),
        dict(max_output_tokens=True),
    ):
        expect_raises(StartupError, Engine, m4pro_checkpoint, start=False, **bad)
    h = cap.health()
    assert h["request_deadline_s"] == 2.0 and h["max_output_tokens"] == 100
    assert off.health()["request_deadline_s"] is None and off.health()["max_output_tokens"] is None
    m = {
        "request_id": "cmpl-secret",
        "seed": 123456789,
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "decode_s": 0.5,
    }
    off.last_metrics, off.recent = m, [dict(m), dict(m)]
    h, st = off.health(), off.stats()
    for d in (h["last_request"], *st["recent"], st["last_request"]):
        assert (
            "seed" not in d
            and "request_id" not in d
            and d["completion_tokens"] == 4
            and d["decode_s"] == 0.5
        )
    assert (
        len(st["recent"]) == 2 and off.last_metrics["seed"] == 123456789
    )  # the request's own x_metrics keep them
    blob = repr((h, st))
    assert "cmpl-secret" not in blob and "123456789" not in blob


def test_tuned_path(m4pro_checkpoint):
    """With no options, the tuned M4 Pro plans G 40 from the tuned table with no search, sizes the context limit to its
    wired limit (262,144 when that holds 31,764 MiB), and resolves decode_backend "auto" at load, speculation on."""
    from jitengine_flint_m2.engine.engine import Engine

    e = Engine(m4pro_checkpoint, start=False, log=lambda m: None)
    assert (e.grid.candidates, e.grid.source) == ((40,), "tuned") and e.grid_cache is None
    assert e.context_limit_source == "auto" and e.decode_backend == "auto" and e.speculation
    if e.facts.wired_bytes >= 31_764 * 2**20:
        assert e.context_limit == 262_144
    h = e.health()
    assert h["device"]["tuned"] and h["device"]["gpu_cores"] == 20 and "simulated" not in h
    assert any("the tuned device" in n for n in e.device_notes)
