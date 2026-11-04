"""The startup checks and the weight-bank layout, on the real checkpoint's config and safetensors headers."""

from itertools import pairwise

from jitengine_flint_m4pro.model.config import (
    StartupError,
    check_context_limit,
    check_profile,
    load_config,
    resolve_context_limit,
    tensor_index,
)
from jitengine_flint_m4pro.weight_banks import BankPlan

GIB = 2**30


def expect_raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return str(e)
    raise AssertionError(f"{fn.__name__}{a} did not raise {exc.__name__}")


def test_profiles_and_limits():
    check_profile("main")
    check_profile("kv8")  # built (labelled, separately reported variant)
    for p in ("relaxed", "bogus", "Main", ""):
        expect_raises(StartupError, check_profile, p)
    for bad in (0, -1, 262145, True, 1.5):
        expect_raises(StartupError, check_context_limit, bad)
    assert check_context_limit(262144) == 262144


def test_layout(flint_headers):
    cfg = load_config(flint_headers)
    plan = BankPlan(cfg, tensor_index(flint_headers, cfg))
    assert len(plan.bank_sizes) <= 4 and max(plan.bank_sizes) < 30_150_672_384
    assert plan.expert_stride == 1_769_472
    assert all(s.offset % 256 == 0 for s in plan.slots.values())
    ends = sorted((s.offset, s.offset + s.nbytes) for s in plan.slots.values())
    assert all(a[1] <= b[0] for a, b in pairwise(ends)), "bank-0 slots overlap"
    fields = sorted((f.offset, f.offset + f.nbytes) for f in plan.expert_fields.values())
    assert fields[0][0] == 0 and all(a[1] == b[0] for a, b in pairwise(fields))
    assert len(plan.expert_home) == len(cfg.moe_layers)
    t = plan.descriptor_table()
    assert t["layer_offsets"].shape[0] == cfg.n_layers


def test_context_limit_for_the_wired_limit(flint_headers):
    """`auto` is what the GPU wired limit holds beside the weights (22,568,715,008 B) and 3 GiB of headroom, at
    28,672 B per token (kv8 15,232), rounded down to 1,024 and capped at 262,144; below 4,096 it is refused. An explicit
    limit that does not fit is refused with what fits and the sysctl that would hold it, never truncated."""
    cfg = load_config(flint_headers)
    wb = BankPlan(cfg, tensor_index(flint_headers, cfg)).total_bytes
    per = {b: cfg.history_bytes_per_token_bits(b) for b in (16, 8)}
    assert wb == 22_568_715_008 and per == {16: 28_672, 8: 15_232}

    def resolve(limit, wired_mib, bits=16, memory=48 * GIB):
        return resolve_context_limit(
            limit,
            weights_bytes=wb,
            bytes_per_token=per[bits],
            wired_bytes=wired_mib * 2**20,
            memory_size=memory,
        )

    table = {  # wired limit (MiB): (main, kv8)
        26 * 1024: (73_728, 139_264),
        27 * 1024: (111_616, 209_920),
        28 * 1024: (148_480, 262_144),
        30 * 1024: (223_232, 262_144),
        31_764: (262_144, 262_144),
        46_080: (262_144, 262_144),
    }
    for mib, want in table.items():
        assert (resolve("auto", mib)[0], resolve(None, mib, 8)[0]) == want, mib
    assert resolve("auto", 31_763) == (261_120, "auto")
    # the 48 GB M4 Pro's 40.2 GB default working set
    assert resolve("auto", 38_338) == (262_144, "auto")
    # explicit: fits as given, or refused
    assert resolve(73_728, 26 * 1024) == (73_728, "explicit")
    assert resolve(1, 26 * 1024) == (1, "explicit")
    msg = expect_raises(StartupError, resolve, 262_144, 26 * 1024)
    assert "context limit 262,144 needs" in msg and "the most that fits is 73,728 tokens" in msg, (
        msg
    )
    assert "(--context-limit auto)" in msg and "sudo sysctl iogpu.wired_limit_mb=31764" in msg, msg
    # the suggested limit is capped at physical memory less 6 GiB: 36 GB -> 30,720 MiB
    msg = expect_raises(StartupError, resolve, 262_144, 27 * 1024, memory=36 * GIB)
    assert "most that fits is 111,616" in msg and "iogpu.wired_limit_mb=30720" in msg, msg
    assert "which holds 223,232 tokens" in msg, msg
    msg = expect_raises(StartupError, resolve, 262_144, 30_720, memory=36 * GIB)
    assert "already at physical memory less 6 GiB" in msg, msg
    # auto below 4,096 tokens: 24,708 MiB needed; a 32 GB Mac can raise its limit, a 24 GB one cannot
    msg = expect_raises(StartupError, resolve, "auto", 24 * 1024, memory=32 * GIB)
    assert (
        "Flint needs 24,708 MiB for 4,096 tokens" in msg and "iogpu.wired_limit_mb=24708" in msg
    ), msg
    msg = expect_raises(StartupError, resolve, "auto", 16 * 1024, memory=24 * GIB)
    assert "Flint needs 24,708 MiB for 4,096 tokens" in msg and "cannot hold it" in msg, msg
    msg = expect_raises(StartupError, resolve, 4096, 22 * 1024, memory=24 * GIB)
    assert "no history fits beside the weights" in msg, msg
    for bad in ("bogus", 0, 262_145, True, 1.5):
        expect_raises(StartupError, resolve, bad, 46_080)
