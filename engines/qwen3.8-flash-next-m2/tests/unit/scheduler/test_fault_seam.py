"""The fault-injection seam: how it is armed, when it fires, and what a fired fault does to the
engine (on MLX's CPU build, with the tiny checkpoint's engine on the stock ops)."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from jitengine_qwen3_8_flash_next_m2.kvcache.kv_cache import KVSlot
from jitengine_qwen3_8_flash_next_m2.scheduler.faults import FaultSeam, RowFaultError, StepGuard


def _check(seam, kind="ple_row", keys=(7,), row_of=None, owners=None, key=None):
    seam.check(kind, np.array(keys), None if row_of is None else np.array(row_of), owners, key=key)


# ---------------------------------------------------------------- arming
@pytest.mark.parametrize("option", [None, {}, False])
def test_no_option_means_no_seam(option):
    assert FaultSeam.from_option(option) is None


def test_a_ple_row_option_arms_a_seam_on_that_row():
    seam = FaultSeam.from_option({"ple_row": 12})
    assert (seam.kind, seam.key, seam.request_id, seam.armed, seam.fired) == (
        "ple_row",
        12,
        None,
        True,
        0,
    )
    assert FaultSeam.from_option({"ple_row": [5]}).key == 5
    assert FaultSeam.from_option({"ple_row": "9", "request_id": 4}).request_id == "4"


def test_an_expert_record_option_arms_a_seam_on_a_layer_and_expert():
    seam = FaultSeam.from_option({"expert_record": [2, 5], "request_id": "r"})
    assert (seam.kind, seam.key, seam.request_id) == ("expert_record", (2, 5), "r")
    assert FaultSeam.from_option({"expert_record": (1, 0)}).key == (1, 0)


@pytest.mark.parametrize(
    "option, message",
    [
        ("ple_row", "must be a dict"),
        ({"request_id": "r"}, "names neither"),
        ({"ple_row": [1, 2]}, "exactly ONE fault"),
        ({"expert_record": 3}, r"\[layer, expert\]"),
        ({"expert_record": [1, 2, 3]}, r"\[layer, expert\]"),
    ],
)
def test_a_malformed_option_is_a_value_error(option, message):
    with pytest.raises(ValueError, match=message):
        FaultSeam.from_option(option)


def test_an_unknown_kind_is_a_value_error():
    with pytest.raises(ValueError, match="expected ple_row or expert_record"):
        FaultSeam("disk", 1)


# ---------------------------------------------------------------- firing
def test_a_read_of_the_armed_key_raises_for_the_rows_that_asked_and_disarms():
    seam = FaultSeam("ple_row", 7)
    with pytest.raises(RowFaultError) as e:
        _check(seam, keys=[3, 7, 9, 7], row_of=[0, 2, 1, 1])
    assert e.value.rows == {1, 2}
    assert isinstance(e.value, OSError) and "ple_row" in str(e.value) and "[1, 2]" in str(e.value)
    assert (seam.armed, seam.fired) == (False, 1)
    _check(seam, keys=[7], row_of=[0])  # one fault fires once


def test_a_read_without_row_numbers_is_row_zero():
    seam = FaultSeam("ple_row", 7)
    with pytest.raises(RowFaultError) as e:
        _check(seam, keys=[7])
    assert e.value.rows == {0}


def test_other_keys_and_other_kinds_do_not_fire():
    seam = FaultSeam("ple_row", 7)
    _check(seam, keys=[1, 2, 3])
    _check(seam, kind="expert_record", keys=[7])
    _check(seam, keys=[])
    assert seam.armed and seam.fired == 0


def test_the_key_argument_overrides_the_armed_key():
    """The expert pool names the record by its global id in `key`; the seam's own key is the
    (layer, expert) pair."""
    seam = FaultSeam("expert_record", (1, 2))
    _check(seam, kind="expert_record", keys=[10], key=9)
    assert seam.armed
    with pytest.raises(RowFaultError, match=r"\(1, 2\)"):
        _check(seam, kind="expert_record", keys=[9, 10], row_of=[0, 1], key=9)


def test_a_request_id_limits_the_fault_to_that_requests_rows():
    seam = FaultSeam("ple_row", 7, request_id="bad")
    owners = ["good", "bad", None]
    _check(seam, keys=[7], row_of=[0], owners=owners)  # the key is read, for another request
    _check(seam, keys=[7], row_of=[2], owners=owners)  # for no request
    _check(seam, keys=[7], row_of=[5], owners=owners)  # for a row outside the unit
    _check(seam, keys=[7], row_of=[0], owners=None)
    assert seam.armed
    with pytest.raises(RowFaultError) as e:
        _check(seam, keys=[7, 7, 7], row_of=[0, 1, 2], owners=owners)
    assert e.value.rows == {1}


def test_the_row_fault_error_keeps_its_rows_as_ints():
    err = RowFaultError(np.array([2, 0]), "boom")
    assert err.rows == {0, 2} and all(type(r) is int for r in err.rows) and str(err) == "boom"


def test_racing_threads_fire_the_fault_once():
    seam = FaultSeam("ple_row", 7)
    barrier, raised = threading.Barrier(8), []

    def read():
        barrier.wait()
        try:
            _check(seam, keys=[7])
        except RowFaultError:
            raised.append(1)

    threads = [threading.Thread(target=read) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(raised) == 1 and seam.fired == 1


# ---------------------------------------------------------------- StepGuard
def test_restore_rewinds_a_delta_net_layer_and_a_ple_slot_and_release_forgets_them():
    delta = SimpleNamespace(conv="conv0", ssm="ssm0")
    ple = SimpleNamespace(context="ctx0", conv="pconv0")
    st = SimpleNamespace(layers=[delta], ple={3: ple}, tokens=[1])
    guard = StepGuard([st])
    st.tokens, delta.conv, delta.ssm = [1, 2], "conv1", "ssm1"
    ple.context, ple.conv = "ctx1", "pconv1"
    guard.restore()
    assert (st.tokens, delta.conv, delta.ssm) == ([1], "conv0", "ssm0")
    assert (ple.context, ple.conv) == ("ctx0", "pconv0")
    st.tokens, delta.conv = [1, 2, 3], "conv2"
    guard.release()
    guard.restore()  # nothing left to undo
    assert (st.tokens, delta.conv) == ([1, 2, 3], "conv2")


def test_a_guard_covers_every_row_of_the_unit():
    slots = [
        KVSlot(SimpleNamespace(num_key_value_heads=1, head_dim=4, indexer_head_dim=4), 64)
        for _ in range(2)
    ]
    states = [SimpleNamespace(layers=[s], ple={}, tokens=[0]) for s in slots]
    guard = StepGuard(states)
    for st, s in zip(states, slots):
        st.tokens, s.offset = [0, 1], 9
    guard.restore()
    assert [(st.tokens, s.offset) for st, s in zip(states, slots)] == [([0], 0), ([0], 0)]


# ---------------------------------------------------------------- the engine
@pytest.fixture
def armed_engine(cpu_flashnext_engine):
    """`armed(request_id)`: a fresh engine whose expert pool starts empty, with one fault armed on
    the first record the warm-up left unread. Armed after the engine is built, because the warm-up
    request has no id and would otherwise take the fault."""

    def armed(request_id):
        e = cpu_flashnext_engine(
            prefix_store_entries=0, expert_pool_seed_per_layer=0, admit_window_s=0.3
        )
        pool = e.expert_pool
        cold = np.flatnonzero(np.asarray(pool.slot_for_id) < 0)
        assert cold.size, "the warm-up read every expert record"
        layer, expert = divmod(int(cold[0]), pool.num_experts)
        seam = FaultSeam("expert_record", (layer, expert), request_id)
        e.fault_seam, pool.seam = seam, seam
        return e, seam

    return armed


def _req(request_id, shift=0, n=30):
    from jitengine_qwen3_8_flash_next_m2.engine.io_struct import GenRequest

    # long enough that every layer routes to nearly every expert, so the cold record is read
    return GenRequest(
        prompt_ids=[t + shift for t in range(10, 10 + n)], max_tokens=3, request_id=request_id
    )


def test_a_fault_in_a_wave_fails_only_the_requests_that_read_it(armed_engine):
    e, seam = armed_engine("bad")
    good, bad, other = e.generate_batch([_req("good"), _req("bad", 1), _req("other", 2)])
    assert bad.finish_reason == "error" and bad.token_ids == []
    assert "injected expert_record read fault" in bad.extra["error"] and "[1]" in bad.extra["error"]
    for r in (good, other):
        assert r.finish_reason == "length" and len(r.token_ids) == 3
    st = e.stats()
    assert (seam.fired, st["fault_injections"], st["row_faults"]) == (1, 1, 1)
    assert st["error_types"] == {"RowFaultError": 1} and e.health()["status"] == "ok"
    # rolled back and re-run, the survivors prefill what a lone request prefills: same first token
    assert good.token_ids[0] == e.generate(_req("solo")).token_ids[0]


def test_a_fault_on_a_lone_row_fails_that_request_and_the_engine_serves_on(armed_engine):
    e, seam = armed_engine("bad")
    with pytest.raises(RowFaultError, match="injected expert_record read fault"):
        e.generate(_req("bad"))
    assert seam.fired == 1 and e.stats()["fault_injections"] == 1
    assert e.health()["status"] == "ok"
    ok = e.generate(_req("bad"))  # disarmed: the same request id now runs
    assert ok.finish_reason == "length" and len(ok.token_ids) == 3


def test_an_unarmed_engine_reports_no_injections(cpu_flashnext_engine):
    e = cpu_flashnext_engine()
    assert e.fault_seam is None and e.stats()["fault_injections"] == 0


def test_an_expert_record_outside_the_pack_is_refused_at_load(cpu_flashnext_engine):
    with pytest.raises(ValueError, match=r"expert_record \(99, 0\) outside the pack"):
        cpu_flashnext_engine(fault_inject={"expert_record": [99, 0]})
