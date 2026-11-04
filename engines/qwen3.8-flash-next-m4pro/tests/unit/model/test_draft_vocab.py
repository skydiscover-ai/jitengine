"""The MTP draft's subset readout (model.draft_vocab) on small stand-ins: its draft against the full
argmax, the per-request prompt tail (dedup, first-occurrence order, truncation, reset between
requests), the shipped id file, the option, its charge beside the head, and its /stats counters."""

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from jitengine_qwen3_8_flash_next_m4pro import cli  # noqa: E402
from jitengine_qwen3_8_flash_next_m4pro.linear import QLinear  # noqa: E402
from jitengine_qwen3_8_flash_next_m4pro.model import draft_vocab as dv  # noqa: E402
from jitengine_qwen3_8_flash_next_m4pro.scheduler import decode  # noqa: E402
from jitengine_qwen3_8_flash_next_m4pro.scheduler.scheduler import SchedulingMixin  # noqa: E402
from jitengine_qwen3_8_flash_next_m4pro.server.stats import Stats  # noqa: E402

N, K_IN = 2048, 128


def _lm(seed, n=N, k=K_IN, hot=(), dup=None):
    """A 4-bit g32 QLinear with bf16 metadata. `hot` rows are scaled up; `dup` = (a, b) gives row b
    row a's exact bytes."""
    w = np.asarray(mx.random.normal((n, k), key=mx.random.key(seed))) * k**-0.5
    for h in hot:
        w[h] *= 6.0
    if dup is not None:
        w[dup[1]] = w[dup[0]]
    q, s, b = mx.quantize(mx.array(w), group_size=32, bits=4)
    lin = QLinear(q, s.astype(mx.bfloat16), b.astype(mx.bfloat16), 32, 4)
    mx.eval(lin.weight, lin.scales, lin.biases)
    return lin


def _x(seed, k=K_IN):
    return mx.random.normal((1, 1, k), key=mx.random.key(seed))


def _want(full_row, s_ids):
    """The rule: the lowest id among the maxima of the full logits restricted to S."""
    v = full_row[s_ids]
    return int(s_ids[v == v.max()].min())


def test_the_draft_is_the_full_argmax_over_the_subset_and_its_rows_are_lm_heads():
    lm = _lm(1)
    rng = np.random.default_rng(7)
    base = np.sort(rng.choice(N, N // 4, replace=False)).astype(np.int32)
    ro = dv.SubsetReadout(lm, base, p_max=64)
    inside = outside = 0
    for i in range(48):
        prompt = rng.choice(N, 80).tolist()
        n, _ = ro.prepare(prompt)
        x = _x(100 + i)
        full = np.asarray(lm(x).reshape(-1))
        s_ids = np.concatenate([base, ro.tail_ids[:n]])
        got = int(np.asarray(ro(x))[0])
        assert got == _want(full, s_ids), i
        top = int(np.argmax(full))
        if top in set(s_ids.tolist()):
            inside += 1
            assert got == top == int(np.asarray(dv.draft_id(lm, None, x))[0])
        else:
            outside += 1
        # the copies' logits are lm_head's rows, by bytes
        xb = x.reshape(1, K_IN)
        assert np.array_equal(
            np.asarray(ro.base(xb)).view(np.uint32), full[base].view(np.uint32)[None]
        )
        tail = np.asarray(ro.tail(xb)).reshape(-1)[:n]
        assert np.array_equal(tail.view(np.uint32), full[ro.tail_ids[:n]].view(np.uint32))
    assert inside > 0 and outside > 0, (inside, outside)  # both branches of the rule ran


def test_ties_go_to_the_lowest_id_wherever_it_sits():
    """Rows 5 and 9 carry the same bytes and are scaled up, so they tie at the maximum: the full
    argmax is 5, and so is the subset's whether 5 is in the tail and 9 in the base or the other way."""
    lm = _lm(2, hot=[5, 9], dup=(5, 9))
    x = mx.dequantize(lm.weight[5:6], lm.scales[5:6], lm.biases[5:6], group_size=32, bits=4)
    x = x.astype(mx.float32).reshape(1, 1, K_IN)
    full = np.asarray(lm(x).reshape(-1))
    assert full[5] == full[9] == full.max() and int(np.argmax(full)) == 5
    for base, prompt in (([9, 100, 200], [5]), ([5, 100, 200], [9]), ([100, 200], [9, 5])):
        ro = dv.SubsetReadout(lm, np.array(base, np.int32), p_max=8)
        ro.prepare(prompt)
        assert int(np.asarray(ro(x))[0]) == 5, (base, prompt)


def test_the_prompt_tail_dedups_keeps_first_occurrence_order_and_truncates():
    lm = _lm(3)
    ro = dv.SubsetReadout(lm, np.arange(10, dtype=np.int32), p_max=4)
    new, cut = ro.prompt_ids([5, 12, 11, 12, 3, 13, 11, -1, N + 7, 14, 15])
    assert (
        new.tolist() == [12, 11, 13, 14] and cut
    )  # 5 and 3 are in the base; -1 and N+7 are not rows
    new, cut = ro.prompt_ids([12, 11, 12])
    assert new.tolist() == [12, 11] and not cut
    n, cut = ro.prepare(list(range(10, 30)))
    assert (n, cut) == (4, True) and ro.tail_ids[:4].tolist() == [10, 11, 12, 13]


def test_the_tail_is_reset_between_requests_and_never_copies_the_base():
    hot = 700
    lm = _lm(4, hot=[hot])
    r = slice(hot, hot + 1)
    x = mx.dequantize(lm.weight[r], lm.scales[r], lm.biases[r], group_size=32, bits=4)
    x = x.astype(mx.float32).reshape(1, 1, K_IN)
    assert int(np.argmax(np.asarray(lm(x).reshape(-1)))) == hot
    base = np.arange(0, 512, dtype=np.int32)
    ro = dv.SubsetReadout(lm, base, p_max=8)
    w0, s0 = ro.base.weight, ro.base.scales
    assert ro.prepare([hot, 600, hot]) == (2, False)
    assert int(np.asarray(ro(x))[0]) == hot
    mask = np.asarray(ro.mask)
    assert (mask[:2] == 0).all() and np.isneginf(mask[2:]).all()
    assert np.asarray(ro.id_map)[base.size : base.size + 2].tolist() == [hot, 600]
    # the next request's prompt lacks the hot id: its row is masked, not drafted
    assert ro.prepare([601]) == (1, False)
    mask = np.asarray(ro.mask)
    assert mask[0] == 0 and np.isneginf(mask[1:]).all()
    assert int(np.asarray(ro(x))[0]) != hot
    assert ro.prepare([]) == (0, False) and np.isneginf(np.asarray(ro.mask)).all()
    assert int(np.asarray(ro(x))[0]) in set(base.tolist())
    assert (
        ro.base.weight is w0 and ro.base.scales is s0
    )  # the base is the load-time copy, untouched


def test_full_is_the_heads_own_readout():
    lm = _lm(5)
    for i in range(8):
        x = _x(200 + i)
        want = mx.argmax(lm(x).reshape(-1)).reshape(1)
        got = dv.draft_id(lm, None, x)
        assert (
            got.dtype == want.dtype
            and got.shape == (1,)
            and np.array_equal(np.asarray(got), np.asarray(want))
        )


def test_the_option():
    assert dv.mode_of({}) == dv.mode_of(None) == "subset" and dv.MODE_DEFAULT == "subset"
    assert (
        dv.mode_of({"draft_vocab": "full"}) == "full"
        and dv.mode_of({"draft_vocab": " Subset "}) == "subset"
    )
    with pytest.raises(ValueError):
        dv.mode_of({"draft_vocab": "top64k"})
    a = cli.build_parser().parse_args(["serve", "--model-path", "m", "--draft-vocab", "full"])
    assert a.draft_vocab == "full" and "draft_vocab" in cli.SERVE_OPTIONS
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["serve", "--model-path", "m", "--draft-vocab", "half"])


def test_the_shipped_ids():
    raw = np.load(dv.IDS_FILE, allow_pickle=False)
    assert raw.dtype == np.int32 and raw.ndim == 1 and raw.size in (16384, 32768, 65536)
    ids = dv.load_ids(248320)
    assert ids.dtype == np.int32 and ids.size == raw.size and (np.diff(ids) > 0).all()
    assert ids[0] >= 0 and ids[-1] < 248320 and set(ids.tolist()) == set(raw.tolist())


def test_bad_id_files_are_refused(tmp_path):
    for bad in (
        np.array([3, 1, 3], np.int32),
        np.array([-1, 2], np.int32),
        np.array([1, 99], np.int32),
        np.zeros((2, 2), np.int32),
        np.array([0.5, 1.0]),
    ):
        p = tmp_path / "ids.npy"
        np.save(p, bad)
        with pytest.raises(ValueError):
            dv.load_ids(50, str(p))


class _Scan:
    """Checkpoint headers for an untied 248,320 x 2,560 4-bit g32 lm_head with bf16 metadata."""

    prefix = "language_model."

    def __init__(self, rows=248320, cols=2560, gs=32):
        w, g = cols * 4 // 32, cols // gs
        hdr = {}
        off = 0
        for part, dt, shape, nb in (
            ("weight", "U32", [rows, w], rows * w * 4),
            ("scales", "BF16", [rows, g], rows * g * 2),
            ("biases", "BF16", [rows, g], rows * g * 2),
        ):
            hdr[f"{self.prefix}lm_head.{part}"] = {
                "dtype": dt,
                "shape": shape,
                "data_offsets": [off, off + nb],
            }
            off += nb
        self.weight_map = {k: "s0" for k in hdr}
        self.headers = {"s0": (hdr, 0)}


def _cfg():
    return SimpleNamespace(vocab_size=248320, quant_of=lambda name: (4, 32))


def test_row_bytes_follow_the_dense_policy():
    scan, cfg = _Scan(), _cfg()
    assert dv.readout_row_bytes(scan, cfg, "float32", "rb") == 1600  # served: bf16 metadata kept
    assert (
        dv.readout_row_bytes(scan, cfg, "float32", "stock") == 1280 + 2 * 80 * 4
    )  # stored as float32
    assert dv.readout_row_bytes(scan, cfg, "bfloat16", "stock") == 1600


def _sched(mode, pool_slots=9150, head=1_470_955_171 + (256 << 20)):
    return SimpleNamespace(
        plan=SimpleNamespace(pool_slots=pool_slots),
        mtp_bytes=head,
        mtp_on=True,
        options={} if mode is None else {"draft_vocab": mode},
        cfg=_cfg(),
        ckpt_scan=_Scan(),
        compute_dtype_name="float32",
        dense_kernel="rb",
    )


def test_the_subset_is_charged_with_the_head_and_full_charges_the_head_alone():
    rec = 2_764_800
    full = _sched("full")
    left_full = SchedulingMixin._charge_mtp_head(full, 2197, rec)
    assert (
        left_full == 2197 - (-(-full.mtp_bytes // rec)) and full.draft_vocab_plan is None
    )  # the charge without a draft vocabulary
    sub = _sched(None)
    left = SchedulingMixin._charge_mtp_head(sub, 2197, rec)
    plan = sub.draft_vocab_plan
    k = dv.load_ids(248320).size
    assert plan is not None and plan.rows == k + dv.P_MAX and plan.row_bytes == 1600
    assert plan.bytes == (k + dv.P_MAX) * (1600 + 4) + dv.P_MAX * 4 + 248320
    assert (
        left == 2197 - (-(-(sub.mtp_bytes + plan.bytes) // rec)) and sub.mtp_bytes == full.mtp_bytes
    )
    assert (2197 - left) * rec >= sub.mtp_bytes + plan.bytes
    off = _sched(None, pool_slots=4000)  # below the slot threshold: no head, no readout
    assert SchedulingMixin._charge_mtp_head(off, 1339, rec) == 1339 and off.draft_vocab_plan is None
    bad = _sched("half")
    with pytest.raises(ValueError):
        SchedulingMixin._charge_mtp_head(bad, 2197, rec)


def test_stats_and_the_decode_hook():
    s = Stats()
    snap = s.snapshot()
    assert (
        snap["draft_vocab"],
        snap["draft_subset_rows"],
        snap["draft_prompt_rows"],
        snap["draft_prompt_truncations"],
        snap["draft_prompt_installs"],
    ) == ("off", 0, 0, 0, 0)
    lm = _lm(6)
    ro = dv.SubsetReadout(lm, np.arange(16, dtype=np.int32), p_max=4)
    row = SimpleNamespace(prompt=[1, 20, 21, 20, 22, 23, 24])
    decode._draft_prompt(s, SimpleNamespace(readout=ro), row)
    snap = s.snapshot()
    assert (
        snap["draft_vocab"],
        snap["draft_subset_rows"],
        snap["draft_prompt_rows"],
        snap["draft_prompt_truncations"],
        snap["draft_prompt_installs"],
    ) == ("subset", 20, 4, 1, 1)
    decode._draft_prompt(s, SimpleNamespace(readout=None), row)
    assert (s.snapshot()["draft_vocab"], s.snapshot()["draft_subset_rows"]) == ("full", 0)
    decode._draft_prompt(s, SimpleNamespace(), row)  # a stand-in head: nothing recorded
    assert s.snapshot()["draft_prompt_installs"] == 1
