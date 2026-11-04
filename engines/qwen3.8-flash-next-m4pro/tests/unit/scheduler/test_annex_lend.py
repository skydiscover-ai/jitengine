"""What the annex borrows (`SchedulingMixin._annex_policy`, on stand-ins): the bytes the prefix store's
entries held, less any entry whose write failed, never the cap they leave unfilled. One-row decode's
transients and MLX's buffer cache live in that slack; an annex sized to the cap put a 24 GB plan at MLX's
memory limit. So with the spill on, one-row decode's footprint is never above the
resident store's."""

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")

from jitengine_qwen3_8_flash_next_m4pro.scheduler.scheduler import SchedulingMixin

SLOT = 2_764_800  # a pack16 slot of the checkpoint
CAP = 587_000_000  # the plan's prefix-store cap at 24 and 40 GB
HELD = 512_501_792  # the benchmark's two entries ("[store] spilled: 512.5 MB" in the benchmark log)


class Pool:
    def __init__(self, ann_slots: int):
        self.ann_slots, self.slot_bytes = ann_slots, SLOT
        self.ann_active, self.ext_active = False, True
        self.grown: list[int] = []

    def regrow_ann(self, n: int) -> int:
        self.grown.append(int(n))
        self.ann_active = n > 0
        return int(n)

    def release_ann(self) -> int:
        self.ann_active = False
        return 0


class Store:
    def __init__(self, held: int, kept: int = 0):
        self.cap_bytes, self.bytes, self.spill_kept = CAP, held, kept
        self.spilled = False
        self.spiller = SimpleNamespace(
            stats=lambda: {
                "spill_entries_written": 2,
                "spill_write_seconds": 0.0,
                "spill_file_bytes": held,
            },
            _pending=[],
        )

    def spill(self) -> int:
        self.spilled = True
        return self.bytes - self.spill_kept

    def restore(self) -> None:
        self.spilled = False


def _sched(held: int, kept: int = 0) -> SchedulingMixin:
    s = SchedulingMixin()
    s.expert_pool = Pool(CAP // SLOT)
    s.store = Store(held, kept)
    s._pending, s._scores = [], []
    s._rows = [SimpleNamespace(phase="decode", retired=False)]
    s.model = SimpleNamespace(drop_compiled=lambda: None)
    return s


@pytest.mark.parametrize(
    "held, kept, slots",
    [
        (HELD, 0, 185),  # the benchmark: 185 of the cap's 212
        (CAP, 0, 212),  # a full store lends the whole annex
        (HELD, 281_217_040, 83),  # an entry whose write failed stays resident and lends nothing
        (0, 0, 0),  # an empty store lends nothing
    ],
)
def test_the_annex_borrows_what_the_entries_held(held, kept, slots):
    s = _sched(held, kept)
    assert s.expert_pool.ann_slots == 212
    s._annex_policy(True)
    assert s.store.spilled and s.expert_pool.grown == [slots]
    # the decode's footprint with the spill on: the kept entries plus the annex, never above the
    # resident store's
    assert kept + slots * SLOT <= held


def test_a_second_row_takes_the_annex_back_and_restores_the_store():
    s = _sched(HELD)
    s._annex_policy(True)
    assert s.expert_pool.ann_active
    s._rows.append(SimpleNamespace(phase="prefill", retired=False))
    s._annex_policy(False)
    assert not s.expert_pool.ann_active and not s.store.spilled
