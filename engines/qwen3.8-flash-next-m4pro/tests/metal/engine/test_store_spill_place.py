"""An engine whose expert pack sits in a read-only directory starts, serves the same tokens, and
writes nothing under the pack directory (the store spill does not run makedirs/mkstemp in
`<pack dir>/jitengine-spill`, which would raise PermissionError on a read-only pack location and
keep the server from listening). With the place
beside it unwritable too, the spill is off and the engine still serves the same tokens."""

import os
import shutil
import stat

OPTS = dict(
    prefix_grid_tokens=64,
    prefix_min_divergence_tokens=16,
    landing_slots=2,
    prefetch_reserve=2,
    pool_ext_slots=4,
    ext_scratch_tokens=48,
    log_requests=False,
)
ANNEX = 6
PROMPT = list(range(1000, 1100)) + [5, 6, 7]
RO = stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH


def _engine(monkeypatch, model, pack, annex=ANNEX, **opts):
    from jitengine_qwen3_8_flash_next_m4pro.engine import engine as fne

    plan_budget = fne.plan_budget

    def capped(*args, **kwargs):
        plan = plan_budget(*args, **kwargs)
        plan.pool_slots = 16  # of 32 records, as test_store_spill_annex
        plan.store_spill_slots = annex
        return plan

    with monkeypatch.context() as mp:
        mp.setattr(fne, "plan_budget", capped)
        return fne.create_engine(model, expert_pack_path=pack, **OPTS, **opts)


def _tokens(e):
    from jitengine_qwen3_8_flash_next_m4pro.engine.io_struct import GenRequest

    return [
        e.generate(GenRequest(prompt_ids=PROMPT, max_tokens=24, stop_at_eos=False)).token_ids
        for _ in range(2)
    ]


def test_a_read_only_pack_directory_starts_serves_and_is_never_written(
    tiny_flashnext, requires_mlx, tmp_path, monkeypatch
):
    model, pack = tiny_flashnext()
    ro = tmp_path / "packs" / "ro"
    ro.mkdir(parents=True)
    shutil.copy2(pack, ro / "tiny.experts")
    ro.chmod(RO)
    try:
        before = (sorted(os.listdir(ro)), os.stat(ro).st_mtime_ns)
        ref = _engine(monkeypatch, model, str(ro / "tiny.experts"), annex=0)  # no spill planned
        assert getattr(ref, "store_spiller", None) is None
        want = _tokens(ref)

        e = _engine(monkeypatch, model, str(ro / "tiny.experts"))
        assert e.health()["status"] == "ok" and e.store_spiller is not None
        assert e.stats()["spill_dir"] == str(tmp_path / "packs" / "jitengine-spill")
        assert _tokens(e) == want
        assert e.stats()["expert_pool_ann_regrows"] >= 1  # the spill ran and the annex was lent
        assert (sorted(os.listdir(ro)), os.stat(ro).st_mtime_ns) == before

        e.store_spiller.close()  # its file and the directory it made go (the engine is done)
        assert not (tmp_path / "packs" / "jitengine-spill").exists()
        (tmp_path / "packs").chmod(RO)  # the place beside it cannot be written either
        off = _engine(monkeypatch, model, str(ro / "tiny.experts"))
        s = off.stats()
        assert (
            off.health()["status"] == "ok"
            and off.store_spiller is None
            and off.store.spiller is None
        )
        assert "spill_dir" not in s and "jitengine-spill" in s["spill_off"]
        assert _tokens(off) == want
        s = off.stats()
        assert s["expert_pool_ann_regrows"] == 0 and s["expert_pool_ann_live"] == 0  # never lent
        assert (sorted(os.listdir(ro)), os.stat(ro).st_mtime_ns) == before
    finally:
        (tmp_path / "packs").chmod(0o755)
        ro.chmod(0o755)
