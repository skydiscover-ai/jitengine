"""The solo routed gather (`SpeculativeGatherMixin._gather_dev`) on the tiny engine: it launches the
grouped kernel at `grouped_expert_qmv.solo_config_for` (VECS `SOLO_VECS`), and its rows are the rows
the same calls give at `config_for`'s VECS, bit for bit.

VECS never enters the kernel's k-order, so the one-row gather may take its own VECS; the decode,
verify and fixture-digest tests hold the end-to-end claim, this one the launch and the rows.
"""

import numpy as np


def _bytes(mx, a) -> bytes:
    return np.asarray(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a).tobytes()


def test_the_solo_gather_runs_at_solo_vecs_with_the_configs_rows(
    flashnext_engine, requires_mlx, monkeypatch
):
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.engine.io_struct import GenRequest
    from jitengine_qwen3_8_flash_next_m4pro.kernels import grouped_expert_qmv as mk

    e = flashnext_engine(pool_ext_slots=0, prefix_store_entries=0)
    e.generate(GenRequest(prompt_ids=list(range(100, 140)), max_tokens=4, stop_at_eos=False))
    pool = e.model.layers[0].moe.pool
    assert pool is not None and pool is e.expert_pool
    E = pool.num_experts
    launches: list = []
    for name in ("grouped_qmv_x", "grouped_qmv_x_p16"):
        real = getattr(mk, name)

        def rec(*a, _real=real, **kw):
            launches.append(a[-4])  # vecs: the first of the four config arguments
            return _real(*a, **kw)

        monkeypatch.setattr(mk, name, rec)
    seen = 0
    for layer in e.model.layers:
        moe = layer.moe
        k = moe.top_k
        resident = np.flatnonzero(
            pool.slot_for_id[moe.layer_idx * E : (moe.layer_idx + 1) * E] >= 0
        )
        if resident.size < k:
            continue
        inds = mx.array(resident[:k].astype(np.int32))
        tab = pool.device_slot_table()
        slots_dev = mx.take(tab, inds + int(moe.layer_idx * E))
        x2 = mx.random.normal((1, moe.hid), key=mx.random.key(moe.layer_idx)).astype(e.model.dtype)
        launches.clear()
        solo = moe._gather_dev(x2, slots_dev, k)
        mx.eval(solo)
        assert launches == [mk.SOLO_VECS] * 3 and mk.SOLO_VECS == 1, launches
        shipped = {
            mk.config_for(lin.out_dims, lin.in_dims, x2.dtype)[0]
            for lin in (moe.gate_proj, moe.down_proj)
        }
        assert 1 not in shipped, shipped
        for v in sorted(shipped | {2}):
            monkeypatch.setattr(mk, "SOLO_VECS", v)
            other = moe._gather_dev(x2, slots_dev, k)
            monkeypatch.setattr(mk, "SOLO_VECS", 1)
            assert _bytes(mx, solo) == _bytes(mx, other), (moe.layer_idx, v)
        seen += 1
    assert seen > 0, "no layer had top_k resident experts"
