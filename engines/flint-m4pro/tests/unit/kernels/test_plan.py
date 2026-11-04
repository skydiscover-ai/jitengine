"""The megakernel plan and the rendered source: layouts, the attention grid, descriptors, parameters, verify rows."""

from jitengine_flint_m4pro.model.config import (
    load_config,
    tensor_index,
)
from jitengine_flint_m4pro.weight_banks import BankPlan


def expect_raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return
    raise AssertionError(f"{fn.__name__}{a} did not raise {exc.__name__}")


def test_megakernel_plan(flint_headers):  # noqa: PLR0915
    import numpy as np

    from jitengine_flint_m4pro.kernels import metal_src as MS
    from jitengine_flint_m4pro.kernels import plan as P

    # layout 6d on the per-position grid S(q) = 8 ceil((q + 1) / span), span 512 while q + 1 <= T
    # else 1,024; the engine's partitions carry T only (no N_max anywhere), <= 64 chunks per kv head on span 512
    assert P.DEFAULT_POLICY == "l6d" and P.SIXD_T == {"l6d": P.T_INF, "l6d2": 1, "l6dk": 32768}
    assert P.default_policy(16) == "l6d" and P.default_policy(8) == P.KV8_POLICY == "l6dk"
    assert P.is_6d("l6d") and P.is_6d("l6d2") and P.is_6d("l6dk") and not P.is_6d("strided")
    for pol, T in P.SIXD_T.items():
        ap = P.attn_params(pol)
        assert ap == P.attn_params(pol, 1025) == P.attn_params(pol, 262144)  # n_max is ignored
        assert (
            ap["six_t"] == T
            and ap["p_f"] == 0
            and ap["p_s"] == P.PARTS_SLIDE
            and P.verify_capable(ap)
        )
        assert (ap["f_sub"], ap["f_asg"], ap["f_flags"]) == (1, 8, 0)
        assert (
            P.make_params(
                pos=0,
                token=1,
                forced_next=None,
                sel_mode=0,
                top_k=1,
                top_p=1.0,
                temperature=0.0,
                u=0.0,
                poison=False,
                cap=8,
                theta=1e4,
                **ap,
            )[P.P_6T]
            == T
        )
        expect_raises(ValueError, P.chunk_sizes, 1025, pol)
    q = np.arange(262144)
    for T, cmax in ((P.T_INF, 64), (1, 128), (32768, 128)):
        S = np.array([P.pos_grid(int(x), T) for x in q[::97]] + [P.pos_grid(262143, T)])
        assert (S % 8 == 0).all() and (S >= 8).all()
        C = -(-(np.r_[q[::97], 262143] + 1) // S)
        assert C.max() <= cmax and C.max() >= cmax - 1
    assert (
        P.pos_grid(511, P.T_INF) == 8
        and P.pos_grid(512, P.T_INF) == 16
        and P.pos_grid(1023, 1) == 8
    )
    assert (
        P.pos_grid(32767, 32768) == 512
        and P.pos_grid(32768, 32768) == 264
        and P.pos_chunks(32768, 32768) == 125
    )
    assert P.pos_grid(262143, P.T_INF) == 4096 and P.pos_grid(262143, 32768) == 2048
    P.check_policy("l6dk")
    P.check_policy("strided")
    expect_raises(ValueError, P.check_policy, "bogus")
    expect_raises(ValueError, P.attn_params, "strided")  # request-fixed partitions need n_max
    # request-fixed partitions (kernel timing only): <= MAX_CHUNKS at any request size
    for n in (1, 100, 1025, 2049, 33793, 66049, 131073, 262144):
        for pol in (
            "strided",
            "strided_all",
            "grid",
            "c4",
            "s32",
            "L60x1a6f1",
            "L20x3a6f0",
            "L120x1a6f4",
        ):
            s_f, s_s, p_f, p_s = P.chunk_sizes(n, pol)
            assert (
                -(-n // s_f) <= P.MAX_CHUNKS
                and -(-P.RING // s_s) <= P.MAX_CHUNKS
                and s_f % 8 == 0
                and s_s % 8 == 0
            )
            assert p_f <= P.MAX_CHUNKS and p_s <= P.MAX_CHUNKS
            ap = P.attn_params(pol, n)
            assert (
                -(-n // ap["s_f"]) * ap["f_sub"] <= P.MAX_CHUNKS
                and 1 <= ap["f_asg"] <= 8
                and ap["six_t"] == 0
            )
    # legacy partition ("strided"): strided full layers up to LONG_FULL, then the long layout-2 stream
    assert P.chunk_sizes(2049, "strided") == (40, 16, 80, 28) and P.chunk_sizes(
        1025, "strided"
    ) == (32, 16, 40, 28)
    assert P.chunk_sizes(33793, "strided") == (568, 16, 0, 28) and P.chunk_sizes(2049, "grid") == (
        32,
        16,
        0,
        0,
    )
    assert P.chunk_sizes(17409, "strided")[2] == 360 and P.chunk_sizes(24577, "strided")[2] == 0
    assert P.verify_capable(P.attn_params("strided", 2049))
    assert not P.verify_capable(P.attn_params("strided", 33793))
    # strided full parts are whole waves of 320 tasks (8 per part) and never a power of two (DRAM channel camping)
    for n in range(1, P.LONG_FULL + 1, 509):
        p_f = P.chunk_sizes(n, "strided")[2]
        assert p_f % 40 == 0 and p_f & (p_f - 1) != 0
    # long full-layer stream (N_max > LONG_FULL): C = 60 chunks per kv head over 6 streaming simdgroups per
    # threadgroup = 240 items on 240 simdgroups at G = 40; request-fixed, never a function of the position
    ap = P.attn_params("strided", 33793)
    assert (ap["f_sub"], ap["f_asg"], ap["f_flags"]) == (1, 6, P.FL_SGFIRST) and -(
        -33793 // ap["s_f"]
    ) == 60
    assert (
        P.attn_params("strided", 2049)["f_asg"] == 8
        and P.attn_params("strided", 258049)["s_f"] == 4304
    )
    # in-kernel error words: every kind is named (decode.py / runner.py messages), the probe and stall seams have words
    assert len({P.P_6T, P.P_STALL, P.P_PROBE, P.P_DIAG}) == 4 and P.P_PROBE < P.P_COS
    for w, s in (
        (7, "grid barrier 7"),
        (P.E_TEAM | 9, "MoE team"),
        (P.E_FLOW_HEADS | 1, "QKV -> ATT) timed out at layer 0"),
        (P.E_FLOW_TASKS | 3, "ATT -> MERGE) timed out at layer 2"),
        (P.E_PREV, "previous pass"),
        (P.E_RING | (39 << 8) | 2, "threadgroup 39, lap 2"),
    ):
        assert s in P.describe_err(w), (w, P.describe_err(w))
    prm = P.make_params(
        pos=5,
        token=1,
        forced_next=None,
        sel_mode=0,
        top_k=1,
        top_p=1.0,
        temperature=0.0,
        u=0.0,
        poison=False,
        cap=8,
        theta=1e4,
        stall=True,
        probe=3,
        **P.attn_params(),
    )
    assert prm[P.P_STALL] == 1 and prm[P.P_PROBE] == 3
    assert P.parse_long("L20x3a6f0") == (20, 3, 6, 0) and P.parse_long("strided") is None
    expect_raises(ValueError, P.parse_long, "L20x3a9f0")
    expect_raises(ValueError, P.parse_long, "L20x3a6f2")
    sched = P.phase_schedule((0, 1))
    assert len(sched) == 343 and sched[-1] == (P.PH["SELECT"], 49) and sched[4] == (P.PH["DGU"], 0)
    assert P.n_barriers((0, 1)) == 150  # MERGE->O, O->MLP (+1 dense), HEAD->SELECT
    assert P.OFF["Y"] + 7 * 2048 * 2 <= P.SCRATCH_BYTES and P.ctr_offset(1) == P.SCRATCH_BYTES
    assert P.MK_SCRATCH_BYTES == P.SCRATCH_BYTES + P.CTR_BYTES and P.CTR_BYTES == 49 * 64 * 64
    assert P.R_QFLAG >= P.R_ATTQ + 32 and P.R_FCNT >= P.R_QFLAG + 40 and P.R_FCNT + 4 <= P.RES_WORDS
    assert P.SCRATCH_BYTES < 32 * 2**20 and all(v % 256 == 0 for v in P.OFF.values())
    assert P.R_ARRIVE % 32 == 0 and P.R_GEN - P.R_ARRIVE >= 32 and P.R_ERR - P.R_GEN >= 32
    prm = P.make_params(
        pos=600,
        token=None,
        forced_next=7,
        sel_mode=1,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        u=0.25,
        poison=False,
        s_f=32,
        s_s=16,
        cap=4096,
        theta=1e4,
    )
    assert prm[P.P_TOKMODE] == 1 and prm[P.P_TOPK] == 20 and prm[P.P_FORCED] == 7
    c = prm[P.P_COS : P.P_COS + 64].view(np.float32)
    assert abs(float(c[0]) - np.cos(600.0)) < 1e-6 and float(
        prm[P.P_INVT : P.P_INVT + 1].view(np.float32)[0]
    ) == np.float32(1 / 0.7)
    cfg = load_config(flint_headers)
    plan = BankPlan(cfg, tensor_index(flint_headers, cfg))
    d = P.pack_descriptors(plan)
    assert (
        d[0 * P.NF + P.D["self_attn.q_proj.weight"]]
        == plan.slots["model.layers.0.self_attn.q_proj.weight"].offset
    )
    assert (
        d[2 * P.NF + P.D_EBANK] == 1
        and d[48 * P.NF + P.D_EBANK] == 2
        and d[0 * P.NF + P.D_EBANK] == 0
    )
    assert d[P.D_GLOBAL + 1] == plan.slots["lm_head.weight"].offset
    for chain in (False, True):
        h, body = MS.render(cfg, P.expert_record_offsets(plan), chain)
        assert f"#define CHAIN {int(chain)}" in h and "grid_sync" in body and "#define DIAG 0" in h
        assert (
            "moe_fused" in body
            and "flow_layer_tasks" in body
            and f"#define S_CTR {P.SCRATCH_BYTES}u" in h
        )
    assert "#define DIAG 1" in MS.render(cfg, P.expert_record_offsets(plan), False, diag=1)[0]
    assert P.P_DIAG < P.P_COS
    # the M = 1 pass reads a threadgroup copy of the descriptor table (offset >> 4 as uint32),
    # verify passes the device table; the encoding is checked lossless on the real table, and refused when it is not
    eo = P.expert_record_offsets(plan)
    for chain in (False, True):
        assert (
            "#define PASS_M1 1" in MS.render(cfg, eo, chain)[0]
            and "dt_[i] =" in MS.render(cfg, eo, chain)[1]
        )
        hv, bv = MS.render(cfg, eo, chain, mr=4)
        assert "#define PASS_M1 0" in hv and "dt_[" not in bv and "hm_" not in bv
    P.check_tg_descriptors(d)
    bad = d.copy()
    bad[P.D["self_attn.q_proj.weight"]] += 8
    expect_raises(ValueError, P.check_tg_descriptors, bad)
    bad = d.copy()
    bad[P.D_GLOBAL] = 1 << 36
    expect_raises(ValueError, P.check_tg_descriptors, bad)


def test_verify_plan(flint_headers):
    import numpy as np

    from jitengine_flint_m4pro.kernels import metal_src as MS
    from jitengine_flint_m4pro.kernels import plan as P

    assert P.VMAX <= 15 + 1  # ring slack: a row never evicts a live window entry of another row
    for M in (2, 5, 8):
        prm = P.make_vparams(
            pos=700,
            tokens=list(range(M)),
            sel_mode=1,
            top_k=20,
            top_p=0.95,
            temperature=1.0,
            us=[0.5] * M,
            poison_rows=0,
            cap=4096,
            theta=1e4,
            s_f=40,
            s_s=16,
            p_f=80,
            p_s=28,
        )
        for m in range(M):
            one = P.make_params(
                pos=700 + m,
                token=m,
                forced_next=None,
                sel_mode=1,
                top_k=20,
                top_p=0.95,
                temperature=1.0,
                u=0.5,
                poison=False,
                s_f=40,
                s_s=16,
                cap=4096,
                theta=1e4,
            )
            assert np.array_equal(
                prm[P.P_RCS + 128 * m : P.P_RCS + 128 * m + 64], one[P.P_COS : P.P_COS + 64]
            )
            assert np.array_equal(
                prm[P.P_RCS + 128 * m + 64 : P.P_RCS + 128 * (m + 1)], one[P.P_SIN : P.P_SIN + 64]
            )
        assert prm[P.P_ROWS] == M and prm[P.P_FORCED] == P.NO_TOKEN
    expect_raises(
        AssertionError,
        P.make_vparams,
        pos=0,
        tokens=[1],
        sel_mode=0,
        top_k=1,
        top_p=1.0,
        temperature=0.0,
        us=[0.0],
        poison_rows=0,
        cap=1,
        theta=1e4,
        s_f=32,
        s_s=16,
        p_f=40,
        p_s=28,
    )
    expect_raises(
        AssertionError,
        P.make_vparams,
        pos=0,
        tokens=[1, 2],
        sel_mode=0,
        top_k=1,
        top_p=1.0,
        temperature=0.0,
        us=[0.0, 0.0],
        poison_rows=0,
        cap=1,
        theta=1e4,
        s_f=568,
        s_s=16,
        p_f=0,
        p_s=28,
    )
    vp = P.make_vparams(
        pos=40000,
        tokens=[1, 2, 3],
        sel_mode=0,
        top_k=1,
        top_p=1.0,
        temperature=0.0,
        us=[0.0] * 3,
        poison_rows=0,
        cap=65536,
        theta=1e4,
        **P.attn_params(),
    )
    assert vp[P.P_6T] == P.T_INF and vp[P.P_PF] == 0 and vp[P.P_ROWS] == 3 and vp[P.P_STALL] == 0
    cfg = load_config(flint_headers)
    eo = P.expert_record_offsets(BankPlan(cfg, tensor_index(flint_headers, cfg)))
    h1, _b1 = MS.render(cfg, eo, False)
    assert "#define MR" not in h1 and "#define R_ARRIVE 128u" in h1 and "vph_qkv" not in h1
    for M in range(2, P.VMAX + 1):
        v = MS.verify_values(M, 40)
        assert v["RG2K"] * 2048 <= P.VXS and v["OG"] * v["KCO"] <= P.VXS and 4096 % v["KCO"] == 0
        h, b = MS.render(cfg, eo, M % 2 == 0, mr=M)
        assert (
            f"#define MR {M}u" in h
            and "vrun_phase" in b
            and f"#define R_ARRIVE {P.VR_ARRIVE}u" in h
        )
        assert (
            "vmoe_fused" in b
            and f"#define S_CTR {P.ctr_offset(P.VMAX)}u" in h
            and f"#define R_QFLAG {P.VR_QFLAG}u" in h
        )
