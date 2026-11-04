
  alignas(16) threadgroup ushort xs_[6144];
  threadgroup float red_[64];
  threadgroup float hb_[128];
  threadgroup uint tku_[16];
  threadgroup float tkw_[8];
  threadgroup uint flag_[4];
  threadgroup uint dt_[NL * NF + 8u];       // the descriptor table, offset >> 4 (the expert bank word raw)
  threadgroup ushort hm_[2048];             // bf(h + o) of the last MoE block (combine8)
  threadgroup uint pk_[12];                 // prm[P_MASK .. P_6T] (PK)
  Ctx c;
  c.b0 = (const device uchar*)b0; c.b1 = (const device uchar*)b1; c.b2 = (const device uchar*)b2;
  c.desc = (const device ulong*)desc;
  c.kf0 = (device ushort*)kf0; c.kf1 = (device ushort*)kf1; c.vf0 = (device ushort*)vf0; c.vf1 = (device ushort*)vf1;
  c.kr = (device ushort*)kr; c.vr = (device ushort*)vr;
  c.st = (device uchar*)scr;
  c.prm = (const device uint*)prm;
  c.res = (device uint*)res;
  c.tid = thread_index_in_threadgroup; c.sg = simdgroup_index_in_threadgroup; c.lane = thread_index_in_simdgroup;
  c.w = threadgroup_position_in_grid.x; c.G = threadgroups_per_grid.x;
  c.pos = c.prm[P_POS]; c.cap = c.prm[P_CAP]; c.gen = 0u; c.spin_max = c.prm[P_SPIN];
  c.tk_layer = -1; c.fcum = 0u;
  c.xs = xs_; c.red = red_; c.hb = hb_; c.tku = tku_; c.tkw = tkw_; c.flag = flag_;
  c.hm = hm_; c.hm_layer = -1;
  for (uint i = c.tid; i < NL * NF + 8u; i += 256u) {
    const ulong d = c.desc[i];
    dt_[i] = (i < NL * NF && i % NF == F_EBANK) ? uint(d) : uint(d >> 4);
  }
  c.dt = dt_;
  if (c.tid < 12u) pk_[c.tid] = c.prm[P_MASK + c.tid];
  c.pk = pk_;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const device uint* pv = (const device uint*)prev;
  (void)dep;
#if DIAG
  c.dnext = (const device uint*)dep;
#endif
#if CHAIN
  run_phase(c, ph[0], ph[1], pv);
#else
  (void)ph;
  if (pass_entry(c, pv)) return;            // previous pass failed / residency probe / kernel_stall seam
  {  // the MoE team counters: zeroed before the first grid barrier (every increment comes after it)
    device uint* ct = (device uint*)(c.st + S_CTR);
    for (uint i = c.w * 256u + c.tid; i < NL * 64u; i += c.G * 256u) ct[i * 16u] = 0u;
  }
  // per layer: QKV -> (flags) ATT -> (task counters) MERGE | O | MoE block (or DGU | DDOWN): 150 grid barriers
  for (uint l = 0; l < NL; ++l) {
    run_phase(c, PH_QKV, l, pv);
    c.fcum += flow_layer_tasks(c, l);
    run_phase(c, PH_ATT, l, pv);
    run_phase(c, PH_MERGE, l, pv); if (!grid_sync(c)) return;
    run_phase(c, PH_O, l, pv); if (!grid_sync(c)) return;
    if (DENSE[l]) {
      run_phase(c, PH_DGU, l, pv); if (!grid_sync(c)) return;
      run_phase(c, PH_DDOWN, l, pv); if (!grid_sync(c)) return;
    } else {
      if ((PK(c, P_MASK) & MOE_BITS) != 0u && !moe_fused(c, l)) return;
      if (!grid_sync(c)) return;
    }
  }
  run_phase(c, PH_HEAD, NL, pv); if (!grid_sync(c)) return;
  run_phase(c, PH_SELECT, NL, pv);
#endif
