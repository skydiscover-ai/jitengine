
  alignas(16) threadgroup ushort xs_[XSV];
  threadgroup float red_[64];
  threadgroup float hb_[RG2K * 128];
  threadgroup uint tku_[16];
  threadgroup float tkw_[8];
  threadgroup uint flag_[4];
  threadgroup uint vtku_[MR * 8];
  threadgroup float vtkw_[MR * 8];
  threadgroup uint vuid_[48];
  threadgroup uint vurk_[48];
  threadgroup uint vnu_[4];
  threadgroup float vxsum_[XSV / 16];
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
  VX vx;
  vx.tku = vtku_; vx.tkw = vtkw_; vx.uid = vuid_; vx.urk = vurk_; vx.nu = vnu_; vx.hb = hb_; vx.xsum = vxsum_; vx.tk_layer = -1;
  (void)dep;
#if CHAIN
  (void)prev;
  vrun_phase(c, vx, ph[0], ph[1]);
#else
  (void)ph;
  if (pass_entry(c, (const device uint*)prev)) return;   // previous pass failed / residency probe / kernel_stall seam
  {  // the MoE team counters: zeroed before the first grid barrier
    device uint* ct = (device uint*)(c.st + S_CTR);
    for (uint i = c.w * 256u + c.tid; i < NL * 64u; i += c.G * 256u) ct[i * 16u] = 0u;
  }
  for (uint l = 0; l < NL; ++l) {
    vrun_phase(c, vx, PH_QKV, l);
    c.fcum += vflow_layer_tasks(c, l);
    vrun_phase(c, vx, PH_ATT, l);
    vrun_phase(c, vx, PH_MERGE, l); if (!grid_sync(c)) return;
    vrun_phase(c, vx, PH_O, l); if (!grid_sync(c)) return;
    if (DENSE[l]) {
      vrun_phase(c, vx, PH_DGU, l); if (!grid_sync(c)) return;
      vrun_phase(c, vx, PH_DDOWN, l); if (!grid_sync(c)) return;
    } else {
      if ((c.prm[P_MASK] & MOE_BITS) != 0u && !vmoe_fused(c, vx, l)) return;
      if (!grid_sync(c)) return;
    }
  }
  vrun_phase(c, vx, PH_HEAD, NL); if (!grid_sync(c)) return;
  vrun_phase(c, vx, PH_SELECT, NL);
#endif
