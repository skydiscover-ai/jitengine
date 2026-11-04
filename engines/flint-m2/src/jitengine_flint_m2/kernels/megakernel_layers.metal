
// ------------------------------------------------------------------ norms and prologues
// v[8] = this thread's elements tid*8..tid*8+7 (fp32 holding bf16 values) -> xs = bf(w * bf(v * rsqrt(mean + eps)))
inline void norm2048(thread Ctx& c, thread const float* v, const device ushort* nw) {
  float ss = 0.0f;
  for (int j = 0; j < 8; ++j) ss += v[j] * v[j];
  ss = simd_sum(ss);
  if (c.lane == 0) c.red[c.sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (int s = 0; s < 8; ++s) tot += c.red[s];
  const float r = precise::rsqrt(tot / 2048.0f + EPS);
  const uint e0 = c.tid * 8;
  for (int j = 0; j < 8; ++j) c.xs[e0 + j] = f2bf(bf2f(nw[e0 + j]) * bfr(v[j] * r));
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
// norm2048 on LN weights loaded beforehand (ld_w8: one uint4 per thread, issued before the prologue's other loads).
// lo16 / hi16 of the packed words = bf2f of the ushorts: the same arithmetic.
inline uint4 ld_w8(thread const Ctx& c, const device ushort* nw) { return *(const device uint4*)(nw + c.tid * 8); }
inline void norm2048_q(thread Ctx& c, thread const float* v, uint4 nq) {
  float ss = 0.0f;
  for (int j = 0; j < 8; ++j) ss += v[j] * v[j];
  ss = simd_sum(ss);
  if (c.lane == 0) c.red[c.sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (int s = 0; s < 8; ++s) tot += c.red[s];
  const float r = precise::rsqrt(tot / 2048.0f + EPS);
  const uint e0 = c.tid * 8;
  const float nv[8] = {lo16(nq.x), hi16(nq.x), lo16(nq.y), hi16(nq.y), lo16(nq.z), hi16(nq.z), lo16(nq.w), hi16(nq.w)};
  for (int j = 0; j < 8; ++j) c.xs[e0 + j] = f2bf(nv[j] * bfr(v[j] * r));
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
inline void load_row8(thread const Ctx& c, const device ushort* row, thread float* v) {
  const uint4 a = *(const device uint4*)(row + c.tid * 8);
  v[0] = lo16(a.x); v[1] = hi16(a.x); v[2] = lo16(a.y); v[3] = hi16(a.y);
  v[4] = lo16(a.z); v[5] = hi16(a.z); v[6] = lo16(a.w); v[7] = hi16(a.w);
}
// hm = bf(h_l + o_l) for this thread's 8 elements
inline void load_mid8(thread const Ctx& c, uint l, thread float* v) {
  float h[8], o[8];
  load_row8(c, HS(c, l), h);
  load_row8(c, OB(c, l), o);
  for (int j = 0; j < 8; ++j) v[j] = bfr(h[j] + o[j]);
}
inline uint pass_token(thread const Ctx& c, const device uint* prev) {
  const uint t = c.prm[P_TOKMODE] != 0u ? prev[R_TOKEN] : c.prm[P_TOKEN];
  return min(t, VOCAB - 1u);
}
// 8-bit embedding row of the pass's token: e = bf(s * q + b)
inline void embed8(thread const Ctx& c, uint tok, thread float* v) {
  const device uchar* wq = GF(c, G_EW) + size_t(tok) * 2048 + c.tid * 8;
  const device ushort* sc = (const device ushort*)GF(c, G_EW + 1) + size_t(tok) * 32;
  const device ushort* bi = (const device ushort*)GF(c, G_EW + 2) + size_t(tok) * 32;
  const uint2 q = *(const device uint2*)wq;
  const float s = bf2f(sc[c.tid / 8]), b = bf2f(bi[c.tid / 8]);
  for (int j = 0; j < 4; ++j) {
    v[j] = bfr(s * float((q.x >> (8 * j)) & 0xffu) + b);
    v[4 + j] = bfr(s * float((q.y >> (8 * j)) & 0xffu) + b);
  }
}
// N bf16 of x into threadgroup memory: every uint4 load is issued before the first store (one memory round trip
// per thread instead of N/512 dependent load -> store trips). N = 4096 (O, DOWN) or 6144 (DDOWN): 2 or 3 uint4.
template <uint N>
inline void ld_vec(thread Ctx& c, const device ushort* src) {
  static_assert(N == 4096u || N == 6144u, "ld_vec stages 512 or 768 uint4 over 256 threads");
  const device uint4* s = (const device uint4*)src;
  threadgroup uint4* d = (threadgroup uint4*)c.xs;
  const uint4 v0 = s[c.tid], v1 = s[c.tid + 256u];
  if (N == 6144u) {
    const uint4 v2 = s[c.tid + 512u];
    d[c.tid] = v0; d[c.tid + 256u] = v1; d[c.tid + 512u] = v2;
  } else {
    d[c.tid] = v0; d[c.tid + 256u] = v1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ------------------------------------------------------------------ history addressing
inline device ushort* kv_head(thread const Ctx& c, uint l, bool isv, uint kvh) {
  const int fi = FIDX[l];
  if (fi >= 0) {
    device ushort* base = isv ? (fi < 7 ? c.vf0 : c.vf1) : (fi < 7 ? c.kf0 : c.kf1);
    return base + size_t(uint(fi % 7) * 4u + kvh) * c.cap * 128u;
  }
  device ushort* base = isv ? c.vr : c.kr;
  return base + size_t(uint(SIDX[l]) * 4u + kvh) * RING * 128u;
}

// ------------------------------------------------------------------ MoE block
// One phase and ONE grid barrier per sparse layer (persistent kernel). Every threadgroup normalises bf(h + o) into
// threadgroup memory; threadgroups w >= MQS compute the router rows and signal the layer's router counter. Pieces
// p = w, w + G, ...: p < MQS = member p of the shared expert's team (needs no routing, so it streams while the router
// completes), p >= MQS = member (p - MQS) % MQ of routed slot (p - MQS) / MQ (slot = rank at M = 1). A threadgroup runs
// its gate/up pieces, signalling the slot's completion counter after each, then its DOWN pieces, each of which waits
// only on its own slot's counter (the grid barrier's fence idiom), never on a grid barrier. DOWN stores every expert's
// output y_r (bf16) in Y[r] (r = rank, 6 = shared); the vendor combine (bf16 after every product and add, rank order,
// then the shared expert, then the residual) runs in the next phase's prologue (QKV of layer l + 1, or HEAD) in every
// threadgroup, and threadgroup 0 stores h_{l+1}. Per-row arithmetic = the earlier RSG / EGU / DOWN units: bit-identical.
// Chain mode (CHAIN 1) runs the same pieces as three plain dispatches: RSG = router + the shared gate/up pieces,
// EGU = the routed gate/up pieces, DOWN = every DOWN piece.
#define MQ 5u                   // routed team: threadgroups per routed expert (union slot)
#define MQS 10u                 // the shared expert's team
#define MDR 4u                  // DOWN output rows per unit (DOWN needs units of >= 2 rows)
#define CT_ROUTER 61u           // counters per layer: 0..47 routed (union) slots, 61 router, 63 shared
#define CT_SHARED 63u
#define MOE_BITS ((1u << PH_RSG) | (1u << PH_EGU) | (1u << PH_DOWN))

// top-6 of the 256 router probabilities (p desc, LOWER id on exact ties), w_r = bf(p_r / S), into threadgroup memory
inline void ensure_top6(thread Ctx& c, uint l) {
  if (c.tk_layer == int(l)) return;
  if (c.sg == 0) {
    const device float* sc = SCb(c, l);
    float v[8];
    for (int k = 0; k < 8; ++k) { const float p = sc[c.lane * 8 + k]; v[k] = (p >= 0.0f) ? p : -1.0f; }
    float sel = 0.0f;
    for (int r = 0; r < 6; ++r) {
      float bv = -4.0f; uint bi = 0xffffu;
      for (int k = 0; k < 8; ++k) if (v[k] > bv) { bv = v[k]; bi = c.lane * 8 + k; }
      const float M = simd_max(bv);
      const uint I = simd_min(bv == M ? bi : 0xffffu);
      if (I / 8u == c.lane) { for (int k = 0; k < 8; ++k) if (uint(k) == I % 8u) v[k] = -3.0f; }
      if (c.lane == uint(r)) sel = M;
      if (c.lane == 0) c.tku[r] = min(I, 255u);
    }
    float S = 0.0f;
    for (int r = 0; r < 6; ++r) S = S + simd_shuffle(sel, ushort(r));
    S = S + 1e-20f;
    if (c.lane < 6) c.tkw[c.lane] = bfr(sel / S);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  c.tk_layer = int(l);
}
inline const device uchar* expert_rec(thread const Ctx& c, uint l, uint e) {
  const ulong bank = desc_word(c, l * NF + F_EBANK);
  const device uchar* base = (bank == 1ul ? c.b1 : c.b2) + desc_off(c, l * NF + F_EBASE);
  return base + size_t(c.tku[e]) * E_STRIDE;
}
inline device ushort* YB(thread const Ctx& c, uint r) { return (device ushort*)(c.st + S_Y) + r * 2048u; }
inline device atomic_uint* CTR(thread const Ctx& c, uint l, uint s) {
  return (device atomic_uint*)(c.st + S_CTR) + (l * 64u + s) * 16u;      // one 64-byte line per counter
}
// h_{l+1} for this thread's 8 elements from layer l's expert outputs: acc = bf(y0 w0); acc = bf(acc + bf(y_r w_r));
// mlp = bf(acc + y_sh); h' = bf(bf(h + o) + mlp) (the earlier DOWN combine; y_r already bf16). Threadgroup 0 stores it.
// bf(h + o) is the value this thread formed in layer l's MoE block (moe_fused keeps it in c.hm)
// when the tag says so; otherwise (chain mode, masked timing arms) it is recomputed from the h and o rows.
inline void combine8(thread Ctx& c, uint l, thread float* v) {
  ensure_top6(c, l);
  float hm[8], y[7][8];
  if (c.hm_layer == int(l)) {
    for (int j = 0; j < 8; ++j) hm[j] = bf2f(c.hm[c.tid * 8 + j]);
  } else {
    load_mid8(c, l, hm);
  }
  for (int r = 0; r < 7; ++r) load_row8(c, YB(c, uint(r)), y[r]);
  for (int j = 0; j < 8; ++j) {
    float acc = bfr(bfr(y[0][j]) * c.tkw[0]);
    for (int r = 1; r < 6; ++r) acc = bfr(acc + bfr(bfr(y[r][j]) * c.tkw[r]));
    const float mlp = bfr(acc + bfr(y[6][j]));
    v[j] = bfr(hm[j] + mlp);
  }
  if (c.w == 0) {
    device uint* hn = (device uint*)HS(c, l + 1u) + c.tid * 4;
    for (int j = 0; j < 4; ++j) hn[j] = pack2(v[2 * j], v[2 * j + 1]);
  }
}
// team-local handoff on a per-(layer, slot) completion counter, with the grid barrier's fence idiom
inline void team_signal(thread Ctx& c, device atomic_uint* ctr) {
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device);
  if (c.tid == 0) atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
}
// Bounded poll of a monotonic counter until it reaches `need`: ONE relaxed load per spin, the
// error word only every 64th spin, and the last value seen decides, with no reload after the loop. `seen` is a value
// already read (grid_sync passes its own RMW's result, so the last arriver never polls). Returns the last value seen:
// below `need` means the wait limit was reached or another wait failed (the caller records its error word and exits).
inline uint spin_until(device atomic_uint* ctr, uint seen, uint need, device atomic_uint* errw, uint spin_max) {
  for (uint spins = 0u; seen < need;) {
    if (++spins >= spin_max) break;
    if ((spins & 63u) == 0u && atomic_load_explicit(errw, memory_order_relaxed) != 0u) break;
    seen = atomic_load_explicit(ctr, memory_order_relaxed);
  }
  return seen;
}
inline bool team_wait(thread Ctx& c, device atomic_uint* ctr, uint need) {
  if (c.tid == 0) {
    device atomic_uint* errw = (device atomic_uint*)(c.res + R_ERR);
    const uint bad = spin_until(ctr, atomic_load_explicit(ctr, memory_order_relaxed), need, errw, c.spin_max) < need ? 1u : 0u;
    if (bad != 0u) set_err(c, E_TEAM | c.gen);
    c.flag[1] = bad;
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  return c.flag[1] == 0u;
}
// gate/up of routed slot s (rank s), team member q: 2-row units q + MQ * sg, ... (per row = the earlier EGU unit)
inline void gu_piece_routed(thread Ctx& c, uint l, uint s, uint q) {
  const device uchar* rec = expert_rec(c, l, s);
  const QM mg = qm_expert(rec, EG_W, EG_S, EG_B), mu = qm_expert(rec, EU_W, EU_S, EU_B);
  device uint* act = (device uint*)(ACT(c, l) + s * 512u);
  for (uint i = q + MQ * c.sg; i < 256u; i += MQ * 8u) {
    float g[2], up[2];
    q4_gu<2048>(g, up, mg, mu, i * 2u, c.xs, c.lane);
    if (c.lane == 0) act[i] = pack2(swiglu(bfr(g[0]), bfr(up[0])), swiglu(bfr(g[1]), bfr(up[1])));
  }
}
// gate/up of the shared expert, team member q (per row = the earlier RSG shared unit)
inline void gu_piece_shared(thread Ctx& c, uint l, uint q) {
  const QM mg = qm_layer(c, l, F_SGW), mu = qm_layer(c, l, F_SUW);
  device uint* act = (device uint*)(ACT(c, l) + 3072u);
  for (uint i = q + MQS * c.sg; i < 512u; i += MQS * 8u) {
    float g[2], up[2];
    q4_gu<2048>(g, up, mg, mu, i * 2u, c.xs, c.lane);
    if (c.lane == 0) act[i] = pack2(swiglu(bfr(g[0]), bfr(up[0])), swiglu(bfr(g[1]), bfr(up[1])));
  }
}
// DOWN of routed slot s, member q: MDR-row units (per row = the earlier routed segment: one block, then simd_sum);
// the slot's act (512) is staged with every uint4 load in flight before the store (ld_vec)
inline void dn_piece_routed(thread Ctx& c, uint l, uint s, uint q) {
  threadgroup ushort* xa = c.xs + 2048u;
  if (c.tid < 64u) ((threadgroup uint4*)xa)[c.tid] = ((const device uint4*)(ACT(c, l) + s * 512u))[c.tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const device uchar* rec = expert_rec(c, l, s);
  const device uchar* W = rec + ED_W;
  const device ushort* S = (const device ushort*)(rec + ED_S);
  const device ushort* B = (const device ushort*)(rec + ED_B);
  const uint xo = c.lane * 16u;
  float xv[16];
  const float xsum = load_x16(xa, xo, xv);
  device ushort* yo = YB(c, s);
  for (uint i = q + MQ * c.sg; i < 2048u / MDR; i += MQ * 8u) {
    float pe[MDR];
    for (uint rr = 0; rr < MDR; ++rr) pe[rr] = q4_blk<512>(W, S, B, i * MDR + rr, xo, xv, xsum);
    for (uint rr = 0; rr < MDR; ++rr) {
      const float y = simd_sum(pe[rr]);
      if (c.lane == 0) yo[i * MDR + rr] = f2bf(y);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
// DOWN of the shared expert, member q (per row = the earlier shared segment: two blocks, then simd_sum)
inline void dn_piece_shared(thread Ctx& c, uint l, uint q) {
  threadgroup ushort* xa = c.xs + 2048u;
  if (c.tid < 128u) ((threadgroup uint4*)xa)[c.tid] = ((const device uint4*)(ACT(c, l) + 3072u))[c.tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const QM ms = qm_layer(c, l, F_SDW);
  device ushort* yo = YB(c, 6u);
  for (uint i = q + MQS * c.sg; i < 2048u / MDR; i += MQS * 8u) {
    for (uint rr = 0; rr < MDR; ++rr) {
      const uint row = i * MDR + rr;
      float pe = 0.0f;
      for (int blk = 0; blk < 2; ++blk) {
        const uint xo = blk * 512 + c.lane * 16u;
        float xv[16];
        const float xsum = load_x16(xa, xo, xv);
        pe += q4_blk<1024>(ms.w, ms.s, ms.b, row, xo, xv, xsum);
      }
      const float y = simd_sum(pe);
      if (c.lane == 0) yo[row] = f2bf(y);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
#if !CHAIN
// the persistent MoE block of sparse layer l. Phase-mask bits (timing arms) select its parts: RSG = router rows,
// EGU = gate/up pieces, DOWN = DOWN pieces; a wait is skipped when its producer is masked off. Returns false on a
// counter timeout (error word set; the threadgroup exits like a grid-barrier timeout).
inline bool moe_fused(thread Ctx& c, uint l) {
  const uint msk = PK(c, P_MASK);
  const bool dr = ((msk >> PH_RSG) & 1u) != 0u, dg = ((msk >> PH_EGU) & 1u) != 0u, dd = ((msk >> PH_DOWN) & 1u) != 0u;
  float v[8];
  const uint4 nq = ld_w8(c, LH(c, l, F_LN2));
  load_mid8(c, l, v);
  norm2048_q(c, v, nq);
  for (int j = 0; j < 8; ++j) c.hm[c.tid * 8 + j] = f2bf(v[j]);    // v = bf(h + o): exact in bf16
  c.hm_layer = int(l);
  const uint NRT = c.G - MQS;
  if (dr && c.w >= MQS) {
    const device ushort* RW = LH(c, l, F_RW);
    device float* sc = SCb(c, l);
    for (uint u = c.sg * NRT + (c.w - MQS); u < 128u; u += NRT * 8u) {
      float acc[2];
      bf_unit(acc, RW, u * 2u, c.xs, c.lane);
      if (c.lane == 0) { sc[u * 2u] = sigmoid_m(bfr(acc[0])); sc[u * 2u + 1u] = sigmoid_m(bfr(acc[1])); }
    }
    team_signal(c, CTR(c, l, CT_ROUTER));
  }
  const uint NP = MQS + 6u * MQ;
  bool ready = false;
  for (uint p = c.w; p < NP; p += c.G) {
    if (p >= MQS && !ready) {
      if (dr && !team_wait(c, CTR(c, l, CT_ROUTER), NRT)) return false;
      ensure_top6(c, l);
      ready = true;
    }
    if (!dg) continue;
    if (p < MQS) { gu_piece_shared(c, l, p); team_signal(c, CTR(c, l, CT_SHARED)); }
    else { const uint r = p - MQS; gu_piece_routed(c, l, r / MQ, r % MQ); team_signal(c, CTR(c, l, r / MQ)); }
  }
  if (dd) {
    for (uint p = c.w; p < NP; p += c.G) {
      const bool sh = p < MQS;
      const uint r = sh ? 0u : p - MQS;
      if (dg && !team_wait(c, CTR(c, l, sh ? CT_SHARED : r / MQ), sh ? MQS : MQ)) return false;
      if (sh) dn_piece_shared(c, l, p);
      else dn_piece_routed(c, l, r / MQ, r % MQ);
    }
  }
  if (!ready) {                                   // shared-team-only threadgroups: the top-6 the combine needs
    if (dr && !team_wait(c, CTR(c, l, CT_ROUTER), NRT)) return false;
    ensure_top6(c, l);
  }
  return true;
}
#endif
// QKV publishes a finished head (persistent kernel, K_FLOW 1): release fence, then res[R_QFLAG + hd] = l + 1
inline void flow_publish(thread Ctx& c, uint l, uint hd) {
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device);
  if (c.tid == 0u) atomic_store_explicit((device atomic_uint*)(c.res + R_QFLAG + hd), l + 1u, memory_order_relaxed);
}

// ------------------------------------------------------------------ QKV: one head per threadgroup
inline void ph_qkv(thread Ctx& c, uint l, const device uint* prev) {
  float v[8];
  const uint4 nq = ld_w8(c, LH(c, l, F_LN1));
  if (l == 0) {
    embed8(c, pass_token(c, prev), v);
    if (c.w == 0) {
      device uint* h0 = (device uint*)HS(c, 0) + c.tid * 4;
      for (int j = 0; j < 4; ++j) h0[j] = pack2(v[2 * j], v[2 * j + 1]);
    }
  } else if (DENSE[l - 1u] != 0u) {
    load_row8(c, HS(c, l), v);
  } else {
    combine8(c, l - 1u, v);        // h_l = layer l - 1's MoE output (threadgroup 0 stores it)
  }
  norm2048_q(c, v, nq);
  const bool full = FIDX[l] >= 0;
  const uint slot = full ? c.pos : c.pos % RING;
  for (uint hd = c.w; hd < 40u; hd += c.G) {
    const uint kind = hd < 32u ? 0u : (hd < 36u ? 1u : 2u);
    const uint local = kind == 0u ? hd : (kind == 1u ? hd - 32u : hd - 36u);
    const uint f = kind == 0u ? F_QW : (kind == 1u ? F_KW : F_VW);
    const device uchar* W = LF(c, l, f) + size_t(local) * 128 * 1024;
    const device ushort* S = LH(c, l, f + 1) + local * 128 * 32;
    const device ushort* B = LH(c, l, f + 2) + local * 128 * 32;
    float tl_nw = 0.0f, tl_cs = 0.0f, tl_sn = 0.0f;      // the tail's loads, issued before the GEMV
    if (c.tid < 128u && kind < 2u) {
      tl_nw = bf2f(LH(c, l, kind == 0u ? F_QN : F_KN)[c.tid]);
      if (!full) { const uint i = c.tid & 63u; tl_cs = as_type<float>(c.prm[P_COS + i]); tl_sn = as_type<float>(c.prm[P_SIN + i]); }
    }
    for (uint u = 0; u < 8u; ++u) {
      const uint r0 = c.sg * 16u + u * 2u;
      float acc[2];
      q4_unit<2048>(acc, W, S, B, r0, c.xs, c.lane);
      if (c.lane == 0) { c.hb[r0] = bfr(acc[0]); c.hb[r0 + 1] = bfr(acc[1]); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float y = 0.0f;
    if (c.tid < 128u) {
      y = c.hb[c.tid];
      if (kind < 2u) {
        const float ss = simd_sum(y * y);
        if (c.lane == 0) c.red[16 + c.sg] = ss;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (c.tid < 128u && kind < 2u) {
      const float tot = ((c.red[16] + c.red[17]) + c.red[18]) + c.red[19];
      const float r = precise::rsqrt(tot / 128.0f + EPS);
      y = bfr(tl_nw * bfr(y * r));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (c.tid < 128u) c.hb[c.tid] = y;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (c.tid < 128u) {
      float out = y;
      if (!full && kind < 2u) {
        const uint i = c.tid & 63u;
        const float x1 = c.hb[i], x2 = c.hb[i + 64u];
        out = c.tid < 64u ? bfr(x1 * tl_cs - x2 * tl_sn) : bfr(x2 * tl_cs + x1 * tl_sn);
      }
      if (kind == 0u) QB(c, l)[local * 128u + c.tid] = f2bf(out);
      else kv_head(c, l, kind == 2u, local)[size_t(slot) * 128u + c.tid] = f2bf(out);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#if !CHAIN
    flow_publish(c, l, hd);
#endif
  }
}

// ------------------------------------------------------------------ attention (layout 2 GQA tile per simdgroup)
struct AttnState { simdgroup_float8x8 O[16]; float m; float l; };
inline short frag_fm(uint lane) { return short(((lane >> 2) & 4) + ((lane >> 1) & 3)); }
inline short frag_fn(uint lane) { return short(((lane >> 2) & 2) * 2 + (lane & 1) * 2); }
inline ushort frag_lane(short r, short col) {
  return ushort((((r >> 2) & 1) << 4) | (((col >> 2) & 1) << 3) | (((r >> 1) & 1) << 2) | ((r & 1) << 1) | ((col >> 1) & 1));
}
// diag(a) as an 8x8 fragment (lane (fm, fn) holds D[fm][fn], D[fm][fn + 1]). The online-softmax rescale of the MMA
// accumulators is ONE fp32 simdgroup MMA per tile by diag(a) instead of element-wise thread_elements() multiplies
// (the element-wise access is slower per rescaled 8-key block per simdgroup). Every off-diagonal product
// is an exact zero, so each element is still round(a * O): bit-identical for finite, non -0.0 accumulators (they
// start at +0 and only accumulate).
inline simdgroup_float8x8 diag_rows(float a_row_fm, short fm, short fn) {      // D[i][i] = a of row i (lane's row fm)
  simdgroup_float8x8 D;
  D.thread_elements()[0] = (fn == fm) ? a_row_fm : 0.0f;
  D.thread_elements()[1] = (fn + 1 == fm) ? a_row_fm : 0.0f;
  return D;
}
inline simdgroup_float8x8 diag_cols(float a_col_fn, float a_col_fn1, short fm, short fn) {   // D[j][j] = a of col j
  simdgroup_float8x8 D;
  D.thread_elements()[0] = (fm == fn) ? a_col_fn : 0.0f;
  D.thread_elements()[1] = (fm == fn + 1) ? a_col_fn1 : 0.0f;
  return D;
}
struct QT2 { simdgroup_bfloat8x8 B[16]; };
inline void load_qt2(thread QT2& q2, const device ushort* q, uint lane) {
  const short fm = frag_fm(lane), fn = frag_fn(lane);
  const device uint4* r0 = (const device uint4*)(q + fn * 128 + (fm >> 1) * 32);
  const device uint4* r1 = (const device uint4*)(q + (fn + 1) * 128 + (fm >> 1) * 32);
  uint a[16], b[16];
  for (int k = 0; k < 4; ++k) {
    const uint4 x = r0[k], y = r1[k];
    a[4 * k] = x.x; a[4 * k + 1] = x.y; a[4 * k + 2] = x.z; a[4 * k + 3] = x.w;
    b[4 * k] = y.x; b[4 * k + 1] = y.y; b[4 * k + 2] = y.z; b[4 * k + 3] = y.w;
  }
  const uint sh = (fm & 1) * 16;
  for (int t = 0; t < 16; ++t) {
    q2.B[t].thread_elements()[0] = as_type<bfloat>(ushort((a[t] >> sh) & 0xffffu));
    q2.B[t].thread_elements()[1] = as_type<bfloat>(ushort((b[t] >> sh) & 0xffffu));
  }
}
// keys [kb, ke) of one kv head; ring = sliding layer (slot valid iff (pos - slot) mod RING < WINDOW)
inline void attend(thread AttnState& st, thread const QT2& q2, const device ushort* Kh, const device ushort* Vh,
                   uint kb, uint ke, uint lane, bool ring, uint pm) {
  const short fm = frag_fm(lane), fn = frag_fn(lane);
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
  for (int t = 0; t < 16; ++t) st.O[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  const ushort la = frag_lane(fn, fm), lb = frag_lane(fn + 1, fm), lal = frag_lane(0, fm);
  const bool odd = (fm & 1) != 0;
  for (uint j0 = kb; j0 < ke; j0 += 8) {
    const uint jr = min(j0 + uint(fm), ke - 1);
    uint kw[16], vw[16];
    const device uint4* pk = (const device uint4*)(Kh + size_t(jr) * 128 + (fn >> 1) * 32);
    const device uint4* pv = (const device uint4*)(Vh + size_t(jr) * 128 + (fn >> 1) * 32);
    for (int k = 0; k < 4; ++k) {
      const uint4 x = pk[k], y = pv[k];
      kw[4 * k] = x.x; kw[4 * k + 1] = x.y; kw[4 * k + 2] = x.z; kw[4 * k + 3] = x.w;
      vw[4 * k] = y.x; vw[4 * k + 1] = y.y; vw[4 * k + 2] = y.z; vw[4 * k + 3] = y.w;
    }
    simdgroup_float8x8 Sa = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Sb = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_bfloat8x8 A;
    for (int t = 0; t < 16; t += 2) {
      A.thread_elements()[0] = as_type<bfloat>(ushort(kw[t] & 0xffffu));
      A.thread_elements()[1] = as_type<bfloat>(ushort(kw[t] >> 16));
      simdgroup_multiply_accumulate(Sa, A, q2.B[t], Sa);
      A.thread_elements()[0] = as_type<bfloat>(ushort(kw[t + 1] & 0xffffu));
      A.thread_elements()[1] = as_type<bfloat>(ushort(kw[t + 1] >> 16));
      simdgroup_multiply_accumulate(Sb, A, q2.B[t + 1], Sb);
    }
    const uint j = j0 + uint(fm);
    bool valid = j < ke;
    if (ring) { const uint d = pm >= j ? pm - j : pm + RING - j; valid = valid && d < WINDOW; }
    const float s0 = valid ? (Sa.thread_elements()[0] + Sb.thread_elements()[0]) * QSCALE : -INFINITY;
    const float s1 = valid ? (Sa.thread_elements()[1] + Sb.thread_elements()[1]) * QSCALE : -INFINITY;
    float b0 = s0, b1 = s1;
    b0 = max(b0, simd_shuffle_xor(b0, ushort(2)));  b1 = max(b1, simd_shuffle_xor(b1, ushort(2)));
    b0 = max(b0, simd_shuffle_xor(b0, ushort(4)));  b1 = max(b1, simd_shuffle_xor(b1, ushort(4)));
    b0 = max(b0, simd_shuffle_xor(b0, ushort(16))); b1 = max(b1, simd_shuffle_xor(b1, ushort(16)));
    const float n0 = max(m0, b0), n1 = max(m1, b1);
    const bool moved = simd_any(n0 > m0 || n1 > m1);
    const float p0 = (s0 == -INFINITY) ? 0.0f : exp2(s0 - n0);
    const float p1 = (s1 == -INFINITY) ? 0.0f : exp2(s1 - n1);
    if (moved) {
      const float a0 = (m0 == -INFINITY) ? 0.0f : exp2(m0 - n0);
      const float a1 = (m1 == -INFINITY) ? 0.0f : exp2(m1 - n1);
      l0 *= a0; l1 *= a1;
      const float2 ap = simd_shuffle(float2(a0, a1), lal);
      const float al = odd ? ap.y : ap.x;
      const simdgroup_float8x8 Dg = diag_rows(al, fm, fn);
      for (int t = 0; t < 16; ++t) {
        simdgroup_float8x8 Tm;
        simdgroup_multiply(Tm, Dg, st.O[t]);
        st.O[t] = Tm;
      }
      m0 = n0; m1 = n1;
    }
    l0 += p0; l1 += p1;
    const float2 pa = simd_shuffle(float2(p0, p1), la), pb = simd_shuffle(float2(p0, p1), lb);
    simdgroup_float8x8 Pm;
    Pm.thread_elements()[0] = odd ? pa.y : pa.x;
    Pm.thread_elements()[1] = odd ? pb.y : pb.x;
    simdgroup_bfloat8x8 Vb;
    for (int t = 0; t < 16; ++t) {
      Vb.thread_elements()[0] = as_type<bfloat>(ushort(vw[t] & 0xffffu));
      Vb.thread_elements()[1] = as_type<bfloat>(ushort(vw[t] >> 16));
      simdgroup_multiply_accumulate(st.O[t], Pm, Vb, st.O[t]);
    }
  }
  l0 += simd_shuffle_xor(l0, ushort(2));  l1 += simd_shuffle_xor(l1, ushort(2));
  l0 += simd_shuffle_xor(l0, ushort(4));  l1 += simd_shuffle_xor(l1, ushort(4));
  l0 += simd_shuffle_xor(l0, ushort(16)); l1 += simd_shuffle_xor(l1, ushort(16));
  const float2 mm = simd_shuffle(float2(m0, m1), lal), ll = simd_shuffle(float2(l0, l1), lal);
  st.m = odd ? mm.y : mm.x;
  st.l = odd ? ll.y : ll.x;
}
// Full-layer stream (long requests): the same arithmetic as attend() with ring = false, over the 8-key blocks
// kb, kb + bstep, kb + 2*bstep, ... < ke (bstep = 8 * NSUB: one sub-stream of a chunk). Every visited block holds
// at least one valid key, so the running max is finite after the first block and the -inf guards of attend() are
// not needed (exp2(-inf) = 0 gives the same values). Zero visited blocks leave m = -inf, l = 0, O = 0 (an empty
// partial, skipped exactly by the merge).
inline void attend_stream(thread AttnState& st, thread const QT2& q2, const device ushort* Kh, const device ushort* Vh,
                          uint kb, uint ke, uint bstep, uint lane) {
  const short fm = frag_fm(lane), fn = frag_fn(lane);
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
  for (int t = 0; t < 16; ++t) st.O[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  const ushort la = frag_lane(fn, fm), lb = frag_lane(fn + 1, fm), lal = frag_lane(0, fm);
  const bool odd = (fm & 1) != 0;
  const device ushort* Kl = Kh + (fn >> 1) * 32;
  const device ushort* Vl = Vh + (fn >> 1) * 32;
  for (uint j0 = kb; j0 < ke; j0 += bstep) {
    const uint j = j0 + uint(fm);
    const uint jr = SABL == 1u ? (min(j, ke - 1) & 15u) : min(j, ke - 1);
    uint kw[16], vw[16];
    const device uint4* pk = (const device uint4*)(Kl + size_t(jr) * 128);
    const device uint4* pv = (const device uint4*)(Vl + size_t(jr) * 128);
    for (int k = 0; k < 4; ++k) {
      const uint4 x = pk[k], y = pv[k];
      kw[4 * k] = x.x; kw[4 * k + 1] = x.y; kw[4 * k + 2] = x.z; kw[4 * k + 3] = x.w;
      vw[4 * k] = y.x; vw[4 * k + 1] = y.y; vw[4 * k + 2] = y.z; vw[4 * k + 3] = y.w;
    }
    if (SABL == 2u) {
      uint x = 0u;
      for (int k = 0; k < 16; ++k) x ^= kw[k] ^ vw[k];
      l0 += float(x & 1u);
      continue;
    }
    simdgroup_float8x8 Sa = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Sb = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_bfloat8x8 A;
    for (int t = 0; t < 16; t += 2) {
      A.thread_elements()[0] = as_type<bfloat>(ushort(kw[t] & 0xffffu));
      A.thread_elements()[1] = as_type<bfloat>(ushort(kw[t] >> 16));
      simdgroup_multiply_accumulate(Sa, A, q2.B[t], Sa);
      A.thread_elements()[0] = as_type<bfloat>(ushort(kw[t + 1] & 0xffffu));
      A.thread_elements()[1] = as_type<bfloat>(ushort(kw[t + 1] >> 16));
      simdgroup_multiply_accumulate(Sb, A, q2.B[t + 1], Sb);
    }
    const bool valid = j < ke;
    const float s0 = valid ? (Sa.thread_elements()[0] + Sb.thread_elements()[0]) * QSCALE : -INFINITY;
    const float s1 = valid ? (Sa.thread_elements()[1] + Sb.thread_elements()[1]) * QSCALE : -INFINITY;
    float b0 = s0, b1 = s1;
    b0 = max(b0, simd_shuffle_xor(b0, ushort(2)));  b1 = max(b1, simd_shuffle_xor(b1, ushort(2)));
    b0 = max(b0, simd_shuffle_xor(b0, ushort(4)));  b1 = max(b1, simd_shuffle_xor(b1, ushort(4)));
    b0 = max(b0, simd_shuffle_xor(b0, ushort(16))); b1 = max(b1, simd_shuffle_xor(b1, ushort(16)));
    const float n0 = max(m0, b0), n1 = max(m1, b1);
    const bool moved = simd_any(n0 > m0 || n1 > m1);
    const float p0 = exp2(s0 - n0), p1 = exp2(s1 - n1);
    if (moved) {
      const float a0 = exp2(m0 - n0), a1 = exp2(m1 - n1);
      l0 *= a0; l1 *= a1;
      const float2 ap = simd_shuffle(float2(a0, a1), lal);
      const float al = odd ? ap.y : ap.x;
      const simdgroup_float8x8 Dg = diag_rows(al, fm, fn);
      for (int t = 0; t < 16; ++t) {
        simdgroup_float8x8 Tm;
        simdgroup_multiply(Tm, Dg, st.O[t]);
        st.O[t] = Tm;
      }
      m0 = n0; m1 = n1;
    }
    l0 += p0; l1 += p1;
    const float2 pa = simd_shuffle(float2(p0, p1), la), pb = simd_shuffle(float2(p0, p1), lb);
    simdgroup_float8x8 Pm;
    Pm.thread_elements()[0] = odd ? pa.y : pa.x;
    Pm.thread_elements()[1] = odd ? pb.y : pb.x;
    simdgroup_bfloat8x8 Vb;
    for (int t = 0; t < 16; ++t) {
      Vb.thread_elements()[0] = as_type<bfloat>(ushort(vw[t] & 0xffffu));
      Vb.thread_elements()[1] = as_type<bfloat>(ushort(vw[t] >> 16));
      simdgroup_multiply_accumulate(st.O[t], Pm, Vb, st.O[t]);
    }
  }
  l0 += simd_shuffle_xor(l0, ushort(2));  l1 += simd_shuffle_xor(l1, ushort(2));
  l0 += simd_shuffle_xor(l0, ushort(4));  l1 += simd_shuffle_xor(l1, ushort(4));
  l0 += simd_shuffle_xor(l0, ushort(16)); l1 += simd_shuffle_xor(l1, ushort(16));
  const float2 mm = simd_shuffle(float2(m0, m1), lal), ll = simd_shuffle(float2(l0, l1), lal);
  st.m = odd ? mm.y : mm.x;
  st.l = odd ? ll.y : ll.x;
}
// partial rows of PROW floats (m, l, -, -, O[128] in natural head-dim order) for the 8 q heads of the tile.
// Lane (fm, fn) holds row fm, head dims (fn/2)*32 + 2t + e of tiles t = 0..15, i.e. 32 CONTIGUOUS dims: they are
// written as 8 float4 stores (16-byte aligned: PROW*4 = 528 = 16*33), not 32 scattered scalars.
inline void store_partial(thread const AttnState& st, device float* rows, uint lane) {
  const short fm = frag_fm(lane), fn = frag_fn(lane);
  device float* pr = rows + fm * PROW;
  if ((lane & 9) == 0) { pr[0] = st.m; pr[1] = st.l; }
  device float4* po = (device float4*)(pr + 4 + (fn >> 1) * 32);
  for (int k = 0; k < 8; ++k)
    po[k] = float4(st.O[2 * k].thread_elements()[0], st.O[2 * k].thread_elements()[1],
                   st.O[2 * k + 1].thread_elements()[0], st.O[2 * k + 1].thread_elements()[1]);
}
// Layout 6d's per-position grid (plan.pos_grid is the same integer formula): keys per chunk of a
// row whose last key is q (keys 0..q), S = 8 ceil((q + 1) / span), span 512 while q + 1 <= T, else 1,024. A pure
// function of the row's own position: no request quantity (N_max, max_tokens, context limit) reaches the partition.
inline uint pos_grid(uint q, uint T) {
  const uint n = q + 1u;
  const uint span = n <= T ? 512u : 1024u;
  return 8u * ((n + span - 1u) / span);
}
// the partition of layer l at the pass's position: strided (parts > 0: C = parts), layout 6d (full layers, the row's
// own pos_grid) or a legacy layout-2 chunk grid (C chunks of S, each split into nsub sub-streams on full layers:
// C * nsub partial rows per kv head, chunk-major)
inline void att_grid(thread const Ctx& c, uint l, thread uint& S, thread uint& nk, thread uint& C, thread uint& parts,
                     thread uint& nsub) {
  const bool full = FIDX[l] >= 0;
  parts = full ? PK(c, P_PF) : PK(c, P_PS);
  S = full ? PK(c, P_SF) : PK(c, P_SS);
  nk = full ? c.pos + 1u : min(c.pos + 1u, RING);
  const uint T6 = full ? PK(c, P_6T) : 0u;
  if (T6 != 0u) { parts = 0u; S = pos_grid(c.pos, T6); }   // layout 6d: chunks of S keys on the absolute grid
  C = parts != 0u ? parts : (nk + S - 1u) / S;
  nsub = (full && parts == 0u && T6 == 0u) ? max(PK(c, P_FSUB), 1u) : 1u;
}

// ------------------------------------------------------------------ flag dataflow around attention (K_FLOW 1)
// Persistent kernel only (the chain reference keeps its plain dispatches). QKV publishes each head (flow_publish); an
// attention task of kv head g first waits for its 8 q heads and the k / v heads of g (acquire), runs unchanged, and
// counts itself done on res[R_FCNT + g]; a merge task of kv head g waits until that counter reaches the cumulative
// task count of g (c.fcum: every threadgroup derives it from the request-fixed partition). This replaces the QKV -> ATT
// and ATT -> MERGE grid barriers; MERGE -> O keeps its barrier. Same tasks, same arithmetic, same merge order:
// bit-identical. Waits are bounded (error word, then every wait aborts; never a hang); producers never wait and the
// persistent grid is co-resident, so the dataflow cannot deadlock.
inline void flow_fail(thread const Ctx& c, uint code) { set_err(c, code); }
// attention tasks per kv head of layer l (0 when the phase mask or a timing ablation skips that layer's tasks)
inline uint flow_layer_tasks(thread const Ctx& c, uint l) {
  if (((PK(c, P_MASK) >> PH_ATT) & 1u) == 0u) return 0u;
  uint abl = PK(c, P_MASK) >> 16;
  if ((abl & 16u) && FIDX[l] < 0) abl = 0u;
  if ((abl & 8u) && FIDX[l] >= 0) return 0u;
  uint S, nk, C, parts, nsub;
  att_grid(c, l, S, nk, C, parts, nsub);
  return parts != 0u ? parts * (8u / HG) : C * nsub;
}
inline void flow_wait_heads(thread const Ctx& c, uint l, uint kvh) {
  if (((PK(c, P_MASK) >> PH_QKV) & 1u) == 0u) return;
  const uint hd = c.lane < 8u ? kvh * 8u + c.lane : (c.lane == 8u ? 32u + kvh : 36u + kvh);
#if PASS_M1
  // ONE load instruction per spin: lanes 0-9 poll the 10 head flags and the other lanes the
  // error word, through a per-lane address (no divergent second load)
  device atomic_uint* fw = (device atomic_uint*)(c.res + (c.lane < 10u ? R_QFLAG + hd : R_ERR));
  bool done = false, abort = false;
  for (uint spins = 0u; spins < c.spin_max; ++spins) {
    const uint x = atomic_load_explicit(fw, memory_order_relaxed);
    const uint st = c.lane < 10u ? (x > l ? 1u : 0u) : (x != 0u ? 2u : 1u);   // 1 ready, 0 not yet, 2 error elsewhere
    if (simd_any(st == 2u)) { abort = true; break; }
    if (simd_all(st == 1u)) { done = true; break; }
  }
#else
  bool done = false, abort = false;
  for (uint spins = 0u; spins < c.spin_max; ++spins) {
    uint st = 1u;                                  // 1 ready, 0 not yet, 2 error elsewhere
    if (c.lane < 10u) st = atomic_load_explicit((device atomic_uint*)(c.res + R_QFLAG + hd), memory_order_relaxed) > l ? 1u : 0u;
    else if (c.lane == 10u) st = atomic_load_explicit((device atomic_uint*)(c.res + R_ERR), memory_order_relaxed) != 0u ? 2u : 1u;
    if (simd_any(st == 2u)) { abort = true; break; }
    if (simd_all(st == 1u)) { done = true; break; }
  }
#endif
  if (!done && !abort && c.lane == 0u) flow_fail(c, E_FLOW_HEADS | (l + 1u));
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
}
inline void flow_done(thread const Ctx& c, uint kvh) {
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  simdgroup_barrier(mem_flags::mem_device);
  if (c.lane == 0u) atomic_fetch_add_explicit((device atomic_uint*)(c.res + R_FCNT + kvh), 1u, memory_order_relaxed);
}
inline void flow_wait_tasks(thread const Ctx& c, uint l, uint kvh) {
  bool done = false, abort = false;
#if PASS_M1
  device atomic_uint* fw = (device atomic_uint*)(c.res + (c.lane == 0u ? R_FCNT + kvh : R_ERR));   // one load per spin
  for (uint spins = 0u; spins < c.spin_max; ++spins) {
    const uint x = atomic_load_explicit(fw, memory_order_relaxed);
    const uint v = simd_broadcast(x, ushort(0));
    const uint e = simd_broadcast(x, ushort(1));
    if (v >= c.fcum) { done = true; break; }
    if (e != 0u) { abort = true; break; }
  }
#else
  for (uint spins = 0u; spins < c.spin_max; ++spins) {
    uint v = 0u, e = 0u;
    if (c.lane == 0u) v = atomic_load_explicit((device atomic_uint*)(c.res + R_FCNT + kvh), memory_order_relaxed);
    else if (c.lane == 1u) e = atomic_load_explicit((device atomic_uint*)(c.res + R_ERR), memory_order_relaxed);
    v = simd_broadcast(v, ushort(0));
    e = simd_broadcast(e, ushort(1));
    if (v >= c.fcum) { done = true; break; }
    if (e != 0u) { abort = true; break; }
  }
#endif
  if (!done && !abort && c.lane == 0u) flow_fail(c, E_FLOW_TASKS | (l + 1u));
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
}
#if CHAIN
#define FLOW_WAIT(c, l, kvh)
#define FLOW_DONE(c, kvh)
#define FLOW_WAIT_TASKS(c, l, kvh)
#else
#define FLOW_WAIT(c, l, kvh) flow_wait_heads(c, l, kvh)
#define FLOW_DONE(c, kvh) flow_done(c, kvh)
#define FLOW_WAIT_TASKS(c, l, kvh) flow_wait_tasks(c, l, kvh)
#endif
// strided scalar attention of one task: q heads g0..g0+HG-1 of kv head kvh over keys part, part + parts, ... < nk.
// Lane owns head dims 4*lane..4*lane+3; score = simd_sum (fixed tree), online softmax in the log2 domain kept
// replicated in every lane; the next key/value row is loaded one step ahead. Writes the task's 8 partial rows.
inline void attend_strided(thread const Ctx& c, uint l, uint kvh, uint g0, uint part, uint parts, uint nk, bool ring,
                           uint pm, device float* rows, uint abl) {
  const device ushort* qp = QB(c, l) + (kvh * 8u + g0) * 128u + c.lane * 4u;
  float qv[HG][4], acc[HG][4], m[HG], z[HG];
  for (int h = 0; h < int(HG); ++h) {
    uint2 w = uint2(0u);
    if ((abl & 1u) == 0u) w = *(const device uint2*)(qp + h * 128);
    qv[h][0] = lo16(w.x) * QSCALE; qv[h][1] = hi16(w.x) * QSCALE;
    qv[h][2] = lo16(w.y) * QSCALE; qv[h][3] = hi16(w.y) * QSCALE;
    for (int j = 0; j < 4; ++j) acc[h][j] = 0.0f;
    m[h] = -INFINITY; z[h] = 0.0f;
  }
  const device ushort* Kh = kv_head(c, l, false, kvh) + c.lane * 4u;
  const device ushort* Vh = kv_head(c, l, true, kvh) + c.lane * 4u;
  const uint end = (abl & 2u) ? 0u : nk;
  // keys in batches of KB (part + j*parts); the next batch's K/V rows are loaded while this batch computes, so a
  // task pays one memory latency per KB keys instead of one per key
  const uint step = KB * parts;
  uint2 kn[KB], vn[KB];
  for (uint j = 0; j < KB; ++j) {
    const uint tj = part + j * parts;
    kn[j] = tj < end ? *(const device uint2*)(Kh + size_t(tj) * 128) : uint2(0u);
    vn[j] = tj < end ? *(const device uint2*)(Vh + size_t(tj) * 128) : uint2(0u);
  }
  for (uint t0 = part; t0 < end; t0 += step) {
    uint2 kw[KB], vw[KB];
    for (uint j = 0; j < KB; ++j) { kw[j] = kn[j]; vw[j] = vn[j]; }
    for (uint j = 0; j < KB; ++j) {
      const uint tj = t0 + step + j * parts;
      if (tj < end) { kn[j] = *(const device uint2*)(Kh + size_t(tj) * 128); vn[j] = *(const device uint2*)(Vh + size_t(tj) * 128); }
    }
    for (uint j = 0; j < KB; ++j) {
      const uint t = t0 + j * parts;
      if (t >= end) break;
      if (ring) { const uint d = pm >= t ? pm - t : pm + RING - t; if (d >= WINDOW) continue; }
      const float k0 = lo16(kw[j].x), k1 = hi16(kw[j].x), k2 = lo16(kw[j].y), k3 = hi16(kw[j].y);
      const float v[4] = {lo16(vw[j].x), hi16(vw[j].x), lo16(vw[j].y), hi16(vw[j].y)};
      float sc[HG];
      for (int h = 0; h < int(HG); ++h) sc[h] = simd_sum(qv[h][0] * k0 + qv[h][1] * k1 + qv[h][2] * k2 + qv[h][3] * k3);
      for (int h = 0; h < int(HG); ++h) {
        if (sc[h] > m[h]) {
          const float a = exp2(m[h] - sc[h]);
          z[h] = z[h] * a + 1.0f;
          for (int q = 0; q < 4; ++q) acc[h][q] = acc[h][q] * a + v[q];
          m[h] = sc[h];
        } else {
          const float pr = exp2(sc[h] - m[h]);
          z[h] += pr;
          for (int q = 0; q < 4; ++q) acc[h][q] += pr * v[q];
        }
      }
    }
  }
  if (abl & 4u) return;
  rows += g0 * PROW;
  for (int h = 0; h < int(HG); ++h) {
    *(device float4*)(rows + h * PROW + 4u + c.lane * 4u) = float4(acc[h][0], acc[h][1], acc[h][2], acc[h][3]);
    if (c.lane == 0) { rows[h * PROW] = m[h]; rows[h * PROW + 1u] = z[h]; }
  }
}
// Layout 6d (layout 6 with the diagonal-MMA rescale): the full-layer path at every
// N_max. One simdgroup computes the partial of one query row's 8 q heads over the keys [kb, ke) of one chunk of the
// per-position absolute grid (S = pos_grid(q, T) keys per chunk: span 512, <= 64 chunks per kv head; kv8 above
// q + 1 = 32,768: span 1024, <= 128; plan.pos_grid), in ascending 8-key
// blocks, transposed-score form:
//   S^T (8 keys x 8 heads) = sum_t K_t (simdgroup_load) * Q^T_t (transposed simdgroup_load, loaded once)
//   O^T_t (8 dims x 8 heads) += V^T_t (transposed simdgroup_load) * P^T (the S^T fragment itself)
// Lane (fm, fn) owns heads fn, fn + 1: the running max and alpha are lane-local and the rescale is O^T x diag(alpha).
// Whole 8-key blocks are loaded and the keys >= ke masked, so history rows up to the next multiple of 8 must be
// finite (zero-initialised history, capacity a multiple of 8: kv_cache.py). The partial depends only on (row position,
// chunk), never on M or on which simdgroup computes it, so a verify row equals the M = 1 pass at its position.
inline void att6(const device ushort* q8, const device ushort* Kh16, const device ushort* Vh16, uint kb, uint ke,
                 device float* rows, uint lane) {
  const short fm = frag_fm(lane), fn = frag_fn(lane);
  const device bfloat* Kh = (const device bfloat*)Kh16;
  const device bfloat* Vh = (const device bfloat*)Vh16;
  simdgroup_bfloat8x8 QT[16];
  {
    const device bfloat* qt = (const device bfloat*)q8;
    for (int t = 0; t < 16; ++t) simdgroup_load(QT[t], qt + 8 * t, 128, ulong2(0, 0), true);
  }
  simdgroup_float8x8 OT[16];
  for (int t = 0; t < 16; ++t) OT[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
  for (uint k0 = kb; k0 < ke; k0 += 8) {
    const device bfloat* Kb = Kh + size_t(k0) * 128;
    const device bfloat* Vb = Vh + size_t(k0) * 128;
    simdgroup_float8x8 Sa = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Sb = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_bfloat8x8 Ka, Kc;
    for (int t = 0; t < 16; t += 2) {
      simdgroup_load(Ka, Kb + 8 * t, 128);
      simdgroup_load(Kc, Kb + 8 * (t + 1), 128);
      simdgroup_multiply_accumulate(Sa, Ka, QT[t], Sa);
      simdgroup_multiply_accumulate(Sb, Kc, QT[t + 1], Sb);
    }
    const bool valid = (k0 + uint(fm)) < ke;
    const float s0 = valid ? (Sa.thread_elements()[0] + Sb.thread_elements()[0]) * QSCALE : -INFINITY;
    const float s1 = valid ? (Sa.thread_elements()[1] + Sb.thread_elements()[1]) * QSCALE : -INFINITY;
    float2 bm = float2(s0, s1);
    bm = max(bm, simd_shuffle_xor(bm, ushort(2)));
    bm = max(bm, simd_shuffle_xor(bm, ushort(4)));
    bm = max(bm, simd_shuffle_xor(bm, ushort(16)));
    const float n0 = max(m0, bm.x), n1 = max(m1, bm.y);
    if (simd_any(n0 > m0 || n1 > m1)) {
      const float a0 = (m0 == -INFINITY) ? 0.0f : exp2(m0 - n0);
      const float a1 = (m1 == -INFINITY) ? 0.0f : exp2(m1 - n1);
      l0 *= a0; l1 *= a1;
      const simdgroup_float8x8 Dg = diag_cols(a0, a1, fm, fn);
      for (int t = 0; t < 16; ++t) {
        simdgroup_float8x8 Tm;
        simdgroup_multiply(Tm, OT[t], Dg);
        OT[t] = Tm;
      }
      m0 = n0; m1 = n1;
    }
    const float p0 = (s0 == -INFINITY) ? 0.0f : exp2(s0 - m0);
    const float p1 = (s1 == -INFINITY) ? 0.0f : exp2(s1 - m1);
    l0 += p0; l1 += p1;
    simdgroup_float8x8 PT;
    PT.thread_elements()[0] = p0;
    PT.thread_elements()[1] = p1;
    simdgroup_bfloat8x8 Va, Vc;
    for (int t = 0; t < 16; t += 2) {
      simdgroup_load(Va, Vb + 8 * t, 128, ulong2(0, 0), true);
      simdgroup_load(Vc, Vb + 8 * (t + 1), 128, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(OT[t], Va, PT, OT[t]);
      simdgroup_multiply_accumulate(OT[t + 1], Vc, PT, OT[t + 1]);
    }
  }
  float2 lv = float2(l0, l1);
  lv += simd_shuffle_xor(lv, ushort(2));
  lv += simd_shuffle_xor(lv, ushort(4));
  lv += simd_shuffle_xor(lv, ushort(16));
  device float* r0 = rows + fn * PROW;
  device float* r1 = rows + (fn + 1) * PROW;
  if (fm == 0) { r0[0] = m0; r0[1] = lv.x; r1[0] = m1; r1[1] = lv.y; }
  for (int t = 0; t < 16; ++t) {
    r0[4 + 8 * t + fm] = OT[t].thread_elements()[0];
    r1[4 + 8 * t + fm] = OT[t].thread_elements()[1];
  }
}
// one full-layer item (kv head, chunk, sub-stream) of the long stream
inline void att_item(thread Ctx& c, uint l, uint it, uint C, uint nsub, uint S, uint nk, uint abl) {
  const uint per = C * nsub;
  const uint kvh = it / per, r = it % per, ch = r / nsub, sub = r % nsub;
  const uint kb = ch * S, ke = min(kb + S, nk);
  QT2 q2;
  if (abl & 1u) { for (int t = 0; t < 16; ++t) q2.B[t] = make_filled_simdgroup_matrix<bfloat, 8, 8>(bfloat(0.0f)); }
  else load_qt2(q2, QB(c, l) + kvh * 8u * 128u, c.lane);
  AttnState st;
  const device ushort* Kh = kv_head(c, l, false, kvh);
  const device ushort* Vh = kv_head(c, l, true, kvh);
  attend_stream(st, q2, Kh, Vh, kb + 8u * sub, (abl & 2u) ? kb : ke, 8u * nsub, c.lane);
  if ((abl & 4u) == 0u) store_partial(st, PART(c) + size_t(kvh * per + r) * 8u * PROW, c.lane);
}
inline void ph_att(thread Ctx& c, uint l) {
  uint S, nk, C, parts, nsub;
  att_grid(c, l, S, nk, C, parts, nsub);
  const bool ring = FIDX[l] < 0;
  const uint pm = c.pos % RING;
  uint abl = PK(c, P_MASK) >> 16;           // timing ablations (bench_mk only): 1 no q load, 2 no keys, 4 no store,
                                            // 8 no full-layer attention at all, 16 = ablations on full layers only
  if ((abl & 16u) && FIDX[l] < 0) abl = 0u;
  if ((abl & 8u) && FIDX[l] >= 0) return;
  if (parts != 0u) {
    const uint NG = 8u / HG;               // tasks per (kv head, part)
    for (uint it = gsg(c); it < 4u * parts * NG; it += nsg(c)) {
      const uint item = it / NG, g0 = (it % NG) * HG;       // item = kvh * parts + part
      FLOW_WAIT(c, l, item / parts);
      attend_strided(c, l, item / parts, g0, item % parts, parts, nk, ring, pm, PART(c) + size_t(item) * 8u * PROW, abl);
      FLOW_DONE(c, item / parts);
    }
    return;
  }
  if (FIDX[l] >= 0 && PK(c, P_6T) != 0u) {
    // layout 6d: items (kv head, chunk), C <= 64 (kv8: <= 128) per kv head, over all 8 simdgroups of every threadgroup
    // (<= 256 items on 320 simdgroups beat 6 streaming simdgroups and half-size grids), dealt threadgroup-first
    // (items beyond one wave spread over every threadgroup)
    for (uint it = gsg(c); it < 4u * C; it += nsg(c)) {
      const uint kvh = it / C, ch = it % C;
      const uint kb = ch * S, ke = (abl & 2u) ? kb : min(kb + S, nk);
      // partial rows (kvh * C + ch) = it; of the timing ablations only 2 (no key blocks) and 8 (above) apply here
      FLOW_WAIT(c, l, kvh);
      att6(QB(c, l) + kvh * 8u * 128u, kv_head(c, l, false, kvh), kv_head(c, l, true, kvh), kb, ke,
           PART(c) + size_t(it) * 8u * PROW, c.lane);
      FLOW_DONE(c, kvh);
    }
    return;
  }
  if (FIDX[l] >= 0) {
    // long full-layer stream: items (kv head, chunk, sub-stream) over the first ASG simdgroups of every threadgroup,
    // dealt statically (threadgroup-first, or simdgroup-first) or taken from this layer's atomic queue. A partial
    // depends only on its item, never on which simdgroup computes it.
    const uint flags = c.prm[P_FFLAGS];
    const uint asg = clamp(c.prm[P_FASG], 1u, 8u);
    if (c.sg >= asg) return;
    const uint NI = 4u * C * nsub;
    if (flags & FL_DYNAMIC) {
      device atomic_uint* q = (device atomic_uint*)(c.res + R_ATTQ + uint(FIDX[l]));
      for (;;) {
        uint it = 0u;
        if (c.lane == 0u) it = atomic_fetch_add_explicit(q, 1u, memory_order_relaxed);
        it = simd_broadcast(it, ushort(0));
        if (it >= NI) break;
        FLOW_WAIT(c, l, it / (C * nsub));
        att_item(c, l, it, C, nsub, S, nk, abl);
        FLOW_DONE(c, it / (C * nsub));
      }
      return;
    }
    const uint ns = c.G * asg;
    const uint me = (flags & FL_SGFIRST) ? c.w * asg + c.sg : c.sg * c.G + c.w;
    for (uint it = me; it < NI; it += ns) {
      FLOW_WAIT(c, l, it / (C * nsub));
      att_item(c, l, it, C, nsub, S, nk, abl);
      FLOW_DONE(c, it / (C * nsub));
    }
    return;
  }
  for (uint it = gsg(c); it < 4u * C; it += nsg(c)) {
    const uint kvh = it / C, ch = it % C;
    const uint kb = ch * S, ke = min(kb + S, nk);
    FLOW_WAIT(c, l, kvh);
    QT2 q2;
    if (abl & 1u) { for (int t = 0; t < 16; ++t) q2.B[t] = make_filled_simdgroup_matrix<bfloat, 8, 8>(bfloat(0.0f)); }
    else load_qt2(q2, QB(c, l) + kvh * 8u * 128u, c.lane);
    AttnState st;
    attend(st, q2, kv_head(c, l, false, kvh), kv_head(c, l, true, kvh), kb, (abl & 2u) ? kb : ke, c.lane, ring, pm);
    if ((abl & 4u) == 0u) store_partial(st, PART(c) + size_t(kvh * C + ch) * 8u * PROW, c.lane);
    FLOW_DONE(c, kvh);
  }
}
// task = (q head, 16-dim slice): lanes own partials lane, lane + 32, ... (ascending, empty ones skipped exactly),
// then simd_sum (fixed tree) and the sink term; out = bf(o / den)
inline void ph_merge(thread Ctx& c, uint l) {
  uint S, nk, C, parts, nsub;
  att_grid(c, l, S, nk, C, parts, nsub);
  C *= nsub;                                  // partial rows per kv head (chunk-major, then sub-stream)
  const device ushort* sinks = LH(c, l, F_SINK);
  for (uint t = gsg(c); t < 256u; t += nsg(c)) {
    FLOW_WAIT_TASKS(c, l, t >> 6);
    const uint h = t >> 3, d0 = (t & 7u) * 16u, kvh = h >> 3, g = h & 7u;
    const device float* base = PART(c) + (size_t(kvh * C) * 8u + g) * PROW;
    const uint stride = 8u * PROW;
    const float sink2 = bf2f(sinks[h]) * kLog2e;
    float mx = sink2;
    float den = 0.0f;
    float4 o0 = float4(0.0f), o1 = float4(0.0f), o2 = float4(0.0f), o3 = float4(0.0f);
    if (C <= 64u) {
      // <= 2 rows per lane (lane, lane + 32), both loaded up front: one memory round trip, then the same order
      const bool h0 = c.lane < C, h1 = c.lane + 32u < C;
      const device float* r0 = base + size_t(c.lane) * stride;
      const device float* r1 = base + size_t(c.lane + 32u) * stride;
      float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
      float4 a0 = float4(0.0f), a1 = float4(0.0f), a2 = float4(0.0f), a3 = float4(0.0f);
      float4 b0 = float4(0.0f), b1 = float4(0.0f), b2 = float4(0.0f), b3 = float4(0.0f);
      if (h0) {
        m0 = r0[0]; l0 = r0[1];
        const device float4* ro = (const device float4*)(r0 + 4u + d0);
        a0 = ro[0]; a1 = ro[1]; a2 = ro[2]; a3 = ro[3];
      }
      if (h1) {
        m1 = r1[0]; l1 = r1[1];
        const device float4* ro = (const device float4*)(r1 + 4u + d0);
        b0 = ro[0]; b1 = ro[1]; b2 = ro[2]; b3 = ro[3];
      }
      if (h0) mx = max(mx, m0);
      if (h1) mx = max(mx, m1);
      mx = simd_max(mx);
      if (h0 && m0 != -INFINITY) {
        const float wgt = exp2(m0 - mx);
        den += l0 * wgt;
        o0 += wgt * a0; o1 += wgt * a1; o2 += wgt * a2; o3 += wgt * a3;
      }
      if (h1 && m1 != -INFINITY) {
        const float wgt = exp2(m1 - mx);
        den += l1 * wgt;
        o0 += wgt * b0; o1 += wgt * b1; o2 += wgt * b2; o3 += wgt * b3;
      }
    } else {
      for (uint ch = c.lane; ch < C; ch += 32u) mx = max(mx, base[size_t(ch) * stride]);
      mx = simd_max(mx);
      for (uint ch = c.lane; ch < C; ch += 32u) {
        const device float* r = base + size_t(ch) * stride;
        const float mc = r[0];
        if (mc == -INFINITY) continue;
        const float wgt = exp2(mc - mx);
        den += r[1] * wgt;
        const device float4* ro = (const device float4*)(r + 4u + d0);
        o0 += wgt * ro[0]; o1 += wgt * ro[1]; o2 += wgt * ro[2]; o3 += wgt * ro[3];
      }
    }
    den = simd_sum(den) + exp2(sink2 - mx);
    o0 = simd_sum(o0); o1 = simd_sum(o1); o2 = simd_sum(o2); o3 = simd_sum(o3);
    if (c.lane < 4u) {
      const float4 v = c.lane == 0u ? o0 : (c.lane == 1u ? o1 : (c.lane == 2u ? o2 : o3));
      device uint* dst = (device uint*)(AB(c, l) + h * 128u + d0 + c.lane * 4u);
      dst[0] = pack2(v.x / den, v.y / den);
      dst[1] = pack2(v.z / den, v.w / den);
    }
  }
}

// ------------------------------------------------------------------ O projection
inline void ph_o(thread Ctx& c, uint l) {
  ld_vec<4096>(c, AB(c, l));
  const QM m = qm_layer(c, l, F_OW);
  device ushort* out = OB(c, l);
  for (uint r = gsg(c); r < 2048u; r += nsg(c)) {
    const float y = q4_row<4096>(m.w, m.s, m.b, r, c.xs, c.lane);
    if (c.lane == 0) out[r] = f2bf(y);
  }
}

// ------------------------------------------------------------------ MoE, chain mode: the block's pieces as plain dispatches
#if CHAIN
inline void ph_rsg(thread Ctx& c, uint l) {        // router + the shared expert's gate/up pieces
  float v[8];
  load_mid8(c, l, v);
  norm2048(c, v, LH(c, l, F_LN2));
  const device ushort* RW = LH(c, l, F_RW);
  device float* sc = SCb(c, l);
  for (uint u = gsg(c); u < 128u; u += nsg(c)) {
    float acc[2];
    bf_unit(acc, RW, u * 2u, c.xs, c.lane);
    if (c.lane == 0) { sc[u * 2u] = sigmoid_m(bfr(acc[0])); sc[u * 2u + 1u] = sigmoid_m(bfr(acc[1])); }
  }
  for (uint p = c.w; p < MQS; p += c.G) gu_piece_shared(c, l, p);
}
inline void ph_egu(thread Ctx& c, uint l) {        // the routed gate/up pieces
  float v[8];
  load_mid8(c, l, v);
  norm2048(c, v, LH(c, l, F_LN2));
  ensure_top6(c, l);
  for (uint p = c.w; p < 6u * MQ; p += c.G) gu_piece_routed(c, l, p / MQ, p % MQ);
}
inline void ph_down(thread Ctx& c, uint l) {       // every DOWN piece
  ensure_top6(c, l);
  for (uint p = c.w; p < MQS + 6u * MQ; p += c.G) {
    if (p < MQS) dn_piece_shared(c, l, p);
    else dn_piece_routed(c, l, (p - MQS) / MQ, (p - MQS) % MQ);
  }
}
#endif

// ------------------------------------------------------------------ dense MLP (layers 0-1)
inline void ph_dgu(thread Ctx& c, uint l) {
  float v[8];
  load_mid8(c, l, v);
  norm2048(c, v, LH(c, l, F_LN2));
  const QM mg = qm_layer(c, l, F_DGW), mu = qm_layer(c, l, F_DUW);
  device uint* act = (device uint*)ACT(c, l);
  for (uint u = gsg(c); u < 3072u; u += nsg(c)) {
    float g[2], up[2];
    q4_gu<2048>(g, up, mg, mu, u * 2u, c.xs, c.lane);
    if (c.lane == 0) act[u] = pack2(swiglu(bfr(g[0]), bfr(up[0])), swiglu(bfr(g[1]), bfr(up[1])));
  }
}
inline void ph_ddown(thread Ctx& c, uint l) {
  ld_vec<6144>(c, ACT(c, l));
  const QM m = qm_layer(c, l, F_DDW);
  device uint* hn = (device uint*)HS(c, l + 1);
  const device uint* hw = (const device uint*)HS(c, l);
  const device uint* ow = (const device uint*)OB(c, l);
  for (uint u = gsg(c); u < 1024u; u += nsg(c)) {
    float acc[2];
    q4_unit<6144>(acc, m.w, m.s, m.b, u * 2u, c.xs, c.lane);
    if (c.lane == 0) {
      const uint hwv = hw[u], owv = ow[u];
      const float m0 = bfr(lo16(hwv) + lo16(owv)), m1 = bfr(hi16(hwv) + hi16(owv));
      hn[u] = pack2(m0 + bfr(acc[0]), m1 + bfr(acc[1]));
    }
  }
}

// ------------------------------------------------------------------ head: logits + per-simdgroup statistics
inline bool tbetter(float v, uint i, float w, uint j) { return v > w || (v == w && i < j); }
inline void ph_head(thread Ctx& c) {
  float v8[8];
  const uint4 nq = ld_w8(c, (const device ushort*)GF(c, G_NORM));
  if (DENSE[NL - 1u] != 0u) load_row8(c, HS(c, NL), v8);
  else combine8(c, NL - 1u, v8);   // h_49 = layer 48's MoE output (threadgroup 0 stores it)
  norm2048_q(c, v8, nq);
  const QM m = QM{GF(c, G_HW), (const device ushort*)GF(c, G_HW + 1), (const device ushort*)GF(c, G_HW + 2)};
  const bool poison = c.prm[P_POISON] != 0u;
  device uint* lg = (device uint*)LG(c);
  float mx = -INFINITY, se = 0.0f;
  uint fin = 1u;
  float tv = -INFINITY; uint ti = 0xffffffffu;      // lane k < 20: k-th best (value desc, id asc)
  for (uint u = gsg(c); u < VOCAB / 2u; u += nsg(c)) {
    float acc[2];
    q4_unit<2048>(acc, m.w, m.s, m.b, u * 2u, c.xs, c.lane);
    float vv[2] = {bfr(acc[0]), bfr(acc[1])};
    if (poison) { vv[0] = vv[0] * NAN; vv[1] = vv[1] * NAN; }
    if (c.lane == 0) lg[u] = pack2(vv[0], vv[1]);
    for (int r = 0; r < 2; ++r) {
      const float x = vv[r];
      const uint id = u * 2u + uint(r);
      fin &= isfinite(x) ? 1u : 0u;
      if (x > mx) { se = se * exp(mx - x) + 1.0f; mx = x; }
      else if (x != -INFINITY) se += exp(x - mx);
      const float thr = simd_shuffle(tv, ushort(19));
      const uint thi = simd_shuffle(ti, ushort(19));
      if (tbetter(x, id, thr, thi)) {
        const float pv = simd_shuffle_up(tv, ushort(1));
        const uint pi = simd_shuffle_up(ti, ushort(1));
        const bool mine = tbetter(x, id, tv, ti);
        const bool prevb = c.lane > 0 && tbetter(x, id, pv, pi);
        if (mine) { if (prevb) { tv = pv; ti = pi; } else { tv = x; ti = id; } }
      }
    }
  }
  device uint* sp = SELP(c) + (c.w * 8u + c.sg) * 48u;
  if (c.lane == 0) { sp[0] = as_type<uint>(mx); sp[1] = as_type<uint>(se); sp[2] = fin; }
  if (c.lane < 20u) { sp[4 + c.lane] = as_type<uint>(tv); sp[24 + c.lane] = ti; }
}

// ------------------------------------------------------------------ select (threadgroup 0, simdgroup 0)
inline void ph_select(thread Ctx& c) {
  if (c.w != 0u || c.sg != 0u) return;
  const uint NS = c.G * 8u;
  const device uint* sp = SELP(c);
  // log-sum-exp: global max, then each lane's lists in a fixed order, then simd_sum
  float ml = -INFINITY;
  for (uint k = c.lane; k < NS; k += 32u) ml = max(ml, as_type<float>(sp[k * 48u]));
  const float M = simd_max(ml);
  float sl = 0.0f;
  uint fin = 1u;
  for (uint k = c.lane; k < NS; k += 32u) {
    const float mk = as_type<float>(sp[k * 48u]);
    if (mk != -INFINITY) sl += as_type<float>(sp[k * 48u + 1u]) * exp(mk - M);
    fin &= sp[k * 48u + 2u];
  }
  const float Ssum = simd_sum(sl);
  fin = simd_min(fin);
  const float lse = M + log(Ssum);
  // top-20 tournament over the NS sorted per-simdgroup lists; lane owns lists lane + 32k
  const uint NK = (NS + 31u) / 32u;
  float hv[16]; uint hid[16]; uint ptr[16];
  for (uint k = 0; k < 16u; ++k) {
    ptr[k] = 0u;
    const uint li = c.lane + 32u * k;
    if (k < NK && li < NS) { hv[k] = as_type<float>(sp[li * 48u + 4u]); hid[k] = sp[li * 48u + 24u]; }
    else { hv[k] = -INFINITY; hid[k] = 0xffffffffu; }
  }
  float myv = -INFINITY; uint myi = 0xffffffffu;
  for (uint r = 0; r < 20u; ++r) {
    float bv = -INFINITY; uint bi = 0xffffffffu, bk = 0u;
    for (uint k = 0; k < 16u; ++k) if (tbetter(hv[k], hid[k], bv, bi)) { bv = hv[k]; bi = hid[k]; bk = k; }
    const float V = simd_max(bv);
    const uint I = simd_min(bv == V ? bi : 0xffffffffu);
    if (bv == V && bi == I && I != 0xffffffffu) {
      for (uint k = 0; k < 16u; ++k) {
        if (k == bk) {
          ptr[k] += 1u;
          const uint li = c.lane + 32u * k;
          if (ptr[k] < 20u) { hv[k] = as_type<float>(sp[li * 48u + 4u + ptr[k]]); hid[k] = sp[li * 48u + 24u + ptr[k]]; }
          else { hv[k] = -INFINITY; hid[k] = 0xffffffffu; }
        }
      }
    }
    if (c.lane == r) { myv = V; myi = I; }
  }
  const device ushort* lg = LG(c);
  if (c.lane < 20u) { c.res[R_TOPI + c.lane] = myi; c.res[R_TOPV + c.lane] = as_type<uint>(myv - lse); }
  const uint amax = simd_shuffle(myi, ushort(0));
  uint chosen = amax;
  if (c.prm[P_SELMODE] == 1u) {
    // T, then top-k (<= 20), then the top-p nucleus, then the inverse CDF at u (sampler.sample's rule)
    const uint K = min(max(c.prm[P_TOPK], 1u), 20u);
    const float invT = as_type<float>(c.prm[P_INVT]), topp = as_type<float>(c.prm[P_TOPP]);
    const float u = as_type<float>(c.prm[P_U]);
    const bool in = c.lane < K;
    const float lf = myv * invT;
    const float lmax = simd_shuffle(lf, ushort(0));
    const float e = in ? exp(lf - lmax) : 0.0f;
    const float Z = simd_sum(e);
    const float p = e / Z;
    const float cs = simd_prefix_inclusive_sum(p);
    const float before = cs - p;
    const float p2 = (in && (topp >= 1.0f || before < topp)) ? p : 0.0f;
    const float cdf = simd_prefix_inclusive_sum(p2);
    const float total = simd_shuffle(cdf, ushort(K - 1u));
    const uint cnt = simd_sum((in && cdf <= u * total) ? 1u : 0u);
    const uint idx = min(cnt, K - 1u);
    chosen = simd_shuffle(myi, ushort(idx));
  }
  chosen = min(chosen, VOCAB - 1u);
  if (c.lane == 0) {
    c.res[R_TOKEN] = chosen;
    c.res[R_FINITE] = fin;
    c.res[R_ARGMAX] = amax;
    c.res[R_LSE] = as_type<uint>(lse);
    c.res[R_CHOSEN_LP] = as_type<uint>(bf2f(lg[chosen]) - lse);
    const uint F = c.prm[P_FORCED];
    c.res[R_FORCED_LP] = as_type<uint>(F < VOCAB ? bf2f(lg[F]) - lse : NAN);
    c.res[R_DONE] = 1u;
  }
}

// ------------------------------------------------------------------ dispatch of one phase, and the grid barrier
inline void run_phase1(thread Ctx& c, uint ph, uint l, const device uint* prev) {
  switch (ph) {
    case PH_QKV: ph_qkv(c, l, prev); break;
    case PH_ATT: ph_att(c, l); break;
    case PH_MERGE: ph_merge(c, l); break;
    case PH_O: ph_o(c, l); break;
#if CHAIN
    case PH_RSG: ph_rsg(c, l); break;
    case PH_EGU: ph_egu(c, l); break;
    case PH_DOWN: ph_down(c, l); break;
#endif
    case PH_DGU: ph_dgu(c, l); break;
    case PH_DDOWN: ph_ddown(c, l); break;
    case PH_HEAD: ph_head(c); break;
    case PH_SELECT: ph_select(c); break;
    default: break;
  }
}
inline void run_phase(thread Ctx& c, uint ph, uint l, const device uint* prev) {
  if (((PK(c, P_MASK) >> ph) & 1u) == 0u) return;
#if DIAG
  // a rolled loop (never unrolled), so each phase's code appears once, as in the production kernel; the next
  // layer of the same kind comes from the host's table in `dep` (a device-side search inflated the code 3.5x)
  const uint dg = c.prm[P_DIAG];
  const uint n = 1u + ((dg >> ph) & 1u);
  uint lr = l;
#pragma clang loop unroll(disable)
  for (uint rep = 0u; rep < n; ++rep) {
    run_phase1(c, ph, lr, prev);
    if (dg & (1u << 16)) lr = min(c.dnext[l], NL);
  }
#else
  run_phase1(c, ph, l, prev);
#endif
}
// Barrier generation g completes when the monotonic arrival counter reaches g * G. Each threadgroup: device fence
// (release), one relaxed RMW on the counter, poll the counter itself until it reaches the target, device fence
// (acquire). The RMWs form one release sequence, so no last-arriver generation word (and its second fence and
// store) is needed. Bounded spin: a timeout sets the error word and every threadgroup exits; never a hang.
inline bool grid_sync(thread Ctx& c) {
  c.gen += 1u;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device);
  if (c.tid == 0) {
    device atomic_uint* arrive = (device atomic_uint*)(c.res + R_ARRIVE);
    device atomic_uint* errw = (device atomic_uint*)(c.res + R_ERR);
    const uint target = c.gen * c.G;
    // the RMW's own result is the first poll: the last arriver never loads the counter
    const uint seen = atomic_fetch_add_explicit(arrive, 1u, memory_order_relaxed) + 1u;
    uint bad = 0u;
    if (spin_until(arrive, seen, target, errw, c.spin_max) < target) {
      bad = 1u;
      set_err(c, c.gen);
    }
    c.flag[0] = bad;
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  return c.flag[0] == 0u;
}
// Residency probe (runner.probe; never part of a decode pass): a flag ring of `laps` laps through every threadgroup, on
// the persistent kernel itself (the pass's resources). Threadgroup w waits until w - 1's flag reaches the lap (w = 0:
// G - 1's previous lap), then publishes its own, so each lap is a chain of G dependent hand-offs that only runs at full
// speed when all G threadgroups run at once (a barrier also completes when late threadgroups are scheduled one by one).
// Flags at R_QFLAG + w (G <= 64 words; a probe dispatch runs no QKV). Bounded spin -> E_RING, never a hang.
inline void probe_ring(thread Ctx& c, uint laps) {
  if (c.tid != 0u) return;
  device atomic_uint* fl = (device atomic_uint*)(c.res + R_QFLAG);
  device atomic_uint* errw = (device atomic_uint*)(c.res + R_ERR);
  const uint src = c.w == 0u ? c.G - 1u : c.w - 1u;
  for (uint lap = 1u; lap <= laps; ++lap) {
    const uint need = c.w == 0u ? lap - 1u : lap;
    uint spins = 0u;
    while (atomic_load_explicit(fl + src, memory_order_relaxed) < need &&
           atomic_load_explicit(errw, memory_order_relaxed) == 0u && spins < c.spin_max) ++spins;
    if (atomic_load_explicit(fl + src, memory_order_relaxed) < need) { set_err(c, E_RING | (c.w << 8) | lap); return; }
    atomic_store_explicit(fl + c.w, lap, memory_order_relaxed);
  }
}
// Entry of every persistent pass (M = 1 and verify). True = this threadgroup does not run the pass:
//   the previous pass failed (prev[R_ERRC]): record E_PREV and return at once in every threadgroup, so a stalled pass
//   costs one wait limit, not one per pass queued behind it, and the host reads the failure at the next sync;
//   the residency probe (prm[P_PROBE] laps): the flag ring instead of the pass;
//   the kernel_stall fault seam (prm[P_STALL], allow_fault_injection only): threadgroup G - 1 returns before any work,
//   so the other threadgroups' real waits (head flags, task counters, barriers) time out exactly as under a real stall.
inline bool pass_entry(thread Ctx& c, const device uint* prev) {
  if (prev[R_ERRC] != 0u) {
    if (c.tid == 0u) set_err(c, E_PREV);
    return true;
  }
  if (c.prm[P_PROBE] != 0u) { probe_ring(c, c.prm[P_PROBE]); return true; }
  return c.prm[P_STALL] != 0u && c.w == c.G - 1u;
}
