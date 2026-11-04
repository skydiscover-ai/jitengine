
// ================================================================== verify pass: MR rows in one dispatch
// The union of the rows' top-6 sets runs on one simdgroup in parallel (vtop6; an earlier build used thread 0's serial
// scan, and every verify row is bit-identical to it). M = 1 untouched.
// Row m feeds prm[P_RTOK + m] at position pos + m. Every row repeats the M = 1 pass's arithmetic at its own
// position: the same GEMV lane blocks, simd_sums and bf16 roundings (the primitives above, called per row), the
// same request-fixed strided attention partition and merge order, the same head deal and select. A weight block
// is loaded once per staging group and dotted with every row's x (the per-row calls share the address, so the
// repeats are cache hits or CSE'd); routed experts are dealt over the union of the group's top-6 sets.
#define MR __MR__u
#define XSV __XSV__u
#define RG2K __RG2K__u          // rows per x staging group: K = 2048 phases (QKV, RSG, EGU, DGU, HEAD)
#define RG4K __RG4K__u          // DOWN (x = 4,096 per row)
#define RGE __RGE__u            // EGU rows per group (<= 2: the 4-accumulator gate/up unit spills at 4 rows)
#define RG6K __RG6K__u          // DDOWN (x = 6,144 per row)
#define OG __OG__u              // O rows per group (UO x OG register accumulators) ...
#define KCO __KCO__u            // ... and the O input staged in K-chunks of KCO columns (same block order)
#define UO __UO__u              // 1-row O / DOWN units per simdgroup: ceil(2048 / (8 G))
#define SCR_ROW __SCR_ROW__ul   // bytes of one row's scratch region (the M = 1 layout)
#define VROW 64u
#define P_ROWS 152u
#define P_RTOK 160u
#define P_RU 168u
#define P_RCS 176u

struct VX {
  threadgroup uint* tku;       // [MR][8] every row's top-6 expert ids
  threadgroup float* tkw;      // [MR][8] their weights
  threadgroup uint* uid;       // [48] union of the staging group's experts
  threadgroup uint* urk;       // [48] rank of uid[k] in row i at nibble i (15 = not routed)
  threadgroup uint* nu;        // [1]
  threadgroup float* hb;       // [RG2K][128] projected heads
  threadgroup float* xsum;     // [XSV / 16] block sums of the staged rows (vprep)
  int tk_layer;
};

inline device uchar* RST(thread const Ctx& c, uint m) { return c.st + size_t(m) * SCR_ROW; }
inline device ushort* HSv(thread const Ctx& c, uint m, uint l) { return (device ushort*)(RST(c, m) + S_HS) + l * 2048u; }
inline device ushort* QBv(thread const Ctx& c, uint m, uint l) { return (device ushort*)(RST(c, m) + S_Q) + l * 4096u; }
inline device ushort* ABv(thread const Ctx& c, uint m, uint l) { return (device ushort*)(RST(c, m) + S_ATTN) + l * 4096u; }
inline device ushort* OBv(thread const Ctx& c, uint m, uint l) { return (device ushort*)(RST(c, m) + S_O) + l * 2048u; }
inline device float* SCv(thread const Ctx& c, uint m, uint l) { return (device float*)(RST(c, m) + S_SC) + l * 256u; }
inline device ushort* ACTv(thread const Ctx& c, uint m, uint l) { return (device ushort*)(RST(c, m) + S_ACT) + l * 6144u; }
inline device float* PARTv(thread const Ctx& c, uint m) { return (device float*)(RST(c, m) + S_PART); }
inline device ushort* LGv(thread const Ctx& c, uint m) { return (device ushort*)(RST(c, m) + S_LOGITS); }
inline device uint* SELPv(thread const Ctx& c, uint m) { return (device uint*)(RST(c, m) + S_SELP); }

// norm2048 into dst (a row slot of the staging buffer); same arithmetic
inline void vnorm2048(thread Ctx& c, thread const float* v, const device ushort* nw, threadgroup ushort* dst) {
  float ss = 0.0f;
  for (int j = 0; j < 8; ++j) ss += v[j] * v[j];
  ss = simd_sum(ss);
  if (c.lane == 0) c.red[c.sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (int s = 0; s < 8; ++s) tot += c.red[s];
  const float r = precise::rsqrt(tot / 2048.0f + EPS);
  const uint e0 = c.tid * 8;
  for (int j = 0; j < 8; ++j) dst[e0 + j] = f2bf(bf2f(nw[e0 + j]) * bfr(v[j] * r));
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
inline void vload_mid8(thread const Ctx& c, const device ushort* hs, const device ushort* ob, thread float* v) {
  float h[8], o[8];
  load_row8(c, hs, h);
  load_row8(c, ob, o);
  for (int j = 0; j < 8; ++j) v[j] = bfr(h[j] + o[j]);
}
// rows i < n of a scratch tensor (row i at src0 + i * SCR_ROW bytes), columns [off, off + len) -> xs + i * len
#define VSTG ((XSV / 8u + 255u) / 256u)
inline void vstage(thread Ctx& c, const device ushort* src0, uint n, uint off, uint len) {
  const uint per = len / 8u, tot = n * per;
  uint4 v[VSTG];
  for (uint j = 0; j < VSTG; ++j) {                       // every load issued before the first store (ld_vec)
    const uint q = c.tid + j * 256u;
    if (q < tot) v[j] = *((const device uint4*)(src0 + size_t(q / per) * (SCR_ROW / 2ul) + off) + q % per);
  }
  for (uint j = 0; j < VSTG; ++j) {
    const uint q = c.tid + j * 256u;
    if (q < tot) ((threadgroup uint4*)(c.xs + (q / per) * len))[q % per] = v[j];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The verify GEMV shares every row-independent step across the rows: each weight block is decoded ONCE (float of
// the masked nibbles) and the x side is prescaled ONCE per row (vprep: load_x16's xv stored back as bf16, exact
// since the scales are powers of two, and its sum in load_x16's order). Every int4 x prescaled-x product is exact in
// fp32, so a row's dot (same products, same summation order as qdot16) and its block value s * dot + b * sum are
// bit-identical to q4_blk's.
inline void vprep(thread Ctx& c, uint n, uint len, threadgroup float* xsum) {
  const uint per = len / 16u;
  for (uint it = c.tid; it < n * per; it += 256u) {
    threadgroup ushort* p = c.xs + (it / per) * len + (it % per) * 16u;
    float xv[16];
    const float sum = load_x16(p, 0u, xv);
    threadgroup uint4* q = (threadgroup uint4*)p;
    q[0] = uint4(pack2(xv[0], xv[1]), pack2(xv[2], xv[3]), pack2(xv[4], xv[5]), pack2(xv[6], xv[7]));
    q[1] = uint4(pack2(xv[8], xv[9]), pack2(xv[10], xv[11]), pack2(xv[12], xv[13]), pack2(xv[14], xv[15]));
    xsum[it] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
inline void load_xp16(const threadgroup ushort* x, uint xo, thread float* xv) {
  const threadgroup uint4* p = (const threadgroup uint4*)(x + xo);
  const uint4 a = p[0], b = p[1];
  const uint wv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  for (int i = 0; i < 8; ++i) { xv[2 * i] = lo16(wv[i]); xv[2 * i + 1] = hi16(wv[i]); }
}
inline void dec16(uint2 wq, thread float* w) {
  const uint ws[4] = {wq.x & 0xffffu, wq.x >> 16, wq.y & 0xffffu, wq.y >> 16};
  for (int j = 0; j < 4; ++j) {
    w[4 * j] = float(ws[j] & 0x000fu); w[4 * j + 1] = float(ws[j] & 0x00f0u);
    w[4 * j + 2] = float(ws[j] & 0x0f00u); w[4 * j + 3] = float(ws[j] & 0xf000u);
  }
}
inline float dot16(thread const float* w, thread const float* xv) {
  float a = 0.0f;
  for (int j = 0; j < 4; ++j) a += xv[4 * j] * w[4 * j] + xv[4 * j + 1] * w[4 * j + 1] + xv[4 * j + 2] * w[4 * j + 2] + xv[4 * j + 3] * w[4 * j + 3];
  return a;
}
// one weight row's 16-input block: decoded nibbles, scale, bias
struct WB { float w[16]; float s; float b; };
template <int K>
inline void wblk(thread WB& o, const device uchar* W, const device ushort* S, const device ushort* B, uint row, uint xo) {
  dec16(*(const device uint2*)(W + size_t(row) * (K / 2) + xo / 2), o.w);
  const size_t gi = size_t(row) * (K / 64) + xo / 64;
  o.s = bf2f(S[gi]);
  o.b = bf2f(B[gi]);
}
// q4_unit per row on prescaled x: weight rows r0, r0 + 1 against rows i < n (x row stride xst, sums stride sst)
template <int K, uint R>
inline void q4_unit_p(thread float* a0, thread float* a1, uint n, const device uchar* W, const device ushort* S,
                      const device ushort* B, uint r0, const threadgroup ushort* x, uint xst,
                      const threadgroup float* xsum, uint sst, uint lane) {
  for (uint i = 0; i < R; ++i) { a0[i] = 0.0f; a1[i] = 0.0f; }
  for (int blk = 0; blk < K / 512; ++blk) {
    const uint xo = blk * 512 + lane * 16;
    WB w0, w1;
    wblk<K>(w0, W, S, B, r0, xo);
    wblk<K>(w1, W, S, B, r0 + 1, xo);
    for (uint i = 0; i < R; ++i) {
      if (i < n) {
        float xv[16];
        load_xp16(x + i * xst, xo, xv);
        const float sm = xsum[i * sst + xo / 16u];
        a0[i] += w0.s * dot16(w0.w, xv) + w0.b * sm;
        a1[i] += w1.s * dot16(w1.w, xv) + w1.b * sm;
      }
    }
  }
  for (uint i = 0; i < R; ++i) { a0[i] = simd_sum(a0[i]); a1[i] = simd_sum(a1[i]); }
}
// q4_gu per row on prescaled x (gate rows then up rows of r0, r0 + 1) for the rows i < n with bit i of mask set
template <int K, uint R>
inline void q4_gu_p(thread float* g, thread float* u, uint n, uint mask, QM mg, QM mu, uint r0,
                    const threadgroup ushort* x, uint xst, const threadgroup float* xsum, uint sst, uint lane) {
  float g0[R], g1[R], u0[R], u1[R];
  for (uint i = 0; i < R; ++i) { g0[i] = 0.0f; g1[i] = 0.0f; u0[i] = 0.0f; u1[i] = 0.0f; }
  for (int blk = 0; blk < K / 512; ++blk) {
    const uint xo = blk * 512 + lane * 16;
    {
      WB w0, w1;
      wblk<K>(w0, mg.w, mg.s, mg.b, r0, xo);
      wblk<K>(w1, mg.w, mg.s, mg.b, r0 + 1, xo);
      for (uint i = 0; i < R; ++i) {
        if (i < n && ((mask >> i) & 1u)) {
          float xv[16];
          load_xp16(x + i * xst, xo, xv);
          const float sm = xsum[i * sst + xo / 16u];
          g0[i] += w0.s * dot16(w0.w, xv) + w0.b * sm;
          g1[i] += w1.s * dot16(w1.w, xv) + w1.b * sm;
        }
      }
    }
    {
      WB w0, w1;
      wblk<K>(w0, mu.w, mu.s, mu.b, r0, xo);
      wblk<K>(w1, mu.w, mu.s, mu.b, r0 + 1, xo);
      for (uint i = 0; i < R; ++i) {
        if (i < n && ((mask >> i) & 1u)) {
          float xv[16];
          load_xp16(x + i * xst, xo, xv);
          const float sm = xsum[i * sst + xo / 16u];
          u0[i] += w0.s * dot16(w0.w, xv) + w0.b * sm;
          u1[i] += w1.s * dot16(w1.w, xv) + w1.b * sm;
        }
      }
    }
  }
  for (uint i = 0; i < R; ++i) {
    g[2 * i] = simd_sum(g0[i]); g[2 * i + 1] = simd_sum(g1[i]); u[2 * i] = simd_sum(u0[i]); u[2 * i + 1] = simd_sum(u1[i]);
  }
}

// ------------------------------------------------------------------ verify: top-6 + union, combine, flags
// every row's top-6 (row m on simdgroup m; ensure_top6's arithmetic per row), then the UNION of the rows' experts in
// first-appearance order (row-major, rank-minor): row 0's experts are slots 0..5 in rank order. uid[k] = expert id of
// slot k, urk[k] = its rank in row m at nibble m (15 = not routed to row m)
inline void vtop6(thread Ctx& c, thread VX& vx, uint l) {
  if (vx.tk_layer == int(l)) return;
  if (c.sg < MR) {
    const uint m = c.sg;
    const device float* sc = SCv(c, m, l);
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
      if (c.lane == 0) vx.tku[m * 8u + uint(r)] = min(I, 255u);
    }
    float S = 0.0f;
    for (int r = 0; r < 6; ++r) S = S + simd_shuffle(sel, ushort(r));
    S = S + 1e-20f;
    if (c.lane < 6) vx.tkw[m * 8u + c.lane] = bfr(sel / S);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (c.sg == 0u) {   // K_X: the union on simdgroup 0 in parallel (earlier: thread 0's serial scan, ~6 MR x nu steps)
    // entry e = 6 m + r (row-major, rank-minor); lane owns entries e and e + 32. Slot of an entry = the number of
    // first appearances before its id's first appearance: first-appearance order, == the serial rule
    const uint NE = 6u * MR;
    const uint e0 = c.lane, e1 = c.lane + 32u;
    const uint id0 = e0 < NE ? vx.tku[(e0 / 6u) * 8u + e0 % 6u] : 0xfffffffeu;
    const uint id1 = e1 < NE ? vx.tku[(e1 / 6u) * 8u + e1 % 6u] : 0xfffffffdu;
    uint f0 = e0, f1 = e1;                           // first entry with the same id
    for (uint j = 0; j < NE; ++j) {
      const uint ja = simd_shuffle(id0, ushort(j & 31u)), jb = simd_shuffle(id1, ushort(j & 31u));
      const uint idj = j < 32u ? ja : jb;
      if (idj == id0 && j < f0) f0 = j;
      if (idj == id1 && j < f1) f1 = j;
    }
    const uint n0 = (e0 < NE && f0 == e0) ? 1u : 0u, n1 = (e1 < NE && f1 == e1) ? 1u : 0u;
    const uint c0 = simd_sum(n0);
    const uint p0 = simd_prefix_exclusive_sum(n0), p1 = c0 + simd_prefix_exclusive_sum(n1);
    const uint q0a = simd_shuffle(p0, ushort(f0 & 31u)), q0b = simd_shuffle(p1, ushort(f0 & 31u));
    const uint q1a = simd_shuffle(p0, ushort(f1 & 31u)), q1b = simd_shuffle(p1, ushort(f1 & 31u));
    const uint s0 = f0 < 32u ? q0a : q0b, s1 = f1 < 32u ? q1a : q1b;   // slot of entries e0, e1
    const uint nu = c0 + simd_sum(n1);
    if (n0) vx.uid[s0] = id0;
    if (n1) vx.uid[s1] = id1;
    uint w0 = 0xffffffffu, w1 = 0xffffffffu;         // rank words of slots lane, lane + 32 (15 = not routed)
    for (uint j = 0; j < NE; ++j) {
      const uint ja = simd_shuffle(s0, ushort(j & 31u)), jb = simd_shuffle(s1, ushort(j & 31u));
      const uint sj = j < 32u ? ja : jb;
      const uint sh = 4u * (j / 6u), rk = (j % 6u) << sh, clr = ~(0xfu << sh);
      if (sj == c.lane) w0 = (w0 & clr) | rk;
      if (sj == c.lane + 32u) w1 = (w1 & clr) | rk;
    }
    if (c.lane < nu) vx.urk[c.lane] = w0;
    if (c.lane + 32u < nu) vx.urk[c.lane + 32u] = w1;
    if (c.lane == 0) vx.nu[0] = nu;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  vx.tk_layer = int(l);
}
inline device ushort* Yv(thread const Ctx& c, uint m, uint r) { return (device ushort*)(RST(c, m) + S_Y) + r * 2048u; }
// combine8 for row m (its own top-6 weights and rank order); threadgroup 0 stores h_{l+1} of row m
inline void vcombine8(thread Ctx& c, thread VX& vx, uint m, uint l, thread float* v) {
  vtop6(c, vx, l);
  float h[8], o[8], y[7][8];
  load_row8(c, HSv(c, m, l), h);
  load_row8(c, OBv(c, m, l), o);
  for (int r = 0; r < 7; ++r) load_row8(c, Yv(c, m, uint(r)), y[r]);
  const threadgroup float* w = vx.tkw + m * 8u;
  for (int j = 0; j < 8; ++j) {
    float acc = bfr(bfr(y[0][j]) * w[0]);
    for (int r = 1; r < 6; ++r) acc = bfr(acc + bfr(bfr(y[r][j]) * w[r]));
    const float mlp = bfr(acc + bfr(y[6][j]));
    v[j] = bfr(bfr(h[j] + o[j]) + mlp);
  }
  if (c.w == 0) {
    device uint* hn = (device uint*)HSv(c, m, l + 1u) + c.tid * 4;
    for (int j = 0; j < 4; ++j) hn[j] = pack2(v[2 * j], v[2 * j + 1]);
  }
}
// flags of this threadgroup's heads, published after ALL rows of them are written
inline void vflow_publish(thread Ctx& c, uint l) {
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device);
  if (c.tid == 0u) {
    for (uint hd = c.w; hd < 40u; hd += c.G)
      atomic_store_explicit((device atomic_uint*)(c.res + R_QFLAG + hd), l + 1u, memory_order_relaxed);
  }
}
// attention tasks per kv head of layer l over every row: 6d = sum of the rows' own chunk counts, strided = M x tasks
inline uint vflow_layer_tasks(thread const Ctx& c, uint l) {
  if (((c.prm[P_MASK] >> PH_ATT) & 1u) == 0u) return 0u;
  const bool full = FIDX[l] >= 0;
  const uint T6 = full ? c.prm[P_6T] : 0u;
  if (T6 != 0u) {
    uint n = 0u;
    for (uint m = 0; m < MR; ++m) {                           // row m: ceil((pos + m + 1) / S_m), its own grid
      const uint Sm = pos_grid(c.pos + m, T6);
      n += (c.pos + m + Sm) / Sm;
    }
    return n;
  }
  return MR * (full ? c.prm[P_PF] : c.prm[P_PS]) * (8u / HG);
}

// ------------------------------------------------------------------ verify QKV: one head per threadgroup, all rows
inline void vph_qkv(thread Ctx& c, thread VX& vx, uint l) {
  const bool full = FIDX[l] >= 0;
  for (uint g0 = 0; g0 < MR; g0 += RG2K) {
    const uint n = min(RG2K, MR - g0);
    for (uint i = 0; i < n; ++i) {
      const uint m = g0 + i;
      float v[8];
      if (l == 0) {
        embed8(c, min(c.prm[P_RTOK + m], VOCAB - 1u), v);
        if (c.w == 0) {
          device uint* h0 = (device uint*)HSv(c, m, 0) + c.tid * 4;
          for (int j = 0; j < 4; ++j) h0[j] = pack2(v[2 * j], v[2 * j + 1]);
        }
      } else if (DENSE[l - 1u] != 0u) {
        load_row8(c, HSv(c, m, l), v);
      } else {
        vcombine8(c, vx, m, l - 1u, v);          // row m's h_l = its layer l - 1 MoE output
      }
      vnorm2048(c, v, LH(c, l, F_LN1), c.xs + i * 2048u);
    }
    vprep(c, n, 2048u, vx.xsum);
    for (uint hd = c.w; hd < 40u; hd += c.G) {
      const uint kind = hd < 32u ? 0u : (hd < 36u ? 1u : 2u);
      const uint local = kind == 0u ? hd : (kind == 1u ? hd - 32u : hd - 36u);
      const uint f = kind == 0u ? F_QW : (kind == 1u ? F_KW : F_VW);
      const device uchar* W = LF(c, l, f) + size_t(local) * 128 * 1024;
      const device ushort* S = LH(c, l, f + 1) + local * 128 * 32;
      const device ushort* B = LH(c, l, f + 2) + local * 128 * 32;
      for (uint u = 0; u < 8u; ++u) {
        const uint r0 = c.sg * 16u + u * 2u;
        float a0[RG2K], a1[RG2K];
        q4_unit_p<2048, RG2K>(a0, a1, n, W, S, B, r0, c.xs, 2048u, vx.xsum, 128u, c.lane);
        if (c.lane == 0) {
          for (uint i = 0; i < RG2K; ++i) if (i < n) { vx.hb[i * 128u + r0] = bfr(a0[i]); vx.hb[i * 128u + r0 + 1] = bfr(a1[i]); }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i0 = 0; i0 < n; i0 += 2u) {
        const uint hh = c.tid >> 7, t = c.tid & 127u;        // row i0 + hh on threads 128 * hh ..
        const uint i = i0 + hh;
        const bool on = i < n;
        const uint m = g0 + i;
        threadgroup float* hb = vx.hb + i * 128u;
        float y = 0.0f;
        if (on) {
          y = hb[t];
          if (kind < 2u) {
            const float ss = simd_sum(y * y);
            if (c.lane == 0) c.red[16 + c.sg] = ss;
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (on && kind < 2u) {
          const uint rb = 16u + 4u * hh;
          const float tot = ((c.red[rb] + c.red[rb + 1]) + c.red[rb + 2]) + c.red[rb + 3];
          const float r = precise::rsqrt(tot / 128.0f + EPS);
          const device ushort* nw = LH(c, l, kind == 0u ? F_QN : F_KN);
          y = bfr(bf2f(nw[t]) * bfr(y * r));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (on) hb[t] = y;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (on) {
          float out = y;
          if (!full && kind < 2u) {
            const uint ii = t & 63u;
            const float x1 = hb[ii], x2 = hb[ii + 64u];
            const float cs = as_type<float>(c.prm[P_RCS + m * 128u + ii]);
            const float sn = as_type<float>(c.prm[P_RCS + m * 128u + 64u + ii]);
            out = t < 64u ? bfr(x1 * cs - x2 * sn) : bfr(x2 * cs + x1 * sn);
          }
          const uint pos = c.pos + m;
          const uint slot = full ? pos : pos % RING;
          if (kind == 0u) QBv(c, m, l)[local * 128u + t] = f2bf(out);
          else kv_head(c, l, kind == 2u, local)[size_t(slot) * 128u + t] = f2bf(out);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  }
#if !CHAIN
  vflow_publish(c, l);
#endif
}

// ------------------------------------------------------------------ verify attention: the strided task of each row
// == attend_strided (abl = 0) with the row's queries, causal limit nk and ring position pm
inline void vattend_strided(thread const Ctx& c, const device ushort* qrow, uint l, uint kvh, uint g0, uint part,
                            uint parts, uint nk, bool ring, uint pm, device float* rows) {
  const device ushort* qp = qrow + (kvh * 8u + g0) * 128u + c.lane * 4u;
  float qv[HG][4], acc[HG][4], m[HG], z[HG];
  for (int h = 0; h < int(HG); ++h) {
    const uint2 w = *(const device uint2*)(qp + h * 128);
    qv[h][0] = lo16(w.x) * QSCALE; qv[h][1] = hi16(w.x) * QSCALE;
    qv[h][2] = lo16(w.y) * QSCALE; qv[h][3] = hi16(w.y) * QSCALE;
    for (int j = 0; j < 4; ++j) acc[h][j] = 0.0f;
    m[h] = -INFINITY; z[h] = 0.0f;
  }
  const device ushort* Kh = kv_head(c, l, false, kvh) + c.lane * 4u;
  const device ushort* Vh = kv_head(c, l, true, kvh) + c.lane * 4u;
  const uint end = nk;
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
  rows += g0 * PROW;
  for (int h = 0; h < int(HG); ++h) {
    *(device float4*)(rows + h * PROW + 4u + c.lane * 4u) = float4(acc[h][0], acc[h][1], acc[h][2], acc[h][3]);
    if (c.lane == 0) { rows[h * PROW] = m[h]; rows[h * PROW + 1u] = z[h]; }
  }
}
// tasks of every row, row fastest (the rows of one (kv head, part / chunk) run together and share K/V through the
// cache; since the threadgroup-first deal, a 6d chunk's rows sit on adjacent threadgroups). Full layers: layout
// 6d -- item (kv head, chunk, row) = att6 of that row's queries over the keys [ch S_m, (ch + 1) S_m) below
// the row's causal limit pos + m + 1, into the row's partials at (kvh * C_m + chunk); S_m = pos_grid(pos + m) and C_m
// are the row's own (computed inline: a per-row array cost more at M = 4 / 8): exactly the M = 1
// pass at position pos + m (chunks wholly beyond a row's limit are not computed, as at M = 1). Rows on either side of a
// grid change simply have different S. Sliding layers (and the strided full partition): the M = 1 strided task
// of each row.
inline void vph_att(thread Ctx& c, uint l) {
  const bool full = FIDX[l] >= 0;
  const uint T6 = full ? c.prm[P_6T] : 0u;
  if (T6 != 0u) {
    uint CL = 0u;                                             // the most chunks per kv head of any row
    for (uint m = 0; m < MR; ++m) { const uint Sm = pos_grid(c.pos + m, T6); CL = max(CL, (c.pos + m + Sm) / Sm); }
    for (uint it = gsg(c); it < 4u * CL * MR; it += nsg(c)) {         // threadgroup-first
      const uint m = it % MR, t = it / MR, kvh = t / CL, ch = t % CL;
      const uint nk = c.pos + m + 1u;
      const uint Sm = pos_grid(nk - 1u, T6), Cm = (nk + Sm - 1u) / Sm;
      if (ch >= Cm) continue;
      const uint kb = ch * Sm, ke = min(kb + Sm, nk);
      FLOW_WAIT(c, l, kvh);
      att6(QBv(c, m, l) + kvh * 8u * 128u, kv_head(c, l, false, kvh), kv_head(c, l, true, kvh), kb, ke,
           PARTv(c, m) + size_t(kvh * Cm + ch) * 8u * PROW, c.lane);
      FLOW_DONE(c, kvh);
    }
    return;
  }
  const uint parts = full ? c.prm[P_PF] : c.prm[P_PS];
  const uint NG = 8u / HG;
  const uint per = 4u * parts * NG;
  for (uint it = gsg(c); it < MR * per; it += nsg(c)) {
    const uint m = it % MR, t = it / MR;
    const uint item = t / NG, g0 = (t % NG) * HG;
    const uint pos = c.pos + m;
    const uint nk = full ? pos + 1u : min(pos + 1u, RING);
    FLOW_WAIT(c, l, item / parts);
    vattend_strided(c, QBv(c, m, l), l, item / parts, g0, item % parts, parts, nk, !full, pos % RING,
                    PARTv(c, m) + size_t(item) * 8u * PROW);
    FLOW_DONE(c, item / parts);
  }
}
// == ph_merge's task t over C partial rows of one row's partials, into that row's attention output
inline void vmerge_task(thread const Ctx& c, uint l, uint t, uint C, const device float* part, device ushort* ab) {
  const device ushort* sinks = LH(c, l, F_SINK);
  const uint h = t >> 3, d0 = (t & 7u) * 16u, kvh = h >> 3, g = h & 7u;
  const device float* base = part + (size_t(kvh * C) * 8u + g) * PROW;
  const uint stride = 8u * PROW;
  const float sink2 = bf2f(sinks[h]) * kLog2e;
  float mx = sink2;
  float den = 0.0f;
  float4 o0 = float4(0.0f), o1 = float4(0.0f), o2 = float4(0.0f), o3 = float4(0.0f);
  if (C <= 64u) {
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
    device uint* dst = (device uint*)(ab + h * 128u + d0 + c.lane * 4u);
    dst[0] = pack2(v.x / den, v.y / den);
    dst[1] = pack2(v.z / den, v.w / den);
  }
}
inline void vph_merge(thread Ctx& c, uint l) {
  const bool full = FIDX[l] >= 0;
  const uint T6 = full ? c.prm[P_6T] : 0u;
  const uint C = full ? c.prm[P_PF] : c.prm[P_PS];
  for (uint it = gsg(c); it < 256u * MR; it += nsg(c)) {
    const uint m = it % MR;
    const uint Sm = T6 != 0u ? pos_grid(c.pos + m, T6) : 1u;
    const uint Cm = T6 != 0u ? (c.pos + m + Sm) / Sm : C;      // 6d: ceil((pos + m + 1) / S_m), the row's own grid
    FLOW_WAIT_TASKS(c, l, (it / MR) >> 6);
    vmerge_task(c, l, it / MR, Cm, PARTv(c, m), ABv(c, m, l));
  }
}

// ------------------------------------------------------------------ verify O: OG rows per group, x in K-chunks
inline void vph_o(thread Ctx& c, thread VX& vx, uint l) {
  const QM mw = qm_layer(c, l, F_OW);
  for (uint g0 = 0; g0 < MR; g0 += OG) {
    const uint n = min(OG, MR - g0);
    float a[UO][OG];
    for (uint k = 0; k < UO; ++k) for (uint i = 0; i < OG; ++i) a[k][i] = 0.0f;
    for (uint ch = 0; ch < 4096u / KCO; ++ch) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      vstage(c, ABv(c, g0, l), n, ch * KCO, KCO);
      vprep(c, n, KCO, vx.xsum);
      for (uint k = 0; k < UO; ++k) {
        const uint r = gsg(c) + k * nsg(c);
        if (r < 2048u) {
          for (uint b = 0; b < KCO / 512u; ++b) {
            const uint xl = b * 512u + c.lane * 16u, xo = ch * KCO + xl;
            WB w;
            wblk<4096>(w, mw.w, mw.s, mw.b, r, xo);
            for (uint i = 0; i < OG; ++i) {
              if (i < n) {
                float xv[16];
                load_xp16(c.xs + i * KCO, xl, xv);
                a[k][i] += w.s * dot16(w.w, xv) + w.b * vx.xsum[i * (KCO / 16u) + xl / 16u];
              }
            }
          }
        }
      }
    }
    for (uint k = 0; k < UO; ++k) {
      const uint r = gsg(c) + k * nsg(c);
      if (r < 2048u) {
        for (uint i = 0; i < OG; ++i) {
          if (i < n) {
            const float y = simd_sum(a[k][i]);
            if (c.lane == 0) OBv(c, g0 + i, l)[r] = f2bf(y);
          }
        }
      }
    }
  }
}

// ------------------------------------------------------------------ verify MoE block (the MoE block's team layout over M rows)
// The M = 1 block's structure (one phase, one grid barrier, router counter, shared team first, per-slot counters) over
// the UNION of the rows' top-6 sets (vtop6). A routed piece reads its expert's rows once and computes gate/up and DOWN
// for every row routed to its slot; row m's act goes to its own rank's slot of row m's ACT, its DOWN output to row m's
// Y[rank]; every row is combined in its own rank order by the next phase's prologue (vcombine8). x is staged in groups
// of RG2K rows and prescaled once (vprep); gate/up runs in sub-batches of RGE rows (the 4-accumulator unit spills
// beyond 2). Per-row arithmetic = the M = 1 pieces' (prescaled-x products are exact, same block order, same simd_sum
// trees; float4 simd_sum == 4 scalar ones): every row is bit-identical to its M = 1 pass.
inline const device uchar* vebase(thread const Ctx& c, uint l) {
  const ulong bank = c.desc[l * NF + F_EBANK];
  return (bank == 1ul ? c.b1 : c.b2) + c.desc[l * NF + F_EBASE];
}
inline uint vrank(uint rk, uint m) { return (rk >> (4u * m)) & 0xfu; }
// rows g0 .. g0 + n - 1 of the block's input (normed bf(h + o)) into xs, prescaled when prep (the router needs raw x)
inline void vmoe_stage_x(thread Ctx& c, thread VX& vx, uint l, uint g0, uint n, bool prep) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = 0; i < n; ++i) {
    float v[8];
    vload_mid8(c, HSv(c, g0 + i, l), OBv(c, g0 + i, l), v);
    vnorm2048(c, v, LH(c, l, F_LN2), c.xs + i * 2048u);
  }
  if (prep) vprep(c, n, 2048u, vx.xsum);
}
// router rows u = first, first + step, ... < 128 (2-row units) of the staged rows (raw x)
inline void vrouter(thread Ctx& c, uint l, uint g0, uint n, uint first, uint step) {
  const device ushort* RW = LH(c, l, F_RW);
  for (uint u = first; u < 128u; u += step) {
    for (uint i = 0; i < RG2K; ++i) {
      if (i < n) {
        float acc[2];
        bf_unit(acc, RW, u * 2u, c.xs + i * 2048u, c.lane);
        if (c.lane == 0) {
          device float* sc = SCv(c, g0 + i, l);
          sc[u * 2u] = sigmoid_m(bfr(acc[0])); sc[u * 2u + 1u] = sigmoid_m(bfr(acc[1]));
        }
      }
    }
  }
}
// gate/up of the shared expert, member q, for the staged rows g0 .. g0 + n - 1
inline void vgu_piece_shared(thread Ctx& c, thread VX& vx, uint l, uint q, uint g0, uint n) {
  const QM mg = qm_layer(c, l, F_SGW), mu = qm_layer(c, l, F_SUW);
  for (uint i = q + MQS * c.sg; i < 512u; i += MQS * 8u) {
    const uint r0 = i * 2u;
    for (uint b = 0; b < n; b += RGE) {
      const uint nb = min(RGE, n - b);
      float g[2 * RGE], up[2 * RGE];
      q4_gu_p<2048, RGE>(g, up, nb, 0xffu, mg, mu, r0, c.xs + b * 2048u, 2048u, vx.xsum + b * 128u, 128u, c.lane);
      if (c.lane == 0) {
        for (uint k = 0; k < RGE; ++k) {
          if (k < nb) {
            device uint* act = (device uint*)(ACTv(c, g0 + b + k, l) + 3072);
            act[i] = pack2(swiglu(bfr(g[2 * k]), bfr(up[2 * k])), swiglu(bfr(g[2 * k + 1]), bfr(up[2 * k + 1])));
          }
        }
      }
    }
  }
}
// gate/up of union slot s, member q, for the staged rows of the group routed to s
inline void vgu_piece_routed(thread Ctx& c, thread VX& vx, uint l, uint s, uint q, uint g0, uint n) {
  const uint rk = vx.urk[s];
  uint gm = 0u;
  for (uint k = 0; k < RG2K; ++k) if (k < n && vrank(rk, g0 + k) != 0xfu) gm |= 1u << k;
  if (gm == 0u) return;
  const device uchar* rec = vebase(c, l) + size_t(vx.uid[s]) * E_STRIDE;
  const QM mg = qm_expert(rec, EG_W, EG_S, EG_B), mu = qm_expert(rec, EU_W, EU_S, EU_B);
  for (uint i = q + MQ * c.sg; i < 256u; i += MQ * 8u) {
    const uint r0 = i * 2u;
    for (uint b = 0; b < n; b += RGE) {
      const uint nb = min(RGE, n - b);
      const uint bm = (gm >> b) & ((1u << RGE) - 1u);
      if (bm == 0u) continue;
      float g[2 * RGE], up[2 * RGE];
      q4_gu_p<2048, RGE>(g, up, nb, bm, mg, mu, r0, c.xs + b * 2048u, 2048u, vx.xsum + b * 128u, 128u, c.lane);
      if (c.lane == 0) {
        for (uint k = 0; k < RGE; ++k) {
          if (k < nb && ((bm >> k) & 1u)) {
            const uint m = g0 + b + k;
            device uint* act = (device uint*)ACTv(c, m, l);
            act[(vrank(rk, m) * 512u + r0) / 2] = pack2(swiglu(bfr(g[2 * k]), bfr(up[2 * k])),
                                                       swiglu(bfr(g[2 * k + 1]), bfr(up[2 * k + 1])));
          }
        }
      }
    }
  }
}
// DOWN of union slot s, member q: MDR-row units; the expert's rows loaded once, every routed row inside
inline void vdn_piece_routed(thread Ctx& c, thread VX& vx, uint l, uint s, uint q) {
  const uint rk = vx.urk[s];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {   // row m's act of its rank of s (512) -> xs + m * 512; every uint4 load issued before the first store
    uint4 v[2];
    for (uint j = 0; j < 2u; ++j) {
      const uint t = c.tid + j * 256u, m = t >> 6;
      const uint rank = m < MR ? vrank(rk, m) : 0xfu;
      if (rank != 0xfu) v[j] = ((const device uint4*)(ACTv(c, m, l) + rank * 512u))[t & 63u];
    }
    for (uint j = 0; j < 2u; ++j) {
      const uint t = c.tid + j * 256u, m = t >> 6;
      if (m < MR && vrank(rk, m) != 0xfu) ((threadgroup uint4*)c.xs)[t] = v[j];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  vprep(c, MR, 512u, vx.xsum);                       // rows not routed to s are prepped and never read
  const device uchar* rec = vebase(c, l) + size_t(vx.uid[s]) * E_STRIDE;
  const device uchar* W = rec + ED_W;
  const device ushort* S = (const device ushort*)(rec + ED_S);
  const device ushort* B = (const device ushort*)(rec + ED_B);
  const uint xo = c.lane * 16u;
  for (uint i = q + MQ * c.sg; i < 2048u / MDR; i += MQ * 8u) {
    uint2 wq[MDR]; float sv[MDR], bv[MDR];
    for (uint rr = 0; rr < MDR; ++rr) {
      const uint row = i * MDR + rr;
      wq[rr] = *(const device uint2*)(W + size_t(row) * 256 + xo / 2);
      const size_t gi = size_t(row) * 8 + xo / 64;
      sv[rr] = bf2f(S[gi]); bv[rr] = bf2f(B[gi]);
    }
    for (uint m = 0; m < MR; ++m) {
      const uint rank = vrank(rk, m);
      if (rank == 0xfu) continue;
      float xv[16];
      load_xp16(c.xs + m * 512u, xo, xv);
      const float sm = vx.xsum[m * 32u + c.lane];
      float pe[MDR];
      for (uint rr = 0; rr < MDR; ++rr) pe[rr] = sv[rr] * qdot16(wq[rr], xv) + bv[rr] * sm;
      const float4 y = simd_sum(float4(pe[0], pe[1], pe[2], pe[3]));
      if (c.lane == 0) {
        device ushort* yo = Yv(c, m, rank) + i * MDR;
        yo[0] = f2bf(y.x); yo[1] = f2bf(y.y); yo[2] = f2bf(y.z); yo[3] = f2bf(y.w);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
// DOWN of the shared expert, member q, for every row
inline void vdn_piece_shared(thread Ctx& c, thread VX& vx, uint l, uint q) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {   // every row's shared act (1024) -> xs + m * 1024; loads first
    uint4 v[4];
    for (uint j = 0; j < 4u; ++j) {
      const uint t = c.tid + j * 256u;
      if (t < MR * 128u) v[j] = ((const device uint4*)(ACTv(c, t >> 7, l) + 3072u))[t & 127u];
    }
    for (uint j = 0; j < 4u; ++j) {
      const uint t = c.tid + j * 256u;
      if (t < MR * 128u) ((threadgroup uint4*)c.xs)[t] = v[j];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  vprep(c, MR, 1024u, vx.xsum);
  const QM ms = qm_layer(c, l, F_SDW);
  for (uint i = q + MQS * c.sg; i < 2048u / MDR; i += MQS * 8u) {
    uint2 wq[MDR][2]; float sv[MDR][2], bv[MDR][2];
    for (uint rr = 0; rr < MDR; ++rr) {
      for (uint blk = 0; blk < 2u; ++blk) {
        const uint row = i * MDR + rr, xo = blk * 512u + c.lane * 16u;
        wq[rr][blk] = *(const device uint2*)(ms.w + size_t(row) * 512 + xo / 2);
        const size_t gi = size_t(row) * 16 + xo / 64;
        sv[rr][blk] = bf2f(ms.s[gi]); bv[rr][blk] = bf2f(ms.b[gi]);
      }
    }
    for (uint m = 0; m < MR; ++m) {
      float pe[MDR];
      for (uint rr = 0; rr < MDR; ++rr) pe[rr] = 0.0f;
      for (uint blk = 0; blk < 2u; ++blk) {
        const uint xo = blk * 512u + c.lane * 16u;
        float xv[16];
        load_xp16(c.xs + m * 1024u, xo, xv);
        const float sm = vx.xsum[m * 64u + xo / 16u];
        for (uint rr = 0; rr < MDR; ++rr) pe[rr] += sv[rr][blk] * qdot16(wq[rr][blk], xv) + bv[rr][blk] * sm;
      }
      const float4 y = simd_sum(float4(pe[0], pe[1], pe[2], pe[3]));
      if (c.lane == 0) {
        device ushort* yo = Yv(c, m, 6u) + i * MDR;
        yo[0] = f2bf(y.x); yo[1] = f2bf(y.y); yo[2] = f2bf(y.z); yo[3] = f2bf(y.w);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
#if !CHAIN
// the persistent verify MoE block of sparse layer l (moe_fused over M rows; same phase-mask semantics)
inline bool vmoe_fused(thread Ctx& c, thread VX& vx, uint l) {
  const uint msk = c.prm[P_MASK];
  const bool dr = ((msk >> PH_RSG) & 1u) != 0u, dg = ((msk >> PH_EGU) & 1u) != 0u, dd = ((msk >> PH_DOWN) & 1u) != 0u;
  const uint NRT = c.G - MQS;
  const bool one = RG2K >= MR;                       // every row in one staging group
  const bool rtg = dr && c.w >= MQS;
  if (rtg) {
    for (uint g0 = 0; g0 < MR; g0 += RG2K) {
      const uint n = min(RG2K, MR - g0);
      vmoe_stage_x(c, vx, l, g0, n, false);
      vrouter(c, l, g0, n, c.sg * NRT + (c.w - MQS), NRT * 8u);
    }
    team_signal(c, CTR(c, l, CT_ROUTER));
  }
  bool ready = false;
  if (dg) {
    for (uint g0 = 0; g0 < MR; g0 += RG2K) {
      const uint n = min(RG2K, MR - g0);
      const bool last = g0 + RG2K >= MR;
      if (one && rtg) { threadgroup_barrier(mem_flags::mem_threadgroup); vprep(c, n, 2048u, vx.xsum); }
      else vmoe_stage_x(c, vx, l, g0, n, true);
      for (uint p = c.w; ; p += c.G) {
        if (p >= MQS) {
          if (!ready) {
            if (dr && !team_wait(c, CTR(c, l, CT_ROUTER), NRT)) return false;
            vtop6(c, vx, l);
            ready = true;
          }
          if (p >= MQS + vx.nu[0] * MQ) break;
          const uint r = p - MQS;
          vgu_piece_routed(c, vx, l, r / MQ, r % MQ, g0, n);
          if (last) team_signal(c, CTR(c, l, r / MQ));
        } else {
          vgu_piece_shared(c, vx, l, p, g0, n);
          if (last) team_signal(c, CTR(c, l, CT_SHARED));
        }
      }
    }
  }
  if (!ready) {
    if (dr && !team_wait(c, CTR(c, l, CT_ROUTER), NRT)) return false;
    vtop6(c, vx, l);
  }
  if (dd) {
    const uint np = MQS + vx.nu[0] * MQ;
    for (uint p = c.w; p < np; p += c.G) {
      const bool sh = p < MQS;
      const uint r = sh ? 0u : p - MQS;
      if (dg && !team_wait(c, CTR(c, l, sh ? CT_SHARED : r / MQ), sh ? MQS : MQ)) return false;
      if (sh) vdn_piece_shared(c, vx, l, p);
      else vdn_piece_routed(c, vx, l, r / MQ, r % MQ);
    }
  }
  return true;
}
#else
// chain mode: the verify block's pieces as plain dispatches (RSG = router + shared gate/up, EGU = routed gate/up,
// DOWN = every DOWN piece)
inline void vph_rsg(thread Ctx& c, thread VX& vx, uint l) {
  for (uint g0 = 0; g0 < MR; g0 += RG2K) {
    const uint n = min(RG2K, MR - g0);
    vmoe_stage_x(c, vx, l, g0, n, false);
    vrouter(c, l, g0, n, gsg(c), nsg(c));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    vprep(c, n, 2048u, vx.xsum);
    for (uint p = c.w; p < MQS; p += c.G) vgu_piece_shared(c, vx, l, p, g0, n);
  }
}
inline void vph_egu(thread Ctx& c, thread VX& vx, uint l) {
  vtop6(c, vx, l);
  const uint np = vx.nu[0] * MQ;
  for (uint g0 = 0; g0 < MR; g0 += RG2K) {
    const uint n = min(RG2K, MR - g0);
    vmoe_stage_x(c, vx, l, g0, n, true);
    for (uint p = c.w; p < np; p += c.G) vgu_piece_routed(c, vx, l, p / MQ, p % MQ, g0, n);
  }
}
inline void vph_down(thread Ctx& c, thread VX& vx, uint l) {
  vtop6(c, vx, l);
  const uint np = MQS + vx.nu[0] * MQ;
  for (uint p = c.w; p < np; p += c.G) {
    if (p < MQS) vdn_piece_shared(c, vx, l, p);
    else vdn_piece_routed(c, vx, l, (p - MQS) / MQ, (p - MQS) % MQ);
  }
}
#endif

// ------------------------------------------------------------------ verify dense MLP (layers 0-1)
inline void vph_dgu(thread Ctx& c, thread VX& vx, uint l) {
  const QM mg = qm_layer(c, l, F_DGW), mu = qm_layer(c, l, F_DUW);
  for (uint g0 = 0; g0 < MR; g0 += RG2K) {
    const uint n = min(RG2K, MR - g0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < n; ++i) {
      float v[8];
      vload_mid8(c, HSv(c, g0 + i, l), OBv(c, g0 + i, l), v);
      vnorm2048(c, v, LH(c, l, F_LN2), c.xs + i * 2048u);
    }
    vprep(c, n, 2048u, vx.xsum);
    for (uint u = gsg(c); u < 3072u; u += nsg(c)) {
      float g[2 * RG2K], up[2 * RG2K];
      q4_gu_p<2048, RG2K>(g, up, n, 0xffu, mg, mu, u * 2u, c.xs, 2048u, vx.xsum, 128u, c.lane);
      if (c.lane == 0) {
        for (uint i = 0; i < RG2K; ++i) {
          if (i < n) {
            device uint* act = (device uint*)ACTv(c, g0 + i, l);
            act[u] = pack2(swiglu(bfr(g[2 * i]), bfr(up[2 * i])), swiglu(bfr(g[2 * i + 1]), bfr(up[2 * i + 1])));
          }
        }
      }
    }
  }
}
inline void vph_ddown(thread Ctx& c, thread VX& vx, uint l) {
  const QM mw = qm_layer(c, l, F_DDW);
  for (uint g0 = 0; g0 < MR; g0 += RG6K) {
    const uint n = min(RG6K, MR - g0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    vstage(c, ACTv(c, g0, l), n, 0u, 6144u);
    vprep(c, n, 6144u, vx.xsum);
    for (uint u = gsg(c); u < 1024u; u += nsg(c)) {
      float a0[RG6K], a1[RG6K];
      q4_unit_p<6144, RG6K>(a0, a1, n, mw.w, mw.s, mw.b, u * 2u, c.xs, 6144u, vx.xsum, 384u, c.lane);
      if (c.lane == 0) {
        for (uint i = 0; i < RG6K; ++i) {
          if (i < n) {
            const uint m = g0 + i;
            const uint hwv = ((const device uint*)HSv(c, m, l))[u], owv = ((const device uint*)OBv(c, m, l))[u];
            const float m0 = bfr(lo16(hwv) + lo16(owv)), m1 = bfr(hi16(hwv) + hi16(owv));
            ((device uint*)HSv(c, m, l + 1))[u] = pack2(m0 + bfr(a0[i]), m1 + bfr(a1[i]));
          }
        }
      }
    }
  }
}

// ------------------------------------------------------------------ verify head: logits + statistics per row
inline void vph_head(thread Ctx& c, thread VX& vx) {
  const QM m = QM{GF(c, G_HW), (const device ushort*)GF(c, G_HW + 1), (const device ushort*)GF(c, G_HW + 2)};
  const uint poison = c.prm[P_POISON];
  for (uint g0 = 0; g0 < MR; g0 += RG2K) {
    const uint n = min(RG2K, MR - g0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < n; ++i) {
      float v8[8];
      if (DENSE[NL - 1u] != 0u) load_row8(c, HSv(c, g0 + i, NL), v8);
      else vcombine8(c, vx, g0 + i, NL - 1u, v8);
      vnorm2048(c, v8, (const device ushort*)GF(c, G_NORM), c.xs + i * 2048u);
    }
    vprep(c, n, 2048u, vx.xsum);
    float mx[RG2K], se[RG2K], tv[RG2K];
    uint fin[RG2K], ti[RG2K];
    for (uint i = 0; i < RG2K; ++i) { mx[i] = -INFINITY; se[i] = 0.0f; fin[i] = 1u; tv[i] = -INFINITY; ti[i] = 0xffffffffu; }
    for (uint u = gsg(c); u < VOCAB / 2u; u += nsg(c)) {
      float a0[RG2K], a1[RG2K];
      q4_unit_p<2048, RG2K>(a0, a1, n, m.w, m.s, m.b, u * 2u, c.xs, 2048u, vx.xsum, 128u, c.lane);
      for (uint i = 0; i < RG2K; ++i) {
        if (i < n) {
          float vv[2] = {bfr(a0[i]), bfr(a1[i])};
          if ((poison >> (g0 + i)) & 1u) { vv[0] = vv[0] * NAN; vv[1] = vv[1] * NAN; }
          if (c.lane == 0) ((device uint*)LGv(c, g0 + i))[u] = pack2(vv[0], vv[1]);
          for (int r = 0; r < 2; ++r) {
            const float x = vv[r];
            const uint id = u * 2u + uint(r);
            fin[i] &= isfinite(x) ? 1u : 0u;
            if (x > mx[i]) { se[i] = se[i] * exp(mx[i] - x) + 1.0f; mx[i] = x; }
            else if (x != -INFINITY) se[i] += exp(x - mx[i]);
            const float thr = simd_shuffle(tv[i], ushort(19));
            const uint thi = simd_shuffle(ti[i], ushort(19));
            if (tbetter(x, id, thr, thi)) {
              const float pv = simd_shuffle_up(tv[i], ushort(1));
              const uint pi = simd_shuffle_up(ti[i], ushort(1));
              const bool mine = tbetter(x, id, tv[i], ti[i]);
              const bool prevb = c.lane > 0 && tbetter(x, id, pv, pi);
              if (mine) { if (prevb) { tv[i] = pv; ti[i] = pi; } else { tv[i] = x; ti[i] = id; } }
            }
          }
        }
      }
    }
    for (uint i = 0; i < RG2K; ++i) {
      if (i < n) {
        device uint* sp = SELPv(c, g0 + i) + (c.w * 8u + c.sg) * 48u;
        if (c.lane == 0) { sp[0] = as_type<uint>(mx[i]); sp[1] = as_type<uint>(se[i]); sp[2] = fin[i]; }
        if (c.lane < 20u) { sp[4 + c.lane] = as_type<uint>(tv[i]); sp[24 + c.lane] = ti[i]; }
      }
    }
  }
}
// select of row m on threadgroup m (simdgroup 0), all rows in parallel; == ph_select with the row's buffers
inline void vph_select(thread Ctx& c) {
  if (c.w >= MR || c.sg != 0u) return;
  const uint row = c.w;
  device uint* res = c.res + row * VROW;
  const uint NS = c.G * 8u;
  const device uint* sp = SELPv(c, row);
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
  const device ushort* lg = LGv(c, row);
  if (c.lane < 20u) { res[R_TOPI + c.lane] = myi; res[R_TOPV + c.lane] = as_type<uint>(myv - lse); }
  const uint amax = simd_shuffle(myi, ushort(0));
  uint chosen = amax;
  if (c.prm[P_SELMODE] == 1u) {
    const uint K = min(max(c.prm[P_TOPK], 1u), 20u);
    const float invT = as_type<float>(c.prm[P_INVT]), topp = as_type<float>(c.prm[P_TOPP]);
    const float u = as_type<float>(c.prm[P_RU + row]);
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
    res[R_TOKEN] = chosen;
    res[R_FINITE] = fin;
    res[R_ARGMAX] = amax;
    res[R_LSE] = as_type<uint>(lse);
    res[R_CHOSEN_LP] = as_type<uint>(bf2f(lg[chosen]) - lse);
    const uint F = c.prm[P_FORCED];
    res[R_FORCED_LP] = as_type<uint>(F < VOCAB ? bf2f(lg[F]) - lse : NAN);
    res[R_DONE] = 1u;
  }
}
inline void vrun_phase(thread Ctx& c, thread VX& vx, uint ph, uint l) {
  if (((c.prm[P_MASK] >> ph) & 1u) == 0u) return;
  switch (ph) {
    case PH_QKV: vph_qkv(c, vx, l); break;
    case PH_ATT: vph_att(c, l); break;
    case PH_MERGE: vph_merge(c, l); break;
    case PH_O: vph_o(c, vx, l); break;
#if CHAIN
    case PH_RSG: vph_rsg(c, vx, l); break;
    case PH_EGU: vph_egu(c, vx, l); break;
    case PH_DOWN: vph_down(c, vx, l); break;
#endif
    case PH_DGU: vph_dgu(c, vx, l); break;
    case PH_DDOWN: vph_ddown(c, vx, l); break;
    case PH_HEAD: vph_head(c, vx); break;
    case PH_SELECT: vph_select(c); break;
    default: break;
  }
}
