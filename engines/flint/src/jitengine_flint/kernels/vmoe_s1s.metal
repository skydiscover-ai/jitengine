// ------------------------------------------------------------------ S1s: one union sweep, MR >= 6
// The router threadgroups publish each row's vprep values (load_x16's prescaled x as bf16 + block sums) into the row's
// LOGITS region (dead from the pass's start until the head rewrites all 150,144 logits after the last MoE block, behind
// a grid barrier) before the router counter; a routed gate/up piece copies only its slot's routed rows (<= XR per
// sub-batch) and walks the union once; the shared team keeps its per-group staging (it starts before routing).
// Every row's router scores, top-6, union, gate/up and DOWN arithmetic and stores are the base block's.
#define XR (XSV / 2048u)
inline device ushort* PUBX(thread const Ctx& c, uint m) { return LGv(c, m); }
inline device float* PUBS(thread const Ctx& c, uint m) { return (device float*)(LGv(c, m) + 2048u); }
// row m's vprep values from its raw staging slot into PUBX / PUBS (vprep's arithmetic: load_x16's xv and sum)
inline void vpublish(thread Ctx& c, uint m, uint slot) {
  if (c.tid < 128u) {
    float xv[16];
    const float sum = load_x16(c.xs + slot * 2048u + c.tid * 16u, 0u, xv);
    device uint4* q = (device uint4*)(PUBX(c, m) + c.tid * 16u);
    q[0] = uint4(pack2(xv[0], xv[1]), pack2(xv[2], xv[3]), pack2(xv[4], xv[5]), pack2(xv[6], xv[7]));
    q[1] = uint4(pack2(xv[8], xv[9]), pack2(xv[10], xv[11]), pack2(xv[12], xv[13]), pack2(xv[14], xv[15]));
    PUBS(c, m)[c.tid] = sum;
  }
}
// S1s: k <= XR routed rows (row ids in nibbles of pk): published x -> xs slots 0 .. k - 1, sums -> xsum, ONE row at a
// time (one uint4 + one float live per thread; a loads-first copy kept XR uint4 + XSJ floats live)
inline void vs1_copy(thread Ctx& c, thread VX& vx, uint pk, uint k) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = 0; j < k; ++j) {
    const uint m = (pk >> (4u * j)) & 0xfu;
    const uint4 v = ((const device uint4*)PUBX(c, m))[c.tid];
    const float s = c.tid < 128u ? PUBS(c, m)[c.tid] : 0.0f;
    ((threadgroup uint4*)c.xs)[j * 256u + c.tid] = v;
    if (c.tid < 128u) vx.xsum[j * 128u + c.tid] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
// gate/up of union slot s, member q, for every row routed to s (sub-batches of <= XR rows; units as the base piece)
inline void vgu_piece_s1(thread Ctx& c, thread VX& vx, uint l, uint s, uint q) {
  const uint rk = vx.urk[s];
  uint rem = 0u;
  for (uint m = 0; m < MR; ++m) if (vrank(rk, m) != 0xfu) rem |= 1u << m;
  const device uchar* rec = vebase(c, l) + size_t(vx.uid[s]) * E_STRIDE;
  const QM mg = qm_expert(rec, EG_W, EG_S, EG_B), mu = qm_expert(rec, EU_W, EU_S, EU_B);
  while (rem != 0u) {
    uint pk = 0u, k = 0u;
    for (; rem != 0u && k < XR; ++k) { pk |= ctz(rem) << (4u * k); rem &= rem - 1u; }
    vs1_copy(c, vx, pk, k);
    for (uint i = q + MQ * c.sg; i < 256u; i += MQ * 8u) {
      const uint r0 = i * 2u;
      for (uint b = 0; b < k; b += RGE) {
        const uint nb = min(RGE, k - b);
        float g[2 * RGE], up[2 * RGE];
        q4_gu_p<2048, RGE>(g, up, nb, 0xffu, mg, mu, r0, c.xs + b * 2048u, 2048u, vx.xsum + b * 128u, 128u, c.lane);
        if (c.lane == 0) {
          for (uint kk = 0; kk < RGE; ++kk) {
            if (kk < nb) {
              const uint m = (pk >> (4u * (b + kk))) & 0xfu;
              device uint* act = (device uint*)ACTv(c, m, l);
              act[(vrank(rk, m) * 512u + r0) / 2] = pack2(swiglu(bfr(g[2 * kk]), bfr(up[2 * kk])),
                                                         swiglu(bfr(g[2 * kk + 1]), bfr(up[2 * kk + 1])));
            }
          }
        }
      }
    }
  }
}
// the S1s block (the base block's phase-mask semantics)
inline bool vmoe_s1(thread Ctx& c, thread VX& vx, uint l, bool dr, bool dg, bool dd) {
  const uint NRT = c.G - MQS;
  if (dr && c.w >= MQS) {                            // router threadgroups: stage, route, publish (row m on j = m % NRT)
    const uint j = c.w - MQS;
    for (uint g0 = 0; g0 < MR; g0 += RG2K) {
      const uint n = min(RG2K, MR - g0);
      vmoe_stage_x(c, vx, l, g0, n, false);
      vrouter(c, l, g0, n, c.sg * NRT + j, NRT * 8u);
      for (uint i = 0; i < n; ++i) if ((g0 + i) % NRT == j) vpublish(c, g0 + i, i);
    }
    team_signal(c, CTR(c, l, CT_ROUTER));
  }
  if (dg) {
    if (c.w < MQS) {                                 // the shared team: its own x per staging group
      for (uint g0 = 0; g0 < MR; g0 += RG2K) {
        const uint n = min(RG2K, MR - g0);
        vmoe_stage_x(c, vx, l, g0, n, true);
        vgu_piece_shared(c, vx, l, c.w, g0, n);
      }
      team_signal(c, CTR(c, l, CT_SHARED));
    }
  }
  if (dr && !team_wait(c, CTR(c, l, CT_ROUTER), NRT)) return false;
  vtop6(c, vx, l);
  if (dg) {
    const uint np = MQS + vx.nu[0] * MQ;
    for (uint p = c.w < MQS ? c.w + c.G : c.w; p < np; p += c.G) {
      const uint r = p - MQS;
      vgu_piece_s1(c, vx, l, r / MQ, r % MQ);
      team_signal(c, CTR(c, l, r / MQ));
    }
  }
  if (dd) {
    const uint np = MQS + vx.nu[0] * MQ;
    for (uint p = c.G - 1u - c.w; p < np; p += c.G) {
      const bool sh = p < MQS;
      const uint r = sh ? 0u : p - MQS;
      if (dg && !team_wait(c, CTR(c, l, sh ? CT_SHARED : r / MQ), sh ? MQS : MQ)) return false;
      if (sh) vdn_piece_shared(c, vx, l, p);
      else vdn_piece_routed(c, vx, l, r / MQ, r % MQ);
    }
  }
  return true;
}
