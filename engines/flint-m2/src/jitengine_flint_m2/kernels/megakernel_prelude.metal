
#include <metal_simdgroup_matrix>
using namespace metal;

#define CHAIN __CHAIN__
#define DIAG __DIAG__
#define PASS_M1 __PASS_M1__   // 1: the M = 1 pass; 0: a verify pass
#define NL 49u
#define VOCAB __VOCAB__u
#define RING __RING__u
#define WINDOW __WINDOW__u
#define EPS __EPS__f
#define NF __NF__u
#define DG __DG__u
constant constexpr float kLog2e = 1.4426950408889634f;
constant constexpr float QSCALE = 0.08838834764831845f * 1.4426950408889634f;

// scratch byte offsets
#define S_HS __OFF_HS__u
#define S_Q __OFF_Q__u
#define S_ATTN __OFF_ATTN__u
#define S_O __OFF_O__u
#define S_SC __OFF_SC__u
#define S_ACT __OFF_ACT__u
#define S_PART __OFF_PART__u
#define S_LOGITS __OFF_LOGITS__u
#define S_SELP __OFF_SELP__u
#define S_Y __OFF_Y__u          // [7][2048] bf16 expert outputs (rank order, 6 = shared)
#define S_CTR __S_CTR__u        // MoE team counters [49][64] on 64-byte lines (after the row regions)

// descriptor fields
#define F_LN1 0u
#define F_LN2 1u
#define F_QW 2u
#define F_KW 5u
#define F_VW 8u
#define F_OW 11u
#define F_QN 14u
#define F_KN 15u
#define F_SINK 16u
#define F_RW 17u
#define F_SGW 18u
#define F_SUW 21u
#define F_SDW 24u
#define F_DGW 27u
#define F_DUW 30u
#define F_DDW 33u
#define F_EBANK 36u
#define F_EBASE 37u
#define G_NORM 0u
#define G_HW 1u
#define G_EW 4u

// expert record
#define E_STRIDE __E_STRIDE__ul
#define EG_W __EG_W__u
#define EG_S __EG_S__u
#define EG_B __EG_B__u
#define EU_W __EU_W__u
#define EU_S __EU_S__u
#define EU_B __EU_B__u
#define ED_W __ED_W__u
#define ED_S __ED_S__u
#define ED_B __ED_B__u

// result words
#define R_TOKEN 0u
#define R_FINITE 1u
#define R_ERRC 2u
#define R_ARGMAX 3u
#define R_LSE 4u
#define R_CHOSEN_LP 5u
#define R_FORCED_LP 6u
#define R_DONE 7u
#define R_TOPI 8u
#define R_TOPV 28u
#define R_ARRIVE __R_ARRIVE__u
#define R_ERR __R_ERR__u
#define R_ATTQ __R_ATTQ__u
#define R_QFLAG __R_QFLAG__u    // 40 per-head 'published layer + 1' words
#define R_FCNT __R_FCNT__u      // 4 per-kv-head attention task counters

// parameters
#define P_POS 0u
#define P_TOKMODE 1u
#define P_TOKEN 2u
#define P_FORCED 3u
#define P_SELMODE 4u
#define P_TOPK 5u
#define P_TOPP 6u
#define P_INVT 7u
#define P_U 8u
#define P_POISON 9u
#define P_MASK 10u
#define P_SPIN 11u
#define P_SF 12u
#define P_SS 13u
#define P_CAP 14u
#define P_PF 15u
#define P_PS 16u
#define P_FSUB 17u
#define P_FASG 18u
#define P_FFLAGS 19u
#define P_DIAG 20u
#define P_6T 21u               // layout 6d: the span switch T of pos_grid (plan.SIXD_T; 0 = strided / layout 2)
#define P_STALL 22u            // fault seam kernel_stall: threadgroup G - 1 returns before any work (pass_entry)
#define P_PROBE 23u            // residency probe: laps of the flag ring (probe_ring), instead of a pass
// in-kernel error words (plan.describe_err); R_ERRC is set beside every one of them (the next pass's entry check)
#define E_TEAM 0x40000000u     // | barrier generation
#define E_FLOW_HEADS 0x20000000u   // | layer + 1
#define E_FLOW_TASKS 0x10000000u   // | layer + 1
#define E_PREV 0x08000000u     // not run: the previous pass had failed
#define E_RING 0x04000000u     // | threadgroup << 8 | lap
#define FL_SGFIRST 1u
#define FL_DYNAMIC 4u
#define P_COS 24u
#define P_SIN 88u
#define PROW 132u
#define KB __KB__u
#define HG __HG__u
#define SABL __SABL__u          // full-layer stream timing ablation (bench only; 0 in production): 1 = K/V rows
                                // taken from 16 L1-resident rows (compute + load issue, no DRAM), 2 = loads only

// phases
#define PH_QKV 0u
#define PH_ATT 1u
#define PH_MERGE 2u
#define PH_O 3u
#define PH_RSG 4u
#define PH_EGU 5u
#define PH_DOWN 6u
#define PH_DGU 7u
#define PH_DDOWN 8u
#define PH_HEAD 9u
#define PH_SELECT 10u

constant short FIDX[49] = {__FIDX__};
constant short SIDX[49] = {__SIDX__};
constant uchar DENSE[49] = {__DENSE__};

inline float bf2f(ushort u) { return as_type<float>(uint(u) << 16); }
inline ushort f2bf(float x) { return as_type<ushort>(bfloat(x)); }
inline float bfr(float x) { return float(bfloat(x)); }
inline float lo16(uint w) { return as_type<float>(w << 16); }
inline float hi16(uint w) { return as_type<float>(w & 0xffff0000u); }
inline uint pack2(float a, float b) { return uint(f2bf(a)) | (uint(f2bf(b)) << 16); }
inline float sigmoid_m(float x) { const float t = 1.0f / (1.0f + exp(fabs(x))); return x < 0.0f ? t : 1.0f - t; }
inline float swiglu(float g, float u) { return bfr(bfr(g * bfr(sigmoid_m(g))) * u); }

struct Ctx {
  const device uchar* b0; const device uchar* b1; const device uchar* b2;
  const device ulong* desc;
  device ushort* kf0; device ushort* kf1; device ushort* vf0; device ushort* vf1;
  device ushort* kr; device ushort* vr;
  device uchar* st;
  const device uint* prm;
  device uint* res;
  uint tid, sg, lane, w, G;
  uint pos, cap, gen, spin_max;
  int tk_layer;
  uint fcum;                   // cumulative attention tasks per kv head (flag dataflow)
  threadgroup ushort* xs;      // 6144 bf16: the GEMV input vector
  threadgroup float* red;      // 64 floats
  threadgroup float* hb;       // 128 floats: one projected head
  threadgroup uint* tku;       // 16: top-6 expert ids
  threadgroup float* tkw;      // 8: top-6 weights (bf16-rounded)
  threadgroup uint* flag;      // 4
  threadgroup const uint* dt;  // M = 1: the descriptor table's threadgroup copy (desc_off)
  threadgroup const uint* pk;  // M = 1: the pass constants prm[P_MASK .. P_6T], copied once per dispatch (PK)
  threadgroup ushort* hm;      // M = 1: bf(h + o) of the last MoE block run by this threadgroup, this thread's 8 ...
  int hm_layer;                // ... elements at tid * 8 (tag: that block's layer; -1 none)
#if DIAG
  const device uint* dnext;    // [NL + 1] next layer of the same kind (bench_mk's diagnostic table, in `dep`)
#endif
};

inline uint gsg(thread const Ctx& c) { return c.sg * c.G + c.w; }   // work dealt threadgroup-first
// record an in-kernel failure: the first error word wins; R_ERRC (word 2 of the result, row 0's in a verify result) marks
// the pass failed for the next pass's entry check (pass_entry), whichever result layout that pass reads
inline void set_err(thread const Ctx& c, uint code) {
  device atomic_uint* errw = (device atomic_uint*)(c.res + R_ERR);
  if (atomic_load_explicit(errw, memory_order_relaxed) == 0u) atomic_store_explicit(errw, code, memory_order_relaxed);
  atomic_store_explicit((device atomic_uint*)(c.res + R_ERRC), 1u, memory_order_relaxed);
}
inline uint nsg(thread const Ctx& c) { return c.G * 8u; }
// descriptor word i: a byte offset (desc_off) or the raw word (desc_word, the expert bank). The M = 1 pass reads its
// threadgroup copy (offset >> 4), a verify pass the device table.
#if PASS_M1
// PK(c, i): pass constant i in P_MASK .. P_6T. The M = 1 pass reads its threadgroup copy
// (prm lines are cold after a fence or a weight stream), a verify pass the device words.
static_assert(P_6T - P_MASK == 11u, "PK copies prm[P_MASK .. P_MASK + 11]");
#define PK(c, i) ((c).pk[(i) - P_MASK])
inline ulong desc_off(thread const Ctx& c, uint i) { return ulong(c.dt[i]) << 4; }
inline ulong desc_word(thread const Ctx& c, uint i) { return ulong(c.dt[i]); }
#else
#define PK(c, i) ((c).prm[i])
inline ulong desc_off(thread const Ctx& c, uint i) { return c.desc[i]; }
inline ulong desc_word(thread const Ctx& c, uint i) { return c.desc[i]; }
#endif
inline const device uchar* LF(thread const Ctx& c, uint l, uint f) { return c.b0 + desc_off(c, l * NF + f); }
inline const device ushort* LH(thread const Ctx& c, uint l, uint f) { return (const device ushort*)(c.b0 + desc_off(c, l * NF + f)); }
inline const device uchar* GF(thread const Ctx& c, uint f) { return c.b0 + desc_off(c, DG + f); }
inline device ushort* HS(thread const Ctx& c, uint l) { return (device ushort*)(c.st + S_HS) + l * 2048u; }
inline device ushort* QB(thread const Ctx& c, uint l) { return (device ushort*)(c.st + S_Q) + l * 4096u; }
inline device ushort* AB(thread const Ctx& c, uint l) { return (device ushort*)(c.st + S_ATTN) + l * 4096u; }
inline device ushort* OB(thread const Ctx& c, uint l) { return (device ushort*)(c.st + S_O) + l * 2048u; }
inline device float* SCb(thread const Ctx& c, uint l) { return (device float*)(c.st + S_SC) + l * 256u; }
inline device ushort* ACT(thread const Ctx& c, uint l) { return (device ushort*)(c.st + S_ACT) + l * 6144u; }
inline device float* PART(thread const Ctx& c) { return (device float*)(c.st + S_PART); }
inline device ushort* LG(thread const Ctx& c) { return (device ushort*)(c.st + S_LOGITS); }
inline device uint* SELP(thread const Ctx& c) { return (device uint*)(c.st + S_SELP); }

// ------------------------------------------------------------------ int4 g64 GEMV (2 rows per unit)
// x (threadgroup, bf16 bits) at offset xo..xo+15 -> xv pre-scaled for masked nibbles; returns sum(x).
inline float load_x16(const threadgroup ushort* x, uint xo, thread float* xv) {
  const threadgroup uint4* p = (const threadgroup uint4*)(x + xo);
  const uint4 a = p[0], b = p[1];
  const uint wv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  float sum = 0.0f;
  for (int i = 0; i < 8; ++i) {
    const float l = lo16(wv[i]), h = hi16(wv[i]);
    sum += l + h;
    xv[2 * i] = (i & 1) ? l * (1.0f / 256.0f) : l;
    xv[2 * i + 1] = (i & 1) ? h * (1.0f / 4096.0f) : h * (1.0f / 16.0f);
  }
  return sum;
}
inline float qdot16(uint2 wq, thread const float* xv) {
  const uint ws[4] = {wq.x & 0xffffu, wq.x >> 16, wq.y & 0xffffu, wq.y >> 16};
  float a = 0.0f;
  for (int j = 0; j < 4; ++j) {
    a += xv[4 * j] * float(ws[j] & 0x000fu) + xv[4 * j + 1] * float(ws[j] & 0x00f0u) +
         xv[4 * j + 2] * float(ws[j] & 0x0f00u) + xv[4 * j + 3] * float(ws[j] & 0xf000u);
  }
  return a;
}
// one 16-input block of row `row` (lane-local partial): s * dot + b * sum(x)
template <int K>
inline float q4_blk(const device uchar* W, const device ushort* S, const device ushort* B, uint row, uint xo,
                    thread const float* xv, float xsum) {
  const uint2 wq = *(const device uint2*)(W + size_t(row) * (K / 2) + xo / 2);
  const size_t gi = size_t(row) * (K / 64) + xo / 64;
  return bf2f(S[gi]) * qdot16(wq, xv) + bf2f(B[gi]) * xsum;
}
// rows r0, r0+1 of one matrix; fixed order: lane-sequential blocks, then simd_sum
template <int K>
inline void q4_unit(thread float* acc, const device uchar* W, const device ushort* S, const device ushort* B,
                    uint r0, const threadgroup ushort* x, uint lane) {
  float a0 = 0.0f, a1 = 0.0f;
  for (int blk = 0; blk < K / 512; ++blk) {
    const uint xo = blk * 512 + lane * 16;
    float xv[16];
    const float xsum = load_x16(x, xo, xv);
    a0 += q4_blk<K>(W, S, B, r0, xo, xv, xsum);
    a1 += q4_blk<K>(W, S, B, r0 + 1, xo, xv, xsum);
  }
  acc[0] = simd_sum(a0);
  acc[1] = simd_sum(a1);
}
// one row (1-row units: no 2-row tail at G = 40); the same per-row arithmetic and order as q4_unit
template <int K>
inline float q4_row(const device uchar* W, const device ushort* S, const device ushort* B, uint row,
                    const threadgroup ushort* x, uint lane) {
  float a0 = 0.0f;
  for (int blk = 0; blk < K / 512; ++blk) {
    const uint xo = blk * 512 + lane * 16;
    float xv[16];
    const float xsum = load_x16(x, xo, xv);
    a0 += q4_blk<K>(W, S, B, row, xo, xv, xsum);
  }
  return simd_sum(a0);
}
// rows r0, r0+1 of two matrices sharing the input (gate | up)
struct QM { const device uchar* w; const device ushort* s; const device ushort* b; };
template <int K>
inline void q4_gu(thread float* g, thread float* u, QM mg, QM mu, uint r0, const threadgroup ushort* x, uint lane) {
  float g0 = 0.0f, g1 = 0.0f, u0 = 0.0f, u1 = 0.0f;
  for (int blk = 0; blk < K / 512; ++blk) {
    const uint xo = blk * 512 + lane * 16;
    float xv[16];
    const float xsum = load_x16(x, xo, xv);
    g0 += q4_blk<K>(mg.w, mg.s, mg.b, r0, xo, xv, xsum);
    g1 += q4_blk<K>(mg.w, mg.s, mg.b, r0 + 1, xo, xv, xsum);
    u0 += q4_blk<K>(mu.w, mu.s, mu.b, r0, xo, xv, xsum);
    u1 += q4_blk<K>(mu.w, mu.s, mu.b, r0 + 1, xo, xv, xsum);
  }
  g[0] = simd_sum(g0); g[1] = simd_sum(g1); u[0] = simd_sum(u0); u[1] = simd_sum(u1);
}
// a quantized matrix whose weight / scales / biases sit at consecutive descriptor fields f, f+1, f+2 of bank 0
inline QM qm_layer(thread const Ctx& c, uint l, uint f) {
  return QM{LF(c, l, f), LH(c, l, f + 1), LH(c, l, f + 2)};
}
inline QM qm_expert(const device uchar* rec, uint wo, uint so, uint bo) {
  return QM{rec + wo, (const device ushort*)(rec + so), (const device ushort*)(rec + bo)};
}
// bf16 x bf16 rows r0, r0+1 (router), K = 2048
inline void bf_unit(thread float* acc, const device ushort* W, uint r0, const threadgroup ushort* x, uint lane) {
  float a0 = 0.0f, a1 = 0.0f;
  for (int blk = 0; blk < 4; ++blk) {
    const uint xo = blk * 512 + lane * 16;
    const threadgroup uint4* px = (const threadgroup uint4*)(x + xo);
    const uint4 xa = px[0], xb = px[1];
    const uint xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
    for (int r = 0; r < 2; ++r) {
      const device uint4* wp = (const device uint4*)(W + size_t(r0 + r) * 2048 + xo);
      const uint4 wa = wp[0], wb = wp[1];
      const uint ww[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
      float s = 0.0f;
      for (int i = 0; i < 8; ++i) s += lo16(xw[i]) * lo16(ww[i]) + hi16(xw[i]) * hi16(ww[i]);
      if (r == 0) a0 += s; else a1 += s;
    }
  }
  acc[0] = simd_sum(a0);
  acc[1] = simd_sum(a1);
}
