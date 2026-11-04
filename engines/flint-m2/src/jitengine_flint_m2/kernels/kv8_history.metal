
// ------------------------------------------------------------------ kv8 history (full layers only)
inline device uchar* kv8_codes(thread const Ctx& c, uint l, bool isv, uint kvh) {
  return (device uchar*)(isv ? c.vf0 : c.kf0) + size_t(uint(FIDX[l]) * 4u + kvh) * c.cap * 128u;
}
inline device uint* kv8_sb(thread const Ctx& c, uint l, bool isv, uint kvh) {
  return (device uint*)(isv ? c.vf1 : c.kf1) + size_t(uint(FIDX[l]) * 4u + kvh) * c.cap * 2u;
}
// MLX affine_quantize (bits 8, group 64) of group g of the 128-value row x (threadgroup fp32 holding bf16 values),
// run by ONE whole simdgroup (lane = 2 values); writes the group's 64 code bytes and its (scale | bias << 16) word
inline void kv8_quant(device uchar* codes, device uint* sb, const threadgroup float* x, uint g, uint lane) {
  constexpr float eps = 1e-7f;
  constexpr float n_bins = 255.0f;
  const float w0 = x[g * 64u + 2u * lane], w1 = x[g * 64u + 2u * lane + 1u];
  float w_min = 3.38953139e38f, w_max = 0.0f;
  w_min = min(w_min, w0); w_max = max(w_max, w0);
  w_min = min(w_min, w1); w_max = max(w_max, w1);
  w_min = simd_min(w_min);
  w_max = simd_max(w_max);
  float scale = max((w_max - w_min) / n_bins, eps);
  const bool side = abs(w_min) > abs(w_max);
  scale = side ? scale : -scale;
  const float edge = side ? w_min : w_max;
  const float q0 = round(edge / scale);
  const bool at_zero = q0 == 0.0f;
  scale = at_zero ? scale : edge / q0;
  const float bias = at_zero ? 0.0f : edge;
  const uint8_t c0 = uint8_t(min(round((w0 - bias) / scale), n_bins));
  const uint8_t c1 = uint8_t(min(round((w1 - bias) / scale), n_bins));
  ((device ushort*)codes)[g * 32u + lane] = ushort(uint(c0) | (uint(c1) << 8));
  if (lane == 0u) sb[g] = uint(f2bf(scale)) | (uint(f2bf(bias)) << 16);
}
// quantise-on-write of full layer l's K (isv false) or V head kvh at history row `slot` from the projected head x
inline void kv8_store(thread const Ctx& c, uint l, bool isv, uint kvh, uint slot, const threadgroup float* x, uint g) {
  kv8_quant(kv8_codes(c, l, isv, kvh) + size_t(slot) * 128u, kv8_sb(c, l, isv, kvh) + size_t(slot) * 2u, x, g, c.lane);
}

