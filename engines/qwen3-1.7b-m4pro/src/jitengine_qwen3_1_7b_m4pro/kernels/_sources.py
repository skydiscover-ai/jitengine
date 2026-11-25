"""Metal sources of the exact small-M 4-bit matmul, byte for byte from the validated research kernel."""

_HEADER = "#include <metal_simdgroup_matrix>\nusing namespace metal;\n"

_SRC_EXACT = """
    uint gid = thread_position_in_grid.x;
    uint part = thread_position_in_grid.y;
    uint sg = gid / 32;
    uint P = (uint)gp[sg];
    if (part >= P) return;
    uint KP = K / P;
    uint lane = gid % 32;
    uint qid = lane / 4;
    uint fm = (qid & 4) + ((lane / 2) % 4);
    uint c = (qid & 2) + (lane % 2);
    uint fn = 2 * c;
    uint nA = sg * 16 + fm;
    uint nB = nA + 8;
    uint kbase = part * KP;
    const device uint32_t* wA = (const device uint32_t*)w + (size_t)nA * (K / 8) + kbase / 8;
    const device uint32_t* wB = (const device uint32_t*)w + (size_t)nB * (K / 8) + kbase / 8;
    const device T* sA = (const device T*)scales + (size_t)nA * (K / 64) + kbase / 64;
    const device T* bA = (const device T*)biases + (size_t)nA * (K / 64) + kbase / 64;
    const device T* sB = (const device T*)scales + (size_t)nB * (K / 64) + kbase / 64;
    const device T* bB = (const device T*)biases + (size_t)nB * (K / 64) + kbase / 64;
    const device T* xp = (const device T*)x + kbase;
    simdgroup_matrix<float, 8, 8> a00 = simdgroup_matrix<float, 8, 8>(0), a01 = simdgroup_matrix<float, 8, 8>(0);
    simdgroup_matrix<float, 8, 8> a10 = simdgroup_matrix<float, 8, 8>(0), a11 = simdgroup_matrix<float, 8, 8>(0);
    bool v0a = fn < M, v0b = fn + 1 < M, v1a = fn + 8 < M, v1b = fn + 9 < M;
    for (uint kb = 0; kb < KP; kb += 64) {
        float s0 = float(sA[kb / 64]), b0 = float(bA[kb / 64]);
        float s1 = float(sB[kb / 64]), b1 = float(bB[kb / 64]);
        uint4 wa0 = ((const device uint4*)(wA + kb / 8))[0], wa1 = ((const device uint4*)(wA + kb / 8))[1];
        uint4 wb0 = ((const device uint4*)(wB + kb / 8))[0], wb1 = ((const device uint4*)(wB + kb / 8))[1];
        uint32_t wAr[8] = {wa0.x, wa0.y, wa0.z, wa0.w, wa1.x, wa1.y, wa1.z, wa1.w};
        uint32_t wBr[8] = {wb0.x, wb0.y, wb0.z, wb0.w, wb1.x, wb1.y, wb1.z, wb1.w};
        #pragma unroll
        for (uint t = 0; t < 8; t++) {
            uint32_t wa = wAr[t] >> (8 * c);
            uint32_t wb = wBr[t] >> (8 * c);
            simdgroup_matrix<T, 8, 8> A0, A1;
            A0.thread_elements()[0] = T(s0 * float(wa & 0xF) + b0);
            A0.thread_elements()[1] = T(s0 * float((wa >> 4) & 0xF) + b0);
            A1.thread_elements()[0] = T(s1 * float(wb & 0xF) + b1);
            A1.thread_elements()[1] = T(s1 * float((wb >> 4) & 0xF) + b1);
            uint k = kb + 8 * t + fm;
            simdgroup_matrix<T, 8, 8> B0, B1;
            B0.thread_elements()[0] = v0a ? xp[(size_t)fn * K + k] : T(0);
            B0.thread_elements()[1] = v0b ? xp[(size_t)(fn + 1) * K + k] : T(0);
            simdgroup_multiply_accumulate(a00, A0, B0, a00);
            simdgroup_multiply_accumulate(a10, A1, B0, a10);
            if (M > 8) {
                B1.thread_elements()[0] = v1a ? xp[(size_t)(fn + 8) * K + k] : T(0);
                B1.thread_elements()[1] = v1b ? xp[(size_t)(fn + 9) * K + k] : T(0);
                simdgroup_multiply_accumulate(a01, A0, B1, a01);
                simdgroup_multiply_accumulate(a11, A1, B1, a11);
            }
        }
    }
    device OT* op = (device OT*)out + (size_t)part * M * N;
    if (v0a) { op[(size_t)fn * N + nA] = OT(a00.thread_elements()[0]); op[(size_t)fn * N + nB] = OT(a10.thread_elements()[0]); }
    if (v0b) { op[(size_t)(fn + 1) * N + nA] = OT(a00.thread_elements()[1]); op[(size_t)(fn + 1) * N + nB] = OT(a10.thread_elements()[1]); }
    if (M > 8) {
        if (v1a) { op[(size_t)(fn + 8) * N + nA] = OT(a01.thread_elements()[0]); op[(size_t)(fn + 8) * N + nB] = OT(a11.thread_elements()[0]); }
        if (v1b) { op[(size_t)(fn + 9) * N + nA] = OT(a01.thread_elements()[1]); op[(size_t)(fn + 9) * N + nB] = OT(a11.thread_elements()[1]); }
    }
"""

_SRC_REDUCE = """
    uint i = thread_position_in_grid.x;
    const device float* pp = (const device float*)parts;
    uint P = (uint)gp[(i % N) / 16];
    float acc = pp[i];
    for (uint p = 1; p < P; p++) acc += pp[(size_t)p * M * N + i];
    ((device T*)out)[i] = T(acc);
"""
