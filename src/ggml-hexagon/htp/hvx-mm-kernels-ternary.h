// GluRun: ported from the core llama.cpp tree's patch 0007 (engines/llamacpp/patches/0007-hexagon-q2_0-q1_0.patch). Q2_0 (ternary, 2-bit) and Q1_0 (1-bit) weights on the Hexagon NPU, HVX mat-vec / small-batch kernels.
// Included by src/ggml-hexagon/htp/matmul-ops.c after hvx-mm-kernels-flat.h (engines/sd/patches/ggml-0004).
//
// Native tiles (32 rows x 32 k, the same tiling every repacked type uses, so the DMA / prefetch / job code is shared):
//
//   Q2_0 tile, HTP_MM_WEIGHT_TILE_SIZE_Q2_0 = 320 bytes (aligned slot 384):
//     bytes 0..255   codes: vector j (j = 0,1), lane r (0..31), byte b (0..3), bit pair p (0..3) holds the 2-bit code of
//                    k = 16*j + 4*p + b, i.e. byte address 128*j + 4*r + b, bits 2p..2p+1. ggml's code c is worth (c-1)*d.
//     bytes 256..319 fp16 scale per row (the ggml block scale; a 64-wide Q2_0 block covers two k-tiles, same scale twice)
//   Q1_0 tile, HTP_MM_WEIGHT_TILE_SIZE_Q1_0 = 192 bytes (aligned slot 256):
//     bytes 0..127   bits: lane r, byte b, bit p (0..7) holds the sign bit of k = 4*p + b (address 4*r + b, bit p).
//                    ggml's bit is worth (2*bit-1)*d.
//     bytes 128..191 fp16 scale per row (a 128-wide Q1_0 block covers four k-tiles)
//
// The layout is chosen for Q6_Vw_vrmpyacc_VwVbVb: shifting the code vector right by 2p (Q1_0: by p) and masking gives,
// for every lane r, the 4 consecutive weights k = 4g..4g+3 of row r as bytes, exactly what the Q8_0 tiled activation
// vector v_act[g] (the same 4 activation bytes replicated in every lane) multiplies. One shift, one and, one subtract
// and one vrmpyacc per 4 k, against Q4_0's shuffle/valign unpack. The block scale is applied once per k-tile like Q4_0.

#ifndef HVX_MM_KERNELS_TERNARY_H
#define HVX_MM_KERNELS_TERNARY_H

#define HTP_MM_Q2_0_ALIGNED_TILE 384
#define HTP_MM_Q1_0_ALIGNED_TILE 256

// --- 2-bit unpack + accumulate ---

// sum over the 32 k of one tile, one activation column: lane r = row r
static inline HVX_Vector accum_q2_0_32x1(const HVX_Vector * restrict vptr, const HVX_Vector * restrict v_act) {
    const HVX_Vector mask_03 = Q6_Vb_vsplat_R(0x03);
    const HVX_Vector one     = Q6_Vb_vsplat_R(1);
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();

    #pragma unroll
    for (int j = 0; j < 2; j++) {
        const HVX_Vector v = vptr[j];
        HVX_Vector w0 = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(v, mask_03), one);
        HVX_Vector w1 = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v, 2), mask_03), one);
        HVX_Vector w2 = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v, 4), mask_03), one);
        HVX_Vector w3 = Q6_Vb_vsub_VbVb(Q6_Vub_vlsr_VubR(v, 6), one);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, w0, v_act[4 * j + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, w1, v_act[4 * j + 1]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, w2, v_act[4 * j + 2]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, w3, v_act[4 * j + 3]);
    }
    return Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

// two activation columns share the unpacked weights
static inline HVX_VectorPair accum_q2_0_32x2(const HVX_Vector * restrict vptr, const HVX_Vector * restrict v_act0, const HVX_Vector * restrict v_act1) {
    const HVX_Vector mask_03 = Q6_Vb_vsplat_R(0x03);
    const HVX_Vector one     = Q6_Vb_vsplat_R(1);
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();

    #pragma unroll
    for (int j = 0; j < 2; j++) {
        const HVX_Vector v = vptr[j];
        HVX_Vector w0 = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(v, mask_03), one);
        HVX_Vector w1 = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v, 2), mask_03), one);
        HVX_Vector w2 = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v, 4), mask_03), one);
        HVX_Vector w3 = Q6_Vb_vsub_VbVb(Q6_Vub_vlsr_VubR(v, 6), one);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, w0, v_act0[4 * j + 0]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, w1, v_act0[4 * j + 1]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, w2, v_act0[4 * j + 2]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, w3, v_act0[4 * j + 3]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, w0, v_act1[4 * j + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, w1, v_act1[4 * j + 1]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, w2, v_act1[4 * j + 2]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, w3, v_act1[4 * j + 3]);
    }
    return Q6_W_vcombine_VV(v_sum1, v_sum0);
}

// --- 1-bit unpack + accumulate: bit -> 2*bit-1 ---

static inline HVX_Vector accum_q1_0_32x1(const HVX_Vector * restrict vptr, const HVX_Vector * restrict v_act) {
    const HVX_Vector mask_01 = Q6_Vb_vsplat_R(0x01);
    const HVX_Vector one     = Q6_Vb_vsplat_R(1);
    const HVX_Vector v = vptr[0];
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();

    #pragma unroll
    for (int g = 0; g < 8; g += 2) {
        HVX_Vector b0 = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v, g), mask_01);
        HVX_Vector b1 = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v, g + 1), mask_01);
        HVX_Vector w0 = Q6_Vb_vsub_VbVb(Q6_Vb_vadd_VbVb(b0, b0), one);
        HVX_Vector w1 = Q6_Vb_vsub_VbVb(Q6_Vb_vadd_VbVb(b1, b1), one);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, w0, v_act[g + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, w1, v_act[g + 1]);
    }
    return Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

static inline HVX_VectorPair accum_q1_0_32x2(const HVX_Vector * restrict vptr, const HVX_Vector * restrict v_act0, const HVX_Vector * restrict v_act1) {
    const HVX_Vector mask_01 = Q6_Vb_vsplat_R(0x01);
    const HVX_Vector one     = Q6_Vb_vsplat_R(1);
    const HVX_Vector v = vptr[0];
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();

    #pragma unroll
    for (int g = 0; g < 8; g++) {
        HVX_Vector b = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v, g), mask_01);
        HVX_Vector w = Q6_Vb_vsub_VbVb(Q6_Vb_vadd_VbVb(b, b), one);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, w, v_act0[g]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, w, v_act1[g]);
    }
    return Q6_W_vcombine_VV(v_sum1, v_sum0);
}

// --- per k-tile scaling shared by every kernel below: sum * (row scale fp16 * activation scale fp16) ---

static inline HVX_Vector ternary_scale_32(HVX_Vector v_sum_i32, HVX_Vector v_scale_w, HVX_Vector v_scale_a) {
    HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
    return hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(v_sum_i32), v_scale_comb);
}

// --- tiled activations (Q8_0 tiled: 8 replicated group vectors + 1 scale vector per 32 k) ---

#define TERNARY_TILED_VEC_DOT_IMPL(NAME, ALIGNED, SCALE_VEC, ACCUM1, ACCUM2)                                                              \
static void tiled_vec_dot_##NAME##_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy,        \
                                        uint32_t valid_rows, const float * restrict sz) {                                                \
    const uint8_t * restrict tile_ptr = vx;                                                                                             \
    const uint8_t * restrict y_q = vy;                                                                                                  \
    HVX_Vector v_sum_float = Q6_V_vzero();                                                                                              \
    const uint32_t n_k_tiles = n / 32;                                                                                                  \
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {                                                                                       \
        const HVX_Vector * restrict vptr  = (const HVX_Vector *) (tile_ptr + kt * ALIGNED);                                             \
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);                                                     \
        HVX_Vector v_sum = ACCUM1(vptr, v_act);                                                                                         \
        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, ternary_scale_32(v_sum, vptr[SCALE_VEC], v_act[8]));                             \
    }                                                                                                                                   \
    if (sz) {                                                                                                                           \
        hvx_vec_store_u(s, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));                                \
    } else {                                                                                                                            \
        hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);                                                                    \
    }                                                                                                                                   \
}                                                                                                                                       \
                                                                                                                                        \
static void tiled_vec_dot_##NAME##_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx,            \
                                        const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows,                       \
                                        const float * restrict sz0, const float * restrict sz1) {                                        \
    const uint8_t * restrict tile_ptr = vx;                                                                                             \
    const uint8_t * restrict y0_q = vy0;                                                                                                \
    const uint8_t * restrict y1_q = vy1;                                                                                                \
    HVX_Vector v_sum_float_c0 = Q6_V_vzero();                                                                                           \
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();                                                                                           \
    const uint32_t n_k_tiles = n / 32;                                                                                                  \
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {                                                                                       \
        const HVX_Vector * restrict vptr   = (const HVX_Vector *) (tile_ptr + kt * ALIGNED);                                            \
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1152);                                                   \
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1152);                                                   \
        HVX_VectorPair v_sums = ACCUM2(vptr, v_act0, v_act1);                                                                           \
        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, ternary_scale_32(Q6_V_lo_W(v_sums), vptr[SCALE_VEC], v_act0[8]));          \
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, ternary_scale_32(Q6_V_hi_W(v_sums), vptr[SCALE_VEC], v_act1[8]));          \
    }                                                                                                                                   \
    if (sz0) {                                                                                                                          \
        hvx_vec_store_u(s0, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));                           \
    } else {                                                                                                                            \
        hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);                                                                \
    }                                                                                                                                   \
    if (sz1) {                                                                                                                          \
        hvx_vec_store_u(s1, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));                           \
    } else {                                                                                                                            \
        hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);                                                                \
    }                                                                                                                                   \
}

TERNARY_TILED_VEC_DOT_IMPL(q2_0, HTP_MM_Q2_0_ALIGNED_TILE, 2, accum_q2_0_32x1, accum_q2_0_32x2)
TERNARY_TILED_VEC_DOT_IMPL(q1_0, HTP_MM_Q1_0_ALIGNED_TILE, 1, accum_q1_0_32x1, accum_q1_0_32x2)

// --- flat activations (Q8_0 flat: 128-byte quant vectors, then fp16 scales per 32 k); the 8 group vectors are built
// with the same vdelta replication the Q4_0 flat kernels use ---

static inline void ternary_flat_act_groups(const uint8_t * restrict y_q, uint32_t kt, HVX_Vector * restrict v_act) {
    static const uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };
    const HVX_Vector v_repl_ctrl = * (const HVX_Vector *) repl;
    const uint32_t block_idx = kt / 4;
    const uint32_t sub_idx   = kt % 4;
    HVX_Vector vx_i8 = * (const HVX_Vector *) (y_q + block_idx * 128);
    HVX_Vector v_act_raw = Q6_V_vror_VR(vx_i8, sub_idx * 32);
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        v_act[g] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 4 * g), v_repl_ctrl);
    }
}

static inline HVX_Vector ternary_flat_act_scale(const __fp16 * restrict y_scales, uint32_t kt) {
    __fp16 scale_a_val = y_scales[kt];
    return hvx_vec_repl_f16(Q6_Vh_vsplat_R(*(const int16_t *) &scale_a_val));
}

#define TERNARY_FLAT_VEC_DOT_IMPL(NAME, ALIGNED, SCALE_VEC, ACCUM1, ACCUM2)                                                               \
static void flat_vec_dot_##NAME##_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy,         \
                                       uint32_t valid_rows, const float * restrict sz) {                                                 \
    const uint8_t * restrict tile_ptr = vx;                                                                                             \
    const uint8_t * restrict y_q = vy;                                                                                                  \
    const uint32_t quants_size = hex_round_up(n, 128);                                                                                  \
    const __fp16 * restrict y_scales = (const __fp16 *) (y_q + quants_size);                                                            \
    HVX_Vector v_sum_float = Q6_V_vzero();                                                                                              \
    const uint32_t n_k_tiles = n / 32;                                                                                                  \
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {                                                                                       \
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * ALIGNED);                                              \
        HVX_Vector v_act[8];                                                                                                            \
        ternary_flat_act_groups(y_q, kt, v_act);                                                                                        \
        HVX_Vector v_sum = ACCUM1(vptr, v_act);                                                                                         \
        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, ternary_scale_32(v_sum, vptr[SCALE_VEC], ternary_flat_act_scale(y_scales, kt))); \
    }                                                                                                                                   \
    if (sz) {                                                                                                                           \
        hvx_vec_store_u(s, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));                                \
    } else {                                                                                                                            \
        hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);                                                                    \
    }                                                                                                                                   \
}                                                                                                                                       \
                                                                                                                                        \
static void flat_vec_dot_##NAME##_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx,             \
                                       const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows,                        \
                                       const float * restrict sz0, const float * restrict sz1) {                                         \
    const uint8_t * restrict tile_ptr = vx;                                                                                             \
    const uint8_t * restrict y0_q = vy0;                                                                                                \
    const uint8_t * restrict y1_q = vy1;                                                                                                \
    const uint32_t quants_size = hex_round_up(n, 128);                                                                                  \
    const __fp16 * restrict y0_scales = (const __fp16 *) (y0_q + quants_size);                                                          \
    const __fp16 * restrict y1_scales = (const __fp16 *) (y1_q + quants_size);                                                          \
    HVX_Vector v_sum_float_c0 = Q6_V_vzero();                                                                                           \
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();                                                                                           \
    const uint32_t n_k_tiles = n / 32;                                                                                                  \
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {                                                                                       \
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * ALIGNED);                                              \
        HVX_Vector v_act0[8];                                                                                                           \
        HVX_Vector v_act1[8];                                                                                                           \
        ternary_flat_act_groups(y0_q, kt, v_act0);                                                                                      \
        ternary_flat_act_groups(y1_q, kt, v_act1);                                                                                      \
        HVX_VectorPair v_sums = ACCUM2(vptr, v_act0, v_act1);                                                                           \
        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, ternary_scale_32(Q6_V_lo_W(v_sums), vptr[SCALE_VEC], ternary_flat_act_scale(y0_scales, kt))); \
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, ternary_scale_32(Q6_V_hi_W(v_sums), vptr[SCALE_VEC], ternary_flat_act_scale(y1_scales, kt))); \
    }                                                                                                                                   \
    if (sz0) {                                                                                                                          \
        hvx_vec_store_u(s0, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));                           \
    } else {                                                                                                                            \
        hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);                                                                \
    }                                                                                                                                   \
    if (sz1) {                                                                                                                          \
        hvx_vec_store_u(s1, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));                           \
    } else {                                                                                                                            \
        hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);                                                                \
    }                                                                                                                                   \
}

TERNARY_FLAT_VEC_DOT_IMPL(q2_0, HTP_MM_Q2_0_ALIGNED_TILE, 2, accum_q2_0_32x1, accum_q2_0_32x2)
TERNARY_FLAT_VEC_DOT_IMPL(q1_0, HTP_MM_Q1_0_ALIGNED_TILE, 1, accum_q1_0_32x1, accum_q1_0_32x2)

#endif // HVX_MM_KERNELS_TERNARY_H
