// GluRun: ported from the core llama.cpp tree's patch 0007 (engines/llamacpp/patches/0007-hexagon-q2_0-q1_0.patch). Q2_0 / Q1_0 native tiles -> the fp16 32x32 HMX weight tile (prompt processing, n > 4 rows).
// Included by src/ggml-hexagon/htp/matmul-ops.c after hmx-mm-kernels-tiled.h (engines/sd/patches/ggml-0004); tile layout in
// hvx-mm-kernels-ternary.h.
//
// The HMX tile the Q4_0 dequant produces is 16 vectors of 64 fp16: vector cp (0..15) holds, for row r, the halves
// (2r, 2r+1) = W[r][2cp], W[r][2cp+1]. From the lane-major group vector g (lane r = bytes k 4g..4g+3 of row r) a
// halfword deal moves the (k 4g, 4g+1) byte pairs of all 32 rows to the low half and the (4g+2, 4g+3) pairs to the
// high half; sign-extending bytes to halves then gives exactly vectors cp = 2g and 2g+1.

#ifndef HMX_MM_KERNELS_TERNARY_H
#define HMX_MM_KERNELS_TERNARY_H

static inline void ternary_store_hmx_group(__fp16 * dst_ptr, int g, HVX_Vector w_i8, HVX_Vector v_scale_dup) {
    HVX_Vector     dealt = Q6_Vh_vdeal_Vh(w_i8);
    HVX_VectorPair pair  = Q6_Wh_vunpack_Vb(dealt);
    HVX_Vector lo = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(pair)), v_scale_dup));
    HVX_Vector hi = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(pair)), v_scale_dup));
    hvx_vmem(dst_ptr + (2 * g + 0) * 64) = lo;
    hvx_vmem(dst_ptr + (2 * g + 1) * 64) = hi;
}

static void dequantize_tiled_weight_to_fp16_task_q2_0(const tiled_dequantize_state_t * state, uint32_t start_tile, uint32_t end_tile) {
    const HVX_Vector mask_03 = Q6_Vb_vsplat_R(0x03);
    const HVX_Vector one     = Q6_Vb_vsplat_R(1);

    for (uint32_t t = start_tile; t < end_tile; t++) {
        const uint8_t * tile_src = state->src + t * state->aligned_tile_size;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_Vector v_sc = hvx_vmem(tile_src + 256);
        HVX_Vector v_scale_dup = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        #pragma unroll
        for (int j = 0; j < 2; j++) {
            const HVX_Vector v = hvx_vmem(tile_src + j * 128);
            ternary_store_hmx_group(dst_ptr, 4 * j + 0, Q6_Vb_vsub_VbVb(Q6_V_vand_VV(v, mask_03), one), v_scale_dup);
            ternary_store_hmx_group(dst_ptr, 4 * j + 1, Q6_Vb_vsub_VbVb(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v, 2), mask_03), one), v_scale_dup);
            ternary_store_hmx_group(dst_ptr, 4 * j + 2, Q6_Vb_vsub_VbVb(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v, 4), mask_03), one), v_scale_dup);
            ternary_store_hmx_group(dst_ptr, 4 * j + 3, Q6_Vb_vsub_VbVb(Q6_Vub_vlsr_VubR(v, 6), one), v_scale_dup);
        }
    }
}

static void dequantize_tiled_weight_to_fp16_task_q1_0(const tiled_dequantize_state_t * state, uint32_t start_tile, uint32_t end_tile) {
    const HVX_Vector mask_01 = Q6_Vb_vsplat_R(0x01);
    const HVX_Vector one     = Q6_Vb_vsplat_R(1);

    for (uint32_t t = start_tile; t < end_tile; t++) {
        const uint8_t * tile_src = state->src + t * state->aligned_tile_size;
        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;

        HVX_Vector v_sc = hvx_vmem(tile_src + 128);
        HVX_Vector v_scale_dup = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc, v_sc, -2));

        const HVX_Vector v = hvx_vmem(tile_src);
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            HVX_Vector b = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v, g), mask_01);
            ternary_store_hmx_group(dst_ptr, g, Q6_Vb_vsub_VbVb(Q6_Vb_vadd_VbVb(b, b), one), v_scale_dup);
        }
    }
}

#endif // HMX_MM_KERNELS_TERNARY_H
