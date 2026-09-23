// GluRun SDK - the DSP tile layout of the ternary weight types Q2_0 / Q1_0, shared by the host side
// (ggml-hexagon.cpp, which repacks weights into it) and the host test that proves the layout
// (tests/images/ternary_tile_test.cpp). The DSP side reads the same bytes in
// htp/hvx-mm-kernels-ternary.h (HVX mat-vec) and htp/hmx-mm-kernels-ternary.h (HMX dequant);
// those two files and this one must agree, and the test checks exactly that.
//
// Added by engines/sd/patches/ggml-0004, ported from the core llama.cpp tree's patch 0007.
// SPDX-License-Identifier: Apache-2.0
#ifndef GGML_HEXAGON_TERNARY_TILES_H
#define GGML_HEXAGON_TERNARY_TILES_H

#include <stdint.h>
#include <string.h>

// Bytes of one 32-row x 32-k tile. Must equal HTP_MM_WEIGHT_TILE_SIZE_Q2_0 / _Q1_0 in
// htp/matmul-ops.h (ggml-hexagon.cpp static_asserts it).
#define GGML_HEX_TERNARY_TILE_Q2_0 320
#define GGML_HEX_TERNARY_TILE_Q1_0 192

// --- GluRun ternary: Q2_0 / Q1_0 -> native 2-bit / 1-bit tiles (32 rows x 32 k), layout in
//   htp/hvx-mm-kernels-ternary.h:
//   Q2_0 (320 bytes): byte 128*(j/16) + 4*row + (j%4), bits 2*((j%16)/4) = code of k = j; fp16 row scales at 256
//   Q1_0 (192 bytes): byte 4*row + (j%4), bit j/4 = sign bit of k = j;                     fp16 row scales at 128
// The ggml block scale (64 or 128 k wide) is repeated in the 2 or 4 k-tiles it covers. Padding rows carry zero
// bits and a zero scale, so they contribute nothing to the dot product.
// `is_q2_0` picks the type: true = Q2_0 (2-bit codes, 64-wide ggml blocks), false = Q1_0 (sign
// bits, 128-wide ggml blocks).
static inline size_t ternary_tile_size(bool is_q2_0) {
    return is_q2_0 ? GGML_HEX_TERNARY_TILE_Q2_0 : GGML_HEX_TERNARY_TILE_Q1_0;
}

static inline void ternary_tile_put_row(bool is_q2_0, uint8_t * tile, int row, const uint8_t * ggml_row, int kt) {
    if (is_q2_0) {
        const block_q2_0 * b = (const block_q2_0 *) ggml_row + (kt / 2);
        const int sub = (kt % 2) * 32;
        for (int j = 0; j < 32; j++) {
            const int e = sub + j;
            const uint8_t c = (b->qs[e / 4] >> ((e % 4) * 2)) & 0x03;
            tile[128 * (j / 16) + 4 * row + (j % 4)] |= (uint8_t) (c << (2 * ((j % 16) / 4)));
        }
        ((ggml_half *) (tile + 256))[row] = b->d;
    } else {
        const block_q1_0 * b = (const block_q1_0 *) ggml_row + (kt / 4);
        const int sub = (kt % 4) * 32;
        for (int j = 0; j < 32; j++) {
            const int e = sub + j;
            const uint8_t bit = (b->qs[e / 8] >> (e % 8)) & 0x01;
            tile[4 * row + (j % 4)] |= (uint8_t) (bit << (j / 4));
        }
        ((ggml_half *) (tile + 128))[row] = b->d;
    }
}

static inline void ternary_tile_get_row(bool is_q2_0, const uint8_t * tile, int row, uint8_t * ggml_row, int kt) {
    if (is_q2_0) {
        block_q2_0 * b = (block_q2_0 *) ggml_row + (kt / 2);
        const int sub = (kt % 2) * 32;
        if (sub == 0) { memset(b->qs, 0, sizeof(b->qs)); b->d = ((const ggml_half *) (tile + 256))[row]; }
        for (int j = 0; j < 32; j++) {
            const int e = sub + j;
            const uint8_t c = (tile[128 * (j / 16) + 4 * row + (j % 4)] >> (2 * ((j % 16) / 4))) & 0x03;
            b->qs[e / 4] |= (uint8_t) (c << ((e % 4) * 2));
        }
    } else {
        block_q1_0 * b = (block_q1_0 *) ggml_row + (kt / 4);
        const int sub = (kt % 4) * 32;
        if (sub == 0) { memset(b->qs, 0, sizeof(b->qs)); b->d = ((const ggml_half *) (tile + 128))[row]; }
        for (int j = 0; j < 32; j++) {
            const int e = sub + j;
            const uint8_t bit = (tile[4 * row + (j % 4)] >> (j / 4)) & 0x01;
            b->qs[e / 8] |= (uint8_t) (bit << (e % 8));
        }
    }
}

#endif  // GGML_HEXAGON_TERNARY_TILES_H
