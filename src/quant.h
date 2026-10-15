/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/*
 * quant.h — dequantization for the k-quant subset Qwen2.5 Q4_K_M uses.
 * Block layouts mirror ggml exactly (we copy blocks verbatim, own the math).
 */
#pragma once
#include <stdint.h>
#include <stddef.h>

#define QK_K 256

/* ggml block layouts (packed; sizes asserted in quant.c). */
typedef struct {
	uint16_t d;          /* fp16 super-block scale of scales */
	uint16_t dmin;       /* fp16 super-block scale of mins */
	uint8_t  scales[12]; /* 6-bit packed scales+mins for 8 sub-blocks */
	uint8_t  qs[128];    /* 4-bit quants (256 values) */
} block_q4_K;            /* 144 bytes */

typedef struct {
	uint8_t  ql[128];    /* lower 4 bits */
	uint8_t  qh[64];     /* upper 2 bits */
	int8_t   scales[16]; /* per-16 block scales, int8 */
	uint16_t d;          /* fp16 super-block scale */
} block_q6_K;            /* 210 bytes */

typedef struct {
	uint16_t d;          /* fp16 block scale */
	int8_t   qs[32];     /* 32 int8 quants */
} block_q8_0;            /* 34 bytes — what small models (dims not a multiple
                            of 256) and every "Q8_0" GGUF are made of */

float f16_to_f32(uint16_t h);

/* Dequantize `nelem` values (nelem must be a multiple of QK_K) into `out`. */
void dequant_q4_K(const void *blocks, float *out, int nelem);
void dequant_q6_K(const void *blocks, float *out, int nelem);
void dequant_q8_0(const void *blocks, float *out, int nelem);   /* nelem multiple of 32 */

/* Type-dispatched dequant. `type` is an enum mlf2_type. F32 = memcpy view,
 * F16 = widen. Returns 0 on success, -1 on unsupported type. */
int dequant_tensor(int type, const void *src, float *out, int nelem);

/* ── int8-activation × int4-weight fast path (Q4_K only) ──
 * Quantize one activation vector to int8 with a per-32 absmax scale. `n` must be
 * a multiple of 32. `xq` holds n int8 values; `xs` holds n/32 float scales. The
 * matmul then does an integer dot against the 4-bit weights and folds scales in,
 * skipping the per-token float dequant of the whole weight matrix. */
/* Also fills axs[n/32] = scale·Σ(quants) per 32-block — the Q4_K min term is
 * dmin·m·Σx, and scale·Σx depends only on the activation, so it is folded here
 * once and shared across all weight rows/threads (skips the per-row Σx). */
void quantize_row_q8_32(const float *x, int n, int8_t *xq, float *xs, float *axs);

/* Fused dot of one Q4_K weight row (n_in cols, n_in multiple of QK_K) with a
 * pre-quantized activation (xq/xs/axs from quantize_row_q8_32). Weights stay
 * exact; only the activation is quantized. Returns the float dot. */
float q4k_vec_dot(const void *w_row, int n_in, const int8_t *xq,
                  const float *xs, const float *axs);

/* Same fused dot for a Q6_K weight row. Q6_K is ~30% of a Q4_K_M model's bytes
 * (attn_v, some ffn_down, lm_head on 7B+); without this it rides the
 * dequant-whole-row float path — the biggest decode cost after the Q4 path. */
float q6k_vec_dot(const void *w_row, int n_in, const int8_t *xq, const float *xs);

/* Q8_0 row · int8 activation: int8×int8 per 32-block, one scale each side. */
float q8_0_vec_dot(const void *w_row, int n_in, const int8_t *xq, const float *xs);
