/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* test_q8.c — the fused Q8_0 dot (SIMD) must agree with dequant·dot in float,
 * and the Q8_0 dequant must match the block formula exactly. Synthetic blocks
 * with a fixed seed: no weights needed, runs under ASan from `make test_q8`. */
#include "../src/quant.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t f32_to_f16(float f) {   /* round-to-nearest, normal range only */
	uint32_t x; memcpy(&x, &f, 4);
	uint32_t sign = (x >> 16) & 0x8000;
	int32_t  exp  = (int32_t)((x >> 23) & 0xff) - 127 + 15;
	uint32_t man  = x & 0x7fffff;
	if (exp <= 0) return (uint16_t)sign;
	if (exp >= 31) return (uint16_t)(sign | 0x7c00);
	uint32_t h = sign | ((uint32_t)exp << 10) | (man >> 13);
	if (man & 0x1000) h++;
	return (uint16_t)h;
}

int main(void) {
	const int N = 4096;                 /* one row: 128 blocks */
	srand(7);
	block_q8_0 *row = malloc((size_t)(N / 32) * sizeof(block_q8_0));
	float *x = malloc((size_t)N * sizeof(float)), *w = malloc((size_t)N * sizeof(float));
	for (int i = 0; i < N / 32; i++) {
		row[i].d = f32_to_f16(0.001f + (float)(rand() % 1000) / 10000.0f);
		for (int l = 0; l < 32; l++) row[i].qs[l] = (int8_t)(rand() % 255 - 127);
	}
	for (int i = 0; i < N; i++) x[i] = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;

	int fails = 0;
	/* 1. dequant matches the formula bit-for-bit */
	dequant_q8_0(row, w, N);
	for (int i = 0; i < N; i++) {
		float want = f16_to_f32(row[i / 32].d) * (float)row[i / 32].qs[i % 32];
		if (w[i] != want) { fails++; if (fails < 4) printf("  FAIL dequant[%d] %g != %g\n", i, w[i], want); }
	}
	printf("  %s dequant_q8_0 matches block formula (%d/%d)\n", fails ? "FAIL" : "ok  ", N - fails, N);

	/* 2. fused dot vs float dot of dequant(w) · quant(x): the only error
	      allowed is the int8 activation rounding, identical on both sides */
	int8_t *xq = malloc(N); float *xs = malloc((size_t)(N / 32) * sizeof(float)), *axs = malloc((size_t)(N / 32) * sizeof(float));
	quantize_row_q8_32(x, N, xq, xs, axs);
	double ref = 0.0;
	for (int i = 0; i < N; i++) ref += (double)w[i] * ((double)xq[i] * xs[i / 32]);
	float got = q8_0_vec_dot(row, N, xq, xs);
	double rel = fabs(got - ref) / (fabs(ref) + 1e-6);
	int ok = rel < 1e-4;
	printf("  %s q8_0_vec_dot vs reference: got %.6f ref %.6f rel %.2e\n", ok ? "ok  " : "FAIL", got, ref, rel);
	fails += !ok;

	/* 3. every 32-block on its own (catches a lane/permute slip that cancels in the sum) */
	int bad = 0;
	for (int i = 0; i < N / 32; i++) {
		double r = 0.0;
		for (int l = 0; l < 32; l++) r += (double)w[i * 32 + l] * ((double)xq[i * 32 + l] * xs[i]);
		float g = q8_0_vec_dot(row + i, 32, xq + i * 32, xs + i);
		if (fabs(g - r) / (fabs(r) + 1e-6) > 1e-4) bad++;
	}
	printf("  %s per-block agreement (%d/%d)\n", bad ? "FAIL" : "ok  ", N / 32 - bad, N / 32);
	fails += bad;

	free(row); free(x); free(w); free(xq); free(xs); free(axs);
	printf(fails ? "test_q8: FAILED\n" : "test_q8: all passed\n");
	return fails ? 1 : 0;
}
