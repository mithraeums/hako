/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* test_quantize.c — the quantizer against the dequant that defines each layout.
   A block is right when dequant gives back what it was made from: exactly for
   f16 and q8_0 (their rounding is fixed), within a type's error otherwise, and
   better than the plain min/max grid the scale search is there to beat. */
#include "../src/quant.h"
#include "../include/mlf2.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
static void ck(int c, const char *w) { printf("  %s %s\n", c ? "ok  " : "FAIL", w); if (!c) fails++; }

static unsigned rng = 7;
static float urand(void) { rng = rng * 1664525u + 1013904223u; return ((rng >> 8) + 0.5f) / 16777216.0f; }
static float gauss(void) { return sqrtf(-2 * logf(urand())) * cosf(6.2831853f * urand()); }

enum { N = 256 * 64 };

static float rel_rmse(int type, const float *x, float *y) {
	static unsigned char buf[N * 4];
	size_t nb = quantize_tensor(type, x, buf, N);
	if (!nb || dequant_tensor(type, buf, y, N)) return INFINITY;
	double e = 0, s = 0;
	for (int i = 0; i < N; i++) { e += (double)(y[i] - x[i]) * (y[i] - x[i]); s += (double)x[i] * x[i]; }
	return (float)sqrt(e / s);
}

/* plain Q4_K grid with no search: scale = (max-min)/15 per 32, same 6-bit storage */
static float naive_q4k(const float *x) {
	double e = 0, s = 0;
	for (int b = 0; b < N; b += 32) {
		float lo = 0, hi = x[b];
		for (int l = 0; l < 32; l++) { if (x[b + l] < lo) lo = x[b + l]; if (x[b + l] > hi) hi = x[b + l]; }
		float sc = (hi - lo) / 15;
		for (int l = 0; l < 32; l++) {
			int q = sc ? (int)lrintf((x[b + l] - lo) / sc) : 0;
			float y = sc * q + lo;
			e += (double)(y - x[b + l]) * (y - x[b + l]); s += (double)x[b + l] * x[b + l];
		}
	}
	return (float)sqrt(e / s);
}

int main(void) {
	/* f16: every finite half survives the trip through float and back */
	int bad = 0;
	for (unsigned h = 0; h < 65536; h++) {
		if ((h & 0x7C00) == 0x7C00 && (h & 0x3FF)) continue;   /* NaN */
		if (f32_to_f16(f16_to_f32((uint16_t)h)) != h) bad++;
	}
	ck(bad == 0, "f32_to_f16(f16_to_f32(h)) == h for all 63,488 non-NaN halves");
	ck(f32_to_f16(1.0f + 1.0f / 2048) == 0x3C00 && f32_to_f16(1.0f + 3.0f / 2048) == 0x3C02,
	   "halfway cases round to even");
	ck(f32_to_f16(1e6f) == 0x7C00 && f32_to_f16(-1e6f) == 0xFC00, "overflow saturates to inf");
	ck(f32_to_f16(1e-8f) == 0 && f32_to_f16(6e-8f) == 1, "subnormals round, not flush");

	/* weights shaped like a real layer: gaussian, scale varies by row, a few outliers */
	float *x = malloc(sizeof(float) * N), *y = malloc(sizeof(float) * N);
	for (int i = 0; i < N; i++) x[i] = gauss() * 0.02f * (1 + (i / 256) % 5) + (i % 997 == 0 ? 0.4f : 0);

	float e8 = rel_rmse(MLF2_TYPE_Q8_0, x, y);
	int exact = 1;
	for (int b = 0; b < N && exact; b += 32) {
		float amax = 0; for (int l = 0; l < 32; l++) if (fabsf(x[b + l]) > amax) amax = fabsf(x[b + l]);
		float d = amax / 127, dd = f16_to_f32(f32_to_f16(d));
		for (int l = 0; l < 32; l++) if (y[b + l] != dd * roundf(x[b + l] / d)) exact = 0;
	}
	ck(exact, "q8_0 is d = absmax/127 (stored f16), q = round(x/d), exactly");
	printf("       relative rmse: q8_0 %.4f", e8);
	float e6 = rel_rmse(MLF2_TYPE_Q6_K, x, y);
	float e4 = rel_rmse(MLF2_TYPE_Q4_K, x, y);
	float en = naive_q4k(x);
	printf("  q6_k %.4f  q4_k %.4f  (plain min/max 4-bit: %.4f)\n", e6, e4, en);
	ck(e8 < 0.006f, "q8_0 error is small");
	ck(e6 < 0.03f, "q6_k error is small");
	ck(e4 < 0.10f && e4 < en, "q4_k error is small, and the scale search beats the plain grid");
	ck(e8 < e6 && e6 < e4, "more bits, less error (q8_0 < q6_k < q4_k)");

	/* a block already on a q4_k grid comes back from a second trip unchanged */
	static unsigned char b1[N], b2[N];
	quantize_tensor(MLF2_TYPE_Q4_K, x, b1, N); dequant_tensor(MLF2_TYPE_Q4_K, b1, y, N);
	float *z = malloc(sizeof(float) * N);
	quantize_tensor(MLF2_TYPE_Q4_K, y, b2, N); dequant_tensor(MLF2_TYPE_Q4_K, b2, z, N);
	double d2 = 0, s2 = 0;
	for (int i = 0; i < N; i++) { d2 += (double)(z[i] - y[i]) * (z[i] - y[i]); s2 += (double)y[i] * y[i]; }
	ck(sqrt(d2 / s2) < 0.01, "requantizing q4_k output is stable (does not drift)");

	memset(x, 0, sizeof(float) * N);
	ck(rel_rmse(MLF2_TYPE_Q4_K, x, y) != INFINITY && y[0] == 0 && y[N - 1] == 0, "an all-zero tensor stays zero (q4_k)");
	ck(rel_rmse(MLF2_TYPE_Q6_K, x, y) != INFINITY && y[0] == 0 && y[N - 1] == 0, "an all-zero tensor stays zero (q6_k)");
	ck(quantize_tensor(MLF2_TYPE_Q4_K, x, b1, 100) == 0, "a size that does not fill whole blocks is refused");

	free(x); free(y); free(z);
	printf(fails ? "test_quantize: %d FAILED\n" : "test_quantize: all passed\n", fails);
	return fails ? 1 : 0;
}
