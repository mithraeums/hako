/*
 * test_kdot.c — validate the fused Q4_K int8 dot (q4k_vec_dot) against a
 * dead-simple scalar reference computing the same formula inline. Exercises the
 * SIMD kernel AND the Σx-hoist (axs from quantize_row_q8_32) on REAL model
 * weight rows, so a bit-unpack or scale-fold regression is caught numerically.
 *
 *   test_kdot <model.mlf2> <q4k-tensor-name>
 */
#include "../src/loader.h"
#include "../src/quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* mirror of quant.c's static get_scale_min_k4 (ggml layout) */
static void gsm(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
	if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
	else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
	       *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

/* scalar golden: dequant-free int dot, Σx computed inline (the OLD formula). */
static float ref_dot(const block_q4_K *b, int n_in, const int8_t *xq, const float *xs) {
	int nb = n_in / QK_K;
	float acc = 0.0f;
	int sb = 0;
	for (int i = 0; i < nb; i++) {
		float d = f16_to_f32(b[i].d), dmin = f16_to_f32(b[i].dmin);
		const uint8_t *q = b[i].qs, *sc = b[i].scales;
		const int8_t *xbb = xq + (size_t)i * QK_K;
		int is = 0;
		for (int g = 0; g < QK_K; g += 64) {
			uint8_t s, m;
			int iqxA = 0, ixA = 0, iqxB = 0, ixB = 0;
			for (int l = 0; l < 32; l++) {
				int xa = xbb[g + l], xb2 = xbb[g + 32 + l];
				iqxA += (q[l] & 0xF) * xa; ixA += xa;
				iqxB += (q[l] >> 4) * xb2; ixB += xb2;
			}
			gsm(is + 0, sc, &s, &m);
			acc += xs[sb++] * (d * s * (float)iqxA - dmin * m * (float)ixA);
			gsm(is + 1, sc, &s, &m);
			acc += xs[sb++] * (d * s * (float)iqxB - dmin * m * (float)ixB);
			q += 32; is += 2;
		}
	}
	return acc;
}

int main(int argc, char **argv) {
	if (argc != 3) { fprintf(stderr, "usage: %s <model.mlf2> <q4k-tensor>\n", argv[0]); return 2; }
	hakm_model mdl;
	if (mlf2_open(argv[1], &mdl) != 0) { fprintf(stderr, "open failed\n"); return 1; }
	const mlf2_tensor *t = mlf2_find(&mdl, argv[2]);
	if (!t) { fprintf(stderr, "tensor %s not found\n", argv[2]); return 1; }
	if (t->type != 12 /* MLF2_TYPE_Q4_K */) { fprintf(stderr, "not Q4_K (type=%u)\n", t->type); return 1; }

	/* orient dims: a row is n_in cols, there are n_out rows, and
	   nbytes == n_out * (n_in/256)*144. Pick whichever assignment matches. */
	int d0 = (int)t->dims[0], d1 = (int)t->dims[1];
	int n_in = d0, n_out = d1;
	size_t rb = (size_t)(n_in / QK_K) * 144;
	if ((n_in % QK_K) || rb * (size_t)n_out != t->nbytes) {
		n_in = d1; n_out = d0;
		rb = (size_t)(n_in / QK_K) * 144;
		if ((n_in % QK_K) || rb * (size_t)n_out != t->nbytes) {
			fprintf(stderr, "can't orient dims [%d,%d] nbytes=%llu\n",
			        d0, d1, (unsigned long long)t->nbytes);
			return 1;
		}
	}
	const unsigned char *base = (const unsigned char *)mlf2_data(&mdl, t);

	int nblk = n_in / 32;
	int8_t *xq = malloc((size_t)n_in);
	float  *xs = malloc((size_t)nblk * sizeof(float));
	float  *axs = malloc((size_t)nblk * sizeof(float));
	float  *x  = malloc((size_t)n_in * sizeof(float));

	srand(1234);
	double worst_rel = 0.0, worst_abs = 0.0, worst_combined = 0.0;
	int rows = n_out < 64 ? n_out : 64;
	for (int r = 0; r < rows; r++) {
		for (int i = 0; i < n_in; i++) x[i] = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
		quantize_row_q8_32(x, n_in, xq, xs, axs);
		const block_q4_K *row = (const block_q4_K *)(base + (size_t)r * rb);
		float got = q4k_vec_dot(row, n_in, xq, xs, axs);
		float ref = ref_dot(row, n_in, xq, xs);
		double abs = fabs(got - ref);
		double rel = abs / (fabs(ref) + 1e-6);
		/* summation-order reordering (FMA path folds groups differently than the
		   scalar ref) makes rel err blow up only when the dot cancels to ~0;
		   the honest metric is abs err vs the term magnitude ~n_in. */
		double comb = abs / ((double)n_in + 1.0);
		if (rel > worst_rel) worst_rel = rel;
		if (abs > worst_abs) worst_abs = abs;
		if (comb > worst_combined) worst_combined = comb;
	}
	printf("q4k_vec_dot vs scalar ref: %d rows, n_in=%d — worst abs=%.3e rel=%.3e "
	       "normalized=%.3e\n", rows, n_in, worst_abs, worst_rel, worst_combined);
	mlf2_close(&mdl);
	free(xq); free(xs); free(axs); free(x);
	/* pass on the magnitude-normalized error; float reorder gives ~1e-5 */
	if (worst_combined > 1e-5) { printf("FAIL\n"); return 1; }
	printf("PASS\n");
	return 0;
}
