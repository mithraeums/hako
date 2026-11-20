/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* test_gpu.c — every GPU matmul kernel against the CPU's definition of the
 * type: dequant (quant.c) and a double dot. Each type, odd row counts, batch
 * sizes on both sides of the BM-row launch split, a bias, and the same tensor
 * twice (the resident copy). Skips, passing, on a machine with no OpenCL GPU. */
#include "../src/gpu.h"
#include "../src/quant.h"
#include "../src/train.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float frand(void) { return (float)rand() / RAND_MAX * 2.0f - 1.0f; }

static int fails, checks;
static void ck(int ok, const char *what) {
	checks++;
	if (!ok) { fails++; printf("  FAIL %s\n", what); }
}

static void one(int type, const char *tn, int n_in, int n_out, int B, float wscale) {
	size_t rb = quant_row_bytes(type, n_in);
	float *wf = malloc((size_t)n_in * n_out * sizeof(float));
	float *x = malloc((size_t)n_in * B * sizeof(float)), *row = malloc((size_t)n_in * sizeof(float));
	float *out = malloc((size_t)n_out * B * sizeof(float)), *bias = malloc((size_t)n_out * sizeof(float));
	unsigned char *wq = malloc(rb * n_out);
	for (size_t i = 0; i < (size_t)n_in * n_out; i++) wf[i] = frand() * wscale;
	for (size_t i = 0; i < (size_t)n_in * B; i++) x[i] = frand();
	for (int r = 0; r < n_out; r++) bias[r] = frand();
	for (int r = 0; r < n_out; r++) quantize_tensor(type, wf + (size_t)r * n_in, wq + rb * r, n_in);

	hakm_model m; memset(&m, 0, sizeof m);
	mlf2_tensor t; memset(&t, 0, sizeof t);
	m.data = wq; t.type = (uint32_t)type; t.n_dims = 2; t.dims[0] = (uint64_t)n_in; t.dims[1] = (uint64_t)n_out;
	t.nbytes = rb * n_out;

	char what[128];
	for (int pass = 0; pass < 2; pass++) {
		for (size_t i = 0; i < (size_t)n_out * B; i++) out[i] = NAN;
		int rc = gpu_matmul(out, x, &m, &t, pass ? bias : NULL, n_out, n_in, B);
		snprintf(what, sizeof what, "%s n_in %d n_out %d B %d%s: ran on the GPU", tn, n_in, n_out, B, pass ? " +bias" : "");
		ck(rc == 0, what);
		if (rc) break;
		double worst = 0, scale = 0;
		for (int r = 0; r < n_out; r++) {
			dequant_tensor(type, wq + rb * r, row, n_in);
			for (int b = 0; b < B; b++) {
				double want = pass ? bias[r] : 0;
				for (int i = 0; i < n_in; i++) want += (double)row[i] * x[(size_t)b * n_in + i];
				double got = out[(size_t)b * n_out + r];
				if (!(fabs(got - want) <= worst)) worst = isnan(got) ? INFINITY : fabs(got - want);
				if (fabs(want) > scale) scale = fabs(want);
			}
		}
		snprintf(what, sizeof what, "%s n_in %d n_out %d B %d%s: matches dequant·dot (worst %.2g of %.2g)",
		         tn, n_in, n_out, B, pass ? " +bias" : "", worst, scale);
		ck(worst <= 1e-4 * (scale > 1 ? scale : scale > 1e-3 ? 1 : 1e-3), what);
	}
	gpu_forget(wq, rb * n_out);   /* the next malloc may hand out this address again */
	free(wf); free(x); free(row); free(out); free(bias); free(wq);
}

/* gpu_gemm against a double sum: A read along either axis, accumulate or not,
   rows with gaps (lda/ldb/ldc past the width) whose gaps must stay as they were */
static void gemm_one(int M, int N, int K, int at, int add, int pad) {
	size_t sm = at ? 1 : (size_t)K + pad, sk = at ? (size_t)M + pad : 1, ldb = (size_t)N + pad, ldc = (size_t)N + pad;
	size_t na = at ? (size_t)K * sk : (size_t)M * sm, nb = (size_t)K * ldb, nc = (size_t)M * ldc;
	float *A = malloc(na * sizeof(float)), *B = malloc(nb * sizeof(float)), *C = malloc(nc * sizeof(float)), *C0 = malloc(nc * sizeof(float));
	for (size_t i = 0; i < na; i++) A[i] = frand();
	for (size_t i = 0; i < nb; i++) B[i] = frand();
	for (size_t i = 0; i < nc; i++) C[i] = C0[i] = frand();
	int rc = gpu_gemm(M, N, K, A, sm, sk, B, ldb, C, ldc, add);
	char what[128];
	snprintf(what, sizeof what, "gemm %dx%dx%d%s%s pad %d: ran on the GPU", M, N, K, at ? " A^T" : "", add ? " +=" : "", pad);
	ck(rc == 0, what);
	int bad = 0, gap = 0;
	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			double s = add ? C0[(size_t)m * ldc + n] : 0, mag = fabs(s);
			for (int k = 0; k < K; k++) {
				double p = (double)A[(size_t)m * sm + (size_t)k * sk] * B[(size_t)k * ldb + n];
				s += p; mag += fabs(p);
			}
			if (fabs(C[(size_t)m * ldc + n] - s) > 1e-5 * mag + 1e-6) bad++;
		}
		for (size_t n = (size_t)N; n < ldc; n++) if (C[(size_t)m * ldc + n] != C0[(size_t)m * ldc + n]) gap++;
	}
	snprintf(what, sizeof what, "gemm %dx%dx%d%s%s pad %d: %d of %d off", M, N, K, at ? " A^T" : "", add ? " +=" : "", pad, bad, M * N);
	ck(!rc && !bad, what);
	snprintf(what, sizeof what, "gemm %dx%dx%d pad %d: the gaps between rows kept", M, N, K, pad);
	ck(!gap, what);
	free(A); free(B); free(C); free(C0);
}

/* one training run's loss and gradients with the products on the GPU against the
   same run on the CPU; a forward pass alone stays on the CPU, bit for bit */
static void train_one(const tr_config *c, const char *tn) {
	tr_model *a = tr_new(c, 5), *b = tr_new(c, 5);
	int Bn = 3, T = c->ctx, n = Bn * T, *in = malloc(n * sizeof(int)), *tg = malloc(n * sizeof(int));
	for (int i = 0; i < n; i++) { in[i] = rand() % c->vocab; tg[i] = rand() % c->vocab; }
	char what[128];
	tr_use_gpu(0);
	double la = tr_run(a, in, tg, Bn, T, 1.0f);
	tr_use_gpu(1);
	long g0 = gpu_gemm_count();
	double lb = tr_run(b, in, tg, Bn, T, 1.0f);
	snprintf(what, sizeof what, "%s: a training run put %ld products on the GPU", tn, gpu_gemm_count() - g0);
	ck(gpu_gemm_count() - g0 > 0, what);
	snprintf(what, sizeof what, "%s: loss %.9g on the GPU, %.9g on the CPU", tn, lb, la);
	ck(la > 0 && fabs(la - lb) <= 1e-5 * la, what);
	const float *ga = tr_grads(a), *gb = tr_grads(b);
	double gmax = 0, dmax = 0;
	for (size_t i = 0; i < tr_n_params(a); i++) {
		if (fabs(ga[i]) > gmax) gmax = fabs(ga[i]);
		if (fabs(ga[i] - gb[i]) > dmax) dmax = fabs(ga[i] - gb[i]);
	}
	snprintf(what, sizeof what, "%s: grads differ by at most %.3g of their largest %.3g", tn, dmax, gmax);
	ck(dmax <= 1e-4 * gmax, what);
	/* three whole steps each (clip, adamw), the last at another batch size, then the
	   weights brought home: the GPU's steps are the CPU's to float rounding */
	for (int st = 0; st < 3; st++) {
		int bn = st == 2 ? 2 : Bn;
		for (int k = 0; k < 2; k++) {
			tr_model *m = k ? b : a;
			tr_use_gpu(k);
			tr_zero_grad(m);
			tr_run(m, in, tg, bn, T, 1.0f / (bn * T));
			tr_clip(m, 0.5);
			tr_adamw(m, 1e-3f, 0.1f, 0.9f, 0.95f, 1e-8f);
		}
	}
	/* adam divides each grad by its own size, so a grad near 0 whose sign is a rounding
	   apart moves its weight lr either way: the steps agree on the loss, not the weights */
	const float *pb = tr_params(b);
	g0 = gpu_gemm_count();
	double f0 = tr_run(a, in, tg, Bn, T, 0), fa = tr_run(a, in, tg, Bn, T, 0), fb = tr_run(b, in, tg, Bn, T, 0);
	snprintf(what, sizeof what, "%s: after 3 steps the loss is %.6g on the GPU's weights, %.6g on the CPU's", tn, fb, fa);
	ck(fabs(fa - fb) <= 1e-3 * fa && fa < la * 0.999, what);
	snprintf(what, sizeof what, "%s: a forward pass alone stays on the CPU, bit for bit", tn);
	ck(f0 == fa && gpu_gemm_count() == g0, what);
	tr_model *c2 = tr_new(c, 5);   /* the same weights through tr_params give the same loss, bit for bit */
	memcpy(tr_params(c2), pb, tr_n_params(b) * sizeof(float));
	snprintf(what, sizeof what, "%s: the GPU's weights, brought home, score the same on the CPU", tn);
	ck(tr_run(c2, in, tg, Bn, T, 0) == fb, what);
	tr_free(c2); tr_free(a); tr_free(b); free(in); free(tg);
}

/* the step's kernels on their own against the same arithmetic in C: sumsq (the
   clip's norm), scale, zero, adamw at an offset inside a buffer */
static void step_kernels(void) {
	enum { N = 100003, OFF = 37 };
	size_t n = N + OFF;
	float *p = malloc(n * 4), *g = malloc(n * 4), *mo = malloc(n * 4), *ve = malloc(n * 4), *o = malloc(n * 4), part[256];
	for (size_t i = 0; i < n; i++) { p[i] = frand(); g[i] = frand() * 1e-2f; mo[i] = frand() * 1e-3f; ve[i] = fabsf(frand()) * 1e-5f; }
	gpu_mem *bp = gpu_new(n * 4), *bg = gpu_new(n * 4), *bm = gpu_new(n * 4), *bv = gpu_new(n * 4), *bt = gpu_new(sizeof part);
	ck(bp && bg && bm && bv && bt && !gpu_put(bp, 0, p, n * 4) && !gpu_put(bg, 0, g, n * 4) &&
	   !gpu_put(bm, 0, mo, n * 4) && !gpu_put(bv, 0, ve, n * 4), "the step kernels' buffers");
	gpu_arg sq[] = { GP(bg, OFF), GP(bt, 0), GL(N) };
	double ss = 0, gs = 0;
	for (size_t i = OFF; i < n; i++) ss += (double)g[i] * g[i];
	ck(!gpu_launch(GK_SUMSQ, 256 * 64, 1, 1, 3, sq) && !gpu_get(bt, 0, part, sizeof part), "sumsq ran");
	for (int i = 0; i < 256; i++) gs += part[i];
	ck(fabs(gs - ss) <= 1e-5 * ss, "sumsq is the sum of squares");
	float lr = 1e-3f, wd = 0.1f, b1 = 0.9f, b2 = 0.95f, eps = 1e-8f, c1 = 1 - powf(b1, 3), c2 = 1 - powf(b2, 3);
	gpu_arg ad[] = { GP(bp, OFF), GP(bg, OFF), GP(bm, OFF), GP(bv, OFF), GL(N), GF(lr), GF(wd), GF(b1), GF(b2), GF(eps), GF(c1), GF(c2) };
	ck(!gpu_launch(GK_ADAMW, N, 1, 1, 12, ad) && !gpu_get(bp, 0, o, n * 4), "adamw ran");
	int bad = 0;
	for (size_t j = 0; j < n; j++) {
		float q = p[j];
		if (j >= OFF) {
			float m = b1 * mo[j] + (1 - b1) * g[j], v = b2 * ve[j] + (1 - b2) * g[j] * g[j];
			q -= lr * ((m / c1) / (sqrtf(v / c2) + eps) + wd * q);
		}
		if (fabsf(o[j] - q) > 1e-6f * (1 + fabsf(q))) bad++;
	}
	ck(!bad, "adamw is the CPU's step, and leaves what is before its offset");
	gpu_arg sc[] = { GP(bg, OFF), GL(N), GF(0.25f) };
	ck(!gpu_launch(GK_SCALE, N, 1, 1, 3, sc) && !gpu_zero(bg, 0, OFF * 4) && !gpu_get(bg, 0, o, n * 4), "scale and zero ran");
	bad = 0;
	for (size_t j = 0; j < n; j++) if (o[j] != (j < OFF ? 0.0f : g[j] * 0.25f)) bad++;
	ck(!bad, "scale scales, zero zeroes");
	gpu_del(bp); gpu_del(bg); gpu_del(bm); gpu_del(bv); gpu_del(bt);
	free(p); free(g); free(mo); free(ve); free(o);
}

int main(void) {
	printf("test_gpu\n");
	const char *name = gpu_name();
	if (!name && gpu_broken()) { printf("test_gpu: FAILED, the kernels did not build\n"); return 1; }
	if (!name) { printf("test_gpu: no OpenCL GPU here, skipped\n"); return 0; }
	printf("  on %s\n", name);
	srand(11);
	static const struct { int type; const char *name; int n_in; } T[] = {
		{ MLF2_TYPE_F32, "f32", 260 }, { MLF2_TYPE_F16, "f16", 516 }, { MLF2_TYPE_BF16, "bf16", 1028 },
		{ MLF2_TYPE_Q8_0, "q8_0", 1056 }, { MLF2_TYPE_Q4_K, "q4_k", 1024 }, { MLF2_TYPE_Q6_K, "q6_k", 1536 },
	};
	static const int Bs[] = { 1, 3, 8, 9, 21 };
	for (size_t k = 0; k < sizeof T / sizeof *T; k++)
		for (size_t b = 0; b < sizeof Bs / sizeof *Bs; b++) one(T[k].type, T[k].name, T[k].n_in, 67, Bs[b], 0.1f);
	/* a row longer than a work-group's stride many times over, and a wide tensor */
	one(MLF2_TYPE_Q4_K, "q4_k", 11008, 33, 2, 0.1f);
	one(MLF2_TYPE_Q6_K, "q6_k", 256, 5003, 1, 0.1f);
	/* the shapes real models have, where a repacked row is exactly as long as a
	   stored one (Q8_0 at 1024: 32 x 34 = 1024 + 64; Q6_K at 2048: 8 x 210 = 1680) */
	one(MLF2_TYPE_Q8_0, "q8_0", 1024, 67, 1, 0.1f);
	one(MLF2_TYPE_Q6_K, "q6_k", 2048, 67, 1, 0.1f);
	/* weights so small their f16 values and their blocks' f16 scales are
	   subnormal: a driver that decodes those wrong fails here */
	one(MLF2_TYPE_F16, "f16 subnormal", 516, 67, 3, 4e-5f);
	one(MLF2_TYPE_Q4_K, "q4_k subnormal", 1024, 67, 3, 1e-3f);
	one(MLF2_TYPE_Q6_K, "q6_k subnormal", 1536, 67, 3, 1e-3f);
	one(MLF2_TYPE_Q8_0, "q8_0 subnormal", 1056, 67, 3, 1e-4f);
	/* a shape with no kernel is left to the CPU, out untouched */
	float o = 7, xx[30] = { 0 };
	hakm_model m; memset(&m, 0, sizeof m);
	mlf2_tensor t; memset(&t, 0, sizeof t);
	t.type = MLF2_TYPE_Q8_0;
	ck(gpu_matmul(&o, xx, &m, &t, NULL, 1, 30, 1) == -1 && o == 7, "a row that is not whole blocks stays on the CPU");
	static const int G3[][3] = { { 1, 1, 1 }, { 5, 7, 3 }, { 64, 64, 16 }, { 67, 130, 33 }, { 130, 67, 300 }, { 200, 513, 64 } };
	for (size_t g = 0; g < sizeof G3 / sizeof *G3; g++)
		for (int at = 0; at < 2; at++)
			for (int add = 0; add < 2; add++) gemm_one(G3[g][0], G3[g][1], G3[g][2], at, add, (int)(g & 1) * 3);
	step_kernels();
	setenv("HAKM_GPU_MIN", "1", 1);   /* every product of the small models below */
	tr_config dense = { 300, 64, 2, 4, 2, 16, 128, 32, 10000.0f, 1e-5f, 0, 0, 0 };
	tr_config moe = { 300, 64, 2, 4, 2, 16, 96, 32, 10000.0f, 1e-5f, 4, 2, 1 };
	train_one(&dense, "dense");
	train_one(&moe, "moe");
	printf("  %zu KB of weights resident\n", gpu_resident() >> 10);
	printf(fails ? "test_gpu: %d of %d FAILED\n" : "test_gpu: all %d passed\n", fails ? fails : checks, checks);
	return fails ? 1 : 0;
}
