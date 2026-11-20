/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* test_train.c — the trainer: every tensor's gradient against a finite
   difference, the engine scoring the saved file the way the trainer does, the
   BPE tokenizer round-tripping through the engine's encoder, a loss that
   falls, and checkpoints that come back bit for bit. Then the same for a moe
   ffn (shared expert or not), plus: routes that hold across the differences,
   the aux losses' gradient, a moe model built from a dense one giving its
   loss and gradients, the .opt versions refusing each other, and the aux
   losses moving the routing the way they say. ASan. */
#include "../src/train.h"
#include "../src/model.h"
#include "../src/bpe.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
static void ck(int c, const char *w) { printf("  %s %s\n", c ? "ok  " : "FAIL", w); if (!c) fails++; }

static uint64_t rs = 7;
static float frand(void) { rs = rs * 6364136223846793005ULL + 1442695040888963407ULL; return ((rs >> 40) & 0xFFFFFF) / 16777216.0f - 0.5f; }

/* ── experts ── */

enum { MB = 2, MT = 12, MN = MB * MT };

/* tensor spans in the trainer's order, written out here so the layout is
   checked, not borrowed: per layer the dense part with the router after
   ffn_norm and the shared ffn only when asked, output_norm, then every
   expert's gate, up, down in (l, e) order */
static int spans(const tr_config *c, size_t *sp) {
	size_t D = c->dim, Q = (size_t)c->heads * c->head_dim, KV = (size_t)c->kv_heads * c->head_dim, F = c->ffn, E = c->experts;
	int n = 0, sh = !E || c->shared;
	sp[n++] = (size_t)c->vocab * D;
	for (int l = 0; l < c->layers; l++) {
		sp[n++] = D; sp[n++] = Q * D; sp[n++] = KV * D; sp[n++] = KV * D; sp[n++] = D * Q; sp[n++] = D;
		if (E) sp[n++] = E * D;
		if (sh) { sp[n++] = F * D; sp[n++] = F * D; sp[n++] = D * F; }
	}
	sp[n++] = D;
	for (size_t i = 0; i < (size_t)c->layers * E * 3; i++) sp[n++] = F * D;
	return n;
}

/* every layer's picks of the last run, side by side, each token's as a sorted
   set: the order of two picks swaps on a near tie and changes nothing (same
   experts, same gates), the set is what the loss depends on */
static void routes_get(const tr_model *m, int N, int *out) {
	const tr_config *c = tr_cfg(m);
	int K = c->top_k;
	for (int l = 0; l < c->layers; l++) {
		const int *r = tr_routes(m, l);
		if (!r) continue;
		int *o = out + (size_t)l * N * K;
		memcpy(o, r, (size_t)N * K * sizeof(int));
		for (int n = 0; n < N; n++)
			for (int s = 1; s < K; s++)
				for (int j = s; j > 0 && o[n * K + j - 1] > o[n * K + j]; j--) {
					int t = o[n * K + j]; o[n * K + j] = o[n * K + j - 1]; o[n * K + j - 1] = t;
				}
	}
}

/* experts a token runs that the one before it (same sequence) did not, per
   token per layer; and the busiest expert's picks over the mean */
static double fresh(const tr_model *m, int B, int T) {
	const tr_config *c = tr_cfg(m);
	int K = c->top_k;
	double f = 0;
	for (int l = 0; l < c->layers; l++) {
		const int *r = tr_routes(m, l);
		for (int n = 0; n < B * T; n++)
			for (int s = 0; n % T && s < K; s++) {
				int seen = 0;
				for (int j = 0; j < K; j++) seen |= r[(n - 1) * K + j] == r[n * K + s];
				f += !seen;
			}
	}
	return f / ((double)B * (T - 1) * c->layers);
}
static double max_load(const tr_model *m, int B, int T) {
	const tr_config *c = tr_cfg(m);
	double w = 0;
	for (int l = 0; l < c->layers; l++) {
		const int *r = tr_routes(m, l);
		int cnt[64] = { 0 }, mx = 0;
		for (int i = 0; i < B * T * c->top_k; i++) cnt[r[i]]++;
		for (int e = 0; e < c->experts; e++) if (cnt[e] > mx) mx = cnt[e];
		w += (double)mx * c->experts / ((double)B * T * c->top_k);
	}
	return w / c->layers;
}

static unsigned char *read_all(const char *path, size_t *n) {
	FILE *f = fopen(path, "rb");
	if (!f) return NULL;
	fseek(f, 0, SEEK_END);
	long len = ftell(f);
	fseek(f, 0, SEEK_SET);
	unsigned char *b = malloc(len > 0 ? (size_t)len : 1);
	*n = b && len >= 0 && fread(b, 1, (size_t)len, f) == (size_t)len ? (size_t)len : 0;
	fclose(f);
	return b;
}
static int write_all(const char *path, const void *b, size_t n) {
	FILE *f = fopen(path, "wb");
	int ok = f && fwrite(b, 1, n, f) == n;
	if (f && fclose(f) != 0) ok = 0;
	return ok;
}

/* one moe shape through everything the dense model goes through, and more */
static void moe_case(int E, int K, int shared, uint64_t seed, const int *ids, const void *blob, size_t blen, int eos) {
	printf("  moe: %d experts, top %d, %s\n", E, K, shared ? "a shared ffn" : "no shared ffn");
	tr_config c = { 320, 16, 2, 4, 2, 8, 24, 12, 10000.0f, 1e-5f, E, K, shared };
	ck(tr_check(&c) == NULL, "the moe shape is valid");
	rs = seed;
	tr_model *m = tr_new(&c, seed);
	size_t np = tr_n_params(m), sp[128], off[128], at = 0;
	int ns = spans(&c, sp), nl = 7 + (shared ? 3 : 0);
	for (int t = 0; t < ns; t++) { off[t] = at; at += sp[t]; }
	ck(at == np, "layout: the dense part with routers after ffn_norm, then the experts");
	int ok = tr_expert_off(m, 0, E) == SIZE_MAX && tr_expert_off(m, c.layers, 0) == SIZE_MAX;
	for (int l = 0; l < c.layers; l++)
		for (int e = 0; e < E; e++) ok &= tr_expert_off(m, l, e) == off[ns - 3 * c.layers * E + 3 * (l * E + e)];
	ck(ok, "tr_expert_off: expert (l, e) is one range of 3 ffn x dim, in order after output_norm");

	float *p = tr_params(m);
	for (size_t i = 0; i < np; i++) p[i] = p[i] == 1.0f ? 1.0f + 0.4f * frand() : 0.8f * frand();
	int in[MN], tg[MN];
	for (int i = 0; i < MN; i++) { in[i] = ids[i * 3]; tg[i] = ids[i * 3 + 1]; }
	tr_set_aux(m, 0, 0);
	double nll0 = tr_run(m, in, tg, MB, MT, 0);
	tr_set_aux(m, 0.5f, 0.5f);
	tr_zero_grad(m);
	double l0 = tr_run(m, in, tg, MB, MT, 1.0f), a0 = tr_aux(m);
	ck(l0 > 0 && l0 == nll0, "forward + backward runs, and tr_run gives the NLL alone whatever the aux weights");
	ck(isfinite(a0) && a0 != 0, "the aux losses have a value (tr_aux)");
	int RN = c.layers * MN * K, *r0 = malloc(RN * sizeof(int)), *r1 = malloc(RN * sizeof(int));
	routes_get(m, MN, r0);
	int every = 1, valid = 1;
	for (int l = 0; l < c.layers; l++)
		for (int e = 0; e < E; e++) {
			int got = 0;
			for (int i = 0; i < MN * K; i++) got += r0[l * MN * K + i] == e;
			every &= got > 0;
		}
	for (int i = 0; i < RN; i++) valid &= r0[i] >= 0 && r0[i] < E;
	for (int l = 0; l < c.layers; l++)   /* a token picks an expert at most once */
		for (int n = 0; n < MN; n++)
			for (int s = 0; s < K; s++)
				for (int j = 0; j < s; j++) valid &= r0[(l * MN + n) * K + s] != r0[(l * MN + n) * K + j];
	ck(valid, "every pick is an expert, no token picks one twice");
	ck(every, "every expert of every layer got a token (none of the checks below is vacuous)");

	/* finite differences of J = NLL + aux on every tensor, routers and experts too */
	float *g = malloc(np * sizeof(float)), *dir = malloc(np * sizeof(float)), *g0 = malloc(np * sizeof(float));
	memcpy(g, tr_grads(m), np * sizeof(float));
	double worst = 0, rworst = 0;
	int held = 1;
	const double h = 1e-3;
	for (int t = 0; t < ns; t++) {
		double an = 0;
		for (size_t i = off[t]; i < off[t] + sp[t]; i++) { dir[i] = frand(); an += (double)dir[i] * g[i]; }
		double J[2];
		for (int k = 0; k < 2; k++) {
			p = tr_params(m);
			for (size_t i = off[t]; i < off[t] + sp[t]; i++) p[i] += (float)(k ? -2 * h : h) * dir[i];
			J[k] = tr_run(m, in, tg, MB, MT, 0) + tr_aux(m);
			routes_get(m, MN, r1);
			held &= !memcmp(r0, r1, RN * sizeof(int));
		}
		p = tr_params(m);
		for (size_t i = off[t]; i < off[t] + sp[t]; i++) p[i] += (float)h * dir[i];
		/* a floor of 1e-2: an expert few tokens picked has a small gradient, and
		   the difference's float noise is a few 1e-4 */
		double fd = (J[0] - J[1]) / (2 * h), rel = fabs(fd - an) / (fabs(an) + 1e-2);
		int router = t > 0 && t < 1 + nl * c.layers && (t - 1) % nl == 6;
		if (rel > worst) worst = rel;
		if (router && rel > rworst) rworst = rel;
		if (rel > 2e-2 || router) printf("    tensor %d%s: analytic %.6f finite %.6f\n", t, router ? " (router)" : "", an, fd);
	}
	printf("    worst relative error %.2e over %d tensors (routers %.2e)\n", worst, ns, rworst);
	ck(held, "every token picks the same experts in every run of the differences (top-k held still)");
	ck(worst < 2e-2, "every tensor's gradient of NLL + aux matches its finite difference");
	/* grad_scale scales the aux gradient with the loss's (microbatches add up) */
	tr_zero_grad(m);
	tr_run(m, in, tg, MB, MT, 0.25f);
	double sdiff = 0, gbig = 0;
	for (size_t i = 0; i < np; i++) {
		if (fabs(g[i]) > gbig) gbig = fabs(g[i]);
		if (fabs(tr_grads(m)[i] - 0.25 * g[i]) > sdiff) sdiff = fabs(tr_grads(m)[i] - 0.25 * g[i]);
	}
	ck(sdiff <= 1e-6 * gbig, "grad_scale 0.25 gives a quarter of every gradient, the aux's too");

	/* each aux loss alone reaches each router: grads with it minus grads without, against tr_aux's differences */
	for (int kind = 0; kind < 2; kind++) {
		tr_set_aux(m, 0, 0);
		tr_zero_grad(m);
		tr_run(m, in, tg, MB, MT, 1.0f);
		memcpy(g0, tr_grads(m), np * sizeof(float));
		tr_set_aux(m, kind ? 0 : 0.5f, kind ? 0.5f : 0);
		tr_zero_grad(m);
		tr_run(m, in, tg, MB, MT, 1.0f);
		const float *gk = tr_grads(m);
		double aw = 0, amin = 1e9;
		for (int l = 0; l < c.layers; l++) {
			int t = 1 + l * nl + 6;
			double an = 0, A[2];
			for (size_t i = off[t]; i < off[t] + sp[t]; i++) { dir[i] = frand(); an += (double)dir[i] * (gk[i] - g0[i]); }
			for (int k = 0; k < 2; k++) {
				p = tr_params(m);
				for (size_t i = off[t]; i < off[t] + sp[t]; i++) p[i] += (float)(k ? -2 * h : h) * dir[i];
				tr_run(m, in, tg, MB, MT, 0);
				A[k] = tr_aux(m);
			}
			p = tr_params(m);
			for (size_t i = off[t]; i < off[t] + sp[t]; i++) p[i] += (float)h * dir[i];
			double fd = (A[0] - A[1]) / (2 * h), rel = fabs(fd - an) / (fabs(an) + 1e-3);
			printf("    %s, layer %d router: analytic %.6f finite %.6f\n", kind ? "balance" : "sticky", l, an, fd);
			if (rel > aw) aw = rel;
			if (fabs(fd) < amin) amin = fabs(fd);
		}
		ck(aw < 2e-2 && amin > 1e-2, kind ? "the balance loss's gradient reaches every router, and is right"
		                                  : "the sticky loss's gradient reaches every router, and is right");
	}
	/* sticky pairs a token with the one before it in its own sequence only:
	   sequences of one token give it nothing, value or gradient */
	tr_set_aux(m, 0, 0);
	tr_zero_grad(m);
	tr_run(m, in, tg, MN, 1, 1.0f);
	memcpy(g0, tr_grads(m), np * sizeof(float));
	tr_set_aux(m, 0.5f, 0);
	tr_zero_grad(m);
	tr_run(m, in, tg, MN, 1, 1.0f);
	ck(tr_aux(m) == 0 && !memcmp(g0, tr_grads(m), np * sizeof(float)), "sticky never reaches across sequences (T = 1: no value, no gradient)");
	tr_set_aux(m, 0.5f, 0.5f);

	/* the engine runs the saved file the trainer's way, batched and token by token */
	char path[64], opt[64], opt2[64];
	snprintf(path, sizeof path, "build/test_train_moe%d.mlf2", shared);
	snprintf(opt, sizeof opt, "build/test_train_moe%d.opt", shared);
	snprintf(opt2, sizeof opt2, "build/test_train_moe%d.2.opt", shared);
	ck(tr_save(m, path, blob, blen, eos) == 0, "save the moe model as MLF2");
	hakm_ctx *e = hakm_ctx_new(path, c.ctx + 1);
	ck(e && e->n_experts == E && e->n_used == K, "hakm loads it with its experts and top-k");
	int seq[MT + 1];
	for (int i = 0; i <= MT; i++) seq[i] = ids[200 + i];
	double eng = e ? hakm_score_batch(e, seq, MT + 1, 0) : -1, dec = 0;
	double trn = tr_run(m, seq, seq + 1, 1, MT, 0);
	if (e) hakm_reset(e);
	for (int i = 0; e && i < MT; i++) {
		const float *z = hakm_forward(e, seq[i]);
		float mx = z[0];
		for (int v = 1; v < c.vocab; v++) if (z[v] > mx) mx = z[v];
		double se = 0;
		for (int v = 0; v < c.vocab; v++) se += exp((double)z[v] - mx);
		dec += log(se) + mx - z[seq[i + 1]];
	}
	printf("    nll: trainer %.6f engine batched %.6f decoded %.6f\n", trn, eng, dec);
	ck(e && fabs(eng - trn) < 2e-3 * trn, "hakm_score_batch gives the trainer's NLL");
	ck(e && fabs(dec - trn) < 2e-3 * trn, "a decode loop through hakm_forward gives it too");
	hakm_ctx_free(e);

	/* weights and the v2 optimizer come back bit for bit */
	tr_adamw(m, 1e-3f, 0.1f, 0.9f, 0.95f, 1e-8f);
	ck(tr_save(m, path, blob, blen, eos) == 0 && tr_save_opt(m, opt, 77) == 0, "save weights + optimizer");
	tr_model *r = tr_load(path, NULL, NULL, NULL);
	const tr_config *rc = r ? tr_cfg(r) : NULL;
	ck(r && rc->experts == E && rc->top_k == K && rc->shared == shared && tr_n_params(r) == np &&
	   !memcmp(tr_params(r), tr_params(m), np * sizeof(float)), "moe weights and shape come back bit for bit");
	uint64_t rng = 0;
	ck(r && tr_load_opt(r, opt, &rng) == 0 && *tr_step(r) == 1 && rng == 77, "the v2 optimizer loads");
	size_t n1 = 0, n2 = 0;
	unsigned char *b1 = read_all(opt, &n1), *b2 = NULL;
	ck(r && tr_save_opt(r, opt2, 77) == 0 && (b2 = read_all(opt2, &n2)) && n1 == n2 && !memcmp(b1, b2, n1),
	   "and writes the same bytes back");
	free(b2);
	if (r) {   /* the moments came back too: one more step on each lands on the same weights */
		tr_set_aux(r, 0.5f, 0.5f);
		tr_zero_grad(m); tr_zero_grad(r);
		tr_run(m, in, tg, MB, MT, 1.0f / MN); tr_run(r, in, tg, MB, MT, 1.0f / MN);
		tr_adamw(m, 1e-3f, 0.1f, 0.9f, 0.95f, 1e-8f); tr_adamw(r, 1e-3f, 0.1f, 0.9f, 0.95f, 1e-8f);
		ck(!memcmp(tr_params(r), tr_params(m), np * sizeof(float)), "a step after the reload matches one without it");
	}
	uint32_t ver = 0;
	int32_t ex[4] = { 0 };
	if (n1 >= 88) { memcpy(&ver, b1 + 4, 4); memcpy(ex, b1 + 72, 16); }
	ck(n1 == 88 + 8 * np && ver == 2 && ex[0] == E && ex[1] == K && ex[2] == shared && ex[3] == 0,
	   "a moe .opt is version 2, experts, top-k, shared and 0 after the header");
	if (r && n1 == 88 + 8 * np) {
		/* the same moments as version 1 (no shape after the header): refused */
		unsigned char *v1 = malloc(n1);
		uint32_t one = 1;
		memcpy(v1, b1, 72); memcpy(v1 + 4, &one, 4); memcpy(v1 + 72, b1 + 88, n1 - 88);
		ck(write_all(opt2, v1, n1 - 16) && tr_load_opt(r, opt2, NULL) != 0, "a version 1 .opt is refused for a moe model");
		/* the shape after the header must be this model's */
		memcpy(v1, b1, n1);
		int32_t k2 = K == 2 ? 3 : 2;
		memcpy(v1 + 76, &k2, 4);
		ck(write_all(opt2, v1, n1) && tr_load_opt(r, opt2, NULL) != 0, "a version 2 .opt with another top-k is refused");
		free(v1);
	}
	free(b1);
	tr_free(r);
	remove(path); remove(opt); remove(opt2);
	free(g); free(g0); free(dir); free(r0); free(r1);
	tr_free(m);
}

/* a moe model whose every expert is a copy of a dense model's ffn (no shared
   one) computes that model: its loss, every trunk gradient, and the dense
   ffn's gradient as the sum over its experts. the router, all gates equal,
   gets no gradient from the loss */
static void moe_dense_equal(const int *ids) {
	tr_config cd = { 320, 16, 2, 4, 2, 8, 24, 12, 10000.0f, 1e-5f, 0, 0, 0 };
	tr_config cm = cd;
	cm.experts = 4; cm.top_k = 2; cm.shared = 0;
	tr_model *d = tr_new(&cd, 21), *x = tr_new(&cm, 22);
	size_t D = cd.dim, F = cd.ffn, FD = F * D, Q = (size_t)cd.heads * cd.head_dim, KV = (size_t)cd.kv_heads * cd.head_dim;
	size_t V = cd.vocab, A = 2 * D + 2 * Q * D + 2 * KV * D, Ld = A + 3 * FD, Lm = A + cm.experts * D;
	float *pd = tr_params(d), *px = tr_params(x);
	for (size_t i = 0; i < tr_n_params(d); i++) pd[i] = pd[i] == 1.0f ? 1.0f + 0.4f * frand() : 0.8f * frand();
	for (size_t i = 0; i < tr_n_params(x); i++) px[i] = 0.8f * frand();   /* routers stay random */
	memcpy(px, pd, V * D * sizeof(float));
	for (int l = 0; l < cd.layers; l++) {
		memcpy(px + V * D + l * Lm, pd + V * D + l * Ld, A * sizeof(float));
		for (int e = 0; e < cm.experts; e++) memcpy(px + tr_expert_off(x, l, e), pd + V * D + l * Ld + A, 3 * FD * sizeof(float));
	}
	memcpy(px + V * D + cd.layers * Lm, pd + V * D + cd.layers * Ld, D * sizeof(float));
	tr_set_aux(x, 0, 0);
	int in[MN], tg[MN];
	for (int i = 0; i < MN; i++) { in[i] = ids[i * 5]; tg[i] = ids[i * 5 + 1]; }
	tr_zero_grad(d); tr_zero_grad(x);
	double ld = tr_run(d, in, tg, MB, MT, 1.0f), lx = tr_run(x, in, tg, MB, MT, 1.0f);
	printf("    nll: dense %.7f moe %.7f\n", ld, lx);
	ck(fabs(ld - lx) <= 1e-5 * ld, "experts that copy a dense ffn give its NLL (within 1e-5)");
	const float *gd = tr_grads(d), *gx = tr_grads(x);
	double big = 0, trunk = 0, ffn = 0, rout = 0;
	for (size_t i = 0; i < tr_n_params(d); i++) if (fabs(gd[i]) > big) big = fabs(gd[i]);
	#define DIFF(acc, a, b) do { double df_ = fabs((double)(a) - (b)); if (df_ > (acc)) (acc) = df_; } while (0)
	for (size_t i = 0; i < V * D; i++) DIFF(trunk, gd[i], gx[i]);
	for (size_t i = 0; i < D; i++) DIFF(trunk, gd[V * D + cd.layers * Ld + i], gx[V * D + cd.layers * Lm + i]);
	for (int l = 0; l < cd.layers; l++) {
		for (size_t i = 0; i < A; i++) DIFF(trunk, gd[V * D + l * Ld + i], gx[V * D + l * Lm + i]);
		for (size_t i = 0; i < 3 * FD; i++) {
			double s = 0;
			for (int e = 0; e < cm.experts; e++) s += gx[tr_expert_off(x, l, e) + i];
			DIFF(ffn, gd[V * D + l * Ld + A + i], s);
		}
		for (size_t i = 0; i < cm.experts * D; i++) DIFF(rout, 0, gx[V * D + l * Lm + A + i]);
	}
	#undef DIFF
	printf("    largest grad %.4f; off by: trunk %.2e, summed experts %.2e, router %.2e\n", big, trunk, ffn, rout);
	ck(trunk < 1e-4 * big, "every trunk gradient matches the dense model's");
	ck(ffn < 1e-4 * big, "the experts' gradients sum to the dense ffn's");
	ck(rout < 1e-4 * big, "the router gets nothing from the loss when every expert is the same");
	tr_free(d); tr_free(x);
}

/* the aux losses do what they say, which differences cannot see the sign of:
   40 steps with a large sticky weight leave fewer fresh experts per token
   than none, and a large balance weight beside it leaves a lower max load */
static void moe_aux_moves(const int *ids) {
	tr_config c = { 320, 16, 2, 4, 2, 8, 24, 12, 10000.0f, 1e-5f, 4, 2, 1 };
	int in[MN], tg[MN];
	for (int i = 0; i < MN; i++) { in[i] = ids[i * 7]; tg[i] = ids[i * 7 + 1]; }
	const float lam[3][2] = { { 0, 0 }, { 1.0f, 0 }, { 1.0f, 1.0f } };
	double fr[3], ld[3], nll[3];
	for (int run = 0; run < 3; run++) {
		tr_model *m = tr_new(&c, 31);
		tr_set_aux(m, lam[run][0], lam[run][1]);
		for (int s = 0; s < 40; s++) {
			tr_zero_grad(m);
			tr_run(m, in, tg, MB, MT, 1.0f / MN);
			tr_clip(m, 1.0);
			tr_adamw(m, 1e-2f, 0.0f, 0.9f, 0.95f, 1e-8f);
		}
		nll[run] = tr_run(m, in, tg, MB, MT, 0) / MN;
		fr[run] = fresh(m, MB, MT); ld[run] = max_load(m, MB, MT);
		printf("    sticky %.1f balance %.1f: %.2f fresh experts/token/layer, max/mean load %.2f, loss %.3f\n",
		       lam[run][0], lam[run][1], fr[run], ld[run], nll[run]);
		tr_free(m);
	}
	ck(fr[1] < fr[0] - 0.25, "a large sticky weight cuts the fresh experts per token");
	ck(ld[2] < ld[1] - 0.1, "a large balance weight beside it lowers the max/mean load");
}

/* the dense .opt keeps the header it always had, and each version refuses the other model */
static void opt_versions(void) {
	tr_config c = { 320, 16, 2, 4, 2, 8, 24, 12, 10000.0f, 1e-5f, 0, 2, 1 };   /* top-k and shared mean nothing here */
	tr_model *m = tr_new(&c, 41);
	ck(m && tr_cfg(m)->experts == 0 && tr_cfg(m)->top_k == 0 && tr_cfg(m)->shared == 0, "experts 0 is dense whatever top-k says");
	ck(m && tr_routes(m, 0) == NULL && tr_aux(m) == 0 && tr_expert_off(m, 0, 0) == SIZE_MAX, "dense: no routes, no aux, no experts");
	tr_adamw(m, 1e-3f, 0.1f, 0.9f, 0.95f, 1e-8f);
	const char *opt = "build/test_train_dense.opt", *opt2 = "build/test_train_dense.2.opt";
	ck(tr_save_opt(m, opt, 5) == 0, "save a dense optimizer");
	size_t n = 0, np = tr_n_params(m);
	unsigned char *b = read_all(opt, &n);
	struct { char magic[4]; uint32_t version; uint64_t n, rng; int64_t step; int shape[8]; float theta, eps; } h0 =
		{ { 'H', 'K', 'O', 'P' }, 1, np, 5, 1, { 320, 16, 2, 4, 2, 8, 24, 12 }, 10000.0f, 1e-5f };
	ck(sizeof h0 == 72 && b && n == 72 + 8 * np && !memcmp(b, &h0, 72), "a dense .opt is version 1, its header byte for byte the old one");
	if (b && n == 72 + 8 * np) {
		unsigned char *v2 = malloc(n + 16);
		uint32_t two = 2;
		int32_t ex[4] = { 0, 0, 0, 0 };
		memcpy(v2, b, 72); memcpy(v2 + 4, &two, 4); memcpy(v2 + 72, ex, 16); memcpy(v2 + 88, b + 72, n - 72);
		ck(write_all(opt2, v2, n + 16) && tr_load_opt(m, opt2, NULL) != 0, "a version 2 .opt is refused for a dense model");
		free(v2);
	}
	ck(tr_load_opt(m, opt, NULL) == 0, "and its own version 1 loads");
	free(b);
	remove(opt); remove(opt2);
	tr_free(m);
	/* shapes the trainer will not take */
	tr_config bad[5] = { c, c, c, c, c };
	bad[0].experts = 1; bad[1].experts = 4; bad[1].top_k = 1; bad[2].experts = 4; bad[2].top_k = 5;
	bad[3].experts = 16; bad[3].top_k = 9; bad[4].experts = 4; bad[4].shared = 2;
	int all = 1;
	for (int i = 0; i < 5; i++) all &= tr_check(&bad[i]) != NULL;
	ck(all, "refused: 1 expert, top-1 (no router gradient), top-k past experts or 8, shared not 0 or 1");
}

static const char *TEXT =
	"the box sat on the shelf and the box was old. the old box held a key, and the key opened "
	"the door at the end of the hall. nobody knew who made the box, or the key, or the door.\n"
	"she took the key from the box and walked to the door. the door was locked, the key turned, "
	"and the hall was quiet again. the box stayed on the shelf, older than before.\n";

int main(void) {
	printf("test_train\n");
	char *text = malloc(64 * 1024);
	text[0] = 0;
	for (int i = 0; i < 40; i++) strcat(text, TEXT);
	size_t tlen = strlen(text);

	/* ── tokenizer: trained here, encoded by the engine ── */
	void *blob; size_t blen; int eos;
	int nm = bpe_train(text, tlen, 320, &blob, &blen, &eos);
	ck(nm > 0 && nm <= 320 - 259, "bpe learns merges up to the vocab");
	ck(tok_blob_len(blob, blen) == blen, "blob length reads back from its own counts");
	hakm_model fm = {0};
	fm.tok = blob; fm.map = blob; fm.map_size = blen; fm.h.eos_id = eos;
	hakm_tok *tk = tok_new(&fm);
	ck(tk != NULL, "the engine loads the blob");
	int *ids = malloc(tlen * sizeof(int));
	int n = tk ? tok_encode(tk, text, ids, (int)tlen) : -1;
	printf("    %zu bytes -> %d tokens with %d merges\n", tlen, n, nm);
	ck(n > 0 && n < (int)tlen / 2, "the engine's encoder uses the merges (half as many tokens as bytes)");
	char *back = malloc(tlen + 1);
	size_t bl = 0;
	for (int i = 0; i < n; i++) bl += tok_decode_id(tk, ids[i], back + bl, (int)(tlen - bl));
	back[bl] = 0;
	ck(bl == tlen && !strcmp(back, text), "encode then decode gives the text back");
	ck(tk && tok_id(tk, "<|endoftext|>") == eos && tok_id(tk, "<|im_end|>") == eos + 2, "specials after the merges");
	{	/* a text with few pairs runs out of merges: pads fill the rest */
		void *pb; size_t pl; int pe;
		int pm = bpe_train("abab abab", 9, 300, &pb, &pl, &pe);
		hakm_model pf = {0};
		pf.tok = pb; pf.map = pb; pf.map_size = pl; pf.h.eos_id = pe;
		hakm_tok *pt = pm > 0 ? tok_new(&pf) : NULL;
		ck(pt && pe == 256 + pm && tok_id(pt, "[PAD299]") == 299, "pads fill the vocab when the pairs run out");
		tok_free(pt); free(pb);
	}

	/* ── gradients: directional finite difference, per tensor ── */
	tr_config c = { 320, 16, 2, 4, 2, 8, 24, 12, 10000.0f, 1e-5f, 0, 0, 0 };
	ck(tr_check(&c) == NULL, "the test shape is valid (GQA 4/2)");
	tr_model *m = tr_new(&c, 3);
	float *p = tr_params(m);
	size_t np = tr_n_params(m);
	for (size_t i = 0; i < np; i++) p[i] = p[i] == 1.0f ? 1.0f + 0.4f * frand() : 0.8f * frand();
	enum { B = 2, T = 12 };
	int in[B * T], tg[B * T];
	for (int i = 0; i < B * T; i++) { in[i] = ids[i * 3]; tg[i] = ids[i * 3 + 1]; }
	tr_zero_grad(m);
	double l0 = tr_run(m, in, tg, B, T, 1.0f);
	ck(l0 > 0, "forward + backward runs");
	float *g = malloc(np * sizeof(float)), *dir = malloc(np * sizeof(float));
	memcpy(g, tr_grads(m), np * sizeof(float));
	/* tensor spans by the layout tr_new uses */
	int D = 16, Q = 32, KV = 16, F = 24, V = 320;
	size_t span[2 + 18], at = 0; int ns = 0;
	span[ns++] = (size_t)V * D;
	for (int l = 0; l < 2; l++) {
		size_t s[9] = { D, (size_t)Q * D, (size_t)KV * D, (size_t)KV * D, (size_t)D * Q, D, (size_t)F * D, (size_t)F * D, (size_t)D * F };
		for (int i = 0; i < 9; i++) span[ns++] = s[i];
	}
	span[ns++] = D;
	double worst = 0;
	for (int t = 0; t < ns; t++) {
		memset(dir, 0, np * sizeof(float));
		double an = 0;
		for (size_t i = at; i < at + span[t]; i++) { dir[i] = frand(); an += (double)dir[i] * g[i]; }
		const double h = 1e-3;
		p = tr_params(m);
		for (size_t i = at; i < at + span[t]; i++) p[i] += (float)h * dir[i];
		double lp = tr_run(m, in, tg, B, T, 0);
		p = tr_params(m);
		for (size_t i = at; i < at + span[t]; i++) p[i] -= 2 * (float)h * dir[i];
		double lm = tr_run(m, in, tg, B, T, 0);
		p = tr_params(m);
		for (size_t i = at; i < at + span[t]; i++) p[i] += (float)h * dir[i];
		double fd = (lp - lm) / (2 * h), rel = fabs(fd - an) / (fabs(an) + 1e-3);
		if (rel > worst) worst = rel;
		if (rel > 2e-2) printf("    tensor %d: analytic %.6f finite %.6f\n", t, an, fd);
		at += span[t];
	}
	printf("    worst relative error %.2e over %d tensors\n", worst, ns);
	ck(worst < 2e-2, "every tensor's gradient matches its finite difference");

	/* ── the engine scores the saved file as the trainer does ── */
	const char *path = "build/test_train.mlf2";
	ck(tr_save(m, path, blob, blen, eos) == 0, "save as MLF2");
	hakm_ctx *e = hakm_ctx_new(path, c.ctx + 1);
	ck(e != NULL, "hakm loads it (arch hako, f32)");
	int seq[T + 1];
	for (int i = 0; i <= T; i++) seq[i] = ids[100 + i];
	double eng = e ? hakm_score_batch(e, seq, T + 1, 0) : -1;
	double trn = tr_run(m, seq, seq + 1, 1, T, 0);
	printf("    nll: trainer %.6f engine %.6f\n", trn, eng);
	ck(e && fabs(eng - trn) < 2e-3 * trn, "same NLL within the f16 KV cache's rounding");
	hakm_ctx_free(e);

	/* ── checkpoints ── */
	tr_adamw(m, 1e-3f, 0.1f, 0.9f, 0.95f, 1e-8f);
	ck(tr_save(m, path, blob, blen, eos) == 0 && tr_save_opt(m, "build/test_train.opt", 99) == 0, "save weights + optimizer");
	void *b2; size_t bl2; int eos2; uint64_t rng = 0;
	tr_model *r = tr_load(path, &b2, &bl2, &eos2);
	ck(r && !memcmp(tr_params(r), tr_params(m), np * sizeof(float)), "weights come back bit for bit");
	ck(r && bl2 == blen && !memcmp(b2, blob, blen) && eos2 == eos, "tokenizer and eos come back");
	ck(r && tr_load_opt(r, "build/test_train.opt", &rng) == 0 && *tr_step(r) == 1 && rng == 99, "optimizer state comes back");
	if (r) { free(b2); tr_free(r); }
	remove(path); remove("build/test_train.opt");

	/* ── it learns: one batch, many steps ── */
	tr_free(m);
	m = tr_new(&c, 5);
	double first = 0, last = 0;
	for (int s = 0; s < 60; s++) {
		tr_zero_grad(m);
		double l = tr_run(m, in, tg, B, T, 1.0f / (B * T));
		tr_clip(m, 1.0);
		tr_adamw(m, 1e-2f, 0.0f, 0.9f, 0.95f, 1e-8f);
		if (!s) first = l; last = l;
	}
	printf("    loss %.3f -> %.3f per token\n", first / (B * T), last / (B * T));
	ck(last < 0.2 * first, "60 steps on one batch cut its loss by 5x");
	tr_free(m);

	/* ── experts ── */
	opt_versions();
	/* seeds whose picks also hold at 5x the differences' step: random weights
	   put some token near a tie in most draws, and a tie that moves is a jump */
	moe_case(4, 2, 1, 8, ids, blob, blen, eos);
	moe_case(5, 3, 0, 13, ids, blob, blen, eos);
	moe_dense_equal(ids);
	moe_aux_moves(ids);

	{	/* frozen ranges: tr_adamw leaves their weights and their moments alone */
		tr_config c = { 320, 16, 2, 4, 2, 8, 24, 12, 10000.0f, 1e-5f, 0, 0, 0 };
		tr_model *a = tr_new(&c, 3), *b = tr_new(&c, 3);
		size_t np = tr_n_params(a), lo = np / 3, len = np / 4, r[2] = { lo, len }, past[2] = { np - 1, 2 };
		float *w0 = malloc(np * 4);
		float *ga = tr_grads(a), *gb = tr_grads(b);
		for (size_t j = 0; j < np; j++) ga[j] = gb[j] = (float)(j % 11) * 0.01f - 0.04f;
		tr_adamw(a, 1e-2f, 0.0f, 0.9f, 0.95f, 1e-8f); tr_adamw(b, 1e-2f, 0.0f, 0.9f, 0.95f, 1e-8f);   /* both warm everywhere */
		memcpy(w0, tr_params(a), np * 4);
		ck(tr_set_live(a, past, 1) != 0, "a live range past the end is refused");
		ck(tr_set_live(a, r, 1) == 0, "one live range set");
		for (int s = 0; s < 20; s++) {   /* a: 20 steps with the rest frozen; b: none */
			for (size_t j = 0; j < np; j++) ga[j] = j >= lo && j < lo + len ? 0.02f : 0.5f;
			tr_adamw(a, 1e-2f, 0.0f, 0.9f, 0.95f, 1e-8f);
		}
		const float *pa = tr_params(a);
		int frozen = !memcmp(pa, w0, lo * 4) && !memcmp(pa + lo + len, w0 + lo + len, (np - lo - len) * 4), live = 0;
		for (size_t j = lo; j < lo + len; j++) live += pa[j] != w0[j];
		ck(frozen && live == (int)len, "frozen weights stay bit for bit under nonzero grads, live ones move");
		/* the frozen moments are what they were: thawed, a and b take the same step there */
		ck(tr_set_live(a, NULL, 0) == 0, "thawed");
		*tr_step(b) = *tr_step(a);
		tr_zero_grad(a); tr_zero_grad(b);
		tr_adamw(a, 1e-2f, 0.0f, 0.9f, 0.95f, 1e-8f); tr_adamw(b, 1e-2f, 0.0f, 0.9f, 0.95f, 1e-8f);
		pa = tr_params(a);
		const float *pb = tr_params(b);
		int same = 1, movedf = 0;
		for (size_t j = 0; j < np; j++) if (j < lo || j >= lo + len) { same &= pa[j] - w0[j] == pb[j] - w0[j]; movedf += pa[j] != w0[j]; }
		ck(same && movedf > (int)(np / 2), "thawed, the frozen weights step exactly as if those 20 steps never ran: their moments were not decayed");
		free(w0); tr_free(a); tr_free(b);
	}

	tok_free(tk);
	free(blob); free(ids); free(back); free(text); free(g); free(dir);
	printf(fails ? "test_train: %d FAILED\n" : "test_train: all ok\n", fails);
	return fails != 0;
}
