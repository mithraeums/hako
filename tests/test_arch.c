/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* test_arch.c — the forward pass against a plain reference, on a tiny random
   model shaped like Qwen3: attention width n_heads*head_dim != d_model, and
   per-head q/k norm. The engine sized q and the attention output by d_model,
   so this shape overflowed both buffers; under ASan the old code fails here. */
#include "../src/model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
static void ck(int c, const char *w) { printf("  %s %s\n", c ? "ok  " : "FAIL", w); if (!c) fails++; }

enum { D = 32, H = 4, KVH = 2, HD = 16, QD = H * HD, KVD = KVH * HD, F = 48, V = 24, NL = 2 };
static const float EPS = 1e-6f, THETA = 10000.0f;

typedef struct { char name[48]; int n_in, n_out; float *w; } wt;
static wt W[64];
static int nw = 0;
static unsigned rng = 12345;
static float frand(void) { rng = rng * 1103515245u + 12345u; return ((rng >> 8) & 0xFFFF) / 65536.0f - 0.5f; }

static float *add(const char *name, int n_in, int n_out, int ones) {
	wt *t = &W[nw++];
	snprintf(t->name, sizeof t->name, "%s", name);
	t->n_in = n_in; t->n_out = n_out;
	t->w = malloc(sizeof(float) * n_in * n_out);
	for (int i = 0; i < n_in * n_out; i++) t->w[i] = ones ? 1.0f + 0.2f * frand() : frand();
	return t->w;
}
static float *get(const char *name) {
	for (int i = 0; i < nw; i++) if (!strcmp(W[i].name, name)) return W[i].w;
	return NULL;
}

static void build(int qk_norm, int only_q_norm) {
	for (int i = 0; i < nw; i++) free(W[i].w);
	nw = 0; rng = 12345;
	char nm[48];
	add("token_embd.weight", D, V, 0);
	add("output_norm.weight", D, 1, 1);
	for (int l = 0; l < NL; l++) {
		#define A(s, i, o, one) snprintf(nm, sizeof nm, "blk.%d." s, l); add(nm, i, o, one);
		A("attn_norm.weight", D, 1, 1);
		A("attn_q.weight", D, QD, 0); A("attn_k.weight", D, KVD, 0);
		A("attn_v.weight", D, KVD, 0); A("attn_output.weight", QD, D, 0);
		if (qk_norm || only_q_norm) { A("attn_q_norm.weight", HD, 1, 1); }
		if (qk_norm) { A("attn_k_norm.weight", HD, 1, 1); }
		A("ffn_norm.weight", D, 1, 1);
		A("ffn_gate.weight", D, F, 0); A("ffn_up.weight", D, F, 0); A("ffn_down.weight", F, D, 0);
		#undef A
	}
}

static void write_model_f(const char *path, uint32_t arch, uint32_t flags) {
	mlf2_header h; memset(&h, 0, sizeof h);
	memcpy(h.magic, MLF2_MAGIC, 4);
	h.version = MLF2_VERSION; h.arch = arch;
	h.n_layers = NL; h.d_model = D; h.n_heads = H; h.n_kv_heads = KVH; h.head_dim = HD;
	h.ffn_dim = F; h.ctx = 128; h.vocab = V; h.rope_theta = THETA; h.rms_eps = EPS;
	h.flags = MLF2_FLAG_TIED_EMBED | flags; h.n_tensors = nw;
	h.tensors_off = sizeof h;
	h.tokenizer_off = h.tensors_off + (uint64_t)nw * sizeof(mlf2_tensor);
	h.data_off = (h.tokenizer_off + 8 + 31) / 32 * 32;
	FILE *f = fopen(path, "wb");
	fwrite(&h, sizeof h, 1, f);
	uint64_t off = 0;
	for (int i = 0; i < nw; i++) {
		mlf2_tensor t; memset(&t, 0, sizeof t);
		snprintf(t.name, sizeof t.name, "%s", W[i].name);
		t.type = MLF2_TYPE_F32;
		t.n_dims = W[i].n_out > 1 ? 2 : 1;
		t.dims[0] = W[i].n_in; t.dims[1] = W[i].n_out;
		t.data_off = off; t.nbytes = (uint64_t)W[i].n_in * W[i].n_out * 4;
		off += (t.nbytes + 31) / 32 * 32;
		fwrite(&t, sizeof t, 1, f);
	}
	uint32_t z[2] = { 0, 0 };
	fwrite(z, sizeof z, 1, f);
	for (long p = ftell(f); p < (long)h.data_off; p++) fputc(0, f);
	for (int i = 0; i < nw; i++) {
		size_t n = (size_t)W[i].n_in * W[i].n_out;
		fwrite(W[i].w, 4, n, f);
		for (size_t p = n * 4; p % 32; p++) fputc(0, f);
	}
	fclose(f);
}

static void write_model(const char *path, uint32_t arch) { write_model_f(path, arch, 0); }

/* ── reference: the textbook forward pass, whole sequence at once ── */
static void r_norm(float *y, const float *x, const float *w, int n) {
	double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
	float s = 1.0f / sqrtf((float)(ss / n) + EPS);
	for (int i = 0; i < n; i++) y[i] = x[i] * s * w[i];
}
static void r_mat(float *y, const float *x, const float *w, int n_in, int n_out) {
	for (int o = 0; o < n_out; o++) { double a = 0; for (int i = 0; i < n_in; i++) a += (double)w[o * n_in + i] * x[i]; y[o] = (float)a; }
}
static int r_norm_rope = 0;
static void r_rope(float *v, int pos) {
	if (r_norm_rope) {
		for (int i = 0; i < HD / 2; i++) {
			double ang = pos * pow(THETA, -(2.0 * i) / HD);
			float a = v[2 * i], b = v[2 * i + 1];
			v[2 * i] = (float)(a * cos(ang) - b * sin(ang));
			v[2 * i + 1] = (float)(a * sin(ang) + b * cos(ang));
		}
		return;
	}
	for (int i = 0; i < HD / 2; i++) {
		double ang = pos * pow(THETA, -(2.0 * i) / HD);
		float a = v[i], b = v[i + HD / 2];
		v[i] = (float)(a * cos(ang) - b * sin(ang));
		v[i + HD / 2] = (float)(a * sin(ang) + b * cos(ang));
	}
}
static void reference(const int *toks, int n, float *logits) {
	static float x[16][D], xn[D], q[16][QD], k[16][KVD], v[16][KVD], att[QD], o[D], g[F], u[F];
	char nm[48];
	for (int t = 0; t < n; t++) memcpy(x[t], get("token_embd.weight") + toks[t] * D, sizeof x[t]);
	for (int l = 0; l < NL; l++) {
		#define G(s) (snprintf(nm, sizeof nm, "blk.%d." s, l), get(nm))
		const float *qn = G("attn_q_norm.weight"), *kn = G("attn_k_norm.weight");
		for (int t = 0; t < n; t++) {
			r_norm(xn, x[t], G("attn_norm.weight"), D);
			r_mat(q[t], xn, G("attn_q.weight"), D, QD);
			r_mat(k[t], xn, G("attn_k.weight"), D, KVD);
			r_mat(v[t], xn, G("attn_v.weight"), D, KVD);
			for (int h = 0; h < H; h++) { if (qn) r_norm(q[t] + h * HD, q[t] + h * HD, qn, HD); r_rope(q[t] + h * HD, t); }
			for (int h = 0; h < KVH; h++) { if (kn) r_norm(k[t] + h * HD, k[t] + h * HD, kn, HD); r_rope(k[t] + h * HD, t); }
		}
		for (int t = 0; t < n; t++) {
			for (int h = 0; h < H; h++) {
				int kh = h / (H / KVH);
				double s[16], mx = -1e30, sum = 0;
				for (int p = 0; p <= t; p++) {
					double d = 0; for (int i = 0; i < HD; i++) d += (double)q[t][h * HD + i] * k[p][kh * HD + i];
					s[p] = d / sqrt((double)HD); if (s[p] > mx) mx = s[p];
				}
				for (int p = 0; p <= t; p++) { s[p] = exp(s[p] - mx); sum += s[p]; }
				for (int i = 0; i < HD; i++) {
					double a = 0; for (int p = 0; p <= t; p++) a += s[p] / sum * v[p][kh * HD + i];
					att[h * HD + i] = (float)a;
				}
			}
			r_mat(o, att, G("attn_output.weight"), QD, D);
			for (int i = 0; i < D; i++) x[t][i] += o[i];
			r_norm(xn, x[t], G("ffn_norm.weight"), D);
			r_mat(g, xn, G("ffn_gate.weight"), D, F);
			r_mat(u, xn, G("ffn_up.weight"), D, F);
			for (int i = 0; i < F; i++) g[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
			r_mat(o, g, G("ffn_down.weight"), F, D);
			for (int i = 0; i < D; i++) x[t][i] += o[i];
		}
		#undef G
	}
	r_norm(xn, x[n - 1], get("output_norm.weight"), D);
	r_mat(logits, xn, get("token_embd.weight"), D, V);
}

static float maxdiff(const float *a, const float *b, int n) {
	float m = 0; for (int i = 0; i < n; i++) { float d = fabsf(a[i] - b[i]); if (d > m) m = d; }
	return m;
}

int main(void) {
	const char *p = "/tmp/hakm-test-arch.mlf2";
	const int toks[] = { 3, 17, 5, 22, 9, 1, 14 };
	const int n = (int)(sizeof toks / sizeof *toks);
	float ref[V], seq[V];

	build(1, 0);
	write_model(p, MLF2_ARCH_QWEN3);
	reference(toks, n, ref);

	hakm_ctx *c = hakm_ctx_new(p, 64);
	ck(c != NULL, "a model with q_dim (64) != d_model (32) and q/k norm loads");
	if (c) {
		ck(c->q_dim == QD && c->layers[0].q_norm && c->layers[0].k_norm, "the shape and the norms are picked up");
		const float *lg = NULL;
		for (int i = 0; i < n; i++) lg = hakm_forward(c, toks[i]);
		memcpy(seq, lg, sizeof seq);
		ck(maxdiff(seq, ref, V) < 1e-3f, "token-at-a-time decode matches the reference");
		hakm_reset(c);
		lg = hakm_forward_batch(c, toks, n);
		ck(lg && maxdiff(lg, ref, V) < 1e-3f, "batched prefill matches the reference");
		/* perplexity scoring: Σ -log p(tok[i+1] | tok[0..i]) from the reference's logits */
		double want = 0;
		for (int i = 2; i < n - 1; i++) {
			float r[V]; reference(toks, i + 1, r);
			double mx = r[0], se = 0;
			for (int v = 1; v < V; v++) if (r[v] > mx) mx = r[v];
			for (int v = 0; v < V; v++) se += exp(r[v] - mx);
			want += log(se) + mx - r[toks[i + 1]];
		}
		hakm_reset(c);
		double got = hakm_score_batch(c, toks, n, 2);
		ck(fabs(got - want) < 1e-3 * (n - 3), "hakm_score_batch (perplexity) matches the reference log-likelihood");
		hakm_ctx_free(c);
	}

	/* the norm is load-bearing: the same weights without it give other logits */
	build(0, 0);
	write_model(p, MLF2_ARCH_QWEN3);
	c = hakm_ctx_new(p, 64);
	if (c) {
		const float *lg = hakm_forward_batch(c, toks, n);
		ck(lg && maxdiff(lg, ref, V) > 1e-2f, "without q/k norm tensors the result differs (the test can fail)");
		float r2[V]; reference(toks, n, r2);
		ck(lg && maxdiff(lg, r2, V) < 1e-3f, "and a model with no q/k norm still matches its own reference");
		hakm_ctx_free(c);
	}

	/* our own arch: every feature from the file. ROPE_NORM must switch the pairing */
	build(1, 0);
	write_model_f(p, MLF2_ARCH_HAKO, MLF2_FLAG_ROPE_NORM);
	r_norm_rope = 1; float rn[V]; reference(toks, n, rn); r_norm_rope = 0;
	c = hakm_ctx_new(p, 64);
	ck(c != NULL, "arch hako with the rope flag loads");
	if (c) {
		const float *lg = hakm_forward_batch(c, toks, n);
		ck(lg && maxdiff(lg, rn, V) < 1e-3f, "and runs (2i, 2i+1) rope, matching the reference");
		ck(lg && maxdiff(lg, ref, V) > 1e-2f, "which differs from neox rope (the flag is load-bearing)");
		hakm_ctx_free(c);
	}
	write_model_f(p, MLF2_ARCH_HAKO, 1u << 31);
	ck(hakm_ctx_new(p, 64) == NULL, "a feature flag this engine does not know is refused");

	build(0, 1);
	write_model(p, MLF2_ARCH_QWEN3);
	ck(hakm_ctx_new(p, 64) == NULL, "a q norm without its k norm is refused");

	build(1, 0);
	write_model(p, MLF2_ARCH_COUNT);
	ck(hakm_ctx_new(p, 64) == NULL, "an arch this engine does not know is refused, not run as qwen2");

	for (int i = 0; i < nw; i++) free(W[i].w);
	remove(p);
	printf(fails ? "test_arch: %d FAILED\n" : "test_arch: all passed\n", fails);
	return fails ? 1 : 0;
}
