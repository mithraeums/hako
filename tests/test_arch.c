/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* test_arch.c — the forward pass against a plain reference, on a tiny random
   model shaped like Qwen3: attention width n_heads*head_dim != d_model, and
   per-head q/k norm. The engine sized q and the attention output by d_model,
   so this shape overflowed both buffers; under ASan the old code fails here.
   then routed experts: a reference that ranks and mixes the experts itself,
   against decode, batch and score, with and without the shared expert. */
#include "../src/model.h"
#include "../src/nn.h"
#include <math.h>
#if defined(HAKM_KV_F16) && defined(__F16C__) && defined(__FMA__)
#include <immintrin.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
static void ck(int c, const char *w) { printf("  %s %s\n", c ? "ok  " : "FAIL", w); if (!c) fails++; }

enum { D = 32, H = 4, KVH = 2, HD = 16, QD = H * HD, KVD = KVH * HD, F = 48, V = 24, NL = 2 };
static const float EPS = 1e-6f, THETA = 10000.0f;

typedef struct { char name[48]; int n_in, n_out; float *w; } wt;
static wt W[128];
static int nw = 0;
static unsigned rng = 12345;
static float frand(void) { rng = rng * 1103515245u + 12345u; return ((rng >> 8) & 0xFFFF) / 65536.0f - 0.5f; }

/* the moe shape build() makes: experts per layer (0 = dense), top-k, whether
   the dense ffn is there as the shared expert, a tensor to leave out, and
   whether one expert's gate gets the wrong height */
static int m_experts = 0, m_used = 0, m_shared = 1, m_bad = 0;
static const char *m_drop = NULL;

static float *add(const char *name, int n_in, int n_out, int ones) {
	if (m_drop && !strcmp(name, m_drop)) return NULL;
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
		if (m_shared) { A("ffn_gate.weight", D, F, 0); A("ffn_up.weight", D, F, 0); A("ffn_down.weight", F, D, 0); }
		if (m_experts) { A("ffn_gate_inp.weight", D, m_experts, 0); }
		#undef A
		for (int e = 0; e < m_experts; e++) {
			snprintf(nm, sizeof nm, "blk.%d.ffn_gate.%d.weight", l, e); add(nm, D, m_bad && l == 1 && e == 2 ? F / 2 : F, 0);
			snprintf(nm, sizeof nm, "blk.%d.ffn_up.%d.weight", l, e);   add(nm, D, F, 0);
			snprintf(nm, sizeof nm, "blk.%d.ffn_down.%d.weight", l, e); add(nm, F, D, 0);
		}
	}
}

static void write_model_x(const char *path, uint32_t arch, uint32_t flags, uint32_t ne, uint32_t nu) {
	mlf2_header h; memset(&h, 0, sizeof h);
	memcpy(h.magic, MLF2_MAGIC, 4);
	h.version = MLF2_VERSION; h.arch = arch;
	h.n_layers = NL; h.d_model = D; h.n_heads = H; h.n_kv_heads = KVH; h.head_dim = HD;
	h.ffn_dim = F; h.ctx = 128; h.vocab = V; h.rope_theta = THETA; h.rms_eps = EPS;
	h.flags = MLF2_FLAG_TIED_EMBED | flags; h.n_tensors = nw;
	h.n_experts = ne; h.n_experts_used = nu;
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

static void write_model_f(const char *path, uint32_t arch, uint32_t flags) { write_model_x(path, arch, flags, 0, 0); }
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
/* the engine keeps K and V in f16 where the hardware converts (model.h); the
   reference rounds them the same way, so the tolerance stays tight on every
   build. without this the test failed on arm64 and on F16C x86 */
static void r_kv(float *v, int n) {
#ifdef HAKM_KV_F16
#if defined(__F16C__) && defined(__FMA__)
	for (int i = 0; i < n; i++) v[i] = _cvtsh_ss(_cvtss_sh(v[i], 0));
#else
	for (int i = 0; i < n; i++) v[i] = (float)(__fp16)v[i];
#endif
#else
	(void)v; (void)n;
#endif
}
/* routed experts the long way: score every expert, rank them all with a stable
   insertion sort (a tie keeps the lower index first), softmax the top k in
   double, add each one's SwiGLU times its gate. records which sets ran and
   whether any k-th/k+1-th pair was close enough to flip on rounding */
static int r_sets[256], r_close = 0;
static void r_moe(float *x, const float *xn, int l) {
	enum { EM = 8 };
	char nm[48];
	float z[EM], g[F], u[F], o[D];
	int ord[EM];
	snprintf(nm, sizeof nm, "blk.%d.ffn_gate_inp.weight", l);
	r_mat(z, xn, get(nm), D, m_experts);
	for (int e = 0; e < m_experts; e++) {
		int j = e;
		while (j > 0 && z[ord[j - 1]] < z[e]) { ord[j] = ord[j - 1]; j--; }
		ord[j] = e;
	}
	double gw[EM], sum = 0;
	for (int s = 0; s < m_used; s++) { gw[s] = exp((double)z[ord[s]] - z[ord[0]]); sum += gw[s]; }
	unsigned set = 0;
	for (int s = 0; s < m_used; s++) {
		int e = ord[s];
		set |= 1u << e;
		snprintf(nm, sizeof nm, "blk.%d.ffn_gate.%d.weight", l, e); r_mat(g, xn, get(nm), D, F);
		snprintf(nm, sizeof nm, "blk.%d.ffn_up.%d.weight", l, e);   r_mat(u, xn, get(nm), D, F);
		for (int i = 0; i < F; i++) g[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
		snprintf(nm, sizeof nm, "blk.%d.ffn_down.%d.weight", l, e); r_mat(o, g, get(nm), F, D);
		for (int i = 0; i < D; i++) x[i] += (float)(gw[s] / sum) * o[i];
	}
	r_sets[set & 255] = 1;
	if (m_used < m_experts && z[ord[m_used - 1]] - z[ord[m_used]] < 1e-3f) r_close++;
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
			r_kv(k[t], KVD); r_kv(v[t], KVD);
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
			if (G("ffn_gate.weight")) {
				r_mat(g, xn, G("ffn_gate.weight"), D, F);
				r_mat(u, xn, G("ffn_up.weight"), D, F);
				for (int i = 0; i < F; i++) g[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
				r_mat(o, g, G("ffn_down.weight"), F, D);
				for (int i = 0; i < D; i++) x[t][i] += o[i];
			}
			if (m_experts) r_moe(x[t], xn, l);
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

/* perplexity scoring: Σ -log p(tok[i+1] | tok[0..i]) from the reference's logits */
static double ref_nll(const int *toks, int n, int from) {
	double want = 0;
	for (int i = from; i < n - 1; i++) {
		float r[V]; reference(toks, i + 1, r);
		double mx = r[0], se = 0;
		for (int v = 1; v < V; v++) if (r[v] > mx) mx = r[v];
		for (int v = 0; v < V; v++) se += exp(r[v] - mx);
		want += log(se) + mx - r[toks[i + 1]];
	}
	return want;
}

/* one moe file (4 experts) through decode, batch and score against the
   reference; leaves the batch logits in got[] */
static void moe_case(const char *p, const int *toks, int n, int shared, int used, float *got) {
	char w[160];
	m_experts = 4; m_used = used; m_shared = shared;
	build(1, 0);
	write_model_x(p, MLF2_ARCH_HAKO, MLF2_FLAG_MOE, 4, used);
	memset(r_sets, 0, sizeof r_sets); r_close = 0;
	float ref[V]; reference(toks, n, ref);
	int sets = 0;
	for (int i = 0; i < 256; i++) sets += r_sets[i];
	snprintf(w, sizeof w, "moe shared=%d k=%d: routes vary across tokens (%d sets) and none is a near tie", shared, used, sets);
	ck(sets >= 3 && !r_close, w);
	hakm_ctx *c = hakm_ctx_new(p, 64);
	snprintf(w, sizeof w, "moe shared=%d k=%d: loads, the expert count and the shared ffn picked up", shared, used);
	ck(c && c->n_experts == 4 && c->n_used == used && !c->layers[1].gate == !shared, w);
	memset(got, 0, V * sizeof *got);
	if (!c) return;
	const float *lg = NULL;
	for (int i = 0; i < n; i++) lg = hakm_forward(c, toks[i]);
	snprintf(w, sizeof w, "moe shared=%d k=%d: token-at-a-time decode matches the reference", shared, used);
	ck(maxdiff(lg, ref, V) < 1e-3f, w);
	hakm_reset(c);
	lg = hakm_forward_batch(c, toks, n);
	snprintf(w, sizeof w, "moe shared=%d k=%d: batched prefill matches the reference", shared, used);
	ck(lg && maxdiff(lg, ref, V) < 1e-3f, w);
	if (lg) memcpy(got, lg, V * sizeof *got);
	hakm_reset(c);
	double s = hakm_score_batch(c, toks, n, 2), want = ref_nll(toks, n, 2);
	snprintf(w, sizeof w, "moe shared=%d k=%d: hakm_score_batch matches the reference log-likelihood", shared, used);
	ck(fabs(s - want) < 1e-3 * (n - 3), w);
	hakm_ctx_free(c);
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
		double want = ref_nll(toks, n, 2);
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

	/* routing: rank order, a tie to the lower index, softmax over just the k */
	{
		const float z[4] = { 1, 3, 3, 0 }, flat[4] = { 2, 2, 2, 2 };
		int sel[3]; float gw[3];
		topk_gates(z, 4, 3, sel, gw);
		ck(sel[0] == 1 && sel[1] == 2 && sel[2] == 0, "topk_gates ranks by score, a tie to the lower index");
		ck(fabsf(gw[0] - gw[1]) < 1e-6f && fabsf(gw[0] + gw[1] + gw[2] - 1) < 1e-6f &&
		   fabsf(gw[0] / gw[2] - expf(2)) < 1e-4f, "and its gates are a softmax over the picked scores only");
		topk_gates(flat, 4, 2, sel, gw);
		ck(sel[0] == 0 && sel[1] == 1 && gw[0] == 0.5f && gw[1] == 0.5f, "all tied: the lowest indices, equal gates");
	}

	/* routed experts, with the shared expert and without, and k as a real knob */
	float k2[V], k3[V], k2s0[V];
	moe_case(p, toks, n, 1, 2, k2);
	moe_case(p, toks, n, 1, 3, k3);
	ck(maxdiff(k2, k3, V) > 1e-2f, "n_experts_used is load-bearing: the same weights at k=3 give other logits");
	moe_case(p, toks, n, 0, 2, k2s0);

	/* each file below is the k=2 one with exactly one thing wrong */
	m_shared = 1; m_used = 2;
	build(1, 0);
	write_model_x(p, MLF2_ARCH_HAKO, MLF2_FLAG_MOE, 0, 0);
	ck(hakm_ctx_new(p, 64) == NULL, "the moe flag without an expert count is refused");
	write_model_x(p, MLF2_ARCH_HAKO, 0, 4, 2);
	ck(hakm_ctx_new(p, 64) == NULL, "an expert count without the moe flag is refused");
	write_model_x(p, MLF2_ARCH_HAKO, MLF2_FLAG_MOE, 4, 5);
	ck(hakm_ctx_new(p, 64) == NULL, "n_experts_used > n_experts is refused");
	write_model_x(p, MLF2_ARCH_HAKO, MLF2_FLAG_MOE, 4, 0);
	ck(hakm_ctx_new(p, 64) == NULL, "n_experts_used = 0 is refused");
	m_drop = "blk.1.ffn_down.3.weight"; build(1, 0);
	write_model_x(p, MLF2_ARCH_HAKO, MLF2_FLAG_MOE, 4, 2);
	ck(hakm_ctx_new(p, 64) == NULL, "a missing expert tensor is refused");
	m_drop = "blk.0.ffn_gate_inp.weight"; build(1, 0);
	write_model_x(p, MLF2_ARCH_HAKO, MLF2_FLAG_MOE, 4, 2);
	ck(hakm_ctx_new(p, 64) == NULL, "a missing router is refused");
	m_drop = "blk.0.ffn_up.weight"; build(1, 0);
	write_model_x(p, MLF2_ARCH_HAKO, MLF2_FLAG_MOE, 4, 2);
	ck(hakm_ctx_new(p, 64) == NULL, "a shared ffn missing one of its three tensors is refused");
	m_drop = NULL; m_bad = 1; build(1, 0);
	write_model_x(p, MLF2_ARCH_HAKO, MLF2_FLAG_MOE, 4, 2);
	ck(hakm_ctx_new(p, 64) == NULL, "an expert of the wrong shape is refused (it would overrun the scratch)");
	m_bad = 0; build(1, 0);
	write_model_x(p, MLF2_ARCH_HAKO, MLF2_FLAG_MOE, 4, 2);
	hakm_ctx *c2 = hakm_ctx_new(p, 64);
	ck(c2 != NULL, "and the same file put right loads (the refusals were for the fault)");
	hakm_ctx_free(c2);
	m_experts = 0; m_used = 0;

	for (int i = 0; i < nw; i++) free(W[i].w);
	remove(p);
	printf(fails ? "test_arch: %d FAILED\n" : "test_arch: all passed\n", fails);
	return fails ? 1 : 0;
}
