/* model.c — Qwen2 incremental forward pass with KV cache. */
#include "model.h"
#include "nn.h"
#include "../include/mlf2.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const mlf2_tensor *need(const hakm_model *m, const char *name) {
	const mlf2_tensor *t = mlf2_find(m, name);
	if (!t) { fprintf(stderr, "hakm: missing tensor %s\n", name); }
	return t;
}

hakm_ctx *hakm_ctx_new(const char *path, int max_seq) {
	hakm_ctx *c = calloc(1, sizeof(*c));
	if (!c) return NULL;
	if (mlf2_open(path, &c->model) != 0) { free(c); return NULL; }

	const mlf2_header *h = &c->model.h;
	c->d_model = h->d_model; c->n_heads = h->n_heads;
	c->n_kv_heads = h->n_kv_heads; c->head_dim = h->head_dim;
	c->kv_dim = h->n_kv_heads * h->head_dim;
	c->ffn_dim = h->ffn_dim; c->n_layers = h->n_layers; c->vocab = h->vocab;
	c->rope_theta = h->rope_theta; c->rms_eps = h->rms_eps;
	c->max_seq = max_seq; c->pos = 0;

	c->tok_embd = need(&c->model, "token_embd.weight");
	c->out_norm = need(&c->model, "output_norm.weight");
	/* lm_head: small Qwen tiers tie it to token_embd (TIED_EMBED flag); 7B+ ship a
	   separate output.weight. Using tok_embd for an untied model = garbage logits. */
	if (h->flags & MLF2_FLAG_TIED_EMBED)
		c->out_weight = c->tok_embd;
	else
		c->out_weight = need(&c->model, "output.weight");

	c->layers = calloc(c->n_layers, sizeof(layer_t));
	if (!c->layers) { hakm_ctx_free(c); return NULL; }
	int ok = (c->tok_embd && c->out_norm && c->out_weight);
	char nm[64];
	for (int l = 0; l < c->n_layers; l++) {
		layer_t *L = &c->layers[l];
		#define T(field, suffix) \
			snprintf(nm, sizeof nm, "blk.%d." suffix, l); \
			L->field = need(&c->model, nm); ok &= (L->field != NULL);
		T(attn_norm, "attn_norm.weight");
		T(wq, "attn_q.weight"); T(wk, "attn_k.weight");
		T(wv, "attn_v.weight"); T(wo, "attn_output.weight");
		T(ffn_norm, "ffn_norm.weight");
		T(gate, "ffn_gate.weight"); T(up, "ffn_up.weight"); T(down, "ffn_down.weight");
		#undef T
		/* biases are optional (present for Qwen2) and not error-checked */
		snprintf(nm, sizeof nm, "blk.%d.attn_q.bias", l); L->bq = mlf2_find(&c->model, nm);
		snprintf(nm, sizeof nm, "blk.%d.attn_k.bias", l); L->bk = mlf2_find(&c->model, nm);
		snprintf(nm, sizeof nm, "blk.%d.attn_v.bias", l); L->bv = mlf2_find(&c->model, nm);
	}
	if (!ok) { hakm_ctx_free(c); return NULL; }

	int D = c->d_model, F = c->ffn_dim, KV = c->kv_dim, V = c->vocab;
	int scratch_max = F > D ? F : D;
	c->xb = calloc(D, sizeof(float));
	c->xn = calloc(D, sizeof(float));
	c->q  = calloc(D, sizeof(float));
	c->k  = calloc(KV, sizeof(float));
	c->v  = calloc(KV, sizeof(float));
	c->att_out = calloc(D, sizeof(float));
	c->gate_buf = calloc(F, sizeof(float));
	c->up_buf   = calloc(F, sizeof(float));
	c->deq      = calloc(scratch_max, sizeof(float));
	c->logits   = calloc(V, sizeof(float));
	c->k_cache  = calloc((size_t)c->n_layers * max_seq * KV, sizeof(hakm_kv_t));
	c->v_cache  = calloc((size_t)c->n_layers * max_seq * KV, sizeof(hakm_kv_t));
	c->att_scores = calloc((size_t)hakm_nthreads() * max_seq, sizeof(float));
	if (!c->xb || !c->xn || !c->q || !c->k || !c->v || !c->att_out ||
	    !c->gate_buf || !c->up_buf || !c->deq ||
	    !c->k_cache || !c->v_cache || !c->logits || !c->att_scores) { hakm_ctx_free(c); return NULL; }
	return c;
}

void hakm_ctx_free(hakm_ctx *c) {
	if (!c) return;
	free(c->layers); free(c->xb); free(c->xn); free(c->q); free(c->k); free(c->v);
	free(c->att_out); free(c->gate_buf); free(c->up_buf); free(c->deq);
	free(c->logits); free(c->k_cache); free(c->v_cache);
	free(c->bxb); free(c->bxn); free(c->bq); free(c->bk); free(c->bv);
	free(c->batt); free(c->bgate); free(c->bup);
	free(c->att_scores);
	mlf2_close(&c->model);
	free(c);
}

void hakm_reset(hakm_ctx *c) { c->pos = 0; }

static const float *fdata(const hakm_ctx *c, const mlf2_tensor *t) {
	return (const float *)mlf2_data(&c->model, t);
}



/* ── KV cache element access (f16 or f32 per HAKM_KV_F16) ── */
#ifdef HAKM_KV_F16
#if defined(__F16C__) && defined(__FMA__)
#include <immintrin.h>
static inline void kv_store(hakm_kv_t *dst, const float *src, int n) {
	for (int d = 0; d < n; d += 8)
		_mm_storeu_si128((__m128i *)(dst + d),
			_mm256_cvtps_ph(_mm256_loadu_ps(src + d), _MM_FROUND_TO_NEAREST_INT));
}
static inline float kv_dot(const float *q, const hakm_kv_t *k, int n) {
	__m256 acc = _mm256_setzero_ps();
	for (int d = 0; d < n; d += 8)
		acc = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k + d))),
		                      _mm256_loadu_ps(q + d), acc);
	__m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
	lo = _mm_hadd_ps(lo, lo);
	lo = _mm_hadd_ps(lo, lo);
	return _mm_cvtss_f32(lo);
}
static inline void kv_axpy(float w, const hakm_kv_t *v, float *out, int n) {
	const __m256 wv = _mm256_set1_ps(w);
	for (int d = 0; d < n; d += 8) {
		__m256 o = _mm256_loadu_ps(out + d);
		o = _mm256_fmadd_ps(wv, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v + d))), o);
		_mm256_storeu_ps(out + d, o);
	}
}
#else	/* aarch64: __fp16 converts in hardware */
static inline void kv_store(hakm_kv_t *dst, const float *src, int n) {
	__fp16 *h = (__fp16 *)dst;
	for (int d = 0; d < n; d++) h[d] = (__fp16)src[d];
}
static inline float kv_dot(const float *q, const hakm_kv_t *k, int n) {
	const __fp16 *h = (const __fp16 *)k;
	float acc = 0.0f;
	for (int d = 0; d < n; d++) acc += q[d] * (float)h[d];
	return acc;
}
static inline void kv_axpy(float w, const hakm_kv_t *v, float *out, int n) {
	const __fp16 *h = (const __fp16 *)v;
	for (int d = 0; d < n; d++) out[d] += w * (float)h[d];
}
#endif
#else	/* f32 cache */
static inline void kv_store(hakm_kv_t *dst, const float *src, int n) {
	memcpy(dst, src, (size_t)n * sizeof(float));
}
static inline float kv_dot(const float *q, const hakm_kv_t *k, int n) {
	float acc = 0.0f;
	for (int d = 0; d < n; d++) acc += q[d] * k[d];
	return acc;
}
static inline void kv_axpy(float w, const hakm_kv_t *v, float *out, int n) {
	for (int d = 0; d < n; d++) out[d] += w * v[d];
}
#endif

/* ── threaded attention ──
 * One GQA attention pass for token-row range [b0,b1) of a batch (each row has
 * its own scores slice in bgate, so rows are independent), or for head range
 * [h0,h1) of a single decode token (each job gets its own att_scores slice).
 * Both run on the same worker pool the matmuls use. */
typedef struct {
	hakm_ctx *c;
	int l;            /* layer */
	int b0, b1;       /* batch rows (batch mode) */
	int h0, h1;       /* heads (decode mode) */
	int pos0;         /* batch: position of row 0; decode: token position */
	const float *q;   /* decode: q vector */
	float *att, *scores; /* decode: output + this job's scores slice */
} att_job;

static void att_heads(hakm_ctx *c, int l, int pos, const float *q,
                      int h0, int h1, float *att, float *scores) {
	int hd = c->head_dim, KV = c->kv_dim;
	int gsize = c->n_heads / c->n_kv_heads;
	float scale = 1.0f / sqrtf((float)hd);
	for (int h = h0; h < h1; h++) {
		int kvh = h / gsize;
		const float *q_h = q + h * hd;
		for (int s = 0; s <= pos; s++) {
			const hakm_kv_t *k_s = c->k_cache + ((size_t)l * c->max_seq + s) * KV + kvh * hd;
			scores[s] = kv_dot(q_h, k_s, hd) * scale;
		}
		softmax(scores, pos + 1);
		float *out_h = att + h * hd;
		for (int s = 0; s <= pos; s++) {
			const hakm_kv_t *v_s = c->v_cache + ((size_t)l * c->max_seq + s) * KV + kvh * hd;
			kv_axpy(scores[s], v_s, out_h, hd);
		}
	}
}

static void att_batch_run(void *arg) {
	att_job *j = (att_job *)arg;
	hakm_ctx *c = j->c;
	int D = c->d_model, F = c->ffn_dim;
	for (int b = j->b0; b < j->b1; b++) {
		float *att = c->batt + (size_t)b * D;
		memset(att, 0, D * sizeof(float));
		att_heads(c, j->l, j->pos0 + b, c->bq + (size_t)b * D,
		          0, c->n_heads, att, c->bgate + (size_t)b * F);
	}
}

static void att_decode_run(void *arg) {
	att_job *j = (att_job *)arg;
	att_heads(j->c, j->l, j->pos0, j->q, j->h0, j->h1, j->att, j->scores);
}

const float *hakm_forward(hakm_ctx *c, int token) {
	int D = c->d_model, H = c->n_heads, KVH = c->n_kv_heads, hd = c->head_dim;
	int KV = c->kv_dim, F = c->ffn_dim, pos = c->pos;

	/* 1. embedding lookup */
	deq_row(&c->model, c->tok_embd, token, D, c->xb);

	for (int l = 0; l < c->n_layers; l++) {
		layer_t *L = &c->layers[l];

		/* 2a. attention rmsnorm */
		rmsnorm(c->xn, c->xb, fdata(c, L->attn_norm), D, c->rms_eps);

		/* 2b. QKV projections (+bias) */
		matmul(c->q, c->xn, &c->model, L->wq, L->bq ? fdata(c, L->bq) : NULL, D,  D);
		matmul(c->k, c->xn, &c->model, L->wk, L->bk ? fdata(c, L->bk) : NULL, KV, D);
		matmul(c->v, c->xn, &c->model, L->wv, L->bv ? fdata(c, L->bv) : NULL, KV, D);

		/* 2c. rope (neox) on each head of q and k */
		for (int h = 0; h < H;   h++) rope_neox(c->q + h * hd, hd, pos, c->rope_theta);
		for (int h = 0; h < KVH; h++) rope_neox(c->k + h * hd, hd, pos, c->rope_theta);

		/* 2d. write k,v into cache at this position */
		kv_store(c->k_cache + ((size_t)l * c->max_seq + pos) * KV, c->k, KV);
		kv_store(c->v_cache + ((size_t)l * c->max_seq + pos) * KV, c->v, KV);

		/* 2e. GQA attention over positions 0..pos — threaded over heads once
		   the context is deep enough to beat the pool dispatch (~tens of µs) */
		memset(c->att_out, 0, D * sizeof(float));
		int T = hakm_nthreads();
		if (pos >= 64 && T > 1) {
			att_job aj[16];
			int per = (H + T - 1) / T, nj = 0;
			for (int t = 0; t < T; t++) {
				int h0 = t * per, h1 = h0 + per;
				if (h1 > H) h1 = H;
				if (h0 >= h1) break;
				aj[nj] = (att_job){ c, l, 0, 0, h0, h1, pos, c->q, c->att_out,
				                    c->att_scores + (size_t)nj * c->max_seq };
				nj++;
			}
			hakm_parallel(att_decode_run, aj, sizeof(att_job), nj);
		} else {
			att_heads(c, l, pos, c->q, 0, H, c->att_out, c->gate_buf);
		}

		/* 2f. output proj, residual */
		matmul(c->xn, c->att_out, &c->model, L->wo, NULL, D, D);
		for (int i = 0; i < D; i++) c->xb[i] += c->xn[i];

		/* 2g. ffn rmsnorm */
		rmsnorm(c->xn, c->xb, fdata(c, L->ffn_norm), D, c->rms_eps);

		/* 2h. SwiGLU */
		matmul(c->gate_buf, c->xn, &c->model, L->gate, NULL, F, D);
		matmul(c->up_buf,   c->xn, &c->model, L->up,   NULL, F, D);
		for (int i = 0; i < F; i++) c->gate_buf[i] = silu_f(c->gate_buf[i]) * c->up_buf[i];
		matmul(c->xn, c->gate_buf, &c->model, L->down, NULL, D, F);
		for (int i = 0; i < D; i++) c->xb[i] += c->xn[i];
	}

	/* 3. final norm + lm_head (out_weight == tok_embd when tied, else output.weight) */
	rmsnorm(c->xn, c->xb, fdata(c, c->out_norm), D, c->rms_eps);
	matmul(c->logits, c->xn, &c->model, c->out_weight, NULL, c->vocab, D);

	c->pos++;
	return c->logits;
}

/* Grow the batch scratch to hold `cap` token rows. Returns 0 ok, -1 oom. */
static int batch_scratch(hakm_ctx *c, int cap) {
	if (cap <= c->batch_cap) return 0;
	int D = c->d_model, F = c->ffn_dim, KV = c->kv_dim;
	free(c->bxb); free(c->bxn); free(c->bq); free(c->bk); free(c->bv);
	free(c->batt); free(c->bgate); free(c->bup);
	c->bxb  = malloc((size_t)cap * D  * sizeof(float));
	c->bxn  = malloc((size_t)cap * D  * sizeof(float));
	c->bq   = malloc((size_t)cap * D  * sizeof(float));
	c->bk   = malloc((size_t)cap * KV * sizeof(float));
	c->bv   = malloc((size_t)cap * KV * sizeof(float));
	c->batt = malloc((size_t)cap * D  * sizeof(float));
	c->bgate = malloc((size_t)cap * F * sizeof(float));
	c->bup   = malloc((size_t)cap * F * sizeof(float));
	if (!c->bxb || !c->bxn || !c->bq || !c->bk || !c->bv ||
	    !c->batt || !c->bgate || !c->bup) { c->batch_cap = 0; return -1; }
	c->batch_cap = cap;
	return 0;
}

const float *hakm_forward_batch(hakm_ctx *c, const int *toks, int n) {
	if (n > c->max_seq - c->pos) n = c->max_seq - c->pos;
	if (n <= 0) return NULL;
	if (n == 1) return hakm_forward(c, toks[0]);
	if (batch_scratch(c, n) != 0) return NULL;

	int D = c->d_model, H = c->n_heads, KVH = c->n_kv_heads, hd = c->head_dim;
	int KV = c->kv_dim, F = c->ffn_dim, pos0 = c->pos;

	/* 1. embedding lookup, all n tokens */
	for (int b = 0; b < n; b++)
		deq_row(&c->model, c->tok_embd, toks[b], D, c->bxb + (size_t)b * D);

	for (int l = 0; l < c->n_layers; l++) {
		layer_t *L = &c->layers[l];

		/* 2a. attention rmsnorm */
		for (int b = 0; b < n; b++)
			rmsnorm(c->bxn + (size_t)b * D, c->bxb + (size_t)b * D,
			        fdata(c, L->attn_norm), D, c->rms_eps);

		/* 2b. QKV projections (+bias), batched */
		matmul_batch(c->bq, c->bxn, &c->model, L->wq, L->bq ? fdata(c, L->bq) : NULL, D,  D, n);
		matmul_batch(c->bk, c->bxn, &c->model, L->wk, L->bk ? fdata(c, L->bk) : NULL, KV, D, n);
		matmul_batch(c->bv, c->bxn, &c->model, L->wv, L->bv ? fdata(c, L->bv) : NULL, KV, D, n);

		/* 2c. rope + 2d. cache write, per token */
		for (int b = 0; b < n; b++) {
			float *qb = c->bq + (size_t)b * D, *kb = c->bk + (size_t)b * KV;
			for (int h = 0; h < H;   h++) rope_neox(qb + h * hd, hd, pos0 + b, c->rope_theta);
			for (int h = 0; h < KVH; h++) rope_neox(kb + h * hd, hd, pos0 + b, c->rope_theta);
			kv_store(c->k_cache + ((size_t)l * c->max_seq + pos0 + b) * KV, kb, KV);
			kv_store(c->v_cache + ((size_t)l * c->max_seq + pos0 + b) * KV, c->bv + (size_t)b * KV, KV);
		}

		/* 2e. GQA attention — token b attends 0..pos0+b (causal; all batch k/v
		   already in cache, the loop bound is the mask). Rows are independent
		   (scores live per-row in bgate) → chunked across the worker pool. */
		{
			int T = hakm_nthreads();
			if (T > n) T = n;
			att_job aj[16];
			int per = (n + T - 1) / T, nj = 0;
			for (int t = 0; t < T; t++) {
				int b0 = t * per, b1 = b0 + per;
				if (b1 > n) b1 = n;
				if (b0 >= b1) break;
				aj[nj++] = (att_job){ c, l, b0, b1, 0, 0, pos0, NULL, NULL, NULL };
			}
			hakm_parallel(att_batch_run, aj, sizeof(att_job), nj);
		}

		/* 2f. output proj, residual */
		matmul_batch(c->bxn, c->batt, &c->model, L->wo, NULL, D, D, n);
		for (int b = 0; b < n; b++)
			for (int i = 0; i < D; i++) c->bxb[(size_t)b * D + i] += c->bxn[(size_t)b * D + i];

		/* 2g. ffn rmsnorm */
		for (int b = 0; b < n; b++)
			rmsnorm(c->bxn + (size_t)b * D, c->bxb + (size_t)b * D,
			        fdata(c, L->ffn_norm), D, c->rms_eps);

		/* 2h. SwiGLU, batched */
		matmul_batch(c->bgate, c->bxn, &c->model, L->gate, NULL, F, D, n);
		matmul_batch(c->bup,   c->bxn, &c->model, L->up,   NULL, F, D, n);
		for (int b = 0; b < n; b++) {
			float *g = c->bgate + (size_t)b * F, *u = c->bup + (size_t)b * F;
			for (int i = 0; i < F; i++) g[i] = silu_f(g[i]) * u[i];
		}
		matmul_batch(c->bxn, c->bgate, &c->model, L->down, NULL, D, F, n);
		for (int b = 0; b < n; b++)
			for (int i = 0; i < D; i++) c->bxb[(size_t)b * D + i] += c->bxn[(size_t)b * D + i];
	}

	/* 3. final norm + lm_head — LAST token only (the rest of the batch only
	   needed to populate the KV cache; skipping their vocab matmul saves ~10%
	   of prefill compute on the 3B). */
	rmsnorm(c->xn, c->bxb + (size_t)(n - 1) * D, fdata(c, c->out_norm), D, c->rms_eps);
	matmul(c->logits, c->xn, &c->model, c->out_weight, NULL, c->vocab, D);

	c->pos = pos0 + n;
	return c->logits;
}
