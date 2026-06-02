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

	c->layers = calloc(c->n_layers, sizeof(layer_t));
	int ok = (c->tok_embd && c->out_norm);
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
	c->k_cache  = calloc((size_t)c->n_layers * max_seq * KV, sizeof(float));
	c->v_cache  = calloc((size_t)c->n_layers * max_seq * KV, sizeof(float));
	if (!c->k_cache || !c->v_cache || !c->logits) { hakm_ctx_free(c); return NULL; }
	return c;
}

void hakm_ctx_free(hakm_ctx *c) {
	if (!c) return;
	free(c->layers); free(c->xb); free(c->xn); free(c->q); free(c->k); free(c->v);
	free(c->att_out); free(c->gate_buf); free(c->up_buf); free(c->deq);
	free(c->logits); free(c->k_cache); free(c->v_cache);
	mlf2_close(&c->model);
	free(c);
}

void hakm_reset(hakm_ctx *c) { c->pos = 0; }

static const float *fdata(const hakm_ctx *c, const mlf2_tensor *t) {
	return (const float *)mlf2_data(&c->model, t);
}

const float *hakm_forward(hakm_ctx *c, int token) {
	int D = c->d_model, H = c->n_heads, KVH = c->n_kv_heads, hd = c->head_dim;
	int KV = c->kv_dim, F = c->ffn_dim, pos = c->pos;
	int gsize = H / KVH;                  /* q-heads per kv-head */
	float scale = 1.0f / sqrtf((float)hd);

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
		float *kc = c->k_cache + ((size_t)l * c->max_seq + pos) * KV;
		float *vc = c->v_cache + ((size_t)l * c->max_seq + pos) * KV;
		memcpy(kc, c->k, KV * sizeof(float));
		memcpy(vc, c->v, KV * sizeof(float));

		/* 2e. GQA attention over positions 0..pos */
		memset(c->att_out, 0, D * sizeof(float));
		float *scores = c->gate_buf;          /* reuse: needs pos+1 <= F */
		for (int h = 0; h < H; h++) {
			int kvh = h / gsize;
			const float *q_h = c->q + h * hd;
			for (int s = 0; s <= pos; s++) {
				const float *k_s = c->k_cache + ((size_t)l * c->max_seq + s) * KV + kvh * hd;
				float dot = 0.0f;
				for (int d = 0; d < hd; d++) dot += q_h[d] * k_s[d];
				scores[s] = dot * scale;
			}
			softmax(scores, pos + 1);
			float *out_h = c->att_out + h * hd;
			for (int s = 0; s <= pos; s++) {
				const float *v_s = c->v_cache + ((size_t)l * c->max_seq + s) * KV + kvh * hd;
				float w = scores[s];
				for (int d = 0; d < hd; d++) out_h[d] += w * v_s[d];
			}
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

	/* 3. final norm + lm_head (tied to token_embd) */
	rmsnorm(c->xn, c->xb, fdata(c, c->out_norm), D, c->rms_eps);
	matmul(c->logits, c->xn, &c->model, c->tok_embd, NULL, c->vocab, D);

	c->pos++;
	return c->logits;
}
