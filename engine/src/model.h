/* model.h — Qwen2 forward pass with KV cache, over MLF2 weights. */
#pragma once
#include "loader.h"

typedef struct {
	const mlf2_tensor *attn_norm, *ffn_norm;
	const mlf2_tensor *wq, *wk, *wv, *wo;
	const mlf2_tensor *bq, *bk, *bv;          /* biases (F32) or NULL */
	const mlf2_tensor *gate, *up, *down;
} layer_t;

typedef struct hakm_ctx {
	hakm_model model;
	layer_t   *layers;
	const mlf2_tensor *tok_embd, *out_norm;   /* lm_head tied to tok_embd */

	int d_model, n_heads, n_kv_heads, head_dim, kv_dim, ffn_dim, n_layers, vocab;
	float rope_theta, rms_eps;

	int max_seq, pos;            /* KV cache capacity + next write position */
	float *k_cache, *v_cache;    /* [n_layers * max_seq * kv_dim] each */

	/* scratch (single allocation) */
	float *xb, *xn, *q, *k, *v, *att_out, *gate_buf, *up_buf, *deq, *logits;
} hakm_ctx;

/* Build a context. max_seq caps the KV cache (context window). */
hakm_ctx *hakm_ctx_new(const char *mlf2_path, int max_seq);
void      hakm_ctx_free(hakm_ctx *c);

/* Run one token at the current position; returns pointer to logits[vocab].
 * Advances the KV cache position by one. */
const float *hakm_forward(hakm_ctx *c, int token);

/* Reset position to 0 (new sequence). */
void hakm_reset(hakm_ctx *c);
