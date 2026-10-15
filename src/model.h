/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* model.h — Qwen2 forward pass with KV cache, over MLF2 weights. */
#pragma once
#include "loader.h"
#include <stdint.h>

/* KV cache element: f16 where the hardware converts cheaply (halves attention
 * memory traffic + cache RAM — ~300MB back at ctx 4096 on the 7B), f32
 * elsewhere (scalar per-element converts would tax weak boxes like iSh). */
#if (defined(__F16C__) && defined(__FMA__)) || defined(__aarch64__)
#define HAKM_KV_F16 1
typedef uint16_t hakm_kv_t;
#else
typedef float hakm_kv_t;
#endif

typedef struct {
	const mlf2_tensor *attn_norm, *ffn_norm;
	const mlf2_tensor *wq, *wk, *wv, *wo;
	const mlf2_tensor *bq, *bk, *bv;          /* biases (F32) or NULL */
	const mlf2_tensor *gate, *up, *down;
} layer_t;

typedef struct hakm_ctx {
	hakm_model model;
	layer_t   *layers;
	const mlf2_tensor *tok_embd, *out_norm;
	const mlf2_tensor *out_weight;            /* lm_head: == tok_embd if TIED_EMBED, else output.weight (7B+) */

	int d_model, n_heads, n_kv_heads, head_dim, kv_dim, ffn_dim, n_layers, vocab;
	float rope_theta, rms_eps;

	int max_seq, pos;            /* KV cache capacity + next write position */
	hakm_kv_t *k_cache, *v_cache; /* [n_layers * max_seq * kv_dim] each */

	/* scratch (single allocation) */
	float *xb, *xn, *q, *k, *v, *att_out, *gate_buf, *up_buf, *deq, *logits;

	/* batch-prefill scratch: B-row versions of the above, lazily sized by
	 * hakm_forward_batch (batch_cap rows). NULL until first batched call. */
	float *bxb, *bxn, *bq, *bk, *bv, *batt, *bgate, *bup;
	/* Attention scores are indexed by POSITION (0..pos), so this is sized by
	   max_seq — not by ffn_dim like the gate/up buffers beside it. Sharing
	   bgate here overflowed the heap on any model whose ffn_dim < ctx. */
	float *bscores;
	int    batch_cap;

	/* per-thread attention score rows for threaded decode attention
	 * (nthreads × max_seq floats) */
	float *att_scores;
} hakm_ctx;

/* Build a context. max_seq caps the KV cache (context window). */
hakm_ctx *hakm_ctx_new(const char *mlf2_path, int max_seq);
void      hakm_ctx_free(hakm_ctx *c);

/* Run one token at the current position; returns pointer to logits[vocab].
 * Advances the KV cache position by one. */
const float *hakm_forward(hakm_ctx *c, int token);

/* Prefill n tokens at once (causal within the batch). Same math per token as
 * hakm_forward, but each weight row is streamed from memory once per batch
 * instead of once per token, and the lm_head runs only for the LAST token.
 * Advances pos by n; returns the last token's logits (NULL on alloc failure
 * or if nothing fits). Use for prompt prefill; decode stays token-at-a-time. */
const float *hakm_forward_batch(hakm_ctx *c, const int *toks, int n);

/* Reset position to 0 (new sequence). */
void hakm_reset(hakm_ctx *c);
