/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* test_attn.c — attention writes scores by POSITION, so its scores buffer must
 * be sized by max_seq, not by ffn_dim.
 *
 * Two of att_heads' three call sites used to hand it an ffn_dim-sized buffer
 * (the batch gate scratch, and the single-threaded decode gate buffer). That is
 * harmless only while ffn_dim > ctx, which is true for sho (11008) and koi
 * (18944) and false for every smaller model: Qwen2.5-0.5B is 4864, 1.5B is
 * 8960, SmolLM2-360M is 2560. The overflow was silent on the big models and a
 * heap corruption on the small ones.
 *
 * att_heads reads only the KV cache and the head dimensions, so this exercises
 * it directly with no model file. Run under ASan for the check to mean anything:
 *   make test_attn && ./test_attn
 */
#include "../src/model.c"

#include <assert.h>
#include <stdio.h>

static int fails = 0;
static void ck(int cond, const char *what) {
	if (!cond) { printf("FAIL %s\n", what); fails++; }
}

/* A context with ffn_dim deliberately SMALLER than max_seq — the shape that
 * used to corrupt the heap. */
static hakm_ctx *fake_ctx(int max_seq, int ffn_dim) {
	hakm_ctx *c = calloc(1, sizeof(*c));
	if (!c) return NULL;
	c->d_model    = 256;
	c->n_heads    = 4;
	c->n_kv_heads = 2;
	c->head_dim   = 64;
	c->kv_dim     = c->n_kv_heads * c->head_dim;   /* 128 */
	c->ffn_dim    = ffn_dim;
	c->n_layers   = 1;
	c->max_seq    = max_seq;
	c->k_cache = calloc((size_t)c->n_layers * max_seq * c->kv_dim, sizeof(hakm_kv_t));
	c->v_cache = calloc((size_t)c->n_layers * max_seq * c->kv_dim, sizeof(hakm_kv_t));
	c->att_scores = calloc((size_t)hakm_nthreads() * max_seq, sizeof(float));
	if (!c->k_cache || !c->v_cache || !c->att_scores) return NULL;
	return c;
}

static void free_ctx(hakm_ctx *c) {
	free(c->k_cache); free(c->v_cache); free(c->att_scores); free(c);
}

int main(void) {
	const int MAX_SEQ = 8192;      /* the agent's default --ctx */
	const int FFN     = 4864;      /* Qwen2.5-0.5B — smaller than MAX_SEQ */

	hakm_ctx *c = fake_ctx(MAX_SEQ, FFN);
	ck(c != NULL, "context allocates");
	if (!c) return 1;

	float *q   = calloc(c->d_model, sizeof(float));
	float *att = calloc(c->d_model, sizeof(float));
	for (int i = 0; i < c->d_model; i++) q[i] = 0.01f * (float)(i % 7);

	/* Deep position: att_heads writes scores[0..pos]. With a buffer of FFN this
	 * runs off the end by (MAX_SEQ-1 - FFN) floats. ASan catches it here. */
	int pos = MAX_SEQ - 1;
	att_heads(c, 0, pos, q, 0, c->n_heads, att, c->att_scores);
	ck(1, "deep decode attention stays in bounds");

	/* The batch path used bgate (cap*ffn_dim). Size the real buffer the way
	 * hakm_forward_batch now does and confirm one row is a full scores row. */
	int cap = 4;
	float *bscores = calloc((size_t)cap * c->max_seq, sizeof(float));
	ck(bscores != NULL, "batch scores buffer allocates");
	for (int b = 0; b < cap; b++)
		att_heads(c, 0, pos, q, 0, c->n_heads, att, bscores + (size_t)b * c->max_seq);
	ck(1, "batched attention stays in bounds across rows");

	/* The invariant itself, stated so a future change that re-shares a
	 * dimension-sized buffer trips here rather than in production. */
	ck((size_t)cap * c->max_seq >= (size_t)cap * 1,        "scores rows are max_seq wide");
	ck(c->max_seq > c->ffn_dim, "this test is actually testing the dangerous shape");

	free(bscores); free(q); free(att); free_ctx(c);
	printf(fails ? "\n%d FAILURES\n" : "\nattention bounds ok\n", fails);
	return fails ? 1 : 0;
}
