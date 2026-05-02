/*
 * mini.c — public API implementation (mini_open, mini_generate, etc.)
 */
#include "../include/mini.h"
#include "loader.h"
#include "model.h"
#include "tokenizer.h"
#include "sample.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct mini_ctx {
	nano_model  model;
	model_bufs *bufs;
	char        desc[128];
};

mini_ctx *mini_open(const char *mlf_path) {
	mini_ctx *ctx = (mini_ctx *)calloc(1, sizeof(mini_ctx));
	if (!ctx) return NULL;

	if (mlf_load(mlf_path, &ctx->model) != 0) {
		free(ctx);
		return NULL;
	}

	ctx->bufs = model_bufs_alloc(&ctx->model.p);
	if (!ctx->bufs) {
		mlf_unload(&ctx->model);
		free(ctx);
		return NULL;
	}

	const mlf_params *p = &ctx->model.p;
	snprintf(ctx->desc, sizeof(ctx->desc),
		"nano  layers=%u d_model=%u heads=%u ffn=%u ctx=%u vocab=%u",
		p->n_layers, p->d_model, p->n_heads, p->ffn_dim, p->ctx, p->vocab);

	return ctx;
}

void mini_close(mini_ctx *ctx) {
	if (!ctx) return;
	model_bufs_free(ctx->bufs);
	mlf_unload(&ctx->model);
	free(ctx);
}

const char *mini_model_desc(const mini_ctx *ctx) {
	return ctx ? ctx->desc : "(null)";
}

int mini_encode(mini_ctx *ctx, const char *text, int *out, int max_tokens) {
	(void)ctx;  /* byte-level: context not needed */
	return tok_encode(text, out, max_tokens);
}

int mini_decode(mini_ctx *ctx, const int *tokens, int n_tokens,
                char *out, int max_bytes) {
	(void)ctx;
	return tok_decode(tokens, n_tokens, out, max_bytes);
}

int mini_generate(mini_ctx *ctx,
                  const int *prompt, int n_prompt,
                  int *out,          int n_max,
                  const mini_sample_opts *opts) {
	if (!ctx || !prompt || !out || n_max <= 0) return -1;

	int max_ctx = (int)ctx->model.p.ctx;
	int vocab   = (int)ctx->model.p.vocab;

	/* Working token buffer: up to max_ctx tokens */
	int *toks = (int *)malloc((size_t)max_ctx * sizeof(int));
	if (!toks) return -1;

	int cur = n_prompt < max_ctx ? n_prompt : max_ctx;
	memcpy(toks, prompt + (n_prompt - cur), (size_t)cur * sizeof(int));

	int generated = 0;
	while (generated < n_max) {
		model_forward(&ctx->model, toks, cur, ctx->bufs);

		/* bufs->logits is overwritten by model_forward — copy before sample */
		float *logits_copy = (float *)malloc((size_t)vocab * sizeof(float));
		if (!logits_copy) break;
		memcpy(logits_copy, ctx->bufs->logits, (size_t)vocab * sizeof(float));

		int next = sample_token(logits_copy, vocab, opts);
		free(logits_copy);

		out[generated++] = next;

		/* Slide window if at capacity */
		if (cur < max_ctx) {
			toks[cur++] = next;
		} else {
			memmove(toks, toks + 1, (size_t)(max_ctx - 1) * sizeof(int));
			toks[max_ctx - 1] = next;
		}
	}

	free(toks);
	return generated;
}

/* Grammar-guided sampling placeholder — implemented at M5/M6. */
int mini_generate_grammar(mini_ctx *ctx,
                          const int *prompt, int n_prompt,
                          int *out,          int n_max,
                          const mini_sample_opts *opts,
                          const struct mini_grammar *grammar) {
	(void)grammar;
	return mini_generate(ctx, prompt, n_prompt, out, n_max, opts);
}
