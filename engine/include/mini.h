/*
 * mini.h — public API for the miniLocal inference engine (ARCH-locked).
 * Supports MLF v0 (nano) and MLF v1 (student/teacher) weight files.
 * All weights are mmap'd read-only; RSS = activations + KV cache only.
 */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mini_ctx mini_ctx;

typedef struct {
	float temperature;  /* 0.0 = greedy */
	float top_p;        /* 0.0 = disabled */
	int   top_k;        /* 0   = disabled */
	uint64_t seed;      /* RNG seed; 0 = use time */
} mini_sample_opts;

/* Open .mlf file, mmap weights, allocate working buffers.
 * Returns NULL on error (check errno). */
mini_ctx *mini_open(const char *mlf_path);

/* Encode UTF-8/byte text into token IDs.
 * Returns number of tokens written, or -1 on overflow. */
int mini_encode(mini_ctx *ctx, const char *text, int *out, int max_tokens);

/* Decode token IDs to a null-terminated UTF-8 string.
 * Returns bytes written (excluding null), or -1 on overflow. */
int mini_decode(mini_ctx *ctx, const int *tokens, int n_tokens,
                char *out, int max_bytes);

/* Autoregressive generation.  Writes new tokens into `out` (not the prompt).
 * Returns number of tokens generated, or -1 on error. */
int mini_generate(mini_ctx *ctx,
                  const int *prompt, int n_prompt,
                  int *out,          int n_max,
                  const mini_sample_opts *opts);

/* Grammar-constrained generation (M5/M6). NULL grammar falls back to
 * mini_generate behaviour. */
struct mini_grammar;
int mini_generate_grammar(mini_ctx *ctx,
                          const int *prompt, int n_prompt,
                          int *out,          int n_max,
                          const mini_sample_opts *opts,
                          const struct mini_grammar *grammar);

/* Release all resources. */
void mini_close(mini_ctx *ctx);

/* Human-readable description of the loaded model (static buffer). */
const char *mini_model_desc(const mini_ctx *ctx);

#ifdef __cplusplus
}
#endif
