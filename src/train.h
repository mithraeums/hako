/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* train.h — train a model of our own on the CPU: byte-level BPE, the engine's
 * forward pass with its backward, AdamW, and an MLF2 file hakm runs as is. */
#pragma once
#include <stddef.h>
#include <stdint.h>

/* The shape is the engine's (RMSNorm → GQA attention → neox RoPE → SwiGLU),
 * no biases, no q/k norm, lm_head tied to the embedding. Written as arch hako. */
typedef struct {
	int vocab, dim, layers, heads, kv_heads, head_dim, ffn, ctx;
	float rope_theta, rms_eps;
} tr_config;

typedef struct { size_t attn_norm, wq, wk, wv, wo, ffn_norm, gate, up, down; } tr_layer;

typedef struct tr_model tr_model;

/* NULL and a message in err when the shape cannot train or cannot run. */
const char *tr_check(const tr_config *c);

tr_model *tr_new(const tr_config *c, uint64_t seed);
void      tr_free(tr_model *m);
const tr_config *tr_cfg(const tr_model *m);
size_t    tr_n_params(const tr_model *m);
float    *tr_params(tr_model *m);
float    *tr_grads(tr_model *m);
long     *tr_step(tr_model *m);

/* Summed NLL (natural log) of tgt given in, over B sequences of T tokens
 * (T <= ctx). With grad_scale > 0 the gradient of grad_scale * NLL is ADDED
 * to the grads (so microbatches accumulate); 0 is a forward pass only.
 * -1 on allocation failure. */
double tr_run(tr_model *m, const int *in, const int *tgt, int B, int T, float grad_scale);

/* Bytes a run of B x T tokens holds: weights, grads, moments, activations, scratch. */
size_t tr_mem_bytes(const tr_config *c, int B, int T);

void   tr_zero_grad(tr_model *m);
/* Global grad norm, then scaled down to max_norm when it is over (0 = no clip). */
double tr_clip(tr_model *m, double max_norm);
void   tr_adamw(tr_model *m, float lr, float wd, float b1, float b2, float eps);

/* Weights to MLF2 (f32) with this tokenizer blob, via <path>.part and a rename. */
int tr_save(const tr_model *m, const char *path, const void *tok, size_t tok_len, int eos);
/* A model this trainer wrote. *tok gets a malloc'd copy of its tokenizer blob. */
tr_model *tr_load(const char *path, void **tok, size_t *tok_len, int *eos);
/* AdamW moments + step, beside the weights; resume needs both files. */
int tr_save_opt(const tr_model *m, const char *path, uint64_t rng);
int tr_load_opt(tr_model *m, const char *path, uint64_t *rng);

/* Byte-level BPE over text, split the way the engine's tokenizer splits
 * (tok_encode in bpe.c), so its merges are the ones that encoder applies.
 * vocab counts 256 byte tokens, the merges, <|endoftext|>, <|im_start|> and
 * <|im_end|>, and [PAD] tokens up to vocab when the text runs out of pairs.
 * Writes a malloc'd MLF2 tokenizer blob. Returns the merge count, -1 on error. */
int bpe_train(const char *text, size_t len, int vocab, void **blob, size_t *blob_len, int *eos);
/* Bytes of a tokenizer blob, read from its own counts; 0 when it is cut short. */
size_t tok_blob_len(const void *blob, size_t avail);
