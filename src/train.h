/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* train.h — train a model of our own on the CPU: byte-level BPE, the engine's
 * forward pass with its backward, AdamW, and an MLF2 file hakm runs as is. */
#pragma once
#include <stddef.h>
#include <stdint.h>

/* The shape is the engine's (RMSNorm → GQA attention → neox RoPE → SwiGLU),
 * no biases, no q/k norm, lm_head tied to the embedding. Written as arch hako.
 * experts > 0 makes the ffn a mixture: a router picks top_k of the layer's
 * experts per token (2 <= top_k <= min(experts, 8)), their SwiGLU outputs are
 * summed by the softmax of the picked scores, and shared = 1 keeps the dense
 * ffn beside them as an expert every token runs. experts = 0 is dense, whatever
 * top_k and shared say (positional inits of the first ten fields stay dense). */
typedef struct {
	int vocab, dim, layers, heads, kv_heads, head_dim, ffn, ctx;
	float rope_theta, rms_eps;
	int experts, top_k, shared;
} tr_config;

/* Offsets into the parameters. Dense order per layer: attn_norm, wq, wk, wv, wo,
 * ffn_norm, gate, up, down; then output_norm. A moe layer has its router
 * [experts x dim] after ffn_norm, then gate, up, down only when shared (else
 * they are 0 and unused). The routed experts come after output_norm, so the
 * dense part keeps its layout: expert (l, e) is one range of 3 x ffn x dim
 * floats at tr_expert_off(m, l, e), gate [ffn x dim], up [ffn x dim], down
 * [dim x ffn], in (l, e) order to the end. experts = expert (l, 0) */
typedef struct { size_t attn_norm, wq, wk, wv, wo, ffn_norm, gate, up, down, router, experts; } tr_layer;

typedef struct tr_model tr_model;

/* NULL and a message in err when the shape cannot train or cannot run. */
const char *tr_check(const tr_config *c);

tr_model *tr_new(const tr_config *c, uint64_t seed);
void      tr_free(tr_model *m);
const tr_config *tr_cfg(const tr_model *m);
size_t    tr_n_params(const tr_model *m);
float    *tr_params(tr_model *m);
float    *tr_grads(tr_model *m);
int64_t  *tr_step(tr_model *m);

/* Summed NLL (natural log) of tgt given in, over B sequences of T tokens
 * (T <= ctx). With grad_scale > 0 the gradient of grad_scale * NLL is ADDED
 * to the grads (so microbatches accumulate); 0 is a forward pass only.
 * -1 on allocation failure. */
double tr_run(tr_model *m, const int *in, const int *tgt, int B, int T, float grad_scale);
/* Built with HAKM_GPU (make GPU=1), a run with grad_scale > 0 puts its big
 * products on the GPU (gpu.h; HAKM_GPU_MIN = the smallest M x N x K it takes,
 * default 2^24). A forward pass alone (grad_scale 0) always runs on the CPU,
 * though its last bits still differ from one kind of CPU to the next: a loss two
 * machines must agree on comes from iloss.h. 0 = CPU for every run, 1 = the
 * default. No effect in a build without the GPU. */
void   tr_use_gpu(int on);
/* The GPU a dense model's training run takes, NULL for the CPU (no GPU, turned off, a CPU build). */
const char *tr_gpu_name(void);

/* Bytes a run of B x T tokens holds: weights, grads, moments, activations, scratch. */
size_t tr_mem_bytes(const tr_config *c, int B, int T);

void   tr_zero_grad(tr_model *m);
/* Global grad norm, then scaled down to max_norm when it is over (0 = no clip). */
double tr_clip(tr_model *m, double max_norm);
void   tr_adamw(tr_model *m, float lr, float wd, float b1, float b2, float eps);
/* Freeze everything outside n ranges of the parameter vector (off, len pairs; they
 * must not overlap): tr_adamw then leaves those weights AND their moments alone, so
 * a frozen weight's optimizer state is not decayed by steps it took no part in.
 * NULL or n = 0 = every weight live again. Not saved with the optimizer. -1 on a
 * range past the end. */
int    tr_set_live(tr_model *m, const size_t *ranges, int n);

/* Moe only. The auxiliary losses tr_run adds to the NLL it differentiates
 * (J = NLL + aux), on each token's softmax p over all the router's scores:
 * sticky = -sticky x the sum over tokens of p on the previous token's experts
 * (same sequence), balance = balance x experts x sum over experts of
 * (share of the batch's picks) x (summed p). Sticky alone collapses every
 * token onto one set: pair it with balance. Defaults sticky 0, balance 0.01. */
void   tr_set_aux(tr_model *m, float sticky, float balance);
/* The aux part of J in the last tr_run, summed over layers (0 for dense). */
double tr_aux(const tr_model *m);
/* The last tr_run's picks for a layer, [B*T][top_k] by rank (highest score
 * first), valid until the next run; NULL for dense or before a run. */
const int *tr_routes(const tr_model *m, int layer);
/* Where expert (l, e) starts in the parameters (and grads, and moments): its
 * 3 x ffn x dim floats, gate, up, down. Every expert follows the dense part,
 * from tr_expert_off(m, 0, 0) to tr_n_params(m). SIZE_MAX when out of range. */
size_t tr_expert_off(const tr_model *m, int l, int e);

/* Weights to MLF2 (f32) with this tokenizer blob, via <path>.part and a rename. */
int tr_save(const tr_model *m, const char *path, const void *tok, size_t tok_len, int eos);
/* A model this trainer wrote. *tok gets a malloc'd copy of its tokenizer blob. */
tr_model *tr_load(const char *path, void **tok, size_t *tok_len, int *eos);
/* AdamW moments + step, beside the weights; resume needs both files. A dense
 * model writes version 1 (the original header), a moe one version 2 with its
 * experts, top_k and shared after the header; each refuses the other. */
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

/* A whole file, NUL-terminated, NULs inside turned to spaces (the encoder stops at one). */
char *tr_slurp(const char *path, size_t *len);
/* text -> ids with the engine's own encoder, the one hakm uses. Cuts text in pieces after
 * blank lines (restored after), so a corpus past 2 GB fits. malloc'd, *n ids. */
int *tr_encode(const void *blob, size_t blen, int eos, char *text, size_t len, int64_t *n);
