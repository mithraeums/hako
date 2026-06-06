/* hakm_api.h — in-process inference API.
 *
 * One header to embed the hako engine directly inside another program (the
 * `hako` agent) with NO subprocess, NO socket, NO server, NO ollama. Link the
 * engine objects (loader/quant/nn/model/bpe/sample + hakm_api) and call
 * hakm_chat() in-process. Weights stay resident in a session; replies stream
 * out token-by-token through a caller callback.
 */
#pragma once
#include <stdint.h>

typedef struct hakm_session hakm_session;

/* One chat message. role is "system" | "user" | "assistant". */
typedef struct {
	const char *role;
	const char *content;
} hakm_msg;

/* Sampling / decode controls. Zero-initialise then override; hakm_params_default
 * fills sane values. */
typedef struct {
	int      n_new;   /* max new tokens for the reply (default 512) */
	float    temp;    /* temperature; 0 => greedy (default 0.7) */
	float    top_p;   /* nucleus; 0 => off (default 0.9) */
	int      top_k;   /* top-k; 0 => off (default 40) */
	uint64_t seed;    /* rng seed; 0 => time-seeded */
} hakm_params;

void hakm_params_default(hakm_params *p);

/* Open a resident session over an MLF2 weight file. max_seq caps the context
 * window (KV cache). Returns NULL on load failure. */
hakm_session *hakm_session_open(const char *mlf2_path, int max_seq);
void          hakm_session_close(hakm_session *s);

/* Model info for banners / discovery. Safe to call after open. */
const char *hakm_session_arch(const hakm_session *s);   /* e.g. "qwen2" */
int         hakm_session_ctxlen(const hakm_session *s);  /* max_seq */

/* Run one chat completion over the full message list.
 *
 * The whole conversation is re-rendered as ChatML each call (the KV cache is
 * reset first), matching a stateless request wire — the caller owns history.
 * `emit` is invoked with successive UTF-8 chunks of the assistant reply as they
 * decode; return non-zero from emit to abort generation early. emit may be NULL.
 *
 * Returns the number of tokens generated (>=0), or -1 on error. */
int hakm_chat(hakm_session *s, const hakm_msg *msgs, int n_msgs,
              const hakm_params *p,
              int (*emit)(const char *chunk, int len, void *ud), void *ud);
