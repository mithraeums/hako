/* bpe.h — Qwen2 byte-level BPE tokenizer, loaded from the MLF2 tok blob. */
#pragma once
#include "loader.h"

typedef struct hakm_tok hakm_tok;

hakm_tok *tok_new(const hakm_model *m);
void      tok_free(hakm_tok *t);

/* Encode UTF-8 text into token ids. Returns count, or -1 on overflow.
 * NOTE: uses a simplified GPT2 pretokenizer (whitespace-delimited pieces with
 * a leading-space marker) — exact for space-separated text without runs of
 * punctuation; good enough for completion smokes. */
int tok_encode(const hakm_tok *t, const char *text, int *out, int max);

/* Decode one token id, appending its bytes to `out` (returns bytes written). */
int tok_decode_id(const hakm_tok *t, int id, char *out, int max);

/* Exact vocab lookup of a literal string (e.g. "<|im_start|>"); -1 if absent.
 * Used to inject special/control tokens that must not be BPE-encoded. */
int tok_id(const hakm_tok *t, const char *literal);

int tok_eos(const hakm_tok *t);
