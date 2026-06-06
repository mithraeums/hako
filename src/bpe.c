/*
 * bpe.c — Qwen2 byte-level BPE.
 *
 * Loads token strings + merge ranks from the MLF2 tokenizer blob (which carries
 * the GGUF gpt2 tokenizer verbatim). Implements GPT2 byte<->unicode mapping and
 * rank-greedy merging. Pretokenization is simplified (see bpe.h).
 */
#include "bpe.h"
#include "../include/mlf2.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ── open-addressing string->int hash ─────────────────────────────────────── */
typedef struct { const char *key; int klen; int val; } hslot;
typedef struct { hslot *s; int cap; } hmap;

static uint64_t fnv(const char *p, int n) {
	uint64_t h = 1469598103934665603ULL;
	for (int i = 0; i < n; i++) { h ^= (unsigned char)p[i]; h *= 1099511628211ULL; }
	return h;
}
static void hm_init(hmap *m, int cap) {
	m->cap = cap; m->s = calloc(cap, sizeof(hslot));
	for (int i = 0; i < cap; i++) m->s[i].val = -1;
}
static void hm_put(hmap *m, const char *k, int klen, int val) {
	uint64_t i = fnv(k, klen) % m->cap;
	while (m->s[i].key) {
		if (m->s[i].klen == klen && !memcmp(m->s[i].key, k, klen)) { m->s[i].val = val; return; }
		i = (i + 1) % m->cap;
	}
	m->s[i].key = k; m->s[i].klen = klen; m->s[i].val = val;
}
static int hm_get(const hmap *m, const char *k, int klen) {
	uint64_t i = fnv(k, klen) % m->cap;
	while (m->s[i].key) {
		if (m->s[i].klen == klen && !memcmp(m->s[i].key, k, klen)) return m->s[i].val;
		i = (i + 1) % m->cap;
	}
	return -1;
}

/* ── tokenizer ────────────────────────────────────────────────────────────── */
struct hakm_tok {
	int n_tokens, n_merges;
	const char **tok_ptr;   /* token string starts (into mlf2 blob) */
	int         *tok_len;
	hmap vocab;             /* token string -> id */
	hmap merge;             /* "a b" -> rank */
	int  byte_enc[256];     /* byte -> mapped codepoint */
	int  cp2byte[512];      /* mapped codepoint -> byte (-1 if none) */
	int  eos;
};

static void build_byte_maps(hakm_tok *t) {
	for (int i = 0; i < 512; i++) t->cp2byte[i] = -1;
	int n = 0;
	for (int b = 0; b < 256; b++) {
		int printable = (b >= '!' && b <= '~') ||
		                (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
		int cp = printable ? b : (256 + n);
		if (!printable) n++;
		t->byte_enc[b] = cp;
		t->cp2byte[cp] = b;
	}
}

/* UTF-8 encode a codepoint (< 0x800) into buf, return bytes. */
static int utf8_put(int cp, char *buf) {
	if (cp < 0x80) { buf[0] = (char)cp; return 1; }
	buf[0] = (char)(0xC0 | (cp >> 6));
	buf[1] = (char)(0x80 | (cp & 0x3F));
	return 2;
}
/* Decode one UTF-8 codepoint from p (<=2 bytes here), return codepoint, set *adv. */
static int utf8_get(const char *p, int *adv) {
	unsigned char c = (unsigned char)p[0];
	if (c < 0x80) { *adv = 1; return c; }
	*adv = 2; return ((c & 0x1F) << 6) | ((unsigned char)p[1] & 0x3F);
}

hakm_tok *tok_new(const hakm_model *m) {
	hakm_tok *t = calloc(1, sizeof(*t));
	const unsigned char *p = m->tok;
	memcpy(&t->n_tokens, p, 4);
	memcpy(&t->n_merges, p + 4, 4);
	p += 8;

	t->tok_ptr = malloc((size_t)t->n_tokens * sizeof(char *));
	t->tok_len = malloc((size_t)t->n_tokens * sizeof(int));
	hm_init(&t->vocab, t->n_tokens * 2);
	for (int i = 0; i < t->n_tokens; i++) {
		/* record: u8 type, u32 len, bytes */
		p += 1;                               /* type (unused for now) */
		int len; memcpy(&len, p, 4); p += 4;
		t->tok_ptr[i] = (const char *)p;
		t->tok_len[i] = len;
		hm_put(&t->vocab, (const char *)p, len, i);
		p += len;
	}
	hm_init(&t->merge, t->n_merges * 2);
	for (int i = 0; i < t->n_merges; i++) {
		int len; memcpy(&len, p, 4); p += 4;
		hm_put(&t->merge, (const char *)p, len, i);   /* rank = i */
		p += len;
	}
	build_byte_maps(t);
	t->eos = (int)m->h.eos_id;
	return t;
}

void tok_free(hakm_tok *t) {
	if (!t) return;
	free(t->tok_ptr); free(t->tok_len);
	free(t->vocab.s); free(t->merge.s);
	free(t);
}

int tok_eos(const hakm_tok *t) { return t->eos; }

int tok_id(const hakm_tok *t, const char *literal) {
	return hm_get(&t->vocab, literal, (int)strlen(literal));
}

/*
 * BPE-merge one pretokenized piece. `work` holds the byte-encoded bytes of the
 * piece; off[]/len[] describe `n` symbols as spans into `work`. Because only
 * adjacent symbols are ever merged, their spans stay contiguous — a merge just
 * grows len[i] and drops entry i+1, no byte movement. Emits ids into out.
 */
static int bpe_piece(const hakm_tok *t, const char *work, int *off, int *len,
                     int n, int *out, int oi, int max) {
	char pair[2048];
	for (;;) {
		int best_rank = 0x7fffffff, best = -1;
		for (int i = 0; i + 1 < n; i++) {
			int a = len[i], b = len[i + 1];
			if (a + 1 + b > (int)sizeof(pair)) continue;
			memcpy(pair, work + off[i], a);
			pair[a] = ' ';
			memcpy(pair + a + 1, work + off[i + 1], b);
			int r = hm_get(&t->merge, pair, a + 1 + b);
			if (r >= 0 && r < best_rank) { best_rank = r; best = i; }
		}
		if (best < 0) break;
		len[best] += len[best + 1];            /* spans are contiguous */
		for (int j = best + 1; j + 1 < n; j++) { off[j] = off[j + 1]; len[j] = len[j + 1]; }
		n--;
	}
	for (int i = 0; i < n; i++) {
		if (oi >= max) return -1;
		int id = hm_get(&t->vocab, work + off[i], len[i]);
		if (id < 0) return -1;        /* unknown symbol — shouldn't happen */
		out[oi++] = id;
	}
	return oi;
}

int tok_encode(const hakm_tok *t, const char *text, int *out, int max) {
	int oi = 0;
	const unsigned char *s = (const unsigned char *)text;
	size_t len = strlen(text), i = 0;

	while (i < len) {
		/* simplified pretokenizer: optional single leading space + run of
		 * non-space bytes => one piece */
		size_t start = i;
		if (s[i] == ' ') i++;               /* keep one leading space in piece */
		while (i < len && s[i] != ' ') i++;
		if (i == start) { i++; continue; }  /* stray space run */

		/* byte-encode the piece into a flat buffer; each source byte becomes
		 * one initial symbol (1-2 bytes of mapped-codepoint UTF-8). */
		char work[4096]; int off[2048], len[2048], n = 0, w = 0;
		for (size_t b = start; b < i && n < 2048 && w + 2 <= (int)sizeof(work); b++) {
			int cp = t->byte_enc[s[b]];
			off[n] = w;
			len[n] = utf8_put(cp, work + w);
			w += len[n];
			n++;
		}
		oi = bpe_piece(t, work, off, len, n, out, oi, max);
		if (oi < 0) return -1;
	}
	return oi;
}

int tok_decode_id(const hakm_tok *t, int id, char *out, int max) {
	if (id < 0 || id >= t->n_tokens) return 0;
	const char *p = t->tok_ptr[id];
	int len = t->tok_len[id], w = 0;
	for (int k = 0; k < len; ) {
		int adv; int cp = utf8_get(p + k, &adv); k += adv;
		int b = (cp < 512) ? t->cp2byte[cp] : -1;
		if (b >= 0 && w < max) out[w++] = (char)b;
	}
	return w;
}
