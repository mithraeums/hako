/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* test_kvc.c — the on-disk KV prefix must round-trip exactly, and must be a
 * MISS (never a crash, never a wrong hit) for a corrupted header, a truncated
 * body, a different model, a different prefix and a bypass. Synthetic context,
 * no weights; runs under ASan from `make test_kvc`. */
#include "../src/kvcache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int fails = 0;
#define CK(cond, msg) do { if (cond) printf("  ok   %s\n", msg); \
	else { printf("  FAIL %s\n", msg); fails++; } } while (0)

static hakm_ctx *ctx_new(int n_layers, int kv_dim, int max_seq) {
	hakm_ctx *c = calloc(1, sizeof(*c));
	c->n_layers = n_layers; c->kv_dim = kv_dim; c->max_seq = max_seq;
	c->k_cache = calloc((size_t)n_layers * max_seq * kv_dim, sizeof(hakm_kv_t));
	c->v_cache = calloc((size_t)n_layers * max_seq * kv_dim, sizeof(hakm_kv_t));
	return c;
}
static void ctx_free(hakm_ctx *c) { free(c->k_cache); free(c->v_cache); free(c); }

static void fill(hakm_ctx *c, int n, unsigned seed) {
	for (int l = 0; l < c->n_layers; l++)
		for (int s = 0; s < n; s++)
			for (int d = 0; d < c->kv_dim; d++) {
				size_t i = ((size_t)l * c->max_seq + s) * c->kv_dim + d;
				c->k_cache[i] = (hakm_kv_t)(seed + i * 7 + 1);
				c->v_cache[i] = (hakm_kv_t)(seed + i * 13 + 2);
			}
	c->pos = n;
}

static int same_prefix(const hakm_ctx *a, const hakm_ctx *b, int n) {
	for (int l = 0; l < a->n_layers; l++) {
		size_t off = (size_t)l * a->max_seq * a->kv_dim, off2 = (size_t)l * b->max_seq * b->kv_dim;
		size_t span = (size_t)n * a->kv_dim * sizeof(hakm_kv_t);
		if (memcmp(a->k_cache + off, b->k_cache + off2, span)) return 0;
		if (memcmp(a->v_cache + off, b->v_cache + off2, span)) return 0;
	}
	return 1;
}

static void touch(const char *path, const char *body) {
	FILE *f = fopen(path, "wb"); fputs(body, f); fclose(f);
}

static long file_size(const char *p) { struct stat st; return stat(p, &st) == 0 ? (long)st.st_size : -1; }

int main(void) {
	char home[256];
	snprintf(home, sizeof(home), "/tmp/hakm_kvc_%ld", (long)getpid());
	mkdir(home, 0700);
	setenv("HAKO_HOME", home, 1);
	char model[300], model2[300];
	snprintf(model, sizeof(model), "%s/fake.mlf2", home);
	snprintf(model2, sizeof(model2), "%s/other.mlf2", home);
	touch(model, "not really weights, only an identity");
	touch(model2, "a different model");

	int toks[40];
	for (int i = 0; i < 40; i++) toks[i] = 1000 + i * 3;
	const int N = 32;

	printf("round trip\n");
	hakm_ctx *a = ctx_new(3, 8, 64);
	fill(a, N, 17);
	CK(kvc_save(a, model, toks, N) == 0, "save succeeds");
	char path[1200];
	CK(kvc_path(path, sizeof(path), a, model, toks, N), "entry path resolves");
	long want = 56 + N * 4 + 2L * 3 * N * 8 * (long)sizeof(hakm_kv_t);
	CK(file_size(path) == want, "file holds header + tokens + used rows only (no max_seq tail)");

	hakm_ctx *b = ctx_new(3, 8, 128);          /* a bigger context: still loads */
	CK(kvc_load(b, model, toks, N) == N, "load into a different max_seq restores N positions");
	CK(b->pos == N, "pos set to the prefix length");
	CK(same_prefix(a, b, N), "K and V rows are bit-identical after the round trip");
	ctx_free(b);

	printf("misses\n");
	b = ctx_new(3, 8, 64);
	int other[40]; memcpy(other, toks, sizeof(toks)); other[N - 1] ^= 1;
	CK(kvc_load(b, model, other, N) == 0 && b->pos == 0, "different prefix is a miss");
	CK(kvc_load(b, model, toks, N - 1) == 0, "shorter prefix is a miss (its own key)");
	CK(kvc_load(b, model2, toks, N) == 0, "different model file is a miss");
	hakm_ctx *w = ctx_new(3, 16, 64);
	CK(kvc_load(w, model, toks, N) == 0, "different kv_dim is a miss");
	ctx_free(w);
	hakm_ctx *tiny = ctx_new(3, 8, 16);
	CK(kvc_load(tiny, model, toks, N) == 0, "prefix longer than max_seq is refused");
	ctx_free(tiny);

	setenv("HAKM_KV_CACHE", "0", 1);
	CK(kvc_load(b, model, toks, N) == 0, "HAKM_KV_CACHE=0 bypasses the load");
	CK(kvc_save(a, model, toks, N) == -1, "HAKM_KV_CACHE=0 bypasses the save");
	unsetenv("HAKM_KV_CACHE");
	CK(kvc_load(b, model, toks, N) == N, "cache is back once the bypass is lifted");

	printf("size cap\n");
	setenv("HAKM_KV_CACHE_MAX_MB", "1", 1);   /* this entry is far under 1MB */
	CK(kvc_save(a, model, toks, N) == 0, "an entry under the cap still saves");
	setenv("HAKM_KV_CACHE_MAX_MB", "0", 1);
	CK(kvc_save(a, model, toks, N) == 0, "cap 0 means no cap");
	{	/* a cap that never refuses anything is not a tested cap: 8MB entry, 1MB cap */
		int big_n = 1024;
		hakm_ctx *big = ctx_new(8, 256, big_n);
		int *bt = malloc((size_t)big_n * sizeof(int));
		for (int i = 0; i < big_n; i++) bt[i] = i * 5 + 1;
		fill(big, big_n, 3);
		setenv("HAKM_KV_CACHE_MAX_MB", "1", 1);
		CK(kvc_save(big, model, bt, big_n) == -1, "an oversize entry is refused by the cap");
		setenv("HAKM_KV_CACHE_MAX_MB", "64", 1);
		CK(kvc_save(big, model, bt, big_n) == 0, "…and saves once the cap allows it");
		char bp[1200];
		if (kvc_path(bp, sizeof(bp), big, model, bt, big_n)) unlink(bp);
		free(bt); ctx_free(big);
	}
	unsetenv("HAKM_KV_CACHE_MAX_MB");

	printf("damage\n");
	/* corrupt a header byte */
	FILE *f = fopen(path, "r+b"); fseek(f, 5, SEEK_SET); fputc('x', f); fclose(f);
	memset(b->k_cache, 0, (size_t)3 * 64 * 8 * sizeof(hakm_kv_t)); b->pos = 0;
	CK(kvc_load(b, model, toks, N) == 0 && b->pos == 0, "corrupted header is a miss");
	CK(kvc_save(a, model, toks, N) == 0 && kvc_load(b, model, toks, N) == N, "rewritten after the bad file");
	/* truncate the body */
	f = fopen(path, "r+b"); fclose(f); truncate(path, want - 100);
	b->pos = 0;
	CK(kvc_load(b, model, toks, N) == 0 && b->pos == 0, "truncated body is a miss, pos stays 0");
	/* model changed on disk (new mtime + size) */
	kvc_save(a, model, toks, N);
	sleep(1);
	touch(model, "same name, new weights, new size");
	CK(kvc_load(b, model, toks, N) == 0, "model file changed → old entry ignored");
	CK(kvc_save(a, model, toks, N) == 0 && kvc_load(b, model, toks, N) == N, "…and a fresh entry works");

	/* --- eviction: one file per distinct system prompt, nothing ever deleted
	   them. 359MB had piled up on the author's box before anyone looked, and
	   the device this has to behave on is a phone. --- */
	{
		char kvdir[400];
		snprintf(kvdir, sizeof(kvdir), "%s/cache/kv", home);
		char cmd[500]; snprintf(cmd, sizeof(cmd), "rm -f '%s'/*.kvc", kvdir); (void)system(cmd);

		/* three 1MB stand-ins with distinct, increasing mtimes */
		char p1[500], p2[500], p3[500];
		snprintf(p1, sizeof(p1), "%s/aaaa000000000001.kvc", kvdir);
		snprintf(p2, sizeof(p2), "%s/aaaa000000000002.kvc", kvdir);
		snprintf(p3, sizeof(p3), "%s/aaaa000000000003.kvc", kvdir);
		const char *paths[3] = { p1, p2, p3 };
		for (int i = 0; i < 3; i++) {
			FILE *f = fopen(paths[i], "wb");
			CK(f != NULL, "eviction fixture created");
			if (!f) break;
			static char blob[1048576];
			fwrite(blob, 1, sizeof(blob), f);
			fclose(f);
			if (i < 2) sleep(1);            /* distinct mtimes, oldest first */
		}

		setenv("HAKM_KV_DIR_MAX_MB", "0", 1);        /* 0 disables */
		kvc_evict(NULL);
		CK(file_size(p1) > 0, "a budget of 0 evicts nothing");

		setenv("HAKM_KV_DIR_MAX_MB", "2", 1);        /* 3MB present, keep 2 */
		kvc_evict(NULL);
		CK(file_size(p1) < 0, "over budget: the OLDEST entry goes first");
		CK(file_size(p3) > 0, "the newest entry survives");

		/* the file just written is exempt even when it is the oldest */
		setenv("HAKM_KV_DIR_MAX_MB", "1", 1);
		kvc_evict(p2);
		CK(file_size(p2) > 0, "the keep path is never evicted");
		unsetenv("HAKM_KV_DIR_MAX_MB");
	}

	ctx_free(a); ctx_free(b);
	char cmd[400]; snprintf(cmd, sizeof(cmd), "rm -rf '%s'", home); (void)system(cmd);
	printf(fails ? "test_kvc: %d FAILED\n" : "test_kvc: all passed\n", fails);
	return fails ? 1 : 0;
}
