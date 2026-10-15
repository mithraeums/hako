/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* kvcache.c — on-disk KV prefix. See kvcache.h for the format and the why. */
#include "kvcache.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#define KVC_MAGIC   "HKVC"
#define KVC_VERSION 1u

/* Fixed-width, little-endian on every box the engine ships to. Not packed:
   the fields are laid out so there is no padding to begin with. */
typedef struct {
	char     magic[4];
	uint32_t version;
	uint32_t n_layers;
	uint32_t kv_dim;
	uint32_t elem_size;     /* sizeof(hakm_kv_t): 2 = f16, 4 = f32 */
	uint32_t n_prefix;
	uint64_t model_size;
	uint64_t model_mtime;
	uint64_t tok_hash;      /* FNV-1a over the prefix token ids */
	uint64_t reserved;
} kvc_header;               /* 56 bytes */

static int kvc_off = 0;

void kvc_disable(void) { kvc_off = 1; }

int kvc_enabled(void) {
	if (kvc_off) return 0;
	const char *e = getenv("HAKM_KV_CACHE");
	return !(e && *e == '0');
}

static uint64_t fnv1a(const void *p, size_t n, uint64_t h) {
	const unsigned char *b = p;
	for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
	return h;
}

static int model_ident(const char *path, uint64_t *size, uint64_t *mtime) {
	struct stat st;
	if (!path || stat(path, &st) != 0) return -1;
	*size  = (uint64_t)st.st_size;
	*mtime = (uint64_t)st.st_mtime;
	return 0;
}

/* $HAKO_HOME/cache/kv, else $HOME/.hako/cache/kv. Made on demand. */
static int cache_dir(char *out, size_t cap) {
	const char *hh = getenv("HAKO_HOME");
	int n;
	if (hh && *hh) n = snprintf(out, cap, "%s", hh);
	else {
		const char *home = getenv("HOME");
		if (!home || !*home) return -1;
		n = snprintf(out, cap, "%s/.hako", home);
	}
	if (n < 0 || (size_t)n >= cap) return -1;
	mkdir(out, 0700);
	size_t l = strlen(out);
	if (l + 10 >= cap) return -1;
	strcat(out, "/cache"); mkdir(out, 0700);
	strcat(out, "/kv");    mkdir(out, 0700);
	struct stat st;
	return (stat(out, &st) == 0 && S_ISDIR(st.st_mode)) ? 0 : -1;
}

static int fill_header(kvc_header *h, const hakm_ctx *c, const char *model_path,
                       const int *toks, int n) {
	memset(h, 0, sizeof(*h));
	if (n <= 0 || !toks) return -1;
	if (model_ident(model_path, &h->model_size, &h->model_mtime) != 0) return -1;
	memcpy(h->magic, KVC_MAGIC, 4);
	h->version   = KVC_VERSION;
	h->n_layers  = (uint32_t)c->n_layers;
	h->kv_dim    = (uint32_t)c->kv_dim;
	h->elem_size = (uint32_t)sizeof(hakm_kv_t);
	h->n_prefix  = (uint32_t)n;
	h->tok_hash  = fnv1a(toks, (size_t)n * sizeof(int), 1469598103934665603ull);
	return 0;
}

int kvc_path(char *out, size_t cap, const hakm_ctx *c, const char *model_path,
             const int *toks, int n) {
	out[0] = '\0';
	kvc_header h;
	char dir[1024];
	if (fill_header(&h, c, model_path, toks, n) != 0 || cache_dir(dir, sizeof(dir)) != 0) return 0;
	/* The file name is the key: everything the header will re-check. */
	uint64_t k = fnv1a(&h.model_size, sizeof(h.model_size), 1469598103934665603ull);
	k = fnv1a(&h.model_mtime, sizeof(h.model_mtime), k);
	k = fnv1a(&h.n_layers,    sizeof(h.n_layers),    k);
	k = fnv1a(&h.kv_dim,      sizeof(h.kv_dim),      k);
	k = fnv1a(&h.elem_size,   sizeof(h.elem_size),   k);
	k = fnv1a(&h.tok_hash,    sizeof(h.tok_hash),    k);
	int w = snprintf(out, cap, "%s/%016llx.kvc", dir, (unsigned long long)k);
	if (w < 0 || (size_t)w >= cap) { out[0] = '\0'; return 0; }
	return 1;
}

/* Bytes of one K (or V) block: n_layers × n_prefix × kv_dim elements.
   Computed in 64 bits and refused if it cannot be a size_t — a 32-bit box
   must never wrap this into a small read that "succeeds". */
static int block_bytes(const kvc_header *h, size_t *out) {
	uint64_t b = (uint64_t)h->n_layers * h->n_prefix * h->kv_dim * h->elem_size;
	if (b == 0 || b > (uint64_t)SIZE_MAX / 4) return -1;
	*out = (size_t)b;
	return 0;
}

/* An entry is 2 × block_bytes. sho at 2.2k tokens is 83MB (f16) / 159MB (f32,
   which is what a box without F16C — iSh — gets); a 7B at a long prefix would
   be several hundred. Cap what ONE entry may cost; kvc_evict() below caps the
   directory as a whole. HAKM_KV_CACHE_MAX_MB overrides. */
static int too_big(size_t block) {
	uint64_t mb = (uint64_t)block * 2 / 1048576;
	const char *e = getenv("HAKM_KV_CACHE_MAX_MB");
	uint64_t cap = e ? strtoull(e, NULL, 10) : 512;
	return cap && mb > cap;
}

int kvc_load(hakm_ctx *c, const char *model_path, const int *toks, int n) {
	if (!kvc_enabled() || n <= 0 || n > c->max_seq) return 0;
	char path[1200];
	if (!kvc_path(path, sizeof(path), c, model_path, toks, n)) return 0;
	FILE *f = fopen(path, "rb");
	if (!f) return 0;

	kvc_header want, got;
	fill_header(&want, c, model_path, toks, n);
	int ok = fread(&got, sizeof(got), 1, f) == 1
	      && memcmp(got.magic, KVC_MAGIC, 4) == 0
	      && got.version   == KVC_VERSION
	      && got.n_layers  == want.n_layers
	      && got.kv_dim    == want.kv_dim
	      && got.elem_size == want.elem_size
	      && got.n_prefix  == want.n_prefix
	      && got.model_size  == want.model_size
	      && got.model_mtime == want.model_mtime
	      && got.tok_hash    == want.tok_hash;
	size_t bb = 0;
	if (ok) ok = block_bytes(&got, &bb) == 0;
	/* Tokens are compared verbatim, not only by hash. */
	int *stored = ok ? malloc((size_t)n * sizeof(int)) : NULL;
	if (ok) ok = stored && fread(stored, sizeof(int), (size_t)n, f) == (size_t)n
	          && memcmp(stored, toks, (size_t)n * sizeof(int)) == 0;
	free(stored);

	/* Layer-major rows straight into the strided live cache. Any short read
	   leaves the cache partly written, so nothing is trusted until the whole
	   file landed: pos stays 0 on failure and the prefill overwrites it all. */
	size_t row = (size_t)c->kv_dim * sizeof(hakm_kv_t);
	size_t span = (size_t)n * row;
	for (int l = 0; ok && l < c->n_layers; l++)
		ok = fread(c->k_cache + (size_t)l * c->max_seq * c->kv_dim, 1, span, f) == span;
	for (int l = 0; ok && l < c->n_layers; l++)
		ok = fread(c->v_cache + (size_t)l * c->max_seq * c->kv_dim, 1, span, f) == span;
	fclose(f);
	if (!ok) { c->pos = 0; return 0; }
	c->pos = n;
	return n;
}

int kvc_save(const hakm_ctx *c, const char *model_path, const int *toks, int n) {
	if (!kvc_enabled() || n <= 0 || n > c->pos) return -1;
	char path[1200], tmp[1240];
	if (!kvc_path(path, sizeof(path), c, model_path, toks, n)) return -1;
	kvc_header h;
	if (fill_header(&h, c, model_path, toks, n) != 0) return -1;
	size_t bb;
	if (block_bytes(&h, &bb) != 0 || too_big(bb)) return -1;
	snprintf(tmp, sizeof(tmp), "%s.%ld.tmp", path, (long)getpid());
	FILE *f = fopen(tmp, "wb");
	if (!f) return -1;

	int ok = fwrite(&h, sizeof(h), 1, f) == 1
	      && fwrite(toks, sizeof(int), (size_t)n, f) == (size_t)n;
	size_t span = (size_t)n * c->kv_dim * sizeof(hakm_kv_t);
	for (int l = 0; ok && l < c->n_layers; l++)
		ok = fwrite(c->k_cache + (size_t)l * c->max_seq * c->kv_dim, 1, span, f) == span;
	for (int l = 0; ok && l < c->n_layers; l++)
		ok = fwrite(c->v_cache + (size_t)l * c->max_seq * c->kv_dim, 1, span, f) == span;
	ok = (fclose(f) == 0) && ok;
	/* Whole file or nothing: readers only ever see the renamed result. */
	if (!ok || rename(tmp, path) != 0) { unlink(tmp); return -1; }
	kvc_evict(path);      /* keep the directory bounded; this entry is exempt */
	return 0;
}

/* Keep the cache directory under a budget, oldest first.

   One file per distinct system prompt and nothing ever deleted them: this box
   had 359MB across five entries before anyone looked, and the device this
   engine most needs to behave on is a phone. Ordering is by mtime — atime is
   off on most filesystems now, so it would rank everything equally.

   Called after a successful save, with that file exempt: it is the one thing
   we know is wanted. HAKM_KV_DIR_MAX_MB overrides; 0 disables eviction. */
#define KVC_DIR_MAX_MB_DEFAULT 1024ULL
#define KVC_EVICT_MAX 256

void kvc_evict(const char *keep)
{
	const char *e = getenv("HAKM_KV_DIR_MAX_MB");
	unsigned long long budget_mb = e ? strtoull(e, NULL, 10) : KVC_DIR_MAX_MB_DEFAULT;
	if (budget_mb == 0) return;
	unsigned long long budget = budget_mb * 1048576ULL;

	char dir[1024];
	if (cache_dir(dir, sizeof(dir)) != 0) return;
	DIR *d = opendir(dir);
	if (!d) return;

	struct ent { char path[1200]; unsigned long long size; long long mtime; };
	static struct ent ents[KVC_EVICT_MAX];
	int n = 0;
	unsigned long long total = 0;
	struct dirent *de;
	while ((de = readdir(d)) != NULL && n < KVC_EVICT_MAX) {
		size_t l = strlen(de->d_name);
		if (l < 5 || strcmp(de->d_name + l - 4, ".kvc") != 0) continue;
		char path[1200];
		if ((size_t)snprintf(path, sizeof(path), "%s/%s", dir, de->d_name) >= sizeof(path)) continue;
		struct stat st;
		if (stat(path, &st) != 0) continue;
		if (keep && strcmp(path, keep) == 0) { total += (unsigned long long)st.st_size; continue; }
		snprintf(ents[n].path, sizeof(ents[n].path), "%s", path);
		ents[n].size  = (unsigned long long)st.st_size;
		ents[n].mtime = (long long)st.st_mtime;
		total += ents[n].size;
		n++;
	}
	closedir(d);
	if (total <= budget) return;

	/* oldest first; n is tiny, so an insertion sort is the whole story */
	for (int i = 1; i < n; i++) {
		struct ent t = ents[i];
		int j = i - 1;
		while (j >= 0 && ents[j].mtime > t.mtime) { ents[j+1] = ents[j]; j--; }
		ents[j+1] = t;
	}
	for (int i = 0; i < n && total > budget; i++)
		if (unlink(ents[i].path) == 0) total -= ents[i].size;
}

int kvc_for_model(unsigned long long size, unsigned long long mtime, int remove) {
	char dir[1024];
	if (cache_dir(dir, sizeof(dir)) != 0) return 0;
	DIR *d = opendir(dir);
	if (!d) return 0;
	int n = 0;
	struct dirent *de;
	while ((de = readdir(d)) != NULL) {
		size_t l = strlen(de->d_name);
		if (l < 5 || strcmp(de->d_name + l - 4, ".kvc") != 0) continue;
		char path[1200];
		snprintf(path, sizeof(path), "%s/%s", dir, de->d_name);
		FILE *f = fopen(path, "rb");
		if (!f) continue;
		kvc_header h;
		int match = fread(&h, sizeof(h), 1, f) == 1 && memcmp(h.magic, KVC_MAGIC, 4) == 0
		         && h.model_size == size && h.model_mtime == mtime;
		fclose(f);
		if (!match) continue;
		n++;
		if (remove) unlink(path);
	}
	closedir(d);
	return n;
}
