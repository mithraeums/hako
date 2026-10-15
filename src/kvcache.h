/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* kvcache.h — persist a KV-cache prefix to disk, so the next process skips
 * the prefill it already paid for.
 *
 * The agent's system prompt is ~2,200 tokens and identical every turn; the
 * resident --serve loop reuses it within a process, but a fresh process (every
 * agent launch) prefilled it again — the whole "warming the model" wait. This
 * writes the first n_prefix positions of every layer's K and V to
 * ~/.hako/cache/kv/<key>.kvc and reads them back when the same model sees the
 * same prefix tokens again.
 *
 * File = header + prefix token ids + K rows + V rows (layer-major, n_prefix
 * rows of kv_dim each). Only the used rows are written, never the empty tail
 * of max_seq, and the file is independent of max_seq: it loads into any
 * context that has room for n_prefix.
 *
 * The key names the model (size + mtime), the KV element type, the layout
 * (n_layers, kv_dim) and an FNV-1a hash of the prefix token ids. Everything in
 * the key is ALSO in the header and re-checked on load; the tokens are stored
 * and compared verbatim, so a hash collision cannot serve the wrong prefix. A
 * bad, short or foreign file is a miss, never a crash. */
#pragma once
#include "model.h"

/* Try to fill the cache with a stored prefix of toks[0..n). Returns the number
 * of positions restored (c->pos is set to it), or 0 on a miss. The caller then
 * prefills from that position. n is the split point (see kvc_prefix_len). */
int kvc_load(hakm_ctx *c, const char *model_path, const int *toks, int n);

/* Store positions [0..n) of the live cache for toks[0..n). Returns 0 on
 * success, -1 if the cache could not be written (disk is optional). */
int kvc_save(const hakm_ctx *c, const char *model_path, const int *toks, int n);

/* Cache is bypassed when HAKM_KV_CACHE=0 is in the environment or
 * kvc_disable() was called (the --no-kv-cache flag). */
void kvc_disable(void);
int  kvc_enabled(void);

/* Path of the entry for this (model, prefix); for diagnostics. Returns 0 and
 * an empty string when no cache dir can be resolved. */
int kvc_path(char *out, size_t cap, const hakm_ctx *c, const char *model_path,
             const int *toks, int n);

/* Count (remove=0) or delete (remove=1) the cache entries that belong to the
 * model file with this size + mtime — the identity the entries are keyed on.
 * Returns the number of entries seen. Used by `hakm list` and `hakm rm`. */
/* Trim the cache directory to HAKM_KV_DIR_MAX_MB (default 1024, 0 = never),
   deleting oldest first. `keep` is a path that is never evicted; pass NULL to
   consider everything. Called automatically after a successful kvc_save. */
void kvc_evict(const char *keep);

int kvc_for_model(unsigned long long size, unsigned long long mtime, int remove);
