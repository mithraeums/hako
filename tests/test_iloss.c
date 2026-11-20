/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* test_iloss.c — the integer loss: exact integers pinned for a dense model and
   for experts with and without the shared ffn (weights made from integers, so
   every machine holds the same floats), the same sum however the windows are
   split, close to the float pass on a model that has learned something, the
   bad windows refused, and a model with absurd weights (1e30, inf, NaN) scored
   without an overflow (UBSan watches). ASan + UBSan. */
#include "../src/iloss.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
static void ck(int c, const char *w) { printf("  %s %s\n", c ? "ok  " : "FAIL", w); if (!c) fails++; }

static uint64_t mix(uint64_t x) { x += 0x9E3779B97F4A7C15ULL; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL; x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL; return x ^ (x >> 31); }

/* every weight k / 8192 for an integer k in [-1000, 1000]: exact in a float */
static void fill(tr_model *m, uint64_t salt) {
	float *p = tr_params(m);
	for (size_t i = 0; i < tr_n_params(m); i++) p[i] = (float)((int)(mix(i ^ salt) % 2001) - 1000) / 8192.0f;
}

enum { NT = 4000, T = 50, NW = 5 };   /* T over one chunk of 32, not two whole ones */

static int64_t score(tr_model *m, const int *ids, const int64_t *at, int n, int t) {
	il_model *I = il_new(m);
	int64_t s = -1;
	if (!I || il_nll(I, ids, at, n, t, &s)) s = -1;
	il_free(I);
	return s;
}

static void pinned(const char *what, const tr_config *c, uint64_t salt, int64_t want, const int *ids, const int64_t *at) {
	tr_model *m = tr_new(c, 1);
	if (!m) { ck(0, what); return; }
	fill(m, salt);
	int64_t all = score(m, ids, at, NW, T), parts = 0;
	for (int w = 0; w < NW; w++) { int64_t one = score(m, ids, at + w, 1, T); parts = one < 0 || parts < 0 ? -1 : parts + one; }
	char msg[160];
	snprintf(msg, sizeof msg, "%s: %lld, the pinned integer", what, (long long)all);
	ck(all == want, msg);
	ck(all == parts, "  the windows scored one at a time sum to the same");
	tr_free(m);
}

int main(void) {
	int *ids = malloc(NT * sizeof *ids);
	for (int i = 0; i < NT; i++) ids[i] = (int)(mix((uint64_t)i * 3 + 1) % 320);
	int64_t at[NW];
	for (int w = 0; w < NW; w++) at[w] = (int64_t)w * 700 + 13;

	printf("pinned integers\n");
	tr_config dense = { 320, 64, 2, 4, 2, 16, 104, 64, 10000.0f, 1e-5f, 0, 0, 0 };
	tr_config shared = { 320, 64, 2, 4, 2, 16, 40, 64, 10000.0f, 1e-5f, 4, 2, 1 };
	tr_config routed = { 320, 64, 2, 4, 2, 16, 40, 64, 10000.0f, 1e-5f, 4, 2, 0 };
	pinned("dense", &dense, 11, 136350250, ids, at);
	pinned("experts with the shared ffn", &shared, 12, 136334845, ids, at);
	pinned("experts alone", &routed, 13, 136415648, ids, at);

	printf("close to the float pass\n");
	{
		/* a pattern it can half learn: each token 7 past the last, mod 40, but one
		   in four drawn at random */
		int *pat = malloc(NT * sizeof *pat);
		pat[0] = 0;
		for (int i = 1; i < NT; i++) pat[i] = mix((uint64_t)i ^ 99) % 4 == 0 ? (int)(mix((uint64_t)i ^ 98) % 40) : (pat[i - 1] + 7) % 40;
		const tr_config *cs[2] = { &dense, &shared };
		for (int k = 0; k < 2; k++) {
			tr_model *m = tr_new(cs[k], 3);
			if (!m) { ck(0, "a model to train"); continue; }
			int in[4 * T], tg[4 * T];
			for (int s = 0; s < 60; s++) {
				for (int b = 0; b < 4; b++) {
					int a = (int)(mix((uint64_t)s * 4 + b) % (NT - T - 1));
					memcpy(in + b * T, pat + a, T * sizeof(int)); memcpy(tg + b * T, pat + a + 1, T * sizeof(int));
				}
				tr_zero_grad(m);
				tr_run(m, in, tg, 4, T, 1.0f / (4 * T));
				tr_clip(m, 1.0);
				tr_adamw(m, 3e-3f, 0.0f, 0.9f, 0.95f, 1e-8f);
			}
			double fl = 0;
			for (int w = 0; w < NW; w++) {
				memcpy(in, pat + at[w], T * sizeof(int)); memcpy(tg, pat + at[w] + 1, T * sizeof(int));
				fl += tr_run(m, in, tg, 1, T, 0);
			}
			fl /= (double)NW * T;
			int64_t r = score(m, pat, at, NW, T);
			double il = r < 0 ? -1 : il_nats(r, (int64_t)NW * T);
			char msg[160];
			snprintf(msg, sizeof msg, "%s: float %.5f, integers %.5f, learned some (under 3) and within 0.01",
			         k ? "experts" : "dense", fl, il);
			ck(fl < 3.0 && fabs(il - fl) < 0.01, msg);
			tr_free(m);
		}
		free(pat);
	}

	printf("refused\n");
	{
		tr_model *m = tr_new(&dense, 1);
		fill(m, 11);
		il_model *I = il_new(m);
		int64_t s;
		ck(I && il_nll(I, ids, at, NW, 65, &s) == -1, "a window longer than ctx");
		int bad[T + 1];
		memcpy(bad, ids, sizeof bad);
		bad[T] = 320;
		int64_t z = 0;
		ck(I && il_nll(I, bad, &z, 1, T, &s) == -1, "a token past the vocab, the last target too");
		bad[T] = -1;
		ck(I && il_nll(I, bad, &z, 1, T, &s) == -1, "a negative token");
		ck(I && il_nll(I, ids, at, 0, T, &s) == 0 && s == 0, "no windows: 0");
		il_free(I);
		tr_free(m);
	}

	printf("absurd weights\n");
	{
		const tr_config *cs[2] = { &dense, &shared };
		for (int k = 0; k < 2; k++) {
			tr_model *m = tr_new(cs[k], 1);
			fill(m, 21);
			float *p = tr_params(m);
			size_t n = tr_n_params(m);
			for (size_t i = 0; i < n; i += 97) p[i] = (i / 97) % 4 == 0 ? 1e30f : (i / 97) % 4 == 1 ? -1e30f : (i / 97) % 4 == 2 ? INFINITY : NAN;
			for (size_t i = 5; i < n; i += 389) p[i] = 1e-40f;   /* subnormal */
			int64_t s = score(m, ids, at, NW, T);
			ck(s >= 0, k ? "experts: scored, no overflow" : "dense: scored, no overflow");
			tr_free(m);
		}
	}

	free(ids);
	printf(fails ? "test_iloss: %d FAILED\n" : "test_iloss: all ok\n", fails);
	return fails != 0;
}
