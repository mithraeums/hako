/* sample.c — greedy / temperature / top-k / top-p sampling + rep penalty. */
#include "sample.h"
#include "nn.h"          /* softmax */

#include <stdlib.h>

void sampler_init(sampler *s, float temp, float top_p, int top_k, uint64_t seed) {
	s->temp = temp; s->top_p = top_p; s->top_k = top_k;
	s->rng = seed ? seed : 0x9e3779b97f4a7c15ULL;
	s->rep_penalty = 1.0f; s->rep_window = 0;
	s->hlen = 0; s->hpos = 0;
}

void sampler_set_rep(sampler *s, float penalty, int window) {
	if (window > SAMPLER_HIST_MAX) window = SAMPLER_HIST_MAX;
	s->rep_penalty = (penalty > 1.0f) ? penalty : 1.0f;
	s->rep_window  = (window  > 0)    ? window  : 0;
	s->hlen = 0; s->hpos = 0;
}

void sampler_reset_hist(sampler *s) { s->hlen = 0; s->hpos = 0; }

static void hist_push(sampler *s, int id) {
	if (s->rep_window <= 0) return;
	s->hist[s->hpos] = id;
	s->hpos = (s->hpos + 1) % SAMPLER_HIST_MAX;
	if (s->hlen < SAMPLER_HIST_MAX) s->hlen++;
}

/* Divide the logit of each token seen in the recent window by rep_penalty (once
 * per unique id). Applied on raw logits, before temperature. */
static void apply_rep_penalty(sampler *s, float *logits) {
	if (s->rep_penalty <= 1.0f || s->rep_window <= 0 || s->hlen == 0) return;
	int w = s->hlen < s->rep_window ? s->hlen : s->rep_window;
	for (int j = 0; j < w; j++) {
		int id = s->hist[(s->hpos - 1 - j + SAMPLER_HIST_MAX) % SAMPLER_HIST_MAX];
		int dup = 0;
		for (int p = 0; p < j; p++)
			if (s->hist[(s->hpos - 1 - p + SAMPLER_HIST_MAX) % SAMPLER_HIST_MAX] == id) { dup = 1; break; }
		if (dup) continue;
		float l = logits[id];
		logits[id] = l > 0.0f ? l / s->rep_penalty : l * s->rep_penalty;
	}
}

static inline uint64_t xorshift(uint64_t *st) {
	uint64_t x = *st;
	x ^= x << 13; x ^= x >> 7; x ^= x << 17;
	return *st = x;
}
static inline float frand(uint64_t *st) {
	return (float)((xorshift(st) >> 40) / (double)(1ULL << 24));  /* [0,1) */
}

typedef struct { float p; int idx; } pidx;
static int cmp_desc(const void *a, const void *b) {
	float d = ((const pidx *)b)->p - ((const pidx *)a)->p;
	return (d > 0) - (d < 0);
}

int sample_token(sampler *s, float *logits, int n) {
	apply_rep_penalty(s, logits);

	if (s->temp <= 0.0f) {                 /* greedy */
		int best = 0; float bv = logits[0];
		for (int i = 1; i < n; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
		hist_push(s, best);
		return best;
	}

	for (int i = 0; i < n; i++) logits[i] /= s->temp;
	softmax(logits, n);                    /* logits now probs */

	/* candidate set: full vocab unless top-k limits it */
	int k = (s->top_k > 0 && s->top_k < n) ? s->top_k : n;
	pidx *c = malloc((size_t)n * sizeof(pidx));
	if (!c) {                              /* OOM: fall back to argmax, don't crash */
		int best = 0; float bv = logits[0];
		for (int i = 1; i < n; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
		hist_push(s, best);
		return best;
	}
	for (int i = 0; i < n; i++) { c[i].p = logits[i]; c[i].idx = i; }
	/* full sort is simplest + correct; vocab sort is cheap vs a forward pass */
	qsort(c, n, sizeof(pidx), cmp_desc);

	/* top-p nucleus over the sorted (and top-k truncated) head */
	float cum = 0.0f; int m = k;
	if (s->top_p > 0.0f && s->top_p < 1.0f) {
		float acc = 0.0f; m = 0;
		for (int i = 0; i < k; i++) { acc += c[i].p; m++; if (acc >= s->top_p) break; }
	}
	for (int i = 0; i < m; i++) cum += c[i].p;

	float r = frand(&s->rng) * cum, run = 0.0f;
	int chosen = c[0].idx;
	for (int i = 0; i < m; i++) { run += c[i].p; if (run >= r) { chosen = c[i].idx; break; } }

	free(c);
	hist_push(s, chosen);
	return chosen;
}
