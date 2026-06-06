/* sample.c — greedy / temperature / top-k / top-p sampling. */
#include "sample.h"
#include "nn.h"          /* softmax */

#include <stdlib.h>

void sampler_init(sampler *s, float temp, float top_p, int top_k, uint64_t seed) {
	s->temp = temp; s->top_p = top_p; s->top_k = top_k;
	s->rng = seed ? seed : 0x9e3779b97f4a7c15ULL;
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
	if (s->temp <= 0.0f) {                 /* greedy */
		int best = 0; float bv = logits[0];
		for (int i = 1; i < n; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
		return best;
	}

	for (int i = 0; i < n; i++) logits[i] /= s->temp;
	softmax(logits, n);                    /* logits now probs */

	/* candidate set: full vocab unless top-k limits it */
	int k = (s->top_k > 0 && s->top_k < n) ? s->top_k : n;
	pidx *c = malloc((size_t)n * sizeof(pidx));
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
	return chosen;
}
