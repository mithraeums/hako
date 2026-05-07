/*
 * sample.c — sampling strategies: greedy, temperature, top-k, top-p.
 */
#include "sample.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t rng_state;

static void rng_seed(uint64_t seed) {
	rng_state = seed ? seed : (uint64_t)time(NULL);
}

/* xorshift64 */
static uint64_t rng_u64(void) {
	rng_state ^= rng_state << 13;
	rng_state ^= rng_state >> 7;
	rng_state ^= rng_state << 17;
	return rng_state;
}

static float rng_f32(void) {
	return (float)(rng_u64() >> 11) / (float)(1ULL << 53);
}

/* in-place softmax with temperature (modifies logits) */
static void apply_temperature(float *logits, int n, float temp) {
	if (temp <= 0.0f) return;  /* greedy handled separately */
	float inv = 1.0f / temp;
	float max = logits[0];
	for (int i = 1; i < n; i++) if (logits[i] > max) max = logits[i];
	float sum = 0.0f;
	for (int i = 0; i < n; i++) { logits[i] = expf((logits[i] - max) * inv); sum += logits[i]; }
	float inv_sum = 1.0f / sum;
	for (int i = 0; i < n; i++) logits[i] *= inv_sum;
}

int sample_token(float *logits, int vocab, const mini_sample_opts *opts) {
	if (opts) rng_seed(opts->seed);

	/* greedy */
	if (!opts || opts->temperature <= 0.0f) {
		int best = 0;
		for (int i = 1; i < vocab; i++)
			if (logits[i] > logits[best]) best = i;
		return best;
	}

	/* convert to probabilities */
	apply_temperature(logits, vocab, opts->temperature);

	/* top-k filtering */
	if (opts->top_k > 0 && opts->top_k < vocab) {
		/* find k-th largest via partial sort on indices */
		int k = opts->top_k;
		/* simple O(V*k) selection — fine for small vocab (nano: 256) */
		int *idx = (int *)malloc((size_t)vocab * sizeof(int));
		if (!idx) goto multinomial;  /* fallback if alloc fails */
		for (int i = 0; i < vocab; i++) idx[i] = i;
		for (int i = 0; i < k; i++) {
			int mx = i;
			for (int j = i + 1; j < vocab; j++)
				if (logits[idx[j]] > logits[idx[mx]]) mx = j;
			int tmp = idx[i]; idx[i] = idx[mx]; idx[mx] = tmp;
		}
		/* zero out tokens beyond top-k */
		for (int i = k; i < vocab; i++) logits[idx[i]] = 0.0f;
		free(idx);
		/* renormalize */
		float sum = 0.0f;
		for (int i = 0; i < vocab; i++) sum += logits[i];
		float inv = 1.0f / sum;
		for (int i = 0; i < vocab; i++) logits[i] *= inv;
	}

	/* top-p (nucleus) filtering */
	if (opts->top_p > 0.0f && opts->top_p < 1.0f) {
		/* sort descending, accumulate until sum >= top_p, zero rest */
		int *idx = (int *)malloc((size_t)vocab * sizeof(int));
		if (!idx) goto multinomial;
		for (int i = 0; i < vocab; i++) idx[i] = i;
		/* simple insertion sort O(V^2) — fine for small V */
		for (int i = 1; i < vocab; i++) {
			int key = idx[i]; int j = i - 1;
			while (j >= 0 && logits[idx[j]] < logits[key]) {
				idx[j+1] = idx[j]; j--;
			}
			idx[j+1] = key;
		}
		float cum = 0.0f;
		int cutoff = vocab;
		for (int i = 0; i < vocab; i++) {
			cum += logits[idx[i]];
			if (cum >= opts->top_p) { cutoff = i + 1; break; }
		}
		for (int i = cutoff; i < vocab; i++) logits[idx[i]] = 0.0f;
		free(idx);
		float sum = 0.0f;
		for (int i = 0; i < vocab; i++) sum += logits[i];
		float inv = 1.0f / sum;
		for (int i = 0; i < vocab; i++) logits[i] *= inv;
	}

multinomial:;
	/* multinomial sample from probabilities */
	float r   = rng_f32();
	float cum = 0.0f;
	for (int i = 0; i < vocab; i++) {
		cum += logits[i];
		if (r <= cum) return i;
	}
	return vocab - 1;  /* rounding fallback */
}
