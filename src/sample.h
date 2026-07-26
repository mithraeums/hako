/* sample.h — logit sampling: greedy, temperature, top-k, top-p, rep penalty. */
#pragma once
#include <stdint.h>

#define SAMPLER_HIST_MAX 256   /* ring capacity for the repetition window */

typedef struct {
	float    temp;    /* <= 0 => greedy argmax */
	float    top_p;   /* 0 => disabled */
	int      top_k;   /* 0 => disabled */
	uint64_t rng;     /* xorshift state; seed non-zero */

	/* Repetition penalty (CTRL-style): recent token logits are divided by
	 * rep_penalty (positive logits) or multiplied (negative). 1.0 => OFF.
	 * Default OFF so greedy structured output (tool-call JSON) stays exact. */
	float    rep_penalty;
	int      rep_window;                  /* recent tokens penalized; 0 => OFF */
	int      hist[SAMPLER_HIST_MAX];      /* ring of recently sampled ids */
	int      hlen;                        /* entries written (caps at MAX) */
	int      hpos;                        /* ring write cursor */
} sampler;

void sampler_init(sampler *s, float temp, float top_p, int top_k, uint64_t seed);

/* Enable/disable the repetition penalty and clear the history ring. penalty<=1
 * or window<=0 disables it. window is clamped to SAMPLER_HIST_MAX. */
void sampler_set_rep(sampler *s, float penalty, int window);

/* Clear the recent-token history (call at the start of each new reply). */
void sampler_reset_hist(sampler *s);

/* Pick a token id from logits[0..n). Mutates s->rng and the history ring. */
int sample_token(sampler *s, float *logits, int n);
