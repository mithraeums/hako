/* sample.h — logit sampling: greedy, temperature, top-k, top-p. */
#pragma once
#include <stdint.h>

typedef struct {
	float    temp;    /* <= 0 => greedy argmax */
	float    top_p;   /* 0 => disabled */
	int      top_k;   /* 0 => disabled */
	uint64_t rng;     /* xorshift state; seed non-zero */
} sampler;

void sampler_init(sampler *s, float temp, float top_p, int top_k, uint64_t seed);

/* Pick a token id from logits[0..n). Mutates s->rng. */
int sample_token(sampler *s, float *logits, int n);
