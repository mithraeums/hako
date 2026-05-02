/* sample.h */
#pragma once
#include "../include/mini.h"

/* Sample next token from logits[0..vocab-1] (may modify logits in-place).
 * Returns token ID. */
int sample_token(float *logits, int vocab, const mini_sample_opts *opts);
