/* kernels_scalar.h — scalar kernel declarations */
#pragma once

void  rmsnorm   (float *out, const float *x, const float *w, int n, float eps);
void  linear    (float *out, const float *x, const float *w, int in_dim, int out_dim);
void  softmax   (float *x, int n);
float silu_f    (float x);
void  rope_apply(float *buf, int T, int n_heads, int head_dim);
