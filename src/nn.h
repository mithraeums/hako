/* nn.h — scalar compute kernels (correctness-first; SIMD is a later pass). */
#pragma once
#include "loader.h"

/* y[d] = x[d] / rms(x) * w[d], rms over d_model with eps. */
void rmsnorm(float *y, const float *x, const float *w, int n, float eps);

/* in-place softmax over n elements. */
void softmax(float *x, int n);

/* SiLU(x) = x * sigmoid(x). */
float silu_f(float x);

/* GPT-NeoX rope (rotate halves) applied in place to one head's vector of
 * length head_dim, at sequence position `pos`. theta = rope_theta. */
void rope_neox(float *vec, int head_dim, int pos, float theta);

/*
 * out[n_out] = W @ x[n_in], where W is the MLF2 tensor `w` (row-major,
 * dims[0]=n_in cols, dims[1]=n_out rows), dequantized on the fly.
 * Parallelized across output rows (pthreads). Optional bias if non-NULL.
 */
void matmul(float *out, const float *x,
            const hakm_model *m, const mlf2_tensor *w,
            const float *bias, int n_out, int n_in);

/* Batched matmul: x is B activation rows (stride n_in), out is B result rows
 * (stride n_out). Each weight row is read/dequantized once and dotted against
 * all B activations while cache-hot — the prefill speedup over B GEMVs. */
void matmul_batch(float *out, const float *x,
                  const hakm_model *m, const mlf2_tensor *w,
                  const float *bias, int n_out, int n_in, int B);

/* Dequantize a single row r (length n_in) of tensor `w` into out[n_in].
 * Used for embedding lookup. */
void deq_row(const hakm_model *m, const mlf2_tensor *w, int r, int n_in, float *out);

/* Generic parallel-for over the persistent worker pool: runs fn on each
 * element of jobs[njobs] (jsize bytes apart); caller's thread takes jobs[0].
 * Same pool the matmuls use — do not nest. */
void hakm_parallel(void (*fn)(void *), void *jobs, size_t jsize, int njobs);
int  hakm_nthreads(void);
