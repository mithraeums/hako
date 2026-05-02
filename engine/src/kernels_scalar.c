/*
 * kernels_scalar.c — scalar reference kernels (always built, no SIMD).
 * These are the iSH-safe paths.  SIMD variants added later as runtime dispatch.
 */
#include "kernels_scalar.h"
#include <math.h>
#include <string.h>

/* ── RMSNorm ─────────────────────────────────────────────────────────────── */

void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
	float ss = 0.0f;
	for (int i = 0; i < n; i++) ss += x[i] * x[i];
	float scale = 1.0f / sqrtf(ss / (float)n + eps);
	for (int i = 0; i < n; i++) out[i] = w[i] * x[i] * scale;
}

/* ── Linear (fully-connected) ────────────────────────────────────────────── */

/* y[i] = dot(x, w[i*in_dim .. (i+1)*in_dim])
 * w layout: [out_dim × in_dim], row-major (PyTorch Linear convention) */
void linear(float *out, const float *x, const float *w,
            int in_dim, int out_dim) {
	for (int i = 0; i < out_dim; i++) {
		float s = 0.0f;
		const float *row = w + (size_t)i * in_dim;
		for (int j = 0; j < in_dim; j++) s += x[j] * row[j];
		out[i] = s;
	}
}

/* ── Softmax (in-place) ──────────────────────────────────────────────────── */

void softmax(float *x, int n) {
	float max = x[0];
	for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
	float sum = 0.0f;
	for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
	float inv = 1.0f / sum;
	for (int i = 0; i < n; i++) x[i] *= inv;
}

/* ── SiLU ────────────────────────────────────────────────────────────────── */

float silu_f(float x) {
	return x / (1.0f + expf(-x));
}

/* ── RoPE ────────────────────────────────────────────────────────────────── */

/*
 * Apply RoPE in-place to buf[T × n_heads × head_dim].
 * buf is logically [T][n_heads][head_dim], stored flat.
 * Positions are 0..T-1.
 *
 * Uses interleaved pairs (v[2i], v[2i+1]) to match the Python trainer's
 * torch.view_as_complex(x.reshape(..., head_dim//2, 2)) convention.
 */
void rope_apply(float *buf, int T, int n_heads, int head_dim) {
	int half = head_dim / 2;
	for (int t = 0; t < T; t++) {
		for (int h = 0; h < n_heads; h++) {
			float *v = buf + ((size_t)t * n_heads + h) * head_dim;
			for (int i = 0; i < half; i++) {
				float theta = (float)t /
					powf(10000.0f, (float)(2 * i) / (float)head_dim);
				float cos_t = cosf(theta);
				float sin_t = sinf(theta);
				float x0 = v[2 * i];
				float x1 = v[2 * i + 1];
				v[2 * i]     = x0 * cos_t - x1 * sin_t;
				v[2 * i + 1] = x0 * sin_t + x1 * cos_t;
			}
		}
	}
}
