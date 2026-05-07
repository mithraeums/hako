/*
 * model.c — transformer forward pass.
 *
 * Runs the full sequence (no KV cache) for M1.  Caller only needs logits at
 * the last position; all T positions are computed for correct causal attention.
 *
 * Working buffer layout inside mini_ctx:
 *   x        [ctx × d_model]   current activations
 *   xn       [ctx × d_model]   normed activations (scratch)
 *   q,k,v    [ctx × d_model]   projected QKV — layout [T][n_heads][head_dim]
 *   attn_out [ctx × d_model]   weighted sum before o_proj
 *   ff1,ff2  [ctx × ffn_dim]   gate and up projections
 *   logits   [vocab]           final output
 *   tmp_d    [d_model]         scratch for linear output (d_model-sized)
 *   tmp_f    [ffn_dim]         scratch for linear output (ffn_dim-sized)
 */
#include "model.h"
#include "kernels_scalar.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── buffer allocation ──────────────────────────────────────────────────── */

model_bufs *model_bufs_alloc(const mlf_params *p) {
	model_bufs *b = (model_bufs *)calloc(1, sizeof(model_bufs));
	if (!b) return NULL;

	size_t ctx = p->ctx, D = p->d_model, F = p->ffn_dim;
	size_t H = p->n_heads, V = p->vocab;

	size_t total =
		ctx * D * 6 +       /* x, xn, q, k, v, attn_out */
		ctx * F * 2 +       /* ff1, ff2 */
		H * ctx * ctx +     /* attn_scores */
		V +                 /* logits */
		D +                 /* tmp_d */
		F;                  /* tmp_f */
	total *= sizeof(float);

	b->base = (float *)malloc(total);
	if (!b->base) { free(b); return NULL; }

	float *ptr = b->base;
	b->x        = ptr; ptr += ctx * D;
	b->xn       = ptr; ptr += ctx * D;
	b->q        = ptr; ptr += ctx * D;
	b->k        = ptr; ptr += ctx * D;
	b->v        = ptr; ptr += ctx * D;
	b->attn_out = ptr; ptr += ctx * D;
	b->ff1      = ptr; ptr += ctx * F;
	b->ff2      = ptr; ptr += ctx * F;
	b->attn     = ptr; ptr += H * ctx * ctx;
	b->logits   = ptr; ptr += V;
	b->tmp_d    = ptr; ptr += D;
	b->tmp_f    = ptr;

	return b;
}

void model_bufs_free(model_bufs *b) {
	if (b) { free(b->base); free(b); }
}

/* ── forward pass ────────────────────────────────────────────────────────── */

/*
 * Compute logits for the next token given `tokens[0..T-1]`.
 * Output written to b->logits[0..vocab-1].
 */
void model_forward(const nano_model *m, const int *tokens, int T,
                   model_bufs *b) {
	const mlf_params *p = &m->p;
	int D  = (int)p->d_model;
	int H  = (int)p->n_heads;
	int hd = D / H;            /* head_dim */
	int F  = (int)p->ffn_dim;
	int V  = (int)p->vocab;

	/* 1. Embed tokens → x[t*D .. (t+1)*D] */
	for (int t = 0; t < T; t++)
		memcpy(b->x + t * D, m->embed + tokens[t] * D, D * sizeof(float));

	/* 2. Transformer layers */
	for (int l = 0; l < (int)p->n_layers; l++) {
		const layer_weights *lw = &m->layers[l];

		/* 2a. Attn norm: xn[t] = rmsnorm(x[t]) */
		for (int t = 0; t < T; t++)
			rmsnorm(b->xn + t * D, b->x + t * D, lw->attn_norm, D, 1e-6f);

		/* 2b. Project Q, K, V: layout [T][n_heads][head_dim] = [T × D] */
		for (int t = 0; t < T; t++) {
			linear(b->q + t * D, b->xn + t * D, lw->q, D, D);
			linear(b->k + t * D, b->xn + t * D, lw->k, D, D);
			linear(b->v + t * D, b->xn + t * D, lw->v, D, D);
		}

		/* 2c. RoPE — interleaved pairs, matches Python train.py */
		rope_apply(b->q, T, H, hd);
		rope_apply(b->k, T, H, hd);

		/* 2d. Causal multi-head attention */
		memset(b->attn_out, 0, (size_t)T * D * sizeof(float));

		for (int h = 0; h < H; h++) {
			float *scores = b->attn + (size_t)h * T * (int)p->ctx;
			float scale   = 1.0f / sqrtf((float)hd);

			for (int t = 0; t < T; t++) {
				const float *q_t = b->q + t * D + h * hd;
				float *row = scores + t * T;

				/* Dot q[t,h] with k[s,h] for s in 0..t */
				for (int s = 0; s <= t; s++) {
					const float *k_s = b->k + s * D + h * hd;
					float dot = 0.0f;
					for (int d = 0; d < hd; d++) dot += q_t[d] * k_s[d];
					row[s] = dot * scale;
				}
				/* Causal mask: future positions to -inf */
				for (int s = t + 1; s < T; s++) row[s] = -1e9f;

				softmax(row, T);

				/* Weighted sum of values → attn_out[t, h] */
				float *out_th = b->attn_out + t * D + h * hd;
				for (int s = 0; s <= t; s++) {
					const float *v_s = b->v + s * D + h * hd;
					float w = row[s];
					for (int d = 0; d < hd; d++) out_th[d] += w * v_s[d];
				}
			}
		}

		/* 2e. O projection: x[t] += o_proj(attn_out[t]) */
		for (int t = 0; t < T; t++) {
			linear(b->tmp_d, b->attn_out + t * D, lw->o, D, D);
			for (int d = 0; d < D; d++) b->x[t * D + d] += b->tmp_d[d];
		}

		/* 2f. FFN norm: xn[t] = rmsnorm(x[t]) */
		for (int t = 0; t < T; t++)
			rmsnorm(b->xn + t * D, b->x + t * D, lw->ffn_norm, D, 1e-6f);

		/* 2g. SwiGLU FFN: gate = linear(xn, gate_w); up = linear(xn, up_w)
		 *                  x += down(silu(gate) * up) */
		for (int t = 0; t < T; t++) {
			float *gate = b->ff1 + t * F;
			float *up   = b->ff2 + t * F;
			linear(gate, b->xn + t * D, lw->gate, D, F);
			linear(up,   b->xn + t * D, lw->up,   D, F);
			for (int i = 0; i < F; i++) gate[i] = silu_f(gate[i]) * up[i];
			linear(b->tmp_d, gate, lw->down, F, D);
			for (int d = 0; d < D; d++) b->x[t * D + d] += b->tmp_d[d];
		}
	}

	/* 3. Final norm on last position */
	rmsnorm(b->xn, b->x + (T - 1) * D, m->final_norm, D, 1e-6f);

	/* 4. Logits = xn @ embed.T  (tied lm_head) */
	for (int v = 0; v < V; v++) {
		float s = 0.0f;
		const float *emb_v = m->embed + (size_t)v * D;
		for (int d = 0; d < D; d++) s += b->xn[d] * emb_v[d];
		b->logits[v] = s;
	}
}
