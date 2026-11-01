/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/*
 * train.c — the trainer: the engine's forward pass with its backward, AdamW,
 * MLF2 in and out, and a byte-level BPE trainer.
 *
 * f32 throughout. Every matrix product is one kernel, C (+)= op(A)·B with B
 * row-major, so the three products of a linear layer need no transposed
 * reads: the forward reads a transposed copy of the weights (refreshed after
 * each update), dX reads W as stored, dW reads dY transposed through strides.
 */
#include "train.h"
#include "nn.h"
#include "loader.h"
#include "../include/mlf2.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef float v8 __attribute__((vector_size(32)));
typedef float v8u __attribute__((vector_size(32), aligned(4)));
#define LD(p)  (*(const v8u *)(p))
#define BC(s)  ((v8){ (s), (s), (s), (s), (s), (s), (s), (s) })
#define MAXT   16

typedef int v8i __attribute__((vector_size(32)));

/* e^x, 8 lanes: x = n ln2 + r, a degree 6 polynomial for e^r (cephes), 2^n
   built in the exponent bits; clamped to [-87, 88] so 2^n stays normal */
static inline v8 vexp(v8 x) {
	const v8 hi = BC(88.0f), lo = BC(-87.0f), magic = BC(12582912.0f);
	v8i mh = x > hi, ml = x < lo;
	x = (v8)(((v8i)x & ~mh) | ((v8i)hi & mh));
	x = (v8)(((v8i)x & ~ml) | ((v8i)lo & ml));
	v8 t = x * BC(1.44269504088896341f) + magic, n = t - magic;
	v8 r = x - n * BC(0.693359375f) - n * BC(-2.12194440e-4f);
	v8 p = BC(1.9875691500e-4f);
	p = p * r + BC(1.3981999507e-3f);
	p = p * r + BC(8.3334519073e-3f);
	p = p * r + BC(4.1665795894e-2f);
	p = p * r + BC(1.6666665459e-1f);
	p = p * r + BC(5.0000001201e-1f);
	v8 y = p * r * r + r + BC(1.0f);
	return y * (v8)((((v8i)t - 0x4B400000) + 127) << 23);
}
static inline float hsum(v8 s) { return ((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7])); }

typedef struct { char name[MLF2_MAX_NAME]; size_t off; int rows, cols; } tr_tensor;
/* what the backward needs of one layer's forward, [rows][width] each */
typedef struct { float *x, *xn1, *r1, *q, *k, *v, *p, *ao, *xm, *xn2, *r2, *g, *u, *h; } tr_acts;

struct tr_model {
	tr_config c;
	size_t n, embd, out_norm;
	tr_layer *L;
	int nt; tr_tensor *t;
	float *p, *g, *mo, *ve, *pt;     /* params, grads, adam moments, transposed matrices */
	int pt_ok;
	long step;
	float *cs, *sn;                  /* rope tables [ctx][head_dim/2] */
	float *dwp;                      /* per-thread norm weight grads [MAXT][dim] */

	int capB, capT;
	size_t ws_n;                     /* per-thread attention scratch, floats */
	void *scr_raw;
	float *act;
	tr_acts *a;
	float *xL, *xf, *rf, *lg;
	float *dx, *dxn, *dao, *dq, *dk, *dv, *dA, *dB, *scr;
};

const tr_config *tr_cfg(const tr_model *m) { return &m->c; }
size_t tr_n_params(const tr_model *m) { return m->n; }
float *tr_params(tr_model *m) { m->pt_ok = 0; return m->p; }
float *tr_grads(tr_model *m) { return m->g; }
long  *tr_step(tr_model *m) { return &m->step; }

/* ── matrix product ───────────────────────────────────────────────────────── */

/* C[M,N] (+)= op(A)[M,K] · B[K,N], op(A)(m,k) = A[m*sm + k*sk]. B is first
   copied into 16-column panels [N/16][K][16] (zero padded), then each thread
   copies MC x KC blocks of its rows into 6-row strips [MC/6][KC][6] that stay
   in L2, and a 6x16 tile of C lives in 12 vector registers across the block:
   both operands are read in order, whatever the strides they came with. */
enum { MR = 6, NR = 16, KC = 256, MC = 96 };

static float *gm_buf;    /* packed B, then one A block per thread */
static void *gm_raw;
static size_t gm_cap;

/* n floats on a 64-byte line (aligned_alloc is not everywhere); free *raw */
static float *alloc64(size_t n, void **raw) {
	*raw = malloc(n * sizeof(float) + 64);
	return *raw ? (float *)(((uintptr_t)*raw + 63) & ~(uintptr_t)63) : NULL;
}

typedef struct {
	int M, N, K, m0, m1, p0, p1, add;
	const float *A; size_t sm, sk;
	const float *B; size_t ldb;
	float *C; size_t ldc;
	float *bp, *ap;
} gemm_job;

/* panel [k][16] of op(B)(k, n0..n0+w) = B[k*sbk + n*sbn], zero past w */
static void pack_panel(float *o, const float *B, size_t sbk, size_t sbn, int K, int n0, int w) {
	for (int k = 0; k < K; k++, o += NR) {
		const float *b = B + (size_t)k * sbk + (size_t)n0 * sbn;
		if (w == NR && sbn == 1) { *(v8u *)o = LD(b); *(v8u *)(o + 8) = LD(b + 8); continue; }
		int c = 0;
		for (; c < w; c++) o[c] = b[(size_t)c * sbn];
		for (; c < NR; c++) o[c] = 0;
	}
}

/* strip [k][6] of op(A)(m0..m0+rows, k0..k0+kc), zero past rows */
static void pack_strip(float *o, const float *A, size_t sm, size_t sk, int m0, int rows, int k0, int kc) {
	for (int i = 0; i < MR; i++) {
		if (i >= rows) { for (int k = 0; k < kc; k++) o[k * MR + i] = 0; continue; }
		const float *a = A + (size_t)(m0 + i) * sm + (size_t)k0 * sk;
		for (int k = 0; k < kc; k++) o[k * MR + i] = a[(size_t)k * sk];
	}
}

static void pack_b(void *arg) {
	const gemm_job *j = arg;
	for (int p = j->p0; p < j->p1; p++)
		pack_panel(j->bp + (size_t)p * j->K * NR, j->B, j->ldb, 1, j->K, p * NR, j->N - p * NR < NR ? j->N - p * NR : NR);
}

static void pack_a(float *o, const gemm_job *j, int m0, int mc, int k0, int kc) {
	for (int t = 0; t < mc; t += MR, o += (size_t)kc * MR)
		pack_strip(o, j->A, j->sm, j->sk, m0 + t, mc - t < MR ? mc - t : MR, k0, kc);
}

static void kern(int K, const float *a, const float *b, float *C, size_t ldc, int rows, int cols, int add) {
	v8 c00 = {0}, c01 = {0}, c10 = {0}, c11 = {0}, c20 = {0}, c21 = {0};
	v8 c30 = {0}, c31 = {0}, c40 = {0}, c41 = {0}, c50 = {0}, c51 = {0};
	for (int k = 0; k < K; k++, a += MR, b += NR) {
		v8 b0 = *(const v8 *)b, b1 = *(const v8 *)(b + 8), x;
		x = BC(a[0]); c00 += x * b0; c01 += x * b1;
		x = BC(a[1]); c10 += x * b0; c11 += x * b1;
		x = BC(a[2]); c20 += x * b0; c21 += x * b1;
		x = BC(a[3]); c30 += x * b0; c31 += x * b1;
		x = BC(a[4]); c40 += x * b0; c41 += x * b1;
		x = BC(a[5]); c50 += x * b0; c51 += x * b1;
	}
	v8 r[12] = { c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51 };
	if (rows == MR && cols == NR) {
		for (int i = 0; i < MR; i++)
			for (int h = 0; h < 2; h++) {
				v8u *o = (v8u *)(C + (size_t)i * ldc + 8 * h);
				*o = add ? *o + r[2 * i + h] : r[2 * i + h];
			}
		return;
	}
	const float *t = (const float *)r;
	for (int i = 0; i < rows; i++)
		for (int c = 0; c < cols; c++) {
			float *o = C + (size_t)i * ldc + c;
			*o = add ? *o + t[i * NR + c] : t[i * NR + c];
		}
}

static void gemm_rows(void *arg) {
	const gemm_job *j = arg;
	int np = (j->N + NR - 1) / NR;
	for (int k0 = 0; k0 < j->K; k0 += KC) {
		int kc = j->K - k0 < KC ? j->K - k0 : KC, add = j->add || k0 > 0;
		for (int m0 = j->m0; m0 < j->m1; m0 += MC) {
			int mc = j->m1 - m0 < MC ? j->m1 - m0 : MC;
			pack_a(j->ap, j, m0, mc, k0, kc);
			for (int p = 0; p < np; p++) {
				const float *b = j->bp + ((size_t)p * j->K + k0) * NR;
				int cols = j->N - p * NR < NR ? j->N - p * NR : NR;
				for (int t = 0; t < mc; t += MR)
					kern(kc, j->ap + (size_t)t * kc, b, j->C + (size_t)(m0 + t) * j->ldc + p * NR, j->ldc,
					     mc - t < MR ? mc - t : MR, cols, add);
			}
		}
	}
}

static void gemm(int M, int N, int K, const float *A, size_t sm, size_t sk,
                 const float *B, size_t ldb, float *C, size_t ldc, int add) {
	int np = (N + NR - 1) / NR, T = hakm_nthreads();
	if (T > MAXT) T = MAXT;
	size_t bsz = (size_t)np * K * NR, need = bsz + (size_t)T * MC * KC;
	if (need > gm_cap) {
		free(gm_raw);
		gm_buf = alloc64(need, &gm_raw);
		gm_cap = gm_buf ? need : 0;
		if (!gm_buf) { fprintf(stderr, "train: out of memory in gemm\n"); abort(); }
	}
	gemm_job jb[MAXT];
	int nj = T < np ? T : np, per = (np + nj - 1) / nj, n = 0;
	for (int t = 0; t < nj; t++) {
		int p0 = t * per, p1 = p0 + per < np ? p0 + per : np;
		if (p0 >= p1) break;
		jb[n++] = (gemm_job){ .N = N, .K = K, .p0 = p0, .p1 = p1, .B = B, .ldb = ldb, .bp = gm_buf };
	}
	hakm_parallel(pack_b, jb, sizeof(gemm_job), n);
	int tiles = (M + MR - 1) / MR;
	nj = T < tiles ? T : tiles;
	per = (tiles + nj - 1) / nj; n = 0;
	for (int t = 0; t < nj; t++) {
		int m0 = t * per * MR, m1 = m0 + per * MR < M ? m0 + per * MR : M;
		if (m0 >= m1) break;
		jb[n] = (gemm_job){ M, N, K, m0, m1, 0, 0, add, A, sm, sk, B, ldb, C, ldc, gm_buf, gm_buf + bsz + (size_t)n * MC * KC };
		n++;
	}
	hakm_parallel(gemm_rows, jb, sizeof(gemm_job), n);
}

/* One thread's C[M,N] (+)= op(A)·op(B) for attention, B at any strides, no K
   blocking (K is head_dim or the sequence). lim drops what causality zeroes:
   1 computes only columns n < m+6 of each 6-row tile, 2 sums only k < m+6,
   3 only k >= m. ws holds K x round16(N) + 6 x K floats, 32-aligned. */
static void gemm_st(int M, int N, int K, const float *A, size_t sm, size_t sk,
                    const float *B, size_t sbk, size_t sbn, float *C, size_t ldc,
                    int add, int lim, float *ws) {
	int np = (N + NR - 1) / NR;
	float *strip = ws + (size_t)np * K * NR;
	for (int p = 0; p < np; p++)
		pack_panel(ws + (size_t)p * K * NR, B, sbk, sbn, K, p * NR, N - p * NR < NR ? N - p * NR : NR);
	for (int m0 = 0; m0 < M; m0 += MR) {
		int rows = M - m0 < MR ? M - m0 : MR, kb = 0, ke = K, pe = np;
		if (lim == 1 && (m0 + MR + NR - 1) / NR < pe) pe = (m0 + MR + NR - 1) / NR;
		if (lim == 2 && m0 + MR < ke) ke = m0 + MR;
		if (lim == 3) kb = m0 < K ? m0 : K;
		pack_strip(strip, A, sm, sk, m0, rows, 0, K);
		for (int p = 0; p < pe; p++)
			kern(ke - kb, strip + (size_t)kb * MR, ws + ((size_t)p * K + kb) * NR,
			     C + (size_t)m0 * ldc + p * NR, ldc, rows, N - p * NR < NR ? N - p * NR : NR, add);
	}
}

/* the three products of a linear layer y[N,O] = x[N,I] · W[O,I]^T */
static void mm_fwd(float *y, const float *x, const float *wt, int N, int O, int I) {
	gemm(N, O, I, x, I, 1, wt, O, y, O, 0);
}
static void mm_dx(float *dx, const float *dy, const float *w, int N, int O, int I, int add) {
	gemm(N, I, O, dy, O, 1, w, I, dx, I, add);
}
static void mm_dw(float *dw, const float *dy, const float *x, int N, int O, int I) {
	gemm(O, I, N, dy, 1, O, x, I, dw, I, 1);
}

/* ── the model ────────────────────────────────────────────────────────────── */

const char *tr_check(const tr_config *c) {
	if (c->vocab < 260 || c->dim < 8 || c->layers < 1 || c->heads < 1 || c->kv_heads < 1 ||
	    c->head_dim < 2 || c->ffn < 8 || c->ctx < 2) return "a size is zero or too small";
	if (c->heads % c->kv_heads) return "heads must be a multiple of kv-heads";
	if (c->head_dim % 8) return "head_dim (dim / heads) must be a multiple of 8";
	if (c->dim % 8 || c->ffn % 8) return "dim and ffn must be multiples of 8";
	return NULL;
}

static uint64_t splitmix(uint64_t *s) {
	uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}
static float gauss(uint64_t *s) {
	double u1 = ((splitmix(s) >> 11) + 1.0) / 9007199254740993.0;
	double u2 = (splitmix(s) >> 11) / 9007199254740992.0;
	return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}

static size_t add_tensor(tr_model *m, size_t *at, const char *name, int rows, int cols) {
	tr_tensor *t = &m->t[m->nt++];
	snprintf(t->name, sizeof t->name, "%s", name);
	t->off = *at; t->rows = rows; t->cols = cols;
	*at += (size_t)rows * cols;
	return t->off;
}

static tr_model *tr_alloc(const tr_config *c) {
	tr_model *m = calloc(1, sizeof *m);
	if (!m) return NULL;
	m->c = *c;
	int D = c->dim, Q = c->heads * c->head_dim, KV = c->kv_heads * c->head_dim, F = c->ffn, V = c->vocab;
	m->L = calloc(c->layers, sizeof *m->L);
	m->t = calloc(2 + 9 * (size_t)c->layers, sizeof *m->t);
	if (!m->L || !m->t) { tr_free(m); return NULL; }
	size_t at = 0;
	char nm[64];
	m->embd = add_tensor(m, &at, "token_embd.weight", V, D);
	for (int l = 0; l < c->layers; l++) {
		tr_layer *L = &m->L[l];
		#define T(field, suffix, r, k) snprintf(nm, sizeof nm, "blk.%d." suffix, l); L->field = add_tensor(m, &at, nm, r, k);
		T(attn_norm, "attn_norm.weight", 1, D);
		T(wq, "attn_q.weight", Q, D); T(wk, "attn_k.weight", KV, D); T(wv, "attn_v.weight", KV, D);
		T(wo, "attn_output.weight", D, Q);
		T(ffn_norm, "ffn_norm.weight", 1, D);
		T(gate, "ffn_gate.weight", F, D); T(up, "ffn_up.weight", F, D); T(down, "ffn_down.weight", D, F);
		#undef T
	}
	m->out_norm = add_tensor(m, &at, "output_norm.weight", 1, D);
	m->n = at;
	m->p  = calloc(at, sizeof(float)); m->g  = calloc(at, sizeof(float));
	m->mo = calloc(at, sizeof(float)); m->ve = calloc(at, sizeof(float));
	m->pt = calloc(at, sizeof(float));
	int half = c->head_dim / 2;
	m->cs = malloc((size_t)c->ctx * half * sizeof(float));
	m->sn = malloc((size_t)c->ctx * half * sizeof(float));
	m->dwp = malloc((size_t)MAXT * D * sizeof(float));
	if (!m->p || !m->g || !m->mo || !m->ve || !m->pt || !m->cs || !m->sn || !m->dwp) { tr_free(m); return NULL; }
	/* the engine's rope_neox arithmetic, so the trained weights see the same angles */
	for (int pos = 0; pos < c->ctx; pos++)
		for (int i = 0; i < half; i++) {
			float freq = powf(c->rope_theta, -((float)(2 * i) / (float)c->head_dim));
			float ang = (float)pos * freq;
			m->cs[(size_t)pos * half + i] = cosf(ang);
			m->sn[(size_t)pos * half + i] = sinf(ang);
		}
	return m;
}

tr_model *tr_new(const tr_config *c, uint64_t seed) {
	if (tr_check(c)) return NULL;
	tr_model *m = tr_alloc(c);
	if (!m) return NULL;
	float out_std = 0.02f / sqrtf(2.0f * c->layers);
	for (int i = 0; i < m->nt; i++) {
		tr_tensor *t = &m->t[i];
		float *w = m->p + t->off;
		size_t n = (size_t)t->rows * t->cols;
		if (t->rows == 1) { for (size_t j = 0; j < n; j++) w[j] = 1.0f; continue; }
		int scaled = strstr(t->name, "attn_output") || strstr(t->name, "ffn_down");
		float sd = scaled ? out_std : 0.02f;
		for (size_t j = 0; j < n; j++) w[j] = sd * gauss(&seed);
	}
	return m;
}

static void free_acts(tr_model *m) {
	free(m->act); free(m->a); free(m->scr_raw);
	m->act = NULL; m->a = NULL; m->scr = NULL; m->scr_raw = NULL; m->capB = m->capT = 0;
}

void tr_free(tr_model *m) {
	if (!m) return;
	free_acts(m);
	free(m->L); free(m->t);
	free(m->p); free(m->g); free(m->mo); free(m->ve); free(m->pt);
	free(m->cs); free(m->sn); free(m->dwp);
	free(m);
	free(gm_raw); gm_raw = gm_buf = NULL; gm_cap = 0;
}

/* floats: activations per layer and after the layers, and attention scratch per thread */
static void act_sizes(const tr_config *c, int B, int T, size_t *per_layer, size_t *tail, size_t *ws_n) {
	size_t N = (size_t)B * T, D = c->dim, Q = (size_t)c->heads * c->head_dim;
	size_t KV = (size_t)c->kv_heads * c->head_dim, F = c->ffn, V = c->vocab;
	*per_layer = N * (4 * D + 2 + 2 * Q + 2 * KV + 3 * F) + (size_t)B * c->heads * T * T;
	*tail = N * (2 * D + 1 + V) + N * (2 * D + 2 * Q + 2 * KV + 2 * F);
	size_t hd = c->head_dim, Tp = (T + 15) / 16 * 16, Hp = (hd + 15) / 16 * 16, Tq = ((size_t)T * T + 15) / 16 * 16;
	size_t w = Tq + (Tp * hd > Hp * T ? Tp * hd : Hp * T) + MR * ((size_t)T > hd ? (size_t)T : hd) + 16;
	*ws_n = (w + 15) / 16 * 16;
}

size_t tr_mem_bytes(const tr_config *c, int B, int T) {
	size_t D = c->dim, Q = (size_t)c->heads * c->head_dim, KV = (size_t)c->kv_heads * c->head_dim;
	size_t F = c->ffn, V = c->vocab, N = (size_t)B * T, per_layer, tail, ws, th = hakm_nthreads();
	size_t np = V * D + c->layers * (2 * D + 2 * Q * D + 2 * KV * D + 3 * F * D) + D;
	act_sizes(c, B, T, &per_layer, &tail, &ws);
	size_t wide = F > Q ? F : Q, packed = N * (wide > D ? wide : D);   /* the widest B a product packs */
	if (V * D > packed) packed = V * D;
	return (5 * np + per_layer * c->layers + tail + th * ws + packed + th * MC * KC) * sizeof(float);
}

static int ensure_acts(tr_model *m, int B, int T) {
	if (B <= m->capB && T <= m->capT) return 0;
	if (B < m->capB) B = m->capB;
	if (T < m->capT) T = m->capT;
	free_acts(m);
	const tr_config *c = &m->c;
	size_t N = (size_t)B * T, D = c->dim, Q = (size_t)c->heads * c->head_dim;
	size_t KV = (size_t)c->kv_heads * c->head_dim, F = c->ffn, V = c->vocab, per_layer, tail;
	act_sizes(c, B, T, &per_layer, &tail, &m->ws_n);
	m->act = malloc((per_layer * c->layers + tail) * sizeof(float));
	m->a = calloc(c->layers, sizeof *m->a);
	m->scr = alloc64((size_t)hakm_nthreads() * m->ws_n, &m->scr_raw);
	if (!m->act || !m->a || !m->scr) { free_acts(m); return -1; }
	float *p = m->act;
	#define TAKE(dst, n) do { (dst) = p; p += (n); } while (0)
	for (int l = 0; l < c->layers; l++) {
		TAKE(m->a[l].x, N * D); TAKE(m->a[l].xn1, N * D); TAKE(m->a[l].r1, N);
		TAKE(m->a[l].q, N * Q); TAKE(m->a[l].k, N * KV); TAKE(m->a[l].v, N * KV);
		TAKE(m->a[l].p, (size_t)B * c->heads * T * T); TAKE(m->a[l].ao, N * Q);
		TAKE(m->a[l].xm, N * D); TAKE(m->a[l].xn2, N * D); TAKE(m->a[l].r2, N);
		TAKE(m->a[l].g, N * F); TAKE(m->a[l].u, N * F); TAKE(m->a[l].h, N * F);
	}
	TAKE(m->xL, N * D); TAKE(m->xf, N * D); TAKE(m->rf, N); TAKE(m->lg, N * V);
	TAKE(m->dx, N * D); TAKE(m->dxn, N * D); TAKE(m->dao, N * Q); TAKE(m->dq, N * Q);
	TAKE(m->dk, N * KV); TAKE(m->dv, N * KV); TAKE(m->dA, N * F); TAKE(m->dB, N * F);
	#undef TAKE
	m->capB = B; m->capT = T;
	return 0;
}

static void refresh_pt(tr_model *m) {
	if (m->pt_ok) return;
	for (int i = 0; i < m->nt; i++) {
		const tr_tensor *t = &m->t[i];
		if (t->rows == 1) continue;
		const float *w = m->p + t->off;
		float *o = m->pt + t->off;
		for (int r = 0; r < t->rows; r++)
			for (int k = 0; k < t->cols; k++) o[(size_t)k * t->rows + r] = w[(size_t)r * t->cols + k];
	}
	m->pt_ok = 1;
}

/* ── per-row kernels ──────────────────────────────────────────────────────── */

static void rms_fwd(float *y, float *r, const float *x, const float *w, int N, int D, float eps) {
	for (int n = 0; n < N; n++) {
		const float *xr = x + (size_t)n * D;
		float ss = 0;
		for (int i = 0; i < D; i++) ss += xr[i] * xr[i];
		float s = 1.0f / sqrtf(ss / (float)D + eps);
		r[n] = s;
		for (int i = 0; i < D; i++) y[(size_t)n * D + i] = xr[i] * s * w[i];
	}
}

/* dx += d(rmsnorm)/dx · dy, dw += d/dw · dy */
static void rms_bwd(float *dx, float *dw, const float *dy, const float *x, const float *r,
                    const float *w, int N, int D) {
	for (int n = 0; n < N; n++) {
		const float *dyr = dy + (size_t)n * D, *xr = x + (size_t)n * D;
		float s = r[n], dot = 0;
		for (int i = 0; i < D; i++) dot += dyr[i] * w[i] * xr[i];
		float k = s * s * s * dot / (float)D;
		float *dxr = dx + (size_t)n * D;
		for (int i = 0; i < D; i++) {
			dxr[i] += s * dyr[i] * w[i] - k * xr[i];
			dw[i] += dyr[i] * xr[i] * s;
		}
	}
}

/* neox rope on every head of each row; dir -1 is the backward (rotate back) */
static void rope_rows(const tr_model *m, float *x, int n0, int n1, int T, int nh, int dir) {
	int hd = m->c.head_dim, half = hd / 2;
	for (int n = n0; n < n1; n++) {
		const float *cs = m->cs + (size_t)(n % T) * half, *sn = m->sn + (size_t)(n % T) * half;
		for (int h = 0; h < nh; h++) {
			float *v = x + ((size_t)n * nh + h) * hd;
			for (int i = 0; i < half; i++) {
				float a = v[i], b = v[i + half], c = cs[i], s = dir * sn[i];
				v[i] = a * c - b * s;
				v[i + half] = a * s + b * c;
			}
		}
	}
}

/* the per-row passes over a range of rows, split across the threads:
   0 x += xa (when given), then out = rmsnorm(x) * w with r the 1/rms;
   1 the rmsnorm backward into out (dx) and this thread's dw partial;
   2 rope on x, nh heads, dir +1 forward or -1 backward */
typedef struct {
	int op; const tr_model *m; int i0, i1, T, nh, dir;
	float *out, *r, *x, *dw; const float *xa, *w, *dy;
} vec_job;

static void vec_rows(void *arg) {
	vec_job *j = arg;
	int D = j->m->c.dim, n = j->i1 - j->i0;
	size_t o = (size_t)j->i0 * D;
	if (j->op == 0) {
		if (j->xa) for (size_t i = o; i < o + (size_t)n * D; i++) j->x[i] += j->xa[i];
		rms_fwd(j->out + o, j->r + j->i0, j->x + o, j->w, n, D, j->m->c.rms_eps);
	} else if (j->op == 1) {
		memset(j->dw, 0, D * sizeof(float));
		rms_bwd(j->out + o, j->dw, j->dy + o, j->x + o, j->r + j->i0, j->w, n, D);
	} else rope_rows(j->m, j->x, j->i0, j->i1, j->T, j->nh, j->dir);
}

static void vec_par(vec_job v, int N) {
	int nt = hakm_nthreads();
	if (nt > MAXT) nt = MAXT;
	if (nt > N) nt = N;
	vec_job jb[MAXT];
	int per = (N + nt - 1) / nt, nj = 0, D = v.m->c.dim;
	float *dw = v.dw;
	for (int t = 0; t < nt; t++) {
		int i0 = t * per, i1 = i0 + per < N ? i0 + per : N;
		if (i0 >= i1) break;
		jb[nj] = v; jb[nj].i0 = i0; jb[nj].i1 = i1;
		if (v.op == 1) jb[nj].dw = v.m->dwp + (size_t)nj * D;
		nj++;
	}
	hakm_parallel(vec_rows, jb, sizeof(vec_job), nj);
	if (v.op == 1)
		for (int t = 0; t < nj; t++)
			for (int i = 0; i < D; i++) dw[i] += jb[t].dw[i];
}

static void add_rms(const tr_model *m, float *out, float *r, float *x, const float *xa, const float *w, int N) {
	vec_par((vec_job){ .op = 0, .m = m, .out = out, .r = r, .x = x, .xa = xa, .w = w }, N);
}
static void rms_back(const tr_model *m, float *dx, float *dw, const float *dy, const float *x, const float *r, const float *w, int N) {
	vec_par((vec_job){ .op = 1, .m = m, .out = dx, .dw = dw, .dy = dy, .x = (float *)x, .r = (float *)r, .w = w }, N);
}
static void rope_par(const tr_model *m, float *x, int N, int T, int nh, int dir) {
	vec_par((vec_job){ .op = 2, .m = m, .x = x, .T = T, .nh = nh, .dir = dir }, N);
}

typedef struct {
	tr_model *m; int l, B, T, i0, i1;
	float scale; const int *tgt; double out; float *scr;
} row_job;

static void run_rows(void (*fn)(void *), tr_model *m, int l, int B, int T, int n,
                     float scale, const int *tgt, double *sum) {
	int nt = hakm_nthreads();
	if (nt > n) nt = n;
	if (nt > MAXT) nt = MAXT;
	row_job jb[MAXT];
	int per = (n + nt - 1) / nt, nj = 0;
	for (int t = 0; t < nt; t++) {
		int i0 = t * per, i1 = i0 + per;
		if (i1 > n) i1 = n;
		if (i0 >= i1) break;
		jb[nj] = (row_job){ m, l, B, T, i0, i1, scale, tgt, 0, m->scr + (size_t)nj * m->ws_n };
		nj++;
	}
	hakm_parallel(fn, jb, sizeof(row_job), nj);
	if (sum) { *sum = 0; for (int i = 0; i < nj; i++) *sum += jb[i].out; }
}

/* causal attention, one (sequence, head) per index: S = QK^T into the stored
   probabilities, softmax in place (zero above the diagonal), O = PV */
static void att_fwd(void *arg) {
	row_job *j = arg;
	tr_model *m = j->m;
	int T = j->T, H = m->c.heads, hd = m->c.head_dim, Q = H * hd, KV = m->c.kv_heads * hd;
	int grp = H / m->c.kv_heads;
	float scale = 1.0f / sqrtf((float)hd);
	const tr_acts *a = &m->a[j->l];
	for (int i = j->i0; i < j->i1; i++) {
		int b = i / H, h = i % H, kh = h / grp;
		float *P = a->p + (size_t)i * T * T;
		const float *kb = a->k + (size_t)b * T * KV + kh * hd, *vb = a->v + (size_t)b * T * KV + kh * hd;
		gemm_st(T, T, hd, a->q + (size_t)b * T * Q + h * hd, Q, 1, kb, 1, KV, P, T, 0, 1, j->scr);
		for (int t = 0; t < T; t++) {
			float *p = P + (size_t)t * T, mx = -INFINITY, sum = 0;
			for (int s = 0; s <= t; s++) { p[s] *= scale; if (p[s] > mx) mx = p[s]; }
			int s = 0;
			v8 vs = {0};
			for (; s + 8 <= t + 1; s += 8) { v8 e = vexp(LD(p + s) - BC(mx)); *(v8u *)(p + s) = e; vs += e; }
			sum = hsum(vs);
			for (; s <= t; s++) { p[s] = expf(p[s] - mx); sum += p[s]; }
			float inv = 1.0f / sum;
			for (int s = 0; s <= t; s++) p[s] *= inv;
			memset(p + t + 1, 0, (size_t)(T - t - 1) * sizeof(float));
		}
		gemm_st(T, hd, T, P, T, 1, vb, KV, 1, a->ao + (size_t)b * T * Q + h * hd, Q, 0, 2, j->scr);
	}
}

/* its backward, one (sequence, kv head) per index: the heads of a group add
   into the same dk/dv rows, so a group is never split across threads.
   dP = dO V^T, dS = P (dP - rowsum(P dP)) / sqrt(hd), dQ = dS K,
   dK += dS^T Q, dV += P^T dO */
static void att_bwd(void *arg) {
	row_job *j = arg;
	tr_model *m = j->m;
	int T = j->T, H = m->c.heads, KH = m->c.kv_heads, hd = m->c.head_dim, Q = H * hd, KV = KH * hd;
	int grp = H / KH;
	float scale = 1.0f / sqrtf((float)hd);
	const tr_acts *a = &m->a[j->l];
	float *dS = j->scr, *ws = j->scr + ((size_t)T * T + 15) / 16 * 16;
	for (int i = j->i0; i < j->i1; i++) {
		int b = i / KH, kh = i % KH;
		size_t kvo = (size_t)b * T * KV + kh * hd;
		for (int h = kh * grp; h < (kh + 1) * grp; h++) {
			size_t qo = (size_t)b * T * Q + h * hd;
			const float *P = a->p + ((size_t)b * H + h) * T * T, *dO = m->dao + qo;
			gemm_st(T, T, hd, dO, Q, 1, a->v + kvo, 1, KV, dS, T, 0, 1, ws);
			for (int t = 0; t < T; t++) {
				const float *p = P + (size_t)t * T;
				float *d = dS + (size_t)t * T, sum = 0;
				for (int u = 0; u <= t; u++) sum += p[u] * d[u];
				for (int u = 0; u <= t; u++) d[u] = p[u] * (d[u] - sum) * scale;
				memset(d + t + 1, 0, (size_t)(T - t - 1) * sizeof(float));
			}
			gemm_st(T, hd, T, dS, T, 1, a->k + kvo, KV, 1, m->dq + qo, Q, 0, 2, ws);
			gemm_st(T, hd, T, dS, 1, T, a->q + qo, Q, 1, m->dk + kvo, KV, 1, 3, ws);
			gemm_st(T, hd, T, P, 1, T, dO, Q, 1, m->dv + kvo, KV, 1, 3, ws);
		}
	}
}

/* SwiGLU over rows: h = silu(g) * u; backward turns dA (dh) into dg, dB into du */
static void swiglu_fwd(void *arg) {
	row_job *j = arg;
	tr_acts *a = &j->m->a[j->l];
	size_t F = j->m->c.ffn;
	for (size_t i = j->i0 * F; i < j->i1 * F; i += 8) {
		v8 g = LD(a->g + i);
		*(v8u *)(a->h + i) = g / (BC(1.0f) + vexp(-g)) * LD(a->u + i);
	}
}
static void swiglu_bwd(void *arg) {
	row_job *j = arg;
	tr_model *m = j->m;
	tr_acts *a = &m->a[j->l];
	size_t F = m->c.ffn;
	for (size_t i = j->i0 * F; i < j->i1 * F; i += 8) {
		v8 g = LD(a->g + i), one = BC(1.0f), sg = one / (one + vexp(-g)), dh = LD(m->dA + i);
		*(v8u *)(m->dB + i) = dh * g * sg;
		*(v8u *)(m->dA + i) = dh * LD(a->u + i) * sg * (one + g * (one - sg));
	}
}

/* softmax cross-entropy per row; the logits become dlogits when scale > 0 */
static void ce_rows(void *arg) {
	row_job *j = arg;
	int V = j->m->c.vocab, V8 = V / 8 * 8, keep = j->scale > 0;
	double nll = 0;
	for (int n = j->i0; n < j->i1; n++) {
		float *z = j->m->lg + (size_t)n * V;
		float mx = z[0];
		for (int i = 1; i < V; i++) if (z[i] > mx) mx = z[i];
		int y = j->tgt[n];
		float zy = z[y];
		v8 vs = {0};
		for (int i = 0; i < V8; i += 8) {
			v8 e = vexp(LD(z + i) - BC(mx));
			vs += e;
			if (keep) *(v8u *)(z + i) = e;
		}
		double se = hsum(vs);
		for (int i = V8; i < V; i++) { float e = expf(z[i] - mx); se += e; if (keep) z[i] = e; }
		nll += log(se) + mx - zy;
		if (keep) {
			float k = (float)(j->scale / se);
			for (int i = 0; i < V; i++) z[i] *= k;
			z[y] -= j->scale;
		}
	}
	j->out = nll;
}

/* ── forward + backward ───────────────────────────────────────────────────── */

double tr_run(tr_model *m, const int *in, const int *tgt, int B, int T, float scale) {
	const tr_config *c = &m->c;
	if (T > c->ctx || B < 1 || T < 1 || ensure_acts(m, B, T)) return -1;
	refresh_pt(m);
	int N = B * T, D = c->dim, H = c->heads, KH = c->kv_heads, hd = c->head_dim;
	int Q = H * hd, KV = KH * hd, F = c->ffn, V = c->vocab;
	const float *P = m->p, *PT = m->pt;

	for (int n = 0; n < N; n++) memcpy(m->a[0].x + (size_t)n * D, P + m->embd + (size_t)in[n] * D, D * sizeof(float));
	for (int l = 0; l < c->layers; l++) {
		const tr_layer *L = &m->L[l];
		tr_acts *a = &m->a[l];
		add_rms(m, a->xn1, a->r1, a->x, l ? m->a[l - 1].xm : NULL, P + L->attn_norm, N);
		mm_fwd(a->q, a->xn1, PT + L->wq, N, Q, D);
		mm_fwd(a->k, a->xn1, PT + L->wk, N, KV, D);
		mm_fwd(a->v, a->xn1, PT + L->wv, N, KV, D);
		rope_par(m, a->q, N, T, H, 1);
		rope_par(m, a->k, N, T, KH, 1);
		run_rows(att_fwd, m, l, B, T, B * H, 0, NULL, NULL);
		mm_fwd(a->xm, a->ao, PT + L->wo, N, D, Q);
		add_rms(m, a->xn2, a->r2, a->xm, a->x, P + L->ffn_norm, N);
		mm_fwd(a->g, a->xn2, PT + L->gate, N, F, D);
		mm_fwd(a->u, a->xn2, PT + L->up, N, F, D);
		run_rows(swiglu_fwd, m, l, B, T, N, 0, NULL, NULL);
		float *next = l + 1 < c->layers ? m->a[l + 1].x : m->xL;
		mm_fwd(next, a->h, PT + L->down, N, D, F);
	}
	add_rms(m, m->xf, m->rf, m->xL, m->a[c->layers - 1].xm, P + m->out_norm, N);
	mm_fwd(m->lg, m->xf, PT + m->embd, N, V, D);
	double nll;
	run_rows(ce_rows, m, 0, B, T, N, scale, tgt, &nll);
	if (scale <= 0) return nll;

	/* backward: m->lg now holds dlogits */
	float *G = m->g;
	mm_dw(G + m->embd, m->lg, m->xf, N, V, D);
	mm_dx(m->dxn, m->lg, P + m->embd, N, V, D, 0);
	memset(m->dx, 0, (size_t)N * D * sizeof(float));
	rms_back(m, m->dx, G + m->out_norm, m->dxn, m->xL, m->rf, P + m->out_norm, N);
	for (int l = c->layers - 1; l >= 0; l--) {
		const tr_layer *L = &m->L[l];
		tr_acts *a = &m->a[l];
		/* ffn: x_next = xm + down(silu(gate) * up) */
		mm_dw(G + L->down, m->dx, a->h, N, D, F);
		mm_dx(m->dA, m->dx, P + L->down, N, D, F, 0);
		run_rows(swiglu_bwd, m, l, B, T, N, 0, NULL, NULL);
		mm_dw(G + L->gate, m->dA, a->xn2, N, F, D);
		mm_dw(G + L->up, m->dB, a->xn2, N, F, D);
		mm_dx(m->dxn, m->dA, P + L->gate, N, F, D, 0);
		mm_dx(m->dxn, m->dB, P + L->up, N, F, D, 1);
		rms_back(m, m->dx, G + L->ffn_norm, m->dxn, a->xm, a->r2, P + L->ffn_norm, N);
		/* attention: xm = x + wo(attn(rope(wq x), rope(wk x), wv x)) */
		mm_dw(G + L->wo, m->dx, a->ao, N, D, Q);
		mm_dx(m->dao, m->dx, P + L->wo, N, D, Q, 0);
		memset(m->dk, 0, (size_t)N * KV * sizeof(float));
		memset(m->dv, 0, (size_t)N * KV * sizeof(float));
		run_rows(att_bwd, m, l, B, T, B * KH, 0, NULL, NULL);
		rope_par(m, m->dq, N, T, H, -1);
		rope_par(m, m->dk, N, T, KH, -1);
		mm_dw(G + L->wq, m->dq, a->xn1, N, Q, D);
		mm_dw(G + L->wk, m->dk, a->xn1, N, KV, D);
		mm_dw(G + L->wv, m->dv, a->xn1, N, KV, D);
		mm_dx(m->dxn, m->dq, P + L->wq, N, Q, D, 0);
		mm_dx(m->dxn, m->dk, P + L->wk, N, KV, D, 1);
		mm_dx(m->dxn, m->dv, P + L->wv, N, KV, D, 1);
		rms_back(m, m->dx, G + L->attn_norm, m->dxn, a->x, a->r1, P + L->attn_norm, N);
	}
	for (int n = 0; n < N; n++) {
		float *ge = G + m->embd + (size_t)in[n] * D;
		const float *d = m->dx + (size_t)n * D;
		for (int i = 0; i < D; i++) ge[i] += d[i];
	}
	return nll;
}

void tr_zero_grad(tr_model *m) { memset(m->g, 0, m->n * sizeof(float)); }

double tr_clip(tr_model *m, double max_norm) {
	double ss = 0;
	for (size_t i = 0; i < m->n; i++) ss += (double)m->g[i] * m->g[i];
	double norm = sqrt(ss);
	if (max_norm > 0 && norm > max_norm) {
		float k = (float)(max_norm / norm);
		for (size_t i = 0; i < m->n; i++) m->g[i] *= k;
	}
	return norm;
}

/* decoupled weight decay on the matrices (the embedding is one), none on norms */
void tr_adamw(tr_model *m, float lr, float wd, float b1, float b2, float eps) {
	m->step++;
	float c1 = 1.0f - powf(b1, (float)m->step), c2 = 1.0f - powf(b2, (float)m->step);
	for (int i = 0; i < m->nt; i++) {
		const tr_tensor *t = &m->t[i];
		float decay = t->rows > 1 ? wd : 0.0f;
		size_t n = (size_t)t->rows * t->cols;
		float *p = m->p + t->off, *g = m->g + t->off, *mo = m->mo + t->off, *ve = m->ve + t->off;
		for (size_t j = 0; j < n; j++) {
			mo[j] = b1 * mo[j] + (1 - b1) * g[j];
			ve[j] = b2 * ve[j] + (1 - b2) * g[j] * g[j];
			float mh = mo[j] / c1, vh = ve[j] / c2;
			p[j] -= lr * (mh / (sqrtf(vh) + eps) + decay * p[j]);
		}
	}
	m->pt_ok = 0;
}

/* ── files ────────────────────────────────────────────────────────────────── */

size_t tok_blob_len(const void *blob, size_t avail) {
	const unsigned char *p = blob, *end = p + avail;
	if (avail < 8) return 0;
	uint32_t nt, nm, len;
	memcpy(&nt, p, 4); memcpy(&nm, p + 4, 4);
	p += 8;
	for (uint32_t i = 0; i < nt; i++) {
		if (end - p < 5) return 0;
		memcpy(&len, p + 1, 4); p += 5;
		if ((size_t)(end - p) < len) return 0;
		p += len;
	}
	for (uint32_t i = 0; i < nm; i++) {
		if (end - p < 4) return 0;
		memcpy(&len, p, 4); p += 4;
		if ((size_t)(end - p) < len) return 0;
		p += len;
	}
	return (size_t)(p - (const unsigned char *)blob);
}

int tr_save(const tr_model *m, const char *path, const void *tok, size_t tok_len, int eos) {
	const tr_config *c = &m->c;
	size_t tok_off = sizeof(mlf2_header) + sizeof(mlf2_tensor) * m->nt;
	size_t data_off = (tok_off + tok_len + MLF2_ALIGN - 1) / MLF2_ALIGN * MLF2_ALIGN;
	mlf2_header h = {0};
	memcpy(h.magic, MLF2_MAGIC, 4);
	h.version = MLF2_VERSION; h.arch = MLF2_ARCH_HAKO;
	h.n_layers = c->layers; h.d_model = c->dim; h.n_heads = c->heads; h.n_kv_heads = c->kv_heads;
	h.head_dim = c->head_dim; h.ffn_dim = c->ffn; h.ctx = c->ctx; h.vocab = c->vocab;
	h.rope_theta = c->rope_theta; h.rms_eps = c->rms_eps;
	h.flags = MLF2_FLAG_TIED_EMBED; h.n_tensors = m->nt;
	h.tensors_off = sizeof(mlf2_header); h.tokenizer_off = tok_off; h.data_off = data_off;
	h.bos_id = eos; h.eos_id = eos;

	char tmp[4096];
	snprintf(tmp, sizeof tmp, "%s.part", path);
	FILE *f = fopen(tmp, "wb");
	if (!f) { perror(tmp); return -1; }
	int ok = fwrite(&h, sizeof h, 1, f) == 1;
	uint64_t rel = 0;
	for (int i = 0; i < m->nt && ok; i++) {
		const tr_tensor *t = &m->t[i];
		mlf2_tensor e = {0};
		memcpy(e.name, t->name, sizeof e.name);
		e.type = MLF2_TYPE_F32;
		e.n_dims = t->rows == 1 ? 1 : 2;
		e.dims[0] = t->cols; e.dims[1] = t->rows == 1 ? 0 : t->rows;
		e.nbytes = (uint64_t)t->rows * t->cols * 4;
		e.data_off = rel;
		rel += (e.nbytes + MLF2_ALIGN - 1) / MLF2_ALIGN * MLF2_ALIGN;
		ok = fwrite(&e, sizeof e, 1, f) == 1;
	}
	static const char zero[MLF2_ALIGN];
	ok = ok && fwrite(tok, 1, tok_len, f) == tok_len;
	ok = ok && fwrite(zero, 1, data_off - tok_off - tok_len, f) == data_off - tok_off - tok_len;
	for (int i = 0; i < m->nt && ok; i++) {
		const tr_tensor *t = &m->t[i];
		size_t nb = (size_t)t->rows * t->cols * 4, pad = (MLF2_ALIGN - nb % MLF2_ALIGN) % MLF2_ALIGN;
		ok = fwrite(m->p + t->off, 1, nb, f) == nb && fwrite(zero, 1, pad, f) == pad;
	}
	if (fclose(f) != 0) ok = 0;
	if (!ok || rename(tmp, path) != 0) { fprintf(stderr, "train: could not write %s\n", path); remove(tmp); return -1; }
	return 0;
}

tr_model *tr_load(const char *path, void **tok, size_t *tok_len, int *eos) {
	hakm_model f;
	if (mlf2_open(path, &f) != 0) return NULL;
	const mlf2_header *h = &f.h;
	tr_config c = { (int)h->vocab, (int)h->d_model, (int)h->n_layers, (int)h->n_heads,
	                (int)h->n_kv_heads, (int)h->head_dim, (int)h->ffn_dim, (int)h->ctx,
	                h->rope_theta, h->rms_eps };
	tr_model *m = NULL;
	if (h->arch != MLF2_ARCH_HAKO || h->flags != MLF2_FLAG_TIED_EMBED || tr_check(&c)) {
		fprintf(stderr, "train: %s is not a model this trainer wrote (arch %s, flags %#x)\n",
		        path, mlf2_arch_name(h->arch), h->flags);
		goto out;
	}
	if (!(m = tr_alloc(&c))) goto out;
	for (int i = 0; i < m->nt; i++) {
		const tr_tensor *t = &m->t[i];
		const mlf2_tensor *e = mlf2_find(&f, t->name);
		if (!e || e->type != MLF2_TYPE_F32 || e->dims[0] != (uint64_t)t->cols ||
		    e->nbytes != (uint64_t)t->rows * t->cols * 4) {
			fprintf(stderr, "train: %s: tensor %s is missing, not f32 or the wrong shape\n", path, t->name);
			tr_free(m); m = NULL; goto out;
		}
		memcpy(m->p + t->off, mlf2_data(&f, e), e->nbytes);
	}
	if (tok) {
		size_t avail = (size_t)((const unsigned char *)f.map + f.map_size - f.tok);
		*tok_len = tok_blob_len(f.tok, avail);
		*tok = *tok_len ? malloc(*tok_len) : NULL;
		if (!*tok) { fprintf(stderr, "train: %s: tokenizer unreadable\n", path); tr_free(m); m = NULL; goto out; }
		memcpy(*tok, f.tok, *tok_len);
	}
	if (eos) *eos = (int)h->eos_id;
out:
	mlf2_close(&f);
	return m;
}

typedef struct { char magic[4]; uint32_t version; uint64_t n, rng; int64_t step; tr_config c; } opt_header;

int tr_save_opt(const tr_model *m, const char *path, uint64_t rng) {
	char tmp[4096];
	snprintf(tmp, sizeof tmp, "%s.part", path);
	FILE *f = fopen(tmp, "wb");
	if (!f) { perror(tmp); return -1; }
	opt_header h = { { 'H', 'K', 'O', 'P' }, 1, m->n, rng, m->step, m->c };
	int ok = fwrite(&h, sizeof h, 1, f) == 1 &&
	         fwrite(m->mo, sizeof(float), m->n, f) == m->n && fwrite(m->ve, sizeof(float), m->n, f) == m->n;
	if (fclose(f) != 0) ok = 0;
	if (!ok || rename(tmp, path) != 0) { fprintf(stderr, "train: could not write %s\n", path); remove(tmp); return -1; }
	return 0;
}

int tr_load_opt(tr_model *m, const char *path, uint64_t *rng) {
	FILE *f = fopen(path, "rb");
	if (!f) { perror(path); return -1; }
	opt_header h;
	int ok = fread(&h, sizeof h, 1, f) == 1 && !memcmp(h.magic, "HKOP", 4) && h.version == 1 &&
	         h.n == m->n && !memcmp(&h.c, &m->c, sizeof h.c);
	ok = ok && fread(m->mo, sizeof(float), m->n, f) == m->n && fread(m->ve, sizeof(float), m->n, f) == m->n;
	fclose(f);
	if (!ok) { fprintf(stderr, "train: %s does not belong to this model\n", path); return -1; }
	m->step = h.step;
	if (rng) *rng = h.rng;
	return 0;
}

/* ── byte-level BPE ───────────────────────────────────────────────────────── */

static uint64_t fnv(const unsigned char *p, size_t n) {
	uint64_t h = 1469598103934665603ULL;
	for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; }
	return h;
}

/* byte string -> int, open addressing, keys borrowed */
typedef struct { const unsigned char **k; int *kl, *v; size_t cap, n; } smap;

static int smap_find(const smap *s, const unsigned char *k, int kl, size_t *slot) {
	size_t i = fnv(k, kl) & (s->cap - 1);
	while (s->k[i]) {
		if (s->kl[i] == kl && !memcmp(s->k[i], k, kl)) { *slot = i; return s->v[i]; }
		i = (i + 1) & (s->cap - 1);
	}
	*slot = i;
	return -1;
}
static int smap_grow(smap *s) {
	smap t = { calloc(s->cap * 2, sizeof *t.k), malloc(s->cap * 2 * sizeof(int)),
	           malloc(s->cap * 2 * sizeof(int)), s->cap * 2, 0 };
	if (!t.k || !t.kl || !t.v) { free(t.k); free(t.kl); free(t.v); return -1; }
	for (size_t i = 0; i < s->cap; i++)
		if (s->k[i]) {
			size_t at; smap_find(&t, s->k[i], s->kl[i], &at);
			t.k[at] = s->k[i]; t.kl[at] = s->kl[i]; t.v[at] = s->v[i]; t.n++;
		}
	free(s->k); free(s->kl); free(s->v);
	*s = t;
	return 0;
}
/* the value for k, inserting val when absent; -2 on oom */
static int smap_get_or_put(smap *s, const unsigned char *k, int kl, int val) {
	if ((s->n + 1) * 2 > s->cap && smap_grow(s)) return -2;
	size_t at;
	int v = smap_find(s, k, kl, &at);
	if (v >= 0) return v;
	s->k[at] = k; s->kl[at] = kl; s->v[at] = val; s->n++;
	return -1;
}

/* pair (a<<32|b) -> count, with a stamp so a merge lists each touched pair once */
typedef struct { uint64_t key; int64_t cnt; int stamp; int used; } pslot;
typedef struct { pslot *s; size_t cap, n; } pmap;

static pslot *pmap_at(pmap *p, uint64_t key) {
	if ((p->n + 1) * 2 > p->cap) {
		pmap t = { calloc(p->cap * 2, sizeof(pslot)), p->cap * 2, 0 };
		if (!t.s) return NULL;
		for (size_t i = 0; i < p->cap; i++)
			if (p->s[i].used) {
				size_t j = (p->s[i].key * 0x9E3779B97F4A7C15ULL) >> 20 & (t.cap - 1);
				while (t.s[j].used) j = (j + 1) & (t.cap - 1);
				t.s[j] = p->s[i]; t.n++;
			}
		free(p->s);
		*p = t;
	}
	size_t i = (key * 0x9E3779B97F4A7C15ULL) >> 20 & (p->cap - 1);
	while (p->s[i].used && p->s[i].key != key) i = (i + 1) & (p->cap - 1);
	if (!p->s[i].used) { p->s[i] = (pslot){ key, 0, -1, 1 }; p->n++; }
	return &p->s[i];
}

/* max-heap of (count, pair), stale entries skipped when popped */
typedef struct { int64_t cnt; uint64_t key; } hent;
typedef struct { hent *e; size_t n, cap; } heap;

static int hless(hent a, hent b) { return a.cnt < b.cnt || (a.cnt == b.cnt && a.key > b.key); }
static int heap_push(heap *h, int64_t cnt, uint64_t key) {
	if (h->n == h->cap) {
		size_t nc = h->cap ? h->cap * 2 : 1024;
		hent *e = realloc(h->e, nc * sizeof *e);
		if (!e) return -1;
		h->e = e; h->cap = nc;
	}
	size_t i = h->n++;
	h->e[i] = (hent){ cnt, key };
	while (i && hless(h->e[(i - 1) / 2], h->e[i])) {
		hent t = h->e[i]; h->e[i] = h->e[(i - 1) / 2]; h->e[(i - 1) / 2] = t;
		i = (i - 1) / 2;
	}
	return 0;
}
static void heap_pop(heap *h) {
	h->e[0] = h->e[--h->n];
	for (size_t i = 0;;) {
		size_t l = 2 * i + 1, r = l + 1, b = i;
		if (l < h->n && hless(h->e[b], h->e[l])) b = l;
		if (r < h->n && hless(h->e[b], h->e[r])) b = r;
		if (b == i) break;
		hent t = h->e[i]; h->e[i] = h->e[b]; h->e[b] = t;
		i = b;
	}
}

static int utf8_put(int cp, unsigned char *buf) {
	if (cp < 0x80) { buf[0] = (unsigned char)cp; return 1; }
	buf[0] = (unsigned char)(0xC0 | (cp >> 6));
	buf[1] = (unsigned char)(0x80 | (cp & 0x3F));
	return 2;
}

static const char *const SPECIALS[] = { "<|endoftext|>", "<|im_start|>", "<|im_end|>" };
enum { N_SPECIAL = 3, MAX_PIECE = 2048 };

int bpe_train(const char *text, size_t len, int vocab, void **blob_out, size_t *blob_len, int *eos) {
	const unsigned char *s = (const unsigned char *)text;
	int target = vocab - N_SPECIAL, ret = -1;
	if (target < 256) return -1;
	smap words = { calloc(1 << 16, sizeof(char *)), malloc((1 << 16) * sizeof(int)),
	               malloc((1 << 16) * sizeof(int)), 1 << 16, 0 };
	smap vmap = { calloc(1 << 14, sizeof(char *)), malloc((1 << 14) * sizeof(int)),
	              malloc((1 << 14) * sizeof(int)), 1 << 14, 0 };
	pmap pairs = { calloc(1 << 16, sizeof(pslot)), 1 << 16, 0 };
	heap hp = {0};
	int nw = 0, wcap = 1 << 16, ntok = 0, nmerge = 0, mcap = vocab;
	int64_t *freq = malloc(wcap * sizeof *freq);
	const unsigned char **wp = malloc(wcap * sizeof *wp);
	int *wl = malloc(wcap * sizeof *wl);
	int *sym = NULL, *wofs = NULL, *wn = NULL;
	unsigned char **tstr = calloc(vocab, sizeof *tstr);
	int *tlen = calloc(vocab, sizeof *tlen);
	int (*merge)[2] = malloc((size_t)vocab * sizeof *merge);
	uint64_t *dirty = NULL; size_t ndirty = 0, dcap = 0;
	unsigned char *blob = NULL;
	if (!words.k || !words.kl || !words.v || !vmap.k || !vmap.kl || !vmap.v || !pairs.s ||
	    !freq || !wp || !wl || !tstr || !tlen || !merge) goto done;

	/* 1. pieces, split exactly as tok_encode splits: one optional leading space,
	   then bytes up to the next space; the encoder reads at most 2048 of them */
	for (size_t i = 0; i < len;) {
		size_t st = i;
		if (s[i] == ' ') i++;
		while (i < len && s[i] != ' ') i++;
		int pl = (int)(i - st) > MAX_PIECE ? MAX_PIECE : (int)(i - st);
		int w = smap_get_or_put(&words, s + st, pl, nw);
		if (w == -2) goto done;
		if (w == -1) {
			if (nw == wcap) {
				wcap *= 2;
				int64_t *f2 = realloc(freq, wcap * sizeof *f2); if (f2) freq = f2;
				const unsigned char **p2 = realloc(wp, wcap * sizeof *p2); if (p2) wp = p2;
				int *l2 = realloc(wl, wcap * sizeof *l2); if (l2) wl = l2;
				if (!f2 || !p2 || !l2) goto done;
			}
			w = nw++;
			freq[w] = 0; wp[w] = s + st; wl[w] = pl;
		}
		freq[w]++;
	}

	/* 2. symbols: each word starts as its bytes (token id = byte value) */
	size_t total = 0;
	for (int w = 0; w < nw; w++) total += wl[w];
	sym = malloc(total * sizeof *sym); wofs = malloc((size_t)nw * sizeof *wofs); wn = malloc((size_t)nw * sizeof *wn);
	if (!sym || !wofs || !wn) goto done;
	total = 0;
	for (int w = 0; w < nw; w++) {
		wofs[w] = (int)total; wn[w] = wl[w];
		for (int i = 0; i < wl[w]; i++) sym[total++] = wp[w][i];
	}
	for (int b = 0, extra = 0; b < 256; b++) {     /* GPT-2's byte -> codepoint map, as in bpe.c */
		int printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
		int cp = printable ? b : 256 + extra++;
		tstr[b] = malloc(2);
		if (!tstr[b]) goto done;
		tlen[b] = utf8_put(cp, tstr[b]);
		if (smap_get_or_put(&vmap, tstr[b], tlen[b], b) == -2) goto done;
	}
	ntok = 256;
	for (int w = 0; w < nw; w++)
		for (int i = 0; i + 1 < wn[w]; i++) {
			pslot *ps = pmap_at(&pairs, (uint64_t)sym[wofs[w] + i] << 32 | (uint32_t)sym[wofs[w] + i + 1]);
			if (!ps) goto done;
			ps->cnt += freq[w];
		}
	for (size_t i = 0; i < pairs.cap; i++)
		if (pairs.s[i].used && pairs.s[i].cnt >= 2 && heap_push(&hp, pairs.s[i].cnt, pairs.s[i].key)) goto done;

	/* 3. merge the most frequent pair until the vocab is full */
	while (ntok < target && hp.n) {
		hent top = hp.e[0];
		heap_pop(&hp);
		pslot *ps = pmap_at(&pairs, top.key);
		if (!ps) goto done;
		if (ps->cnt != top.cnt || top.cnt < 2) continue;
		int A = (int)(top.key >> 32), B = (int)(uint32_t)top.key;
		unsigned char *str = malloc(tlen[A] + tlen[B]);
		if (!str) goto done;
		memcpy(str, tstr[A], tlen[A]); memcpy(str + tlen[A], tstr[B], tlen[B]);
		int Z = smap_get_or_put(&vmap, str, tlen[A] + tlen[B], ntok);
		if (Z == -2) { free(str); goto done; }
		if (Z == -1) { Z = ntok++; tstr[Z] = str; tlen[Z] = tlen[A] + tlen[B]; }
		else free(str);       /* the same string by another pair: one id, two merges */
		if (nmerge == mcap) {
			int (*m2)[2] = realloc(merge, (size_t)mcap * 2 * sizeof *merge);
			if (!m2) goto done;
			merge = m2; mcap *= 2;
		}
		merge[nmerge][0] = A; merge[nmerge][1] = B; nmerge++;

		ndirty = 0;
		for (int w = 0; w < nw; w++) {
			int *q = sym + wofs[w], n = wn[w], hit = 0;
			for (int i = 0; i + 1 < n && !hit; i++) hit = q[i] == A && q[i + 1] == B;
			if (!hit) continue;
			for (int pass = 0; pass < 2; pass++) {
				for (int i = 0; i + 1 < n; i++) {
					pslot *e = pmap_at(&pairs, (uint64_t)q[i] << 32 | (uint32_t)q[i + 1]);
					if (!e) goto done;
					e->cnt += pass ? freq[w] : -freq[w];
					if (e->stamp != nmerge) {
						e->stamp = nmerge;
						if (ndirty == dcap) {
							dcap = dcap ? dcap * 2 : 4096;
							uint64_t *d2 = realloc(dirty, dcap * sizeof *d2);
							if (!d2) goto done;
							dirty = d2;
						}
						dirty[ndirty++] = e->key;
					}
				}
				if (pass) break;
				int o = 0;
				for (int i = 0; i < n;) {
					if (i + 1 < n && q[i] == A && q[i + 1] == B) { q[o++] = Z; i += 2; }
					else q[o++] = q[i++];
				}
				wn[w] = n = o;
			}
		}
		for (size_t i = 0; i < ndirty; i++) {
			pslot *e = pmap_at(&pairs, dirty[i]);
			if (!e || (e->cnt >= 2 && heap_push(&hp, e->cnt, e->key))) goto done;
		}
	}

	/* 4. the blob: tokens (bytes, merges, specials, pads), then "A B" merges */
	size_t bl = 8;
	for (int i = 0; i < ntok; i++) bl += 5 + tlen[i];
	for (int i = 0; i < N_SPECIAL; i++) bl += 5 + strlen(SPECIALS[i]);
	bl += (size_t)(vocab - ntok - N_SPECIAL) * (5 + 16);
	for (int i = 0; i < nmerge; i++) bl += 4 + tlen[merge[i][0]] + 1 + tlen[merge[i][1]];
	if (!(blob = malloc(bl))) goto done;
	unsigned char *o = blob;
	uint32_t u = (uint32_t)vocab; memcpy(o, &u, 4);
	u = (uint32_t)nmerge; memcpy(o + 4, &u, 4);
	o += 8;
	#define TOKREC(type, str, n) do { *o++ = (type); uint32_t l_ = (uint32_t)(n); memcpy(o, &l_, 4); \
		memcpy(o + 4, (str), l_); o += 4 + l_; } while (0)
	for (int i = 0; i < ntok; i++) TOKREC(MLF2_TOKTYPE_NORMAL, tstr[i], tlen[i]);
	for (int i = 0; i < N_SPECIAL; i++) TOKREC(MLF2_TOKTYPE_CONTROL, SPECIALS[i], strlen(SPECIALS[i]));
	for (int i = ntok + N_SPECIAL; i < vocab; i++) {
		char pad[16];
		int n = snprintf(pad, sizeof pad, "[PAD%d]", i);
		TOKREC(5, pad, n);
	}
	#undef TOKREC
	for (int i = 0; i < nmerge; i++) {
		int a = merge[i][0], b = merge[i][1];
		uint32_t l = (uint32_t)(tlen[a] + 1 + tlen[b]);
		memcpy(o, &l, 4); o += 4;
		memcpy(o, tstr[a], tlen[a]); o += tlen[a];
		*o++ = ' ';
		memcpy(o, tstr[b], tlen[b]); o += tlen[b];
	}
	*blob_out = blob; *blob_len = (size_t)(o - blob);
	*eos = ntok;
	blob = NULL;
	ret = nmerge;
done:
	if (ret < 0) fprintf(stderr, "train: out of memory training the tokenizer\n");
	free(blob);
	free(words.k); free(words.kl); free(words.v);
	free(vmap.k); free(vmap.kl); free(vmap.v);
	free(pairs.s); free(hp.e); free(dirty);
	free(freq); free(wp); free(wl); free(sym); free(wofs); free(wn);
	if (tstr) for (int i = 0; i < vocab; i++) free(tstr[i]);
	free(tstr); free(tlen); free(merge);
	return ret;
}
