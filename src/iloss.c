/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* iloss.c — a trainer model's loss in integers (iloss.h). Every intermediate is
 * held to a range its next step cannot overflow (int32 values saturate, wide sums
 * go to __int128 when their bounds say so), so a model with absurd weights scores
 * badly rather than differently. */
#include "iloss.h"
#include "nn.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#ifdef __FAST_MATH__
#error "iloss.c needs IEEE doubles as written: build it without -ffast-math"
#endif
#ifndef __SIZEOF_INT128__
#error "iloss.c needs __int128 (gcc or clang on a 64-bit host, clang on wasm32)"
#endif
_Static_assert((-3 >> 1) == -2 && (-1LL >> 1) == -1, "signed right shifts are arithmetic");

typedef __int128 i128;
#define LN2 0.6931471805599453094
#define LOG2E (1.0 / LN2)
#define BLK 32               /* values sharing one exponent in an int8 input */
#define I32 2147483647

/* ── tables, in doubles ──
 * Only + - / sqrt and products, each product through a volatile so it is rounded
 * on its own: a compiler allowed to fuse it into the add after it (-ffp-contract
 * =fast, the trainer's flag) would round once where another device rounds twice.
 * From libm only floor, nearbyint, ldexp and frexp, which are exact. */
static double mul(double a, double b) { volatile double r = a * b; return r; }
static double d_ln(double x) {
	int k = 0;
	while (x > 1.5) { x *= 0.5; k++; }
	while (x < 0.75) { x *= 2; k--; }
	double y = (x - 1) / (x + 1), y2 = mul(y, y), t = y, s = 0;
	for (int n = 0; n < 60; n++) { s += t / (2 * n + 1); t = mul(t, y2); }
	return 2 * s + mul(k, LN2);
}
static double d_exp(double x) {
	double k = nearbyint(x / LN2), r = x - mul(k, LN2), t = 1, s = 1;
	for (int n = 1; n < 30; n++) { t = mul(t, r / n); s += t; }
	return ldexp(s, (int)k);
}
static void d_sincos(double a, double *sn, double *cs) {
	const double TWO_PI = 6.283185307179586477;
	double r = a - mul(nearbyint(a / TWO_PI), TWO_PI), r2 = mul(r, r), ts = r, tc = 1, s = r, c = 1;
	for (int n = 1; n < 30; n++) {
		ts = mul(ts, -r2 / ((2 * n) * (2 * n + 1))); s += ts;
		tc = mul(tc, -r2 / ((2 * n - 1) * (2 * n))); c += tc;
	}
	*sn = s; *cs = c;
}
/* a double to the nearest int32, held to range; NaN 0 */
static int32_t q32(double v) {
	if (!(v == v)) return 0;
	v = floor(v + 0.5);
	return v > I32 ? I32 : v < -I32 ? -I32 : (int32_t)v;
}

static int64_t EXP2T[1025], LOG2T[1025], C_L2E;
static pthread_once_t tables_once = PTHREAD_ONCE_INIT;
static void tables(void) {
	for (int k = 0; k <= 1024; k++) {
		EXP2T[k] = q32(mul(d_exp(mul(-(double)k, LN2) / 1024), 1073741824.0));   /* 2^(-k/1024), Q30 */
		LOG2T[k] = q32(mul(mul(d_ln(1 + k / 1024.0), LOG2E), 65536.0));        /* log2(1 + k/1024), Q16 */
	}
	C_L2E = q32(mul(LOG2E, 1073741824.0));                                       /* log2 e, Q30 */
}

/* ── fixed point ── */
static inline int64_t rsh(int64_t v, int k) { return k <= 0 ? v * ((int64_t)1 << -k) : (v + ((int64_t)1 << (k - 1))) >> k; }
static inline int32_t sat32(int64_t v) { return v > I32 ? I32 : v < -I32 ? -I32 : (int32_t)v; }
/* v x 2^-k rounded, held to int32; |v| < 2^120 (2^84 when k < 0) */
static inline int32_t fix32(i128 v, int k) {
	if (k > 0) { if (k > 100) return 0; v = (v + ((i128)1 << (k - 1))) >> k; }
	else if (k < 0) { if (!v) return 0; if (-k > 40) return v > 0 ? I32 : -I32; v *= (i128)1 << -k; }
	return v > I32 ? I32 : v < -I32 ? -I32 : (int32_t)v;
}
static inline int64_t divr(int64_t n, int64_t d) { return n >= 0 ? (n + d / 2) / d : -((-n + d / 2) / d); }
static inline int64_t divr128(i128 n, int64_t d) { return (int64_t)(n >= 0 ? (n + d / 2) / d : -((-n + d / 2) / d)); }
static inline int bits(uint64_t v) { return v ? 64 - __builtin_clzll(v) : 0; }
/* 2^(d / 2^16) for d <= 0, Q30 */
static inline int64_t exp2_neg(int64_t d) {
	int64_t a = -d, n = a >> 16;
	if (n >= 31) return 0;
	int f = (int)(a & 0xFFFF), i = f >> 6, fr = f & 63;
	return (EXP2T[i] + (((EXP2T[i + 1] - EXP2T[i]) * fr) >> 6)) >> n;
}
/* log2(S / 2^30), Q16, S > 0 */
static inline int64_t log2_q30(int64_t S) {
	int b = bits((uint64_t)S) - 1;
	int64_t m = S - ((int64_t)1 << b);
	int64_t f = b >= 30 ? m >> (b - 30) : m << (30 - b);   /* Q30 fraction */
	int i = (int)(f >> 20), fr = (int)((f >> 10) & 1023);
	return ((int64_t)(b - 30) << 16) + LOG2T[i] + (((LOG2T[i + 1] - LOG2T[i]) * fr) >> 10);
}
static int64_t isqrt128(i128 v) {
	if (v <= 0) return 0;
	int64_t r = (int64_t)sqrt((double)v);   /* a guess; the loops make it exact */
	while (r > 0 && (i128)r * r > v) r--;
	while ((i128)(r + 1) * (r + 1) <= v) r++;
	return r;
}

/* ── the integer model ── */
typedef struct { int rows, cols; int8_t *q; int32_t *mant; int16_t *ex; } qmat;

/* int8 rows, each scaled by its largest |w| / 127 */
static int qmat_make(qmat *Q, const float *w, int rows, int cols) {
	Q->rows = rows; Q->cols = cols;
	Q->q = malloc((size_t)rows * cols); Q->mant = malloc((size_t)rows * sizeof *Q->mant); Q->ex = malloc((size_t)rows * sizeof *Q->ex);
	if (!Q->q || !Q->mant || !Q->ex) return -1;
	for (int r = 0; r < rows; r++) {
		const float *wr = w + (size_t)r * cols;
		int8_t *q = Q->q + (size_t)r * cols;
		double mx = 0;
		for (int c = 0; c < cols; c++) { double a = fabs(wr[c]); if (a > mx && a <= 3.4028234663852886e38) mx = a; }
		double s = mx > 0 ? mx / 127 : 1;
		for (int c = 0; c < cols; c++) {
			double v = wr[c];
			int32_t x = v != v ? 0 : v > 3.4028234663852886e38 ? 127 : v < -3.4028234663852886e38 ? -127 : q32(v / s);
			q[c] = (int8_t)(x > 127 ? 127 : x < -127 ? -127 : x);
		}
		int e; double f = frexp(s, &e);
		Q->mant[r] = q32(f * 2147483648.0);
		Q->ex[r] = (int16_t)(e - 31);
	}
	return 0;
}
static void qmat_free(qmat *Q) { if (Q) { free(Q->q); free(Q->mant); free(Q->ex); } }

struct il_model {
	int V, D, L, H, KH, hd, F, ctx, Q, KV, E, K, dense, w;   /* w: the widest input, max(D, Q, F) */
	int32_t *emb;                     /* Q16 */
	qmat head;                        /* the lm head, tied to the embedding */
	int32_t *nw;                      /* norm weights Q14: per layer attn, ffn; then out */
	qmat *wq, *wk, *wv, *wo, *wg, *wu, *wd, *router;   /* per layer; wg wu wd when dense or shared */
	qmat *xg, *xu, *xd;               /* experts, l x E + e */
	int32_t *cs, *sn;                 /* rope Q14 [ctx][hd / 2] */
	int64_t c_att;                    /* log2 e / sqrt(hd), Q30 */
	int64_t eps;                      /* Q32 */
};

void il_free(il_model *I) {
	if (!I) return;
	qmat *per[] = { I->wq, I->wk, I->wv, I->wo, I->wg, I->wu, I->wd, I->router };
	for (int j = 0; j < 8; j++) { if (per[j]) for (int l = 0; l < I->L; l++) qmat_free(&per[j][l]); free(per[j]); }
	qmat *ex[] = { I->xg, I->xu, I->xd };
	for (int j = 0; j < 3; j++) { if (ex[j]) for (int i = 0; i < I->L * I->E; i++) qmat_free(&ex[j][i]); free(ex[j]); }
	qmat_free(&I->head);
	free(I->emb); free(I->nw); free(I->cs); free(I->sn);
	free(I);
}

il_model *il_new(tr_model *m) {
	pthread_once(&tables_once, tables);
	const tr_config *c = tr_cfg(m);
	const float *p = tr_params(m);
	il_model *I = calloc(1, sizeof *I);
	if (!I) return NULL;
	I->V = c->vocab; I->D = c->dim; I->L = c->layers; I->H = c->heads; I->KH = c->kv_heads; I->hd = c->head_dim;
	I->F = c->ffn; I->ctx = c->ctx; I->Q = I->H * I->hd; I->KV = I->KH * I->hd;
	I->E = c->experts; I->K = c->experts ? c->top_k : 0; I->dense = !c->experts || c->shared;
	I->w = I->D > I->Q ? I->D : I->Q;
	if (I->F > I->w) I->w = I->F;
	int D = I->D, V = I->V, Q = I->Q, KV = I->KV, F = I->F, L = I->L, E = I->E;
	size_t at = 0;
	I->emb = malloc((size_t)V * D * sizeof *I->emb);
	I->nw = malloc(((size_t)2 * L + 1) * D * sizeof *I->nw);
	I->wq = calloc(L, sizeof(qmat)); I->wk = calloc(L, sizeof(qmat)); I->wv = calloc(L, sizeof(qmat)); I->wo = calloc(L, sizeof(qmat));
	I->wg = calloc(L, sizeof(qmat)); I->wu = calloc(L, sizeof(qmat)); I->wd = calloc(L, sizeof(qmat)); I->router = calloc(L, sizeof(qmat));
	if (E) { I->xg = calloc((size_t)L * E, sizeof(qmat)); I->xu = calloc((size_t)L * E, sizeof(qmat)); I->xd = calloc((size_t)L * E, sizeof(qmat)); }
	int half = I->hd / 2;
	I->cs = malloc((size_t)I->ctx * half * sizeof *I->cs); I->sn = malloc((size_t)I->ctx * half * sizeof *I->sn);
	if (!I->emb || !I->nw || !I->wq || !I->wk || !I->wv || !I->wo || !I->wg || !I->wu || !I->wd || !I->router ||
	    (E && (!I->xg || !I->xu || !I->xd)) || !I->cs || !I->sn) goto bad;
	/* the trainer's layout (train.h) */
	for (size_t i = 0; i < (size_t)V * D; i++) I->emb[i] = q32(p[i] * 65536.0);
	if (qmat_make(&I->head, p, V, D)) goto bad;
	at += (size_t)V * D;
	#define NORM(slot) do { for (int i = 0; i < D; i++) I->nw[(size_t)(slot) * D + i] = q32(p[at + i] * 16384.0); at += D; } while (0)
	#define MAT(dst, r, k) do { if (qmat_make(dst, p + at, r, k)) goto bad; at += (size_t)(r) * (k); } while (0)
	for (int l = 0; l < L; l++) {
		NORM(2 * l);
		MAT(&I->wq[l], Q, D); MAT(&I->wk[l], KV, D); MAT(&I->wv[l], KV, D); MAT(&I->wo[l], D, Q);
		NORM(2 * l + 1);
		if (E) MAT(&I->router[l], E, D);
		if (I->dense) { MAT(&I->wg[l], F, D); MAT(&I->wu[l], F, D); MAT(&I->wd[l], D, F); }
	}
	NORM(2 * L);
	for (int l = 0; l < L; l++)
		for (int e = 0; e < E; e++) {
			if (at != tr_expert_off(m, l, e)) goto bad;
			qmat *x = &I->xg[(size_t)l * E + e];
			MAT(x, F, D); MAT(&I->xu[(size_t)l * E + e], F, D); MAT(&I->xd[(size_t)l * E + e], D, F);
		}
	#undef NORM
	#undef MAT
	if (at != tr_n_params(m)) goto bad;
	double lt = d_ln(c->rope_theta);
	for (int i = 0; i < half; i++) {
		double freq = d_exp(mul(-((double)(2 * i) / I->hd), lt));
		for (int pos = 0; pos < I->ctx; pos++) {
			double s, co;
			d_sincos(mul(pos, freq), &s, &co);
			I->cs[(size_t)pos * half + i] = q32(co * 16384.0);
			I->sn[(size_t)pos * half + i] = q32(s * 16384.0);
		}
	}
	I->c_att = q32(LOG2E / sqrt((double)I->hd) * 1073741824.0);
	I->eps = (int64_t)floor((double)c->rms_eps * 4294967296.0 + 0.5);
	return I;
bad:
	il_free(I);
	return NULL;
}

/* ── one window, TC tokens at a time where tokens do not mix ── */
#define TC 32
typedef struct {
	int32_t *x, *q, *k, *v, *ao;             /* the window: [T][D], [T][Q], [T][KV] x2, [T][Q] */
	int64_t *e, *acc; i128 *acc2;            /* attention: [T], [hd], [hd] */
	int32_t *xn, *g, *u, *y, *z, *lg;        /* a chunk: [TC][D], [TC][F] x2, [TC][D], [TC][E], [TC][V] */
	int8_t *qa, *qn, *gq;                    /* int8 inputs [TC][w], the normed ffn input [TC][D], an expert's rows [TC][D] */
	int *eb, *en, *geb, *ea, *ean, *gea;     /* their block exponents and sum exponents */
	int *sel, *gt; int64_t *gw, *lz;         /* picks [TC][K], an expert's tokens [TC], gates [TC][K], log2 logits [V] */
} scratch;

/* Q f ints to int8, one power-of-two exponent per BLK of them (value = q x 2^eb):
 * returns the exponent the products are summed at, the largest less 24 or the
 * smallest, whichever is higher, so a row's sum never passes 2^52 */
static int act8(int8_t *q, int *eb, const int32_t *y, int n, int f) {
	int emin = 1 << 20, emax = -(1 << 20);
	for (int k = 0; k * BLK < n; k++) {
		int c0 = k * BLK, m = n - c0 < BLK ? n - c0 : BLK, sh = 0;
		int64_t mx = 0;
		for (int i = 0; i < m; i++) { int64_t a = y[c0 + i] < 0 ? -(int64_t)y[c0 + i] : y[c0 + i]; if (a > mx) mx = a; }
		while (rsh(mx, sh) > 127) sh++;
		for (int i = 0; i < m; i++) { int64_t v = rsh(y[c0 + i], sh); q[c0 + i] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v); }
		eb[k] = sh - f;
		if (eb[k] < emin) emin = eb[k];
		if (eb[k] > emax) emax = eb[k];
	}
	return emax - 24 > emin ? emax - 24 : emin;
}
static void act8_rows(int8_t *q, int ldq, int *eb, int ldb, int *ea, const int32_t *y, int ldy, int n, int cols, int f) {
	for (int t = 0; t < n; t++) ea[t] = act8(q + (size_t)t * ldq, eb + (size_t)t * ldb, y + (size_t)t * ldy, cols, f);
}
/* out[t][r] (Q qo) = W[r] . a[t] for n rows of inputs, or += with add; a weight row
 * is read once for all of them */
static void imatmul(int32_t *out, int ldo, const qmat *W, const int8_t *a, int lda, const int *eb, int ldb, const int *ea,
                    int n, int qo, int add) {
	int nb = (W->cols + BLK - 1) / BLK;
	for (int r = 0; r < W->rows; r++) {
		const int8_t *w = W->q + (size_t)r * W->cols;
		for (int t = 0; t < n; t++) {
			const int8_t *at = a + (size_t)t * lda;
			const int *bt = eb + (size_t)t * ldb;
			int64_t acc = 0;
			for (int k = 0; k < nb; k++) {
				int c0 = k * BLK, m = W->cols - c0 < BLK ? W->cols - c0 : BLK;
				int32_t s = 0;
				if (m == BLK) for (int c = 0; c < BLK; c++) s += w[c0 + c] * at[c0 + c];
				else for (int c = 0; c < m; c++) s += w[c0 + c] * at[c0 + c];
				int d = bt[k] - ea[t];
				acc += d >= 0 ? (int64_t)s * ((int64_t)1 << d) : rsh(s, -d);
			}
			int32_t v = fix32((i128)acc * W->mant[r], -(ea[t] + W->ex[r] + qo));
			int32_t *o = out + (size_t)t * ldo + r;
			*o = add ? sat32((int64_t)*o + v) : v;
		}
	}
}
/* rmsnorm of a Q16 row times Q14 weights -> Q12 */
static void inorm(int32_t *y, const int32_t *x, const int32_t *w, int D, int64_t eps) {
	i128 ss = 0;
	for (int i = 0; i < D; i++) ss += (i128)x[i] * x[i];
	int64_t r = isqrt128(ss / D + eps);   /* Q16 */
	if (r < 1) r = 1;
	int64_t inv = ((int64_t)1 << 46) / r; /* Q30 */
	for (int i = 0; i < D; i++) y[i] = fix32((i128)x[i] * inv * w[i], 48);
}
/* silu(g) x u, Q12, into g */
static void swiglu(int32_t *g, const int32_t *u, size_t n) {
	for (size_t i = 0; i < n; i++) {
		int64_t x = g[i], t = rsh(x * C_L2E, 12 + 30 - 16), a = t < 0 ? -t : t;
		int64_t sp = ((int64_t)1 << 60) / (((int64_t)1 << 30) + exp2_neg(-a));   /* sigmoid(|g|), Q30 */
		int64_t sig = x >= 0 ? sp : ((int64_t)1 << 30) - sp;
		g[i] = sat32(rsh(rsh(x * sig, 30) * u[i], 12));
	}
}
static uint64_t maxabs(const int32_t *x, size_t n) {
	uint64_t m = 0;
	for (size_t i = 0; i < n; i++) { uint64_t a = x[i] < 0 ? (uint64_t)(-(int64_t)x[i]) : (uint64_t)x[i]; if (a > m) m = a; }
	return m;
}

static void attention(const il_model *I, int T, scratch *S) {
	int Q = I->Q, KV = I->KV, hd = I->hd, grp = I->H / I->KH;
	/* exact sums either way: int64 when the bounds allow, else __int128 */
	int qk64 = bits(maxabs(S->q, (size_t)T * Q)) + bits(maxabs(S->k, (size_t)T * KV)) + bits((uint64_t)hd) <= 62;
	int vb = bits(maxabs(S->v, (size_t)T * KV));
	for (int h = 0; h < I->H; h++) {
		int kh = h / grp;
		for (int i = 0; i < T; i++) {
			const int32_t *qi = S->q + (size_t)i * Q + h * hd;
			int64_t mz = INT64_MIN;
			for (int j = 0; j <= i; j++) {
				const int32_t *kj = S->k + (size_t)j * KV + kh * hd;
				i128 s;
				if (qk64) { int64_t s64 = 0; for (int d = 0; d < hd; d++) s64 += (int64_t)qi[d] * kj[d]; s = s64; }
				else { s = 0; for (int d = 0; d < hd; d++) s += (i128)qi[d] * kj[d]; }
				i128 zz = (s * I->c_att + ((i128)1 << 37)) >> 38;   /* Q24 x Q30 -> Q16 log2 units */
				int64_t z = zz > ((i128)1 << 40) ? (int64_t)1 << 40 : zz < -((i128)1 << 40) ? -((int64_t)1 << 40) : (int64_t)zz;
				S->e[j] = z;
				if (z > mz) mz = z;
			}
			int64_t Es = 0;
			for (int j = 0; j <= i; j++) { S->e[j] = exp2_neg(S->e[j] - mz); Es += S->e[j]; }
			int32_t *o = S->ao + (size_t)i * Q + h * hd;
			if (bits((uint64_t)Es) + vb <= 62) {
				memset(S->acc, 0, hd * sizeof *S->acc);
				for (int j = 0; j <= i; j++) {
					int64_t ej = S->e[j];
					const int32_t *vj = S->v + (size_t)j * KV + kh * hd;
					if (ej) for (int d = 0; d < hd; d++) S->acc[d] += ej * vj[d];
				}
				for (int d = 0; d < hd; d++) o[d] = sat32(divr(S->acc[d], Es));
			} else {
				memset(S->acc2, 0, hd * sizeof *S->acc2);
				for (int j = 0; j <= i; j++) {
					int64_t ej = S->e[j];
					const int32_t *vj = S->v + (size_t)j * KV + kh * hd;
					if (ej) for (int d = 0; d < hd; d++) S->acc2[d] += (i128)ej * vj[d];
				}
				for (int d = 0; d < hd; d++) o[d] = sat32(divr128(S->acc2[d], Es));
			}
		}
	}
}

/* the ffn of n tokens at x (normed input in qn, en, ean): the dense one, then the
 * routed experts, each one's tokens gathered so its rows are read once */
static void ffn(const il_model *I, int l, int32_t *x, int n, scratch *S) {
	int D = I->D, F = I->F, E = I->E, K = I->K, W = I->w, nbD = (D + BLK - 1) / BLK, nbW = (W + BLK - 1) / BLK;
	if (I->dense) {
		imatmul(S->g, F, &I->wg[l], S->qn, D, S->en, nbD, S->ean, n, 12, 0);
		imatmul(S->u, F, &I->wu[l], S->qn, D, S->en, nbD, S->ean, n, 12, 0);
		swiglu(S->g, S->u, (size_t)n * F);
		act8_rows(S->qa, W, S->eb, nbW, S->ea, S->g, F, n, F, 12);
		imatmul(x, D, &I->wd[l], S->qa, W, S->eb, nbW, S->ea, n, 16, 1);
	}
	if (!E) return;
	/* the router's scores (Q16), top_k of them, ties to the lower index, gates =
	 * the softmax over the picks (Q30) */
	imatmul(S->z, E, &I->router[l], S->qn, D, S->en, nbD, S->ean, n, 16, 0);
	for (int t = 0; t < n; t++) {
		const int32_t *z = S->z + (size_t)t * E;
		int *sel = S->sel + (size_t)t * K;
		int64_t lz[8], sum = 0, *gw = S->gw + (size_t)t * K;
		for (int s = 0; s < K; s++) {
			int b = -1;
			for (int e = 0; e < E; e++) {
				int used = 0;
				for (int r = 0; r < s; r++) used |= sel[r] == e;
				if (!used && (b < 0 || z[e] > z[b])) b = e;
			}
			sel[s] = b;
			lz[s] = rsh((int64_t)z[b] * C_L2E, 30);
		}
		for (int s = 0; s < K; s++) { gw[s] = exp2_neg(lz[s] - lz[0]); sum += gw[s]; }
		for (int s = 0; s < K; s++) gw[s] = divr(gw[s] << 30, sum);
	}
	for (int e = 0; e < E; e++) {
		int ne = 0;
		for (int t = 0; t < n; t++)
			for (int s = 0; s < K; s++)
				if (S->sel[(size_t)t * K + s] == e) {
					memcpy(S->gq + (size_t)ne * D, S->qn + (size_t)t * D, D);
					memcpy(S->geb + (size_t)ne * nbD, S->en + (size_t)t * nbD, nbD * sizeof(int));
					S->gea[ne] = S->ean[t];
					S->gt[ne++] = t * K + s;
				}
		if (!ne) continue;
		size_t ix = (size_t)l * E + e;
		imatmul(S->g, F, &I->xg[ix], S->gq, D, S->geb, nbD, S->gea, ne, 12, 0);
		imatmul(S->u, F, &I->xu[ix], S->gq, D, S->geb, nbD, S->gea, ne, 12, 0);
		swiglu(S->g, S->u, (size_t)ne * F);
		act8_rows(S->qa, W, S->eb, nbW, S->ea, S->g, F, ne, F, 12);
		imatmul(S->y, D, &I->xd[ix], S->qa, W, S->eb, nbW, S->ea, ne, 16, 0);
		for (int r = 0; r < ne; r++) {
			int t = S->gt[r] / K;
			int64_t gs = S->gw[S->gt[r]];
			int32_t *xt = x + (size_t)t * D;
			const int32_t *y = S->y + (size_t)r * D;
			for (int i = 0; i < D; i++) xt[i] = sat32((int64_t)xt[i] + rsh(gs * y[i], 30));
		}
	}
}

/* summed NLL of one window, Q16 log2 units */
static int64_t window(const il_model *I, const int *in, const int *tg, int T, scratch *S) {
	int D = I->D, Q = I->Q, KV = I->KV, H = I->H, hd = I->hd, half = hd / 2, W = I->w;
	int nbD = (D + BLK - 1) / BLK, nbW = (W + BLK - 1) / BLK;
	for (int t = 0; t < T; t++) memcpy(S->x + (size_t)t * D, I->emb + (size_t)in[t] * D, D * sizeof(int32_t));
	for (int l = 0; l < I->L; l++) {
		for (int t0 = 0; t0 < T; t0 += TC) {
			int n = T - t0 < TC ? T - t0 : TC;
			for (int t = 0; t < n; t++) inorm(S->xn + (size_t)t * D, S->x + (size_t)(t0 + t) * D, I->nw + (size_t)(2 * l) * D, D, I->eps);
			act8_rows(S->qa, W, S->eb, nbW, S->ea, S->xn, D, n, D, 12);
			imatmul(S->q + (size_t)t0 * Q, Q, &I->wq[l], S->qa, W, S->eb, nbW, S->ea, n, 12, 0);
			imatmul(S->k + (size_t)t0 * KV, KV, &I->wk[l], S->qa, W, S->eb, nbW, S->ea, n, 12, 0);
			imatmul(S->v + (size_t)t0 * KV, KV, &I->wv[l], S->qa, W, S->eb, nbW, S->ea, n, 12, 0);
		}
		for (int t = 0; t < T; t++) {
			const int32_t *cs = I->cs + (size_t)t * half, *sn = I->sn + (size_t)t * half;
			for (int h = 0; h < H + I->KH; h++) {
				int32_t *v = h < H ? S->q + (size_t)t * Q + h * hd : S->k + (size_t)t * KV + (h - H) * hd;
				for (int i = 0; i < half; i++) {
					int64_t a = v[i], b = v[i + half];
					v[i] = sat32(rsh(a * cs[i] - b * sn[i], 14));
					v[i + half] = sat32(rsh(a * sn[i] + b * cs[i], 14));
				}
			}
		}
		attention(I, T, S);
		for (int t0 = 0; t0 < T; t0 += TC) {
			int n = T - t0 < TC ? T - t0 : TC;
			int32_t *x = S->x + (size_t)t0 * D;
			act8_rows(S->qa, W, S->eb, nbW, S->ea, S->ao + (size_t)t0 * Q, Q, n, Q, 12);
			imatmul(x, D, &I->wo[l], S->qa, W, S->eb, nbW, S->ea, n, 16, 1);
			for (int t = 0; t < n; t++) inorm(S->xn + (size_t)t * D, x + (size_t)t * D, I->nw + (size_t)(2 * l + 1) * D, D, I->eps);
			act8_rows(S->qn, D, S->en, nbD, S->ean, S->xn, D, n, D, 12);
			ffn(I, l, x, n, S);
		}
	}
	int64_t nll = 0;
	for (int t0 = 0; t0 < T; t0 += TC) {
		int n = T - t0 < TC ? T - t0 : TC;
		for (int t = 0; t < n; t++) inorm(S->xn + (size_t)t * D, S->x + (size_t)(t0 + t) * D, I->nw + (size_t)(2 * I->L) * D, D, I->eps);
		act8_rows(S->qa, W, S->eb, nbW, S->ea, S->xn, D, n, D, 12);
		imatmul(S->lg, I->V, &I->head, S->qa, W, S->eb, nbW, S->ea, n, 16, 0);
		for (int t = 0; t < n; t++) {
			const int32_t *lg = S->lg + (size_t)t * I->V;
			int64_t mx = INT64_MIN, sum = 0;
			for (int v = 0; v < I->V; v++) { S->lz[v] = rsh((int64_t)lg[v] * C_L2E, 30); if (S->lz[v] > mx) mx = S->lz[v]; }
			for (int v = 0; v < I->V; v++) sum += exp2_neg(S->lz[v] - mx);
			nll += mx + log2_q30(sum) - S->lz[tg[t0 + t]];
		}
	}
	return nll;
}

typedef struct { const il_model *I; const int *ids; const int64_t *at; int n, T, t0, step, err; int64_t sum; } il_job;

static void il_work(void *arg) {
	il_job *J = arg;
	const il_model *I = J->I;
	size_t T = (size_t)J->T, D = (size_t)I->D, F = (size_t)I->F, W = (size_t)I->w, K = I->K ? (size_t)I->K : 1;
	size_t nbD = (D + BLK - 1) / BLK, nbW = (W + BLK - 1) / BLK, E = I->E ? (size_t)I->E : 1;
	scratch S = {
		malloc(T * D * 4), malloc(T * I->Q * 4), malloc(T * I->KV * 4), malloc(T * I->KV * 4), malloc(T * I->Q * 4),
		malloc(T * 8), malloc((size_t)I->hd * 8), malloc((size_t)I->hd * sizeof(i128)),
		malloc(TC * D * 4), malloc(TC * F * 4), malloc(TC * F * 4), malloc(TC * D * 4), malloc(TC * E * 4), malloc(TC * (size_t)I->V * 4),
		malloc(TC * W), malloc(TC * D), malloc(TC * D),
		malloc(TC * nbW * sizeof(int)), malloc(TC * nbD * sizeof(int)), malloc(TC * nbD * sizeof(int)),
		malloc(TC * sizeof(int)), malloc(TC * sizeof(int)), malloc(TC * sizeof(int)),
		malloc(TC * K * sizeof(int)), malloc(TC * K * sizeof(int)), malloc(TC * K * 8), malloc((size_t)I->V * 8),
	};
	void *all[] = { S.x, S.q, S.k, S.v, S.ao, S.e, S.acc, S.acc2, S.xn, S.g, S.u, S.y, S.z, S.lg, S.qa, S.qn, S.gq,
	                S.eb, S.en, S.geb, S.ea, S.ean, S.gea, S.sel, S.gt, S.gw, S.lz };
	int na = (int)(sizeof all / sizeof *all);
	for (int i = 0; i < na; i++) if (!all[i]) J->err = 1;
	for (int i = J->t0; !J->err && i < J->n; i += J->step) {
		const int *in = J->ids + J->at[i];
		for (int t = 0; t <= J->T; t++) if (in[t] < 0 || in[t] >= I->V) J->err = 1;
		if (!J->err) J->sum += window(I, in, in + 1, J->T, &S);
	}
	for (int i = 0; i < na; i++) free(all[i]);
}

int il_nll(const il_model *I, const int *ids, const int64_t *at, int n, int T, int64_t *nll) {
	if (!I || T < 1 || T > I->ctx || n < 0) return -1;
	il_job jb[64];
	int nt = hakm_nthreads();
	if (nt > 64) nt = 64;
	if (nt > n) nt = n;
	for (int t = 0; t < nt; t++) jb[t] = (il_job){ I, ids, at, n, T, t, nt, 0, 0 };
	hakm_parallel(il_work, jb, sizeof *jb, nt);
	int64_t s = 0;
	for (int t = 0; t < nt; t++) { if (jb[t].err) return -1; s += jb[t].sum; }
	*nll = s;
	return 0;
}

double il_nats(int64_t nll, int64_t tokens) { return (double)nll * (LN2 / 65536) / (double)tokens; }
