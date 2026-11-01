/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* quantize.c — float weights into the types the engine runs. The inverse
   of the dequant in quant.c, which is the definition of each layout: a block is
   right when dequant_tensor gives back the values it was made from, near enough.
   Offline only (hakm quantize); nothing here is on the inference path. */
#include "quant.h"
#include "loader.h"
#include "../include/mlf2.h"

#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* float -> IEEE half, round to nearest even. Overflow saturates to inf. */
uint16_t f32_to_f16(float f) {
	uint32_t x; memcpy(&x, &f, 4);
	uint32_t sign = (x >> 16) & 0x8000, ax = x & 0x7FFFFFFF;
	if (ax >= 0x7F800000) return (uint16_t)(sign | 0x7C00 | (ax > 0x7F800000 ? 0x200 : 0));
	if (ax >= 0x477FF000) return (uint16_t)(sign | 0x7C00);          /* rounds past 65504 */
	if (ax < 0x38800000) {                                             /* subnormal half */
		if (ax < 0x33000000) return (uint16_t)sign;
		int e = (int)(ax >> 23), shift = 126 - e;
		uint32_t m = (ax & 0x7FFFFF) | 0x800000;
		uint32_t h = m >> shift, rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
		if (rem > half || (rem == half && (h & 1))) h++;
		return (uint16_t)(sign | h);
	}
	uint32_t h = ((ax - 0x38000000) >> 13), rem = ax & 0x1FFF;
	if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
	return (uint16_t)(sign | h);
}

static inline int nearest(float v) { return (int)lrintf(v); }

/* ── Q8_0: d = absmax / 127, q = round(x / d) ── */
static void quant_q8_0(const float *x, block_q8_0 *b, int n) {
	for (int i = 0; i < n / 32; i++, x += 32) {
		float amax = 0;
		for (int l = 0; l < 32; l++) if (fabsf(x[l]) > amax) amax = fabsf(x[l]);
		float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
		b[i].d = f32_to_f16(d);
		for (int l = 0; l < 32; l++) b[i].qs[l] = (int8_t)roundf(x[l] * id);
	}
}

/* Fit x ≈ scale * q - mn with q in 0..nmax, by weighted least squares over a few
   candidate grids around the plain min/max one. mn >= 0 is what gets subtracted
   (the block format stores a non-negative min). Returns scale, sets *mn. */
static float fit_asym(const float *x, const float *w, int n, int nmax, uint8_t *q, float *mn) {
	float lo = x[0], hi = x[0];
	for (int i = 1; i < n; i++) { if (x[i] < lo) lo = x[i]; if (x[i] > hi) hi = x[i]; }
	if (lo > 0) lo = 0;
	if (hi == lo) { memset(q, 0, n); *mn = -lo; return 0; }
	/* start from the plain min/max grid, so q is set even if no candidate below
	   fits. FLT_MAX, not INFINITY: hakm builds with -ffast-math, which may fold
	   any test against infinity away and leave q uninitialized */
	float best_s = (hi - lo) / nmax, best_m = lo, best_err = FLT_MAX;
	for (int i = 0; i < n; i++) { int l = nearest((x[i] - lo) / best_s); q[i] = (uint8_t)(l < 0 ? 0 : l > nmax ? nmax : l); }
	uint8_t tq[32];
	for (int step = -10; step <= 10; step++) {
		float is = (nmax + 0.1f * step) / (hi - lo);
		float sw = 0, sx = 0, sl = 0, sll = 0, sxl = 0;
		for (int i = 0; i < n; i++) {
			int l = nearest(is * (x[i] - lo));
			tq[i] = (uint8_t)(l < 0 ? 0 : l > nmax ? nmax : l);
			sw += w[i]; sx += w[i] * x[i]; sl += w[i] * tq[i];
			sll += w[i] * tq[i] * tq[i]; sxl += w[i] * x[i] * tq[i];
		}
		float det = sw * sll - sl * sl;
		if (det <= 0) continue;
		float s = (sw * sxl - sx * sl) / det, m = (sll * sx - sl * sxl) / det;
		if (m > 0) { m = 0; s = sll ? sxl / sll : 0; }          /* min stays <= 0 */
		float err = 0;
		for (int i = 0; i < n; i++) { float e = s * tq[i] + m - x[i]; err += w[i] * e * e; }
		if (err < best_err) { best_err = err; best_s = s; best_m = m; memcpy(q, tq, n); }
	}
	*mn = -best_m;
	return best_s;
}

/* Fit x ≈ scale * q with q in -nlo..nlo-1 (signed, asymmetric range), trying
   grids that map the largest-magnitude value to either end. Returns scale. */
static float fit_sym(const float *x, const float *w, int n, int nlo) {
	float amax = 0, vmax = 0;
	for (int i = 0; i < n; i++) if (fabsf(x[i]) > amax) { amax = fabsf(x[i]); vmax = x[i]; }
	if (!amax) return 0;
	float best_s = -vmax / nlo, best = -1;
	for (int step = -9; step <= 9; step++) {
		float is = -(nlo + 0.1f * step) / vmax, sxl = 0, sll = 0;
		for (int i = 0; i < n; i++) {
			int l = nearest(is * x[i]);
			l = l < -nlo ? -nlo : l > nlo - 1 ? nlo - 1 : l;
			sxl += w[i] * x[i] * l; sll += w[i] * l * l;
		}
		if (sll > 0 && sxl * sxl / sll > best) { best = sxl * sxl / sll; best_s = sxl / sll; }
	}
	return best_s;
}

/* ── Q4_K: 8 sub-blocks of 32, each its own scale and min, both stored as 6-bit
   multiples of two per-block f16 values ── */
static void quant_q4_K(const float *x, block_q4_K *b, int n) {
	for (int i = 0; i < n / QK_K; i++, x += QK_K) {
		float sc[8], mn[8], w[32], ss = 0;
		uint8_t q[QK_K];
		for (int l = 0; l < QK_K; l++) ss += x[l] * x[l];
		float av = sqrtf(ss / QK_K);
		for (int j = 0; j < 8; j++) {
			for (int l = 0; l < 32; l++) w[l] = av + fabsf(x[32 * j + l]);
			sc[j] = fit_asym(x + 32 * j, w, 32, 15, q + 32 * j, &mn[j]);
		}
		float smax = 0, mmax = 0;
		for (int j = 0; j < 8; j++) { if (sc[j] > smax) smax = sc[j]; if (mn[j] > mmax) mmax = mn[j]; }
		uint16_t dh = f32_to_f16(smax / 63.0f), mh = f32_to_f16(mmax / 63.0f);
		float d = f16_to_f32(dh), dm = f16_to_f32(mh);
		uint8_t ls[8], lm[8];
		for (int j = 0; j < 8; j++) {
			int a = d ? nearest(sc[j] / d) : 0, c = dm ? nearest(mn[j] / dm) : 0;
			ls[j] = (uint8_t)(a < 0 ? 0 : a > 63 ? 63 : a);
			lm[j] = (uint8_t)(c < 0 ? 0 : c > 63 ? 63 : c);
		}
		b[i].d = dh; b[i].dmin = mh;
		for (int j = 0; j < 4; j++) {
			b[i].scales[j]     = (uint8_t)(ls[j] | ((ls[j + 4] >> 4) << 6));
			b[i].scales[j + 4] = (uint8_t)(lm[j] | ((lm[j + 4] >> 4) << 6));
			b[i].scales[j + 8] = (uint8_t)((ls[j + 4] & 0xF) | ((lm[j + 4] & 0xF) << 4));
		}
		/* requantize against the scales as stored, not as fitted */
		for (int j = 0; j < 8; j++) {
			float s = d * ls[j], m = dm * lm[j];
			for (int l = 0; l < 32; l++) {
				int v = s ? nearest((x[32 * j + l] + m) / s) : 0;
				q[32 * j + l] = (uint8_t)(v < 0 ? 0 : v > 15 ? 15 : v);
			}
		}
		for (int c = 0; c < 4; c++)
			for (int l = 0; l < 32; l++)
				b[i].qs[32 * c + l] = (uint8_t)(q[64 * c + l] | (q[64 * c + 32 + l] << 4));
	}
}

/* ── Q6_K: 16 sub-blocks of 16, signed 6-bit values, int8 scales over one f16 ── */
static void quant_q6_K(const float *x, block_q6_K *b, int n) {
	for (int i = 0; i < n / QK_K; i++, x += QK_K) {
		float sc[16], w[16], amax = 0, vmax = 0;
		for (int j = 0; j < 16; j++) {
			for (int l = 0; l < 16; l++) w[l] = x[16 * j + l] * x[16 * j + l];
			sc[j] = fit_sym(x + 16 * j, w, 16, 32);
			if (fabsf(sc[j]) > amax) { amax = fabsf(sc[j]); vmax = sc[j]; }
		}
		memset(&b[i], 0, sizeof b[i]);
		if (!amax) continue;
		float is = -128.0f / vmax;
		b[i].d = f32_to_f16(1.0f / is);
		float d = f16_to_f32(b[i].d);
		uint8_t L[QK_K];
		for (int j = 0; j < 16; j++) {
			int s = nearest(is * sc[j]);
			b[i].scales[j] = (int8_t)(s > 127 ? 127 : s < -128 ? -128 : s);
			float ds = d * b[i].scales[j];
			for (int l = 0; l < 16; l++) {
				int v = ds ? nearest(x[16 * j + l] / ds) : 0;
				L[16 * j + l] = (uint8_t)((v < -32 ? -32 : v > 31 ? 31 : v) + 32);
			}
		}
		for (int h = 0; h < 2; h++) {
			const uint8_t *Lh = L + 128 * h;
			uint8_t *ql = b[i].ql + 64 * h, *qh = b[i].qh + 32 * h;
			for (int l = 0; l < 32; l++) {
				ql[l]      = (uint8_t)((Lh[l] & 0xF)      | ((Lh[l + 64] & 0xF) << 4));
				ql[l + 32] = (uint8_t)((Lh[l + 32] & 0xF) | ((Lh[l + 96] & 0xF) << 4));
				qh[l] = (uint8_t)((Lh[l] >> 4) | ((Lh[l + 32] >> 4) << 2) |
				                  ((Lh[l + 64] >> 4) << 4) | ((Lh[l + 96] >> 4) << 6));
			}
		}
	}
}

size_t quantize_tensor(int type, const float *src, void *dst, int nelem) {
	size_t nb = quant_row_bytes(type, nelem);
	if (!nb) return 0;
	switch (type) {
	case MLF2_TYPE_F32:  memcpy(dst, src, nb); break;
	case MLF2_TYPE_F16:  for (int i = 0; i < nelem; i++) ((uint16_t *)dst)[i] = f32_to_f16(src[i]); break;
	case MLF2_TYPE_BF16:
		for (int i = 0; i < nelem; i++) {
			uint32_t u; memcpy(&u, &src[i], 4);
			((uint16_t *)dst)[i] = (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
		}
		break;
	case MLF2_TYPE_Q8_0: quant_q8_0(src, dst, nelem); break;
	case MLF2_TYPE_Q4_K: quant_q4_K(src, dst, nelem); break;
	case MLF2_TYPE_Q6_K: quant_q6_K(src, dst, nelem); break;
	}
	return nb;
}

/* ── hakm quantize: rewrite a model file at another type ── */
/* The per-tensor policy for "q4_k" is the usual q4_k_m mix: 1-D tensors stay
   f32, the output projection and the attention-v / ffn-down of some layers
   (first and last eighth, and every third between) get q6_k, the rest q4_k.
   A row that does not fill whole blocks falls back to q8_0, then f16. */
static int more_bits(int l, int n) { return l < n / 8 || l >= 7 * n / 8 || (l - n / 8) % 3 == 2; }

static int pick_type(const mlf2_header *h, const mlf2_tensor *t, int want) {
	if (t->n_dims < 2) return MLF2_TYPE_F32;
	int type = want, l = -1;
	if (want == MLF2_TYPE_Q4_K) {
		int tied = (h->flags & MLF2_FLAG_TIED_EMBED) != 0;
		if (!strcmp(t->name, "output.weight") || (tied && !strcmp(t->name, "token_embd.weight")))
			type = MLF2_TYPE_Q6_K;
		else if (sscanf(t->name, "blk.%d.", &l) == 1 &&
		         (strstr(t->name, "attn_v.") || strstr(t->name, "ffn_down.")) && more_bits(l, (int)h->n_layers))
			type = MLF2_TYPE_Q6_K;
	}
	int cols = (int)t->dims[0];
	if (!quant_row_bytes(type, cols)) type = quant_row_bytes(MLF2_TYPE_Q8_0, cols) ? MLF2_TYPE_Q8_0 : MLF2_TYPE_F16;
	if (want == MLF2_TYPE_F16 || want == MLF2_TYPE_BF16 || want == MLF2_TYPE_F32) type = want;
	return type;
}

typedef struct { const hakm_model *m; const mlf2_tensor *t; int type, cols, r0, r1, err; unsigned char *out; } qjob;

static void *qwork(void *arg) {
	qjob *j = arg;
	size_t ib = quant_row_bytes((int)j->t->type, j->cols), ob = quant_row_bytes(j->type, j->cols);
	const unsigned char *src = (const unsigned char *)mlf2_data(j->m, j->t);
	float *row = malloc(sizeof(float) * j->cols);
	if (!row || !ib) { j->err = 1; free(row); return NULL; }
	for (int r = j->r0; r < j->r1; r++) {
		if (dequant_tensor((int)j->t->type, src + (size_t)r * ib, row, j->cols)) { j->err = 1; break; }
		quantize_tensor(j->type, row, j->out + (size_t)r * ob, j->cols);
	}
	free(row);
	return NULL;
}

int hakm_quantize(const char *in, const char *out, const char *type_name) {
	int want = !strcmp(type_name, "q4_k") || !strcmp(type_name, "q4_k_m") ? MLF2_TYPE_Q4_K :
	           !strcmp(type_name, "q6_k") ? MLF2_TYPE_Q6_K : !strcmp(type_name, "q8_0") ? MLF2_TYPE_Q8_0 :
	           !strcmp(type_name, "f16") ? MLF2_TYPE_F16 : !strcmp(type_name, "bf16") ? MLF2_TYPE_BF16 :
	           !strcmp(type_name, "f32") ? MLF2_TYPE_F32 : -1;
	if (want < 0) { fprintf(stderr, "hakm quantize: unknown type '%s' (q4_k, q6_k, q8_0, f16, bf16, f32)\n", type_name); return -1; }
	hakm_model m;
	if (mlf2_open(in, &m)) return -1;
	const mlf2_header *h = &m.h;
	uint32_t nt = h->n_tensors;
	mlf2_tensor *dir = calloc(nt, sizeof *dir);
	int nth = (int)sysconf(_SC_NPROCESSORS_ONLN);
	if (nth < 1) nth = 1;
	if (nth > 64) nth = 64;
	char tmp[4096];
	snprintf(tmp, sizeof tmp, "%s.part", out);
	FILE *f = dir ? fopen(tmp, "wb") : NULL;
	if (!f) { fprintf(stderr, "hakm quantize: cannot write %s\n", tmp); free(dir); mlf2_close(&m); return -1; }

	mlf2_header oh = *h;
	size_t tok_len = (size_t)(h->data_off - h->tokenizer_off);
	oh.tensors_off = sizeof oh;
	oh.tokenizer_off = oh.tensors_off + (uint64_t)nt * sizeof(mlf2_tensor);
	oh.data_off = (oh.tokenizer_off + tok_len + MLF2_ALIGN - 1) / MLF2_ALIGN * MLF2_ALIGN;
	uint64_t off = 0, in_bytes = 0;
	for (uint32_t i = 0; i < nt; i++) {
		dir[i] = m.tensors[i];
		int ty = pick_type(h, &m.tensors[i], want);
		uint64_t rows = 1;
		for (uint32_t d = 1; d < dir[i].n_dims; d++) rows *= dir[i].dims[d];
		size_t rb = quant_row_bytes(ty, (int)dir[i].dims[0]);
		if (!quant_row_bytes((int)m.tensors[i].type, (int)dir[i].dims[0]) || !rb) {
			fprintf(stderr, "hakm quantize: %s: cannot convert %s\n", dir[i].name, mlf2_type_name((int)m.tensors[i].type));
			fclose(f); remove(tmp); free(dir); mlf2_close(&m); return -1;
		}
		in_bytes += m.tensors[i].nbytes;
		dir[i].type = (uint32_t)ty;
		dir[i].nbytes = rows * rb;
		dir[i].data_off = off;
		off += (dir[i].nbytes + MLF2_ALIGN - 1) / MLF2_ALIGN * MLF2_ALIGN;
	}
	int ok = fwrite(&oh, sizeof oh, 1, f) == 1 && fwrite(dir, sizeof *dir, nt, f) == nt &&
	         fwrite(m.tok, 1, tok_len, f) == tok_len;
	for (long p = ftell(f); ok && p < (long)oh.data_off; p++) ok = fputc(0, f) != EOF;

	int counts[32] = { 0 };
	for (uint32_t i = 0; ok && i < nt; i++) {
		const mlf2_tensor *t = &m.tensors[i];
		int cols = (int)t->dims[0], rows = (int)(dir[i].nbytes / quant_row_bytes((int)dir[i].type, cols));
		unsigned char *buf = malloc(dir[i].nbytes + MLF2_ALIGN);
		if (!buf) { ok = 0; break; }
		if (dir[i].type == t->type) memcpy(buf, mlf2_data(&m, t), dir[i].nbytes);
		else {
			int n = rows < nth ? rows : nth;
			qjob jobs[64]; pthread_t th[64];
			for (int k = 0; k < n; k++) {
				jobs[k] = (qjob){ &m, t, (int)dir[i].type, cols, rows * k / n, rows * (k + 1) / n, 0, buf };
				pthread_create(&th[k], NULL, qwork, &jobs[k]);
			}
			for (int k = 0; k < n; k++) { pthread_join(th[k], NULL); if (jobs[k].err) ok = 0; }
		}
		size_t pad = (MLF2_ALIGN - dir[i].nbytes % MLF2_ALIGN) % MLF2_ALIGN;
		memset(buf + dir[i].nbytes, 0, pad);
		if (ok && fwrite(buf, 1, dir[i].nbytes + pad, f) != dir[i].nbytes + pad) ok = 0;
		free(buf);
		counts[dir[i].type & 31]++;
		if (isatty(2)) fprintf(stderr, "\r  %u/%u tensors", i + 1, nt);
	}
	if (isatty(2)) fprintf(stderr, "\r\x1b[K");
	ok &= fclose(f) == 0;
	if (ok && rename(tmp, out) != 0) ok = 0;
	if (!ok) { remove(tmp); fprintf(stderr, "hakm quantize: failed writing %s\n", out); }
	else {
		double u = in_bytes >= 1e9 ? 1e9 : 1e6;      /* our own models start small */
		const char *un = u == 1e9 ? "GB" : "MB";
		fprintf(stderr, "%s: %.2f %s -> %.2f %s  (", out, in_bytes / u, un, off / u, un);
		const int order[] = { MLF2_TYPE_Q4_K, MLF2_TYPE_Q6_K, MLF2_TYPE_Q8_0, MLF2_TYPE_F16, MLF2_TYPE_BF16, MLF2_TYPE_F32 };
		const char *sep = "";
		for (int k = 0; k < 6; k++) if (counts[order[k]]) { fprintf(stderr, "%s%d %s", sep, counts[order[k]], mlf2_type_name(order[k])); sep = ", "; }
		fprintf(stderr, ")\n");
	}
	free(dir); mlf2_close(&m);
	return ok ? 0 : -1;
}
