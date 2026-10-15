/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/*
 * quant.c — k-quant dequant (Q4_K, Q6_K) + fp16 widen.
 *
 * Bit layouts follow ggml's reference dequantize_row_q4_K / _q6_K so blocks
 * copied verbatim from a GGUF decode identically. Scalar and correctness-first;
 * SIMD matmul paths live in kernels, not here.
 */
#include "quant.h"
#include "../include/mlf2.h"

#include <string.h>
#include <math.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>       /* every NEON path needs it, not only the dot-product one */
#endif
#if defined(__ARM_FEATURE_DOTPROD)
#define HAKM_NEON_DOTPROD 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define HAKM_AVX2 1
#endif

#if defined(HAKM_AVX2) && !defined(__FMA__)
/* horizontal sum of 8 int32 lanes (pre-FMA fallback path only — the FMA
 * q4k path keeps lane sums vectorized and hsums floats once per row). */
static inline int hsum256_epi32(__m256i v) {
	__m128i s = _mm_add_epi32(_mm256_castsi256_si128(v),
	                          _mm256_extracti128_si256(v, 1));
	s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
	s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
	return _mm_cvtsi128_si32(s);
}
#endif

/* IEEE half -> float, branch-light, no _Float16 dependency. */
float f16_to_f32(uint16_t h) {
	uint32_t sign = (uint32_t)(h & 0x8000) << 16;
	uint32_t exp  = (h >> 10) & 0x1F;
	uint32_t mant = h & 0x3FF;
	uint32_t bits;
	if (exp == 0) {
		if (mant == 0) {
			bits = sign;                       /* +/- zero */
		} else {
			/* subnormal: normalize */
			exp = 127 - 15 + 1;
			while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
			mant &= 0x3FF;
			bits = sign | (exp << 23) | (mant << 13);
		}
	} else if (exp == 0x1F) {
		bits = sign | 0x7F800000u | (mant << 13);  /* inf/nan */
	} else {
		bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
	}
	float f;
	memcpy(&f, &bits, sizeof(f));
	return f;
}

/* Unpack 6-bit scale and min for sub-block j (ggml get_scale_min_k4). */
static inline void get_scale_min_k4(int j, const uint8_t *q,
                                    uint8_t *d, uint8_t *m) {
	if (j < 4) {
		*d = q[j] & 63;
		*m = q[j + 4] & 63;
	} else {
		*d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
		*m = (q[j + 4] >>  4) | ((q[j]     >> 6) << 4);
	}
}

void dequant_q4_K(const void *blocks, float *out, int nelem) {
	const block_q4_K *x = (const block_q4_K *)blocks;
	int nb = nelem / QK_K;
	float *y = out;

	for (int i = 0; i < nb; i++) {
		const float d    = f16_to_f32(x[i].d);
		const float dmin = f16_to_f32(x[i].dmin);
		const uint8_t *q = x[i].qs;
		const uint8_t *sc = x[i].scales;

		int is = 0;
		for (int j = 0; j < QK_K; j += 64) {
			uint8_t s, m;
			get_scale_min_k4(is + 0, sc, &s, &m);
			const float d1 = d * s, m1 = dmin * m;
			get_scale_min_k4(is + 1, sc, &s, &m);
			const float d2 = d * s, m2 = dmin * m;
			for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
			for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4)  - m2;
			q  += 32;
			is += 2;
		}
	}
}

void dequant_q6_K(const void *blocks, float *out, int nelem) {
	const block_q6_K *x = (const block_q6_K *)blocks;
	int nb = nelem / QK_K;
	float *y = out;

	for (int i = 0; i < nb; i++) {
		const float d = f16_to_f32(x[i].d);
		const uint8_t *ql = x[i].ql;
		const uint8_t *qh = x[i].qh;
		const int8_t  *sc = x[i].scales;

		for (int n = 0; n < QK_K; n += 128) {
			for (int l = 0; l < 32; l++) {
				const int is = l / 16;
				const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
				const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
				const int8_t q3 = (int8_t)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
				const int8_t q4 = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
				y[l +  0] = d * sc[is + 0] * q1;
				y[l + 32] = d * sc[is + 2] * q2;
				y[l + 64] = d * sc[is + 4] * q3;
				y[l + 96] = d * sc[is + 6] * q4;
			}
			y  += 128;
			ql += 64;
			qh += 32;
			sc += 8;
		}
	}
}

/* ── int8-activation × int4-weight fast path ── */

/* Quantize + also emit axs[k] = scale · Σ(quants) for 32-block k. The Q4_K min
 * term is dmin·m·Σx, and Σx (hence scale·Σx) depends only on the activation — so
 * folding it here once lets q4k_vec_dot skip the per-row Σx reduction AND the
 * int→float convert in its hot loop, reusing axs across every weight row/thread. */
void quantize_row_q8_32(const float *x, int n, int8_t *xq, float *xs, float *axs) {
	for (int b = 0; b < n; b += 32) {
		float amax = 0.0f;
		for (int l = 0; l < 32; l++) {
			float a = fabsf(x[b + l]);
			if (a > amax) amax = a;
		}
		float scale = amax / 127.0f;
		float inv   = scale > 0.0f ? 1.0f / scale : 0.0f;
		xs[b >> 5] = scale;
		int s = 0;
		for (int l = 0; l < 32; l++) {
			int v = (int)lrintf(x[b + l] * inv);
			if (v >  127) v =  127;
			if (v < -127) v = -127;
			xq[b + l] = (int8_t)v;
			s += v;
		}
		axs[b >> 5] = scale * (float)s;
	}
}

/* One Q4_K row · int8 activation. For each 32-wide sub-block:
 *   dot += xs · ( d·s · Σ(q·xq)  −  dmin·m · Σ(xq) )
 * q is the 4-bit weight (0..15), s/m the 6-bit sub-block scale/min, d/dmin the
 * fp16 super-block scales. Σ(q·xq) is the per-row integer MAC; the min term is
 * activation-only, so scale·Σ(xq) arrives precomputed as `axs` (per 32-block,
 * reused across every weight row/thread) — the kernel just accumulates m·axs,
 * skipping the second per-row Σx reduction. */
float q4k_vec_dot(const void *w_row, int n_in, const int8_t *xq,
                  const float *xs, const float *axs) {
	const block_q4_K *b = (const block_q4_K *)w_row;
	int nb = n_in / QK_K;
	float acc = 0.0f;
	int sb = 0;                       /* global 32-block index → xs[]/axs[] */
#if defined(HAKM_AVX2) && defined(__FMA__)
	__m256 facc = _mm256_setzero_ps();   /* +term: Σ d·s·(q·x), 8 lanes */
	__m256 fmin = _mm256_setzero_ps();   /* -term: Σ dmin·m·(scale·Σx), 8 lanes */
#endif

	for (int i = 0; i < nb; i++) {
		const float d    = f16_to_f32(b[i].d);
		const float dmin = f16_to_f32(b[i].dmin);
		const uint8_t *q  = b[i].qs;
		const uint8_t *sc = b[i].scales;
		const int8_t  *xb = xq + (size_t)i * QK_K;
#if defined(HAKM_AVX2) && defined(__FMA__)
		float minf[8];                /* per-superblock m·axs, folded once below */
#endif

		int is = 0;
		for (int g = 0; g < QK_K; g += 64) {
			uint8_t s, m;
#if !(defined(HAKM_AVX2) && defined(__FMA__))
			int iqxA, iqxB;
#endif

#ifdef HAKM_NEON_DOTPROD
			/* 64 weights (32 bytes) × 64 int8 activations via hardware sdot.
			   Low nibbles → sub-block A (xb[g..g+31]); high → B (xb[g+32..g+63]).
			   Min term rides the precomputed axs in the scalar fold — no ones-dot. */
			const uint8x16_t q0 = vld1q_u8(q);
			const uint8x16_t q1 = vld1q_u8(q + 16);
			const uint8x16_t lo = vdupq_n_u8(0x0F);
			const int8x16_t loA0 = vreinterpretq_s8_u8(vandq_u8(q0, lo));
			const int8x16_t loA1 = vreinterpretq_s8_u8(vandq_u8(q1, lo));
			const int8x16_t hiB0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
			const int8x16_t hiB1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));
			const int8x16_t xA0 = vld1q_s8(xb + g);
			const int8x16_t xA1 = vld1q_s8(xb + g + 16);
			const int8x16_t xB0 = vld1q_s8(xb + g + 32);
			const int8x16_t xB1 = vld1q_s8(xb + g + 48);

			int32x4_t aq = vdupq_n_s32(0), bq = vdupq_n_s32(0);
			aq = vdotq_s32(aq, loA0, xA0); aq = vdotq_s32(aq, loA1, xA1);
			bq = vdotq_s32(bq, hiB0, xB0); bq = vdotq_s32(bq, hiB1, xB1);
			iqxA = vaddvq_s32(aq); iqxB = vaddvq_s32(bq);
#elif defined(HAKM_AVX2) && defined(__FMA__)
			/* FMA path: keep the 8-lane int32 partial sums as VECTORS and fold
			   the per-subblock scale in float lanes — ONE horizontal sum per ROW.
			   The +term (d·s·Σq·x) needs the per-row integer MAC; the -term
			   (dmin·m·scale·Σx) is activation-only (axs) so only m·axs is
			   computed per subblock (one scalar mul), then the whole superblock's
			   8 are folded into fmin with a single vector fnmadd — no per-row Σx
			   maddubs (frees half the port-0 MACs) and no serial scalar chain.
			   int32→float converts are exact (lane sums ≤ 4·15·127 = 7620). */
			{
				const __m256i qv     = _mm256_loadu_si256((const __m256i *)q);
				const __m256i lomask = _mm256_set1_epi8(0x0F);
				const __m256i lowA = _mm256_and_si256(qv, lomask);
				const __m256i hiB  = _mm256_and_si256(_mm256_srli_epi16(qv, 4), lomask);
				const __m256i xA   = _mm256_loadu_si256((const __m256i *)(xb + g));
				const __m256i xB   = _mm256_loadu_si256((const __m256i *)(xb + g + 32));
				const __m256i ones16 = _mm256_set1_epi16(1);

				const __m256i pA = _mm256_madd_epi16(_mm256_maddubs_epi16(lowA, xA), ones16);
				const __m256i pB = _mm256_madd_epi16(_mm256_maddubs_epi16(hiB,  xB), ones16);

				get_scale_min_k4(is + 0, sc, &s, &m);
				facc = _mm256_fmadd_ps(_mm256_set1_ps(xs[sb] * d * s),
				                       _mm256_cvtepi32_ps(pA), facc);
				minf[is + 0] = (float)m * axs[sb];
				sb++;
				get_scale_min_k4(is + 1, sc, &s, &m);
				facc = _mm256_fmadd_ps(_mm256_set1_ps(xs[sb] * d * s),
				                       _mm256_cvtepi32_ps(pB), facc);
				minf[is + 1] = (float)m * axs[sb];
				sb++;
			}
			q  += 32;
			is += 2;
			continue;
#elif defined(HAKM_AVX2)
			/* 32 weight bytes → low nibbles = sub-block A (xb[g..g+31]),
			   high nibbles = sub-block B (xb[g+32..g+63]). maddubs does the
			   uint4×int8 MAC into int16 pairs (no saturation: max 15·127·2 =
			   3810 < 32767); madd_epi16 folds pairs to int32; then hsum. The min
			   term uses precomputed axs (= scale·Σx), see the scalar fold. */
			{
				const __m256i qv     = _mm256_loadu_si256((const __m256i *)q);
				const __m256i lomask = _mm256_set1_epi8(0x0F);
				const __m256i lowA = _mm256_and_si256(qv, lomask);
				const __m256i hiB  = _mm256_and_si256(_mm256_srli_epi16(qv, 4), lomask);
				const __m256i xA   = _mm256_loadu_si256((const __m256i *)(xb + g));
				const __m256i xB   = _mm256_loadu_si256((const __m256i *)(xb + g + 32));
				const __m256i ones16 = _mm256_set1_epi16(1);

				iqxA = hsum256_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(lowA, xA), ones16));
				iqxB = hsum256_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(hiB,  xB), ones16));
			}
#else
			iqxA = iqxB = 0;
			for (int l = 0; l < 32; l++) {
				int xa = xb[g + l], xbv = xb[g + 32 + l];
				iqxA += (q[l] & 0xF) * xa;
				iqxB += (q[l] >> 4)  * xbv;
			}
#endif
#if !(defined(HAKM_AVX2) && defined(__FMA__))
			/* +term needs the per-row integer MAC; -term folds precomputed
			   axs = scale·Σx (so xs·(… − dmin·m·Σx) = … − dmin·m·axs). */
			get_scale_min_k4(is + 0, sc, &s, &m);
			acc += xs[sb] * d * s * (float)iqxA - dmin * m * axs[sb]; sb++;
			get_scale_min_k4(is + 1, sc, &s, &m);
			acc += xs[sb] * d * s * (float)iqxB - dmin * m * axs[sb]; sb++;

			q  += 32;
			is += 2;
#endif
		}
#if defined(HAKM_AVX2) && defined(__FMA__)
		/* fold this superblock's 8 min terms (m·axs) into fmin with one vector
		   FMA — dmin is constant over the superblock. */
		fmin = _mm256_fmadd_ps(_mm256_set1_ps(dmin), _mm256_loadu_ps(minf), fmin);
#endif
	}
#if defined(HAKM_AVX2) && defined(__FMA__)
	{	/* one horizontal sum per row: (Σ +term) − (Σ −term) */
		__m256 v = _mm256_sub_ps(facc, fmin);
		__m128 lo = _mm_add_ps(_mm256_castps256_ps128(v),
		                       _mm256_extractf128_ps(v, 1));
		lo = _mm_hadd_ps(lo, lo);
		lo = _mm_hadd_ps(lo, lo);
		acc += _mm_cvtss_f32(lo);
	}
#endif
	return acc;
}

/* One Q6_K row · int8 activation. Element layout mirrors dequant_q6_K: each
 * 256-block is two 128-halves; within a half, the four 32-element groups at
 * offsets {0,32,64,96} draw their 4-bit nibbles from {ql[0..31] low,
 * ql[32..63] low, ql[0..31] high, ql[32..63] high} and 2 extra bits per value
 * from qh[0..31] (bit pairs 0/2/4/6), biased -32. Weight scales are int8 per
 * 16 elements; activation scales (xs) per 32. All MACs stay integer; ONE
 * float fold per 32-group (and one hsum per row on the AVX2 path). */
float q6k_vec_dot(const void *w_row, int n_in, const int8_t *xq, const float *xs) {
	const block_q6_K *b = (const block_q6_K *)w_row;
	int nb = n_in / QK_K;
	float acc = 0.0f;
	int sb = 0;                       /* global 32-block index -> xs[] */
#if defined(HAKM_AVX2) && defined(__FMA__)
	__m256 facc = _mm256_setzero_ps();
#endif

	for (int i = 0; i < nb; i++) {
		const float d = f16_to_f32(b[i].d);
		const uint8_t *ql = b[i].ql;
		const uint8_t *qh = b[i].qh;
		const int8_t  *sc = b[i].scales;
		const int8_t  *xb = xq + (size_t)i * QK_K;

		for (int half = 0; half < 2; half++) {
#if defined(HAKM_AVX2) && defined(__FMA__)
			/* q built UNSIGNED (0..63) so maddubs applies (u8*s8 pairs,
			   |2*63*127| < 32767); the -32 bias becomes 32*sum(x) subtracted on
			   the int32 lanes. madd folds to 8 int32 lanes of 4 elements each:
			   lanes 0-3 = the group's first 16 elements, so the per-16 weight
			   scales apply with one mullo. */
			const __m256i qlA  = _mm256_loadu_si256((const __m256i *)ql);        /* ql[0..31] */
			const __m256i qlB  = _mm256_loadu_si256((const __m256i *)(ql + 32)); /* ql[32..63] */
			const __m256i qh32 = _mm256_loadu_si256((const __m256i *)qh);        /* qh[0..31] */
			const __m256i lo4   = _mm256_set1_epi8(0x0F);
			const __m256i hi2m  = _mm256_set1_epi8(3);
			const __m256i m0x30 = _mm256_set1_epi8(0x30);
			const __m256i ones8  = _mm256_set1_epi8(1);
			const __m256i ones16 = _mm256_set1_epi16(1);

			for (int g = 0; g < 4; g++) {
				const __m256i nib = (g == 0) ? _mm256_and_si256(qlA, lo4)
				                  : (g == 1) ? _mm256_and_si256(qlB, lo4)
				                  : (g == 2) ? _mm256_and_si256(_mm256_srli_epi16(qlA, 4), lo4)
				                             : _mm256_and_si256(_mm256_srli_epi16(qlB, 4), lo4);
				const __m256i hb = _mm256_and_si256(
					_mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh32, 2 * g), hi2m), 4),
					m0x30);
				const __m256i qu = _mm256_or_si256(nib, hb);
				const __m256i xv = _mm256_loadu_si256((const __m256i *)(xb + g * 32));

				const __m256i ps  = _mm256_madd_epi16(_mm256_maddubs_epi16(qu, xv), ones16);
				const __m256i sx  = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, xv), ones16);
				const __m256i sgn = _mm256_sub_epi32(ps, _mm256_slli_epi32(sx, 5));   /* -32 bias */
				const __m256i sv  = _mm256_set_epi32(sc[2*g+1], sc[2*g+1], sc[2*g+1], sc[2*g+1],
				                                     sc[2*g],   sc[2*g],   sc[2*g],   sc[2*g]);
				const __m256i scaled = _mm256_mullo_epi32(sgn, sv);
				facc = _mm256_fmadd_ps(_mm256_set1_ps(xs[sb + g] * d),
				                       _mm256_cvtepi32_ps(scaled), facc);
			}
#else
			/* scalar reference: 4 interleaved groups, per-16 int sums */
			int32_t s0a=0,s0b=0,s1a=0,s1b=0,s2a=0,s2b=0,s3a=0,s3b=0;
			for (int l = 0; l < 32; l++) {
				const int q1 = ((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
				const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
				const int q3 = ((ql[l]      >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
				const int q4 = ((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
				if (l < 16) {
					s0a += q1 * xb[l];      s1a += q2 * xb[l + 32];
					s2a += q3 * xb[l + 64]; s3a += q4 * xb[l + 96];
				} else {
					s0b += q1 * xb[l];      s1b += q2 * xb[l + 32];
					s2b += q3 * xb[l + 64]; s3b += q4 * xb[l + 96];
				}
			}
			acc += xs[sb + 0] * d * (float)(sc[0] * s0a + sc[1] * s0b);
			acc += xs[sb + 1] * d * (float)(sc[2] * s1a + sc[3] * s1b);
			acc += xs[sb + 2] * d * (float)(sc[4] * s2a + sc[5] * s2b);
			acc += xs[sb + 3] * d * (float)(sc[6] * s3a + sc[7] * s3b);
#endif
			sb += 4;
			xb += 128; ql += 64; qh += 32; sc += 8;
		}
	}
#if defined(HAKM_AVX2) && defined(__FMA__)
	{
		__m128 lo = _mm_add_ps(_mm256_castps256_ps128(facc),
		                       _mm256_extractf128_ps(facc, 1));
		lo = _mm_hadd_ps(lo, lo);
		lo = _mm_hadd_ps(lo, lo);
		acc += _mm_cvtss_f32(lo);
	}
#endif
	return acc;
}

int dequant_tensor(int type, const void *src, float *out, int nelem) {
	switch (type) {
	case MLF2_TYPE_F32:
		memcpy(out, src, (size_t)nelem * sizeof(float));
		return 0;
	case MLF2_TYPE_F16: {
		const uint16_t *h = (const uint16_t *)src;
		for (int i = 0; i < nelem; i++) out[i] = f16_to_f32(h[i]);
		return 0;
	}
	case MLF2_TYPE_Q4_K: dequant_q4_K(src, out, nelem); return 0;
	case MLF2_TYPE_Q6_K: dequant_q6_K(src, out, nelem); return 0;
	case MLF2_TYPE_Q8_0: dequant_q8_0(src, out, nelem); return 0;
	default: return -1;
	}
}

/* ── Q8_0 ──
 * 32 int8 + one f16 scale per block: what ggml gives any tensor whose row is
 * not a multiple of 256 (so every sub-1B model's attention), and what a
 * "Q8_0" GGUF is made of throughout. */
void dequant_q8_0(const void *blocks, float *out, int nelem) {
	const block_q8_0 *b = (const block_q8_0 *)blocks;
	for (int i = 0; i < nelem / 32; i++) {
		float d = f16_to_f32(b[i].d);
		for (int l = 0; l < 32; l++) out[i * 32 + l] = d * (float)b[i].qs[l];
	}
}

/* Row · activation with both sides int8: acc += d_w · s_x · Σ q·x per block. */
float q8_0_vec_dot(const void *w_row, int n_in, const int8_t *xq, const float *xs) {
	const block_q8_0 *b = (const block_q8_0 *)w_row;
	int nb = n_in / 32;
#if defined(HAKM_AVX2) && defined(__FMA__)
	/* maddubs is u8·s8: feed |x| as the unsigned side and w with x's sign
	   moved onto it — the products are unchanged and stay in range. */
	__m256 facc = _mm256_setzero_ps();
	const __m256i ones16 = _mm256_set1_epi16(1);
	for (int i = 0; i < nb; i++) {
		__m256i w = _mm256_loadu_si256((const __m256i *)b[i].qs);
		__m256i x = _mm256_loadu_si256((const __m256i *)(xq + (size_t)i * 32));
		__m256i ax = _mm256_sign_epi8(x, x);
		__m256i sw = _mm256_sign_epi8(w, x);
		__m256i p  = _mm256_madd_epi16(_mm256_maddubs_epi16(ax, sw), ones16);
		float s = f16_to_f32(b[i].d) * xs[i];
		facc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(s), facc);
	}
	__m128 lo = _mm256_castps256_ps128(facc), hi = _mm256_extractf128_ps(facc, 1);
	lo = _mm_add_ps(lo, hi);
	lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
	lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
	return _mm_cvtss_f32(lo);
#elif defined(__ARM_NEON)
	float acc = 0.0f;
	for (int i = 0; i < nb; i++) {
		int8x16_t w0 = vld1q_s8(b[i].qs), w1 = vld1q_s8(b[i].qs + 16);
		int8x16_t x0 = vld1q_s8(xq + (size_t)i * 32), x1 = vld1q_s8(xq + (size_t)i * 32 + 16);
		int32x4_t s = vdupq_n_s32(0);
#if defined(HAKM_NEON_DOTPROD)
		s = vdotq_s32(s, w0, x0);
		s = vdotq_s32(s, w1, x1);
#else
		int16x8_t p = vmull_s8(vget_low_s8(w0), vget_low_s8(x0));
		p = vmlal_s8(p, vget_high_s8(w0), vget_high_s8(x0));
		p = vmlal_s8(p, vget_low_s8(w1),  vget_low_s8(x1));
		p = vmlal_s8(p, vget_high_s8(w1), vget_high_s8(x1));
		s = vpadalq_s16(s, p);
#endif
		acc += f16_to_f32(b[i].d) * xs[i] * (float)vaddvq_s32(s);
	}
	return acc;
#else
	float acc = 0.0f;
	for (int i = 0; i < nb; i++) {
		const int8_t *xb = xq + (size_t)i * 32;
		int s = 0;
		for (int l = 0; l < 32; l++) s += (int)b[i].qs[l] * (int)xb[l];
		acc += f16_to_f32(b[i].d) * xs[i] * (float)s;
	}
	return acc;
#endif
}
