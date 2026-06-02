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

#if defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#define HAKM_NEON_DOTPROD 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define HAKM_AVX2 1
/* horizontal sum of 8 int32 lanes. */
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

void quantize_row_q8_32(const float *x, int n, int8_t *xq, float *xs) {
	for (int b = 0; b < n; b += 32) {
		float amax = 0.0f;
		for (int l = 0; l < 32; l++) {
			float a = fabsf(x[b + l]);
			if (a > amax) amax = a;
		}
		float scale = amax / 127.0f;
		float inv   = scale > 0.0f ? 1.0f / scale : 0.0f;
		xs[b >> 5] = scale;
		for (int l = 0; l < 32; l++) {
			int v = (int)lrintf(x[b + l] * inv);
			if (v >  127) v =  127;
			if (v < -127) v = -127;
			xq[b + l] = (int8_t)v;
		}
	}
}

/* One Q4_K row · int8 activation. For each 32-wide sub-block:
 *   dot += xs · ( d·s · Σ(q·xq)  −  dmin·m · Σ(xq) )
 * q is the 4-bit weight (0..15), s/m the 6-bit sub-block scale/min, d/dmin the
 * fp16 super-block scales. The two Σ are pure integer MACs (vectorizable). */
float q4k_vec_dot(const void *w_row, int n_in, const int8_t *xq, const float *xs) {
	const block_q4_K *b = (const block_q4_K *)w_row;
	int nb = n_in / QK_K;
	float acc = 0.0f;
	int sb = 0;                       /* global 32-block index → xs[] */

	for (int i = 0; i < nb; i++) {
		const float d    = f16_to_f32(b[i].d);
		const float dmin = f16_to_f32(b[i].dmin);
		const uint8_t *q  = b[i].qs;
		const uint8_t *sc = b[i].scales;
		const int8_t  *xb = xq + (size_t)i * QK_K;

		int is = 0;
		for (int g = 0; g < QK_K; g += 64) {
			uint8_t s, m;
			int iqxA, ixA, iqxB, ixB;

#ifdef HAKM_NEON_DOTPROD
			/* 64 weights (32 bytes) × 64 int8 activations via hardware sdot.
			   Low nibbles → sub-block A (xb[g..g+31]); high → B (xb[g+32..g+63]). */
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
			const int8x16_t ones = vdupq_n_s8(1);

			int32x4_t aq = vdupq_n_s32(0), ax = vdupq_n_s32(0);
			aq = vdotq_s32(aq, loA0, xA0); aq = vdotq_s32(aq, loA1, xA1);
			ax = vdotq_s32(ax, ones, xA0); ax = vdotq_s32(ax, ones, xA1);
			iqxA = vaddvq_s32(aq); ixA = vaddvq_s32(ax);

			int32x4_t bq = vdupq_n_s32(0), bx = vdupq_n_s32(0);
			bq = vdotq_s32(bq, hiB0, xB0); bq = vdotq_s32(bq, hiB1, xB1);
			bx = vdotq_s32(bx, ones, xB0); bx = vdotq_s32(bx, ones, xB1);
			iqxB = vaddvq_s32(bq); ixB = vaddvq_s32(bx);
#elif defined(HAKM_AVX2)
			/* 32 weight bytes → low nibbles = sub-block A (xb[g..g+31]),
			   high nibbles = sub-block B (xb[g+32..g+63]). maddubs does the
			   uint4×int8 MAC into int16 pairs (no saturation: max 15·127·2 =
			   3810 < 32767); madd_epi16 folds pairs to int32; then hsum.
			   Σxq uses maddubs against a vector of 1s. */
			{
				const __m256i qv     = _mm256_loadu_si256((const __m256i *)q);
				const __m256i lomask = _mm256_set1_epi8(0x0F);
				const __m256i lowA = _mm256_and_si256(qv, lomask);
				const __m256i hiB  = _mm256_and_si256(_mm256_srli_epi16(qv, 4), lomask);
				const __m256i xA   = _mm256_loadu_si256((const __m256i *)(xb + g));
				const __m256i xB   = _mm256_loadu_si256((const __m256i *)(xb + g + 32));
				const __m256i ones8  = _mm256_set1_epi8(1);
				const __m256i ones16 = _mm256_set1_epi16(1);

				iqxA = hsum256_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(lowA, xA), ones16));
				iqxB = hsum256_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(hiB,  xB), ones16));
				ixA  = hsum256_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(ones8, xA), ones16));
				ixB  = hsum256_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(ones8, xB), ones16));
			}
#else
			iqxA = ixA = iqxB = ixB = 0;
			for (int l = 0; l < 32; l++) {
				int xa = xb[g + l], xbv = xb[g + 32 + l];
				iqxA += (q[l] & 0xF) * xa; ixA += xa;
				iqxB += (q[l] >> 4)  * xbv; ixB += xbv;
			}
#endif
			get_scale_min_k4(is + 0, sc, &s, &m);
			acc += xs[sb++] * (d * s * (float)iqxA - dmin * m * (float)ixA);
			get_scale_min_k4(is + 1, sc, &s, &m);
			acc += xs[sb++] * (d * s * (float)iqxB - dmin * m * (float)ixB);

			q  += 32;
			is += 2;
		}
	}
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
	default: return -1;
	}
}
