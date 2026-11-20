/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* gpu.c — OpenCL matmuls (see gpu.h). Only the handful of OpenCL 1.2 calls used
 * here are declared, so no OpenCL headers are needed to build it. */
#include "gpu.h"
#include "quant.h"
#include "../include/mlf2.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef int64_t cl_long;
typedef uint64_t cl_bitfield;
typedef struct cl_opaque *cl_platform_id, *cl_device_id, *cl_context, *cl_command_queue,
	*cl_mem, *cl_program, *cl_kernel, *cl_event;

#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE 0x1010
#define CL_DEVICE_NAME 0x102B
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_COPY_HOST_PTR (1 << 5)
#define CL_TRUE 1
#define CL_PROGRAM_BUILD_LOG 0x1183

static struct {
	cl_int (*GetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
	cl_int (*GetDeviceIDs)(cl_platform_id, cl_bitfield, cl_uint, cl_device_id *, cl_uint *);
	cl_int (*GetDeviceInfo)(cl_device_id, cl_uint, size_t, void *, size_t *);
	cl_context (*CreateContext)(const intptr_t *, cl_uint, const cl_device_id *, void *, void *, cl_int *);
	cl_command_queue (*CreateCommandQueue)(cl_context, cl_device_id, cl_bitfield, cl_int *);
	cl_program (*CreateProgramWithSource)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
	cl_int (*BuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *, void *, void *);
	cl_int (*GetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void *, size_t *);
	cl_kernel (*CreateKernel)(cl_program, const char *, cl_int *);
	cl_mem (*CreateBuffer)(cl_context, cl_bitfield, size_t, void *, cl_int *);
	cl_int (*ReleaseMemObject)(cl_mem);
	cl_int (*SetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
	cl_int (*EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *,
		const size_t *, cl_uint, const cl_event *, cl_event *);
	cl_int (*EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void *,
		cl_uint, const cl_event *, cl_event *);
	cl_int (*EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void *,
		cl_uint, const cl_event *, cl_event *);
	cl_int (*Finish)(cl_command_queue);
} cl;

/* a work-group of WG items an output row; BM activation rows per launch */
#define WG 64
#define BM 8

/* every load is a whole aligned vector: through a byte pointer the compiler
   cannot assume alignment and splits it, which held the 555X at 26 GB/s of its
   74 (an f32 row read as float4s reached 72). So Q8_0 and Q6_K, whose blocks
   are 34 and 210 bytes, are repacked at upload into aligned arrays per row
   (gpu_rb below); Q4_K's 144-byte blocks are aligned as they are. One wave of
   64 items a row: a load is then 64 neighbouring vectors of one row (a wave
   split over 4 rows got 39 GB/s). mad, not fma(): fma is quarter rate on some
   GPUs (345 GFLOPS on the 555X against 1365 with mad) */
static const char *KSRC =
"#define WG 64\n#define BM 8\n"
"#define ARGS global const uchar *W, global const float *X, global float *Y, int n_in, int n_out, int b0, int B, int rb\n"
"#define HEAD local float red[BM * WG]; int t = get_local_id(0), r = get_group_id(0);"
" global const uchar *row = W + (size_t)r * rb; float acc[BM];"
" for (int b = 0; b < BM; b++) acc[b] = 0.0f;\n"
"#define END tail(red, acc, B, Y, r, n_out, b0);\n"
"#define XR(b) ((global const float4 *)(X + (size_t)(b0 + (b)) * n_in))\n"
"#define EACH for (int b = 0; b < BM; b++) if (b < B)\n"
"inline float sum4(float4 v) { return (v.x + v.y) + (v.z + v.w); }\n"
"inline float dot16(float16 w, global const float4 *x) {\n"
"	return sum4(w.s0123 * x[0] + w.s4567 * x[1] + w.s89ab * x[2] + w.scdef * x[3]);\n"
"}\n"
/* item b sums the 64 partials of activation row b */
"inline void tail(local float *red, float *acc, int B, global float *Y, int r, int n_out, int b0) {\n"
"	int t = get_local_id(0);\n"
"	for (int b = 0; b < BM; b++) red[b * WG + t] = acc[b];\n"
"	barrier(CLK_LOCAL_MEM_FENCE);\n"
"	if (t < B) {\n"
"		local const float *p = red + t * WG;\n"
"		float s = 0.0f;\n"
"		for (int i = 0; i < WG; i++) s += p[i];\n"
"		Y[(size_t)(b0 + t) * n_out + r] = s;\n"
"	}\n"
"}\n"
"kernel void mv_f32(ARGS) { HEAD\n"
"	for (int u = t; u < n_in / 4; u += WG) {\n"
"		float4 w = ((global const float4 *)row)[u];\n"
"		EACH acc[b] += sum4(w * XR(b)[u]);\n"
"	}\n"
"	END\n"
"}\n"
"kernel void mv_f16(ARGS) { HEAD\n"
"	for (int u = t; u < n_in / 4; u += WG) {\n"
"		float4 w = vloada_half4(u, (global const half *)row);\n"
"		EACH acc[b] += sum4(w * XR(b)[u]);\n"
"	}\n"
"	END\n"
"}\n"
"kernel void mv_bf16(ARGS) { HEAD\n"
"	for (int u = t; u < n_in / 4; u += WG) {\n"
"		float4 w = as_float4(convert_uint4(((global const ushort4 *)row)[u]) << 16);\n"
"		EACH acc[b] += sum4(w * XR(b)[u]);\n"
"	}\n"
"	END\n"
"}\n"
/* Q8_0 repacked: n_in int8 quants, then a half scale per 32; a unit is 16 */
"kernel void mv_q8_0(ARGS) { HEAD\n"
"	global const half *d = (global const half *)(row + n_in);\n"
"	for (int u = t; u < n_in / 16; u += WG) {\n"
"		float16 w = convert_float16(((global const char16 *)row)[u]) * vload_half(u >> 1, d);\n"
"		EACH acc[b] += dot16(w, XR(b) + u * 4);\n"
"	}\n"
"	END\n"
"}\n"
/* Q4_K as stored: 144-byte blocks of 256 (9 aligned 16-byte words: d, dmin and
   12 scale bytes, then 128 of quants) in 4 chunks of 64; a unit is half a chunk,
   16 bytes whose low nibbles are 16 values of sub-block 2j, high of 2j+1 */
"inline void smk(int s, uint4 hd, float *d, float *m) {\n"
"	int k = (s & 3) * 8;\n"
"	uint a = (hd.y >> k) & 255, b = (hd.z >> k) & 255, c = (hd.w >> k) & 255;\n"
"	if (s < 4) { *d = a & 63; *m = b & 63; }\n"
"	else { *d = (c & 15) | ((a >> 6) << 4); *m = (c >> 4) | ((b >> 6) << 4); }\n"
"}\n"
"kernel void mv_q4_k(ARGS) { HEAD\n"
"	for (int u = t; u < (n_in >> 8) * 8; u += WG) {\n"
"		int i = u >> 3, j = (u >> 1) & 3, hh = u & 1;\n"
"		global const uint4 *blk = (global const uint4 *)(row + i * 144);\n"
"		uint4 hd = blk[0];\n"
"		float2 dd = vloada_half2(0, (global const half *)blk);\n"
"		float s1, m1, s2, m2;\n"
"		smk(2 * j, hd, &s1, &m1); smk(2 * j + 1, hd, &s2, &m2);\n"
"		uchar16 q = as_uchar16(blk[1 + 2 * j + hh]);\n"
"		float16 lo = convert_float16(q & (uchar16)15) * (dd.x * s1) - dd.y * m1;\n"
"		float16 hi = convert_float16(q >> (uchar16)4) * (dd.x * s2) - dd.y * m2;\n"
"		int at = (i * 256 + j * 64 + hh * 16) >> 2;\n"
"		EACH acc[b] += dot16(lo, XR(b) + at) + dot16(hi, XR(b) + at + 8);\n"
"	}\n"
"	END\n"
"}\n"
/* Q6_K repacked: a row's ql (128 a block), qh (64), scales (16), then d as
   halves; a unit is 16 positions l of a block's half, giving l, l+32, l+64 and
   l+96 */
"kernel void mv_q6_k(ARGS) { HEAD\n"
"	int nb = n_in >> 8;\n"
"	global const uint4 *ql = (global const uint4 *)row, *qh = ql + nb * 8;\n"
"	global const uint2 *sc = (global const uint2 *)(qh + nb * 4);\n"
"	global const half *dv = (global const half *)(row + nb * 208);\n"
"	for (int u = t; u < nb * 4; u += WG) {\n"
"		int i = u >> 2, h = (u >> 1) & 1, lg = u & 1;\n"
"		uchar16 A = as_uchar16(ql[i * 8 + h * 4 + lg]), C = as_uchar16(ql[i * 8 + h * 4 + 2 + lg]);\n"
"		uchar16 H = as_uchar16(qh[i * 4 + h * 2 + lg]);\n"
/* scales lg, lg+2, lg+4, lg+6 of the half's 8, by shifts: as_char8(..).even
   gave 16-bit pairs of neighbouring scales on the 555X (Apple's compiler) */
"		uint2 sv = sc[i * 2 + h]; int k = lg * 8;\n"
"		int4 s4 = as_int4((uint4)(sv.x << (24 - k), sv.x << (8 - k), sv.y << (24 - k), sv.y << (8 - k))) >> 24;\n"
"		float4 f = convert_float4(s4) * vload_half(i, dv);\n"
"		uchar16 m = (uchar16)15, three = (uchar16)3;\n"
"		float16 q1 = (convert_float16((A & m) | ((H & three) << (uchar16)4)) - 32.0f) * f.x;\n"
"		float16 q2 = (convert_float16((C & m) | (((H >> (uchar16)2) & three) << (uchar16)4)) - 32.0f) * f.y;\n"
"		float16 q3 = (convert_float16((A >> (uchar16)4) | (((H >> (uchar16)4) & three) << (uchar16)4)) - 32.0f) * f.z;\n"
"		float16 q4 = (convert_float16((C >> (uchar16)4) | (((H >> (uchar16)6) & three) << (uchar16)4)) - 32.0f) * f.w;\n"
"		int at = (i * 256 + h * 128 + lg * 16) >> 2;\n"
"		EACH acc[b] += dot16(q1, XR(b) + at) + dot16(q2, XR(b) + at + 8) + dot16(q3, XR(b) + at + 16) + dot16(q4, XR(b) + at + 24);\n"
"	}\n"
"	END\n"
"}\n";

/* the trainer's kernels (gpu.h, GK_*). Every operand is a buffer and an offset
   in elements, so one buffer holds many tensors. gemm_f32: C[M,N] (+)= op(A) .
   op(B), op(A)(m,k) = A[m*sam + k*sak], op(B)(k,n) = B[k*sbk + n*sbn]: a
   work-group of 16 x 16 items makes a 64 x 64 tile of C, 4 x 4 an item at a
   stride of 16 (neighbouring items store neighbouring floats), through 16-deep
   slices of A and B in local memory, each read along its stride-1 axis. The
   rest mirror train.c's CPU passes one for one */
static const char *TSRC =
"#define PA(t, n) global t *n##_, long n##o\n"
"#define AT(t, n) global t *n = n##_ + n##o;\n"
"#define TS 64\n#define TK 16\n"
"kernel void gemm_f32(PA(const float, A), PA(const float, B), PA(float, C),\n"
"	int M, int N, int K, int sam, int sak, int sbk, int sbn, int ldc, int add,\n"
"	int zh, int hm, int h0, long sab, long sah, int dva, long sbb, long sbh, int dvb, long scb, long sch, int dvc) {\n"
"	AT(const float, A) AT(const float, B) AT(float, C)\n"
"	int z = get_group_id(2), zb = z / zh, h = z % zh * hm + h0;\n"
"	A += zb * sab + (long)(h / dva) * sah; B += zb * sbb + (long)(h / dvb) * sbh; C += zb * scb + (long)(h / dvc) * sch;\n"
"	local float As[TK][TS], Bs[TK][TS];\n"
"	int tx = get_local_id(0), ty = get_local_id(1), id = ty * 16 + tx, at = sak != 1, bt = sbn != 1;\n"
"	int m0 = get_group_id(1) * TS, n0 = get_group_id(0) * TS;\n"
"	float acc[4][4];\n"
"	for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;\n"
"	for (int k0 = 0; k0 < K; k0 += TK) {\n"
"		for (int q = 0; q < 4; q++) {\n"
"			int e = id + 256 * q, r = at ? e & 63 : e >> 4, k = at ? e >> 6 : e & 15;\n"
"			int gm = m0 + r, gk = k0 + k;\n"
"			As[k][r] = gm < M && gk < K ? A[(long)gm * sam + (long)gk * sak] : 0.0f;\n"
"			int c = bt ? e >> 4 : e & 63, kb = bt ? e & 15 : e >> 6, gn = n0 + c, gb = k0 + kb;\n"
"			Bs[kb][c] = gn < N && gb < K ? B[(long)gb * sbk + (long)gn * sbn] : 0.0f;\n"
"		}\n"
"		barrier(CLK_LOCAL_MEM_FENCE);\n"
"		for (int k = 0; k < TK; k++) {\n"
"			float a[4], b[4];\n"
"			for (int i = 0; i < 4; i++) { a[i] = As[k][ty + 16 * i]; b[i] = Bs[k][tx + 16 * i]; }\n"
"			for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i][j] = mad(a[i], b[j], acc[i][j]);\n"
"		}\n"
"		barrier(CLK_LOCAL_MEM_FENCE);\n"
"	}\n"
"	for (int i = 0; i < 4; i++) {\n"
"		int gm = m0 + ty + 16 * i;\n"
"		if (gm >= M) continue;\n"
"		for (int j = 0; j < 4; j++) {\n"
"			int gn = n0 + tx + 16 * j;\n"
"			if (gn >= N) continue;\n"
"			long o = (long)gm * ldc + gn;\n"
"			C[o] = add ? C[o] + acc[i][j] : acc[i][j];\n"
"		}\n"
"	}\n"
"}\n"
/* a work-group of 64 sums v over its items into red[0] */
"inline float wsum(local float *red, float v) {\n"
"	int t = get_local_id(0);\n"
"	red[t] = v; barrier(CLK_LOCAL_MEM_FENCE);\n"
"	for (int h = 32; h > 0; h >>= 1) { if (t < h) red[t] += red[t + h]; barrier(CLK_LOCAL_MEM_FENCE); }\n"
"	float s = red[0]; barrier(CLK_LOCAL_MEM_FENCE);\n"
"	return s;\n"
"}\n"
"kernel void embed(PA(float, x), PA(const float, E), PA(const int, ids), int D, int N) {\n"
"	AT(float, x) AT(const float, E) AT(const int, ids)\n"
"	long i = get_global_id(0);\n"
"	if (i < (long)N * D) x[i] = E[(long)ids[i / D] * D + i % D];\n"
"}\n"
/* a row a work-group: x += xa (has), out = x / rms(x) * w, r = 1 / rms */
"kernel void add_rms(PA(float, out), PA(float, r), PA(float, x), PA(const float, xa), PA(const float, w),\n"
"	int D, float eps, int has) {\n"
"	AT(float, out) AT(float, r) AT(float, x) AT(const float, xa) AT(const float, w)\n"
"	local float red[64];\n"
"	long n = get_group_id(0); int t = get_local_id(0);\n"
"	global float *xr = x + n * D;\n"
"	float ss = 0.0f;\n"
"	for (int i = t; i < D; i += 64) { float v = xr[i]; if (has) { v += xa[n * D + i]; xr[i] = v; } ss += v * v; }\n"
"	float s = 1.0f / sqrt(wsum(red, ss) / (float)D + eps);\n"
"	if (!t) r[n] = s;\n"
"	for (int i = t; i < D; i += 64) out[n * D + i] = xr[i] * s * w[i];\n"
"}\n"
"kernel void rms_dx(PA(float, dx), PA(const float, dy), PA(const float, x), PA(const float, r), PA(const float, w), int D) {\n"
"	AT(float, dx) AT(const float, dy) AT(const float, x) AT(const float, r) AT(const float, w)\n"
"	local float red[64];\n"
"	long n = get_group_id(0); int t = get_local_id(0);\n"
"	float s = r[n], dot = 0.0f;\n"
"	for (int i = t; i < D; i += 64) dot += dy[n * D + i] * w[i] * x[n * D + i];\n"
"	float k = s * s * s * wsum(red, dot) / (float)D;\n"
"	for (int i = t; i < D; i += 64) dx[n * D + i] += s * dy[n * D + i] * w[i] - k * x[n * D + i];\n"
"}\n"
/* the norm weight's gradient in two passes: rows in blocks of 128 into part, then the blocks summed */
"kernel void rms_dw(PA(float, part), PA(const float, dy), PA(const float, x), PA(const float, r), int D, int N) {\n"
"	AT(float, part) AT(const float, dy) AT(const float, x) AT(const float, r)\n"
"	int i = get_global_id(0), y = get_global_id(1);\n"
"	if (i >= D) return;\n"
"	float s = 0.0f;\n"
"	for (int n = y * 128; n < N && n < y * 128 + 128; n++) s += dy[(long)n * D + i] * x[(long)n * D + i] * r[n];\n"
"	part[(long)y * D + i] = s;\n"
"}\n"
"kernel void colsum(PA(float, dw), PA(const float, part), int D, int rows) {\n"
"	AT(float, dw) AT(const float, part)\n"
"	int i = get_global_id(0);\n"
"	if (i >= D) return;\n"
"	float s = 0.0f;\n"
"	for (int y = 0; y < rows; y++) s += part[(long)y * D + i];\n"
"	dw[i] += s;\n"
"}\n"
"kernel void rope(PA(float, x), PA(const float, cs), PA(const float, sn), int N, int T, int nh, int hd, float dir) {\n"
"	AT(float, x) AT(const float, cs) AT(const float, sn)\n"
"	int hh = hd / 2;\n"
"	long id = get_global_id(0);\n"
"	if (id >= (long)N * nh * hh) return;\n"
"	int i = id % hh; long nh_ = id / hh; int h = nh_ % nh; long n = nh_ / nh; int pos = n % T;\n"
"	global float *v = x + (n * nh + h) * hd;\n"
"	float c = cs[pos * hh + i], s = dir * sn[pos * hh + i], a = v[i], b = v[i + hh];\n"
"	v[i] = a * c - b * s; v[i + hh] = a * s + b * c;\n"
"}\n"
/* causal attention's softmax over S = QK^T in place, a row of a head a work-group of 64
   (row = (sequence x heads + head) x T + position): p = softmax(S x sc) on s <= t, 0 after */
"kernel void att_sm(PA(float, p), int T, float sc) {\n"
"	AT(float, p)\n"
"	local float red[64];\n"
"	long row = get_group_id(0); int t = row % T, l = get_local_id(0);\n"
"	global float *P = p + row * T;\n"
"	float mx = -INFINITY;\n"
"	for (int s = l; s <= t; s += 64) mx = fmax(mx, P[s] * sc);\n"
"	red[l] = mx; barrier(CLK_LOCAL_MEM_FENCE);\n"
"	for (int h = 32; h > 0; h >>= 1) { if (l < h) red[l] = fmax(red[l], red[l + h]); barrier(CLK_LOCAL_MEM_FENCE); }\n"
"	mx = red[0]; barrier(CLK_LOCAL_MEM_FENCE);\n"
"	float sum = 0.0f;\n"
"	for (int s = l; s <= t; s += 64) { float e = exp(P[s] * sc - mx); P[s] = e; sum += e; }\n"
"	float inv = 1.0f / wsum(red, sum);\n"
"	for (int s = l; s < T; s += 64) P[s] = s <= t ? P[s] * inv : 0.0f;\n"
"}\n"
/* its backward on dP = dO V^T in place: dS = P (dP - rowsum(P dP)) x sc */
"kernel void att_smb(PA(const float, p), PA(float, ds), int T, float sc) {\n"
"	AT(const float, p) AT(float, ds)\n"
"	local float red[64];\n"
"	long row = get_group_id(0); int t = row % T, l = get_local_id(0);\n"
"	global const float *P = p + row * T; global float *S = ds + row * T;\n"
"	float d = 0.0f;\n"
"	for (int s = l; s <= t; s += 64) d += P[s] * S[s];\n"
"	d = wsum(red, d);\n"
"	for (int s = l; s < T; s += 64) S[s] = s <= t ? P[s] * (S[s] - d) * sc : 0.0f;\n"
"}\n"
"kernel void swiglu_f(PA(const float, g), PA(const float, u), PA(float, h), long n) {\n"
"	AT(const float, g) AT(const float, u) AT(float, h)\n"
"	long i = get_global_id(0);\n"
"	if (i < n) h[i] = g[i] / (1.0f + exp(-g[i])) * u[i];\n"
"}\n"
"kernel void swiglu_b(PA(const float, g), PA(const float, u), PA(float, dA), PA(float, dB), long n) {\n"
"	AT(const float, g) AT(const float, u) AT(float, dA) AT(float, dB)\n"
"	long i = get_global_id(0);\n"
"	if (i >= n) return;\n"
"	float x = g[i], sg = 1.0f / (1.0f + exp(-x)), dh = dA[i];\n"
"	dB[i] = dh * x * sg;\n"
"	dA[i] = dh * u[i] * sg * (1.0f + x * (1.0f - sg));\n"
"}\n"
/* softmax cross-entropy, a row a work-group of 64: its nll, and with scale > 0 the logits become scale x dlogits */
"kernel void ce(PA(float, z), PA(const int, tgt), PA(float, nll), int V, float scale) {\n"
"	AT(float, z) AT(const int, tgt) AT(float, nll)\n"
"	local float red[64];\n"
"	long n = get_group_id(0); int t = get_local_id(0), y = tgt[n];\n"
"	global float *zr = z + n * V;\n"
"	float mx = -INFINITY, zy = zr[y];\n"
"	for (int i = t; i < V; i += 64) mx = fmax(mx, zr[i]);\n"
"	red[t] = mx; barrier(CLK_LOCAL_MEM_FENCE);\n"
"	for (int h = 32; h > 0; h >>= 1) { if (t < h) red[t] = fmax(red[t], red[t + h]); barrier(CLK_LOCAL_MEM_FENCE); }\n"
"	mx = red[0]; barrier(CLK_LOCAL_MEM_FENCE);\n"
"	float se = 0.0f;\n"
"	for (int i = t; i < V; i += 64) { float e = exp(zr[i] - mx); se += e; if (scale > 0) zr[i] = e; }\n"
"	se = wsum(red, se);\n"
"	if (!t) nll[n] = log(se) + mx - zy;\n"
"	if (scale > 0) { float k = scale / se; for (int i = t; i < V; i += 64) zr[i] = zr[i] * k - (i == y ? scale : 0.0f); }\n"
"}\n"
/* the embedding's gradient, a (distinct token, element) an item: ids holds the tokens, then
   where each one's positions start (nu + 1), then the positions grouped by token */
"kernel void emb_grad(PA(float, G), PA(const float, dx), PA(const int, ids), int nu, int D) {\n"
"	AT(float, G) AT(const float, dx) AT(const int, ids)\n"
"	long id = get_global_id(0);\n"
"	if (id >= (long)nu * D) return;\n"
"	int u = id / D, i = id % D;\n"
"	global const int *off = ids + nu, *pos = off + nu + 1;\n"
"	float s = 0.0f;\n"
"	for (int j = off[u]; j < off[u + 1]; j++) s += dx[(long)pos[j] * D + i];\n"
"	G[(long)ids[u] * D + i] += s;\n"
"}\n"
"kernel void sumsq(PA(const float, x), PA(float, part), long n) {\n"
"	AT(const float, x) AT(float, part)\n"
"	local float red[64];\n"
"	float s = 0.0f;\n"
"	for (long i = get_global_id(0); i < n; i += get_global_size(0)) s += x[i] * x[i];\n"
"	s = wsum(red, s);\n"
"	if (!get_local_id(0)) part[get_group_id(0)] = s;\n"
"}\n"
"kernel void zero(PA(float, x), long n) {\n"
"	AT(float, x)\n"
"	long i = get_global_id(0);\n"
"	if (i < n) x[i] = 0.0f;\n"
"}\n"
"kernel void scale(PA(float, x), long n, float k) {\n"
"	AT(float, x)\n"
"	long i = get_global_id(0);\n"
"	if (i < n) x[i] *= k;\n"
"}\n"
"kernel void adamw(PA(float, p), PA(const float, g), PA(float, mo), PA(float, ve), long n,\n"
"	float lr, float decay, float b1, float b2, float eps, float c1, float c2) {\n"
"	AT(float, p) AT(const float, g) AT(float, mo) AT(float, ve)\n"
"	long j = get_global_id(0);\n"
"	if (j >= n) return;\n"
"	float m = b1 * mo[j] + (1.0f - b1) * g[j], v = b2 * ve[j] + (1.0f - b2) * g[j] * g[j];\n"
"	mo[j] = m; ve[j] = v;\n"
"	p[j] -= lr * ((m / c1) / (sqrt(v / c2) + eps) + decay * p[j]);\n"
"}\n";

/* the matvecs, then the trainer's kernels in gpu.h's GK_ order */
enum { K_F32, K_F16, K_BF16, K_Q8_0, K_Q4_K, K_Q6_K, K_GK, K_N = K_GK + GK_N };
static const char *KNAME[K_N] = { "mv_f32", "mv_f16", "mv_bf16", "mv_q8_0", "mv_q4_k", "mv_q6_k",
	"gemm_f32", "embed", "add_rms", "rms_dx", "rms_dw", "colsum", "rope", "att_sm", "att_smb",
	"swiglu_f", "swiglu_b", "ce", "emb_grad", "sumsq", "zero", "scale", "adamw" };
/* a work-group's shape per GK_ kernel (y 1 unless 2-D) */
static const unsigned char GLX[GK_N] = { 16, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64 };

typedef struct { const void *data; cl_mem mem; size_t bytes; } wslot;

static struct {
	int state;                 /* 0 not tried, 1 ready, -1 none or failed */
	char name[128];
	cl_context cx; cl_command_queue q; cl_kernel k[K_N];
	cl_mem xb, yb; size_t xcap, ycap;
	cl_mem ab, bb, cb; size_t acap, bcap, ccap;   /* gpu_gemm's operands */
	wslot *w; int nw, capw;
	size_t used, budget, max_alloc;
	long gemms;
	int broken;                /* a GPU was there and the kernels did not build on it */
	pthread_mutex_t lock;
} G = { .lock = PTHREAD_MUTEX_INITIALIZER };

static void *lib_open(void) {
	static const char *paths[] = {
#ifdef __APPLE__
		"/System/Library/Frameworks/OpenCL.framework/OpenCL",
#endif
		"libOpenCL.so.1", "libOpenCL.so",
		"/vendor/lib64/libOpenCL.so", "/system/vendor/lib64/libOpenCL.so",   /* Android */
		NULL };
	for (int i = 0; paths[i]; i++) { void *h = dlopen(paths[i], RTLD_NOW | RTLD_LOCAL); if (h) return h; }
	return NULL;
}

static int load(void *h) {
#define SYM(f) if (!(*(void **)&cl.f = dlsym(h, "cl" #f))) return -1;
	SYM(GetPlatformIDs) SYM(GetDeviceIDs) SYM(GetDeviceInfo) SYM(CreateContext) SYM(CreateCommandQueue)
	SYM(CreateProgramWithSource) SYM(BuildProgram) SYM(GetProgramBuildInfo) SYM(CreateKernel)
	SYM(CreateBuffer) SYM(ReleaseMemObject) SYM(SetKernelArg) SYM(EnqueueNDRangeKernel)
	SYM(EnqueueWriteBuffer) SYM(EnqueueReadBuffer) SYM(Finish)
#undef SYM
	return 0;
}

static int init(void) {
	const char *e = getenv("HAKM_GPU");
	if (e && !strcmp(e, "0")) return -1;
	void *h = lib_open();
	if (!h || load(h)) return -1;
	cl_platform_id pl[8]; cl_uint np = 0;
	if (cl.GetPlatformIDs(8, pl, &np) || !np) return -1;
	int want = getenv("HAKM_GPU_DEVICE") ? atoi(getenv("HAKM_GPU_DEVICE")) : 0, seen = 0;
	cl_device_id dev = NULL;
	for (cl_uint p = 0; p < np && !dev; p++) {
		cl_device_id d[8]; cl_uint nd = 0;
		if (cl.GetDeviceIDs(pl[p], CL_DEVICE_TYPE_GPU, 8, d, &nd)) continue;
		for (cl_uint i = 0; i < nd && !dev; i++) if (seen++ == want) dev = d[i];
	}
	if (!dev) return -1;
	cl_ulong mem = 0, ma = 0;
	cl.GetDeviceInfo(dev, CL_DEVICE_NAME, sizeof G.name, G.name, NULL);
	cl.GetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof mem, &mem, NULL);
	cl.GetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof ma, &ma, NULL);
	G.max_alloc = (size_t)ma;
	/* 2/3, not all: a GPU that also drives a display holds its framebuffers in
	   the same memory, and past what is free the driver pages buffers over PCIe
	   on every call (sho on the 555X: 1.7 tok/s at a 1843 MB budget, 7.6 at 1200) */
	G.budget = getenv("HAKM_GPU_MB") ? (size_t)atol(getenv("HAKM_GPU_MB")) << 20 : (size_t)(mem / 3 * 2);
	cl_int err;
	G.cx = cl.CreateContext(NULL, 1, &dev, NULL, NULL, &err);
	if (err) return -1;
	G.q = cl.CreateCommandQueue(G.cx, dev, 0, &err);
	if (err) return -1;
	const char *src[2] = { KSRC, TSRC };
	cl_program pr = cl.CreateProgramWithSource(G.cx, 2, src, NULL, &err);
	if (err) return -1;
	if (cl.BuildProgram(pr, 1, &dev, "-cl-mad-enable", NULL, NULL)) {
		char log[4096] = "";
		cl.GetProgramBuildInfo(pr, dev, CL_PROGRAM_BUILD_LOG, sizeof log - 1, log, NULL);
		fprintf(stderr, "hakm: gpu kernels did not build on %s, using the CPU:\n%s\n", G.name, log);
		G.broken = 1;
		return -1;
	}
	for (int k = 0; k < K_N; k++) { G.k[k] = cl.CreateKernel(pr, KNAME[k], &err); if (err) return -1; }
	return 0;
}

static int ready(void) {
	if (!G.state) {
		G.state = init() ? -1 : 1;
		if (G.state > 0) fprintf(stderr, "hakm: gpu %s, %zu MB for weights\n", G.name, G.budget >> 20);
	}
	return G.state > 0;
}

/* bytes a row takes on the device: as stored, except the two repacked types */
static size_t gpu_rb(int type, int n_in) {
	size_t nb = (size_t)n_in / QK_K;
	if (type == MLF2_TYPE_Q8_0) return ((size_t)n_in + (size_t)n_in / 32 * 2 + 15) & ~(size_t)15;
	if (type == MLF2_TYPE_Q6_K) return (nb * 208 + nb * 2 + 15) & ~(size_t)15;
	return quant_row_bytes(type, n_in);
}

/* one row as stored to its device layout (see KSRC) */
static void repack(int type, const unsigned char *src, unsigned char *dst, int n_in) {
	if (type == MLF2_TYPE_Q8_0) {
		const block_q8_0 *b = (const block_q8_0 *)src;
		uint16_t *d = (uint16_t *)(dst + n_in);
		for (int i = 0; i < n_in / 32; i++) { memcpy(dst + i * 32, b[i].qs, 32); d[i] = b[i].d; }
	} else {
		const block_q6_K *b = (const block_q6_K *)src;
		int nb = n_in / QK_K;
		uint16_t *d = (uint16_t *)(dst + (size_t)nb * 208);
		for (int i = 0; i < nb; i++) {
			memcpy(dst + (size_t)i * 128, b[i].ql, 128);
			memcpy(dst + (size_t)nb * 128 + i * 64, b[i].qh, 64);
			memcpy(dst + (size_t)nb * 192 + i * 16, b[i].scales, 16);
			d[i] = b[i].d;
		}
	}
}

/* the device copy of a weight tensor, made on first use; NULL keeps it on the CPU */
static cl_mem weights(const void *data, int type, int n_in, int n_out) {
	for (int i = 0; i < G.nw; i++) if (G.w[i].data == data) return G.w[i].mem;
	if (G.nw == G.capw) {
		int cap = G.capw ? G.capw * 2 : 256;
		wslot *n = realloc(G.w, (size_t)cap * sizeof *n);
		if (!n) return NULL;
		G.w = n; G.capw = cap;
	}
	size_t rb = quant_row_bytes(type, n_in), grb = gpu_rb(type, n_in), bytes = grb * (size_t)n_out;
	cl_mem m = NULL;
	if (G.used + bytes <= G.budget && bytes <= G.max_alloc) {
		unsigned char *tmp = NULL;
		const void *up = data;
		if (type == MLF2_TYPE_Q8_0 || type == MLF2_TYPE_Q6_K) {   /* by type: at n_in 1024 a repacked
		                                                           Q8_0 row is as long as a stored one */
			if (!(tmp = calloc(bytes, 1))) return NULL;
			for (int r = 0; r < n_out; r++)
				repack(type, (const unsigned char *)data + rb * r, tmp + grb * r, n_in);
			up = tmp;
		}
		cl_int err;
		m = cl.CreateBuffer(G.cx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, (void *)up, &err);
		free(tmp);
		if (err) m = NULL;
		else G.used += bytes;
	}
	G.w[G.nw++] = (wslot){ data, m, bytes };   /* a NULL slot remembers "CPU", so it is not retried */
	return m;
}

static int grow(cl_mem *b, size_t *cap, size_t need) {
	if (need <= *cap) return 0;
	if (*b) cl.ReleaseMemObject(*b);
	cl_int err;
	*b = cl.CreateBuffer(G.cx, CL_MEM_READ_WRITE, need, NULL, &err);
	*cap = err ? 0 : need;
	if (err) *b = NULL;
	return err ? -1 : 0;
}

static int kernel_for(int type, int n_in) {
	switch (type) {
	case MLF2_TYPE_F32:  return n_in % 4 ? -1 : K_F32;
	case MLF2_TYPE_F16:  return n_in % 4 ? -1 : K_F16;
	case MLF2_TYPE_BF16: return n_in % 4 ? -1 : K_BF16;
	case MLF2_TYPE_Q8_0: return n_in % 32 ? -1 : K_Q8_0;
	case MLF2_TYPE_Q4_K: return n_in % QK_K ? -1 : K_Q4_K;
	case MLF2_TYPE_Q6_K: return n_in % QK_K ? -1 : K_Q6_K;
	}
	return -1;
}

static int run(float *out, const float *x, cl_mem wm, int k, int n_out, int n_in, int B, int rb) {
	size_t xs = (size_t)B * n_in * sizeof(float), ys = (size_t)B * n_out * sizeof(float);
	if (grow(&G.xb, &G.xcap, xs) || grow(&G.yb, &G.ycap, ys)) return -1;
	if (cl.EnqueueWriteBuffer(G.q, G.xb, 0, 0, xs, x, 0, NULL, NULL)) return -1;
	cl_kernel kr = G.k[k];
	size_t gs = (size_t)n_out * WG, ls = WG;
	for (int b0 = 0; b0 < B; b0 += BM) {
		int nb = B - b0 < BM ? B - b0 : BM;
		if (cl.SetKernelArg(kr, 0, sizeof wm, &wm) || cl.SetKernelArg(kr, 1, sizeof G.xb, &G.xb) ||
		    cl.SetKernelArg(kr, 2, sizeof G.yb, &G.yb) || cl.SetKernelArg(kr, 3, sizeof n_in, &n_in) ||
		    cl.SetKernelArg(kr, 4, sizeof n_out, &n_out) || cl.SetKernelArg(kr, 5, sizeof b0, &b0) ||
		    cl.SetKernelArg(kr, 6, sizeof nb, &nb) || cl.SetKernelArg(kr, 7, sizeof rb, &rb))
			return -1;
		if (cl.EnqueueNDRangeKernel(G.q, kr, 1, NULL, &gs, &ls, 0, NULL, NULL)) return -1;
	}
	return cl.EnqueueReadBuffer(G.q, G.yb, 1, 0, ys, out, 0, NULL, NULL) ? -1 : 0;
}

int gpu_matmul(float *out, const float *x, const hakm_model *m, const mlf2_tensor *w,
               const float *bias, int n_out, int n_in, int B) {
	int k = kernel_for((int)w->type, n_in);
	if (k < 0 || B < 1 || n_out < 1) return -1;
	size_t rb = gpu_rb((int)w->type, n_in);
	if (!rb || rb > INT32_MAX) return -1;
	pthread_mutex_lock(&G.lock);
	int rc = -1;
	if (ready()) {
		cl_mem wm = weights(mlf2_data(m, w), (int)w->type, n_in, n_out);
		if (wm) {
			rc = run(out, x, wm, k, n_out, n_in, B, (int)rb);
			if (rc) { G.state = -1; fprintf(stderr, "hakm: gpu error, using the CPU from here\n"); }
		}
	}
	pthread_mutex_unlock(&G.lock);
	if (!rc && bias)
		for (int b = 0; b < B; b++) for (int r = 0; r < n_out; r++) out[(size_t)b * n_out + r] += bias[r];
	return rc;
}

/* the span of a strided operand, in floats: its last element + 1 */
static size_t span(int rows, int cols, size_t sr, size_t sc) { return (size_t)(rows - 1) * sr + (size_t)(cols - 1) * sc + 1; }

static int gemm_run(int M, int N, int K, const float *A, size_t sm, size_t sk, const float *B, size_t ldb,
                    float *C, size_t ldc, int add) {
	size_t as = span(M, K, sm, sk) * 4, bs = span(K, N, ldb, 1) * 4, cs = span(M, N, ldc, 1) * 4;
	if (as > G.max_alloc || bs > G.max_alloc || cs > G.max_alloc || as + bs + cs > G.budget) return 1;   /* too big: the CPU */
	if (grow(&G.ab, &G.acap, as) || grow(&G.bb, &G.bcap, bs) || grow(&G.cb, &G.ccap, cs)) return -1;
	if (cl.EnqueueWriteBuffer(G.q, G.ab, 0, 0, as, A, 0, NULL, NULL) ||
	    cl.EnqueueWriteBuffer(G.q, G.bb, 0, 0, bs, B, 0, NULL, NULL) ||
	    ((add || ldc != (size_t)N) && cl.EnqueueWriteBuffer(G.q, G.cb, 0, 0, cs, C, 0, NULL, NULL)))   /* rows with gaps: keep them */
		return -1;
	cl_kernel kr = G.k[K_GK + GK_GEMM];
	cl_long z = 0;
	int a[10] = { M, N, K, (int)sm, (int)sk, (int)ldb, 1, (int)ldc, add };
	cl_mem mm[3] = { G.ab, G.bb, G.cb };
	for (int i = 0; i < 3; i++) if (cl.SetKernelArg(kr, 2 * i, sizeof(cl_mem), &mm[i]) || cl.SetKernelArg(kr, 2 * i + 1, sizeof z, &z)) return -1;
	for (int i = 0; i < 9; i++) if (cl.SetKernelArg(kr, 6 + i, sizeof(int), &a[i])) return -1;
	int one = 1;   /* one product, no batch: zh 1, hm 1, h0 0, every batch stride 0, divisors 1 */
	for (int i = 0; i < 12; i++) {
		int at = 15 + i, isl = i == 3 || i == 4 || i == 6 || i == 7 || i == 9 || i == 10;
		if (isl ? cl.SetKernelArg(kr, at, sizeof z, &z) : cl.SetKernelArg(kr, at, sizeof one, i == 2 ? (const void *)&a[9] : (const void *)&one)) return -1;
	}
	size_t gs[3] = { (size_t)(N + 63) / 64 * 16, (size_t)(M + 63) / 64 * 16, 1 }, ls[3] = { 16, 16, 1 };
	if (cl.EnqueueNDRangeKernel(G.q, kr, 3, NULL, gs, ls, 0, NULL, NULL)) return -1;
	if (cl.EnqueueReadBuffer(G.q, G.cb, 1, 0, cs, C, 0, NULL, NULL)) return -1;
	G.gemms++;
	return 0;
}

int gpu_gemm(int M, int N, int K, const float *A, size_t sm, size_t sk, const float *B, size_t ldb,
             float *C, size_t ldc, int add) {
	if (M < 1 || N < 1 || K < 1 || (sm != 1 && sk != 1) || sm > INT32_MAX || sk > INT32_MAX ||
	    ldb > INT32_MAX || ldc > INT32_MAX || (size_t)N > ldb || (size_t)N > ldc)
		return -1;
	pthread_mutex_lock(&G.lock);
	int rc = -1;
	if (ready()) {
		rc = gemm_run(M, N, K, A, sm, sk, B, ldb, C, ldc, add);
		if (rc < 0) { G.state = -1; fprintf(stderr, "hakm: gpu error, using the CPU from here\n"); }
		if (rc > 0) rc = -1;
	}
	pthread_mutex_unlock(&G.lock);
	return rc;
}

/* ── buffers and kernels for the trainer (gpu.h) ─────────────────────────── */

struct gpu_mem { cl_mem m; size_t bytes; };

/* an error past this point leaves the device's state unknown: off for good */
static int fail(void) { G.state = -1; fprintf(stderr, "hakm: gpu error, using the CPU from here\n"); return -1; }

gpu_mem *gpu_new(size_t bytes) {
	pthread_mutex_lock(&G.lock);
	gpu_mem *b = NULL;
	if (ready() && bytes && bytes <= G.max_alloc && (b = calloc(1, sizeof *b))) {
		cl_int err;
		b->m = cl.CreateBuffer(G.cx, CL_MEM_READ_WRITE, bytes, NULL, &err);
		b->bytes = bytes;
		if (err) { free(b); b = NULL; }
	}
	pthread_mutex_unlock(&G.lock);
	return b;
}

void gpu_del(gpu_mem *b) {
	if (!b) return;
	pthread_mutex_lock(&G.lock);
	cl.ReleaseMemObject(b->m);
	pthread_mutex_unlock(&G.lock);
	free(b);
}

static int io(gpu_mem *b, size_t off, void *p, size_t n, int put) {
	if (!b || off > b->bytes || n > b->bytes - off) return -1;
	if (!n) return 0;
	pthread_mutex_lock(&G.lock);
	int rc = G.state > 0 ? 0 : -1;
	if (!rc && (put ? cl.EnqueueWriteBuffer(G.q, b->m, CL_TRUE, off, n, p, 0, NULL, NULL)
	                : cl.EnqueueReadBuffer(G.q, b->m, CL_TRUE, off, n, p, 0, NULL, NULL))) rc = fail();
	pthread_mutex_unlock(&G.lock);
	return rc;
}
int gpu_put(gpu_mem *b, size_t off, const void *src, size_t n) { return io(b, off, (void *)src, n, 1); }
int gpu_get(gpu_mem *b, size_t off, void *dst, size_t n) { return io(b, off, dst, n, 0); }

/* a kernel, not clEnqueueFillBuffer: Apple's fills with garbage (2.4e-29 on the 555X) */
int gpu_zero(gpu_mem *b, size_t off, size_t n) {
	if (!b || off > b->bytes || n > b->bytes - off || off % 4 || n % 4) return -1;
	if (!n) return 0;
	gpu_arg a[] = { GP(b, off / 4), GL((int64_t)(n / 4)) };
	return gpu_launch(GK_ZERO, n / 4, 1, 1, 2, a);
}

/* HAKM_GPU_PROF=1: each launch waited for and timed, a table per kernel at exit */
static double prof_s[GK_N]; static long prof_n[GK_N]; static int prof = -1;
static double mono(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void prof_out(void) {
	double all = 0;
	for (int k = 0; k < GK_N; k++) all += prof_s[k];
	fprintf(stderr, "hakm: gpu time by kernel, %.2f s in all\n", all);
	for (int k = 0; k < GK_N; k++)
		if (prof_n[k]) fprintf(stderr, "  %-9s %8ld launches %8.3f s %5.1f%%\n", KNAME[K_GK + k], prof_n[k], prof_s[k], 100 * prof_s[k] / all);
}

int gpu_launch(int k, size_t gx, size_t gy, size_t gz, int n, const gpu_arg *a) {
	if (k < 0 || k >= GK_N || !gx || !gy || !gz) return -1;
	if (prof < 0) { const char *e = getenv("HAKM_GPU_PROF"); prof = e && *e == '1'; if (prof) atexit(prof_out); }
	pthread_mutex_lock(&G.lock);
	double t0 = prof ? (cl.Finish(G.q), mono()) : 0;
	int rc = G.state > 0 ? 0 : -1;
	cl_kernel kr = G.k[K_GK + k];
	for (int i = 0, at = 0; !rc && i < n; i++) {
		cl_int e;
		if (a[i].t == GA_PTR) {
			cl_long o = a[i].off;
			e = cl.SetKernelArg(kr, at++, sizeof(cl_mem), &a[i].m->m);
			if (!e) e = cl.SetKernelArg(kr, at++, sizeof o, &o);
		} else if (a[i].t == GA_LONG) { cl_long v = a[i].off; e = cl.SetKernelArg(kr, at++, sizeof v, &v); }
		else e = cl.SetKernelArg(kr, at++, 4, a[i].t == GA_INT ? (const void *)&a[i].i : (const void *)&a[i].f);
		if (e) rc = fail();
	}
	size_t lx = GLX[k], ly = k == GK_GEMM ? 16 : 1;
	size_t gs[3] = { (gx + lx - 1) / lx * lx, (gy + ly - 1) / ly * ly, gz }, ls[3] = { lx, ly, 1 };
	if (!rc && cl.EnqueueNDRangeKernel(G.q, kr, 3, NULL, gs, ls, 0, NULL, NULL)) rc = fail();
	if (!rc && k == GK_GEMM) G.gemms++;
	if (prof && !rc) { cl.Finish(G.q); prof_s[k] += mono() - t0; prof_n[k]++; }
	pthread_mutex_unlock(&G.lock);
	return rc;
}

void gpu_forget(const void *base, size_t n) {
	const unsigned char *lo = base, *hi = lo + n;
	pthread_mutex_lock(&G.lock);
	for (int i = 0; i < G.nw;) {
		const unsigned char *d = G.w[i].data;
		if (d >= lo && d < hi) {
			if (G.w[i].mem) { cl.ReleaseMemObject(G.w[i].mem); G.used -= G.w[i].bytes; }
			G.w[i] = G.w[--G.nw];
		} else i++;
	}
	pthread_mutex_unlock(&G.lock);
}

const char *gpu_name(void) {
	pthread_mutex_lock(&G.lock);
	int ok = ready();
	pthread_mutex_unlock(&G.lock);
	return ok ? G.name : NULL;
}

size_t gpu_resident(void) { return G.used; }
long gpu_gemm_count(void) { return G.gemms; }
int gpu_broken(void) { gpu_name(); return G.broken; }
