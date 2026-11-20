/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* gpu.h — the matmuls on a GPU through OpenCL, loaded at run time (dlopen), so
 * the build needs no SDK, no headers and no toolchain: the kernels are C source
 * the driver compiles on the machine it runs on, which is also what lets a GPU
 * newer than this file run them. Built only with `make GPU=1` (-DHAKM_GPU).
 *
 * Weights go up once, on first use, and stay (keyed by their place in the
 * mapping); a tensor that would pass the memory budget stays on the CPU. The
 * CPU path stays the reference: every result here is checked against it in
 * tests/test_gpu.c, and any OpenCL error turns the GPU off for the process.
 *
 * env: HAKM_GPU=0 off; HAKM_GPU_DEVICE=N which GPU (0 = first);
 *      HAKM_GPU_MB=N weight budget (default 2/3 of the device's memory) */
#pragma once
#include <stdint.h>
#include "loader.h"

/* out = W @ x for B rows, as matmul_batch (nn.h) defines it. 0 when the GPU did
 * it; -1 when the caller must (no GPU, a type or shape without a kernel, over
 * the budget, an error). out is untouched on -1. */
int gpu_matmul(float *out, const float *x, const hakm_model *m, const mlf2_tensor *w,
               const float *bias, int n_out, int n_in, int B);

/* the trainer's product (train.c gemm): C[M,N] (+)= op(A)[M,K] . B[K,N] in f32,
 * op(A)(m,k) = A[m*sm + k*sk] with sm or sk 1, B and C row-major at ldb and ldc,
 * add = accumulate into C. Operands go up and C comes back each call (weights
 * change every step). 0 when the GPU did it; -1 when the caller must (no GPU,
 * too big for the device, an error), C untouched. */
int gpu_gemm(int M, int N, int K, const float *A, size_t sm, size_t sk, const float *B, size_t ldb,
             float *C, size_t ldc, int add);

/* Buffers and kernels for training on the device (train.c keeps a dense model's
 * weights, grads, moments and activations in them). An operand is a buffer and
 * an offset in elements; ints and floats are 4 bytes alike. gpu_put and gpu_get
 * wait for everything queued before them; a launch only queues. Any failure
 * turns the GPU off for the process (the device's state is then unknown). */
typedef struct gpu_mem gpu_mem;
enum { GK_GEMM, GK_EMBED, GK_ADD_RMS, GK_RMS_DX, GK_RMS_DW, GK_COLSUM, GK_ROPE, GK_ATT_SM, GK_ATT_SMB,
       GK_SWIGLU_F, GK_SWIGLU_B, GK_CE, GK_EMB_GRAD, GK_SUMSQ, GK_ZERO, GK_SCALE, GK_ADAMW, GK_N };
/* GA_LONG is 64 bits on every host: C's long is 32 on wasm32 and 32-bit phones, and a
 * stride or count past 2^31 there would wrap before it reached the kernel's long */
enum { GA_PTR, GA_INT, GA_FLOAT, GA_LONG };
typedef struct { int t; gpu_mem *m; int64_t off; int i; float f; } gpu_arg;
_Static_assert(sizeof(((gpu_arg *)0)->off) == 8, "gpu_arg offsets and longs are 64 bits on every host");
#define GP(mem, o) ((gpu_arg){ GA_PTR, (mem), (int64_t)(o), 0, 0 })
#define GI(v) ((gpu_arg){ GA_INT, NULL, 0, (int)(v), 0 })
#define GF(v) ((gpu_arg){ GA_FLOAT, NULL, 0, 0, (float)(v) })
#define GL(v) ((gpu_arg){ GA_LONG, NULL, (int64_t)(v), 0, 0 })
gpu_mem *gpu_new(size_t bytes);   /* NULL: no GPU, or past what one buffer may be */
void gpu_del(gpu_mem *b);
int  gpu_put(gpu_mem *b, size_t off, const void *src, size_t n);   /* bytes */
int  gpu_get(gpu_mem *b, size_t off, void *dst, size_t n);
int  gpu_zero(gpu_mem *b, size_t off, size_t n);
/* kernel k over gx x gy x gz items (rounded up to its work-group; a row kernel's gx is
 * rows x 64). GK_GEMM's z is a batch of products: item z is head h = z % zh x hm + h0
 * of sequence z / zh, and each operand X moves by sequence x sXb + (h / dX) x sXh */
int  gpu_launch(int k, size_t gx, size_t gy, size_t gz, int n, const gpu_arg *a);

/* drop the device copies of weights inside [base, base + n): a model being
 * closed, before its mapping (whose addresses a later one may reuse) goes */
void gpu_forget(const void *base, size_t n);

/* the device in use, or NULL when there is none (initializes on first call) */
const char *gpu_name(void);
/* bytes of weights resident on the device */
size_t gpu_resident(void);
/* products the GPU has done in this process (gpu_gemm calls and GK_GEMM launches) */
long gpu_gemm_count(void);
/* 1 when there is a GPU and the kernels failed to build on it (a bug here, not a missing GPU) */
int gpu_broken(void);
