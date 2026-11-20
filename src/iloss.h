/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* iloss.h — a trainer model's loss worked out in integers, the same number on
 * every machine. The trainer's float forward pass rounds differently from one
 * device to the next (the order of a sum, fused multiply-adds, exp and log
 * libraries, matrix units that are not fp32 at all), so two machines scoring one
 * model agree only to about 1e-5. Here every product is int8 x int8 summed in
 * int32 or wider, and integer addition gives the same bits in any order, on any
 * device, matrix units included:
 *   weights   int8 rows, each with its scale (a 31-bit mantissa and a power of two)
 *   inputs    int8 with one power-of-two exponent per 32 values (OCP MXINT8's layout)
 *   residual  Q16; norms, attention and the ffn Q12; norm weights and rope Q14
 *   rmsnorm   exact integer square root
 *   softmax, silu, cross-entropy  in log2 units, from a 2^-x table and a log2 table
 *   routing   the experts picked by integer scores, ties to the lower index
 * The tables are built once in doubles from + - * / and sqrt, each product
 * rounded on its own, so a compiler that fuses multiply-adds builds the same ones.
 * Close to the float pass (about 3e-4 nats per token on a trained model), not
 * equal to it: a chain that checks with this one checks with nothing else. */
#pragma once
#include "train.h"
#include <stdint.h>

typedef struct il_model il_model;

/* The model's weights as the integer pass holds them; NULL when out of memory.
 * Weights that are not finite count as 0 (NaN) or the row's largest (inf). */
il_model *il_new(tr_model *m);
void      il_free(il_model *I);

/* Summed NLL of n windows into *nll, in 1/65536ths of a bit: window w predicts
 * ids[at[w] + 1 .. at[w] + T] from ids[at[w] .. at[w] + T - 1]. On the engine's
 * threads; the sum is the same however they split it. 0, or -1 when out of
 * memory, T is past ctx, or a token is past the vocab. */
int il_nll(const il_model *I, const int *ids, const int64_t *at, int n, int T, int64_t *nll);

/* nll / tokens in nats per token, one multiply and one divide: the same double
 * everywhere for the same integers. */
double il_nats(int64_t nll, int64_t tokens);
