/* model.h — forward pass and working buffer types */
#pragma once
#include "loader.h"

typedef struct {
	float *base;       /* single allocation — do not free sub-pointers */
	float *x;          /* [ctx × d_model] */
	float *xn;         /* [ctx × d_model] */
	float *q, *k, *v;  /* [ctx × d_model] */
	float *attn_out;   /* [ctx × d_model] */
	float *ff1, *ff2;  /* [ctx × ffn_dim] */
	float *attn;       /* [n_heads × ctx × ctx] scores */
	float *logits;     /* [vocab] */
	float *tmp_d;      /* [d_model] scratch */
	float *tmp_f;      /* [ffn_dim] scratch */
} model_bufs;

model_bufs *model_bufs_alloc(const mlf_params *p);
void        model_bufs_free (model_bufs *b);

void model_forward(const nano_model *m, const int *tokens, int T,
                   model_bufs *b);
