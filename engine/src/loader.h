/* loader.h — internal model types shared across engine source files. */
#pragma once
#include <stddef.h>
#include <stdint.h>

#define NANO_MAX_LAYERS 32

typedef struct {
	uint32_t n_layers;
	uint32_t d_model;
	uint32_t n_heads;
	uint32_t ffn_dim;
	uint32_t ctx;
	uint32_t vocab;
} mlf_params;

typedef struct {
	const float *attn_norm;   /* [d_model] */
	const float *q, *k, *v;  /* [d_model × d_model] each */
	const float *o;           /* [d_model × d_model] */
	const float *ffn_norm;    /* [d_model] */
	const float *gate, *up;   /* [ffn_dim × d_model] each */
	const float *down;        /* [d_model × ffn_dim] */
} layer_weights;

typedef struct {
	mlf_params   p;
	const float *embed;                        /* [vocab × d_model] */
	layer_weights layers[NANO_MAX_LAYERS];
	const float  *final_norm;                  /* [d_model] */
	void         *mmap_base;
	size_t        mmap_size;
} nano_model;

int  mlf_load  (const char *path, nano_model *m);
void mlf_unload(nano_model *m);
