/* loader.h — mmap an MLF2 file and resolve tensors by name. */
#pragma once
#include <stddef.h>
#include "../include/mlf2.h"

typedef struct {
	mlf2_header        h;
	const mlf2_tensor *tensors;   /* directory, points into the mapping */
	const unsigned char *data;    /* base + h.data_off */
	const unsigned char *tok;     /* base + h.tokenizer_off */
	void  *map;
	size_t map_size;
} hakm_model;

/* Returns 0 on success, -1 on error (message on stderr). */
int  mlf2_open(const char *path, hakm_model *m);
void mlf2_close(hakm_model *m);

/* Find a tensor by exact name, or NULL. */
const mlf2_tensor *mlf2_find(const hakm_model *m, const char *name);

/* Raw pointer to a tensor's data blob. */
const void *mlf2_data(const hakm_model *m, const mlf2_tensor *t);
