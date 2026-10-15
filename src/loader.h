/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
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

/* Fault the whole mapping in. mmap is lazy, so the FIRST forward pass pays for
 * reading every weight off disk — measured 8s of the first turn on a 1.8GB
 * model with a cold page cache, and nothing at all when it is warm. Call this
 * off the critical path (a thread at startup) so the read overlaps the user
 * typing instead of landing inside their first turn. Returns bytes touched. */
size_t mlf2_prefault(const hakm_model *m);
