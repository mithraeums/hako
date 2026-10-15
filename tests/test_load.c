/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* test_load.c — the loader must refuse a damaged model file, never crash on it.
   A download cut short keeps a valid header; that used to be a SIGSEGV. */
#include "../src/loader.h"
#include "../src/bpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
static void ck(int c, const char *w) { printf("  %s %s\n", c ? "ok  " : "FAIL", w); if (!c) fails++; }

/* header + one tensor + a tokenizer blob, with the tensor's data at `data_end` */
static void write_model(const char *path, uint64_t tensor_bytes, int32_t n_tokens, int32_t tok_len) {
	mlf2_header h; memset(&h, 0, sizeof h);
	memcpy(h.magic, MLF2_MAGIC, 4);
	h.version = MLF2_VERSION; h.n_tensors = 1;
	h.tensors_off = sizeof h;
	h.tokenizer_off = sizeof h + sizeof(mlf2_tensor);
	h.data_off = 512;
	mlf2_tensor t; memset(&t, 0, sizeof t);
	snprintf(t.name, sizeof t.name, "token_embd.weight");
	t.n_dims = 1; t.dims[0] = tensor_bytes; t.nbytes = tensor_bytes;
	unsigned char buf[600]; memset(buf, 0, sizeof buf);
	memcpy(buf, &h, sizeof h);
	memcpy(buf + h.tensors_off, &t, sizeof t);
	unsigned char *k = buf + h.tokenizer_off;
	int32_t nm = 0;
	memcpy(k, &n_tokens, 4); memcpy(k + 4, &nm, 4);
	k[8] = 1; memcpy(k + 9, &tok_len, 4); memcpy(k + 13, "a", 1);
	FILE *f = fopen(path, "wb"); fwrite(buf, 1, sizeof buf, f); fclose(f);   /* data: bytes 512..599 */
}

int main(void) {
	const char *p = "/tmp/hakm-test-load.mlf2";
	hakm_model m;

	write_model(p, 64, 1, 1);
	ck(mlf2_open(p, &m) == 0, "a complete file opens");
	hakm_tok *t = tok_new(&m);
	ck(t != NULL, "and its tokenizer loads");
	tok_free(t); mlf2_close(&m);

	write_model(p, 1u << 30, 1, 1);
	ck(mlf2_open(p, &m) != 0, "a tensor past the end of the file is refused (truncated download)");

	write_model(p, UINT64_MAX - 100, 1, 1);
	ck(mlf2_open(p, &m) != 0, "an offset+size that overflows is refused");

	write_model(p, 64, 1, 1 << 30);
	ck(mlf2_open(p, &m) == 0, "header fine, tokenizer hostile");
	ck(tok_new(&m) == NULL, "a token length past the end is refused, not read");
	mlf2_close(&m);

	write_model(p, 64, -5, 1);
	mlf2_open(p, &m);
	ck(tok_new(&m) == NULL, "a negative token count is refused");
	mlf2_close(&m);

	remove(p);
	printf(fails ? "test_load: %d FAILED\n" : "test_load: all passed\n", fails);
	return fails ? 1 : 0;
}
