/*
 * test_quant.c — dequant a tensor from an MLF2 file and dump floats.
 *
 *   test_quant <model.mlf2> <tensor-name> <nelem> <out.bin>
 *
 * Pairs with q_ref.py, which dequants the same blocks independently in Python
 * and asserts the C output matches. Validates Q4_K / Q6_K bit unpacking on
 * real model weights.
 */
#include "../src/loader.h"
#include "../src/quant.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
	if (argc != 5) {
		fprintf(stderr, "usage: %s <model.mlf2> <tensor> <nelem> <out.bin>\n", argv[0]);
		return 2;
	}
	const char *path = argv[1], *tname = argv[2], *out = argv[4];
	int nelem = atoi(argv[3]);

	hakm_model m;
	if (mlf2_open(path, &m) != 0) return 1;

	const mlf2_tensor *t = mlf2_find(&m, tname);
	if (!t) { fprintf(stderr, "tensor %s not found\n", tname); mlf2_close(&m); return 1; }
	fprintf(stderr, "tensor %s type=%u dims=[%llu,%llu] nbytes=%llu\n",
	        tname, t->type,
	        (unsigned long long)t->dims[0], (unsigned long long)t->dims[1],
	        (unsigned long long)t->nbytes);

	float *buf = malloc((size_t)nelem * sizeof(float));
	if (dequant_tensor((int)t->type, mlf2_data(&m, t), buf, nelem) != 0) {
		fprintf(stderr, "unsupported type %u\n", t->type);
		free(buf); mlf2_close(&m); return 1;
	}

	FILE *f = fopen(out, "wb");
	fwrite(buf, sizeof(float), (size_t)nelem, f);
	fclose(f);

	fprintf(stderr, "first 6: ");
	for (int i = 0; i < 6 && i < nelem; i++) fprintf(stderr, "% .5f ", buf[i]);
	fprintf(stderr, "\n");

	free(buf);
	mlf2_close(&m);
	return 0;
}
