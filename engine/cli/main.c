/*
 * engine/cli/main.c — miniLocal inference CLI.
 *
 * Usage:
 *   mini <model.mlf> [OPTIONS] [PROMPT]
 *
 * Options:
 *   -n <N>       max new tokens (default 200)
 *   -t <float>   temperature, 0=greedy (default 0.8)
 *   -p <float>   top-p / nucleus (default 0=disabled)
 *   -k <int>     top-k (default 0=disabled)
 *   -s <int>     RNG seed (default 0=time)
 *   --info       print model info and exit
 */
#include "../include/mini.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PROMPT_BYTES 4096
#define MAX_OUT_TOKENS   2048

static void usage(const char *prog) {
	fprintf(stderr,
		"usage: %s <model.mlf> [-n N] [-t temp] [-p top_p] [-k top_k]"
		" [-s seed] [--info] [PROMPT]\n",
		prog);
}

int main(int argc, char **argv) {
	if (argc < 2) { usage(argv[0]); return 1; }

	const char *mlf_path = argv[1];
	int n_new        = 200;
	float temperature = 0.8f;
	float top_p       = 0.0f;
	int   top_k       = 0;
	int   seed        = 0;
	int   info_only   = 0;
	char  prompt_buf[MAX_PROMPT_BYTES] = {0};

	/* Parse arguments */
	for (int i = 2; i < argc; i++) {
		if      (!strcmp(argv[i], "-n")     && i+1 < argc) n_new        = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-t")     && i+1 < argc) temperature  = atof(argv[++i]);
		else if (!strcmp(argv[i], "-p")     && i+1 < argc) top_p        = atof(argv[++i]);
		else if (!strcmp(argv[i], "-k")     && i+1 < argc) top_k        = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-s")     && i+1 < argc) seed         = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--info"))                info_only    = 1;
		else {
			/* Remaining args concatenated as prompt */
			size_t cur = strlen(prompt_buf);
			size_t rem = sizeof(prompt_buf) - cur - 1;
			if (cur > 0 && rem > 1) { prompt_buf[cur] = ' '; cur++; rem--; }
			strncat(prompt_buf, argv[i], rem);
		}
	}

	mini_ctx *ctx = mini_open(mlf_path);
	if (!ctx) {
		fprintf(stderr, "mini: failed to open %s\n", mlf_path);
		return 1;
	}

	fprintf(stderr, "%s\n", mini_model_desc(ctx));

	if (info_only) {
		mini_close(ctx);
		return 0;
	}

	/* Encode prompt */
	int prompt_toks[MAX_PROMPT_BYTES];
	int n_prompt = 0;
	if (prompt_buf[0]) {
		n_prompt = mini_encode(ctx, prompt_buf, prompt_toks, MAX_PROMPT_BYTES);
		if (n_prompt < 0) {
			fprintf(stderr, "mini: prompt too long\n");
			mini_close(ctx);
			return 1;
		}
		/* Echo prompt as-is */
		fwrite(prompt_buf, 1, strlen(prompt_buf), stdout);
		fflush(stdout);
	}

	/* If no prompt, use a single newline token to seed generation */
	if (n_prompt == 0) {
		prompt_toks[0] = '\n';
		n_prompt = 1;
	}

	/* Generate */
	mini_sample_opts opts = {
		.temperature = temperature,
		.top_p       = top_p,
		.top_k       = top_k,
		.seed        = (uint64_t)seed,
	};

	int out_toks[MAX_OUT_TOKENS];
	int n_gen = mini_generate(ctx, prompt_toks, n_prompt,
	                          out_toks, n_new, &opts);
	if (n_gen < 0) {
		fprintf(stderr, "\nmini: generation failed\n");
		mini_close(ctx);
		return 1;
	}

	/* Decode and print token by token */
	char byte_buf[2] = {0, 0};
	for (int i = 0; i < n_gen; i++) {
		byte_buf[0] = (char)(out_toks[i] & 0xFF);
		fwrite(byte_buf, 1, 1, stdout);
		fflush(stdout);
	}
	putchar('\n');

	fprintf(stderr, "\n[generated %d tokens]\n", n_gen);
	mini_close(ctx);
	return 0;
}
