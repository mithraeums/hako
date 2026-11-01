/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* test_train.c — the trainer: every tensor's gradient against a finite
   difference, the engine scoring the saved file the way the trainer does, the
   BPE tokenizer round-tripping through the engine's encoder, a loss that
   falls, and checkpoints that come back bit for bit. ASan. */
#include "../src/train.h"
#include "../src/model.h"
#include "../src/bpe.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
static void ck(int c, const char *w) { printf("  %s %s\n", c ? "ok  " : "FAIL", w); if (!c) fails++; }

static uint64_t rs = 7;
static float frand(void) { rs = rs * 6364136223846793005ULL + 1442695040888963407ULL; return ((rs >> 40) & 0xFFFFFF) / 16777216.0f - 0.5f; }

static const char *TEXT =
	"the box sat on the shelf and the box was old. the old box held a key, and the key opened "
	"the door at the end of the hall. nobody knew who made the box, or the key, or the door.\n"
	"she took the key from the box and walked to the door. the door was locked, the key turned, "
	"and the hall was quiet again. the box stayed on the shelf, older than before.\n";

int main(void) {
	printf("test_train\n");
	char *text = malloc(64 * 1024);
	text[0] = 0;
	for (int i = 0; i < 40; i++) strcat(text, TEXT);
	size_t tlen = strlen(text);

	/* ── tokenizer: trained here, encoded by the engine ── */
	void *blob; size_t blen; int eos;
	int nm = bpe_train(text, tlen, 320, &blob, &blen, &eos);
	ck(nm > 0 && nm <= 320 - 259, "bpe learns merges up to the vocab");
	ck(tok_blob_len(blob, blen) == blen, "blob length reads back from its own counts");
	hakm_model fm = {0};
	fm.tok = blob; fm.map = blob; fm.map_size = blen; fm.h.eos_id = eos;
	hakm_tok *tk = tok_new(&fm);
	ck(tk != NULL, "the engine loads the blob");
	int *ids = malloc(tlen * sizeof(int));
	int n = tk ? tok_encode(tk, text, ids, (int)tlen) : -1;
	printf("    %zu bytes -> %d tokens with %d merges\n", tlen, n, nm);
	ck(n > 0 && n < (int)tlen / 2, "the engine's encoder uses the merges (half as many tokens as bytes)");
	char *back = malloc(tlen + 1);
	size_t bl = 0;
	for (int i = 0; i < n; i++) bl += tok_decode_id(tk, ids[i], back + bl, (int)(tlen - bl));
	back[bl] = 0;
	ck(bl == tlen && !strcmp(back, text), "encode then decode gives the text back");
	ck(tk && tok_id(tk, "<|endoftext|>") == eos && tok_id(tk, "<|im_end|>") == eos + 2, "specials after the merges");
	{	/* a text with few pairs runs out of merges: pads fill the rest */
		void *pb; size_t pl; int pe;
		int pm = bpe_train("abab abab", 9, 300, &pb, &pl, &pe);
		hakm_model pf = {0};
		pf.tok = pb; pf.map = pb; pf.map_size = pl; pf.h.eos_id = pe;
		hakm_tok *pt = pm > 0 ? tok_new(&pf) : NULL;
		ck(pt && pe == 256 + pm && tok_id(pt, "[PAD299]") == 299, "pads fill the vocab when the pairs run out");
		tok_free(pt); free(pb);
	}

	/* ── gradients: directional finite difference, per tensor ── */
	tr_config c = { 320, 16, 2, 4, 2, 8, 24, 12, 10000.0f, 1e-5f };
	ck(tr_check(&c) == NULL, "the test shape is valid (GQA 4/2)");
	tr_model *m = tr_new(&c, 3);
	float *p = tr_params(m);
	size_t np = tr_n_params(m);
	for (size_t i = 0; i < np; i++) p[i] = p[i] == 1.0f ? 1.0f + 0.4f * frand() : 0.8f * frand();
	enum { B = 2, T = 12 };
	int in[B * T], tg[B * T];
	for (int i = 0; i < B * T; i++) { in[i] = ids[i * 3]; tg[i] = ids[i * 3 + 1]; }
	tr_zero_grad(m);
	double l0 = tr_run(m, in, tg, B, T, 1.0f);
	ck(l0 > 0, "forward + backward runs");
	float *g = malloc(np * sizeof(float)), *dir = malloc(np * sizeof(float));
	memcpy(g, tr_grads(m), np * sizeof(float));
	/* tensor spans by the layout tr_new uses */
	int D = 16, Q = 32, KV = 16, F = 24, V = 320;
	size_t span[2 + 18], at = 0; int ns = 0;
	span[ns++] = (size_t)V * D;
	for (int l = 0; l < 2; l++) {
		size_t s[9] = { D, (size_t)Q * D, (size_t)KV * D, (size_t)KV * D, (size_t)D * Q, D, (size_t)F * D, (size_t)F * D, (size_t)D * F };
		for (int i = 0; i < 9; i++) span[ns++] = s[i];
	}
	span[ns++] = D;
	double worst = 0;
	for (int t = 0; t < ns; t++) {
		memset(dir, 0, np * sizeof(float));
		double an = 0;
		for (size_t i = at; i < at + span[t]; i++) { dir[i] = frand(); an += (double)dir[i] * g[i]; }
		const double h = 1e-3;
		p = tr_params(m);
		for (size_t i = at; i < at + span[t]; i++) p[i] += (float)h * dir[i];
		double lp = tr_run(m, in, tg, B, T, 0);
		p = tr_params(m);
		for (size_t i = at; i < at + span[t]; i++) p[i] -= 2 * (float)h * dir[i];
		double lm = tr_run(m, in, tg, B, T, 0);
		p = tr_params(m);
		for (size_t i = at; i < at + span[t]; i++) p[i] += (float)h * dir[i];
		double fd = (lp - lm) / (2 * h), rel = fabs(fd - an) / (fabs(an) + 1e-3);
		if (rel > worst) worst = rel;
		if (rel > 2e-2) printf("    tensor %d: analytic %.6f finite %.6f\n", t, an, fd);
		at += span[t];
	}
	printf("    worst relative error %.2e over %d tensors\n", worst, ns);
	ck(worst < 2e-2, "every tensor's gradient matches its finite difference");

	/* ── the engine scores the saved file as the trainer does ── */
	const char *path = "build/test_train.mlf2";
	ck(tr_save(m, path, blob, blen, eos) == 0, "save as MLF2");
	hakm_ctx *e = hakm_ctx_new(path, c.ctx + 1);
	ck(e != NULL, "hakm loads it (arch hako, f32)");
	int seq[T + 1];
	for (int i = 0; i <= T; i++) seq[i] = ids[100 + i];
	double eng = e ? hakm_score_batch(e, seq, T + 1, 0) : -1;
	double trn = tr_run(m, seq, seq + 1, 1, T, 0);
	printf("    nll: trainer %.6f engine %.6f\n", trn, eng);
	ck(e && fabs(eng - trn) < 2e-3 * trn, "same NLL within the f16 KV cache's rounding");
	hakm_ctx_free(e);

	/* ── checkpoints ── */
	tr_adamw(m, 1e-3f, 0.1f, 0.9f, 0.95f, 1e-8f);
	ck(tr_save(m, path, blob, blen, eos) == 0 && tr_save_opt(m, "build/test_train.opt", 99) == 0, "save weights + optimizer");
	void *b2; size_t bl2; int eos2; uint64_t rng = 0;
	tr_model *r = tr_load(path, &b2, &bl2, &eos2);
	ck(r && !memcmp(tr_params(r), tr_params(m), np * sizeof(float)), "weights come back bit for bit");
	ck(r && bl2 == blen && !memcmp(b2, blob, blen) && eos2 == eos, "tokenizer and eos come back");
	ck(r && tr_load_opt(r, "build/test_train.opt", &rng) == 0 && *tr_step(r) == 1 && rng == 99, "optimizer state comes back");
	if (r) { free(b2); tr_free(r); }
	remove(path); remove("build/test_train.opt");

	/* ── it learns: one batch, many steps ── */
	tr_free(m);
	m = tr_new(&c, 5);
	double first = 0, last = 0;
	for (int s = 0; s < 60; s++) {
		tr_zero_grad(m);
		double l = tr_run(m, in, tg, B, T, 1.0f / (B * T));
		tr_clip(m, 1.0);
		tr_adamw(m, 1e-2f, 0.0f, 0.9f, 0.95f, 1e-8f);
		if (!s) first = l; last = l;
	}
	printf("    loss %.3f -> %.3f per token\n", first / (B * T), last / (B * T));
	ck(last < 0.2 * first, "60 steps on one batch cut its loss by 5x");
	tr_free(m);

	tok_free(tk);
	free(blob); free(ids); free(back); free(text); free(g); free(dir);
	printf(fails ? "test_train: %d FAILED\n" : "test_train: all ok\n", fails);
	return fails != 0;
}
