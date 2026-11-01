/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/*
 * hakm-train — train a hako model from scratch on the CPU.
 *
 *   hakm-train --data TEXT --out MODEL.mlf2 [options]
 *
 * Trains a byte-level BPE tokenizer on TEXT (or takes one from --tok), then the
 * model, and writes MODEL.mlf2 (f32, arch hako) that hakm runs and quantizes as
 * is. MODEL.mlf2.opt beside it holds the optimizer, so --resume continues a run
 * and ctrl-c saves before it stops. It runs at the lowest priority unless told
 * otherwise, so it gives the cores back to whatever else is running, and --at
 * holds it (and its memory) until a time of day.
 */
#include "../src/train.h"
#include "../src/bpe.h"
#include "../src/loader.h"
#include "../src/nn.h"

#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

static double wall(void) {
	struct timeval t; gettimeofday(&t, NULL);
	return (double)t.tv_sec + (double)t.tv_usec * 1e-6;
}

static volatile sig_atomic_t stop = 0;
static void on_int(int s) { (void)s; stop = 1; signal(SIGINT, SIG_DFL); }

static char *slurp(const char *path, size_t *len) {
	FILE *f = fopen(path, "rb");
	if (!f) { perror(path); return NULL; }
	fseek(f, 0, SEEK_END);
	long n = ftell(f);
	fseek(f, 0, SEEK_SET);
	char *b = n >= 0 ? malloc((size_t)n + 1) : NULL;
	if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "train: cannot read %s\n", path); free(b); fclose(f); return NULL; }
	fclose(f);
	for (long i = 0; i < n; i++) if (!b[i]) b[i] = ' ';   /* the encoder stops at a NUL */
	b[n] = 0;
	*len = (size_t)n;
	return b;
}

/* text -> ids with the engine's own encoder, the one hakm will use. Encoded in pieces cut
 * after a blank line, so a corpus past 2 GB fits the encoder's int lengths */
#define ENC_PIECE ((size_t)64 << 20)
static int *encode(const void *blob, size_t blen, int eos, char *text, size_t len, long *n) {
	hakm_model fm = {0};
	fm.tok = blob; fm.map = (void *)blob; fm.map_size = blen; fm.h.eos_id = (uint32_t)eos;
	hakm_tok *tk = tok_new(&fm);
	size_t cap = len / 2 + 1024, got = 0;
	int *ids = tk ? malloc(cap * sizeof(int)) : NULL, ok = ids != NULL;
	for (size_t at = 0; ok && at < len;) {
		size_t end = len - at > ENC_PIECE ? at + ENC_PIECE : len;
		while (end < len && !(text[end - 1] == '\n' && text[end] == '\n')) end++;
		while (end < len && text[end] == '\n') end++;
		if (cap - got < end - at + 1) {
			size_t nc = cap * 2 > got + (end - at) + 1 ? cap * 2 : got + (end - at) + 1;
			int *nb = realloc(ids, nc * sizeof(int));
			if (!nb) { ok = 0; break; }
			ids = nb; cap = nc;
		}
		char keep = text[end];
		text[end] = 0;
		int k = tok_encode(tk, text + at, ids + got, (int)(end - at) + 1);
		text[end] = keep;
		if (k < 0) ok = 0;
		else got += (size_t)k;
		at = end;
	}
	tok_free(tk);
	if (!ok) { fprintf(stderr, "train: could not encode the text\n"); free(ids); return NULL; }
	int *fit = realloc(ids, (got + 1) * sizeof(int));
	*n = (long)got;
	return fit ? fit : ids;
}

/* sleep until the next HH:MM local time; -1 when it does not parse */
static int wait_until(const char *hm) {
	int h, mi;
	char extra;
	if (sscanf(hm, "%d:%d%c", &h, &mi, &extra) != 2 || h < 0 || h > 23 || mi < 0 || mi > 59) return -1;
	time_t now = time(NULL);
	struct tm t = *localtime(&now);
	t.tm_hour = h; t.tm_min = mi; t.tm_sec = 0;
	time_t at = mktime(&t);
	if (at <= now) { t.tm_mday++; at = mktime(&t); }
	fprintf(stderr, "waiting until %02d:%02d (%.0f min)\n", h, mi, difftime(at, now) / 60);
	while (!stop && (now = time(NULL)) < at) sleep(at - now > 60 ? 60 : (unsigned)(at - now));
	return 0;
}

static void usage(void) {
	fprintf(stderr,
		"usage: hakm-train --data TEXT --out MODEL.mlf2 [options]\n"
		"  --val TEXT        held-out text for the val loss (default: the last 2%% of --data)\n"
		"  --tok MODEL.mlf2  take that model's tokenizer instead of training one\n"
		"  --resume          continue MODEL.mlf2 from MODEL.mlf2.opt (shape and tokenizer from the file)\n"
		"shape:  --vocab N (4096)  --dim N (256)  --layers N (4)  --heads N (4)  --kv-heads N (= heads)\n"
		"        --ffn N (3 x dim)  --ctx N (256)  --theta F (10000)\n"
		"run:    --steps N (2000)  --batch N (16)  --accum N (1)  --lr F (1e-3)  --min-lr F (lr / 10)\n"
		"        --warmup N (100)  --wd F (0.1)  --clip F (1)  --seed N (1)\n"
		"        --eval-every N (200)  --eval-tokens N (65536)  --save-every N (500)  --log-every N (10)\n"
		"system: --threads N (every core)  --nice N (19, the lowest; 0 runs at normal priority)\n"
		"        --at HH:MM  wait until then before loading anything (ctrl-c cancels)\n");
}

int main(int argc, char **argv) {
	const char *data = NULL, *val = NULL, *out = NULL, *tokf = NULL;
	tr_config c = { 4096, 256, 4, 4, 0, 0, 0, 256, 10000.0f, 1e-5f };
	long steps = 2000, warmup = 100, eval_every = 200, eval_tokens = 65536, save_every = 500, log_every = 10;
	int batch = 16, accum = 1, resume = 0, niceness = 19;
	const char *at_time = NULL;
	float lr = 1e-3f, min_lr = -1, wd = 0.1f, clip = 1.0f;
	uint64_t seed = 1;
	for (int i = 1; i < argc; i++) {
		const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
		#define OPT(name) (!strcmp(a, name) && v && ++i)
		if      (OPT("--data")) data = v;
		else if (OPT("--val")) val = v;
		else if (OPT("--out")) out = v;
		else if (OPT("--tok")) tokf = v;
		else if (!strcmp(a, "--resume")) resume = 1;
		else if (OPT("--vocab")) c.vocab = atoi(v);
		else if (OPT("--dim")) c.dim = atoi(v);
		else if (OPT("--layers")) c.layers = atoi(v);
		else if (OPT("--heads")) c.heads = atoi(v);
		else if (OPT("--kv-heads")) c.kv_heads = atoi(v);
		else if (OPT("--ffn")) c.ffn = atoi(v);
		else if (OPT("--ctx")) c.ctx = atoi(v);
		else if (OPT("--theta")) c.rope_theta = (float)atof(v);
		else if (OPT("--steps")) steps = atol(v);
		else if (OPT("--batch")) batch = atoi(v);
		else if (OPT("--accum")) accum = atoi(v);
		else if (OPT("--lr")) lr = (float)atof(v);
		else if (OPT("--min-lr")) min_lr = (float)atof(v);
		else if (OPT("--warmup")) warmup = atol(v);
		else if (OPT("--wd")) wd = (float)atof(v);
		else if (OPT("--clip")) clip = (float)atof(v);
		else if (OPT("--seed")) seed = strtoull(v, NULL, 10);
		else if (OPT("--eval-every")) eval_every = atol(v);
		else if (OPT("--eval-tokens")) eval_tokens = atol(v);
		else if (OPT("--save-every")) save_every = atol(v);
		else if (OPT("--log-every")) log_every = atol(v);
		else if (OPT("--threads")) setenv("HAKM_THREADS", v, 1);
		else if (OPT("--nice")) niceness = atoi(v);
		else if (OPT("--at")) at_time = v;
		else { usage(); return 2; }
		#undef OPT
	}
	if (!data || !out || batch < 1 || accum < 1 || steps < 1) { usage(); return 2; }
	if (min_lr < 0) min_lr = lr / 10;
	if (log_every < 1) log_every = 1;
	if (niceness > 0 && setpriority(PRIO_PROCESS, 0, niceness) != 0) perror("train: setpriority");
	signal(SIGINT, on_int);
	if (at_time && wait_until(at_time) < 0) { fprintf(stderr, "train: --at wants HH:MM, got %s\n", at_time); return 2; }
	if (stop) return 0;
	signal(SIGINT, on_int);

	size_t tlen, vlen = 0;
	char *text = slurp(data, &tlen), *vtext = NULL;
	if (!text) return 1;
	if (val && !(vtext = slurp(val, &vlen))) return 1;

	char optp[4096];
	snprintf(optp, sizeof optp, "%s.opt", out);
	void *blob = NULL; size_t blen = 0; int eos = 0;
	tr_model *m = NULL;
	uint64_t rng = seed * 0x9E3779B97F4A7C15ULL + 1;
	if (resume) {
		if (!(m = tr_load(out, &blob, &blen, &eos)) || tr_load_opt(m, optp, &rng)) return 1;
		c = *tr_cfg(m);
		fprintf(stderr, "resuming %s at step %ld\n", out, *tr_step(m));
	} else {
		if (!c.kv_heads) c.kv_heads = c.heads;
		if (c.heads > 0) c.head_dim = c.dim / c.heads;
		if (!c.ffn) c.ffn = 3 * c.dim;
		if (tokf) {
			hakm_model tm;
			if (mlf2_open(tokf, &tm)) return 1;
			size_t avail = (size_t)((const unsigned char *)tm.map + tm.map_size - tm.tok);
			blen = tok_blob_len(tm.tok, avail);
			blob = blen ? malloc(blen) : NULL;
			if (!blob) { fprintf(stderr, "train: no tokenizer in %s\n", tokf); return 1; }
			memcpy(blob, tm.tok, blen);
			eos = (int)tm.h.eos_id; c.vocab = (int)tm.h.vocab;
			mlf2_close(&tm);
		} else {
			double t0 = wall();
			int nm = bpe_train(text, tlen, c.vocab, &blob, &blen, &eos);
			if (nm < 0) return 1;
			fprintf(stderr, "tokenizer: %d merges, vocab %d, %.1fs\n", nm, c.vocab, wall() - t0);
		}
		const char *bad = tr_check(&c);
		if (c.dim % c.heads) bad = "dim must be a multiple of heads";
		if (bad) { fprintf(stderr, "train: %s\n", bad); return 2; }
		if (!(m = tr_new(&c, seed))) { fprintf(stderr, "train: out of memory for the model\n"); return 1; }
	}

	double t0 = wall();
	long ntr, nval = 0;
	int *ids = encode(blob, blen, eos, text, tlen, &ntr), *vids = NULL, own_vids = vtext != NULL;
	if (!ids) return 1;
	if (vtext) { if (!(vids = encode(blob, blen, eos, vtext, vlen, &nval))) return 1; }
	else { nval = ntr / 50; ntr -= nval; vids = ids + ntr; }
	size_t vbytes = vtext ? vlen : 0;
	if (!vtext) { /* bytes of the held-out tail, for bits per byte */
		hakm_model fm = {0};
		fm.tok = blob; fm.map = blob; fm.map_size = blen; fm.h.eos_id = (uint32_t)eos;
		hakm_tok *tk = tok_new(&fm);
		char buf[256];
		for (long i = 0; tk && i < nval; i++) vbytes += (size_t)tok_decode_id(tk, vids[i], buf, sizeof buf);
		tok_free(tk);
	}
	free(text); free(vtext);
	int T = c.ctx;
	if (ntr < T + 2 || nval < T + 2) { fprintf(stderr, "train: not enough text for ctx %d\n", T); return 1; }
	fprintf(stderr, "data: %ld train tokens, %ld val tokens (%.2f bytes per token), encoded in %.1fs\n",
	        ntr, nval, (double)vbytes / nval, wall() - t0);
	size_t np = tr_n_params(m), emb = (size_t)c.vocab * c.dim;
	fprintf(stderr, "model: %.2fM params (%.2fM without the embedding), dim %d, %d layers, %d/%d heads, ffn %d, ctx %d\n",
	        np / 1e6, (np - emb) / 1e6, c.dim, c.layers, c.heads, c.kv_heads, c.ffn, c.ctx);
	fprintf(stderr, "run: %ld steps of %d x %d x %d tokens, lr %.1e -> %.1e, %d threads, nice %d\n",
	        steps, batch, accum, T, lr, min_lr, hakm_nthreads(), niceness);
	fprintf(stderr, "memory: about %.0f MB (%.0f MB of it the token ids)\n",
	        (tr_mem_bytes(&c, batch, T) + (size_t)(ntr + (own_vids ? nval : 0)) * sizeof(int)) / 1e6,
	        (size_t)(ntr + (own_vids ? nval : 0)) * sizeof(int) / 1e6);

	int *in = malloc((size_t)batch * T * sizeof(int)), *tg = malloc((size_t)batch * T * sizeof(int));
	if (!in || !tg) return 1;
	double tick = wall(), lsum = 0;
	long lcount = 0, toks = 0;
	int rc = 0;
	for (long *step = tr_step(m); *step < steps && !stop;) {
		long s = *step;
		float rate = s < warmup ? lr * (s + 1) / warmup
		           : min_lr + 0.5f * (lr - min_lr) * (1 + cosf(3.14159265f * (s - warmup) / (float)(steps - warmup > 0 ? steps - warmup : 1)));
		tr_zero_grad(m);
		double loss = 0;
		for (int a = 0; a < accum; a++) {
			for (int b = 0; b < batch; b++) {
				uint64_t z = (rng += 0x9E3779B97F4A7C15ULL);
				z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; z ^= z >> 31;
				long at = (long)(z % (uint64_t)(ntr - T - 1));
				memcpy(in + (size_t)b * T, ids + at, T * sizeof(int));
				memcpy(tg + (size_t)b * T, ids + at + 1, T * sizeof(int));
			}
			double l = tr_run(m, in, tg, batch, T, 1.0f / ((float)batch * T * accum));
			if (l < 0) { fprintf(stderr, "train: out of memory for the activations\n"); return 1; }
			loss += l;
		}
		loss /= (double)batch * T * accum;
		if (!isfinite(loss)) { fprintf(stderr, "train: loss is %f at step %ld, stopping without saving\n", loss, s); rc = 1; break; }
		double gn = tr_clip(m, clip);
		tr_adamw(m, rate, wd, 0.9f, 0.95f, 1e-8f);
		lsum += loss; lcount++; toks += (long)batch * T * accum;
		s = *step;
		int last = s == steps || stop;
		if (s % log_every == 0 || last) {
			double now = wall(), tps = toks / (now - tick);
			fprintf(stderr, "step %5ld/%ld  loss %.4f  lr %.2e  grad %.2f  %.0f tok/s  eta %.0fm\n",
			        s, steps, lsum / lcount, rate, gn, tps, (steps - s) * (double)batch * T * accum / tps / 60);
			tick = now; lsum = 0; lcount = 0; toks = 0;
		}
		if ((eval_every > 0 && s % eval_every == 0) || last) {
			double e0 = wall(), nll = 0;
			long nt = 0;
			for (long at = 0; at + T + 1 <= nval && nt < eval_tokens;) {
				int b = 0;
				for (; b < batch && at + T + 1 <= nval; b++, at += T) {
					memcpy(in + (size_t)b * T, vids + at, T * sizeof(int));
					memcpy(tg + (size_t)b * T, vids + at + 1, T * sizeof(int));
				}
				nll += tr_run(m, in, tg, b, T, 0);
				nt += (long)b * T;
			}
			double vl = nll / nt;
			fprintf(stderr, "  val loss %.4f  ppl %.2f  %.3f bits/byte  (%ld tokens, %.1fs)\n",
			        vl, exp(vl), vl / log(2.0) * nval / (double)vbytes, nt, wall() - e0);
			tick += wall() - e0;
		}
		if ((save_every > 0 && s % save_every == 0) || last) {
			if (tr_save(m, out, blob, blen, eos) || tr_save_opt(m, optp, rng)) { rc = 1; break; }
			fprintf(stderr, "  saved %s at step %ld\n", out, s);
		}
	}
	if (stop) fprintf(stderr, "stopped; --resume continues from step %ld\n", *tr_step(m));
	free(in); free(tg); free(ids);
	if (own_vids) free(vids);
	free(blob);
	tr_free(m);
	return rc;
}
