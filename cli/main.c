/*
 * hakm — hako native inference CLI.  No ollama, no llama.cpp.
 *
 *   hakm <model.mlf2> [opts] [PROMPT...]
 *
 * Modes:
 *   PROMPT given, --raw   : raw text completion of PROMPT
 *   PROMPT given          : single ChatML turn (system optional via --sys)
 *   no PROMPT             : interactive multi-turn chat REPL (KV cache reused)
 *
 * Options:
 *   -n N        max new tokens per reply (default 128)
 *   -t F        temperature (default 0.7; 0 = greedy)
 *   -p F        top-p nucleus (default 0.9)
 *   -k N        top-k (default 40)
 *   -s N        rng seed (default time)
 *   --sys TEXT  system prompt (chat modes)
 *   --raw       raw completion instead of ChatML
 *   --info      print model info and exit
 *   --chat-stdin   one-shot: read a framed conversation from stdin, render
 *                  ChatML, print ONLY the reply to stdout (stats to stderr).
 *                  Wire for the hako agent's subprocess transport. Frame:
 *                    N\n                      (message count)
 *                    role\n len\n <len bytes>  (repeated N times)
 *   --serve     resident: serve framed conversations from stdin in a loop,
 *                  reusing the KV cache across requests (prefix-matched, only
 *                  the unseen tail is prefilled). Request = chat-stdin frame
 *                  with an optional per-request token cap: "N [n_new]\n" then
 *                  N x ( role\n len\n <len bytes> ). Response on stdout:
 *                    R <len>\n <len bytes>   (assistant reply)
 *                    E <len>\n <len bytes>   (error; KV cache was reset)
 *                  Exits on EOF / desynced frame. Stats land on stderr.
 *   --ctx N     context window in tokens (default 2048)
 */
#include "../src/model.h"
#include "../src/bpe.h"
#include "../src/sample.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

static double wall(void) {
	struct timeval t; gettimeofday(&t, NULL);
	return (double)t.tv_sec + (double)t.tv_usec * 1e-6;
}

/* append BPE-encoded text to buf at *n; returns 0 ok, -1 overflow. */
static int push_text(const hakm_tok *tk, const char *s, int *buf, int *n, int max) {
	int got = tok_encode(tk, s, buf + *n, max - *n);
	if (got < 0) return -1;
	*n += got;
	return 0;
}
static int push_id(int id, int *buf, int *n, int max) {
	if (id < 0 || *n >= max) return -1;
	buf[(*n)++] = id;
	return 0;
}

/* Feed tokens through the model in batched chunks; return last logits.
 * Batched prefill streams each weight row once per chunk instead of once per
 * token (HAKM_BATCH overrides the chunk size; decode stays token-at-a-time). */
static const float *feed(hakm_ctx *c, const int *toks, int n) {
	static int B = 0;
	if (!B) {
		const char *e = getenv("HAKM_BATCH");
		B = (e && atoi(e) > 0) ? atoi(e) : 16;
	}
	const float *lg = NULL;
	for (int i = 0; i < n && c->pos < c->max_seq; ) {
		int chunk = n - i < B ? n - i : B;
		const float *t = hakm_forward_batch(c, toks + i, chunk);
		if (!t) break;
		lg = t;
		i += chunk;
	}
	return lg;
}

/* Sample + stream a reply from the given starting logits. Stops on eos or cap.
 * Returns tokens actually generated (callers report HONEST tok/s — counting
 * the requested cap inflated the rate ~4x whenever eos came early). */
static int generate(hakm_ctx *c, hakm_tok *tk, sampler *sp,
                    const float *logits, int n_new) {
	char buf[16];
	int g = 0;
	sampler_reset_hist(sp);
	for (; g < n_new && c->pos < c->max_seq; g++) {
		int id = sample_token(sp, c->logits, c->vocab);
		(void)logits;
		if (id == tok_eos(tk)) break;
		int w = tok_decode_id(tk, id, buf, sizeof(buf));
		fwrite(buf, 1, w, stdout); fflush(stdout);
		hakm_forward(c, id);
	}
	putchar('\n');
	return g;
}

/* One-shot prefill+decode with HONEST split timing. The combined (n+gen)/wall
 * figure hid a slow decode behind a fast prefill (or vice-versa); report the
 * prefill and decode rates separately, like the --serve path already does. */
static void run_oneshot(hakm_ctx *c, hakm_tok *tk, sampler *sp,
                        const int *buf, int n, int n_new) {
	double t0 = wall();
	const float *lg = feed(c, buf, n);
	double t1 = wall();
	int gen = generate(c, tk, sp, lg, n_new);
	double t2 = wall();
	double pt = t1 - t0, dt = t2 - t1;
	fprintf(stderr,
	        "[%.1fs wall: %d prefilled (%.1fs, %.2f tok/s) + %d generated (%.1fs, %.2f tok/s)]\n",
	        t2 - t0, n, pt, pt > 0 ? n / pt : 0.0,
	        gen, dt, dt > 0 ? gen / dt : 0.0);
}

/* Build a ChatML turn into buf: [im_start]role\n{text}[im_end]\n , optionally
 * leaving the assistant header open for generation. */
static int chatml_turn(const hakm_tok *tk, int im_start, int im_end,
                       const char *role, const char *text, int open,
                       int *buf, int *n, int max) {
	if (push_id(im_start, buf, n, max)) return -1;
	if (push_text(tk, role, buf, n, max)) return -1;
	if (push_text(tk, "\n", buf, n, max)) return -1;
	if (text && push_text(tk, text, buf, n, max)) return -1;
	if (!open) {
		if (push_id(im_end, buf, n, max)) return -1;
		if (push_text(tk, "\n", buf, n, max)) return -1;
	}
	return 0;
}

#ifndef HAKM_VERSION
#define HAKM_VERSION "0.0.3"
#endif

int main(int argc, char **argv) {
	if (argc >= 2 && (!strcmp(argv[1], "--version") || !strcmp(argv[1], "-v"))) {
		printf("hakm v%s\n", HAKM_VERSION);
		return 0;
	}
	if (argc < 2) {
		fprintf(stderr, "usage: %s <model.mlf2> [-n N -t F -p F -k N --rep F --rep-window N -s N --sys TEXT --raw --info --serve --chat-stdin] [PROMPT]\n", argv[0]);
		return 1;
	}
	const char *path = argv[1], *sys = NULL;
	int n_new = 128, top_k = 40, info = 0, raw = 0, chat_stdin = 0, serve = 0;
	int max_seq = 2048;
	float temp = 0.7f, top_p = 0.9f, rep = 1.0f;
	int rep_win = 64;
	uint64_t seed = (uint64_t)time(NULL);
	char prompt[8192] = {0};

	for (int i = 2; i < argc; i++) {
		if      (!strcmp(argv[i], "-n") && i + 1 < argc) n_new = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-t") && i + 1 < argc) temp  = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "-p") && i + 1 < argc) top_p = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "-k") && i + 1 < argc) top_k = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--rep") && i + 1 < argc) rep = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "--rep-window") && i + 1 < argc) rep_win = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed  = strtoull(argv[++i], NULL, 10);
		else if (!strcmp(argv[i], "--sys") && i + 1 < argc) sys = argv[++i];
		else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) max_seq = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--raw"))  raw = 1;
		else if (!strcmp(argv[i], "--chat-stdin")) chat_stdin = 1;
		else if (!strcmp(argv[i], "--serve")) serve = 1;
		else if (!strcmp(argv[i], "--info")) info = 1;
		else { if (prompt[0]) strncat(prompt, " ", sizeof(prompt) - strlen(prompt) - 1);
		       strncat(prompt, argv[i], sizeof(prompt) - strlen(prompt) - 1); }
	}

	if (max_seq <= 0) max_seq = 2048;
	hakm_ctx *c = hakm_ctx_new(path, max_seq);
	if (!c) { fprintf(stderr, "hakm: failed to load %s\n", path); return 1; }

	/* Total parameter count = sum of element counts over every tensor in the
	   directory (product of its dims). Reported in billions like the model name. */
	double params = 0.0;
	for (uint32_t t = 0; t < c->model.h.n_tensors; t++) {
		const mlf2_tensor *w = &c->model.tensors[t];
		double ne = 1.0;
		for (uint32_t dd = 0; dd < w->n_dims; dd++) ne *= (double)w->dims[dd];
		params += ne;
	}

	fprintf(stderr,
	        "hakm: hako engine · arch qwen2 · %.2fB params · %d layers · "
	        "d_model %d · heads %d/%d (GQA) · ffn %d · vocab %d · ctx %d · rope_theta %.0f\n",
	        params / 1e9, c->n_layers, c->d_model, c->n_heads, c->n_kv_heads,
	        c->ffn_dim, c->vocab, max_seq, (double)c->rope_theta);
	if (info) { hakm_ctx_free(c); return 0; }

	hakm_tok *tk = tok_new(&c->model);
	sampler sp; sampler_init(&sp, temp, top_p, top_k, seed);
	sampler_set_rep(&sp, rep, rep_win);
	int im_start = tok_id(tk, "<|im_start|>");
	int im_end   = tok_id(tk, "<|im_end|>");
	int chat_ok  = (im_start >= 0 && im_end >= 0);

	int *buf = malloc(max_seq * sizeof(int));

	/* ── raw completion ── */
	if (raw && prompt[0]) {
		int n = 0;
		if (push_text(tk, prompt, buf, &n, max_seq)) { fprintf(stderr, "encode failed\n"); return 1; }
		fputs(prompt, stdout); fflush(stdout);
		run_oneshot(c, tk, &sp, buf, n, n_new);
		goto done;
	}

	if (!chat_ok) { fprintf(stderr, "hakm: no ChatML special tokens; use --raw\n"); goto done; }

	/* ── resident serve loop (persistent agent transport, KV cache reused) ──
	   Same frame as --chat-stdin, repeated until EOF, with an optional
	   per-request token cap on the header line ("N n_new\n"). Each request is
	   the FULL conversation; we prefix-match its rendered tokens against what
	   is already in the KV cache and prefill only the unseen tail — the system
	   prompt and prior turns cost nothing after the first request. Reply is
	   length-prefixed ("R <len>\n<bytes>"); errors use "E". */
	if (serve) {
		int *kv_toks = malloc(max_seq * sizeof(int));
		int kv_n = 0;	/* tokens whose KV entries are valid; == c->pos between requests */
		char hdr[64];
		while (kv_toks && fgets(hdr, sizeof(hdr), stdin)) {
			int nmsgs = 0, req_new = 0, mt = -1;
			if (sscanf(hdr, "%d %d %d", &nmsgs, &req_new, &mt) < 1 || nmsgs <= 0) break;
			if (req_new <= 0 || req_new > 65536) req_new = n_new;
			/* Optional 3rd header field = per-request temperature in millis
			   (mt/1000). The agent sends 0 on tool turns to force greedy
			   sampling — same prompt → same tool call, every time. Absent (mt<0,
			   old frames / --chat-stdin) leaves the CLI -t default untouched. */
			if (mt >= 0) sp.temp = mt / 1000.0f;

			int n = 0, overflow = 0, sync_lost = 0;
			for (int i = 0; i < nmsgs; i++) {
				char role[64];
				if (!fgets(role, sizeof(role), stdin)) { sync_lost = 1; break; }
				size_t rl = strlen(role); while (rl && (role[rl-1] == '\n' || role[rl-1] == '\r')) role[--rl] = 0;
				if (!fgets(hdr, sizeof(hdr), stdin)) { sync_lost = 1; break; }
				long clen = atol(hdr);
				if (clen < 0) clen = 0;
				char *content = malloc((size_t)clen + 1);
				if (!content) { sync_lost = 1; break; }
				size_t got = clen ? fread(content, 1, (size_t)clen, stdin) : 0;
				content[got] = 0;
				if (got != (size_t)clen) { free(content); sync_lost = 1; break; }
				/* keep consuming the frame even after overflow so the stream stays synced */
				if (!overflow && chatml_turn(tk, im_start, im_end, role, content, 0, buf, &n, max_seq))
					overflow = 1;
				free(content);
			}
			if (sync_lost) break;
			if (!overflow && chatml_turn(tk, im_start, im_end, "assistant", NULL, 1, buf, &n, max_seq))
				overflow = 1;
			if (overflow) {
				const char *msg = "conversation exceeds context window — trim history or raise --ctx";
				printf("E %zu\n", strlen(msg)); fwrite(msg, 1, strlen(msg), stdout); fflush(stdout);
				hakm_reset(c); kv_n = 0;
				continue;
			}

			/* Prefix match, then roll the cache back to the divergence point.
			   Clamp to n-1 so at least the last token is fed — generation needs
			   its logits fresh. */
			int p = 0;
			while (p < kv_n && p < n - 1 && kv_toks[p] == buf[p]) p++;
			c->pos = p;
			double t0 = wall();
			for (int i = p; i < n; i++) kv_toks[i] = buf[i];
			feed(c, buf + p, n - p);   /* batched prefill of the unseen tail */
			kv_n = n;
			double t1 = wall();

			/* Generate into a buffer; the reply ships length-prefixed in one piece. */
			size_t ocap = 4096, olen = 0;
			char *out = malloc(ocap);
			int gen = 0;
			char piece[16];
			sampler_reset_hist(&sp);
			while (out && gen < req_new && c->pos < c->max_seq) {
				int id = sample_token(&sp, c->logits, c->vocab);
				if (id == tok_eos(tk)) break;
				int w = tok_decode_id(tk, id, piece, sizeof(piece));
				if (olen + (size_t)w >= ocap) {
					char *t = realloc(out, ocap * 2);
					if (!t) break;
					out = t; ocap *= 2;
				}
				memcpy(out + olen, piece, (size_t)w); olen += (size_t)w;
				hakm_forward(c, id);
				kv_toks[kv_n++] = id;	/* keep kv_toks == fed tokens; kv_n tracks c->pos */
				gen++;
				/* Stop the moment a tool call closes. The agent's Qwen dialect is
				   one <tool_call>…</tool_call> per turn: emit it, hand off, wait
				   for the observation. Stopping here means a call can never be
				   truncated mid-JSON by req_new, and no wasted "now run it in
				   your terminal" epilogue burns tokens after the call.
				   </write_file> is the raw-body file-write channel (no JSON
				   escaping of the file content) — same one-shot, hand-off rule. */
				if ((olen >= 12 && memcmp(out + olen - 12, "</tool_call>",  12) == 0) ||
				    (olen >= 13 && memcmp(out + olen - 13, "</write_file>", 13) == 0)) break;
			}
			printf("R %zu\n", olen);
			if (out) fwrite(out, 1, olen, stdout);
			fflush(stdout);
			free(out);
			double t2 = wall();
			fprintf(stderr, "[serve: %d cached + %d prefilled (%.1fs) + %d generated (%.1fs, %.2f tok/s)]\n",
			        p, n - p, t1 - t0, gen, t2 - t1, gen > 0 ? gen / (t2 - t1) : 0.0);
		}
		free(kv_toks);
		goto done;
	}

	/* ── one-shot framed conversation from stdin (agent subprocess wire) ──
	   Frame: "N\n" then N × ( "role\n" "len\n" <len bytes> ). Reply → stdout
	   only; the agent reads stdout, stats land on stderr. */
	if (chat_stdin) {
		char hdr[64];
		if (!fgets(hdr, sizeof(hdr), stdin)) { fprintf(stderr, "hakm: empty stdin\n"); goto done; }
		int nmsgs = atoi(hdr);
		if (nmsgs <= 0) { fprintf(stderr, "hakm: no messages\n"); goto done; }

		int n = 0, overflow = 0;
		for (int i = 0; i < nmsgs; i++) {
			char role[64];
			if (!fgets(role, sizeof(role), stdin)) { fprintf(stderr, "hakm: short frame (role)\n"); goto done; }
			size_t rl = strlen(role); while (rl && (role[rl-1] == '\n' || role[rl-1] == '\r')) role[--rl] = 0;

			if (!fgets(hdr, sizeof(hdr), stdin)) { fprintf(stderr, "hakm: short frame (len)\n"); goto done; }
			long clen = atol(hdr);
			if (clen < 0) clen = 0;

			char *content = malloc((size_t)clen + 1);
			if (!content) { fprintf(stderr, "hakm: oom\n"); goto done; }
			size_t got = clen ? fread(content, 1, (size_t)clen, stdin) : 0;
			content[got] = 0;

			if (chatml_turn(tk, im_start, im_end, role, content, 0, buf, &n, max_seq)) overflow = 1;
			free(content);
			if (overflow) break;   /* ran past ctx — generate from what fit */
		}
		if (chatml_turn(tk, im_start, im_end, "assistant", NULL, 1, buf, &n, max_seq)) {
			fprintf(stderr, "hakm: prompt over ctx budget\n"); goto done;
		}
		if (overflow) fprintf(stderr, "hakm: conversation truncated to ctx %d\n", max_seq);

		run_oneshot(c, tk, &sp, buf, n, n_new);
		goto done;
	}

	/* ── single-shot chat ── */
	if (prompt[0]) {
		int n = 0;
		if (sys) chatml_turn(tk, im_start, im_end, "system", sys, 0, buf, &n, max_seq);
		chatml_turn(tk, im_start, im_end, "user", prompt, 0, buf, &n, max_seq);
		chatml_turn(tk, im_start, im_end, "assistant", NULL, 1, buf, &n, max_seq);
		run_oneshot(c, tk, &sp, buf, n, n_new);
		goto done;
	}

	/* ── interactive multi-turn REPL (KV cache reused across turns) ── */
	fprintf(stderr, "hakm chat — multi-turn, ctx %d. blank line or EOF to quit.\n", max_seq);
	int first = 1;
	char line[8192];
	for (;;) {
		fputs("\n> ", stderr);
		if (!fgets(line, sizeof(line), stdin)) break;
		size_t L = strlen(line);
		if (L && line[L - 1] == '\n') line[--L] = 0;
		if (L == 0) break;

		int n = 0;
		if (first && sys) chatml_turn(tk, im_start, im_end, "system", sys, 0, buf, &n, max_seq);
		first = 0;
		chatml_turn(tk, im_start, im_end, "user", line, 0, buf, &n, max_seq);
		chatml_turn(tk, im_start, im_end, "assistant", NULL, 1, buf, &n, max_seq);

		if (c->pos + n >= max_seq) { fprintf(stderr, "[context full — restarting]\n"); hakm_reset(c); first = 1; }
		const float *lg = feed(c, buf, n);
		generate(c, tk, &sp, lg, n_new);
	}

done:
	free(buf);
	tok_free(tk);
	hakm_ctx_free(c);
	return 0;
}
