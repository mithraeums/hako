/* hakm_api.c — in-process inference API. See hakm_api.h.
 *
 * Wraps the resident engine (model + tokenizer) behind a stateless chat call so
 * a host program (the `hako` agent) can generate locally with no subprocess,
 * socket, server, or ollama. The conversation is re-rendered as ChatML each
 * call; the KV cache is reset per call (caller owns history).
 */
#include "hakm_api.h"
#include "model.h"
#include "bpe.h"
#include "sample.h"

#include <stdlib.h>
#include <string.h>

struct hakm_session {
	hakm_ctx  *ctx;
	hakm_tok  *tok;
	int        im_start, im_end;   /* ChatML control ids, or -1 */
	int        max_seq;
};

void hakm_params_default(hakm_params *p)
{
	if (!p) return;
	p->n_new = 512;
	p->temp  = 0.7f;
	p->top_p = 0.9f;
	p->top_k = 40;
	p->seed  = 0;
}

hakm_session *hakm_session_open(const char *mlf2_path, int max_seq)
{
	if (max_seq <= 0) max_seq = 2048;
	hakm_session *s = calloc(1, sizeof *s);
	if (!s) return NULL;

	s->ctx = hakm_ctx_new(mlf2_path, max_seq);
	if (!s->ctx) { free(s); return NULL; }

	s->tok = tok_new(&s->ctx->model);
	if (!s->tok) { hakm_ctx_free(s->ctx); free(s); return NULL; }

	s->im_start = tok_id(s->tok, "<|im_start|>");
	s->im_end   = tok_id(s->tok, "<|im_end|>");
	s->max_seq  = max_seq;
	return s;
}

void hakm_session_close(hakm_session *s)
{
	if (!s) return;
	tok_free(s->tok);
	hakm_ctx_free(s->ctx);
	free(s);
}

const char *hakm_session_arch(const hakm_session *s)
{
	(void)s;
	return "qwen2";   /* only arch the engine implements today */
}

int hakm_session_ctxlen(const hakm_session *s)
{
	return s ? s->max_seq : 0;
}

/* ── token buffer helpers (mirror cli/main.c) ── */

static int push_text(const hakm_tok *tk, const char *str, int *buf, int *n, int max)
{
	int got = tok_encode(tk, str, buf + *n, max - *n);
	if (got < 0) return -1;
	*n += got;
	return 0;
}

static int push_id(int id, int *buf, int *n, int max)
{
	if (id < 0 || *n >= max) return -1;
	buf[(*n)++] = id;
	return 0;
}

/* Render one ChatML turn into buf. open=1 leaves the assistant header dangling
 * (no closing <|im_end|>) so generation continues from there. */
static int chatml_turn(const hakm_tok *tk, int im_start, int im_end,
                       const char *role, const char *text, int open,
                       int *buf, int *n, int max)
{
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

/* Feed prompt tokens; return last logits (held in c->logits). */
static const float *feed(hakm_ctx *c, const int *toks, int n)
{
	const float *lg = NULL;
	for (int i = 0; i < n && c->pos < c->max_seq; i++)
		lg = hakm_forward(c, toks[i]);
	return lg;
}

int hakm_chat(hakm_session *s, const hakm_msg *msgs, int n_msgs,
              const hakm_params *p,
              int (*emit)(const char *chunk, int len, void *ud), void *ud)
{
	if (!s || !msgs || n_msgs <= 0) return -1;
	if (s->im_start < 0 || s->im_end < 0) return -1;   /* needs ChatML model */

	hakm_params dp;
	if (!p) { hakm_params_default(&dp); p = &dp; }

	hakm_ctx *c = s->ctx;
	hakm_tok *tk = s->tok;
	int max = s->max_seq;

	/* Stateless: rebuild the whole conversation each call. */
	hakm_reset(c);

	int *buf = malloc((size_t)max * sizeof(int));
	if (!buf) return -1;
	int n = 0;

	for (int i = 0; i < n_msgs; i++) {
		const char *role = msgs[i].role ? msgs[i].role : "user";
		if (chatml_turn(tk, s->im_start, s->im_end, role,
		                msgs[i].content, 0, buf, &n, max)) {
			free(buf); return -1;
		}
	}
	/* open assistant header for the model to complete */
	if (chatml_turn(tk, s->im_start, s->im_end, "assistant", NULL, 1, buf, &n, max)) {
		free(buf); return -1;
	}

	if (c->pos + n >= max) { free(buf); return -1; }   /* prompt over budget */

	feed(c, buf, n);
	free(buf);

	sampler sp;
	sampler_init(&sp, p->temp, p->top_p, p->top_k, p->seed);

	int eos = tok_eos(tk);
	int produced = 0;
	char piece[16];
	for (int g = 0; g < p->n_new && c->pos < c->max_seq; g++) {
		int id = sample_token(&sp, c->logits, c->vocab);
		if (id == eos) break;
		int w = tok_decode_id(tk, id, piece, sizeof piece);
		if (w > 0 && emit) {
			if (emit(piece, w, ud)) { produced++; break; }   /* caller abort */
		}
		produced++;
		hakm_forward(c, id);
	}
	return produced;
}
