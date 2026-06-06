/* test_api.c — smoke the in-process hakm_api.
 *
 *   ./test_api <model.mlf2> ["user prompt"]
 *
 * Opens a resident session, runs one chat turn through hakm_chat(), streams the
 * reply to stdout via the emit callback. Proves the embed path the agent uses —
 * no subprocess, no socket, no ollama. */
#include "../src/hakm_api.h"
#include <stdio.h>
#include <string.h>

static int emit_cb(const char *chunk, int len, void *ud)
{
	(void)ud;
	fwrite(chunk, 1, len, stdout);
	fflush(stdout);
	return 0;   /* keep going */
}

int main(int argc, char **argv)
{
	if (argc < 2) { fprintf(stderr, "usage: %s <model.mlf2> [prompt]\n", argv[0]); return 1; }
	const char *path = argv[1];
	const char *user = argc > 2 ? argv[2] : "Say hello in one short sentence.";

	hakm_session *s = hakm_session_open(path, 2048);
	if (!s) { fprintf(stderr, "test_api: open failed\n"); return 1; }
	fprintf(stderr, "test_api: arch=%s ctx=%d\n", hakm_session_arch(s), hakm_session_ctxlen(s));

	hakm_msg msgs[] = {
		{ "system", "You are hako, a concise local coding assistant." },
		{ "user",   user },
	};

	hakm_params p; hakm_params_default(&p);
	p.n_new = 64;

	printf("--- reply ---\n");
	int got = hakm_chat(s, msgs, (int)(sizeof msgs / sizeof msgs[0]), &p, emit_cb, NULL);
	printf("\n--- %d tokens ---\n", got);

	hakm_session_close(s);
	return got < 0 ? 1 : 0;
}
