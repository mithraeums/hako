/*
 * tokenizer.c — byte-level tokenizer (MLF v0 / nano).
 *
 * Byte-level: encode = memcpy bytes to int array, decode = bytes back.
 * BPE (MLF v1) will replace this at M2, but the byte-level path stays.
 */
#include "tokenizer.h"
#include <string.h>

int tok_encode(const char *text, int *out, int max_tokens) {
	int n = 0;
	const unsigned char *p = (const unsigned char *)text;
	while (*p) {
		if (n >= max_tokens) return -1;
		out[n++] = (int)*p++;
	}
	return n;
}

int tok_encode_bytes(const unsigned char *data, int len,
                     int *out, int max_tokens) {
	if (len > max_tokens) return -1;
	for (int i = 0; i < len; i++) out[i] = (int)data[i];
	return len;
}

int tok_decode(const int *tokens, int n_tokens, char *out, int max_bytes) {
	if (n_tokens >= max_bytes) return -1;
	for (int i = 0; i < n_tokens; i++) {
		out[i] = (char)(tokens[i] & 0xFF);
	}
	out[n_tokens] = '\0';
	return n_tokens;
}
