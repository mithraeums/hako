/* tokenizer.h */
#pragma once

/* Encode null-terminated text to token IDs.  Returns token count or -1 on overflow. */
int tok_encode      (const char *text, int *out, int max_tokens);
int tok_encode_bytes(const unsigned char *data, int len, int *out, int max_tokens);

/* Decode token IDs to null-terminated string.  Returns byte count or -1 on overflow. */
int tok_decode(const int *tokens, int n_tokens, char *out, int max_bytes);
