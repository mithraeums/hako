# hakm engine — GPLv3 local inference. From-scratch C; no ggml/llama.cpp, no torch.
CC      ?= cc
# ARCH: native tuning by default. For cross / universal builds, override e.g.
#   make ARCH="-arch arm64 -arch x86_64"   (macOS universal2)
#   make ARCH=""                           (portable generic, no -march)
# -march=native and -arch are mutually exclusive — don't pass both.
ARCH    ?= -march=native
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -ffast-math -pthread $(ARCH)
LDLIBS  := -lm -lpthread

SRC := src/loader.c src/quant.c src/nn.c src/model.c src/bpe.c src/sample.c
# embed API source: OPTIONAL in-process library. The hako-code agent does NOT use
# this — it spawns the `hakm` CLI as a subprocess. Kept for third-party embedders.
LIB := $(SRC) src/hakm_api.c
OBJ := $(SRC:.c=.o)

# Static lib — an optional embeddable build (libhakm.a + src/hakm_api.h) for anyone
# who wants in-process inference. NOT on the agent's path (it spawns `hakm`).
# Objects carry the engine's own kernel flags.
LIBA      := libhakm.a
EMBED_OBJ := $(LIB:.c=.o)

all: hakm

hakm: cli/main.c $(SRC)
	$(CC) $(CFLAGS) -o $@ cli/main.c $(SRC) $(LDLIBS)

# libhakm.a: archive engine + embed API for optional in-process embedding
# (not used by the agent — it spawns the hakm subprocess).
$(LIBA): $(EMBED_OBJ)
	ar rcs $@ $(EMBED_OBJ)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

test_quant: tests/test_quant.c src/loader.c src/quant.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# smoke the optional in-process embed API (hakm_chat). NB the agent uses the
# hakm subprocess, not this — this only exercises the embeddable library.
test_api: tests/test_api.c $(LIB)
	$(CC) $(CFLAGS) -o $@ tests/test_api.c $(LIB) $(LDLIBS)

# convert an ollama/HF gguf into our native format
%.mlf2:
	@echo "run: python3 tools/gguf2mlf.py <model.gguf> $@"

clean:
	rm -f hakm test_quant test_api $(LIBA) $(EMBED_OBJ) $(OBJ)

.PHONY: all clean lib
lib: $(LIBA)
