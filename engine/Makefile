# hakm engine — proprietary local inference. No ggml/llama.cpp, no torch.
CC      ?= cc
# ARCH: native tuning by default. For cross / universal builds, override e.g.
#   make ARCH="-arch arm64 -arch x86_64"   (macOS universal2)
#   make ARCH=""                           (portable generic, no -march)
# -march=native and -arch are mutually exclusive — don't pass both.
ARCH    ?= -march=native
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -ffast-math -pthread $(ARCH)
LDLIBS  := -lm -lpthread

SRC := src/loader.c src/quant.c src/nn.c src/model.c src/bpe.c src/sample.c
# embed API: the in-process entry the `hako` agent links against (no server).
LIB := $(SRC) src/hakm_api.c
OBJ := $(SRC:.c=.o)

# Static lib — the models project's embeddable deliverable. The hako-code agent
# links this prebuilt archive (+ src/hakm_api.h), so the agent build never reaches
# into engine sources. Objects carry the engine's own kernel flags.
LIBA      := libhakm.a
EMBED_OBJ := $(LIB:.c=.o)

all: hakm

hakm: cli/main.c $(SRC)
	$(CC) $(CFLAGS) -o $@ cli/main.c $(SRC) $(LDLIBS)

# libhakm.a: archive engine + embed API for in-process linking by the agent.
$(LIBA): $(EMBED_OBJ)
	ar rcs $@ $(EMBED_OBJ)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

test_quant: tests/test_quant.c src/loader.c src/quant.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# smoke the in-process embed API (hakm_chat) — same path the agent uses.
test_api: tests/test_api.c $(LIB)
	$(CC) $(CFLAGS) -o $@ tests/test_api.c $(LIB) $(LDLIBS)

# convert an ollama/HF gguf into our native format
%.mlf2:
	@echo "run: python3 tools/gguf2mlf.py <model.gguf> $@"

clean:
	rm -f hakm test_quant test_api $(LIBA) $(EMBED_OBJ) $(OBJ)

.PHONY: all clean lib
lib: $(LIBA)
