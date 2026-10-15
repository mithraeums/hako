# hakm engine — GPLv3 local inference. From-scratch C; no ggml/llama.cpp, no torch.
CC      ?= cc
# ARCH: native tuning by default. For cross / universal builds, override e.g.
#   make ARCH="-arch arm64 -arch x86_64"   (macOS universal2)
#   make ARCH=""                           (portable generic, no -march)
# -march=native and -arch are mutually exclusive — don't pass both.
#
# EXCEPT on 32-bit x86, which in practice means iSh: there the "CPU" is an
# emulator that ADVERTISES features it does not implement, so -march=native
# happily emits AVX/FMA and the engine dies with SIGILL the first time it
# executes one. (That is signal 4 partway through a prefill — it looks like a
# crash in the model code and is not.) A conservative baseline there; every
# other target keeps native tuning.
# Ask the COMPILER what it targets, not uname: uname under an emulator reports
# whatever the emulator feels like, and this decision must not depend on that.
# __i386__ means 32-bit x86, which in practice means iSh.
ARCH_DEFAULT := -march=native
ifneq (,$(shell $(CC) -dM -E - </dev/null 2>/dev/null | grep -c '__i386__' | grep -v '^0$$'))
  ARCH_DEFAULT :=
endif
ARCH    ?= $(ARCH_DEFAULT)
# -std=c11 hides everything that is not ISO C, and on musl (Alpine / iSh) that
# means madvise/MADV_*, popen/pclose and usleep all vanish — the engine simply
# does not compile there. glibc is lax enough to let them through, so this only
# ever showed up on the phone. _DEFAULT_SOURCE asks for the usual POSIX+BSD
# surface on both libcs; it is the standard companion to a strict -std=.
CFLAGS  ?= -O3 -std=c11 -D_DEFAULT_SOURCE -Wall -Wextra -ffast-math -pthread $(ARCH)
LDLIBS  := -lm -lpthread

SRC := src/loader.c src/quant.c src/nn.c src/model.c src/bpe.c src/sample.c src/kvcache.c src/models.c
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
	@echo "hakm: ARCH=$(if $(ARCH),$(ARCH),none (portable baseline))"
	$(CC) $(CFLAGS) -o $@ cli/main.c $(SRC) $(LDLIBS)

# libhakm.a: archive engine + embed API for optional in-process embedding
# (not used by the agent — it spawns the hakm subprocess).
$(LIBA): $(EMBED_OBJ)
	ar rcs $@ $(EMBED_OBJ)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Test binaries build into build/, out of the way of the source. `make test_attn`
# and friends still work: each name is an alias for build/<name>.
B := build
TESTS := test_attn test_kvc test_q8 test_load test_quant test_kdot test_api
TCFLAGS = -O1 -g -std=c11 -D_DEFAULT_SOURCE -Wall -Wextra -pthread $(SAN)

$(B):
	@mkdir -p $(B)

# Attention scores are indexed by position, not by a model dimension. Run under
# ASan — that is what makes the bounds check real.
$(B)/test_attn: tests/test_attn.c src/model.c src/loader.c src/quant.c src/nn.c $(wildcard src/*.h include/*.h) | $(B)
	$(CC) $(TCFLAGS) -o $@ tests/test_attn.c src/loader.c src/quant.c src/nn.c -lm -lpthread

# On-disk KV prefix: round-trip into a synthetic context, refuse a corrupted
# header, a truncated body and a foreign model. ASan, like test_attn.
$(B)/test_kvc: tests/test_kvc.c src/kvcache.c $(wildcard src/*.h include/*.h) | $(B)
	$(CC) $(TCFLAGS) -o $@ tests/test_kvc.c src/kvcache.c -lm -lpthread

# Q8_0 fused dot vs float reference, per block and whole row. ASan + the real
# SIMD path (-march=native), because the kernel is what is under test.
$(B)/test_q8: tests/test_q8.c src/quant.c $(wildcard src/*.h include/*.h) | $(B)
	$(CC) $(TCFLAGS) $(ARCH) -o $@ tests/test_q8.c src/quant.c -lm -lpthread

# A truncated or hostile model file must be refused, not mapped and trusted.
$(B)/test_load: tests/test_load.c src/loader.c src/bpe.c $(wildcard src/*.h include/*.h) | $(B)
	$(CC) $(TCFLAGS) -o $@ tests/test_load.c src/loader.c src/bpe.c -lm -lpthread

# Need real weights, so not part of `make test`:
# dequant a tensor for tests/q_ref.py to check bit-exact
$(B)/test_quant: tests/test_quant.c src/loader.c src/quant.c | $(B)
	$(CC) $(CFLAGS) -o $@ tests/test_quant.c src/loader.c src/quant.c $(LDLIBS)

# the fused Q4_K int8 dot (SIMD + Σx hoist) vs a scalar reference
$(B)/test_kdot: tests/test_kdot.c src/loader.c src/quant.c | $(B)
	$(CC) $(CFLAGS) -o $@ tests/test_kdot.c src/loader.c src/quant.c $(LDLIBS)

# smoke the optional in-process embed API (hakm_chat), not the agent's path
$(B)/test_api: tests/test_api.c $(LIB) | $(B)
	$(CC) $(CFLAGS) -o $@ tests/test_api.c $(LIB) $(LDLIBS)

$(TESTS): %: $(B)/%

# The tests run under ASan+UBSan wherever the compiler can link them. musl
# (Alpine, iSh) ships neither, and forcing them made `make test` fail there.
SAN := $(shell printf 'int main(void){return 0;}\n' | $(CC) -fsanitize=address,undefined -x c - -o /dev/null 2>/dev/null && echo -fsanitize=address,undefined)

# The ASan gates, one command. Built as files, RUN every time: with the run
# inside the build rule, a second `make test` said "nothing to be done" and
# tested nothing.
test: $(B)/test_attn $(B)/test_kvc $(B)/test_q8 $(B)/test_load
	@[ -n "$(SAN)" ] || echo "test: this compiler has no ASan/UBSan — running the tests without them"
	@$(B)/test_attn && $(B)/test_kvc && $(B)/test_q8 && $(B)/test_load

# convert an ollama/HF gguf into our native format
%.mlf2:
	@echo "run: python3 tools/gguf2mlf.py <model.gguf> $@"

clean:
	rm -rf hakm $(B) $(LIBA) $(EMBED_OBJ) $(OBJ)
	rm -rf test_quant test_kdot test_api test_kvc test_attn test_q8 test_load *.dSYM

# The NEON paths, compiled for arm64 with warnings as errors. x86 builds never see that
# code, and v0.0.4's first tag failed on both ARM builders for a missing include.
# Needs a clang that can target arm64 (Apple's can); elsewhere it says so and passes.
arm-check:
	@if ! printf 'int x;' | $(CC) -target arm64-apple-macos11 -x c -c - -o /dev/null 2>/dev/null; then \
		echo "arm-check: this compiler cannot target arm64, skipped"; exit 0; fi; \
	for m in armv8-a armv8.2-a+dotprod; do for f in src/*.c cli/*.c tests/*.c; do \
		$(CC) -target arm64-apple-macos11 -march=$$m -std=c11 -D_DEFAULT_SOURCE -O2 -Wall -Wextra -Werror \
			-Iinclude -Isrc -c $$f -o /dev/null || exit 1; done; done; \
	echo "arm-check: every NEON path compiles for arm64 (armv8-a and +dotprod)"

.PHONY: all clean lib test arm-check $(TESTS)
lib: $(LIBA)
