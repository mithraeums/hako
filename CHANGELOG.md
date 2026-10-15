# Changelog

All notable changes to the hako engine (`hakm`).

## [v0.0.4] - 2026-10-15

ships together with hako-code v0.2.4 (token streaming and the model paths span both).

### changed: the new look (2026-10-12)
- README banner and badges in the new inks (the suite's new look: two inks, ultramarine and vermilion, on washi, printed the way a risograph misprints)

### fixed
- **the ARM builds did not compile** (linux-arm64, macos-universal): the Q8_0 NEON path used `<arm_neon.h>` only when the compiler had the dot-product extension, and the release targets baseline armv8-a; included for every NEON build now, and `make arm-check` compiles all of it for arm64 on a Mac before a tag
- in a terminal the timing line could land on the same line as the answer (`Program crashes.[7.5s wall: …]`): the closing newline sat in stdout's buffer while the stats went straight to stderr. flushed first now

### changed
- test binaries build into `build/` instead of the repo root; `make test_attn` and the rest still work. `make clean` removes both

### fixed: the test gate (2026-09-30)
- **a truncated model file crashed the engine** (SIGSEGV, no message); a download cut short keeps a valid header. every tensor is checked against the file now: `hakm: model file is incomplete — 48 of 1839 MB … run hakm pull again (it resumes)`. the directory-size check could overflow on 32-bit. the tokenizer reader had no bounds at all; a corrupt length walked off the mapping. new `test_load` builds damaged files and checks each is refused; proven to fail (and segfault) on the old loader
- **`make test` did not build on linux or musl.** the test rules hard-coded `-std=c11` without `_DEFAULT_SOURCE` (glibc then hides `posix_madvise`), and forced ASan/UBSan, which musl (alpine, iSh) does not ship. sanitizers are used where the compiler links them. verified in debian, alpine and 32-bit i386 alpine containers
- `hakm list` skips a model path too long to represent instead of stat-ing a clipped one (gcc `-Wformat-truncation`)
- `hakm --help` / `-h` print usage (they were taken as a model path: "open: no such file or directory"); `hakm -v` carries a build stamp like the other tools; release CI runs `make test` before building
- **a second `make test` tested nothing.** each test ran inside its build rule, so once built, `make test` said "nothing to be done". the tests now run on every `make test`, and `test_attn` depends on `src/model.c`; the file it exists to guard, which it included but did not list

### added: the KV cache directory is bounded
- one `.kvc` per distinct system prompt and nothing ever deleted them: 359 MB across five
  entries on the author's box before anyone looked, and the device this engine most needs to
  behave on is a phone. `kvc_evict()` trims oldest-first (by **mtime**; atime is off on most
  filesystems and would rank everything equally) to `HAKM_KV_DIR_MAX_MB`, default 1024, `0`
  disables. runs after every successful save with the just-written entry exempt. four rows in
  `make test_kvc`, under ASan

- **`--serve` streams tokens.** a 4th field on the request header asks for it; each token then
  ships as `T <len>\n<bytes>` as it is sampled, and the usual `R <len>\n<bytes>` still follows with
  the whole reply. callers that do not ask see byte-identical old behavior, and a caller that
  asks of an older engine simply gets the old reply; the field is ignored by a 3-field `sscanf`.
  this is what lets the agent type a local answer out in real time instead of sitting silent for
  the length of the generation


### fixed: attention wrote past its scores buffer on any small model
- `att_heads` indexes its scores array by **position** (`0..pos`), but two of its three call sites handed it a buffer sized by `ffn_dim`: the batch-prefill path (always taken) passed the gate scratch, and the single-threaded decode fallback passed `gate_buf`. that is safe only while `ffn_dim > ctx`, which is true for `hako-sho` (11008) and `hako-koi` (18944) and false for every smaller model; Qwen2.5-0.5B is 4864, 1.5B is 8960, SmolLM2-360M is 2560. it was a silent heap overflow waiting for the first small model, which is exactly what running on a phone requires
- batch prefill now has its own `bscores` buffer sized `cap * max_seq`; the decode fallback uses `att_scores`, which was already the right shape. `make test_attn` is a new ASan regression test that reproduces the overflow if either is undone

### fixed: a model larger than the address space crashed instead of explaining
- the loader assigned `st_size` (64-bit) into a `size_t`. on a 32-bit build a 4.7GB model truncated to about 0.4GB, every header offset still validated because they all live in the first few megabytes, and the first tensor read past the mapping was a SIGBUS with no message. oversized models are now refused with the size and the build width

