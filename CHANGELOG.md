# Changelog

All notable changes to the hako engine (`hakm`).

## [v0.0.5] - 2026-11-01

### added
- **release builds are ~3x faster on x86.** they were `x86-64-v2`, which leaves out the AVX2/FMA paths: on an i3-8100 sho decoded at 2.7 tok/s against 8.3-9.5 for an `x86-64-v3` build (prefill 4x). linux now also ships `hakm-linux-x86_64-v3`, which `install.sh` picks when `/proc/cpuinfo` has the v3 flags (else the v2 build, as before); the macOS universal binary gains an `x86_64h` slice, which macOS picks on a Haswell+ Intel Mac by itself. Apple Silicon was already fine (Apple's default arm64 target has dotprod)
- **qwen3 models run**: per-head q/k norm, used whenever a file has the tensors, and an attention width (`n_heads * head_dim`) that no longer has to equal `d_model`. Qwen3-0.6B Q8_0 answers correctly at ~30 tok/s decode on the i3; with the norm switched off it produces garbage, so the step is load-bearing
- `make test` gains `test_arch`: a tiny random model with the qwen3 shape checked against a plain reference forward pass, decode and batched prefill both. on the v0.0.4 source it is a heap overflow under ASan
- `gguf2mlf.py` takes `qwen3` and reads `head_dim` from `attention.key_length`
- **`tools/hf2mlf.py`**: a Hugging Face model folder (safetensors, sharded or not, plus `tokenizer.json`) straight to MLF2, at the dtype it holds. no GGUF in between, which is the path for any model nobody made a GGUF of, ours included. checked on Qwen3-0.6B: header, tokenizer (4.2 MB) and every f32 tensor byte-identical to the GGUF route
- **bf16 tensors**: what safetensors usually hold, run as-is (no widening to f32 at twice the size)
- **`hakm quantize <model> <out> q4_k|q6_k|q8_0|f16|bf16|f32`**: our own quantizer, from any type the engine reads. `q4_k` is the q4_k_m mix (q6_k for the output projection and some attn_v / ffn_down). Qwen3-0.6B bf16 → q4_k in 14 s on the i3 (1.19 → 0.39 GB). q8_0 is the exact formula; `make test` gains `test_quantize` (every f16 value round-trips, each type against the dequant that defines it, q4_k beats a plain min/max grid)
- **`hakm <model> --ppl FILE`**: perplexity over a text file (windows of 512, second half scored), so a quantization or a training run has a number. Qwen3-0.6B on wikitext-2 (2048 tokens): bf16 21.16, our q8_0 21.20, our q4_k 22.72. pinned in `test_arch` against the reference log-likelihood
- **arch `hako` and the `ROPE_NORM` flag**: the RoPE pairing is a flag in the file now, so a model of our own is arch `hako` plus whichever features it uses, with no engine code per architecture. a file with a feature bit this engine does not know is refused, not run without it. old llama files still read by arch id
- **`hakm-train` (`make train`)**: a model of our own from scratch, on the CPU, in C. trains a byte-level BPE tokenizer split the way the engine's encoder splits, then the engine's own shape (RMSNorm, GQA, neox RoPE, SwiGLU, tied embedding) with AdamW, warmup + cosine, grad clipping and accumulation, and writes an f32 MLF2 file, arch `hako`, that hakm runs and quantizes as is. `--resume` continues from `MODEL.mlf2.opt`, ctrl-c saves first, `--tok` reuses another model's tokenizer. first run: 4.5M params on wikitext-2 train, 2 epochs in 13 min on the i3 (7,400 tok/s; 26 min before the speedup below), val ppl 171 → 60. on wiki.test it scores 62.1 at f32, 62.1 at q8_0, 62.3 at q4_k
- the trainer is 2-2.6x faster: the matrix product packs both operands into cache-sized panels (it read B a page apart per step), attention runs as blocked products on the same kernel and skips what causality zeroes, exp is 8 lanes at a time, and the norms, residual adds and rope run on every core. 4.5M model 3,500 → 7,800 tok/s, 13M model 1,100 → 2,800. losses match the old build to 4 decimals
- `hakm-train` gives way: it runs at nice 19 unless `--nice 0`, `--threads N` caps the cores, `--at HH:MM` waits before loading anything, and it prints the memory the run will hold (13M params at ctx 512: 2.5 GB, measured 2.54)
- **`tools/mkcorpus.py`**: a training corpus from permissive sources only (public domain, MIT, BSD, ISC, Apache-2.0, zlib, 0BSD, CC BY 4.0), with a manifest of every source and its license. repo tarballs are streamed and filtered in memory (no clones), FineWeb-Edu is read by byte range (64 MB downloaded for 100 MB of text). deduplicated, shuffled, 1% held out. the first build: 298 MB of C, Go, Rust, Zig, shell and English in 41 s
- `make test` gains `test_train`: every tensor's gradient against a finite difference (worst 2e-3), hakm scoring the saved file within 1e-5 of the trainer, the tokenizer round-tripping through the engine's encoder, checkpoints back bit for bit, a loss that falls
- `--ppl` also prints bits per byte, which compares models with different tokenizers (qwen3-0.6b 1.05, the 4.5M wiki model 1.69)
- models whose vocab has `<think>` (qwen3) answer straight away: the reply opens with an empty think block, as pushed token ids, which is the model's own non-thinking template. `--think` lets it reason first

### changed
- model features are read from the file, not the arch name: q/k/v bias and q/k norm by tensor presence, the RoPE pairing by flag. one forward pass for every model, including ones we design ourselves
- a model file with an unknown arch, or only one of the q/k norms, is refused instead of run as qwen2
- `--info`, `hakm list` and the embed API name the arch from one table (the API always said qwen2); tensor type names and row sizes come from one table each too
- a tensor of a type this engine cannot run is refused at load with its name
- `--info`, `hakm list` and `hakm quantize` show small models in M and MB instead of 0.00B and 0.00 GB
- hako-sho and hako-koi output is byte-identical to v0.0.4

### fixed
- the q4_k fitter's fallback tested against `INFINITY`, which `-ffast-math` (hakm's flags) lets the compiler fold away, leaving a block's quants uninitialized if no candidate grid fit. it starts from the plain grid now, against `FLT_MAX`; output is byte-identical (Qwen3-0.6B q4_k, 390.75 MB). `make test` gains `werror-check`, hakm's sources at the release flags with `-Werror`, which fails on the old source under clang

## [v0.0.4] - 2026-10-15

ships together with hako-code v0.2.4 (token streaming and the model paths span both).

### changed: the new look (2026-10-12)
- README banner and badges in the new inks (the suite's new look: two inks, ultramarine and vermilion, on washi, printed the way a risograph misprints)

### fixed
- **the ARM builds did not compile** (linux-arm64, macos-universal): the Q8_0 NEON path used `<arm_neon.h>` only when the compiler had the dot-product extension, and the release targets baseline armv8-a; included for every NEON build now, and `make arm-check` compiles all of it for arm64 on a Mac before a tag
- **Q8_0 on ARM without the dot-product extension summed wrong** on large weights: four int8 products were stacked in one 16-bit lane (up to 4 x 16384) and overflowed; each product is widened to 32 bits now (bit-exact emulation: 530/20000 random blocks wrong before, 0 after; the macOS runner caught it)
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

