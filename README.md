<p align="center">
  <a href="https://mithraeums.github.io">
    <img src="https://mithraeums.github.io/assets/banner-hako.png" alt="hako — a from-scratch C inference engine for the inner chamber" width="100%"/>
  </a>
</p>

<p align="center">
  <em>A from-scratch C inference engine for the hako model family. Own loader, own kernels, own tokenizer. <code>hakm</code>.</em>
</p>

<p align="center">
  <a href="https://github.com/mithraeums/hako/releases"><img src="https://img.shields.io/badge/version-v0.0.4-e2402e?style=flat-square&labelColor=14130f" alt="v0.0.4"/></a>
  <img src="https://img.shields.io/badge/license-GPL--3.0-2636c4?style=flat-square&labelColor=14130f" alt="GPL-3.0"/>
  <img src="https://img.shields.io/badge/deps-libc%20%2B%20libm%20%2B%20pthread-2636c4?style=flat-square&labelColor=14130f" alt="deps"/>
  <img src="https://img.shields.io/badge/runtime-no%20llama.cpp%20%C2%B7%20no%20ollama-2636c4?style=flat-square&labelColor=14130f" alt="no llama.cpp / no ollama"/>
  <img src="https://img.shields.io/badge/models-hako--sho%20%C2%B7%20hako--koi-2636c4?style=flat-square&labelColor=14130f" alt="models"/>
</p>

<p align="center">
  <sub><a href="https://mithraeums.github.io">site</a> &nbsp;·&nbsp; <a href="https://github.com/mithraeums/hako/releases">releases</a> &nbsp;·&nbsp; <a href="https://huggingface.co/mithraeum">models</a> &nbsp;·&nbsp; <a href="https://github.com/mithraeums/hako-code">hako-code</a> (agent) &nbsp;·&nbsp; <a href="https://github.com/mithraeums/hako-edit">hako-edit</a> (editor) &nbsp;·&nbsp; <a href="https://github.com/mithraeums/hako-studio">hako-studio</a> (browser) &nbsp;·&nbsp; <a href="https://github.com/mithraeums">org</a></sub>
</p>

<br>

<!-- HERO: vhs/demo.tape — hakm --version, then a live streaming generation ending on the honest prefill/decode split line. -->
<p align="center">
  <img src="https://github.com/mithraeums/mithraeums.github.io/blob/main/assets/readme-screenshots/hako/demo.gif?raw=true" alt="hakm live generation" width="82%"/>
</p>

<table align="center">
  <tr>
    <td align="center" width="33%">
      <!-- vhs/info.tape — `hakm --info` arch banner. -->
      <img src="https://github.com/mithraeums/mithraeums.github.io/blob/main/assets/readme-screenshots/hako/screenshot-info.png?raw=true" alt="hakm --info arch banner" width="100%"/><br/>
      <sub><code>--info</code> · arch · params · GQA · ctx</sub>
    </td>
    <td align="center" width="33%">
      <!-- vhs/raw.tape — raw completion. -->
      <img src="https://github.com/mithraeums/mithraeums.github.io/blob/main/assets/readme-screenshots/hako/screenshot-raw.png?raw=true" alt="raw completion" width="100%"/><br/>
      <sub><code>--raw</code> · greedy completion</sub>
    </td>
    <td align="center" width="33%">
      <!-- vhs/chat.tape — one chat turn with the honest prefill/decode split. -->
      <img src="https://github.com/mithraeums/mithraeums.github.io/blob/main/assets/readme-screenshots/hako/screenshot-chat.png?raw=true" alt="one chat turn" width="100%"/><br/>
      <sub>one ChatML turn · honest tok/s split</sub>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td width="33%" valign="top"><b>From scratch.</b><br/><sub>Own GGUF→MLF2 loader, own Q4_K/Q6_K + int8 kernels, own byte-level BPE, own Qwen2 forward pass. No ggml, no llama.cpp, no torch.</sub></td>
    <td width="33%" valign="top"><b>Three deps.</b><br/><sub>libc + libm + pthread. That is the whole stack. <code>make</code> builds <code>./hakm</code> on Linux / macOS / FreeBSD.</sub></td>
    <td width="33%" valign="top"><b>Resident + fast.</b><br/><sub><code>--serve</code> keeps the model warm and reuses the KV prefix across turns. Batched prefill, FMA q4k, a Q6_K int8 path, threaded attention, f16 KV — <code>~8 tok/s</code> decode on a CPU-only i3.</sub></td>
  </tr>
</table>

<p align="center"><sub><b>—— I ——</b></sub></p>

## Overview

This repo is **the engine**. It runs the [hako models](https://huggingface.co/mithraeum)
end to end. [hako-code](https://github.com/mithraeums/hako-code) (the agent) runs
it as a `hakm` subprocess — a long-lived `--serve` process that keeps the model
resident and reuses the KV cache — and [hako-edit](https://github.com/mithraeums/hako-edit)
(the editor) reaches it through the agent. You can also run it directly.

- **No ggml, no llama.cpp, no torch, no ollama.** The loader, the k-quant +
  int8 kernels, the BPE tokenizer, and the Qwen2 forward pass are all
  hand-written C11. libc + libm + pthread is the entire runtime.
- **Own container format — MLF2.** `tools/gguf2mlf.py` converts a Qwen2.5-Coder
  GGUF to a native MLF2 file once (pure stdlib): k-quant blocks copied verbatim,
  tokenizer embedded, arch params in a fixed 128-byte header. The runtime mmaps
  it — RSS is just activations + KV cache.
- **Runs the real weights.** Qwen2.5-Coder **3B** (`hako-sho`) and **7B**
  (`hako-koi`) generate end to end — greedy + sampling, ChatML, multi-turn,
  tool-call dialect. Quant output is **bit-exact** vs an independent Python
  reimpl.
- **CPU-only, and honest about it.** This is correctness-first; the goal is
  owning the stack on old hardware, not beating tuned GPU runtimes. Speed wins
  come from int8 fast paths and KV reuse, measured on a 4-core i3-8100.

<p align="center"><sub><b>—— II ——</b></sub></p>

## Build & Run

```sh
curl -fsSL https://mithraeums.github.io/hakm.sh | sh    # release binary, or a source build where there is none (iSh)

git clone https://github.com/mithraeums/hako && cd hako
make                            # builds ./hakm — libc + libm + pthread, no deps
make test                       # the ASan gates (no weights needed)
```

> Deps: libc + libm + pthread. No third-party libraries. Built `-march=native` by
> default; pass `ARCH=""` for a portable generic build, or `ARCH="-arch arm64 -arch x86_64"`
> for a macOS universal binary.

### Get a model

Weights are **never** stored here — pull them from HuggingFace
([huggingface.co/mithraeum](https://huggingface.co/mithraeum)) or convert your own:

```sh
hakm pull hako-sho      # resumable, checked, installed to ~/.hako/models/
hakm list               # what is installed
# (inside the agent: :pull hako-sho does the same)

# or convert a Qwen2.5-Coder GGUF yourself, once:
python3 tools/gguf2mlf.py model.gguf ~/.hako/models/hako-sho/hako-sho.mlf2
```

| model | tier | base | license |
|---|---|---|---|
| [`hako-sho`](https://huggingface.co/mithraeum/hako-sho) | mini · 3B | Qwen2.5-Coder-3B-Instruct | ⚠️ Qwen-research (non-commercial) |
| [`hako-koi`](https://huggingface.co/mithraeum/hako-koi) | mid · 7B | Qwen2.5-Coder-7B-Instruct | Apache-2.0 |

Future tiers (14B/32B fine-tune, 50B+ max) are queued. Stock wraps carry no
version; the first real fine-tune of a tier earns `v0.0.1`.

### Run

```sh
M=hako-sho                                            # an id, org/id, or a .mlf2 path

./hakm "$M" --info                                   # arch banner, then exit
./hakm "$M" --raw -t 0 "def fibonacci(n):"           # raw greedy completion
./hakm "$M" --sys "You are hako." "ring buffers in C" # one ChatML turn
./hakm "$M" --sys "You are hako."                    # interactive multi-turn REPL
./hakm --version
```

Flags: `-n` new-tokens · `-t` temp (0 = greedy) · `-p` top_p · `-k` top_k ·
`--rep F` repetition penalty (1 = off) · `--rep-window N` (default 64) ·
`-s` seed · `--sys TEXT` · `--ctx N` · `--raw` · `--info` · `--chat-stdin` · `--serve` ·
`--no-kv-cache`. `hakm --help` lists them, plus the environment knobs.

<p align="center"><sub><b>—— III ——</b></sub></p>

## How it works

1. **Offline conversion** — `tools/gguf2mlf.py` reads a GGUF and emits an
   **MLF2** file: the native container. Quant blocks are copied verbatim (ggml's
   k-quant layout preserved), tokenizer (vocab + merges) embedded, arch params in
   a fixed 128-byte header. Pure stdlib — no torch / llama.cpp dependency.

2. **Runtime** (C11, libc + libm + pthread):
   - `loader.c` — mmap the MLF2, resolve tensors by name. RSS = activations + KV.
   - `quant.c` — Q4_K / Q6_K dequant + **int8 fast-path dot** (quantize the
     activation to int8 once, dot straight against the k-quant weights; weights
     stay exact). Bit-exact vs `tests/q_ref.py`. AVX2-FMA + scalar paths.
   - `nn.c` — rmsnorm, softmax, SiLU, **NeoX rope**, int8 matmul, and a generic
     `hakm_parallel` worker pool (matmul rows + attention heads share it).
   - `model.c` — Qwen2 forward: **GQA**, QKV bias, KV cache (f16 when `__F16C__`),
     batched prefill, threaded attention, lm_head (tied on 3B, `output.weight` on 7B+).
   - `bpe.c` — Qwen2 byte-level BPE (GPT2 byte map + rank-greedy merges).
   - `cli/main.c` — the `hakm` CLI: one-shot, `--raw`, chat REPL, `--chat-stdin`,
     and the resident `--serve` loop.

3. **Serve loop** — `--serve` is a resident framed-request loop. Each request's
   rendered conversation is prefix-matched against the KV-cached tokens; rollback
   is just `pos = p`; only the unseen tail prefills. This is how the agent gets
   warm turns in seconds instead of re-prefilling the whole system prompt every
   turn. Optional per-request temperature (greedy on tool turns) and token cap.

<p align="center"><sub><b>—— IV ——</b></sub></p>

## Validated

Runs the real **Qwen2.5-Coder 3B and 7B** weights end to end (greedy + sampling,
ChatML, multi-turn REPL, `--serve`):

- **3B** (`hako-sho`): qwen2 · 36 layers · d_model 2048 · 16/2 heads (GQA) · ffn
  11008 · vocab 151936 · tied embeddings.
- **7B** (`hako-koi`): qwen2 · 28 layers · d_model 3584 · 28/4 heads (GQA) · ffn
  18944 · vocab 152064 · **untied** (`output.weight`).

Quant correctness gate: C output **bit-exact** vs `tests/q_ref.py` (Q4_K + Q6_K,
maxdiff 0.0). Strict-FP build is batch-vs-single bit-identical. ASan + UBSan clean
on chat + serve paths.

```sh
make test                          # attention bounds, KV cache, Q8_0, damaged model files (ASan)
make test_quant                    # build/test_quant: dequant a real tensor for tests/q_ref.py
make test_kdot                     # build/test_kdot: fused Q4_K dot vs scalar, on real weights
```

<p align="center"><sub><b>—— V ——</b></sub></p>

## Speed

CPU-only, measured on a 4-core **i3-8100** (the staging box itself):

- **Decode ~8 tok/s, flat with depth** (was 2.35 → **~3.4×**), via the Q6_K int8
  dot path, threaded attention, and an f16 KV cache that stops decode sagging as
  context grows.
- **Batched prefill ~4.7×** — `matmul_batch` streams each weight row from RAM
  once per batch (hot in L1 for the other B−1 dots); the FMA q4k kernel keeps
  8-lane int32 partials vectorized with one hsum per row.
- **Prefill ~1.3× on top of that** (compute-bound once weight streaming is
  amortized) — the q4k min term (`dmin·m·Σx`) is activation-only, so `scale·Σx`
  is folded once per activation and reused across every weight row, freeing half
  the per-group port-0 MACs. Greedy output stays bit-identical.
- **Warm agent turns ~3–5s** (were 5+ minutes) — `--serve` keeps the model
  resident and reuses the KV prefix, so only the new tail prefills.
- **7B (`hako-koi`) is viable on 8GB** — ~4 tok/s decode, RSS ~4.4GB.

Honest framing: the int8 paths keep weights exact (greedy output is
bit-identical to the float path), and the `tok/s` print **splits prefill from
decode** rather than blending them. Decode is near the box's bandwidth/compute
floor; the next jumps are speculative decode (needs a draft model) or a
register-blocked prefill GEMM micro-kernel (amortize the weight nibble-unpack
across batch columns) — not more threads.

<p align="center"><sub><b>—— VI ——</b></sub></p>

## Change Log

### Latest — engine v0.0.4

- **Local answers stream** — `--serve` sends each token as it is sampled
  (`T <len>` frames; opt-in via a 4th header field, so old callers and old
  engines are unaffected). The agent types a local reply out in real time.
- **KV prefix on disk** — the prefilled system prompt is saved under
  `~/.hako/cache/kv/` and restored on the next run: cold first turn on the i3
  went from **255 s to 2.3 s**. Keyed on model size+mtime, layers, kv dim, kv
  type and the exact token ids; `--no-kv-cache` opts out.
- **…and the cache is bounded** — one file per distinct system prompt, and
  nothing ever deleted them (359 MB piled up before anyone looked). Oldest
  first to `HAKM_KV_DIR_MAX_MB` (default 1024; `0` disables), the just-written
  entry exempt.
- **Small models no longer overflow** — `att_heads` indexed its scores by
  position but two call sites passed a buffer sized by `ffn_dim`; safe for
  sho/koi, a heap overflow for anything smaller. `make test_attn` reproduces
  it under ASan if undone.
- **32-bit / iSh** — builds on musl (`-D_DEFAULT_SOURCE`, `posix_madvise`); no
  `-march=native` on i386, where the emulator advertises AVX/FMA it cannot run
  (that was SIGILL mid-prefill); prefault off; a model larger than the address
  space is refused with a message instead of a SIGBUS.
- **Model store** — `hakm list | pull | rm`, resumable pulls, atomic install;
  **llama arch** (SmolLM2 proven) and **Q8_0** (`make test_q8`).
- **Damaged files are refused** — a truncated download or a corrupt tokenizer
  gets a message (`model file is incomplete … run hakm pull again`) instead of a
  segfault; `make test` builds damaged files and checks each one.

Earlier versions, and the full detail: [CHANGELOG.md](CHANGELOG.md).

<p align="center"><sub><b>—— VII ——</b></sub></p>

## Known gaps

- **Windows:** the loader uses POSIX `mmap`, which MinGW lacks. Linux (glibc and musl, 64- and
  32-bit), macOS and FreeBSD build today.
- **Tokenizer pretokenizer is simplified** (whitespace-delimited; merges, symbols and special
  tokens are correct), so digit and punctuation runs can split differently from the reference.

<p align="center"><sub><b>—— VIII ——</b></sub></p>

## Source layout

```
src/loader.c       mmap MLF2, resolve tensors, refuse damaged files
src/quant.c        Q4_K / Q6_K / Q8_0 dequant + int8 fast dot (bit-exact)
src/nn.c           rmsnorm, softmax, SiLU, rope, matmul, worker pool
src/model.c        Qwen2 / llama forward — GQA, KV cache, batched prefill, threaded attn
src/bpe.c          byte-level BPE (encode + decode)
src/sample.c       temp / top-k / top-p / greedy, repetition penalty
src/kvcache.c      the system-prompt KV prefix on disk, with eviction
src/models.c       hakm list / pull / rm — the model store
src/hakm_api.c     optional in-process library (libhakm.a); not the agent's path
cli/main.c         the hakm CLI — one-shot / raw / REPL / --chat-stdin / --serve
include/mlf2.h     the native container format
tools/gguf2mlf.py  offline GGUF → MLF2 converter (pure stdlib)
tests/             make test (ASan) + the real-weights checks and q_ref.py
install.sh         the installer behind mithraeums.github.io/hakm.sh
```

## Contributing
If you share the belief that simplicity empowers creativity, feel free to contribute.

### Contribution is welcome in the form of:
- Forking this repo
- Submitting a Pull Request
- Bug reports and feature requests

Please ensure your code follows the existing style — C11, tabs, `snake_case`,
ASan + UBSan clean, quant changes proven bit-exact against `tests/q_ref.py`.

## License

- **Engine code — GPL-3.0** (`LICENSE`). From-scratch C; no ggml/llama.cpp/torch.
- **Model weights keep their required upstream licenses** (on HuggingFace, not
  here): `hako-koi` (7B) Apache-2.0; `hako-sho` (3B) **Qwen RESEARCH —
  non-commercial only** (the 3B base is research-licensed). See each model repo's
  `LICENSE` / `NOTICE`.

<p align="center"><sub><a href="LICENSE">— SEE LICENSE —</a> &nbsp;·&nbsp; GPL-3.0</sub></p>

<p align="center"><sub><em>— deus sol invictus mithras —</em></sub></p>

