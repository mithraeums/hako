<p align="center">
  <a href="https://mithraeums.github.io">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://mithraeums.github.io/assets/banner-hako-night.png"/>
      <img src="https://mithraeums.github.io/assets/banner-hako.png" alt="hako — a from-scratch C inference engine for the inner chamber" width="100%"/>
    </picture>
  </a>
</p>

<p align="center">
  <em>A from-scratch C inference engine for the hako model family. Own loader, own kernels, own tokenizer. <code>hakm</code>.</em>
</p>

<p align="center">
  <a href="https://github.com/mithraeums/hako/releases"><img src="https://img.shields.io/badge/version-v0.0.6-e2402e?style=flat-square&labelColor=14130f" alt="v0.0.6"/></a>
  <img src="https://img.shields.io/badge/license-GPL--3.0-2636c4?style=flat-square&labelColor=14130f" alt="GPL-3.0"/>
  <img src="https://img.shields.io/badge/deps-libc%20%2B%20libm%20%2B%20pthread-2636c4?style=flat-square&labelColor=14130f" alt="deps"/>
  <img src="https://img.shields.io/badge/runtime-no%20llama.cpp%20%C2%B7%20no%20ollama-2636c4?style=flat-square&labelColor=14130f" alt="no llama.cpp / no ollama"/>
  <img src="https://img.shields.io/badge/models-hako--sho%20%C2%B7%20hako--koi-2636c4?style=flat-square&labelColor=14130f" alt="models"/>
</p>

<p align="center">
  <sub><a href="https://mithraeums.github.io">site</a> &nbsp;·&nbsp; <a href="https://github.com/mithraeums/hako/releases">releases</a> &nbsp;·&nbsp; <a href="https://huggingface.co/mithraeum">models</a> &nbsp;·&nbsp; <a href="https://github.com/mithraeums/hako-code">hako-code</a> (agent) &nbsp;·&nbsp; <a href="https://github.com/mithraeums/hako-edit">hako-edit</a> (editor) &nbsp;·&nbsp; <a href="https://github.com/mithraeums/hako-studio">hako-studio</a> (browser) &nbsp;·&nbsp; <a href="https://github.com/mithraeums">org</a></sub>
</p>

<br>

<!-- terminal media: real sessions through a pty, drawn in the night terminal at 2x (Code/mithraeums/.claude/web/look/media.py).
     one image per row, width only, never a height: GitHub can shrink them but never stretch them -->
<p align="center">
  <img src="https://github.com/mithraeums/mithraeums.github.io/blob/main/assets/readme-screenshots/hako/demo.gif?raw=true" alt="hakm --version, hakm list, then a live streaming answer from hako-sho ending on the prefill/decode split" width="100%"/>
</p>

<p align="center">
  <img src="https://github.com/mithraeums/mithraeums.github.io/blob/main/assets/readme-screenshots/hako/info.png?raw=true" alt="hakm hako-sho --info: arch, parameters, layers, heads with GQA, context" width="100%"/><br/>
  <sub><code>--info</code>: arch, params, GQA, context</sub>
</p>

<p align="center">
  <img src="https://github.com/mithraeums/mithraeums.github.io/blob/main/assets/readme-screenshots/hako/raw.png?raw=true" alt="a raw greedy completion of def fibonacci(n):" width="100%"/><br/>
  <sub><code>--raw</code>: a plain greedy completion, then the honest tok/s split</sub>
</p>

<table>
  <tr>
    <td width="33%" valign="top"><b>From scratch.</b><br/><sub>Own GGUF and safetensors → MLF2 loader, own quantizer, own Q4_K/Q6_K + int8 kernels, own byte-level BPE, own forward pass (qwen2, qwen3, llama), own trainer. No ggml, no llama.cpp, no torch.</sub></td>
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
  int8 kernels, the BPE tokenizer, and the forward pass (qwen2, qwen3, llama) are all
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

# or convert one yourself, once — from a GGUF, or from a Hugging Face folder:
python3 tools/gguf2mlf.py model.gguf ~/.hako/models/hako-sho/hako-sho.mlf2
python3 tools/hf2mlf.py ./Qwen3-0.6B qwen3-0.6b-bf16.mlf2

# then make it small, and check what that cost:
hakm quantize qwen3-0.6b-bf16.mlf2 qwen3-0.6b.mlf2 q4_k
hakm qwen3-0.6b.mlf2 --ppl wiki.test.raw

# or train one of your own, tokenizer included (make train):
hakm-train --data wiki.train.raw --val wiki.valid.raw --out tiny.mlf2 --steps 1500 --lr 2e-3
hakm tiny.mlf2 --raw "The history of the city"

# it runs at the lowest priority and prints the memory it will hold. overnight,
# out of the way (caffeinate keeps a mac awake; --at holds even the memory until then):
caffeinate -i hakm-train --data wiki.train.raw --out tiny.mlf2 --at 01:00 --threads 4
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

### Latest — engine v0.0.6

- **A model of our own, end to end** — `hakm-train` (`make train`) trains a
  byte-level BPE and the engine's own shape from scratch on the CPU, in C, and
  writes an MLF2 file hakm runs and quantizes as is. 4.5M params on wikitext-2:
  val ppl **171 → 60** in 13 min on the i3. Every gradient is checked against a
  finite difference in `make test`.
- **Own quantizer and a number to judge it by** — `hakm quantize` to q4_k /
  q6_k / q8_0 / f16 / bf16, and `--ppl` (plus bits per byte). Qwen3-0.6B on
  wikitext-2: bf16 21.16, q8_0 21.20, q4_k 22.72.
- **Hugging Face straight in** — `tools/hf2mlf.py` reads safetensors (bf16
  runs as is), byte-identical to the GGUF route. No GGUF in between.
- **Qwen3 runs** — per-head q/k norm and attention width apart from `d_model`;
  model features come from the file, not the arch name.
- **x86 release builds ~3x faster** — a `x86-64-v3` build with the AVX2/FMA
  paths, picked by `install.sh` (and an `x86_64h` slice on macOS).
- **`tools/mkcorpus.py`** — a training corpus from permissive sources only,
  with a manifest of every source and its license.

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
src/quantize.c     float → f16 / bf16 / Q8_0 / Q4_K / Q6_K, and `hakm quantize`
src/nn.c           rmsnorm, softmax, SiLU, rope, matmul, worker pool
src/model.c        the forward pass (qwen2 / qwen3 / llama / hako) — GQA, KV cache, batched prefill, threaded attn
src/bpe.c          byte-level BPE (encode + decode)
src/sample.c       temp / top-k / top-p / greedy, repetition penalty
src/kvcache.c      the system-prompt KV prefix on disk, with eviction
src/models.c       hakm list / pull / rm — the model store
src/train.c        the trainer: backward pass, AdamW, BPE training, MLF2 out
src/hakm_api.c     optional in-process library (libhakm.a); not the agent's path
cli/main.c         the hakm CLI — one-shot / raw / REPL / --chat-stdin / --serve
cli/train.c        hakm-train, the training loop (make train)
include/mlf2.h     the native container format
tools/gguf2mlf.py  offline GGUF → MLF2 converter (pure stdlib)
tools/hf2mlf.py    offline safetensors → MLF2 converter (pure stdlib)
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

