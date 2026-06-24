<p align="center">
  <a href="https://mithraeums.github.io">
    <img src="https://mithraeums.github.io/assets/banner-hako-dark.svg" alt="hako — a from-scratch C inference engine for the inner chamber" width="100%"/>
  </a>
</p>

<p align="center">
  <em>A from-scratch C inference engine for the hako model family. Own loader, own kernels, own tokenizer. <code>hakm</code>.</em>
</p>

<p align="center">
  <a href="https://github.com/mithraeums/hako/releases"><img src="https://img.shields.io/badge/engine-v0.0.2-b89656?style=flat-square&labelColor=14130f" alt="v0.0.2"/></a>
  <img src="https://img.shields.io/badge/license-GPL--3.0-c8c2b2?style=flat-square&labelColor=14130f" alt="GPL-3.0"/>
  <img src="https://img.shields.io/badge/deps-libc%20%2B%20libm%20%2B%20pthread-c8c2b2?style=flat-square&labelColor=14130f" alt="deps"/>
  <img src="https://img.shields.io/badge/runtime-no%20llama.cpp%20%C2%B7%20no%20ollama-c8c2b2?style=flat-square&labelColor=14130f" alt="no llama.cpp / no ollama"/>
  <img src="https://img.shields.io/badge/models-hako--sho%20%C2%B7%20hako--koi-c8c2b2?style=flat-square&labelColor=14130f" alt="models"/>
</p>

<p align="center">
  <sub><a href="https://mithraeums.github.io">site</a> &nbsp;·&nbsp; <a href="https://github.com/mithraeums/hako/releases">releases</a> &nbsp;·&nbsp; <a href="https://huggingface.co/mithraeum">models</a> &nbsp;·&nbsp; <a href="https://github.com/mithraeums/hako-code">hako-code</a> (agent) &nbsp;·&nbsp; <a href="https://github.com/mithraeums/hako-edit">hako-edit</a> (editor) &nbsp;·&nbsp; <a href="https://github.com/mithraeums">org</a></sub>
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
end to end and is spawned by [hako-code](https://github.com/mithraeums/hako-code)
(the agent) and [hako-edit](https://github.com/mithraeums/hako-edit) (the editor)
as a `hakm` subprocess — one-shot `--chat-stdin` per turn, or a long-lived
`--serve` process that keeps the model resident and reuses the KV cache.

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
git clone https://github.com/mithraeums/hako && cd hako
make                            # builds ./hakm — libc + libm + pthread, no deps
make UNIVERSAL=1                # (cross) override ARCH for a fat / portable build
```

> Deps: libc + libm + pthread. No third-party libraries. Built `-march=native` by
> default; pass `ARCH=""` for a portable generic build, or `ARCH="-arch arm64 -arch x86_64"`
> for a macOS universal binary.

### Get a model

Weights are **never** stored here — pull them from HuggingFace
([huggingface.co/mithraeum](https://huggingface.co/mithraeum)) or convert your own:

```sh
# let the agent fetch + convert for you:
hako   →   :pull hako-sho

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
M=~/.hako/models/hako-sho/hako-sho.mlf2

./hakm "$M" --info                                   # arch banner, then exit
./hakm "$M" --raw -t 0 "def fibonacci(n):"           # raw greedy completion
./hakm "$M" --sys "You are hako." "ring buffers in C" # one ChatML turn
./hakm "$M" --sys "You are hako."                    # interactive multi-turn REPL
./hakm --version                                     # hakm v0.0.2
```

Flags: `-n` new-tokens · `-t` temp (0 = greedy) · `-p` top_p · `-k` top_k ·
`-s` seed · `--sys TEXT` · `--ctx N` · `--raw` · `--info` · `--chat-stdin` · `--serve`.

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
on chat + serve paths. Build `tests`:

```sh
make test_quant && ./test_quant      # Q4_K/Q6_K dequant vs reference
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
- **Warm agent turns ~3–5s** (were 5+ minutes) — `--serve` keeps the model
  resident and reuses the KV prefix, so only the new tail prefills.
- **7B (`hako-koi`) is viable on 8GB** — ~4 tok/s decode, RSS ~4.4GB.

Honest framing: the int8 paths keep weights exact (greedy output is
bit-identical to the float path), and the `tok/s` print **splits prefill from
decode** rather than blending them. Decode is near the box's bandwidth/compute
floor; the next jumps are speculative decode (needs a draft model) or a
prefill GEMM micro-kernel — not more threads.

<p align="center"><sub><b>—— VI ——</b></sub></p>

## Change Log

### engine v0.0.2 (Latest)

- **`--serve`** — resident framed-request loop with **KV prefix reuse**; warm
  agent turns in seconds instead of re-prefilling every turn.
- **Batched prefill** — `matmul_batch` + `hakm_forward_batch` (causal within
  batch, lm_head only for the last token). `HAKM_BATCH` knob.
- **Q6_K int8 dot path** — decode 2.88 → 8.40 tok/s; was riding the float
  dequant path and dominating decode.
- **Threaded attention + f16 KV cache** — decode flat with depth; koi KV at ctx
  4096 halved.
- **FMA q4k kernel** — one hsum per row; prefill + decode both up.
- **Per-request temperature + stop sequences** in `--serve` (greedy tool turns,
  break on `</tool_call>` / `</write_file>`).
- **Honest tok/s** — every one-shot path now prints prefill and decode rates
  **separately** (the blended figure hid a slow decode behind a fast prefill).
- **`--version` / `-v`** → `hakm v0.0.2`.

### engine v0.0.1

- First native runner: MLF2 loader, Q4_K/Q6_K dequant, Qwen2 GQA forward,
  NeoX rope, byte-level BPE, int8-activation × int4-weight fast dot, pthread
  matmul. Runs 3B + 7B end to end, no llama.cpp / no ollama.

<p align="center"><sub><b>—— VII ——</b></sub></p>

## Known gaps / next

- **Windows:** the loader uses POSIX `mmap`; MinGW lacks `sys/mman.h`. An
  `#ifdef _WIN32` mmap shim is the unlock. Linux / macOS / FreeBSD build today.
- **Tokenizer pretokenizer is simplified** (whitespace-delimited; merges,
  symbols, special tokens correct). Needs the full Qwen2 regex for
  digit/punctuation-run boundaries to exactly match reference tokenization.
- **Decode ceiling:** near the i3's bandwidth/compute floor. Real breaks =
  speculative decode (sho as a draft model for koi — needs the training lane) or
  a smaller quant tier.
- **Prefill:** a q4k/q6k GEMM micro-kernel (hoist nibble-unpack across the batch)
  is the next pure-engine lever; NEON q6k path for arm64.

## Roadmap

- [x] MLF2 loader + own k-quant dequant (bit-exact)
- [x] Qwen2 GQA forward, NeoX rope, byte-level BPE
- [x] int8-activation × int4-weight fast dot (AVX2-FMA + scalar)
- [x] `--serve` resident loop + KV prefix reuse
- [x] batched prefill, Q6_K int8 path, threaded attention, f16 KV
- [x] runs 3B (`hako-sho`) + 7B (`hako-koi`) end to end
- [ ] Windows `mmap` shim
- [ ] full Qwen2 pretokenizer regex
- [ ] NEON q6k path (arm64) + prefill GEMM micro-kernel
- [ ] speculative decode (draft model from the training lane)

<p align="center"><sub><b>—— VIII ——</b></sub></p>

## Source layout

```
src/loader.c     mmap MLF2, resolve tensors
src/quant.c      Q4_K / Q6_K dequant + int8 fast dot (bit-exact)
src/nn.c         rmsnorm, softmax, SiLU, NeoX rope, matmul, worker pool
src/model.c      Qwen2 forward — GQA, KV cache, batched prefill, threaded attn
src/bpe.c        Qwen2 byte-level BPE (encode + decode)
src/sample.c     temp / top-k / top-p / greedy
cli/main.c       the hakm CLI — one-shot / raw / REPL / --chat-stdin / --serve
tools/gguf2mlf.py  offline GGUF → MLF2 converter (pure stdlib)
include/mlf2.h   the native container format
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

