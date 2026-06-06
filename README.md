<div align="center">
  <a href="https://mithraeums.github.io">
    <img src="https://mithraeums.github.io/assets/banner-hako-dark.svg" alt="hako — from-scratch C inference engine" width="100%"/>
  </a>

  <p><em>hako: a from-scratch C inference engine for the hako model family. <code>hakm</code>.</em></p>

  <img src="https://img.shields.io/badge/license-GPL--3.0-c8c2b2?style=flat-square&labelColor=14130f" alt="GPL-3.0"/>
  <img src="https://img.shields.io/badge/deps-libc%20%2B%20libm%20%2B%20pthread-c8c2b2?style=flat-square&labelColor=14130f" alt="deps"/>
  <img src="https://img.shields.io/badge/runtime-no%20llama.cpp%20%C2%B7%20no%20ollama-c8c2b2?style=flat-square&labelColor=14130f" alt="no llama.cpp / no ollama"/>

  <sub><a href="https://mithraeums.github.io">site</a> &nbsp;·&nbsp; <a href="https://github.com/mithraeums/hako-code">hako-code</a> (agent) &nbsp;·&nbsp; <a href="https://github.com/mithraeums/hako-edit">hako-edit</a> (editor) &nbsp;·&nbsp; <a href="https://github.com/mithraeums">org</a></sub>
</div>

---

This repo is **the engine** <br>Own GGUF -> MLF2 loader, own Q4_K/Q6_K + int8 kernels,
own BPE tokenizer, own Qwen2 forward pass. **No ggml, no llama.cpp, no torch, no
ollama at runtime.** libc + libm + pthread, that's the stack.

It runs the hako models end to end and is spawned by [hako-code](https://github.com/mithraeums/hako-code)
(the agent) and [hako-edit](https://github.com/mithraeums/hako-edit) (the editor)
as a one-shot `hakm --chat-stdin` subprocess.

## Models

Weights aren't stored here; the engine pulls them from HuggingFace on request
(`hako :pull <id>`, or download + convert yourself). Hosted under
[**huggingface.co/mithraeum**](https://huggingface.co/mithraeum):

| model | tier | base | license |
|---|---|---|---|
| [`hako-sho`](https://huggingface.co/mithraeum/hako-sho) | mini · 3B | Qwen2.5-Coder-3B-Instruct | [ ! ] Qwen-research (non-commercial) |
| [`hako-koi`](https://huggingface.co/mithraeum/hako-koi) | mid · 7B | Qwen2.5-Coder-7B-Instruct | Apache-2.0 |

Future tiers (14B/32B fine-tune, 50B+ max) are queued. Stock wraps carry no
version; the first real fine-tune of a tier earns `v0.0.1`.

## Build & run

```sh
git clone https://github.com/mithraeums/hako && cd hako
make                                            # builds ./hakm — libc + libm + pthread, no deps

# convert a Qwen2.5-Coder GGUF (HF download, or an existing ollama blob) once:
python3 tools/gguf2mlf.py model.gguf ~/.hako/models/hako-sho.mlf2

./hakm ~/.hako/models/hako-sho.mlf2 --raw -t 0 "def fibonacci(n):"      # raw completion
./hakm ~/.hako/models/hako-sho.mlf2 --sys "You are hako." "ring buffers in C"  # one chat turn
./hakm ~/.hako/models/hako-sho.mlf2 --sys "You are hako."               # interactive REPL
# flags: -n new-tokens  -t temp(0=greedy)  -p top_p  -k top_k  -s seed  --raw --info --chat-stdin
```

Or just let the agent fetch a model: `hako` → `:pull hako-sho`.

## How it works

1. **Offline conversion** - `tools/gguf2mlf.py` reads a GGUF and emits an **MLF2**
   file: the native container. Quant blocks copied verbatim (ggml's k-quant
   layout, preserved), tokenizer (vocab + merges) embedded, arch params in a fixed
   128-byte header. Pure stdlib; no torch/llama.cpp dependency.

2. **Runtime** (C11, libc + libm + pthread):
   - `loader.c` - mmap the MLF2, resolve tensors by name. RSS = activations + KV.
   - `quant.c` - Q4_K / Q6_K dequant + int8 fast-path dot. Bit-exact vs an
     independent Python reimpl (see `tests/`).
   - `nn.c` - rmsnorm, softmax, SiLU, **NeoX rope**, dequant-on-the-fly + int8 matmul.
   - `model.c` - Qwen2 forward: **GQA**, QKV bias, KV cache, lm_head (tied to
     token_embd on small tiers, separate `output.weight` on 7B+).
   - `bpe.c` - Qwen2 byte-level BPE (GPT2 byte map + rank-greedy merges).
   - `cli/main.c` - the `hakm` CLI (one-shot / chat REPL / `--chat-stdin`).

## Validated

Runs the real **Qwen2.5-Coder 3B and 7B** weights end to end (greedy + sampling,
ChatML, multi-turn REPL):

- **3B** (`hako-sho`): qwen2 · 36 layers · d_model 2048 · 16/2 heads (GQA) · ffn
  11008 · vocab 151936 · tied embeddings.
- **7B** (`hako-koi`): qwen2 · 28 layers · d_model 3584 · 28/4 heads (GQA) · ffn
  18944 · vocab 152064 · **untied** (`output.weight`).

Quant correctness gate: C output bit-exact vs `tests/q_ref.py` (maxdiff 0.0).

## Speed

2.2–2.5 tok/s wall on a 4-core x86_64 (greedy, 3B, Q4_K_M), up from 1.07; a
**~2.1× win** from the int8 fast path: quantize the *activation* to int8 once
(`quantize_row_q8_32`), dot it straight against the 4-bit weights (`q4k_vec_dot`)

 weights stay exact, output bit-identical to the float path. AVX2 kernel
(`_mm256_maddubs_epi16`) + a dormant NEON `vdotq_s32` path for arm64. Matmul is
multithreaded over output rows (persistent pthread pool), built `-march=native`.

measured thread scaling (T=1/2/4 -> 0.73/1.44/2.44) shows it's
**compute-bound near the 4-core ceiling**. Further speed needs *less work*
(speculative decode, smaller draft model) or wider SIMD, not more threads. This is
correctness-first; the goal is owning the stack, not beating tuned GPU runtimes.

## Known gaps / next

- **Tokenizer pretokenizer is simplified** (whitespace-delimited; merges, symbols,
  special tokens correct). Needs the full Qwen2 regex for digit/punctuation-run
  boundaries to exactly match reference tokenization.
- **Windows:** the loader uses POSIX `mmap`; MinGW lacks `sys/mman.h`. An `#ifdef
  _WIN32` mmap shim is the unlock. Linux/macOS/FreeBSD build today.
- Next SIMD: NEON for arm64, tighter AVX2, then speculative decode for the real
  ceiling break.

## License

- **Engine code - GPL-3.0** (`LICENSE`). From-scratch C; no ggml/llama.cpp/torch.
- **Model weights keep their required upstream licenses** (on HuggingFace, not
  here): `hako-koi` (7B) Apache-2.0; `hako-sho` (3B) **Qwen RESEARCH — non-commercial
  only** (the 3B base is research-licensed). See each model repo's `LICENSE`/`NOTICE`.

