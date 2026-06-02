# hakm engine — hako native inference

Proprietary local inference for the hako model family. **No ggml, no llama.cpp,
no torch, no ollama at runtime.** We own the loader, the dequant kernels, the
forward pass, and the tokenizer end to end.

Status (2026-05-30): **usable chat working.** Runs the real Qwen2.5-Coder-3B
(`hako-sho-stock`) weights through our own stack — greedy + sampling
(temp/top-k/top-p), ChatML with special tokens, and a multi-turn REPL:

```
$ ./hakm sho.mlf2 -n 5 "The capital of France is"
The capital of France is Paris. Paris is the
$ ./hakm sho.mlf2 -n 24 "def fibonacci(n):"
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
```

## How it works

1. **Offline conversion** — `tools/gguf2mlf.py` reads a GGUF (e.g. an ollama
   blob) and emits an **MLF2** file: our native container. Quant blocks are
   copied verbatim (we keep ggml's k-quant layout — it's good and preserves the
   ~1.9 GB file size), tokenizer (vocab + merges) is embedded, arch params land
   in a fixed 128-byte header. Pure stdlib; no torch/llama.cpp dependency.

2. **Runtime** (C11, libc + libm only):
   - `loader.c` — mmap the MLF2, resolve tensors by name. RSS = activations + KV.
   - `quant.c` — Q4_K / Q6_K dequant + fp16 widen. Bit-exact vs an independent
     Python reimpl (see `tests/`).
   - `nn.c` — rmsnorm, softmax, SiLU, **NeoX rope**, dequant-on-the-fly matmul.
   - `model.c` — Qwen2 forward: **GQA** (16 q-heads / 2 kv-heads), QKV bias,
     KV cache, tied lm_head.
   - `bpe.c` — Qwen2 byte-level BPE (GPT2 byte map + rank-greedy merges).
   - `cli/main.c` — `hakm` greedy decode CLI.

## Build & run

```sh
make                                            # builds ./hakm
python3 tools/gguf2mlf.py model.gguf model.mlf2 # one-time convert

./hakm model.mlf2 --raw -t 0 "def fibonacci(n):"          # raw completion
./hakm model.mlf2 --sys "You are hako." "list comp for squares 0-9"  # one chat turn
./hakm model.mlf2 --sys "You are hako."                    # interactive REPL

# flags: -n new-tokens  -t temp(0=greedy)  -p top_p  -k top_k  -s seed  --raw --info
make test_quant && ./test_quant model.mlf2 blk.0.ffn_gate.weight 256 /tmp/c.bin
python3 tests/q_ref.py model.mlf2 blk.0.ffn_gate.weight 256 /tmp/c.bin  # PASS
```

## Validated facts (Qwen2.5-Coder-3B)

arch qwen2 · 36 layers · d_model 2048 · 16 heads / 2 kv-heads (head_dim 128) ·
ffn 11008 · vocab 151936 · rope_theta 1e6 · rms_eps 1e-6 · QKV bias · tied
embeddings · quant mix Q4_K (216) / Q6_K (37) / F32 (181).

## Speed

~2.2–2.5 tok/s wall on a 4-core x86_64 (greedy, 3B, Q4_K_M), up from ~1.07 — a
**~2.1× win** from the int8 fast path. Matmul is multithreaded over output rows
(pthreads + persistent pool) and built `-march=native`.

**int8-activation × int4-weight dot (shipped):** instead of dequantizing ~3B
params to float every token, we quantize the *activation* to int8 once
(`quantize_row_q8_32`) and dot it straight against the 4-bit weights on the quant
blocks (`q4k_vec_dot`) — weights stay exact, output is bit-identical to the float
path. There's an AVX2 kernel (`_mm256_maddubs_epi16`) and a dormant NEON
`vdotq_s32` path for arm64.

Honest framing: measured thread scaling (T=1/2/4 → 0.73/1.44/2.44) shows it's
**compute-bound near the 4-core ceiling** — not bandwidth or overhead. Further
speed needs *less work* (speculative decode, smaller draft model) or wider SIMD,
not more threads. This is correctness-first; the goal is owning the stack, not
beating tuned GPU runtimes.

## Known gaps / next

- **✅ int8×int4 matmul — shipped** (see Speed). Next SIMD: NEON for arm64,
  tighter AVX2, then speculative decode for the real ceiling break.
- **Tokenizer pretokenizer is simplified** (whitespace-delimited pieces; merge,
  symbol, and special-token injection are correct). Needs the full Qwen2 regex
  for digit/punctuation-run boundaries to exactly match reference tokenization.
- **Embedding via `libhakm.a`** — the engine links directly into hako-code (the
  agent) and runs in-process: no server, no socket, no ollama. (`make lib`.)
- 7B (koi-mini) is the same arch — should convert + run unchanged (more RAM).
