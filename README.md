# miniLocal

Train a transformer from scratch. Distill and quantize it down to something that runs on **iSH** — x86 user-mode Linux on an iPhone — at ≥2 tok/s from mmap'd weights, in under 500MB RAM.

No cloud inference. No API calls. The model runs on the device, offline, forever.

```
corpus ──► teacher (fp32) ──► distill ──► student (≤300M) ──► INT4 quant ──► iSH
           train from scratch              ≤300M params         <500MB on disk
```

---

## Why

Most "run LLMs locally" projects still need a GPU, 16GB RAM, or a fast laptop. iSH is the hard constraint that forces real smallness. If it runs there, it runs anywhere CPU-only.

The student model ships with a standard ChatML template and JSON tool-call format, so any agent runtime can use it without modification.

---

## Architecture

| Component | Description |
|-----------|-------------|
| **nano/** | Single-file byte-level GPT, ~250 lines. Trains in <10 min on CPU. The executable spec — never deleted. |
| **train/** | Production trainer (`pretrain.py`) + distillation (`distill.py`) + SFT (`sft.py`). |
| **engine/** | C17 inference engine. Zero deps beyond libc/libm. Loads `.mlf` files via mmap. |
| **quant/** | INT8 and sub-4-bit quantization scripts + calibration. |
| **data/** | Corpus pipeline: filter, dedup, tokenize. |
| **tokenizer/** | BPE trainer (16k vocab) + export to `.mlf` section. C decoder at inference. |

**Model family:**

| Model | Params | Format | Target |
|-------|--------|--------|--------|
| nano | ~1.1M | fp32 | smoke harness, always runnable |
| student | ≤300M | INT4 | iSH, ≥2 tok/s, <500MB |
| teacher | TBD at M7 | fp32 | distillation source |

---

## Stack

- **Python** — training, distillation, quantization (`torch`, nothing else for nano)
- **C17** — inference engine, no dependencies, scalar-first kernels
- **CMake** — engine build, cross-compiles clean for iSH (x86 musl)
- **`.mlf`** — miniLocal Format, mmap-friendly weight file (magic `MLF1`)

---

## Milestones

| # | Description | Status |
|---|-------------|--------|
| M0 | Repo + agent scaffold + ARCH spec frozen | ✅ |
| M1 | nano/ end-to-end: train → export → C engine generates | 🔄 in progress |
| M2 | Real BPE tokenizer (16k vocab) replaces byte-level | |
| M3 | Real data pipeline (filtered code corpus) | |
| M4 | Production trainer (`train/pretrain.py`) | |
| M5 | Engine running on iSH at any speed | |
| M6 | INT8 quant, <2% perplexity regression | |
| M7 | Teacher trained, distillation → student | |
| M8 | Sub-4-bit quant, iSH ≥1 tok/s sustained | |
| M8.5 | Agent SFT: tool-call traces + ChatML tuning | |
| M9 | Release: HF + Ollama + model card | |

---

## Quick Start

**Train nano (byte-level, any text file):**
```bash
python3 nano/train.py data.txt --steps 2000
```

**Export to `.mlf`:**
```bash
python3 nano/export.py nano/checkpoints/final.pt nano.mlf
```

**Build the engine:**
```bash
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build
```

**Run inference:**
```bash
engine/build/mini nano.mlf -n 200 -t 0.8 "Once upon"
engine/build/mini nano.mlf --info
```

**Options:**
```
-n <N>      max new tokens (default 200)
-t <float>  temperature, 0 = greedy (default 0.8)
-p <float>  top-p nucleus sampling
-k <int>    top-k
-s <int>    RNG seed
--info      print model info and exit
```

---

## Weight Format

`.mlf` (miniLocal Format) is a flat binary designed for `mmap`. No compression — page faults are the access model.

```
[0:64]   header  magic="MLF1" | version | n_layers | d_model | n_heads |
                 ffn_dim | ctx | vocab | scheme | n_params | reserved
[64:]    weights fp32 (v0) or quant-native (v1+), row-major, contiguous
```

v0 (nano) and v1 (student) are both supported by the engine forever.

---

## Design Constraints

- Engine: no malloc in the per-token hot path after warmup
- Engine: no SIMD assumed — scalar kernels are the iSH path, AVX2 is a runtime-dispatched bonus
- Engine: KV cache is the only large runtime allocation, ring buffer, fixed at `mini_open()`
- Weights: mmap'd read-only, RSS = activations + KV cache only
- Chat template: standard ChatML — `<|im_start|>` / `<|im_end|>`
- Tool calls: `<|tool_call|>{"name": ..., "arguments": {...}}<|tool_call|>`
- Grammar-guided sampling (GBNF) enforces valid JSON tool calls at M5+

---

## Philosophy

nanoGPT energy. Smallness is a virtue, not a limitation.

`nano/` exists so the entire pipeline — train, export, load, generate — is runnable in under 10 minutes on a laptop CPU, with no infra, forever. Every milestone replaces one nano component with a real one. The end-to-end loop is never broken.

No GPU required. No cloud required. No framework required at inference.

---

## License

MIT
