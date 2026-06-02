<p align="center">
  <a href="https://mithraeums.github.io">
    <img src="https://mithraeums.github.io/assets/banner-hako-dark.svg" alt="hako — local models trained for the cursor" width="100%"/>
  </a>
</p>

<p align="center">
  <em>Local-first AI models from Mithraeum. Quiet weights for people who write code on their own machine.</em>
</p>

<p align="center">
  <a href="https://github.com/mithraeums/hako/releases"><img src="https://img.shields.io/badge/version-v0-b89656?style=flat-square&labelColor=14130f" alt="v0"/></a>
  <img src="https://img.shields.io/badge/license-Apache--2.0%20%2F%20MIT-c8c2b2?style=flat-square&labelColor=14130f" alt="Apache 2.0 / MIT"/>
  <img src="https://img.shields.io/badge/runtime-mithraeum-c8c2b2?style=flat-square&labelColor=14130f" alt="mithraeum runtime"/>
  <img src="https://img.shields.io/badge/tiers-4-c8c2b2?style=flat-square&labelColor=14130f" alt="4 tiers"/>
  <img src="https://img.shields.io/badge/offline-forever-c8c2b2?style=flat-square&labelColor=14130f" alt="offline forever"/>
</p>

<p align="center">
  <sub><a href="https://mithraeums.github.io">site</a> &nbsp;·&nbsp; <a href="https://github.com/mithraeums/hako/releases">releases</a> &nbsp;·&nbsp; <a href="https://github.com/mithraeums/hako-code">hako-code</a> &nbsp;·&nbsp; <a href="https://github.com/mithraeums/hako-edit">hako-edit</a> &nbsp;·&nbsp; <a href="https://github.com/mithraeums">org</a></sub>
</p>

<br>

<p align="center">
  <img src="https://github.com/mithraeums/mithraeums.github.io/blob/main/assets/readme-screenshots/hako-code/screenrecord-hako.gif?raw=true" alt="a hako model running via the hako agent" width="82%"/>
</p>

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="https://github.com/mithraeums/mithraeums.github.io/blob/main/assets/readme-screenshots/hako-code/screenshot-splash.png?raw=true" alt="hako agent with a hako model defaulted" width="100%"/><br/>
      <sub><code>hako-sho-stock</code> as the default in <a href="https://github.com/mithraeums/hako-code"><b>hako-code</b></a></sub>
    </td>
    <td align="center" width="50%">
      <img src="https://github.com/mithraeums/mithraeums.github.io/blob/main/assets/readme-screenshots/hako-code/screenshot-models.png?raw=true" alt=":models output" width="100%"/><br/>
      <sub><code>:models</code> · live stock-wraps + queued fine-tunes</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="https://github.com/mithraeums/mithraeums.github.io/blob/main/assets/readme-screenshots/hako-edit/screenshot-rei.png?raw=true" alt="koi inside hake editor" width="100%"/><br/>
      <sub>a hako model inside <a href="https://github.com/mithraeums/hako-edit"><b>hake</b></a> (the editor)</sub>
    </td>
    <td align="center" width="50%">
      <!-- TODO: hakm-specific captures (hakm-splash / hakm-run / hakm-list) coming. Until then, themes shot stands in to show suite-wide palette consistency. -->
      <img src="https://github.com/mithraeums/mithraeums.github.io/blob/main/assets/readme-screenshots/hako-edit/screenshot-themes.png?raw=true" alt="suite-wide palette" width="100%"/><br/>
      <sub>shared mithraeum palette across the suite</sub>
    </td>
  </tr>
</table>

> **Engine captures pending.** `hakm` run/chat shots will replace the reused
> suite images above as they're recorded. The stock-wraps (`hako-sho-stock` 3B,
> `hako-koi-mini-stock` 7B) run on the native engine end-to-end today.

<br>

**hako** is a growing model family.

```
╭───────────────────────────────────────────╮
│        ⠀⠀⠀⠀⠀⠀⠀⢀⣀⡀⠀⠀⠀⠀⠀⠀⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀         │
│        ⠀⠀⠀⢀⣠⣶⣾⣿⣿⣿⣦⡀⠀⠀⢀⣴⣿⣿⣿⣷⣶⣄⡀⠀⠀⠀         │
│        ⣠⣴⣾⣿⣿⣿⣿⣿⣿⠿⠛⠉⢠⡄⠉⠛⠿⣿⣿⣿⣿⣿⣿⣷⣦⣄         │
│        ⠀⠻⣿⣿⣿⡿⠟⠋⠁⠀⠀⠀⢸⡇⠀⠀⠀⠈⠙⠻⢿⣿⣿⣿⠟⠁         │
│        ⠀⠀⠈⠋⠁⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠈⠙⠁⠀⠀         │
│        ⠀⠀⣰⣷⣦⣄⡀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⢀⣠⣴⣾⣆⠀⠀         │
│        ⢠⣾⣿⣿⣿⣿⣿⣷⣦⣄⠀⠀⢸⡇⠀⠀⣠⣴⣾⣿⣿⣿⣿⣿⣷⡄         │
│        ⠙⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠖⠀⠀⠲⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠋         │
│        ⠀⠀⠀⠉⠛⠿⣿⣿⣿⠟⢁⣴⡇⢸⣦⡈⠻⣿⣿⣿⠿⠛⠉⠀⠀⠀         │
│        ⠀⠀⠀⢸⣷⣦⣄⡉⢁⣴⣿⣿⡇⢸⣿⣿⣦⡈⢉⣠⣴⣾⡇⠀⠀⠀         │
│        ⠀⠀⠀⠈⠙⠻⢿⣿⣿⣿⣿⣿⡇⢸⣿⣿⣿⣿⣿⡿⠟⠋⠁⠀⠀⠀         │
│        ⠀⠀⠀⠀⠀⠀⠀⠈⠙⠻⢿⣿⡇⢸⣿⡿⠟⠋⠁⠀⠀⠀⠀⠀⠀⠀         │
│        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠁⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀      
│                                           │
│ hako 0.1.6 · mithraeum · hako-sho-stock   │
│ trust on · session resumed                │
│ :help :providers :models :login :theme    │
╰───────────────────────────────────────────╯
```
```
 ▄█████▄
██ ███ ██ sho       : mini       · Qwen2.5-Coder-3B  · stock-wrap live
█████████ koi-mini  : mid-small  · Qwen2.5-Coder-7B  · stock-wrap live
█████████ koi       : mid        · 14B / 32B target  · fine-tune queued
▀█▀▀█▀▀█▀ samurai   : max        · 50B+ reserved     · waits on hardware
```

<p align="center"><sub><b>—— I ——</b></sub></p>

## Overview

- **Local first.** Models run on your device. Offline. No cloud inference. No API calls at runtime.
- **Four tiers.** **mini** (sho, 3B, phone-reachable), **mid-small** (koi-mini, 7B, M-class iPad and laptop), **mid** (koi, 14B/32B target, desktop), **max** (50B+ reserved). Quantization is the brand promise: older hardware stays useful.
- **Honest framing.** Stock-wrap is labeled stock-wrap. Fine-tune is labeled fine-tune. Every Modelfile SYSTEM and model card says exactly what it is.
- **Native engine, zero deps.** `hako/engine/` is a from-scratch C inference runtime — own GGUF→MLF2 loader, own Q4_K/Q6_K dequant + matmul kernels (int8 fast path, AVX2), own BPE tokenizer. **No llama.cpp, no ggml, no torch, no ollama at runtime.** Runs the real Qwen2.5-Coder weights end to end. Links into `hako` (the agent) as `libhakm.a` and runs **in-process** — no subprocess, no server, no port.
- **Wired into the suite.** `hako` (the agent) auto-defaults to whichever `hako-*-v*` is installed (preference: koi > koi-mini > sho). `hake` (the editor) embeds the agent in a split pane. Zero config when all three are present.
- **Tool-use ready.** Each Modelfile SYSTEM teaches the `<tool name="...">{...}</tool>` schema the agent parses. Works on a 3B model, not just chat.
- **ChatML throughout.** `<|im_start|>` / `<|im_end|>` template. No vendor lock.
- **Naming.** Stock-wraps carry no version (`hako-sho-stock`). The first real fine-tune of a tier earns `v0.0.1` (`hako-koi-v0.0.1`). `hakm` resolves shorthand: `sho` → `hako-sho-stock`, `koi-mini` → `hako-koi-mini-stock`; `koi` and `samurai` are queued.

<p align="center"><sub><b>—— II ——</b></sub></p>

## Install

### Build the engine

```sh
git clone https://github.com/mithraeums/hako.git && cd hako/engine
make                        # builds ./hakm — libc + libm + pthread only, no deps
```

### Get weights (one-time, offline)

The runtime loads `.mlf2` files — our native container. Convert any Qwen2.5-Coder
GGUF (from HuggingFace, or an existing ollama blob) with the pure-stdlib converter:

```sh
python3 tools/gguf2mlf.py qwen2.5-coder-3b-instruct-q4_k_m.gguf hako-sho-stock.mlf2
mkdir -p ~/.hako/models && mv hako-sho-stock.mlf2 ~/.hako/models/
```

No ollama needed — the GGUF is just a byte source for the one-time conversion.

### Run

```sh
./hakm ~/.hako/models/hako-sho-stock.mlf2 -n 5 "The capital of France is"
# hakm: hako engine · arch qwen2 · 3.09B params · 36 layers · ...
# Paris

./hakm ~/.hako/models/hako-sho-stock.mlf2 --sys "You are hako."   # chat REPL
```

Drop a `.mlf2` in `~/.hako/models/` and the `hako` agent picks it up automatically
(provider `mithraeum`, in-process).

<p align="center"><sub><b>—— III ——</b></sub></p>

## Tiers

All tiers run on the **native hako engine** (`engine/`) — no ollama, no llama.cpp.

| Tier | Project | Base | Today | Target | Hardware |
|---|---|---|---|---|---|
| **mini** | [sho](sho/) | Qwen2.5-Coder-3B-Instruct | stock-wrap, runs on the engine | 3B fine-tune on mithraeum docs | iPhone, iPad, iSh, anything CPU |
| **mid-small** | [koi-mini](koi/) | Qwen2.5-Coder-7B-Instruct | stock-wrap (same arch, converts unchanged) | 7B fine-tune | M-class iPad, any laptop |
| **mid** | [koi](koi/) | Qwen2.5-Coder-14B (or 24B) | not pulled | 32B fine-tune on rented A100 | desktop, workstation |
| **max** | samurai | TBD | not pulled | 50B (later 128B, 200B) | workstation, dGPU |

### Model catalog

| Tag | Tier | What | Status |
|---|---|---|---|
| `hako-sho-stock` | mini | Qwen2.5-Coder-3B-Instruct + hako SYSTEM wrap | **live** |
| `hako-koi-mini-stock` | mid-small | Qwen2.5-Coder-7B-Instruct + hako SYSTEM wrap | **live** |
| `hako-sho-v0.0.1` | mini | 3B fine-tune on mithraeum docs + curated code corpus | queued |
| `hako-koi-mini-v0.0.1` | mid-small | same recipe at 7B | queued |
| `hako-koi-v0.0.1` | mid | 14B/32B base + LoRA fine-tune via rented A100 | queued |
| `hako-samurai-v0.0.1` | max | 50B+ base + fine-tune | reserved, hardware-blocked |

Scratch-from-scratch research (byte-level GPT, own C17 inference, `.mlf` format) lives in `experimental/` and is gitignored. Not part of the release pipeline.

<p align="center"><sub><b>—— IV ——</b></sub></p>

## Use

### Standalone via the engine

```sh
cd engine && make
./hakm ~/.hako/models/hako-sho-stock.mlf2 "explain ring buffers in C"   # one-shot
./hakm ~/.hako/models/hako-sho-stock.mlf2 --sys "You are hako."         # chat REPL
# flags: -n new-tokens  -t temp(0=greedy)  -p top_p  -k top_k  --raw  --info
```

Convert weights once with `tools/gguf2mlf.py` (see Install). The `.mlf2` is our
native container — k-quant blocks copied verbatim, tokenizer embedded, ~1.9 GB
for the 3B.

### Via [hako-code](https://github.com/mithraeums/hako-code) (the agent)

If any installed `hako-*` model is in the local list, `hako` auto-defaults on first launch (preference order: koi > koi-mini > sho, fine-tunes over stock-wraps), no config:

```sh
hako                             # detects local hako model, defaults to mithraeum provider
›  list the files in this directory
◆  <tool name="list_dir">{"path": "."}</tool>
r list_dir(.)
  README.md  hakm  koi/  sho/  LICENSE
◆  README, the hakm CLI, two model dirs, and a LICENSE file.
```

Or set explicitly:

```sh
:provider mithraeum
:model hako-sho-stock           # or hako-koi-mini-stock / hako-koi-v0.0.1 when shipped
```

### Via [hake](https://github.com/mithraeums/hako-edit) (the editor)

Open `hake`, drop into the AI pane (`Ctrl-W r`), set `:provider mithraeum` `:model hako-sho-stock` (or any installed `hako-*` model). Tool calls run in the editor's split pane against the file you're looking at.

<p align="center"><sub><b>—— V ——</b></sub></p>

## Layout

```
hako/                          (this repo, mithraeums/hako)
├─ engine/                     native C inference runtime — the deliverable
│   ├─ src/                    loader, quant (Q4_K/Q6_K + int8 dot), nn, model, bpe, sample
│   ├─ hakm_api.{h,c}          in-process embed API → libhakm.a (linked by hako-code)
│   ├─ cli/main.c              hakm CLI (one-shot / chat REPL)
│   ├─ tools/gguf2mlf.py       pure-stdlib GGUF → MLF2 converter
│   └─ Makefile                make hakm | make lib | make test_api
├─ koi/                        mid + mid-small fine-tune recipes (axolotl/QLoRA)
├─ sho/                        mini tier notes
└─ experimental/               from-scratch research, gitignored, not shipping
```

<p align="center"><sub><b>—— VI ——</b></sub></p>

## Design constraints (shared across tiers)

- **ChatML** template (`<|im_start|>` / `<|im_end|>`)
- **Tool-call format** compatible with [hako-code](https://github.com/mithraeums/hako-code)'s `HK_TOOLS` schema (prose `<tool name="X">{...}</tool>` parsed at the agent boundary)
- **Local-only inference.** No cloud at runtime.
- **Honest framing** in every Modelfile SYSTEM and model card. Stock is stock. Fine-tunes name their base and corpus.
- **C-first runtime, zero deps.** The `engine/` runs the models with nothing but
  libc + libm + pthread — own loader, own Q4_K/Q6_K dequant + matmul (int8 fast
  path, AVX2 today, NEON for arm64), own BPE. No llama.cpp, no ggml, no torch, no
  ollama. Honest on speed: it's correctness-first (~2-3 tok/s on a 3B, 4-core
  Intel) — the point is owning the whole stack, not beating tuned GPU runtimes.

<p align="center"><sub><b>—— VII ——</b></sub></p>

## The suite

| Product | CLI | Source | Role |
|---|---|---|---|
| Models (this) | `hakm <model.mlf2> <prompt>` | [mithraeums/hako](https://github.com/mithraeums/hako) | native C engine + weights |
| Agent | `hako` | [mithraeums/hako-code](https://github.com/mithraeums/hako-code) | terminal AI agent |
| Editor | `hake` | [mithraeums/hako-edit](https://github.com/mithraeums/hako-edit) | modal terminal editor (embeds the agent) |

<p align="center"><sub><b>—— VIII ——</b></sub></p>

## Roadmap

- **`hako-sho-v0.0.1`** first real fine-tune on mithraeum docs + curated code corpus, 3B base, rented A100.
- **`hako-koi-mini-v0.0.1`** same recipe at 7B.
- **`hako-koi-v0.0.1`** mid-tier debut: 14B/32B base, LoRA on rented A100. Tooling staged in `koi/sft/`.
- **`hako-samurai-v0.0.1`** max tier, 50B+, reserved on hardware.
- **✅ Native engine — SHIPPED.** Own loader, dequant, kernels, tokenizer in C.
  No llama.cpp ever — we skipped the "vendor it first" step and built the whole
  stack. Runs in-process in hako-code via `libhakm.a`. Zero ollama.
- **✅ Own kernels — SHIPPED.** Q4_K/Q6_K dequant + int8-activation × int4-weight
  matmul, AVX2. Next: NEON (arm64), tighter SIMD, speculative decode (sho as a
  draft model) to push past the single-machine compute ceiling.
- **samurai (max)** waits on hardware (target: RTX 5090 + 128GB DDR5).
- **Scale plan** 2x-4x each tier when training budget allows. Eventual lineup roughly 6B sho, 14B-24B koi-mini, 64B-128B koi, 200B max.

<p align="center"><sub><b>—— IX ——</b></sub></p>

## License

- **sho, koi-mini, koi**: Apache 2.0 (matches Qwen2.5-Coder base)
- Top-level `LICENSE` covers shared assets (READMEs, configs, the `hakm` CLI)
- `experimental/` (gitignored, scratch research): MIT when published, not shipping today

<br>

<p align="center"><sub><em>— deus sol invictus mithras —</em></sub></p>

