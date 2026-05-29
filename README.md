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

> **hakm CLI captures pending.** `hakm run`, `hakm list`, `hakm pull` shots will replace the reused suite images above as they're recorded. The agent + editor stand-ins demonstrate the live stock-wraps (`hako-sho-stock`, `hako-koi-mini-stock`) working end-to-end today.

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
- **One CLI: `hakm`.** Standalone runner. `hakm run sho "..."`, `hakm chat koi-mini`, `hakm list`, `hakm pull sho`. Shell stub today, native runner in v0.1.7.
- **Wired into the suite.** `hako` (the agent) auto-defaults to whichever `hako-*-v*` is installed (preference: koi > koi-mini > sho). `hake` (the editor) embeds the agent in a split pane. Zero config when all three are present.
- **Tool-use ready.** Each Modelfile SYSTEM teaches the `<tool name="...">{...}</tool>` schema the agent parses. Works on a 3B model, not just chat.
- **ChatML throughout.** `<|im_start|>` / `<|im_end|>` template. No vendor lock.
- **Naming.** Stock-wraps carry no version (`hako-sho-stock`). The first real fine-tune of a tier earns `v0.0.1` (`hako-koi-v0.0.1`). `hakm` resolves shorthand: `sho` → `hako-sho-stock`, `koi-mini` → `hako-koi-mini-stock`; `koi` and `samurai` are queued.

<p align="center"><sub><b>—— II ——</b></sub></p>

## Install

### One-line install (recommended)

```sh
curl -fsSL https://mithraeums.github.io/hakm.sh | sh
```

Installs `hakm` + pulls the live stock-wraps: `hako-sho-stock` (3B) and `hako-koi-mini-stock` (7B). Run `hakm run sho "hello"` immediately after.

> **v0.1.6 note:** transport is ollama today (transitional). The native `hakm-server` lands v0.1.7 and drops every external runtime dep.

### Manual

```sh
# 1. clone
git clone https://github.com/mithraeums/hako.git && cd hako

# 2. install hakm into ~/.local/bin (or anywhere on PATH)
install -m 0755 hakm ~/.local/bin/hakm

# 3. build a stock-wrap (base + bundled Modelfile)
hakm pull sho               # or: hakm pull koi-mini
```

### Verify

```sh
hakm --version              # hakm 0.1.0
hakm list                   # NAME                  ID   SIZE   MODIFIED
                            # hako-sho-stock        ...
hakm run sho "write fib"    # streams response
```

<p align="center"><sub><b>—— III ——</b></sub></p>

## Tiers

| Tier | Project | Base | Today | Target | Runtime | Hardware |
|---|---|---|---|---|---|---|
| **mini** | [sho](sho/) | Qwen2.5-Coder-3B-Instruct | stock-wrap | 3B fine-tune on mithraeum docs | ollama (hakm-server v0.1.7) | iPhone, iPad, iSh, anything CPU |
| **mid-small** | [koi-mini](koi/) | Qwen2.5-Coder-7B-Instruct | stock-wrap | 7B fine-tune | ollama (hakm-server v0.1.7) | M-class iPad, any laptop |
| **mid** | [koi](koi/) | Qwen2.5-Coder-14B (or 24B) | not pulled | 32B fine-tune on rented A100 | ollama, then hakm-server | desktop, workstation |
| **max** | samurai | TBD | not pulled | 50B (later 128B, 200B) | hakm-server | workstation, dGPU |

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

### Standalone via `hakm`

```sh
hakm run sho "explain ring buffers in C"
hakm chat koi-mini               # interactive REPL
hakm list                        # installed hako-* models
hakm pull sho                    # download base + create hako-sho-stock
hakm models                      # full catalog
```

`hakm <short>` resolves shorthand to the current tag (`sho` → `hako-sho-stock`, `koi-mini` → `hako-koi-mini-stock`, `koi` → `hako-koi-v0.0.1` when shipped).

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
├─ hakm                        standalone CLI runner (shell stub today, C in v0.2)
├─ koi/                        mid + mid-small tiers, Qwen-based, mithraeum runtime
│   ├─ sft/                    LoRA fine-tune pipeline (axolotl)
│   ├─ models/                 Modelfiles + (gitignored) GGUF weights
│   │   └─ hako-koi-mini-stock/   7B stock-wrap, tool-aware SYSTEM
│   └─ data/                   fine-tune datasets (gitignored)
├─ sho/                        mini tier, 3B stock-wrap today
│   └─ models/
│       └─ hako-sho-stock/        3B stock-wrap, tool-aware SYSTEM
└─ experimental/               from-scratch research, gitignored, not shipping
    ├─ nano/                   byte-level GPT trainer (Python + torch)
    ├─ engine/                 C17 inference, libc + libm, mmap'd .mlf
    ├─ models/                 .mlf weight files
    └─ data/                   pretrain corpora
```

<p align="center"><sub><b>—— VI ——</b></sub></p>

## Design constraints (shared across tiers)

- **ChatML** template (`<|im_start|>` / `<|im_end|>`)
- **Tool-call format** compatible with [hako-code](https://github.com/mithraeums/hako-code)'s `HK_TOOLS` schema (prose `<tool name="X">{...}</tool>` parsed at the agent boundary)
- **Local-only inference.** No cloud at runtime.
- **Honest framing** in every Modelfile SYSTEM and model card. Stock is stock. Fine-tunes name their base and corpus.
- **C-first runtime.** Ollama is the transport today. `hakm-server` (v0.1.7) replaces it: minimal llama.cpp subset, CPU-only, drops the runtime dep. v0.2 swaps the vendored kernels for hand-tuned RoPE / RMSNorm / SwiGLU / attention / Q4_K dequant in AVX2 + NEON.

<p align="center"><sub><b>—— VII ——</b></sub></p>

## The suite

| Product | CLI | Source | Role |
|---|---|---|---|
| Models (this) | `hakm <model> <prompt>` | [mithraeums/hako](https://github.com/mithraeums/hako) | local weights + runner |
| Agent | `hako` | [mithraeums/hako-code](https://github.com/mithraeums/hako-code) | terminal AI agent |
| Editor | `hake` | [mithraeums/hako-edit](https://github.com/mithraeums/hako-edit) | modal terminal editor (embeds the agent) |

<p align="center"><sub><b>—— VIII ——</b></sub></p>

## Roadmap

- **`hako-sho-v0.0.1`** first real fine-tune on mithraeum docs + curated code corpus, 3B base, rented A100.
- **`hako-koi-mini-v0.0.1`** same recipe at 7B.
- **`hako-koi-v0.0.1`** mid-tier debut: 14B/32B base, LoRA on rented A100. Tooling staged in `koi/sft/`.
- **`hako-samurai-v0.0.1`** max tier, 50B+, reserved on hardware.
- **Native engine (v0.1.7)** `hakm-server`: vendored minimal llama.cpp subset for Qwen2 Q4_K_M, CPU-only. Drops ollama runtime dep.
- **Own kernels (v0.2)** RoPE / RMSNorm / SwiGLU / attention / Q4_K dequant, hand-tuned AVX2 + NEON. Drops every line of vendored code.
- **mithraeum-native provider in hako-code** today `:provider mithraeum` aliases to ollama transport; v0.1.7 swaps to `hakm-server` over a local port.
- **samurai (max)** waits on hardware (target: RTX 5090 + 128GB DDR5).
- **Scale plan** 2x-4x each tier when training budget allows. Eventual lineup roughly 6B sho, 14B-24B koi-mini, 64B-128B koi, 200B max.

<p align="center"><sub><b>—— IX ——</b></sub></p>

## License

- **sho, koi-mini, koi**: Apache 2.0 (matches Qwen2.5-Coder base)
- Top-level `LICENSE` covers shared assets (READMEs, configs, the `hakm` CLI)
- `experimental/` (gitignored, scratch research): MIT when published, not shipping today

<br>

<p align="center"><sub><em>— deus sol invictus mithras —</em></sub></p>

