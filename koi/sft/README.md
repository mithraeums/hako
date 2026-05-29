# koi/sft — supervised fine-tune pipeline

Scaffold for `hako-koi-v0.0.1`: LoRA fine-tune of Qwen2.5-Coder-14B-Instruct (koi, mid tier) on hako/mithraeum code + public code-instruct sets.

**Status: dormant.** Sits ready until budget unlocks (~$20 on Vast.ai A100 80GB spot).

## Files
- `prep_dataset.py` — builds `dataset.jsonl` from mithraeum repo + public seeds (Magicoder, Code-Alpaca) + personality bumpers
- `qwen2.5-coder-14b-qlora.yaml` — axolotl QLoRA config, A100 80GB sized
- `Modelfile.tmpl` — ollama Modelfile for merged+quantized result
- `RUN.md` — one-page Vast.ai recipe end-to-end

## Quick path
```bash
python3 prep_dataset.py --repo-root ../../.. --include-public --out dataset.jsonl
# ...rent A100, see RUN.md for the rest
```

## Live today (no fine-tune yet)
- `hako-sho-stock` — Qwen2.5-Coder-3B-Instruct + hako SYSTEM wrap (mini). See `../../sho/models/hako-sho-stock/`.
- `hako-koi-mini-stock` — Qwen2.5-Coder-7B-Instruct + hako SYSTEM wrap (mid-small). See `../models/hako-koi-mini-stock/`.

The koi (mid) tier ships its first weights as `hako-koi-v0.0.1` when this pipeline runs.
