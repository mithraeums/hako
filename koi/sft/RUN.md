# SFT RUN — Qwen2.5-Coder-14B → hako-koi-v0.0.1

One-page recipe. Target: Vast.ai A100 80GB spot, ~$15-25 total, ~8-12 hours wall time.

## Prerequisites (locally, before renting)
1. Build dataset:
   ```bash
   cd hako/koi/sft
   python3 prep_dataset.py --repo-root ../../.. --include-public --target-n 5000 --out dataset.jsonl
   ```
   `dataset.jsonl` lands ~10-30MB. Inspect a few rows.
2. **Replace placeholder summaries in `dataset.jsonl`** — `prep_dataset.py` stamps a `[NOTE: replace this placeholder...]` line on synthesized rows. Either: (a) regenerate locally by piping each row through Qwen2.5-Coder-7B-instruct via ollama, or (b) drop the synth-explain rows entirely and rely on public sets + personality. **Real bumper data > fake synth data.**
3. SSH key on Vast.ai account.

## Rent the pod
- Vast.ai → Console → Search → filter: GPU=A100 80GB, image=`pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`, disk≥80GB, spot.
- Pick lowest $/hr (~$1.80 typical). Hit RENT. Wait ~2min for boot.

## On the pod
```bash
# 1. Install axolotl
pip install -U axolotl[flash-attn,deepspeed] huggingface_hub
huggingface-cli login  # paste your HF token if pushing

# 2. Sync data + config from local laptop
# (run on LAPTOP, replace IP/port from Vast.ai console)
rsync -avz -e "ssh -p <PORT>" dataset.jsonl qwen2.5-coder-14b-qlora.yaml root@<IP>:/workspace/

# 3. On pod: train
cd /workspace
axolotl train qwen2.5-coder-14b-qlora.yaml

# Expect: ~8-12 hours for 5k examples × 2 epochs at micro_batch=2, accum=4.
# Watch `nvidia-smi` and the logging line every 10 steps.

# 4. Merge LoRA into base
axolotl merge-lora qwen2.5-coder-14b-qlora.yaml
# Produces ./out-hako-koi-v0.0.1/merged/

# 5. Convert to GGUF Q4_K_M
git clone https://github.com/ggerganov/llama.cpp /tmp/llama.cpp
pip install -r /tmp/llama.cpp/requirements.txt
python3 /tmp/llama.cpp/convert_hf_to_gguf.py ./out-hako-koi-v0.0.1/merged --outfile hako-koi-v0.0.1.fp16.gguf
make -C /tmp/llama.cpp llama-quantize
/tmp/llama.cpp/llama-quantize hako-koi-v0.0.1.fp16.gguf hako-koi-v0.0.1.Q4_K_M.gguf Q4_K_M

# 6. Download to laptop (run on LAPTOP)
rsync -avz -e "ssh -p <PORT>" root@<IP>:/workspace/hako-koi-v0.0.1.Q4_K_M.gguf .

# 7. DESTROY the pod (Vast.ai console → instance → Destroy). Stops billing.
```

## On laptop after download
```bash
# Place gguf next to Modelfile.tmpl
cp hako-koi-v0.0.1.Q4_K_M.gguf hako/koi/sft/
cd hako/koi/sft
ollama create hako-koi-v0.0.1 -f Modelfile.tmpl
ollama run hako-koi-v0.0.1 "who are you?"
```

Should respond as hako with v0.0.1 framing. Wire into hako-code: `:provider ollama` → `:model hako-koi-v0.0.1`.

## Budget bookkeeping
- Pod: $1.80/hr × 10hr = $18
- Egress (rsync gguf out, ~1GB): often free on Vast.ai
- HF storage if pushing: free
- **Worst-case: $25-30.** Set Vast.ai instance auto-stop after 14hr if you might forget to destroy.

## When it fails
- OOM mid-train → drop `micro_batch_size` to 1, raise `gradient_accumulation_steps` to 8
- Loss not dropping → check dataset quality, especially that placeholder rows were stripped/replaced
- ChatML template errors → confirm `chat_template: chatml` in axolotl config, base model has `<|im_start|>` in tokenizer
- ollama refuses gguf → verify Modelfile.tmpl `FROM` path is correct and gguf is valid (`/tmp/llama.cpp/llama-cli -m hako-koi-v0.0.1.Q4_K_M.gguf -p "test"`)
