#!/usr/bin/env python3
"""
hf2mlf.py — convert a Hugging Face model folder (safetensors) into MLF2.

The format training tools write, so this is the path for any model that has no
GGUF, our own included. Pure stdlib. Weights are copied at the dtype they hold
(bf16, f16, f32); `hakm quantize` makes the smaller files from the result.

Usage:
    python3 hf2mlf.py <model_dir> <out.mlf2>

model_dir holds config.json, tokenizer.json and model.safetensors (or shards
named by model.safetensors.index.json). Supported: model_type qwen2, qwen3,
llama (byte-level BPE tokenizers: qwen, llama-3, smollm).
"""
import json, os, struct, sys

F32, F16, BF16 = 0, 1, 30
DTYPES = {"F32": (F32, 4), "F16": (F16, 2), "BF16": (BF16, 2)}
ARCHES = {"qwen2": 0, "llama": 1, "qwen3": 2}
TT_NORMAL, TT_CONTROL, TT_USER, TT_UNUSED = 1, 3, 4, 5

# Hugging Face names -> the engine's (GGUF-style) names
TOP = {"model.embed_tokens.weight": "token_embd.weight",
       "model.norm.weight": "output_norm.weight",
       "lm_head.weight": "output.weight"}
LAYER = {"input_layernorm.weight": "attn_norm.weight",
         "post_attention_layernorm.weight": "ffn_norm.weight",
         "self_attn.q_proj.weight": "attn_q.weight", "self_attn.q_proj.bias": "attn_q.bias",
         "self_attn.k_proj.weight": "attn_k.weight", "self_attn.k_proj.bias": "attn_k.bias",
         "self_attn.v_proj.weight": "attn_v.weight", "self_attn.v_proj.bias": "attn_v.bias",
         "self_attn.o_proj.weight": "attn_output.weight",
         "self_attn.q_norm.weight": "attn_q_norm.weight",
         "self_attn.k_norm.weight": "attn_k_norm.weight",
         "mlp.gate_proj.weight": "ffn_gate.weight",
         "mlp.up_proj.weight": "ffn_up.weight",
         "mlp.down_proj.weight": "ffn_down.weight"}

def our_name(hf):
    if hf in TOP: return TOP[hf]
    p = hf.split(".")
    if len(p) > 3 and p[0] == "model" and p[1] == "layers" and p[2].isdigit():
        rest = ".".join(p[3:])
        if rest in LAYER: return f"blk.{p[2]}.{LAYER[rest]}"
    return None

def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 100 << 20: raise SystemExit(f"{path}: header of {n} bytes, not a safetensors file")
        hdr = json.loads(f.read(n))
    base = 8 + n
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__": continue
        a, b = v["data_offsets"]
        out[k] = {"file": path, "dtype": v["dtype"], "shape": v["shape"], "off": base + a, "len": b - a}
    return out

def load_weights(d):
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        files = sorted(set(json.load(open(idx))["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    w = {}
    for fn in files: w.update(read_safetensors(os.path.join(d, fn)))
    return w

def tokenizer_blob(d, vocab_size):
    tj = json.load(open(os.path.join(d, "tokenizer.json"), encoding="utf-8"))
    model = tj["model"]
    pre = json.dumps(tj.get("pre_tokenizer")) + json.dumps(tj.get("decoder"))
    if model.get("type") != "BPE" or "ByteLevel" not in pre:
        raise SystemExit("tokenizer is not byte-level BPE: the engine speaks GPT-2 style BPE "
                         "(qwen, llama-3, smollm); SentencePiece is not written yet")
    toks = {i: (TT_NORMAL, t) for t, i in model["vocab"].items()}
    for a in tj.get("added_tokens", []):
        c = a["content"]                  # <|...|> is control even when not marked special (as GGUF does)
        toks[a["id"]] = (TT_CONTROL if a.get("special") or (c.startswith("<|") and c.endswith("|>")) else TT_USER, c)
    n = max(vocab_size, max(toks) + 1)
    merges = [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]
    buf = bytearray(struct.pack("<II", n, len(merges)))
    for i in range(n):
        tt, s = toks.get(i, (TT_UNUSED, f"[PAD{i}]"))
        b = s.encode("utf-8")
        buf += struct.pack("<BI", tt, len(b)) + b
    for m in merges:
        b = m.encode("utf-8")
        buf += struct.pack("<I", len(b)) + b
    return n, buf

def bf16_to_f32(b):
    out = bytearray(len(b) * 2)
    out[2::4] = b[0::2]; out[3::4] = b[1::2]
    return bytes(out)

def f16_to_f32(b):
    n = len(b) // 2
    return struct.pack(f"<{n}f", *struct.unpack(f"<{n}e", b))

def main():
    if len(sys.argv) != 3: raise SystemExit(__doc__)
    d, dst = sys.argv[1], sys.argv[2]
    cfg = json.load(open(os.path.join(d, "config.json")))
    mt = cfg.get("model_type")
    if mt not in ARCHES:
        raise SystemExit(f"unsupported model_type {mt!r} (one of {sorted(ARCHES)})")
    act = cfg.get("hidden_act", "silu")
    if act != "silu": raise SystemExit(f"hidden_act {act!r}: the engine runs SwiGLU (silu)")
    rs = cfg.get("rope_scaling")
    if rs and rs.get("rope_type", rs.get("type")) not in (None, "default"):
        raise SystemExit(f"rope_scaling {rs}: not implemented in the engine")

    d_model = cfg["hidden_size"]; n_heads = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_heads)
    head_dim = cfg.get("head_dim") or d_model // n_heads
    n_layers = cfg["num_hidden_layers"]
    ctx = cfg.get("max_position_embeddings", 4096)
    vocab, tokbuf = tokenizer_blob(d, cfg["vocab_size"])

    w = load_weights(d)
    tensors = []
    for hf, t in w.items():
        name = our_name(hf)
        if name is None:
            if hf.endswith("rotary_emb.inv_freq"): continue
            raise SystemExit(f"unknown tensor {hf}: not a weight this engine runs")
        if t["dtype"] not in DTYPES: raise SystemExit(f"{hf}: dtype {t['dtype']} not supported")
        t["name"] = name
        t["dims"] = list(reversed(t["shape"]))        # ggml order: fastest first
        tensors.append(t)
    names = {t["name"] for t in tensors}
    tied = "output.weight" not in names
    if not tied and cfg.get("tie_word_embeddings"):
        tensors = [t for t in tensors if t["name"] != "output.weight"]; tied = True
    flags = (1 if "blk.0.attn_q.bias" in names else 0) | (2 if tied else 0)
    # Hugging Face rope is rotate_half, the neox pairing, for llama too: no ROPE_NORM.
    # (GGUF's llama permutes q/k rows to the adjacent pairing; these weights are unpermuted.)

    for t in tensors:
        ty, es = DTYPES[t["dtype"]]
        if len(t["dims"]) == 1: ty, es = F32, 4         # norms and biases are read as f32
        t["type"] = ty
        nelem = 1
        for x in t["dims"]: nelem *= x
        t["nbytes"] = nelem * es
        if len(t["name"]) >= 48: raise SystemExit(f"tensor name too long: {t['name']}")
    tensors.sort(key=lambda t: (t["name"] != "token_embd.weight", t["name"]))

    print(f"arch={mt} layers={n_layers} d_model={d_model} heads={n_heads} kv_heads={n_kv} "
          f"head_dim={head_dim} ffn={cfg['intermediate_size']} vocab={vocab} ctx={ctx}")
    print(f"rope_theta={cfg.get('rope_theta', 10000.0)} rms_eps={cfg.get('rms_norm_eps', 1e-6)} "
          f"tied={tied} qkv_bias={bool(flags & 1)} qk_norm={'blk.0.attn_q_norm.weight' in names}")

    HDR, TENT = 128, 104
    tok_off = HDR + TENT * len(tensors)
    data_off = (tok_off + len(tokbuf) + 31) // 32 * 32
    cur = 0
    for t in tensors:
        t["rel"] = cur; cur += (t["nbytes"] + 31) // 32 * 32

    bos = cfg.get("bos_token_id"); eos = cfg.get("eos_token_id")
    if isinstance(eos, list): eos = eos[0]
    tmp = dst + ".part"
    with open(tmp, "wb") as o:
        o.write(struct.pack("<4sIIIIIIIIIIffII QQQ II 36s", b"MLF2", 2, ARCHES[mt],
                            n_layers, d_model, n_heads, n_kv, head_dim, cfg["intermediate_size"], ctx, vocab,
                            float(cfg.get("rope_theta", 10000.0)), float(cfg.get("rms_norm_eps", 1e-6)),
                            flags, len(tensors), HDR, tok_off, data_off,
                            bos if bos is not None else 0, eos if eos is not None else 0, b"\0" * 36))
        for t in tensors:
            o.write(t["name"].encode().ljust(48, b"\0"))
            o.write(struct.pack("<II", t["type"], len(t["dims"])))
            o.write(struct.pack("<4Q", *(t["dims"] + [0, 0, 0, 0])[:4]))
            o.write(struct.pack("<QQ", t["rel"], t["nbytes"]))
        o.write(tokbuf)
        o.write(b"\0" * (data_off - o.tell()))
        for i, t in enumerate(tensors):
            o.seek(data_off + t["rel"])
            with open(t["file"], "rb") as f:
                f.seek(t["off"])
                if t["type"] == F32 and t["dtype"] != "F32":
                    b = f.read(t["len"])
                    o.write(bf16_to_f32(b) if t["dtype"] == "BF16" else f16_to_f32(b))
                else:
                    left = t["len"]
                    while left:
                        b = f.read(min(left, 64 << 20))
                        if not b: raise SystemExit(f"short read on {t['name']}")
                        o.write(b); left -= len(b)
            if sys.stderr.isatty(): print(f"\r  {i + 1}/{len(tensors)} tensors", end="", file=sys.stderr)
        o.seek(data_off + cur); o.truncate()
    if sys.stderr.isatty(): print("\r\x1b[K", end="", file=sys.stderr)
    os.replace(tmp, dst)
    print(f"wrote {dst}  ({os.path.getsize(dst) / 1e9:.2f} GB, {len(tensors)} tensors)")

if __name__ == "__main__":
    main()
