#!/usr/bin/env python3
"""
gguf2mlf.py — convert a GGUF model into the hako-native MLF2 container.

Offline tooling. Pure stdlib: parses GGUF directly (no `gguf`/`torch`/`llama.cpp`
dependency) and copies quant blocks verbatim. The runtime engine then owns the
dequant math. See ../include/mlf2.h for the on-disk format.

Usage:
    python3 gguf2mlf.py <model.gguf> <out.mlf2>

Supported arch: qwen2.  Supported tensor types: F32, F16, Q4_K, Q6_K
(the exact set Qwen2.5-Coder Q4_K_M ships).
"""
import struct, sys, os

# ── ggml type ids (mirrored in mlf2.h) ───────────────────────────────────────
GGML_F32, GGML_F16, GGML_Q4_K, GGML_Q6_K = 0, 1, 12, 14
TYPE_NAME = {GGML_F32: "F32", GGML_F16: "F16", GGML_Q4_K: "Q4_K", GGML_Q6_K: "Q6_K"}
QK_K = 256
# bytes per block of QK_K=256 elements (or per element for F32/F16)
def type_nbytes(t, nelem):
    if t == GGML_F32:  return 4 * nelem
    if t == GGML_F16:  return 2 * nelem
    if t == GGML_Q4_K: return (nelem // QK_K) * 144
    if t == GGML_Q6_K: return (nelem // QK_K) * 210
    raise SystemExit(f"unsupported tensor type {t}")

# ── GGUF reader ──────────────────────────────────────────────────────────────
class G:
    def __init__(s, path):
        s.f = open(path, "rb")
    def rd(s, n): return s.f.read(n)
    def u32(s):  return struct.unpack("<I", s.rd(4))[0]
    def i32(s):  return struct.unpack("<i", s.rd(4))[0]
    def u64(s):  return struct.unpack("<Q", s.rd(8))[0]
    def i64(s):  return struct.unpack("<q", s.rd(8))[0]
    def f32(s):  return struct.unpack("<f", s.rd(4))[0]
    def f64(s):  return struct.unpack("<d", s.rd(8))[0]
    def gstr(s):
        n = s.u64(); return s.rd(n)
    def val(s, t):
        if t == 8:  return s.gstr()                       # string -> bytes
        if t == 6:  return s.f32()
        if t == 4:  return s.u32()
        if t == 5:  return s.i32()
        if t == 10: return s.u64()
        if t == 11: return s.i64()
        if t == 12: return s.f64()
        if t == 7:  return bool(s.rd(1)[0])
        if t in (0, 1): return s.rd(1)[0]
        if t in (2, 3): return struct.unpack("<H", s.rd(2))[0]
        if t == 9:                                         # array
            et = s.u32(); n = s.u64()
            return [s.val(et) for _ in range(n)]
        raise SystemExit(f"bad gguf value type {t}")

def load_gguf(path):
    g = G(path)
    if g.rd(4) != b"GGUF": raise SystemExit("not a GGUF file")
    ver = g.u32()
    if ver != 3: print(f"warn: gguf version {ver} (expected 3)", file=sys.stderr)
    ntensors = g.u64(); nkv = g.u64()
    md = {}
    for _ in range(nkv):
        k = g.gstr().decode("utf-8"); t = g.u32(); md[k] = g.val(t)
    tensors = []
    for _ in range(ntensors):
        name = g.gstr().decode("utf-8")
        nd = g.u32(); dims = [g.u64() for _ in range(nd)]
        gt = g.u32(); off = g.u64()
        tensors.append({"name": name, "dims": dims, "type": gt, "off": off})
    # data section starts at next alignment boundary after the tensor infos
    align = md.get("general.alignment", 32)
    pos = g.f.tell()
    data_start = (pos + align - 1) // align * align
    return md, tensors, data_start, g

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    md, tensors, data_start, g = load_gguf(src)

    arch = md["general.architecture"].decode()
    if arch != "qwen2":
        raise SystemExit(f"unsupported arch {arch!r} (only qwen2)")

    def m(key): return md[f"qwen2.{key}"]
    n_layers   = m("block_count")
    d_model    = m("embedding_length")
    n_heads    = m("attention.head_count")
    n_kv_heads = m("attention.head_count_kv")
    ffn_dim    = m("feed_forward_length")
    ctx        = m("context_length")
    rms_eps    = m("attention.layer_norm_rms_epsilon")
    rope_theta = md.get("qwen2.rope.freq_base", 1000000.0)
    head_dim   = d_model // n_heads

    names = {t["name"] for t in tensors}
    tied  = "output.weight" not in names
    has_bias = "blk.0.attn_q.bias" in names
    vocab  = len(md["tokenizer.ggml.tokens"])
    bos = md.get("tokenizer.ggml.bos_token_id", 0)
    eos = md.get("tokenizer.ggml.eos_token_id", 0)

    flags = 0
    if has_bias: flags |= 1
    if tied:     flags |= 2

    print(f"arch={arch} layers={n_layers} d_model={d_model} heads={n_heads} "
          f"kv_heads={n_kv_heads} head_dim={head_dim} ffn={ffn_dim} vocab={vocab}")
    print(f"rope_theta={rope_theta} rms_eps={rms_eps} tied={tied} qkv_bias={has_bias}")

    # validate every tensor type is supported, compute nbytes
    for t in tensors:
        nelem = 1
        for d in t["dims"]: nelem *= d
        t["nelem"] = nelem
        t["nbytes"] = type_nbytes(t["type"], nelem)
        if len(t["name"]) >= 48:
            raise SystemExit(f"tensor name too long: {t['name']}")

    # ── build tokenizer blob ──
    toks   = md["tokenizer.ggml.tokens"]          # list[bytes]
    ttypes = md.get("tokenizer.ggml.token_type", [1] * vocab)
    merges = md["tokenizer.ggml.merges"]          # list[bytes]
    tokbuf = bytearray()
    tokbuf += struct.pack("<II", len(toks), len(merges))
    for i, tk in enumerate(toks):
        tt = ttypes[i] if i < len(ttypes) else 1
        tokbuf += struct.pack("<BI", tt & 0xFF, len(tk)) + tk
    for mg in merges:
        tokbuf += struct.pack("<I", len(mg)) + mg

    # ── lay out the file ──
    HDR = 128
    TENT = 48 + 4 + 4 + 32 + 8 + 8           # mlf2_tensor packed size = 104
    assert TENT == 104
    tensors_off   = HDR
    tokenizer_off = tensors_off + TENT * len(tensors)
    raw_off       = tokenizer_off + len(tokbuf)
    data_off      = (raw_off + 32 - 1) // 32 * 32

    # assign each tensor a 32-aligned offset within the data section
    cur = 0
    for t in tensors:
        t["rel"] = cur
        cur += (t["nbytes"] + 31) // 32 * 32
    data_total = cur

    with open(dst, "wb") as o:
        hdr = struct.pack(
            "<4sIIIIIIIIIIffII QQQ II 36s",
            b"MLF2", 2, 0,
            n_layers, d_model, n_heads, n_kv_heads, head_dim,
            ffn_dim, ctx, vocab,
            float(rope_theta), float(rms_eps), flags, len(tensors),
            tensors_off, tokenizer_off, data_off,
            bos, eos, b"\x00" * 36)
        assert len(hdr) == HDR, len(hdr)
        o.write(hdr)
        # tensor directory
        for t in tensors:
            dims = (t["dims"] + [0, 0, 0, 0])[:4]
            o.write(t["name"].encode().ljust(48, b"\x00"))
            o.write(struct.pack("<II", t["type"], len(t["dims"])))
            o.write(struct.pack("<4Q", *dims))
            o.write(struct.pack("<QQ", t["rel"], t["nbytes"]))
        # tokenizer
        o.write(tokbuf)
        # pad to data_off
        o.write(b"\x00" * (data_off - o.tell()))
        # tensor data, copied verbatim from gguf
        base = o.tell()
        for t in tensors:
            g.f.seek(data_start + t["off"])
            blob = g.f.read(t["nbytes"])
            if len(blob) != t["nbytes"]:
                raise SystemExit(f"short read on {t['name']}")
            o.seek(base + t["rel"])
            o.write(blob)
        # ensure file extends to full data section
        o.seek(base + data_total)
        o.truncate()

    sz = os.path.getsize(dst)
    print(f"wrote {dst}  ({sz/1e9:.2f} GB, {len(tensors)} tensors, "
          f"tokenizer {len(tokbuf)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
