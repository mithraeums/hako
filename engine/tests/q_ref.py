#!/usr/bin/env python3
"""
q_ref.py — independent Python reference for Q4_K / Q6_K dequant.

Reads the same MLF2 tensor the C test dequants, recomputes it from the ggml
block layout, and asserts the C output (a float32 .bin) matches within tol.

    python3 q_ref.py <model.mlf2> <tensor-name> <nelem> <c_out.bin>
"""
import struct, sys

QK_K = 256

def f16(u):
    s = (u >> 15) & 1; e = (u >> 10) & 0x1F; m = u & 0x3FF
    if e == 0:
        v = m * 2**-24
    elif e == 0x1F:
        v = float("inf") if m == 0 else float("nan")
    else:
        v = (1 + m / 1024.0) * 2.0 ** (e - 15)
    return -v if s else v

def get_scale_min_k4(j, q):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
    m = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4)
    return d, m

def deq_q4k(buf, nelem):
    out = []
    nb = nelem // QK_K
    o = 0
    for _ in range(nb):
        d    = f16(struct.unpack_from("<H", buf, o)[0])
        dmin = f16(struct.unpack_from("<H", buf, o + 2)[0])
        sc = buf[o + 4 : o + 16]
        qs = buf[o + 16 : o + 144]
        o += 144
        y = [0.0] * QK_K
        qi = 0; yi = 0; is_ = 0
        for _j in range(0, QK_K, 64):
            s1, m1 = get_scale_min_k4(is_ + 0, sc)
            s2, m2 = get_scale_min_k4(is_ + 1, sc)
            d1, mm1 = d * s1, dmin * m1
            d2, mm2 = d * s2, dmin * m2
            for l in range(32): y[yi + l]      = d1 * (qs[qi + l] & 0xF) - mm1
            for l in range(32): y[yi + 32 + l] = d2 * (qs[qi + l] >> 4)  - mm2
            qi += 32; yi += 64; is_ += 2
        out.extend(y)
    return out

def deq_q6k(buf, nelem):
    out = []
    nb = nelem // QK_K
    o = 0
    for _ in range(nb):
        ql = buf[o : o + 128]
        qh = buf[o + 128 : o + 192]
        sc = struct.unpack_from("<16b", buf, o + 192)
        d  = f16(struct.unpack_from("<H", buf, o + 208)[0])
        o += 210
        y = [0.0] * QK_K
        ybase = 0; qlo = 0; qho = 0; sco = 0
        for _n in range(0, QK_K, 128):
            for l in range(32):
                is_ = l // 16
                q1 = ((ql[qlo + l]      & 0xF) | (((qh[qho + l] >> 0) & 3) << 4)) - 32
                q2 = ((ql[qlo + l + 32] & 0xF) | (((qh[qho + l] >> 2) & 3) << 4)) - 32
                q3 = ((ql[qlo + l]      >> 4)  | (((qh[qho + l] >> 4) & 3) << 4)) - 32
                q4 = ((ql[qlo + l + 32] >> 4)  | (((qh[qho + l] >> 6) & 3) << 4)) - 32
                y[ybase + l]      = d * sc[sco + is_ + 0] * q1
                y[ybase + l + 32] = d * sc[sco + is_ + 2] * q2
                y[ybase + l + 64] = d * sc[sco + is_ + 4] * q3
                y[ybase + l + 96] = d * sc[sco + is_ + 6] * q4
            ybase += 128; qlo += 64; qho += 32; sco += 8
        out.extend(y)
    return out

# ── minimal MLF2 reader ──
def open_mlf2(path):
    f = open(path, "rb")
    hdr = f.read(128)
    (magic, ver, arch, n_layers, d_model, n_heads, n_kv, head_dim, ffn, ctx,
     vocab, rope, eps, flags, n_tensors, tens_off, tok_off, data_off,
     bos, eos) = struct.unpack_from("<4sIIIIIIIIIIffIIQQQII", hdr, 0)
    assert magic == b"MLF2"
    tdir = {}
    f.seek(tens_off)
    for _ in range(n_tensors):
        name = f.read(48).rstrip(b"\x00").decode()
        ttype, nd = struct.unpack("<II", f.read(8))
        dims = struct.unpack("<4Q", f.read(32))
        rel, nbytes = struct.unpack("<QQ", f.read(16))
        tdir[name] = (ttype, dims, rel, nbytes)
    return f, data_off, tdir

def main():
    path, tname, nelem, cbin = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    f, data_off, tdir = open_mlf2(path)
    ttype, dims, rel, nbytes = tdir[tname]
    f.seek(data_off + rel)
    raw = f.read(nbytes)
    if   ttype == 12: ref = deq_q4k(raw, nelem)
    elif ttype == 14: ref = deq_q6k(raw, nelem)
    else: raise SystemExit(f"ref only covers Q4_K/Q6_K, got type {ttype}")

    cdata = open(cbin, "rb").read()
    cvals = struct.unpack(f"<{nelem}f", cdata[: nelem * 4])

    maxdiff = max(abs(a - b) for a, b in zip(ref, cvals))
    print(f"{tname}: type={ttype} nelem={nelem} maxdiff={maxdiff:.3e}")
    print("  ref[:6]:", [f"{v: .5f}" for v in ref[:6]])
    print("  C  [:6]:", [f"{v: .5f}" for v in cvals[:6]])
    if maxdiff > 1e-3:
        print("  FAIL: C and Python dequant disagree"); sys.exit(1)
    print("  PASS")

if __name__ == "__main__":
    main()
