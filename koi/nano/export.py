"""
nano/export.py — torch checkpoint → nano.mlf (MLF v0, fp32).

Usage:
    python nano/export.py nano/checkpoints/final.pt nano.mlf

MLF v0 header (64 bytes, little-endian):
  [0:4]   magic    "MLF1"
  [4:8]   version  u32 = 0
  [8:12]  n_layers u32
  [12:16] d_model  u32
  [16:20] n_heads  u32
  [20:24] ffn_dim  u32
  [24:28] ctx      u32
  [28:32] vocab    u32
  [32:36] scheme   u32  0=fp32
  [36:44] n_params u64
  [44:64] reserved (zeros)

Weight sections (fp32 LE, contiguous, no padding between tensors):
  embed              [vocab × d_model]
  per layer i in 0..n_layers-1:
    attn_norm.weight [d_model]
    q.weight         [d_model × d_model]   (out_dim × in_dim, row-major)
    k.weight         [d_model × d_model]
    v.weight         [d_model × d_model]
    o.weight         [d_model × d_model]
    ffn_norm.weight  [d_model]
    gate.weight      [ffn_dim × d_model]
    up.weight        [ffn_dim × d_model]
    down.weight      [d_model × ffn_dim]
  final_norm.weight  [d_model]
  (lm_head is tied to embed — not written)
"""
import sys, struct
import torch

_NANO_N_HEADS = 4
_NANO_CTX     = 256


def export(ckpt_path: str, out_path: str) -> None:
	ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
	sd   = ckpt["model"] if "model" in ckpt else ckpt

	vocab, d_model = sd["embed.weight"].shape
	n_layers = sum(1 for k in sd
				   if k.startswith("blocks.") and k.endswith(".attn_norm.weight"))
	ffn_dim  = sd["blocks.0.ffn.gate.weight"].shape[0]
	n_heads  = _NANO_N_HEADS
	ctx      = _NANO_CTX

	def _w(key: str) -> torch.Tensor:
		return sd[key].detach().float().contiguous()

	tensors = [_w("embed.weight")]
	for i in range(n_layers):
		b = f"blocks.{i}"
		tensors += [
			_w(f"{b}.attn_norm.weight"),
			_w(f"{b}.attn.q.weight"),
			_w(f"{b}.attn.k.weight"),
			_w(f"{b}.attn.v.weight"),
			_w(f"{b}.attn.o.weight"),
			_w(f"{b}.ffn_norm.weight"),
			_w(f"{b}.ffn.gate.weight"),
			_w(f"{b}.ffn.up.weight"),
			_w(f"{b}.ffn.down.weight"),
		]
	tensors.append(_w("norm.weight"))

	n_params = sum(int(t.numel()) for t in tensors)

	with open(out_path, "wb") as f:
		# 64-byte header: 4s + 9×I + Q + 20s = 4+36+8+20 = 68 — wrong, recalc:
		# 4s=4, I×8=32 (version+n_layers+d_model+n_heads+ffn_dim+ctx+vocab+scheme) wait
		# fields: magic(4s) version(I) n_layers(I) d_model(I) n_heads(I) ffn_dim(I)
		#         ctx(I) vocab(I) scheme(I) n_params(Q) reserved(20s)
		# = 4 + 8*4 + 8 + 20 = 4 + 32 + 8 + 20 = 64 ✓
		hdr = struct.pack("<4s8IQ20s",
			b"MLF1",
			0,         # version
			n_layers,
			d_model,
			n_heads,
			ffn_dim,
			ctx,
			vocab,
			0,         # scheme = fp32
			n_params,
			b"\x00" * 20,
		)
		assert len(hdr) == 64, f"header size {len(hdr)} != 64"
		f.write(hdr)

		for t in tensors:
			raw = t.detach().float().contiguous()
			try:
				f.write(raw.numpy().astype("<f4").tobytes())
			except Exception:
				import struct as _struct
				vals = raw.flatten().tolist()
				f.write(_struct.pack(f"<{len(vals)}f", *vals))

	size_mb = (64 + n_params * 4) / (1024 * 1024)
	print(f"exported  {out_path}  params={n_params:,}  {size_mb:.2f} MB")
	print(f"  n_layers={n_layers}  d_model={d_model}  n_heads={n_heads}"
		  f"  ffn_dim={ffn_dim}  ctx={ctx}  vocab={vocab}")


if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("usage: python nano/export.py <ckpt.pt> <output.mlf>")
		sys.exit(1)
	export(sys.argv[1], sys.argv[2])
