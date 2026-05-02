"""
nano/train.py — byte-level GPT trainer.
ARCH spec: 4 layers, d_model=128, heads=4, ffn=512, ctx=256, vocab=256
Trains in <10 min on a laptop CPU.  GPU optional, not required.

Usage:
    python nano/train.py [data_file]          # data_file defaults to data.txt
    python nano/train.py data.txt --steps 2000
"""
import math, os, sys, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── hyper (ARCH-frozen for nano) ──────────────────────────────────────────────
N_LAYERS     = 4
D_MODEL      = 128
N_HEADS      = 4
FFN_DIM      = 512
CTX          = 256
VOCAB        = 256   # byte-level

BATCH        = 32
LR           = 3e-4
MAX_STEPS    = 5000
WARMUP_STEPS = 200
EVAL_EVERY   = 500
SAVE_EVERY   = 1000
GRAD_CLIP    = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── data ──────────────────────────────────────────────────────────────────────

def load_data(path: str):
	raw   = open(path, "rb").read()
	data  = torch.tensor(list(raw), dtype=torch.long)
	split = int(len(data) * 0.9)
	return data[:split], data[split:]


def get_batch(data: torch.Tensor, batch: int, ctx: int, device: str):
	ix = torch.randint(len(data) - ctx, (batch,))
	x  = torch.stack([data[i:i+ctx]   for i in ix])
	y  = torch.stack([data[i+1:i+ctx+1] for i in ix])
	return x.to(device), y.to(device)

# ── model ─────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
	def __init__(self, d: int, eps: float = 1e-6):
		super().__init__()
		self.eps    = eps
		self.weight = nn.Parameter(torch.ones(d))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
		return self.weight * (x / rms)


def _rope_freqs(head_dim: int, ctx: int, base: float = 10000.0) -> torch.Tensor:
	theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
	t     = torch.arange(ctx, dtype=torch.float32)
	freqs = torch.outer(t, theta)                        # [ctx, head_dim/2]
	return torch.polar(torch.ones_like(freqs), freqs)    # complex64


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
	"""x: [B, T, H, head_dim] → same shape with RoPE applied."""
	B, T, H, D = x.shape
	xc = torch.view_as_complex(x.float().reshape(B, T, H, D // 2, 2))
	xc = xc * freqs[:T].unsqueeze(0).unsqueeze(2)       # broadcast B, H
	return torch.view_as_real(xc).reshape(B, T, H, D).to(x.dtype)


class Attention(nn.Module):
	def __init__(self):
		super().__init__()
		self.head_dim = D_MODEL // N_HEADS
		self.q     = nn.Linear(D_MODEL, D_MODEL, bias=False)
		self.k     = nn.Linear(D_MODEL, D_MODEL, bias=False)
		self.v     = nn.Linear(D_MODEL, D_MODEL, bias=False)
		self.o     = nn.Linear(D_MODEL, D_MODEL, bias=False)
		self.scale = self.head_dim ** -0.5

	def forward(self, x: torch.Tensor, freqs: torch.Tensor,
				mask: torch.Tensor) -> torch.Tensor:
		B, T, _ = x.shape
		q = self.q(x).reshape(B, T, N_HEADS, self.head_dim)
		k = self.k(x).reshape(B, T, N_HEADS, self.head_dim)
		v = self.v(x).reshape(B, T, N_HEADS, self.head_dim)
		q, k   = _apply_rope(q, freqs), _apply_rope(k, freqs)
		q, k, v = (t.transpose(1, 2) for t in (q, k, v))   # [B,H,T,head_dim]
		scores  = (q @ k.transpose(-2, -1)) * self.scale     # [B,H,T,T]
		scores  = scores.masked_fill(
					mask[:T, :T].unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
		scores  = F.softmax(scores.float(), dim=-1).to(x.dtype)
		out     = (scores @ v).transpose(1, 2).reshape(B, T, D_MODEL)
		return self.o(out)


class FFN(nn.Module):
	def __init__(self):
		super().__init__()
		self.gate = nn.Linear(D_MODEL, FFN_DIM, bias=False)
		self.up   = nn.Linear(D_MODEL, FFN_DIM, bias=False)
		self.down = nn.Linear(FFN_DIM, D_MODEL, bias=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
	def __init__(self):
		super().__init__()
		self.attn_norm = RMSNorm(D_MODEL)
		self.attn      = Attention()
		self.ffn_norm  = RMSNorm(D_MODEL)
		self.ffn       = FFN()

	def forward(self, x: torch.Tensor, freqs: torch.Tensor,
				mask: torch.Tensor) -> torch.Tensor:
		x = x + self.attn(self.attn_norm(x), freqs, mask)
		x = x + self.ffn(self.ffn_norm(x))
		return x


class NanoModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.embed  = nn.Embedding(VOCAB, D_MODEL)
		self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
		self.norm   = RMSNorm(D_MODEL)
		head_dim = D_MODEL // N_HEADS
		self.register_buffer("freqs", _rope_freqs(head_dim, CTX))
		self.register_buffer("mask",  torch.tril(torch.ones(CTX, CTX)))

	def forward(self, idx: torch.Tensor) -> torch.Tensor:
		x = self.embed(idx)
		for block in self.blocks:
			x = block(x, self.freqs, self.mask)
		x = self.norm(x)
		return x @ self.embed.weight.T   # tied lm_head

	@torch.no_grad()
	def generate(self, prompt: bytes, max_new: int = 200,
				 temperature: float = 1.0) -> bytes:
		dev  = next(self.parameters()).device
		toks = torch.tensor(list(prompt), dtype=torch.long, device=dev).unsqueeze(0)
		for _ in range(max_new):
			logits = self(toks[:, -CTX:])[0, -1]
			if temperature <= 0.0:
				nxt = logits.argmax(keepdim=True)
			else:
				nxt = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1)
			toks = torch.cat([toks, nxt.unsqueeze(0)], dim=1)
		return bytes(toks[0, len(prompt):].tolist())

# ── training ──────────────────────────────────────────────────────────────────

def cosine_lr(step: int, warmup: int, total: int, peak: float) -> float:
	if step < warmup:
		return peak * step / max(1, warmup)
	ratio = (step - warmup) / max(1, total - warmup)
	return peak * 0.5 * (1.0 + math.cos(math.pi * ratio))


def train(data_path: str, ckpt_dir: str = "nano/checkpoints",
		  max_steps: int = MAX_STEPS):
	os.makedirs(ckpt_dir, exist_ok=True)
	train_data, val_data = load_data(data_path)

	model   = NanoModel().to(DEVICE)
	n_param = sum(p.numel() for p in model.parameters())
	print(f"nano  params={n_param:,}  device={DEVICE}  data={len(train_data):,}B")

	opt = torch.optim.AdamW(model.parameters(), lr=LR,
							betas=(0.9, 0.95), weight_decay=0.1)
	t0  = time.time()

	for step in range(1, max_steps + 1):
		for g in opt.param_groups:
			g["lr"] = cosine_lr(step, WARMUP_STEPS, max_steps, LR)

		x, y   = get_batch(train_data, BATCH, CTX, DEVICE)
		logits = model(x)
		loss   = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))

		opt.zero_grad(set_to_none=True)
		loss.backward()
		nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
		opt.step()

		if step % EVAL_EVERY == 0:
			model.eval()
			xv, yv   = get_batch(val_data, BATCH, CTX, DEVICE)
			with torch.no_grad():
				val_loss = F.cross_entropy(
					model(xv).reshape(-1, VOCAB), yv.reshape(-1))
			elapsed = time.time() - t0
			tps     = step * BATCH * CTX / elapsed
			print(f"step {step:5d}  loss {loss.item():.4f}"
				  f"  val {val_loss.item():.4f}  tok/s {tps:.0f}"
				  f"  lr {opt.param_groups[0]['lr']:.2e}")
			model.train()

		if step % SAVE_EVERY == 0:
			ck = os.path.join(ckpt_dir, f"step_{step:05d}.pt")
			torch.save({"step": step, "model": model.state_dict(),
						"opt":   opt.state_dict()}, ck)
			print(f"  saved {ck}")

	final = os.path.join(ckpt_dir, "final.pt")
	torch.save({"step": max_steps, "model": model.state_dict(),
				"opt":  opt.state_dict()}, final)
	print(f"done → {final}")

	# quick smoke: generate from the empty prompt
	model.eval()
	sample = model.generate(b"", max_new=64, temperature=0.8)
	print("sample:", repr(sample))
	return model


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("data",    nargs="?", default="data.txt")
	ap.add_argument("--steps", type=int,  default=MAX_STEPS)
	ap.add_argument("--ckpt",  default="nano/checkpoints")
	args = ap.parse_args()
	train(args.data, args.ckpt, args.steps)
