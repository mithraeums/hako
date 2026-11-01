#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser
"""mkcorpus.py — build a training corpus for hakm-train from permissively
licensed sources, without cloning anything: each repo's tarball is streamed and
only the wanted files are kept, in memory.

  python3 tools/mkcorpus.py OUTDIR [--english-mb N] [--code-scale F] [--scale F] [--stack c=MB,go=MB,...]

Writes OUTDIR/train.txt, OUTDIR/val.txt (1% of documents, picked by hash) and
OUTDIR/MANIFEST.md (every source, its license and what it gave). Documents are
deduplicated, shuffled, and joined by blank lines. English comes from
FineWeb-Edu (ODC-By), read with HTTP range requests so only the row groups used
are downloaded; it needs pyarrow and is skipped without it. More code can come
from The Stack (dedup v1.2, collected 2022, so no model output), also by range
request; its terms must be accepted on Hugging Face first, and every file's
license is checked again against STACK_OK.

Nothing written by a Claude model goes in, and nothing copyleft: every source
below is public domain or permissive (MIT, BSD, ISC, Apache-2.0, zlib, 0BSD,
curl) or CC BY 4.0. Add a source only after checking its license.
"""
import hashlib, io, os, random, re, sys, tarfile, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

C, GO, RS, ZIG, SH = (".c", ".h"), (".go",), (".rs",), (".zig",), (".sh", ".bash", ".zsh")
# (repo, license, extensions, language, MB cap for this repo)
SOURCES = [
	("sqlite/sqlite", "public domain", C, "c", 14),
	("lua/lua", "MIT", C, "c", 4),
	("curl/curl", "curl (MIT style)", C, "c", 10),
	("libuv/libuv", "MIT", C, "c", 5),
	("valkey-io/valkey", "BSD-3-Clause", C, "c", 10),
	("jqlang/jq", "MIT", C, "c", 3),
	("facebook/zstd", "BSD-3-Clause (dual with GPL-2.0, taken as BSD)", C, "c", 6),
	("landley/toybox", "0BSD", C, "c", 6),
	("tmux/tmux", "ISC", C, "c", 5),
	("nginx/nginx", "BSD-2-Clause", C, "c", 6),
	("bminor/musl", "MIT", C, "c", 6),
	("nothings/stb", "MIT or public domain", C, "c", 4),
	("janet-lang/janet", "MIT", C, "c", 3),
	("mruby/mruby", "MIT", C, "c", 4),
	("madler/zlib", "zlib", C, "c", 2),
	("the-tcpdump-group/tcpdump", "BSD-3-Clause", C, "c", 6),
	("golang/go", "BSD-3-Clause", GO, "go", 30),
	("junegunn/fzf", "MIT", GO, "go", 3),
	("cli/cli", "MIT", GO, "go", 8),
	("caddyserver/caddy", "Apache-2.0", GO, "go", 5),
	("etcd-io/etcd", "Apache-2.0", GO, "go", 10),
	("spf13/cobra", "Apache-2.0", GO, "go", 1),
	("charmbracelet/bubbletea", "MIT", GO, "go", 2),
	("gohugoio/hugo", "Apache-2.0", GO, "go", 8),
	("BurntSushi/ripgrep", "MIT or Unlicense", RS, "rust", 4),
	("tokio-rs/tokio", "MIT", RS, "rust", 8),
	("serde-rs/serde", "MIT or Apache-2.0", RS, "rust", 3),
	("sharkdp/fd", "MIT or Apache-2.0", RS, "rust", 1),
	("sharkdp/bat", "MIT or Apache-2.0", RS, "rust", 2),
	("alacritty/alacritty", "Apache-2.0", RS, "rust", 3),
	("rust-lang/cargo", "MIT or Apache-2.0", RS, "rust", 10),
	("rust-lang/regex", "MIT or Apache-2.0", RS, "rust", 4),
	("clap-rs/clap", "MIT or Apache-2.0", RS, "rust", 4),
	("nushell/nushell", "MIT", RS, "rust", 10),
	("ziglang/zig", "MIT", ZIG, "zig", 25),
	("tigerbeetle/tigerbeetle", "Apache-2.0", ZIG, "zig", 8),
	("ghostty-org/ghostty", "MIT", ZIG, "zig", 8),
	("zigtools/zls", "MIT", ZIG, "zig", 3),
	("ohmyzsh/ohmyzsh", "MIT", SH, "shell", 4),
	("nvm-sh/nvm", "MIT", SH, "shell", 2),
	("bats-core/bats-core", "MIT", SH, "shell", 1),
	("Bash-it/bash-it", "MIT", SH, "shell", 2),
	("rbenv/rbenv", "MIT", SH, "shell", 1),
	("pyenv/pyenv", "MIT", SH, "shell", 2),
	("romkatv/powerlevel10k", "MIT", SH, "shell", 2),
	("kward/shunit2", "Apache-2.0", SH, "shell", 1),
	("dylanaraps/pure-bash-bible", "MIT", (".md",), "shell", 1),
	("dylanaraps/pure-sh-bible", "MIT", (".md",), "shell", 1),
	("tldr-pages/tldr", "CC BY 4.0", (".md",), "shell", 12),
	("google/oss-fuzz", "Apache-2.0", SH, "shell", 6),
	("GameServerManagers/LinuxGSM", "MIT", SH, "shell", 3),
	("dokku/dokku", "MIT", SH, "shell", 3),
	("sdkman/sdkman-cli", "Apache-2.0", SH, "shell", 1),
	("shellspec/shellspec", "MIT", SH, "shell", 2),
	("ohmybash/oh-my-bash", "MIT", SH, "shell", 2),
	("sorin-ionescu/prezto", "MIT", SH, "shell", 1),
	("zimfw/zimfw", "MIT", SH, "shell", 1),
	("zsh-users/zsh-autosuggestions", "MIT", SH, "shell", 1),
	("zsh-users/zsh-syntax-highlighting", "BSD-3-Clause", SH, "shell", 1),
	("tfutils/tfenv", "MIT", SH, "shell", 1),
	("nodenv/nodenv", "MIT", SH, "shell", 1),
	("nodenv/node-build", "MIT", SH, "shell", 1),
	("rbenv/ruby-build", "MIT", SH, "shell", 1),
	("pyenv/pyenv-virtualenv", "MIT", SH, "shell", 1),
	("tj/n", "MIT", SH, "shell", 1),
	("dehydrated-io/dehydrated", "MIT", SH, "shell", 1),
	("docker/docker-install", "Apache-2.0", SH, "shell", 1),
	("Homebrew/install", "BSD-2-Clause", SH, "shell", 1),
	("docker-library/postgres", "MIT", SH, "shell", 1),
	("mathiasbynens/dotfiles", "MIT", SH, "shell", 1),
	("holman/dotfiles", "MIT", SH, "shell", 1),
	("kward/shflags", "Apache-2.0", SH, "shell", 1),
	("dylanaraps/pfetch", "MIT", SH, "shell", 1),
]
SKIP_DIRS = {"vendor", "third_party", "thirdparty", "third-party", "deps", "external", "testdata",
             "node_modules", ".git", "fixtures", "libc", "compat"}
SHEBANG = re.compile(rb"^#!\s*(/usr/bin/env\s+)?(/bin/|/usr/bin/|/usr/local/bin/)?(ba|z|k|da)?sh\b")
FINEWEB = ["https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/%03d_00000.parquet" % i
           for i in range(14)]


def keep_path(repo, path, exts):
	parts = path.split("/")[1:]   # drop the tarball's top directory
	if not parts or any(p in SKIP_DIRS for p in parts[:-1]):
		return False
	if repo == "tldr-pages/tldr":   # English pages only
		return len(parts) == 3 and parts[0] == "pages"
	return True


def clean(raw):
	if not 64 <= len(raw) <= 256 * 1024:
		return None
	try:
		t = raw.decode("utf-8")
	except UnicodeDecodeError:
		return None
	head = t[:1024].lower()
	if ("code generated" in head and "do not edit" in head) or "autogenerated" in head or "auto-generated" in head:
		return None
	lines = t.split("\n")
	if max(len(l) for l in lines) > 400:
		return None
	if sum(ch.isalpha() for ch in t) < 0.25 * len(t):
		return None
	return t.replace("\r\n", "\n").strip("\n") + "\n"


def fetch_repo(src):
	repo, lic, exts, lang, cap = src
	url = f"https://github.com/{repo}/archive/HEAD.tar.gz"
	docs, size, seen = [], 0, 0
	try:
		with urllib.request.urlopen(url, timeout=60) as r, tarfile.open(fileobj=r, mode="r|gz") as tf:
			for m in tf:
				if not m.isfile() or not keep_path(repo, m.name, exts):
					continue
				base = m.name.rsplit("/", 1)[-1]
				by_ext = base.endswith(exts)
				if not by_ext and (exts != SH or "." in base or m.size > 256 * 1024):
					continue
				raw = tf.extractfile(m).read()
				if not by_ext and not SHEBANG.match(raw):
					continue
				seen += 1
				t = clean(raw)
				if t:
					docs.append(t)
					size += len(t)
	except Exception as e:
		print(f"  {repo}: FAILED {e}", file=sys.stderr)
		return src, [], 0
	random.Random(repo).shuffle(docs)
	out, total = [], 0
	for d in docs:
		if total >= cap * 1e6:
			break
		out.append(d); total += len(d)
	print(f"  {repo}: {len(out)}/{seen} files, {total / 1e6:.1f} MB", file=sys.stderr)
	return src, out, seen


class _NoAuthAcrossHosts(urllib.request.HTTPRedirectHandler):
	"""Hugging Face redirects files to a signed CDN URL; the token must not follow."""
	def redirect_request(self, req, fp, code, msg, headers, newurl):
		new = super().redirect_request(req, fp, code, msg, headers, newurl)
		if new is not None and urllib.parse.urlsplit(newurl).hostname != urllib.parse.urlsplit(req.full_url).hostname:
			new.remove_header("Authorization")
		return new


_open = urllib.request.build_opener(_NoAuthAcrossHosts()).open


class HttpFile(io.RawIOBase):
	"""A read-only remote file for pyarrow, fetched by byte range."""
	def __init__(self, url, headers=None):
		self.headers = headers or {}
		with _open(urllib.request.Request(url, method="HEAD", headers=self.headers), timeout=60) as r:
			self.url, self.size = r.url, int(r.headers["Content-Length"])
		if urllib.parse.urlsplit(self.url).hostname != urllib.parse.urlsplit(url).hostname:
			self.headers = {}
		self.pos, self.fetched = 0, 0
	def seekable(self): return True
	def readable(self): return True
	def tell(self): return self.pos
	def seek(self, off, whence=0):
		self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
		return self.pos
	def read(self, n=-1):
		end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
		if end <= self.pos:
			return b""
		req = urllib.request.Request(self.url, headers={**self.headers, "Range": f"bytes={self.pos}-{end - 1}"})
		with urllib.request.urlopen(req, timeout=120) as r:
			b = r.read()
		self.pos += len(b); self.fetched += len(b)
		return b


def fetch_english(mb, emit):
	"""Stream FineWeb-Edu documents to emit() until mb is reached, shard by shard,
	so the English never has to fit in memory."""
	try:
		import pyarrow.parquet as pq
	except ImportError:
		print("  fineweb-edu: no pyarrow, skipped", file=sys.stderr)
		return 0, 0
	ndocs, total, fetched = 0, 0, 0
	for url in FINEWEB:
		f = HttpFile(url)
		pf = pq.ParquetFile(f)
		for g in range(pf.num_row_groups):
			for t in pf.read_row_group(g, columns=["text"]).column("text").to_pylist():
				total += emit(t.strip() + "\n"); ndocs += 1
			print(f"  fineweb-edu: {total / 1e6:.0f} MB", file=sys.stderr, end="\r")
			if total >= mb * 1e6:
				break
		fetched += f.fetched
		if total >= mb * 1e6:
			break
	print(f"  fineweb-edu: {ndocs} documents, {total / 1e6:.1f} MB, {fetched / 1e6:.0f} MB downloaded", file=sys.stderr)
	return ndocs, total


STACK = "https://huggingface.co/datasets/bigcode/the-stack-dedup/resolve/main/data/%s/data-%05d-of-%05d.parquet"
STACK_TREE = "https://huggingface.co/api/datasets/bigcode/the-stack-dedup/tree/main/data/%s"
# The Stack is permissively licensed by construction; this list is checked again per file,
# so nothing outside it gets in whatever the dataset's own filter allowed
STACK_OK = {"mit", "mit-0", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "bsd-3-clause-clear", "0bsd", "isc",
            "zlib", "unlicense", "cc0-1.0", "bsl-1.0", "wtfpl", "curl", "x11", "ncsa", "postgresql",
            "python-2.0", "psf-2.0", "cc-by-4.0", "cc-by-3.0", "bsd-4-clause", "upl-1.0"}


def hf_token():
	t = os.environ.get("HF_TOKEN")
	if t:
		return t
	try:
		return open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
	except OSError:
		return None


def fetch_stack(lang, mb, emit):
	"""Stream one language of The Stack (dedup, v1.2, collected 2022) to emit() until mb is
	reached. Needs the dataset's terms accepted on Hugging Face and a token."""
	import json
	try:
		import pyarrow.parquet as pq
	except ImportError:
		print(f"  stack/{lang}: no pyarrow, skipped", file=sys.stderr)
		return 0, 0, 0
	tok = hf_token()
	hdr = {"Authorization": f"Bearer {tok}"} if tok else {}
	with urllib.request.urlopen(urllib.request.Request(STACK_TREE % lang, headers=hdr), timeout=60) as r:
		n = len([x for x in json.load(r) if x["path"].endswith(".parquet")])
	ndocs, total, skipped, fetched = 0, 0, 0, 0
	for i in range(n):
		f = HttpFile(STACK % (lang, i, n), hdr)
		pf = pq.ParquetFile(f)
		names = pf.schema_arrow.names
		lcol = next((c for c in ("max_stars_repo_licenses", "licenses", "max_forks_repo_licenses") if c in names), None)
		if lcol is None:
			sys.exit(f"stack/{lang}: no license column in {names}")
		for g in range(pf.num_row_groups):
			tb = pf.read_row_group(g, columns=["content", lcol])
			for t, lic in zip(tb.column("content").to_pylist(), tb.column(lcol).to_pylist()):
				if not lic or any(l.lower() not in STACK_OK for l in lic):
					skipped += 1
					continue
				d = clean(t.encode("utf-8", "replace"))
				if d:
					total += emit(d); ndocs += 1
			print(f"  stack/{lang}: {total / 1e6:.0f} MB", file=sys.stderr, end="\r")
			if total >= mb * 1e6:
				break
		fetched += f.fetched
		if total >= mb * 1e6:
			break
	print(f"  stack/{lang}: {ndocs} files, {total / 1e6:.1f} MB, {skipped} refused by license, {fetched / 1e6:.0f} MB downloaded", file=sys.stderr)
	return ndocs, total, skipped


def main():
	args = sys.argv[1:]
	if not args or args[0].startswith("-"):
		sys.exit(__doc__)
	outdir, english_mb, scale = args[0], 100.0, 1.0
	if "--english-mb" in args: english_mb = float(args[args.index("--english-mb") + 1])
	if "--scale" in args: scale = float(args[args.index("--scale") + 1])
	code_scale = float(args[args.index("--code-scale") + 1]) if "--code-scale" in args else scale
	stack = {}
	if "--stack" in args:   # e.g. c=500,go=300 (MB per language of The Stack)
		for kv in args[args.index("--stack") + 1].split(","):
			k, v = kv.split("=")
			stack[k] = float(v)
	os.makedirs(outdir, exist_ok=True)
	srcs = [(r, l, e, g, c * code_scale) for r, l, e, g, c in SOURCES]
	print(f"fetching {len(srcs)} repos + fineweb-edu", file=sys.stderr)
	with ThreadPoolExecutor(6) as ex:
		results = list(ex.map(fetch_repo, srcs))

	# English is written to train.txt as it streams; code and the held-out set stay in
	# memory and are shuffled. hakm-train samples at random offsets, so the order of
	# train.txt does not matter, but its eval reads val.txt from the start, so val is mixed.
	seen, code, val, rows, by_lang = set(), [], [], [], {}
	tr = open(os.path.join(outdir, "train.txt"), "w")
	nt = nv = 0
	def add(d, tag, stream=False):
		nonlocal nt, nv
		h = hashlib.sha1(re.sub(r"\s+", "", d).encode()).digest()
		if h in seen:
			return 0
		seen.add(h)
		by_lang[tag] = by_lang.get(tag, 0) + len(d)
		if h[0] < 3:   # ~1%
			val.append(d); nv += len(d)
		elif stream:
			tr.write(d + "\n\n"); nt += len(d)
		else:
			code.append(d); nt += len(d)
		return len(d)
	for (repo, lic, exts, lang, cap), out, nseen in results:
		mb = sum(add(d, lang) for d in out) / 1e6
		rows.append(f"| [{repo}](https://github.com/{repo}) | {lang} | {lic} | {len(out)} | {mb:.1f} |")
	results = None
	nen, mb = fetch_english(english_mb * scale, lambda d: add(d, "english", True)) if english_mb > 0 else (0, 0)
	rows.append(f"| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) sample/10BT | english | ODC-By 1.0 | {nen} | {by_lang.get('english', 0) / 1e6:.1f} |")
	for lang, smb in stack.items():
		before = by_lang.get(lang, 0)
		nf, _, nskip = fetch_stack(lang, smb, lambda d, lang=lang: add(d, lang, True))
		rows.append(f"| [The Stack dedup v1.2](https://huggingface.co/datasets/bigcode/the-stack-dedup) data/{lang} | {lang} | permissive only, per file: {', '.join(sorted(STACK_OK))} ({nskip} files refused) | {nf} | {(by_lang.get(lang, 0) - before) / 1e6:.1f} |")

	random.Random(1).shuffle(code)
	for d in code:
		tr.write(d + "\n\n")
	tr.close()
	random.Random(2).shuffle(val)
	with open(os.path.join(outdir, "val.txt"), "w") as va:
		for d in val:
			va.write(d + "\n\n")

	with open(os.path.join(outdir, "MANIFEST.md"), "w") as f:
		f.write("# corpus manifest\n\nbuilt by tools/mkcorpus.py; documents deduplicated, shuffled, 1% held out by hash\n\n")
		f.write(f"train {nt / 1e6:.1f} MB, val {nv / 1e6:.1f} MB\n\n")
		f.write("| language | MB |\n|---|---|\n")
		for k, v in sorted(by_lang.items(), key=lambda kv: -kv[1]):
			f.write(f"| {k} | {v / 1e6:.1f} |\n")
		f.write("\n| source | language | license | files | MB |\n|---|---|---|---|---|\n")
		f.write("\n".join(rows) + "\n")
	print(f"train {nt / 1e6:.1f} MB, val {nv / 1e6:.1f} MB -> {outdir}", file=sys.stderr)
	for k, v in sorted(by_lang.items(), key=lambda kv: -kv[1]):
		print(f"  {k:8s} {v / 1e6:6.1f} MB", file=sys.stderr)


if __name__ == "__main__":
	main()
