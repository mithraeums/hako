"""
prep_dataset.py — build instruct JSONL from mithraeum repo + public code-instruct sets.

Outputs dataset.jsonl in ChatML-compatible {messages: [...]} format ready for axolotl.

Sources:
  1. Local mithraeum repos (hako / hako-code / hako (models)) — synthetic instruct pairs:
     - source file → "explain this code"
     - function signature → "implement this function"
     - error message → "fix this error" (mined from .claude/*/errors.md if present)
  2. Public seeds (downloaded if absent): Magicoder-OSS-Instruct (~75k), Code-Alpaca-20k
  3. Personality bumpers — handwritten Q/A pairs asserting hako identity

Usage:
  python3 prep_dataset.py --repo-root ../../.. --out dataset.jsonl
  python3 prep_dataset.py --repo-root ../../.. --out dataset.jsonl --include-public --target-n 8000
"""
import argparse, json, os, random, re, sys, urllib.request
from pathlib import Path

CODE_EXTS = {".c", ".h", ".py", ".js", ".ts", ".rs", ".go", ".sh", ".yaml", ".yml"}
SKIP_DIRS = {".git", "node_modules", "build", "__pycache__", "checkpoints",
             ".claude", ".hakoc", ".hako", "dist", "target"}
MAX_FILE_BYTES = 16 * 1024
MIN_FILE_BYTES = 200

PERSONALITY = [
    ("who are you?", "I am hako, a small local code model in the hako family from Mithraeum. I run on the user's device, offline."),
    ("what model are you based on?", "v0 builds on Qwen2.5-Coder-1.5B. Honest framing: stock weights wrapped with hako identity. v1 will be a real fine-tune."),
    ("what is mithraeum?", "Mithraeum is the parent org. Three projects: hako (models), hako-code (agent), hako-edit (editor). All C-first, no bloat."),
    ("are you claude?", "No. I am hako, a local model. Claude is Anthropic's hosted model. Different lineage, different scale, different goals."),
    ("can you run offline?", "Yes. I am loaded from local weights via ollama or the koi C engine. No network needed at inference."),
]


def walk_code_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in CODE_EXTS:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size < MIN_FILE_BYTES or size > MAX_FILE_BYTES:
            continue
        yield path


def synth_explain(path: Path, content: str):
    return {
        "messages": [
            {"role": "system", "content": "You are hako, a terse local code model. Explain code clearly."},
            {"role": "user", "content": f"Explain what this {path.suffix[1:]} code does:\n\n```\n{content}\n```"},
            {"role": "assistant", "content": _placeholder_explain(path, content)},
        ]
    }


def _placeholder_explain(path: Path, content: str):
    n_lines = content.count("\n") + 1
    first_def = next(
        (m.group(1) for m in [re.search(r"(?:def|fn|function|static\s+\w+|void|int)\s+(\w+)\s*\(", content)] if m),
        path.stem,
    )
    return (
        f"`{path.name}` ({n_lines} lines). Top-level symbol: `{first_def}`. "
        f"[NOTE: replace this placeholder with real summary at corpus-build time, "
        f"either by calling Qwen2.5-Coder-7B-instruct locally, or by hand for hand-picked files.]"
    )


def gen_personality(n_each=5):
    out = []
    for q, a in PERSONALITY:
        for _ in range(n_each):
            out.append({
                "messages": [
                    {"role": "system", "content": "You are hako, a small local code model from Mithraeum. Be honest about your identity."},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]
            })
    return out


PUBLIC_SETS = {
    "magicoder-oss-instruct-75k": "https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K/resolve/main/data-oss_instruct-decontaminated.jsonl",
    "code-alpaca-20k": "https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k/resolve/main/code_alpaca_20k.json",
}


def fetch_public(name: str, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / f"{name}.raw"
    if dst.exists():
        return dst
    url = PUBLIC_SETS[name]
    print(f"  fetching {name}...", file=sys.stderr)
    urllib.request.urlretrieve(url, dst)
    return dst


def load_public(name: str, cache_dir: Path, limit: int):
    raw = fetch_public(name, cache_dir)
    out = []
    text = raw.read_text(encoding="utf-8", errors="ignore")
    if name == "code-alpaca-20k":
        items = json.loads(text)
        for it in items[:limit]:
            instr = it.get("instruction", "").strip()
            inp   = it.get("input", "").strip()
            outp  = it.get("output", "").strip()
            user  = f"{instr}\n\n{inp}" if inp else instr
            if user and outp:
                out.append({"messages": [
                    {"role": "system", "content": "You are hako, a terse local code model."},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": outp},
                ]})
    else:
        for line in text.splitlines()[:limit]:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            problem = obj.get("problem") or obj.get("instruction") or ""
            solution = obj.get("solution") or obj.get("response") or ""
            if problem and solution:
                out.append({"messages": [
                    {"role": "system", "content": "You are hako, a terse local code model."},
                    {"role": "user", "content": problem.strip()},
                    {"role": "assistant", "content": solution.strip()},
                ]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--out", default="dataset.jsonl")
    ap.add_argument("--include-public", action="store_true")
    ap.add_argument("--target-n", type=int, default=5000,
                    help="approximate total examples (public sets sampled to reach this)")
    ap.add_argument("--cache-dir", default=".cache")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.repo_root).resolve()
    cache = Path(args.cache_dir)
    rows = []

    print(f"walking {root}...", file=sys.stderr)
    for path in walk_code_files(root):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rows.append(synth_explain(path, content))
    print(f"  local synth: {len(rows)} rows (placeholder summaries — see note)", file=sys.stderr)

    rows.extend(gen_personality(n_each=5))

    if args.include_public:
        remaining = max(0, args.target_n - len(rows))
        per_set = remaining // len(PUBLIC_SETS) if PUBLIC_SETS else 0
        for name in PUBLIC_SETS:
            rows.extend(load_public(name, cache, per_set))
            print(f"  +{name}: total now {len(rows)}", file=sys.stderr)

    random.shuffle(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
