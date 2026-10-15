#!/usr/bin/env sh
# hakm installer — the hako engine. Fetches the latest GitHub release for this
# OS/arch, or builds from source where there is no release asset (iSh: 32-bit
# x86 under musl). No ollama, no llama.cpp: hakm is its own runtime.
#
# Usage:  curl -fsSL https://raw.githubusercontent.com/mithraeums/hako/main/install.sh | sh
# Env:    REPO=owner/hako      (override default)
#         PREFIX=/usr/local    (install dir; defaults to ~/.local if not writable)
#         VERIFY=0             (skip sha256 verify; default 1)
#         SOURCE=1             (force a source build even where a release exists)
#
# Weights are NOT downloaded here — sho is ~1.9 GB. Afterwards:
#         hakm pull hako-sho       # 3B, runs on a laptop
#         hakm pull hako-koi       # 7B
# or let the agent do it on first use (`hako`, then `:pull hako-sho`).

set -eu

REPO="${REPO:-mithraeums/hako}"
PREFIX="${PREFIX:-}"
VERIFY="${VERIFY:-1}"
SOURCE="${SOURCE:-0}"

uname_s="$(uname -s)"
uname_m="$(uname -m)"

if [ -z "$PREFIX" ]; then
	if [ -w "/usr/local/bin" ] || [ "$(id -u)" = "0" ]; then
		PREFIX=/usr/local
	else
		PREFIX="${HOME}/.local"
	fi
fi
mkdir -p "${PREFIX}/bin"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# ---- source build: iSh and anything else without a release asset ----------
build_from_source() {
	command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 || {
		echo "no C compiler. on iSh: apk add build-base git" >&2; exit 1; }
	command -v make >/dev/null 2>&1 || { echo "make not found. on iSh: apk add build-base" >&2; exit 1; }
	command -v git  >/dev/null 2>&1 || { echo "git not found. on iSh: apk add git" >&2; exit 1; }
	echo "building hakm from source (${uname_s} ${uname_m})..."
	git clone --depth 1 "https://github.com/${REPO}.git" "${tmp}/src" >/dev/null 2>&1
	# The Makefile drops -march=native on 32-bit x86 by itself (the emulator
	# advertises AVX/FMA it cannot execute); nothing to pass here.
	(cd "${tmp}/src" && make hakm) || { echo "build failed" >&2; exit 1; }
	install -m 0755 "${tmp}/src/hakm" "${PREFIX}/bin/hakm"
}

asset=""
case "$uname_s" in
	Linux*)
		case "$uname_m" in
			x86_64|amd64)  asset="hakm-linux-x86_64.tar.gz"; dirname="hakm-linux-x86_64" ;;
			arm64|aarch64) asset="hakm-linux-arm64.tar.gz";  dirname="hakm-linux-arm64" ;;
			i386|i486|i586|i686) asset="" ;;         # iSh, 32-bit: source
			*) asset="" ;;
		esac
		;;
	Darwin*) asset="hakm-macos-universal.tar.gz"; dirname="hakm-macos-universal" ;;
	*) asset="" ;;
esac

if [ "$SOURCE" = 1 ] || [ -z "$asset" ]; then
	[ -z "$asset" ] && echo "no release asset for ${uname_s} ${uname_m}; building from source."
	build_from_source
else
	api="https://api.github.com/repos/${REPO}/releases/latest"
	echo "fetching latest release of ${REPO}..."
	tag="$(curl -fsSL "$api" | grep -m1 '"tag_name":' | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/' || true)"
	if [ -z "$tag" ]; then
		echo "could not resolve latest tag (rate-limited?); falling back to a source build" >&2
		build_from_source
	else
		echo "latest: $tag"
		url="https://github.com/${REPO}/releases/download/${tag}/${asset}"
		sha_name="${asset%.tar.gz}.sha256"
		sha_url="https://github.com/${REPO}/releases/download/${tag}/${sha_name}"
		echo "downloading $asset"
		curl -fL --progress-bar -o "${tmp}/${asset}" "$url"
		tar -xzf "${tmp}/${asset}" -C "$tmp"
		if [ "$VERIFY" = "1" ]; then
			if curl -fsSL -o "${tmp}/${sha_name}" "$sha_url"; then
				(cd "${tmp}/${dirname}" && \
					if command -v sha256sum >/dev/null 2>&1; then sha256sum -c "../${sha_name}" >/dev/null; \
					elif command -v shasum >/dev/null 2>&1;    then shasum -a 256 -c "../${sha_name}" >/dev/null; \
					else echo "no sha256 tool found, skipping verify" >&2; fi) \
					|| { echo "sha256 verify FAILED" >&2; exit 1; }
				echo "sha256 ok."
			else
				echo "warning: sha sidecar missing — skipping verify (set VERIFY=0 to silence)" >&2
			fi
		fi
		install -m 0755 "${tmp}/${dirname}/hakm" "${PREFIX}/bin/hakm"
		if [ "$uname_s" = "Darwin" ] && command -v xattr >/dev/null 2>&1; then
			xattr -d com.apple.quarantine "${PREFIX}/bin/hakm" 2>/dev/null || true
		fi
	fi
fi

echo "installed: ${PREFIX}/bin/hakm"
"${PREFIX}/bin/hakm" --version || true

# The agent looks here first, so it never runs a stale engine off PATH.
mkdir -p "${HOME}/.hako/bin"
cp -f "${PREFIX}/bin/hakm" "${HOME}/.hako/bin/hakm"

case ":${PATH}:" in
	*":${PREFIX}/bin:"*) ;;
	*) echo "note: ${PREFIX}/bin not in PATH. add to your shell rc:"
	   echo "    export PATH=\"${PREFIX}/bin:\$PATH\"" ;;
esac

echo ""
echo "next: hakm pull hako-sho      # 3B weights (~1.9 GB) from huggingface.co/mithraeum"
echo "      hako                    # the agent finds it at ~/.hako/bin/hakm"
