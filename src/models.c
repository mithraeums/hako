/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* models.c — the local model store. See models.h. */
#include "models.h"
#include "kvcache.h"
#include "loader.h"

#include <dirent.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

static void mkdirp1(const char *p) { mkdir(p, 0755); }

int hakm_models_dir(char *out, size_t cap) {
	const char *hh = getenv("HAKO_HOME");
	int n;
	if (hh && *hh) n = snprintf(out, cap, "%s", hh);
	else {
		const char *home = getenv("HOME");
		if (!home || !*home) return -1;
		n = snprintf(out, cap, "%s/.hako", home);
	}
	if (n < 0 || (size_t)n + 8 >= cap) return -1;
	mkdirp1(out);
	strcat(out, "/models");
	mkdirp1(out);
	struct stat st;
	return (stat(out, &st) == 0 && S_ISDIR(st.st_mode)) ? 0 : -1;
}

/* An id is one path segment: letters, digits, . _ - only. That is what keeps
   `rm` from ever walking anywhere but <models>/<id>. */
static int id_ok(const char *id) {
	if (!id || !*id || *id == '.') return 0;
	for (const char *p = id; *p; p++)
		if (!((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
		      (*p >= '0' && *p <= '9') || *p == '.' || *p == '_' || *p == '-')) return 0;
	return strlen(id) < 128;
}

/* "org/id" → org, id; "id" → default org. Both outputs sized 128+. */
static int split_spec(const char *spec, char *org, char *id) {
	const char *slash = strchr(spec, '/');
	const char *def = getenv("HAKO_HF_ORG");
	if (!def || !*def) def = "mithraeum";
	if (slash) {
		size_t ol = (size_t)(slash - spec);
		if (ol == 0 || ol >= 128) return -1;
		memcpy(org, spec, ol); org[ol] = '\0';
		snprintf(id, 128, "%s", slash + 1);
	} else {
		snprintf(org, 128, "%s", def);
		snprintf(id, 128, "%s", spec);
	}
	/* a trailing .mlf2 on an id is a habit worth tolerating */
	size_t il = strlen(id);
	if (il > 5 && !strcmp(id + il - 5, ".mlf2")) id[il - 5] = '\0';
	return id_ok(org) && id_ok(id) ? 0 : -1;
}

int hakm_resolve(const char *spec, char *out, size_t cap) {
	if (!spec || !*spec) return -1;
	struct stat st;
	/* A path: it exists as a file, is spelled as one (/ ./ ../ ~), or has
	   more than one slash. "org/id" is the one slash-form that is a name. */
	int slashes = 0;
	for (const char *p = spec; *p; p++) slashes += (*p == '/');
	if ((stat(spec, &st) == 0 && S_ISREG(st.st_mode)) ||
	    spec[0] == '/' || spec[0] == '.' || spec[0] == '~' || slashes > 1) {
		int n = snprintf(out, cap, "%s", spec);
		return (n > 0 && (size_t)n < cap) ? 0 : -1;
	}
	char org[128], id[128], dir[1024];
	if (split_spec(spec, org, id) != 0) return -1;
	if (hakm_models_dir(dir, sizeof(dir)) != 0) return -1;
	int n = snprintf(out, cap, "%s/%s/%s.mlf2", dir, id, id);
	return (n > 0 && (size_t)n < cap) ? 0 : -1;
}

/* The quant a model "is" = the type carrying most of its bytes. */
static const char *quant_name(const hakm_model *m) {
	uint64_t by[32] = {0};
	for (uint32_t i = 0; i < m->h.n_tensors; i++)
		if (m->tensors[i].type < 32) by[m->tensors[i].type] += m->tensors[i].nbytes;
	uint32_t best = 0;
	for (uint32_t t = 1; t < 32; t++) if (by[t] > by[best]) best = t;
	switch (best) {
	case MLF2_TYPE_F32:  return "f32";
	case MLF2_TYPE_F16:  return "f16";
	case MLF2_TYPE_Q4_K: return "q4_k";
	case MLF2_TYPE_Q6_K: return "q6_k";
	case MLF2_TYPE_Q8_0: return "q8_0";
	default: return "?";
	}
}

static const char *arch_name(uint32_t a) {
	return a == MLF2_ARCH_QWEN2 ? "qwen2" : a == MLF2_ARCH_LLAMA ? "llama" : "?";
}

int hakm_list(void) {
	char dir[1024];
	if (hakm_models_dir(dir, sizeof(dir)) != 0) { fprintf(stderr, "hakm: no models dir\n"); return 0; }
	DIR *d = opendir(dir);
	if (!d) return 0;
	int n = 0;
	struct dirent *de;
	printf("%-20s %8s  %-5s  %-6s  %6s  %s\n", "ID", "SIZE", "QUANT", "ARCH", "PARAMS", "KV CACHE");
	while ((de = readdir(d)) != NULL) {
		if (de->d_name[0] == '.') continue;
		char w[1300]; struct stat st;
		int wn = snprintf(w, sizeof(w), "%s/%s/%s.mlf2", dir, de->d_name, de->d_name);
		if (wn < 0 || (size_t)wn >= sizeof(w)) continue;   /* a clipped path would name the wrong file */
		if (stat(w, &st) != 0) continue;
		hakm_model m;
		/* mmap is lazy: opening reads only the header + directory. */
		if (mlf2_open(w, &m) != 0) continue;
		double params = 0.0;
		for (uint32_t t = 0; t < m.h.n_tensors; t++) {
			double ne = 1.0;
			for (uint32_t dd = 0; dd < m.tensors[t].n_dims; dd++) ne *= (double)m.tensors[t].dims[dd];
			params += ne;
		}
		int kv = kvc_for_model((unsigned long long)st.st_size, (unsigned long long)st.st_mtime, 0);
		printf("%-20s %6.2fGB  %-5s  %-6s  %5.1fB  ", de->d_name,
		       (double)st.st_size / 1e9, quant_name(&m), arch_name(m.h.arch), params / 1e9);
		if (kv) printf("%d\n", kv); else printf("-\n");
		mlf2_close(&m);
		n++;
	}
	closedir(d);
	if (n == 0) printf("(none installed — hakm pull <id>)\n");
	return n;
}

static double now_s(void) { struct timeval t; gettimeofday(&t, NULL); return t.tv_sec + t.tv_usec * 1e-6; }

static void fmt_bytes(double b, char *out, size_t cap) {
	if (b >= 1e9)      snprintf(out, cap, "%.2f GB", b / 1e9);
	else if (b >= 1e6) snprintf(out, cap, "%.1f MB", b / 1e6);
	else               snprintf(out, cap, "%.0f KB", b / 1e3);
}

/* Content-Length after redirects, via a HEAD; 0 if unknown. */
static long long remote_size(const char *url) {
	char cmd[2200];
	snprintf(cmd, sizeof(cmd), "curl -sIL '%s' 2>/dev/null", url);
	FILE *p = popen(cmd, "r");
	if (!p) return 0;
	long long len = 0;
	char line[512];
	while (fgets(line, sizeof(line), p))
		if (!strncasecmp(line, "content-length:", 15)) len = atoll(line + 15);
	pclose(p);
	return len;
}

int hakm_pull(const char *spec) {
	char org[128], id[128], dir[1024];
	if (split_spec(spec, org, id) != 0) { fprintf(stderr, "hakm: bad model id '%s'\n", spec); return -1; }
	if (hakm_models_dir(dir, sizeof(dir)) != 0) { fprintf(stderr, "hakm: no models dir\n"); return -1; }
	const char *base = getenv("HAKO_HF_BASE");
	if (!base || !*base) base = "https://huggingface.co";
	char mdir[1200], dest[1400], tmp[1420], url[1800];
	snprintf(mdir, sizeof(mdir), "%s/%s", dir, id);
	snprintf(dest, sizeof(dest), "%s/%s.mlf2", mdir, id);
	snprintf(tmp,  sizeof(tmp),  "%s.part", dest);
	snprintf(url,  sizeof(url),  "%s/%s/%s/resolve/main/%s.mlf2", base, org, id, id);

	struct stat st;
	if (stat(dest, &st) == 0) { printf("already installed: %s (%.2f GB)\n", dest, st.st_size / 1e9); return 0; }
	if (system("command -v curl >/dev/null 2>&1") != 0) { fprintf(stderr, "hakm: pull needs `curl` on PATH\n"); return -1; }
	mkdirp1(mdir);

	long long total = remote_size(url);
	long long have = (stat(tmp, &st) == 0) ? (long long)st.st_size : 0;
	int tty = isatty(STDERR_FILENO);
	fprintf(stderr, "pulling %s/%s%s\n", org, id, have ? " (resuming)" : "");

	/* curl in a child, quiet; the parent watches the file grow and draws the
	   progress itself. -C - resumes a .part; -f fails on 404 instead of
	   saving an HTML error page as weights. A server that refuses byte ranges
	   (curl exit 33) gets one clean restart. */
	double t0 = now_s();
	int status = 0;
retry:;
	pid_t pid = fork();
	if (pid < 0) return -1;
	if (pid == 0) {
		if (have) execlp("curl", "curl", "-fsSL", "-C", "-", "-o", tmp, url, (char *)NULL);
		else      execlp("curl", "curl", "-fsSL", "-o", tmp, url, (char *)NULL);
		_exit(127);
	}
	double ts = now_s();
	for (;;) {
		pid_t r = waitpid(pid, &status, WNOHANG);
		if (r == pid) break;
		if (r < 0 && errno != EINTR) break;
		usleep(200000);
		if (!tty) continue;
		long long cur = (stat(tmp, &st) == 0) ? (long long)st.st_size : 0;
		double t = now_s();
		double rate = (t - ts) > 0.2 ? (double)(cur - have) / (t - ts) : 0;   /* since this attempt began */
		char c[32], tt[32], rr[32];
		fmt_bytes((double)cur, c, sizeof c); fmt_bytes((double)total, tt, sizeof tt); fmt_bytes(rate, rr, sizeof rr);
		if (total > 0) fprintf(stderr, "\r  %5.1f%%  %s / %s  %s/s   ", 100.0 * cur / total, c, tt, rr);
		else           fprintf(stderr, "\r  %s  %s/s   ", c, rr);
		fflush(stderr);
	}
	if (tty) fprintf(stderr, "\r%60s\r", "");
	if (have && WIFEXITED(status) && WEXITSTATUS(status) == 33) {
		fprintf(stderr, "  server cannot resume — starting over\n");
		unlink(tmp); have = 0;
		goto retry;
	}
	int ok = WIFEXITED(status) && WEXITSTATUS(status) == 0 && stat(tmp, &st) == 0;
	/* Not just "curl said yes": the bytes have to be an MLF2 file. */
	if (ok) {
		char magic[4] = {0};
		FILE *f = fopen(tmp, "rb");
		ok = f && fread(magic, 1, 4, f) == 4 && memcmp(magic, MLF2_MAGIC, 4) == 0 && st.st_size > 1024 * 1024;
		if (f) fclose(f);
		if (!ok) { fprintf(stderr, "hakm: %s is not an MLF2 file\n", url); unlink(tmp); rmdir(mdir); return -1; }
	} else {
		fprintf(stderr, "hakm: download failed (curl exit %d); the .part is kept for --resume: %s\n",
		        WIFEXITED(status) ? WEXITSTATUS(status) : -1, tmp);
		return -1;
	}
	if (rename(tmp, dest) != 0) { fprintf(stderr, "hakm: could not move into place: %s\n", strerror(errno)); return -1; }
	char sz[32]; fmt_bytes((double)st.st_size, sz, sizeof sz);
	printf("installed %s (%s, %.1fs)\n", dest, sz, now_s() - t0);
	return 0;
}

int hakm_rm(const char *id) {
	if (!id_ok(id) || strchr(id, '/')) { fprintf(stderr, "hakm: rm takes an exact model id (see hakm list)\n"); return -1; }
	char dir[1024], mdir[1200], w[1400], part[1420];
	if (hakm_models_dir(dir, sizeof(dir)) != 0) return -1;
	snprintf(mdir, sizeof(mdir), "%s/%s", dir, id);
	snprintf(w, sizeof(w), "%s/%s.mlf2", mdir, id);
	snprintf(part, sizeof(part), "%s.part", w);
	struct stat st;
	if (stat(w, &st) != 0) {
		if (stat(part, &st) == 0) { unlink(part); rmdir(mdir); printf("removed partial download %s\n", part); return 0; }
		fprintf(stderr, "hakm: '%s' is not installed\n", id); return -1;
	}
	/* The cache entries are keyed on size+mtime, so take those before the file goes. */
	int kv = kvc_for_model((unsigned long long)st.st_size, (unsigned long long)st.st_mtime, 1);
	if (unlink(w) != 0) { fprintf(stderr, "hakm: %s: %s\n", w, strerror(errno)); return -1; }
	unlink(part);
	rmdir(mdir);
	printf("removed %s (%.2f GB)%s", id, st.st_size / 1e9, kv ? "" : "\n");
	if (kv) printf(" and %d KV cache entr%s\n", kv, kv == 1 ? "y" : "ies");
	return 0;
}
