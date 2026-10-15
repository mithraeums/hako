/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* loader.c — MLF2 mmap + tensor directory. */
#include "loader.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int mlf2_open(const char *path, hakm_model *m) {
	memset(m, 0, sizeof(*m));

	int fd = open(path, O_RDONLY);
	if (fd < 0) { perror("hakm: open"); return -1; }

	struct stat st;
	if (fstat(fd, &st) < 0) { perror("hakm: fstat"); close(fd); return -1; }
	/* off_t is 64-bit even on 32-bit musl; size_t is not. Without this a 4.7GB
	   model truncates to ~0.4GB, every header offset still validates because
	   they all live in the first few MB, and the first tensor read past the
	   mapping is a SIGBUS with no message. */
	if (st.st_size < 0 || (uintmax_t)st.st_size > (uintmax_t)SIZE_MAX) {
		fprintf(stderr, "hakm: model is %ju bytes — too large for a %zu-bit build\n",
		        (uintmax_t)st.st_size, sizeof(size_t) * 8);
		close(fd);
		return -1;
	}
	size_t size = (size_t)st.st_size;

	void *base = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
	close(fd);
	if (base == MAP_FAILED) { perror("hakm: mmap"); return -1; }

	if (size < sizeof(mlf2_header)) {
		fprintf(stderr, "hakm: file too small\n");
		munmap(base, size); return -1;
	}
	memcpy(&m->h, base, sizeof(mlf2_header));

	if (memcmp(m->h.magic, MLF2_MAGIC, 4) != 0) {
		fprintf(stderr, "hakm: bad magic (not MLF2)\n");
		munmap(base, size); return -1;
	}
	if (m->h.version != MLF2_VERSION) {
		fprintf(stderr, "hakm: unsupported MLF2 version %u\n", m->h.version);
		munmap(base, size); return -1;
	}
	/* 64-bit, overflow-safe: on a 32-bit build n_tensors * sizeof wrapped. */
	uint64_t fsz = (uint64_t)size;
	uint64_t dir_bytes = (uint64_t)m->h.n_tensors * sizeof(mlf2_tensor);
	if (m->h.tensors_off > fsz || dir_bytes > fsz - m->h.tensors_off ||
	    m->h.data_off > fsz || m->h.tokenizer_off > fsz) {
		fprintf(stderr, "hakm: corrupt offsets\n");
		munmap(base, size); return -1;
	}
	/* Every tensor must lie inside the file. A download cut short keeps a
	   perfectly valid header, and the first read past the end was a SIGSEGV
	   with no message — so say what it is, and how big it should be. */
	const mlf2_tensor *dir = (const mlf2_tensor *)((const unsigned char *)base + m->h.tensors_off);
	uint64_t need = m->h.data_off;
	for (uint32_t i = 0; i < m->h.n_tensors; i++) {
		const mlf2_tensor *t = &dir[i];
		if (t->n_dims > 4 || memchr(t->name, '\0', MLF2_MAX_NAME) == NULL ||
		    t->data_off > UINT64_MAX - m->h.data_off ||
		    t->nbytes > UINT64_MAX - m->h.data_off - t->data_off) {
			fprintf(stderr, "hakm: corrupt tensor directory (entry %u)\n", i);
			munmap(base, size); return -1;
		}
		uint64_t end = m->h.data_off + t->data_off + t->nbytes;
		if (end > need) need = end;
	}
	if (need > fsz) {
		fprintf(stderr, "hakm: model file is incomplete — %.0f of %.0f MB. "
		        "The download was cut short: run `hakm pull <id>` again (it resumes).\n",
		        fsz / 1048576.0, need / 1048576.0);
		munmap(base, size); return -1;
	}

	m->map      = base;
	m->map_size = size;
	m->tensors  = (const mlf2_tensor *)((const unsigned char *)base + m->h.tensors_off);
	m->data     = (const unsigned char *)base + m->h.data_off;
	m->tok      = (const unsigned char *)base + m->h.tokenizer_off;
	return 0;
}

void mlf2_close(hakm_model *m) {
	if (m->map && m->map != MAP_FAILED) munmap(m->map, m->map_size);
	m->map = NULL;
}

const mlf2_tensor *mlf2_find(const hakm_model *m, const char *name) {
	for (uint32_t i = 0; i < m->h.n_tensors; i++)
		if (strncmp(m->tensors[i].name, name, MLF2_MAX_NAME) == 0)
			return &m->tensors[i];
	return NULL;
}

const void *mlf2_data(const hakm_model *m, const mlf2_tensor *t) {
	return m->data + t->data_off;
}

size_t mlf2_prefault(const hakm_model *m) {
	if (!m->map || !m->map_size) return 0;
	/* Hint first (the kernel can then read ahead in big chunks), then touch one
	   byte per page so the mapping is resident even where the hint is a no-op.
	   `volatile` keeps the read; a plain load here is dead code to the
	   optimizer and the pages never arrive. */
	/* posix_madvise, not madvise: the POSIX spelling is declared by every libc
	   without extra feature macros, while plain madvise is a BSD extension
	   that musl hides behind them. Same hint, one less thing to configure. */
	(void)posix_madvise((void *)m->map, m->map_size, POSIX_MADV_WILLNEED);
	const volatile unsigned char *p = m->map;
	size_t step = (size_t)sysconf(_SC_PAGESIZE);
	if (step == 0 || step > m->map_size) step = 4096;
	unsigned char sink = 0;
	for (size_t off = 0; off < m->map_size; off += step) sink ^= p[off];
	(void)sink;
	return m->map_size;
}
