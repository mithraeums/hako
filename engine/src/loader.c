/* loader.c — MLF2 mmap + tensor directory. */
#include "loader.h"

#include <errno.h>
#include <fcntl.h>
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
	if (m->h.tensors_off + (size_t)m->h.n_tensors * sizeof(mlf2_tensor) > size ||
	    m->h.data_off > size || m->h.tokenizer_off > size) {
		fprintf(stderr, "hakm: corrupt offsets\n");
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
