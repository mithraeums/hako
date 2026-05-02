/*
 * loader.c — MLF v0 parser, mmap, weight pointer setup.
 *
 * MLF v0 header layout (64 bytes, little-endian):
 *   [0:4]   magic    "MLF1"
 *   [4:8]   version  u32
 *   [8:12]  n_layers u32
 *  [12:16]  d_model  u32
 *  [16:20]  n_heads  u32
 *  [20:24]  ffn_dim  u32
 *  [24:28]  ctx      u32
 *  [28:32]  vocab    u32
 *  [32:36]  scheme   u32  (0=fp32)
 *  [36:44]  n_params u64
 *  [44:64]  reserved
 *
 * Weight layout after header (fp32, row-major):
 *   embed              [vocab × d_model]
 *   per layer i:
 *     attn_norm_w      [d_model]
 *     q_weight         [d_model × d_model]
 *     k_weight         [d_model × d_model]
 *     v_weight         [d_model × d_model]
 *     o_weight         [d_model × d_model]
 *     ffn_norm_w       [d_model]
 *     gate_weight      [ffn_dim × d_model]
 *     up_weight        [ffn_dim × d_model]
 *     down_weight      [d_model × ffn_dim]
 *   final_norm_w       [d_model]
 *   (lm_head tied to embed)
 */
#include "loader.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define MLF_MAGIC    "MLF1"
#define MLF_HDR_SIZE 64
#define MLF_V0       0

#define MAX_LAYERS   32

/* Packed header mirrors the on-disk layout exactly. */
typedef struct __attribute__((packed)) {
	char     magic[4];
	uint32_t version;
	uint32_t n_layers;
	uint32_t d_model;
	uint32_t n_heads;
	uint32_t ffn_dim;
	uint32_t ctx;
	uint32_t vocab;
	uint32_t scheme;
	uint64_t n_params;
	uint8_t  reserved[20];
} mlf_header_v0;

static int parse_header(const void *buf, size_t file_size, mlf_params *p) {
	if (file_size < MLF_HDR_SIZE) {
		fprintf(stderr, "mini: file too small (%zu bytes)\n", file_size);
		return -1;
	}
	const mlf_header_v0 *h = (const mlf_header_v0 *)buf;

	if (memcmp(h->magic, MLF_MAGIC, 4) != 0) {
		fprintf(stderr, "mini: bad magic (not MLF1)\n");
		return -1;
	}
	if (h->version != MLF_V0) {
		fprintf(stderr, "mini: unsupported MLF version %u\n", h->version);
		return -1;
	}
	if (h->scheme != 0) {
		fprintf(stderr, "mini: unsupported scheme %u (only fp32=0)\n", h->scheme);
		return -1;
	}
	if (h->n_layers == 0 || h->n_layers > MAX_LAYERS) {
		fprintf(stderr, "mini: invalid n_layers %u\n", h->n_layers);
		return -1;
	}
	if (h->d_model == 0 || h->n_heads == 0 || h->d_model % h->n_heads != 0) {
		fprintf(stderr, "mini: d_model %u not divisible by n_heads %u\n",
				h->d_model, h->n_heads);
		return -1;
	}

	uint64_t expected_params =
		(uint64_t)h->vocab * h->d_model +                         /* embed */
		(uint64_t)h->n_layers * (
			h->d_model +                                           /* attn_norm */
			4ULL * h->d_model * h->d_model +                      /* q,k,v,o */
			h->d_model +                                           /* ffn_norm */
			2ULL * h->ffn_dim * h->d_model +                      /* gate, up */
			(uint64_t)h->d_model * h->ffn_dim                     /* down */
		) +
		h->d_model;                                                /* final_norm */

	if (h->n_params != expected_params) {
		fprintf(stderr, "mini: n_params mismatch: header=%llu expected=%llu\n",
				(unsigned long long)h->n_params,
				(unsigned long long)expected_params);
		return -1;
	}
	uint64_t expected_bytes = MLF_HDR_SIZE + expected_params * sizeof(float);
	if (file_size < expected_bytes) {
		fprintf(stderr, "mini: file too small: %zu < %llu\n",
				file_size, (unsigned long long)expected_bytes);
		return -1;
	}

	p->n_layers = h->n_layers;
	p->d_model  = h->d_model;
	p->n_heads  = h->n_heads;
	p->ffn_dim  = h->ffn_dim;
	p->ctx      = h->ctx;
	p->vocab    = h->vocab;
	return 0;
}

int mlf_load(const char *path, nano_model *m) {
	memset(m, 0, sizeof(*m));

	int fd = open(path, O_RDONLY);
	if (fd < 0) {
		perror("mini: open");
		return -1;
	}
	struct stat st;
	if (fstat(fd, &st) < 0) {
		perror("mini: fstat");
		close(fd);
		return -1;
	}
	size_t size = (size_t)st.st_size;

	void *base = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
	close(fd);
	if (base == MAP_FAILED) {
		perror("mini: mmap");
		return -1;
	}

	if (parse_header(base, size, &m->p) != 0) {
		munmap(base, size);
		return -1;
	}

	m->mmap_base = base;
	m->mmap_size = size;

	/* Walk weight pointer arithmetic. */
	const float *ptr = (const float *)((const char *)base + MLF_HDR_SIZE);

	m->embed = ptr;
	ptr += (size_t)m->p.vocab * m->p.d_model;

	for (uint32_t l = 0; l < m->p.n_layers; l++) {
		m->layers[l].attn_norm = ptr; ptr += m->p.d_model;
		m->layers[l].q        = ptr; ptr += (size_t)m->p.d_model * m->p.d_model;
		m->layers[l].k        = ptr; ptr += (size_t)m->p.d_model * m->p.d_model;
		m->layers[l].v        = ptr; ptr += (size_t)m->p.d_model * m->p.d_model;
		m->layers[l].o        = ptr; ptr += (size_t)m->p.d_model * m->p.d_model;
		m->layers[l].ffn_norm = ptr; ptr += m->p.d_model;
		m->layers[l].gate     = ptr; ptr += (size_t)m->p.ffn_dim * m->p.d_model;
		m->layers[l].up       = ptr; ptr += (size_t)m->p.ffn_dim * m->p.d_model;
		m->layers[l].down     = ptr; ptr += (size_t)m->p.d_model * m->p.ffn_dim;
	}

	m->final_norm = ptr;

	return 0;
}

void mlf_unload(nano_model *m) {
	if (m->mmap_base && m->mmap_base != MAP_FAILED) {
		munmap(m->mmap_base, m->mmap_size);
		m->mmap_base = NULL;
	}
}
