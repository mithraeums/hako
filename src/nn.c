/* nn.c — kernels: rmsnorm, softmax, silu, neox-rope, threaded quant matmul. */
#include "nn.h"
#include "quant.h"
#include "../include/mlf2.h"

#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

void rmsnorm(float *y, const float *x, const float *w, int n, float eps) {
	float ss = 0.0f;
	for (int i = 0; i < n; i++) ss += x[i] * x[i];
	float scale = 1.0f / sqrtf(ss / (float)n + eps);
	for (int i = 0; i < n; i++) y[i] = x[i] * scale * w[i];
}

void softmax(float *x, int n) {
	float mx = x[0];
	for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
	float sum = 0.0f;
	for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
	float inv = 1.0f / sum;
	for (int i = 0; i < n; i++) x[i] *= inv;
}

float silu_f(float x) { return x / (1.0f + expf(-x)); }

/* NeoX rope: rotate (vec[i], vec[i + hd/2]) pairs. Matches llama.cpp qwen2. */
void rope_neox(float *vec, int head_dim, int pos, float theta) {
	int half = head_dim / 2;
	for (int i = 0; i < half; i++) {
		float freq = powf(theta, -((float)(2 * i) / (float)head_dim));
		float ang  = (float)pos * freq;
		float c = cosf(ang), s = sinf(ang);
		float a = vec[i], b = vec[i + half];
		vec[i]        = a * c - b * s;
		vec[i + half] = a * s + b * c;
	}
}

/* bytes occupied by one row of n_in elements of the given type. */
static size_t row_nbytes(int type, int n_in) {
	switch (type) {
	case MLF2_TYPE_F32:  return (size_t)n_in * 4;
	case MLF2_TYPE_F16:  return (size_t)n_in * 2;
	case MLF2_TYPE_Q4_K: return (size_t)(n_in / QK_K) * 144;
	case MLF2_TYPE_Q6_K: return (size_t)(n_in / QK_K) * 210;
	default:             return 0;
	}
}

void deq_row(const hakm_model *m, const mlf2_tensor *w, int r, int n_in, float *out) {
	const unsigned char *base = (const unsigned char *)mlf2_data(m, w);
	size_t rb = row_nbytes((int)w->type, n_in);
	dequant_tensor((int)w->type, base + (size_t)r * rb, out, n_in);
}

#define MM_MAXT 16

typedef struct {
	const hakm_model *m; const mlf2_tensor *w;
	const float *x, *bias; float *out;
	const int8_t *xq; const float *xs;   /* int8-quantized x for the k-quant path */
	const float *axs;                    /* per-32-block scale·Σx, Q4_K min term */
	int r0, r1, n_in, type; size_t rb;
	int B, n_out;                        /* batch rows in x/out; out row stride */
} mm_job;

static void mm_worker(void *arg) {
	mm_job *j = (mm_job *)arg;
	const unsigned char *base = (const unsigned char *)mlf2_data(j->m, j->w);

	/* Q4_K fast path: integer dot against pre-quantized activation, no per-row
	   float dequant of the weight matrix. The b-loop sits INSIDE the row loop on
	   purpose: one weight row (~1.1 KB at d_model 2048) is read from RAM once and
	   dotted against all B activations while hot in L1 — this is what makes
	   batched prefill faster than B separate GEMVs. */
	if (j->type == MLF2_TYPE_Q4_K || j->type == MLF2_TYPE_Q6_K) {
		int xsn = j->n_in / 32;
		int q6 = (j->type == MLF2_TYPE_Q6_K);
		for (int r = j->r0; r < j->r1; r++) {
			const unsigned char *row = base + (size_t)r * j->rb;
			for (int b = 0; b < j->B; b++) {
				const int8_t *xqb = j->xq + (size_t)b * j->n_in;
				const float  *xsb = j->xs + (size_t)b * xsn;
				const float  *axb = j->axs + (size_t)b * xsn;
				float acc = q6 ? q6k_vec_dot(row, j->n_in, xqb, xsb)
				               : q4k_vec_dot(row, j->n_in, xqb, xsb, axb);
				float *o = j->out + (size_t)b * j->n_out;
				o[r] = j->bias ? acc + j->bias[r] : acc;
			}
		}
		return;
	}

	/* Generic path: dequant the row to float ONCE, then dot all B activations.
	   Heap scratch sized to n_in — must hold a full row (ffn_dim), which scales
	   with the model (3B 11008, 7B 18944, larger tiers more); a fixed stack
	   array overflowed. */
	float *scratch = malloc((size_t)j->n_in * sizeof(float));
	if (!scratch) return;
	for (int r = j->r0; r < j->r1; r++) {
		dequant_tensor(j->type, base + (size_t)r * j->rb, scratch, j->n_in);
		for (int b = 0; b < j->B; b++) {
			const float *xb = j->x + (size_t)b * j->n_in;
			float acc = 0.0f;
			for (int i = 0; i < j->n_in; i++) acc += scratch[i] * xb[i];
			float *o = j->out + (size_t)b * j->n_out;
			o[r] = j->bias ? acc + j->bias[r] : acc;
		}
	}
	free(scratch);
}

int hakm_nthreads(void) {
	static int n = 0;
	if (!n) {
		const char *env = getenv("HAKM_THREADS");   /* perf knob; 0/unset = auto */
		long c = (env && atoi(env) > 0) ? atoi(env) : sysconf(_SC_NPROCESSORS_ONLN);
		n = c < 1 ? 1 : (c > MM_MAXT ? MM_MAXT : (int)c);
	}
	return n;
}

/* ── persistent thread pool ──
 * Per-token inference issues ~250 matmuls; spawning+joining threads each time
 * costs ~1000 pthread_create/join cycles per token. Instead spawn the workers
 * once and dispatch matmul chunks to them through a condvar barrier. Worker w
 * (1..T-1) runs jb[w]; the calling thread runs jb[0] itself, then waits for the
 * workers' completion counter. No mutex is held during the actual compute. */
typedef struct {
	pthread_mutex_t lock;
	pthread_cond_t  go;      /* signalled when a new batch is ready */
	pthread_cond_t  done;    /* signalled as each worker finishes */
	void  (*fn)(void *);     /* job runner for this batch */
	void   *jobs;            /* job array base; jb[1..njobs] go to workers */
	size_t  jsize;           /* size of one job element */
	int     njobs;           /* number of worker jobs this batch (excludes jb[0]) */
	uint64_t gen;            /* batch generation; workers wake when it changes */
	int     remaining;       /* worker jobs still running this batch */
	int     started;         /* pool spawned? */
	int     shutdown;
	pthread_t th[MM_MAXT];
} mm_pool;

static mm_pool POOL = { PTHREAD_MUTEX_INITIALIZER,
                        PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER,
                        NULL, NULL, 0, 0, 0, 0, 0, 0, {0} };

static void *mm_pool_worker(void *arg) {
	long id = (long)arg;          /* 1..T-1 */
	uint64_t seen = 0;
	for (;;) {
		pthread_mutex_lock(&POOL.lock);
		while (POOL.gen == seen && !POOL.shutdown)
			pthread_cond_wait(&POOL.go, &POOL.lock);
		if (POOL.shutdown) { pthread_mutex_unlock(&POOL.lock); return NULL; }
		seen = POOL.gen;
		int mine = (id <= POOL.njobs);
		pthread_mutex_unlock(&POOL.lock);

		if (mine) POOL.fn((char *)POOL.jobs + (size_t)id * POOL.jsize);   /* compute outside the lock */

		pthread_mutex_lock(&POOL.lock);
		if (mine && --POOL.remaining == 0) pthread_cond_signal(&POOL.done);
		pthread_mutex_unlock(&POOL.lock);
	}
}

static void mm_pool_start(int T) {
	if (POOL.started) return;
	POOL.started = 1;
	for (long t = 1; t < T; t++)
		pthread_create(&POOL.th[t], NULL, mm_pool_worker, (void *)t);
}

/* Generic parallel-for over a job array: jobs[1..n-1] go to pool workers, the
 * caller runs jobs[0], then waits. Reused for matmul chunks AND attention. */
void hakm_parallel(void (*fn)(void *), void *jobs, size_t jsize, int njobs_total) {
	if (njobs_total <= 1) { if (njobs_total == 1) fn(jobs); return; }
	mm_pool_start(hakm_nthreads());
	int nw = njobs_total - 1;     /* worker jobs (jobs[1..nw]) */

	pthread_mutex_lock(&POOL.lock);
	POOL.fn = fn;
	POOL.jobs = jobs;
	POOL.jsize = jsize;
	POOL.njobs = nw;
	POOL.remaining = nw;
	POOL.gen++;
	pthread_cond_broadcast(&POOL.go);
	pthread_mutex_unlock(&POOL.lock);

	fn(jobs);                     /* caller does chunk 0 */

	pthread_mutex_lock(&POOL.lock);
	while (POOL.remaining > 0) pthread_cond_wait(&POOL.done, &POOL.lock);
	pthread_mutex_unlock(&POOL.lock);
}

void matmul_batch(float *out, const float *x,
                  const hakm_model *m, const mlf2_tensor *w,
                  const float *bias, int n_out, int n_in, int B) {
	int T = hakm_nthreads();
	if (n_out < T) T = 1;
	size_t rb = row_nbytes((int)w->type, n_in);

	/* For the k-quant fast paths, quantize all B activation rows once and
	   share them across all output rows / threads (read-only). n_in is always
	   a multiple of 32. */
	const int8_t *xqp = NULL; const float *xsp = NULL; const float *axsp = NULL;
	int8_t *xq = NULL; float *xs = NULL; float *axs = NULL;
	if ((int)w->type == MLF2_TYPE_Q4_K || (int)w->type == MLF2_TYPE_Q6_K) {
		/* heap-sized to n_in (multiple of 32) so it scales past the 3B; a fixed
		   stack buffer overflowed on the 7B's ffn_dim (18944). */
		xq  = malloc((size_t)B * n_in);
		xs  = malloc((size_t)B * (n_in / 32) * sizeof(float));
		axs = malloc((size_t)B * (n_in / 32) * sizeof(float));
		for (int b = 0; b < B; b++)
			quantize_row_q8_32(x + (size_t)b * n_in, n_in,
			                   xq + (size_t)b * n_in, xs + (size_t)b * (n_in / 32),
			                   axs + (size_t)b * (n_in / 32));
		xqp = xq; xsp = xs; axsp = axs;
	}

	mm_job jb[MM_MAXT];
	int per = (n_out + T - 1) / T, nt = 0;
	for (int t = 0; t < T; t++) {
		int r0 = t * per, r1 = r0 + per;
		if (r1 > n_out) r1 = n_out;
		if (r0 >= r1) break;
		jb[nt++] = (mm_job){ m, w, x, bias, out, xqp, xsp, axsp, r0, r1, n_in, (int)w->type, rb, B, n_out };
	}
	hakm_parallel(mm_worker, jb, sizeof(mm_job), nt);
	free(xq); free(xs); free(axs);
}

void matmul(float *out, const float *x,
            const hakm_model *m, const mlf2_tensor *w,
            const float *bias, int n_out, int n_in) {
	matmul_batch(out, x, m, w, bias, n_out, n_in, 1);
}
