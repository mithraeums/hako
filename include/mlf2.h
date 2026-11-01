/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/*
 * mlf2.h — on-disk format for the hakm engine (MLF v2).
 *
 * MLF2 is the hako-native weight container. It is produced offline by
 * tools/gguf2mlf.py from a GGUF file and consumed at runtime by the engine.
 * No llama.cpp, no ggml, no torch at runtime — we own loader, kernels, and
 * tokenizer end to end.
 *
 * Design notes
 * ------------
 *  - Quant blocks are copied VERBATIM from GGUF (k-quant layout is good and
 *    keeps the brand-promise file size). We re-implement only the dequant math,
 *    so the runtime depends on nobody else's code. Tensor `type` values below
 *    intentionally mirror ggml's enum so the converter is a pure byte copy.
 *  - All weights are mmap'd read-only; process RSS = activations + KV cache.
 *  - Little-endian. All offsets are absolute file offsets unless noted.
 *
 * File layout
 * -----------
 *   [0]                 mlf2_header           (fixed 128 bytes)
 *   [hdr.tensors_off]   mlf2_tensor[n_tensors]  directory (name-sorted not required)
 *   [hdr.tokenizer_off] tokenizer blob        (see mlf2_tok_* below)
 *   [hdr.data_off]      tensor data           (32-byte aligned; quant blocks)
 */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MLF2_MAGIC     "MLF2"
#define MLF2_VERSION   2u
#define MLF2_ALIGN     32u
#define MLF2_MAX_NAME  48u   /* tensor names fit Qwen ("blk.35.ffn_down.weight") */

/* Architecture id: a label, not a code path. The forward pass is one shape
 * (RMSNorm → GQA attention → RoPE → SwiGLU) and every difference between
 * models is a feature of the FILE: q/k/v bias and per-head q/k norm are used
 * when their tensors exist, the attention width is n_heads * head_dim (need not
 * equal d_model), and the RoPE pairing is MLF2_FLAG_ROPE_NORM. A model of our
 * own is arch "hako" plus whichever features it uses.
 *   qwen2  neox rope (i, i+d/2), q/k/v bias
 *   llama  norm rope (2i, 2i+1), no bias — also mistral, smollm, tinyllama
 *   qwen3  neox rope, no bias, q/k norm, head_dim 128 at every size
 * Files written before ROPE_NORM existed say llama by arch id alone; the engine
 * still reads them that way. */
enum mlf2_arch {
	MLF2_ARCH_QWEN2 = 0,
	MLF2_ARCH_LLAMA = 1,
	MLF2_ARCH_QWEN3 = 2,
	MLF2_ARCH_HAKO  = 3,
	MLF2_ARCH_COUNT
};

static inline const char *mlf2_arch_name(uint32_t a) {
	static const char *const n[MLF2_ARCH_COUNT] = { "qwen2", "llama", "qwen3", "hako" };
	return a < MLF2_ARCH_COUNT ? n[a] : "?";
}

/* Tensor element types — values MUST match ggml_type so the converter copies
 * blocks without transcoding. We only implement the subset Qwen Q4_K_M uses. */
enum mlf2_type {
	MLF2_TYPE_F32  = 0,
	MLF2_TYPE_F16  = 1,
	MLF2_TYPE_Q8_0 = 8,
	MLF2_TYPE_Q4_K = 12,
	MLF2_TYPE_Q6_K = 14,
	MLF2_TYPE_BF16 = 30,   /* what Hugging Face safetensors usually hold */
};

static inline const char *mlf2_type_name(uint32_t t) {
	switch (t) {
	case MLF2_TYPE_F32:  return "f32";
	case MLF2_TYPE_F16:  return "f16";
	case MLF2_TYPE_BF16: return "bf16";
	case MLF2_TYPE_Q8_0: return "q8_0";
	case MLF2_TYPE_Q4_K: return "q4_k";
	case MLF2_TYPE_Q6_K: return "q6_k";
	default:             return "?";
	}
}

/* header.flags bits */
#define MLF2_FLAG_QKV_BIAS    (1u << 0)  /* attn q/k/v have bias vectors */
#define MLF2_FLAG_TIED_EMBED  (1u << 1)  /* lm_head == token_embd (no output.weight) */
#define MLF2_FLAG_ROPE_NORM   (1u << 2)  /* rope pairs (2i, 2i+1); clear = neox (i, i+d/2) */
/* A bit outside this mask is a feature this engine does not implement: the file
 * is refused rather than run without it, which would give wrong logits. */
#define MLF2_FLAGS_KNOWN      (MLF2_FLAG_QKV_BIAS | MLF2_FLAG_TIED_EMBED | MLF2_FLAG_ROPE_NORM)

typedef struct __attribute__((packed)) {
	char     magic[4];        /* "MLF2" */
	uint32_t version;         /* = MLF2_VERSION */
	uint32_t arch;            /* enum mlf2_arch */
	uint32_t n_layers;        /* block_count */
	uint32_t d_model;         /* embedding_length */
	uint32_t n_heads;         /* attention.head_count */
	uint32_t n_kv_heads;      /* attention.head_count_kv (GQA) */
	uint32_t head_dim;        /* attention.key_length, else d_model / n_heads */
	uint32_t ffn_dim;         /* feed_forward_length */
	uint32_t ctx;             /* context_length (max) */
	uint32_t vocab;           /* token count */
	float    rope_theta;      /* rope.freq_base (1e6 for Qwen2.5) */
	float    rms_eps;         /* attention.layer_norm_rms_epsilon */
	uint32_t flags;           /* MLF2_FLAG_* */
	uint32_t n_tensors;
	uint64_t tensors_off;     /* offset of tensor directory */
	uint64_t tokenizer_off;   /* offset of tokenizer blob */
	uint64_t data_off;        /* offset of tensor data (MLF2_ALIGN-aligned) */
	uint32_t bos_id;
	uint32_t eos_id;
	uint8_t  reserved[36];
} mlf2_header;                /* sizeof == 128 */

/* One entry in the tensor directory. `data_off` is relative to header.data_off. */
typedef struct __attribute__((packed)) {
	char     name[MLF2_MAX_NAME];  /* null-padded */
	uint32_t type;                 /* enum mlf2_type */
	uint32_t n_dims;
	uint64_t dims[4];              /* ggml order: dims[0] = fastest (cols) */
	uint64_t data_off;            /* from header.data_off */
	uint64_t nbytes;              /* size of this tensor's data */
} mlf2_tensor;

/*
 * Tokenizer blob (at header.tokenizer_off):
 *   uint32_t  n_tokens
 *   uint32_t  n_merges
 *   then n_tokens records:  uint8_t type; uint32_t len; char bytes[len]
 *       (bytes are the GGUF token strings, GPT2 byte-encoded)
 *   then n_merges records:  uint32_t len; char bytes[len]   ("A B" form)
 * Special token ids live in the header (bos_id/eos_id).
 */
#define MLF2_TOKTYPE_NORMAL   1
#define MLF2_TOKTYPE_UNKNOWN  2
#define MLF2_TOKTYPE_CONTROL  3
#define MLF2_TOKTYPE_USER     4
#define MLF2_TOKTYPE_BYTE     6

#ifdef __cplusplus
}
#endif
