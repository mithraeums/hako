/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* models.h — the local model store: ~/.hako/models/<id>/<id>.mlf2.
 *
 * `hakm` used to take a file path and nothing else; the agent built the path
 * itself. This is the engine's own idea of where models live, so a bare id
 * works everywhere: `hakm hako-sho`, `hakm pull mithraeum/hako-sho`,
 * `hakm list`, `hakm rm hako-sho`. */
#pragma once
#include <stddef.h>

/* $HAKO_HOME/models, else $HOME/.hako/models. Created on demand. 0 = ok. */
int hakm_models_dir(char *out, size_t cap);

/* Resolve what the user typed to a weight file:
 *   a path that exists (or ends in .mlf2)  → as is
 *   org/id                                  → <models>/<id>/<id>.mlf2
 *   id                                      → <models>/<id>/<id>.mlf2
 * Returns 0 and fills `out`; -1 if the id is malformed. Does NOT check that
 * the file exists — callers say what they were looking for. */
int hakm_resolve(const char *spec, char *out, size_t cap);

/* One row per installed model on stdout. Returns the count. */
int hakm_list(void);

/* Download org/id (default org "mithraeum", or $HAKO_HF_ORG) from Hugging Face
 * with `curl`. Resumes a partial .part, verifies the MLF2 magic, renames into
 * place atomically. Progress on stderr, one \r-refreshed line, only at a tty.
 * 0 = installed (or already there), -1 = failed. */
int hakm_pull(const char *spec);

/* Remove an installed model and every KV-cache entry keyed to it. Exact id
 * only — a path or a partial name is refused. 0 = removed, -1 = refused. */
int hakm_rm(const char *id);
