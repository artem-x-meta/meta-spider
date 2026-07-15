#!/usr/bin/env bash
# Two-pass Doubter run in llama.cpp (wrapper over llama-meta-generate): base vs Doubter.
# Requires the built fork (see the meta-attention library: cpp/llama_patch/README.md).
#
# Usage: run_meta_generate.sh <base.gguf> <sidecar.gguf> <layers-csv> "<prompt>" [ngen]
#   layers-csv = cross_attn_layers from run.json (e.g. 16,17,18,19,20,21,22,23)
#   LLAMA_BIN  = path to llama-meta-generate (default ~/llamacpp-build/build/bin/llama-meta-generate)
set -euo pipefail
BASE="${1:?base.gguf}"; SIDE="${2:?sidecar.gguf}"; LAY="${3:?layers csv}"; PROMPT="${4:?prompt}"
NGEN="${5:-64}"
BIN="${LLAMA_BIN:-$HOME/llamacpp-build/build/bin/llama-meta-generate}"
common=(-m "$BASE" -c 2048 -t 4)

echo "=== BASE (clean, oracle) ==="
META_BASE=1 META_SIDECAR="$SIDE" META_LAYERS="$LAY" META_PROMPT="$PROMPT" META_NGEN="$NGEN" \
    "$BIN" "${common[@]}" 2>/dev/null

echo "=== DOUBTER (Pass1 tap → encoder → Pass2 CA injection) ==="
META_SIDECAR="$SIDE" META_LAYERS="$LAY" META_PROMPT="$PROMPT" META_NGEN="$NGEN" \
    "$BIN" "${common[@]}" 2>/dev/null
