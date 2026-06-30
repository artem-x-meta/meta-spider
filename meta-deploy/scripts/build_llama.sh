#!/usr/bin/env bash
# Build the meta-spider llama.cpp fork (llama-meta-generate) from scratch: clone the pinned base,
# apply the meta-injection patch, copy the examples, and configure+build for a chosen backend.
# Codifies the verified recipe (incl. the CUDA "CUDA::cuda_driver target not found" fix on hosts
# where the toolkit stubs/ dir is empty or read-only, e.g. Kaggle/Colab).
#
# Usage:
#   ./build_llama.sh --backend cuda --arch 75 --out ~/llamacpp-build   # T4 (sm_75)
#   ./build_llama.sh --backend cpu  --out ~/llamacpp-build
#   ./build_llama.sh --backend metal --out ~/llamacpp-build            # Apple
# Then run with scripts/run_meta_generate.sh (META_SIDECAR/META_LAYERS/META_PROMPT...).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="$(cd "$HERE/../llama_patch" && pwd)"
BASE_COMMIT="b9619"                       # pinned base (patch made against d8a24cc / tag b9619)
BACKEND="cuda"; ARCH=""; OUT="$HOME/llamacpp-build"
while [ $# -gt 0 ]; do case "$1" in
  --backend) BACKEND="$2"; shift 2;;
  --arch)    ARCH="$2";    shift 2;;       # CUDA arch, e.g. 75 (T4), 80 (A100), 89 (L4/4090)
  --out)     OUT="$2";     shift 2;;
  *) echo "unknown arg: $1"; exit 2;;
esac; done

echo "==> clone llama.cpp @ $BASE_COMMIT -> $OUT"
[ -d "$OUT" ] || git clone -q https://github.com/ggml-org/llama.cpp "$OUT"
git -C "$OUT" checkout -q "$BASE_COMMIT"

echo "==> apply meta-spider patch + copy examples"
git -C "$OUT" apply "$PATCH_DIR/meta-spider-llama.patch" 2>/dev/null \
  || { echo "patch already applied or hunks differ on this base"; }
cp -r "$PATCH_DIR/examples/meta-generate" "$OUT/examples/" 2>/dev/null || true
cp -r "$PATCH_DIR/examples/extract-activations" "$OUT/examples/" 2>/dev/null || true

CMAKE_FLAGS=(-DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release)
case "$BACKEND" in
  cuda)
    CMAKE_FLAGS+=(-DGGML_CUDA=ON -DCUDAToolkit_ROOT="${CUDA_HOME:-/usr/local/cuda}")
    [ -n "$ARCH" ] && CMAKE_FLAGS+=(-DCMAKE_CUDA_ARCHITECTURES="$ARCH")
    # FIX: on Kaggle/Colab the toolkit stubs/ is empty AND read-only -> FindCUDAToolkit fails to
    # create CUDA::cuda_driver. Point cmake at the real libcuda.so explicitly.
    for c in /usr/local/nvidia/lib64/libcuda.so /usr/local/cuda*/compat/libcuda.so \
             /usr/lib/x86_64-linux-gnu/libcuda.so; do
      [ -e "$c" ] && { CMAKE_FLAGS+=(-DCUDA_cuda_driver_LIBRARY="$c" -DCMAKE_LIBRARY_PATH="$(dirname "$c")"); break; }
    done ;;
  metal) CMAKE_FLAGS+=(-DGGML_METAL=ON) ;;
  cpu)   : ;;
  *) echo "unknown backend: $BACKEND (cuda|cpu|metal)"; exit 2;;
esac

echo "==> cmake configure (${BACKEND}): ${CMAKE_FLAGS[*]}"
cmake -B "$OUT/build" -S "$OUT" "${CMAKE_FLAGS[@]}"
echo "==> build llama-meta-generate"
cmake --build "$OUT/build" --target llama-meta-generate -j "$(nproc 2>/dev/null || echo 4)"

BIN="$(find "$OUT/build" -name llama-meta-generate -type f | head -1)"
echo "==> done: $BIN"
