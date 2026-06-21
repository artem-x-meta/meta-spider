#!/usr/bin/env bash
# Build the standalone ggml sidecar validators (meta_encoder/meta_ca/meta_selective).
# The "MinGW Makefiles" cmake generator needs mingw32-make (often missing) → we build with g++
# directly, mirroring the CMakeLists flags. CMakeLists remains the canonical dependency description.
#
# Requires: g++ (mingw/gcc), a built llama.cpp/ggml in $LLAMA_DIR with ggml*.a.
# Usage:  LLAMA_DIR=~/llamacpp-build ./build.sh [meta_selective ...]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$HOME/llamacpp-build}"
INC="$LLAMA_DIR/ggml/include"
LIBD="$LLAMA_DIR/build/ggml/src"
mkdir -p "$HERE/build"

for lib in ggml.a ggml-cpu.a ggml-base.a; do
    [ -f "$LIBD/$lib" ] || { echo "MISSING $LIBD/$lib — build llama.cpp/ggml (LLAMA_DIR=$LLAMA_DIR)"; exit 1; }
done

tools=("$@")
[ ${#tools[@]} -eq 0 ] && tools=(meta_encoder meta_ca meta_selective)

for t in "${tools[@]}"; do
    echo "→ g++ $t.cpp"
    g++ -std=c++17 -O2 -fopenmp "$HERE/$t.cpp" \
        -I"$INC" \
        -Wl,--start-group "$LIBD/ggml.a" "$LIBD/ggml-cpu.a" "$LIBD/ggml-base.a" -Wl,--end-group \
        -fopenmp -o "$HERE/build/$t.exe"
done
echo "OK → $HERE/build/{$(IFS=,; echo "${tools[*]}")}.exe"
