#!/usr/bin/env bash
# Milestone 0 Step 1: HF Gemma-2-2b → GGUF f16 → quant ladder.
#
# Requires: a built llama.cpp (llama-quantize), convert_hf_to_gguf.py,
#           a downloaded HF model.
#
# Env:
#   LLAMACPP   — path to the llama.cpp clone (default /c/Users/Impi/llamacpp-build)
#   HF_MODEL   — path to the downloaded HF model
#   GGUF_DIR   — where to put GGUF (default $LLAMACPP/_gguf)
set -e

LLAMACPP="${LLAMACPP:-/c/Users/Impi/llamacpp-build}"
HF_MODEL="${HF_MODEL:-$LLAMACPP/_gemma2b_hf}"
GGUF_DIR="${GGUF_DIR:-$LLAMACPP/_gguf}"
QUANTIZE="$LLAMACPP/build/bin/llama-quantize.exe"

mkdir -p "$GGUF_DIR"
F16="$GGUF_DIR/gemma2b-f16.gguf"

if [ ! -f "$F16" ]; then
  echo "=== convert HF → GGUF f16 ==="
  python "$LLAMACPP/convert_hf_to_gguf.py" "$HF_MODEL" \
      --outfile "$F16" --outtype f16
fi

# Quant ladder
for Q in Q8_0 Q6_K Q5_K_M Q4_K_M Q3_K_M Q2_K; do
  OUT="$GGUF_DIR/gemma2b-${Q}.gguf"
  if [ ! -f "$OUT" ]; then
    echo "=== quantize $Q ==="
    "$QUANTIZE" "$F16" "$OUT" "$Q"
  fi
done

echo ""
echo "GGUF ready:"
ls -la "$GGUF_DIR"/*.gguf | awk '{print $5, $9}'
