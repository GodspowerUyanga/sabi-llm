#!/usr/bin/env bash
# Download the SABI GGUF model (sabi-3b, Qwen2.5-Coder-3B-Instruct base,
# Q4_K_M quantization, ~1.8 GB) from Hugging Face. Idempotent, no credentials
# required. Path matches _runtime.model_path in metadata.json.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/models"
MODEL_FILE="$MODEL_DIR/sabi-3b.Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/Doctorgp1/sabi-v1/resolve/main/sabi-3b.Q4_K_M.gguf"

mkdir -p "$MODEL_DIR"

if [[ -f "$MODEL_FILE" ]]; then
  echo "model already present at $MODEL_FILE — skipping download"
  exit 0
fi

echo "downloading $MODEL_URL -> $MODEL_FILE (~1.8 GB)..."
if command -v curl >/dev/null 2>&1; then
  curl -L --fail --progress-bar -o "$MODEL_FILE.partial" "$MODEL_URL"
elif command -v wget >/dev/null 2>&1; then
  wget --show-progress -O "$MODEL_FILE.partial" "$MODEL_URL"
else
  echo "neither curl nor wget available" >&2
  exit 1
fi
mv "$MODEL_FILE.partial" "$MODEL_FILE"
echo "done: $MODEL_FILE"
