#!/usr/bin/env bash
# Download the SABI GGUF model (sabi-v1, Qwen2.5-Coder-3B-Instruct base,
# Q4_K_M quantization, ~2.0 GB) from Hugging Face. Idempotent, no credentials
# required. Path matches _runtime.model_path in metadata.json.
#
# Sourced from our own Doctorgp1/sabi-v1 repo (verified 2,104,932,800 bytes,
# SHA256 724fb256bec1ff062b2f65e4569e871ad2e95ab2a3989723d1769c54294730b7 —
# an unmodified mirror of Qwen's official Q4_K_M build, with LICENSE/NOTICE
# for the Qwen Research License) — see REPORT.md §13.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/models"
MODEL_FILE="$MODEL_DIR/sabi-v1.Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/Doctorgp1/sabi-v1/resolve/main/sabi-v1.Q4_K_M.gguf"

mkdir -p "$MODEL_DIR"

if [[ -f "$MODEL_FILE" ]]; then
  echo "model already present at $MODEL_FILE — skipping download"
  exit 0
fi

echo "downloading $MODEL_URL -> $MODEL_FILE (~2.0 GB)..."
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
