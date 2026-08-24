#!/usr/bin/env bash
# Download the SABI GGUF model (sabi-v1, Qwen2.5-Coder-3B-Instruct base,
# Q4_K_M quantization, ~2.0 GB) from Hugging Face. Idempotent, no credentials
# required. Path matches _runtime.model_path in metadata.json.
#
# Sourced directly from Qwen's official GGUF repo (verified 2,104,932,800 bytes
# for this exact file via the HF API) rather than our own Doctorgp1/sabi-v1
# upload, which was found to be mismatched at 4.68 GB (7B-class size, not the
# 3B build this submission claims and benchmarks) — see REPORT.md §6/§13.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/models"
MODEL_FILE="$MODEL_DIR/sabi-v1.Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf"

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
