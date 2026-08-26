#!/usr/bin/env bash
# Download every quantized model SABI needs from Hugging Face: sabi-v1 (the
# Qwen2.5-Coder-3B-Instruct-based GGUF, Q4_K_M, ~2.0 GB) and sabi-yoruba-llm
# (the English<->Yoruba translation layer, int8 CTranslate2, ~635 MB). Both
# steps are idempotent (safe to re-run) and need only curl or wget — no
# Python, no credentials. sabi-v1's path matches _runtime.model_path in
# metadata.json.
#
# sabi-v1 is sourced from our own Doctorgp1/sabi-v1 repo (verified
# 2,104,932,800 bytes, SHA256
# 724fb256bec1ff062b2f65e4569e871ad2e95ab2a3989723d1769c54294730b7 — an
# unmodified mirror of Qwen's official Q4_K_M build, with LICENSE/NOTICE for
# the Qwen Research License) — see REPORT.md §13. sabi-yoruba-llm is sourced
# from our own Doctorgp1/sabi-yoruba-llm repo (CC-BY-NC-4.0, disclosed in
# REPORT.md §11).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/models"

fetch() {
  local url="$1" out="$2" label="$3"
  if [[ -f "$out" ]]; then
    echo "$label already present at $out — skipping"
    return 0
  fi
  echo "downloading $url -> $out"
  mkdir -p "$(dirname "$out")"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --progress-bar -o "$out.partial" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget --show-progress -O "$out.partial" "$url"
  else
    echo "neither curl nor wget available" >&2
    exit 1
  fi
  mv "$out.partial" "$out"
  echo "done: $out"
}

# ---- sabi-v1 (coder model, ~2.0 GB) ----
fetch \
  "https://huggingface.co/Doctorgp1/sabi-v1/resolve/main/sabi-v1.Q4_K_M.gguf" \
  "$MODEL_DIR/sabi-v1.Q4_K_M.gguf" \
  "sabi-v1"

# ---- sabi-yoruba-llm (translation layer, ~635 MB across 7 files) ----
YORUBA_DIR="$MODEL_DIR/sabi-yoruba-tts"
YORUBA_REPO="https://huggingface.co/Doctorgp1/sabi-yoruba-llm/resolve/main"
for f in config.json model.bin sentencepiece.bpe.model shared_vocabulary.json \
         special_tokens_map.json tokenizer.json tokenizer_config.json; do
  fetch "$YORUBA_REPO/$f" "$YORUBA_DIR/$f" "sabi-yoruba-llm/$f"
done

echo "all models ready in $MODEL_DIR"
