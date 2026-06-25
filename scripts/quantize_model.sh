#!/usr/bin/env bash
#
# quantize_model.sh - build llama.cpp, download a Hugging Face model, convert it
# to GGUF, quantize it (default Q4_K_M), and leave ONLY your single named model
# file in the models/ folder.
#
# Usage:
#   ./scripts/quantize_model.sh
#   ./scripts/quantize_model.sh --hf Qwen/Qwen2.5-Coder-3B-Instruct \
#                               --out sabi-3b.Q4_K_M.gguf --quant Q4_K_M
#   ./scripts/quantize_model.sh --keep-src      # keep the raw download to re-quantize later
#
# Result:  models/<your-out-name>.gguf   (nothing else added to models/)
#
# Requirements (Ubuntu): build-essential cmake git libcurl4-openssl-dev python3
#   sudo apt-get install -y build-essential cmake git libcurl4-openssl-dev
#   pip install "huggingface_hub[cli]"
#
# Notes:
#   * A 3B conversion needs ~15 GB free disk and ideally 8-16 GB RAM.
#   * No GPU is required to convert/quantize.
#   * For the ADTC 7 GB ceiling, a 3B at Q4_K_M is the recommended sweet spot
#     (~2 GB on disk, ~3.5-4.5 GB at runtime). If you need even smaller, try
#     --hf Qwen/Qwen2.5-Coder-1.5B-Instruct, or --quant Q4_0 for a tighter file.
#
set -euo pipefail

# ---- Defaults ----
HF_MODEL="Qwen/Qwen2.5-Coder-3B-Instruct"
OUT_NAME="sabi-3b.Q4_K_M.gguf"
QUANT="Q4_K_M"
KEEP_SRC=0
LLAMA_DIR=".llama.cpp"          # hidden build dir (not inside models/)
BUILD_DIR=".model-build"        # hidden scratch dir for the raw download + f16

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$ROOT/models"
SRC_DIR="$ROOT/$BUILD_DIR/qwen-src"

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf)       HF_MODEL="$2"; shift 2;;
    --out)      OUT_NAME="$2"; shift 2;;
    --quant)    QUANT="$2";    shift 2;;
    --keep-src) KEEP_SRC=1;    shift 1;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

echo "=============================================="
echo " SABI model quantizer"
echo "   base model : $HF_MODEL"
echo "   quant type : $QUANT"
echo "   output     : models/$OUT_NAME   (only this file is kept)"
echo "=============================================="

mkdir -p "$MODELS_DIR" "$ROOT/$BUILD_DIR"

# ---- 1. Build llama.cpp (once) ----
if [[ ! -x "$ROOT/$LLAMA_DIR/build/bin/llama-quantize" ]]; then
  echo ">> Cloning and building llama.cpp ..."
  if [[ ! -d "$ROOT/$LLAMA_DIR" ]]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$ROOT/$LLAMA_DIR"
  fi
  ( cd "$ROOT/$LLAMA_DIR" \
      && cmake -B build \
      && cmake --build build --config Release -j \
      && pip install -r requirements.txt )
else
  echo ">> llama.cpp already built, skipping."
fi

CONVERT="$ROOT/$LLAMA_DIR/convert_hf_to_gguf.py"
QUANTIZE="$ROOT/$LLAMA_DIR/build/bin/llama-quantize"
CLI="$ROOT/$LLAMA_DIR/build/bin/llama-cli"

# ---- 2. Download base model (into hidden scratch dir, NOT models/) ----
echo ">> Downloading $HF_MODEL ..."
huggingface-cli download "$HF_MODEL" --local-dir "$SRC_DIR"

# ---- 3. Convert HF -> GGUF (FP16 intermediate, in scratch dir) ----
F16="$ROOT/$BUILD_DIR/${OUT_NAME%.gguf}.f16.gguf"
echo ">> Converting to FP16 GGUF ..."
python "$CONVERT" "$SRC_DIR" --outtype f16 --outfile "$F16"

# ---- 4. Quantize straight into models/ with YOUR name ----
OUT="$MODELS_DIR/$OUT_NAME"
echo ">> Quantizing to $QUANT -> models/$OUT_NAME ..."
"$QUANTIZE" "$F16" "$OUT" "$QUANT"

# ---- 5. Smoke test ----
echo ">> Smoke-testing the quantized model ..."
"$CLI" -m "$OUT" -p "Write a Python function that returns the square of a number." -n 60 || true

# ---- 6. Clean up so models/ holds ONLY your named file ----
echo ">> Cleaning up intermediates ..."
rm -f "$F16"
if [[ "$KEEP_SRC" -eq 0 ]]; then
  rm -rf "$ROOT/$BUILD_DIR"
  echo "   removed raw download + scratch (use --keep-src to keep it next time)"
else
  echo "   kept raw download at $SRC_DIR (--keep-src)"
fi

echo ""
echo "Done. models/ now contains only: $OUT_NAME"
ls -lh "$MODELS_DIR" | grep -E "\.gguf$" || true
echo ""
echo "Next steps:"
echo "  1) Verify:  sabi doctor"
echo "  2) Upload to your HF repo:"
echo "       huggingface-cli login"
echo "       huggingface-cli upload godspoweruyanga/sabi-llm-gguf \"\$OUT\" \"\$OUT_NAME\""
