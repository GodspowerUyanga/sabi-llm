#!/usr/bin/env bash
# =============================================================================
# Sabi setup — one command to a working, offline assistant.
# Tested on Ubuntu 22.04 LTS (the ADTC Standard Laptop OS).
#
# Usage:
#   ./setup.sh              # fetch the model from GitHub Releases (default)
#   ./setup.sh hf 3b        # fall back to Hugging Face, 3B model
#   ./setup.sh hf 1.5b      # Hugging Face, 1.5B (budget profile)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

SOURCE="${1:-github}"
SIZE="${2:-3b}"

echo "==> Sabi setup starting (model source: $SOURCE)"

# 1. System build tools for the CPU llama.cpp wheel (first run only)
if ! command -v cmake >/dev/null 2>&1; then
  echo "==> Installing build tools (needs sudo)"
  sudo apt-get update -y
  sudo apt-get install -y build-essential cmake python3-venv python3-dev python3-pip
fi

# 2. Virtual environment
[ -d ".venv" ] || { echo "==> Creating .venv"; python3 -m venv .venv; }
# shellcheck disable=SC1091
source .venv/bin/activate

# 3. Python dependencies
echo "==> Installing Python dependencies"
pip install --upgrade pip wheel >/dev/null
pip install -r requirements.txt
pip install -e .

# 4. Download the model (the only step that uses the network)
if [ "$SOURCE" = "hf" ]; then
  echo "==> Downloading model from Hugging Face ($SIZE) and branding as Sabi-1"
  python scripts/download_model.py --source hf --size "$SIZE"
else
  echo "==> Downloading Sabi-1 from GitHub Releases"
  python scripts/download_model.py --source github
fi

# 5. Build the RAG index over the sample corpus
echo "==> Building the document index"
python -m sabi index

echo ""
echo "==> Setup complete."
echo "    Start Sabi:     ./run.sh         (then open http://127.0.0.1:8000)"
echo "    Terminal chat:  python -m sabi chat"
echo "    Benchmark:      python -m sabi bench"
echo ""
