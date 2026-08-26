#!/usr/bin/env python3
"""
Fetch sabi-yoruba-llm — the English<->Yoruba translation layer.

Source: Meta's official facebook/nllb-200-distilled-600M checkpoint (NLLB-200,
"No Language Left Behind"; Yoruba = yor_Latn), licensed CC-BY-NC-4.0 —
non-commercial. Unlike SABI's other model components (MIT), this restricts
commercial use of the weights themselves; disclosed in REPORT.md §11 — fine
for this non-commercial hackathon submission, but check it fits any other use.
It runs as an int8-quantized CTranslate2 model so SABI's runtime never needs
`torch` (only `ctranslate2` + a tokenizer). This mirrors how download_model.sh
turns a full-precision GGUF download into the quantized model SABI actually
runs.

Two ways to get it:

  Path A — fast (default): download the already-converted int8 model
  directly from SABI's own Hugging Face repo:
  https://huggingface.co/Doctorgp1/sabi-yoruba-llm  (~635 MB, no torch needed)

      pip install huggingface_hub
      python scripts/download_yoruba_model.py

  Path B — from scratch: convert the original fp32 checkpoint yourself
  (needs `torch` just for this one-time step; downloads ~2.4 GB):

      pip install --index-url https://download.pytorch.org/whl/cpu torch
      pip install ctranslate2 transformers
      python scripts/download_yoruba_model.py --from-scratch
      pip uninstall -y torch          # optional: not needed after conversion

After either path, the model lands at:
    models/sabi-yoruba-tts/         (int8 CTranslate2 model + tokenizer files)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sabi.config import load_config        # noqa: E402
from sabi import downloader                 # noqa: E402

MODELS = ROOT / "models"
SABI_HF_REPO = downloader.YORUBA_HF_REPO
HF_REPO = "facebook/nllb-200-distilled-600M"
OUT_DIR = MODELS / "sabi-yoruba-tts"
CACHE_DIR = MODELS / ".hf_cache"


def download(force: bool = False) -> None:
    try:
        downloader.download_yoruba_model(load_config(root=ROOT), force=force)
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)
    print("  yoruba_enabled: true is already the default in config/default.yaml —")
    print("  SABI will translate Yoruba turns automatically.\n")


def convert(force: bool = False) -> None:
    if OUT_DIR.exists() and (OUT_DIR / "model.bin").exists() and not force:
        print(f"  sabi-yoruba-llm already present at {OUT_DIR} (use --force to redo)")
        return

    try:
        import ctranslate2  # noqa: F401
    except ImportError:
        print("Install deps first:  pip install ctranslate2 transformers")
        sys.exit(1)
    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "This one-time conversion step needs a CPU build of torch "
            "(not required at runtime):\n"
            "  pip install --index-url https://download.pytorch.org/whl/cpu torch\n"
        )
        sys.exit(1)

    from ctranslate2.converters import TransformersConverter

    print(f"\nDownloading + converting {HF_REPO} -> int8 CTranslate2 ({OUT_DIR})")
    print("This downloads ~2.4 GB of fp32 weights once; only the ~600 MB int8")
    print("conversion output is kept.\n")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    converter = TransformersConverter(HF_REPO, load_as_float16=False)
    converter.convert(str(OUT_DIR), quantization="int8", force=True)

    # TransformersConverter only writes CT2's own vocab format; translate.py's
    # AutoTokenizer needs the actual HF tokenizer files saved alongside it.
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(HF_REPO).save_pretrained(str(OUT_DIR))

    print(f"\n  Done. sabi-yoruba-llm is ready at {OUT_DIR}")
    print("  yoruba_enabled: true is already the default in config/default.yaml —")
    print("  SABI will translate Yoruba turns automatically.\n")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Download sabi-yoruba-llm.")
    ap.add_argument("--force", action="store_true", help="redo even if already present")
    ap.add_argument(
        "--from-scratch",
        action="store_true",
        help="convert from the original fp32 checkpoint instead of downloading "
        "the pre-converted model (needs torch, ctranslate2, transformers)",
    )
    args = ap.parse_args()
    if args.from_scratch:
        convert(force=args.force)
    else:
        download(force=args.force)


if __name__ == "__main__":
    main()
