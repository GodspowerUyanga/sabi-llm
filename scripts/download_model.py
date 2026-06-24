#!/usr/bin/env python3
"""Download the SABI GGUF model from Hugging Face into ./models/.

The model is large, so it is hosted on Hugging Face rather than committed to
Git. This downloads it directly (no Hugging Face account or extra package
needed) using the repo/filename configured in config/default.yaml.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --repo Doctorgp1/sabi-v1 --file sabi-v1.Q4_K_M.gguf
    python scripts/download_model.py --force        # re-download even if present
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sabi.config import load_config        # noqa: E402
from sabi import downloader                 # noqa: E402


def main() -> int:
    cfg = load_config(root=ROOT)
    p = argparse.ArgumentParser(description="Download the SABI GGUF model.")
    p.add_argument("--repo", default=cfg.hf_repo_id, help="Hugging Face repo id")
    p.add_argument("--file", default=cfg.hf_filename, help="GGUF filename in the repo")
    p.add_argument("--revision", default=cfg.hf_revision, help="branch / tag / commit")
    p.add_argument("--out", default=str(cfg.abs_model_path()), help="output path")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    args = p.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"Model already present: {out}\n(use --force to re-download)")
        return 0

    try:
        path = downloader.download_model(
            cfg, repo=args.repo, filename=args.file,
            revision=args.revision, out=args.out, force=args.force,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nDone. Model saved as: {path}")
    print("Next:  sabi doctor   then   sabi run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
