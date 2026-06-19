#!/usr/bin/env python3
"""
Download the Sabi-1 model — by default from THIS project's GitHub Releases.

Why GitHub Releases?
  - A 1–2 GB model cannot live inside a git repo, but it CAN be attached to a
    GitHub *Release* as an asset (up to 2 GiB per file).
  - Hosting the weights on your own repo makes the audit one reliable download
    from a single source you control — no Hugging Face account, token, or proxy.
  - This is the ONLY step that touches the network. After it completes, Sabi
    runs 100% offline.

Default source (edit RELEASE_* below or pass --repo/--tag to change):
    https://github.com/GodspowerUyanga/sabi-llm/releases/download/<tag>/<asset>

Usage:
    python scripts/download_model.py                  # GitHub release (default)
    python scripts/download_model.py --tag v1.0       # pick a release tag
    python scripts/download_model.py --source hf --size 3b   # fallback: Hugging Face

After this, the files land at:
    models/sabi-1.gguf       (the chat model, already branded Sabi-1)
    models/embedding.gguf    (the RAG embedding model)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

# ---- GitHub release configuration (your repo) ------------------------------
RELEASE_REPO = "GodspowerUyanga/sabi-llm"
RELEASE_TAG = "v1.0"
CHAT_ASSET = "sabi-1.gguf"
EMBED_ASSET = "embedding.gguf"

# ---- Hugging Face fallback configuration -----------------------------------
HF_CHAT = {
    "1.5b": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    "3b":   ("Qwen/Qwen2.5-3B-Instruct-GGUF",   "qwen2.5-3b-instruct-q4_k_m.gguf"),
}
HF_EMBED = ("CompendiumLabs/bge-small-en-v1.5-gguf", "bge-small-en-v1.5-f16.gguf")


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def download_stream(url: str, dest: Path, label: str) -> bool:
    """Stream a file to disk with a progress bar and HTTP-range resume."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}

    with requests.get(url, headers=headers, stream=True, timeout=30, allow_redirects=True) as r:
        if r.status_code == 416:  # already complete
            tmp.rename(dest)
            print(f"  {label}: already downloaded")
            return True
        if r.status_code not in (200, 206):
            print(f"  {label}: server returned HTTP {r.status_code} for {url}")
            return False
        total = int(r.headers.get("content-length", 0)) + existing
        mode = "ab" if existing and r.status_code == 206 else "wb"
        if mode == "wb":
            existing = 0
        done = existing
        bar_w = 40
        with open(tmp, mode) as fh:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                fh.write(chunk)
                done += len(chunk)
                if total:
                    frac = done / total
                    filled = int(bar_w * frac)
                    sys.stdout.write(
                        f"\r  {label}: |{'█' * filled}{' ' * (bar_w - filled)}| "
                        f"{frac*100:5.1f}%  {_human(done)}/{_human(total)}"
                    )
                else:
                    sys.stdout.write(f"\r  {label}: {_human(done)}")
                sys.stdout.flush()
        sys.stdout.write("\n")
    tmp.rename(dest)
    return True


def from_github(tag: str, repo: str) -> None:
    base = f"https://github.com/{repo}/releases/download/{tag}"
    print(f"\nDownloading Sabi-1 from GitHub release {repo}@{tag}\n")
    ok_chat = download_stream(f"{base}/{CHAT_ASSET}", MODELS / "sabi-1.gguf", "sabi-1.gguf")
    if not ok_chat:
        print(
            "\n  Could not fetch the chat model from GitHub.\n"
            "  Make sure the release exists and the asset is uploaded:\n"
            f"    https://github.com/{repo}/releases/tag/{tag}\n"
            "  See docs/PUBLISH_MODEL.md to publish it, or use:\n"
            "    python scripts/download_model.py --source hf --size 3b\n"
        )
        sys.exit(1)
    # Embedding is optional — RAG falls back to a local lexical embedder if absent.
    download_stream(f"{base}/{EMBED_ASSET}", MODELS / "embedding.gguf", "embedding.gguf")
    print("\n  Done. Sabi-1 is ready and will run fully offline.")
    print("  Next:  python -m sabi index   then   python -m sabi serve\n")


def from_huggingface(size: str) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install deps first:  pip install -r requirements.txt")
        sys.exit(1)
    import subprocess
    MODELS.mkdir(parents=True, exist_ok=True)
    repo, filename = HF_CHAT[size]
    print(f"\nDownloading base model from Hugging Face ({size})…\n  ↓ {repo} :: {filename}")
    raw = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(MODELS))
    try:
        erepo, efile = HF_EMBED
        print(f"  ↓ {erepo} :: {efile}")
        epath = hf_hub_download(repo_id=erepo, filename=efile, local_dir=str(MODELS))
        target = MODELS / "embedding.gguf"
        if not target.exists():
            target.write_bytes(Path(epath).read_bytes())
    except Exception as exc:
        print(f"  ! embedding download failed ({exc}); RAG will use the local lexical fallback.")
    print("\nBranding → Sabi-1 …")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "customize_model.py"),
                    "--input", str(raw)], check=False)
    print("\n  Done. Next:  python -m sabi index  then  python -m sabi serve\n")


def main():
    ap = argparse.ArgumentParser(description="Download the Sabi-1 model.")
    ap.add_argument("--source", choices=["github", "hf"], default="github",
                    help="where to fetch the model (default: github)")
    ap.add_argument("--repo", default=RELEASE_REPO, help="GitHub owner/repo")
    ap.add_argument("--tag", default=RELEASE_TAG, help="GitHub release tag")
    ap.add_argument("--size", choices=list(HF_CHAT), default="3b",
                    help="Hugging Face model size (only with --source hf)")
    args = ap.parse_args()

    if args.source == "github":
        from_github(args.tag, args.repo)
    else:
        from_huggingface(args.size)


if __name__ == "__main__":
    main()
