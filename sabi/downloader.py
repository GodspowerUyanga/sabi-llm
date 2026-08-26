"""Model downloader.

Fetches the quantized GGUF from Hugging Face into ./models/ with zero required
dependencies: it streams the file directly from the public ``resolve`` URL using
the standard library (urllib), showing progress. If that fails and
``huggingface_hub`` happens to be installed, it falls back to that (useful for
private repos / auth).

Because the download is direct, judges who clone the repo only need to run one
command (or just start SABI) -- no Hugging Face account or extra package needed.
"""

from __future__ import annotations

import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

import shutil

from .config import Config

# SABI's own pre-converted Yoruba translation layer (int8 CTranslate2).
# See scripts/download_yoruba_model.py for the from-scratch conversion path.
YORUBA_HF_REPO = "Doctorgp1/sabi-yoruba-llm"


def resolve_url(repo_id: str, filename: str, revision: str = "main") -> str:
    """Build the direct-download URL for a public Hugging Face file."""
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _http_download(url: str, out: Path, progress: bool = True) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "sabi-downloader/1.0"})
    tmp = out.with_suffix(out.suffix + ".part")
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (trusted host)
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        chunk = 1024 * 256
        with open(tmp, "wb") as fh:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                fh.write(buf)
                done += len(buf)
                if progress and total:
                    pct = done / total * 100
                    bar_len = 28
                    filled = int(bar_len * done / total)
                    bar = "█" * filled + "·" * (bar_len - filled)
                    sys.stdout.write(
                        f"\r  [{bar}] {pct:5.1f}%  {_human(done)} / {_human(total)}"
                    )
                    sys.stdout.flush()
    if progress:
        sys.stdout.write("\n")
    tmp.replace(out)


def download_model(
    config: Config,
    repo: Optional[str] = None,
    filename: Optional[str] = None,
    revision: Optional[str] = None,
    out: Optional[str] = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Download the model into models/ and return its path.

    Skips the download if the file already exists (unless ``force``).
    """
    repo = repo or config.hf_repo_id
    filename = filename or config.hf_filename
    revision = revision or getattr(config, "hf_revision", "main")
    out_path = Path(out) if out else config.abs_model_path()

    if out_path.exists() and not force:
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = resolve_url(repo, filename, revision)

    if progress:
        print(f"Downloading {filename}\n  from {url}\n  to   {out_path}")

    # 1) Primary: direct HTTPS (no extra dependency needed).
    try:
        _http_download(url, out_path, progress=progress)
        return out_path
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        direct_err = exc

    # 2) Fallback: huggingface_hub, if available (handles auth / private repos).
    try:
        from huggingface_hub import hf_hub_download
        import shutil

        downloaded = Path(hf_hub_download(
            repo_id=repo, filename=filename, revision=revision,
            local_dir=str(out_path.parent),
        ))
        if downloaded.resolve() != out_path.resolve():
            if out_path.exists():
                out_path.unlink()
            shutil.move(str(downloaded), str(out_path))
        cache = out_path.parent / ".cache"
        if cache.exists():
            shutil.rmtree(cache, ignore_errors=True)
        return out_path
    except ImportError:
        raise RuntimeError(
            f"Could not download the model directly ({direct_err}). "
            "Check your internet connection, or install huggingface_hub "
            "(`pip install huggingface_hub`) and try again."
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Model download failed.\n  direct: {direct_err}\n  hub: {exc}\n"
            f"Verify the file exists at {url}"
        )


def download_yoruba_model(config: Config, force: bool = False, progress: bool = True) -> Path:
    """Fetch the pre-converted sabi-yoruba-llm translation layer into models/.

    Downloads the already-quantized int8 CTranslate2 model directly from
    SABI's own Hugging Face repo (~635 MB) -- no torch, no conversion step.
    Skips the download if it's already present (unless ``force``).
    """
    out_dir = config.abs_yoruba_model_path()
    if out_dir.exists() and (out_dir / "model.bin").exists() and not force:
        return out_dir

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError(
            "Could not download the Yoruba translation model: huggingface_hub "
            "is not installed. Install the translate extra "
            '(`pip install "sabi-llm[translate]"`) or `pip install huggingface_hub`.'
        )

    if progress:
        print(f"Downloading sabi-yoruba-llm from {YORUBA_HF_REPO} (~635 MB) to {out_dir}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        YORUBA_HF_REPO,
        local_dir=str(out_dir),
        allow_patterns=["*.bin", "*.json", "*.model"],
    )

    if progress:
        print(f"sabi-yoruba-llm ready at {out_dir}")
    return out_dir
