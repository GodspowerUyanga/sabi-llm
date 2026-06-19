# Publishing the Sabi-1 model to GitHub Releases

The model weights are too large to commit to a normal git repo, so they live as
**release assets** on your repository. This makes `python scripts/download_model.py`
a single, reliable download from a source you control — perfect for the audit.

Repo: **https://github.com/GodspowerUyanga/sabi-llm**

You only do this **once** (and again whenever you update the model).

## Step 0 — Get the model file

If you already ran the Hugging Face path, you have it:
```
models/sabi-1.gguf
```
If not, fetch + brand it first (needs internet):
```bash
python scripts/download_model.py --source hf --size 1.5b   # ~1 GB, fits a release asset comfortably
# or --size 3b  (~2 GB, near GitHub's 2 GiB per-file limit)
```
You should also have `models/embedding.gguf` (optional but recommended).

> Tip: the **1.5B** model is the safest choice for a GitHub release asset (well
> under the 2 GiB limit) and also strengthens your Speed/Efficiency scores and
> the budget-laptop bonus. Use 3B only if your file is under 2 GiB.

## Step 1 — Create the release and upload the assets

### Option A — GitHub CLI (fastest)
```bash
# install once:  sudo apt-get install gh   &&   gh auth login
gh release create v1.0 \
  models/sabi-1.gguf models/embedding.gguf \
  --repo GodspowerUyanga/sabi-llm \
  --title "Sabi-1 model weights v1.0" \
  --notes "Quantised Sabi-1 (GGUF) + bge-small embedding model. Downloaded by scripts/download_model.py."
```

### Option B — GitHub website
1. Go to **https://github.com/GodspowerUyanga/sabi-llm/releases/new**
2. Tag: `v1.0`  ·  Title: `Sabi-1 model weights v1.0`
3. Drag `models/sabi-1.gguf` (and `models/embedding.gguf`) into **“Attach binaries.”**
4. Click **Publish release**. Wait for the upload to finish.

## Step 2 — Verify the download works

On any machine (or after deleting your local `models/*.gguf`):
```bash
python scripts/download_model.py            # pulls from your release
python -m sabi index
python -m sabi serve
```

That's it. Anyone — including the judges — can now reproduce Sabi with one
command, fully offline after the download.

## Notes
- The download script resumes interrupted transfers, so a flaky connection is fine.
- If you bump the model, upload to a new tag (e.g. `v1.1`) and run
  `python scripts/download_model.py --tag v1.1`.
- Keep `models/*.gguf` in `.gitignore` (already configured) — never commit weights.
