# Models

The quantized GGUF model is **not** stored in this repository — it is large, so
it is hosted on **Hugging Face** and downloaded on demand.

## Download

```bash
python scripts/download_model.py
```

This fetches the file named in `config/default.yaml` (default
`sabi-v1.Q4_K_M.gguf` from `godspoweruyanga/sabi-llm-gguf`) into this folder.

To use a different repo/file:

```bash
python scripts/download_model.py --repo <hf-repo-id> --file <model.gguf>
```

Or set the path directly via the `SABI_MODEL_PATH` environment variable / the
`model_path` key in `config/default.yaml`.

After downloading, verify with:

```bash
sabi doctor
```

> `*.gguf` files are git-ignored, so the downloaded model stays out of version
> control.
