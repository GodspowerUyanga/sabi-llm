# Getting the SABI Model (Qwen2.5-Coder → quantized GGUF)

SABI runs a single quantized **GGUF** model through llama.cpp. Before you upload
anything to Hugging Face, you first need that quantized `.gguf` file. There are
two ways to get it.

> **Recommended base model:** `Qwen/Qwen2.5-Coder-3B-Instruct` (Apache-2.0) — chosen for the ADTC 7 GB ceiling (~2 GB on disk, ~3.5-4.5 GB at runtime).
> **Recommended quant:** `Q4_K_M` (best quality/size trade-off under the 7 GB ceiling).

---

## Path A — Fast: download a ready-made GGUF (recommended, ~4.68 GB)

Qwen already publishes pre-quantized GGUFs, so you can skip building llama.cpp.
The cleanest way is SABI's own download script, which pulls **only** the single
quantized file (~4.68 GB — not the 15 GB full model), renames it to your name,
and cleans up so `models/` holds only that one file:

```bash
pip install "huggingface_hub[cli]"

python scripts/download_model.py \
  --repo Qwen/Qwen2.5-Coder-3B-Instruct-GGUF \
  --file qwen2.5-coder-7b-instruct-q4_k_m.gguf
# -> models/sabi-3b.Q4_K_M.gguf  (your configured name; nothing else added)

sabi doctor    # verify
sabi run
```

(The output name comes from `config/default.yaml` → `model_path`. Change it
there if you want a different name.)

Once you upload this file to **your** repo (see "Upload" below), end users just
run `python scripts/download_model.py` with no arguments to fetch it.

---

## Path B — DIY: convert and quantize it yourself

Use this if you want full control (your own filename, your own fine-tune first,
or a different quant level).

**Requirements:** ~35 GB free disk, ideally 16 GB RAM. No GPU needed.

### Automated (one command)

```bash
sudo apt-get install -y build-essential cmake git libcurl4-openssl-dev
pip install "huggingface_hub[cli]"

./scripts/quantize_model.sh
# or customise:
./scripts/quantize_model.sh --hf Qwen/Qwen2.5-Coder-3B-Instruct \
                            --out sabi-3b.Q4_K_M.gguf --quant Q4_K_M
```

This builds llama.cpp, downloads the base model, converts to FP16 GGUF,
quantizes to Q4_K_M, smoke-tests it, and writes `models/sabi-3b.Q4_K_M.gguf`.

**It cleans up after itself.** The raw ~15 GB Qwen download and the FP16
intermediate go into a hidden `.model-build/` scratch folder and are deleted at
the end, so `models/` is left containing **only your one named file**
(`sabi-3b.Q4_K_M.gguf`). Pass `--keep-src` if you'd rather keep the raw download
to re-quantize at other levels without re-downloading.

### Manual (the same steps, by hand)

```bash
# 1. Build llama.cpp
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build --config Release
pip install -r requirements.txt

# 2. Download the base model
huggingface-cli download Qwen/Qwen2.5-Coder-3B-Instruct --local-dir models/qwen-src

# 3. Convert HF -> GGUF (FP16)
python convert_hf_to_gguf.py models/qwen-src \
  --outtype f16 --outfile models/sabi-3b.f16.gguf

# 4. Quantize -> Q4_K_M
./build/bin/llama-quantize \
  models/sabi-3b.f16.gguf models/sabi-3b.Q4_K_M.gguf Q4_K_M

# 5. Test
./build/bin/llama-cli -m models/sabi-3b.Q4_K_M.gguf -p "hello" -n 40
```

---

## Naming the file "your" name

The output filename is **yours to choose** — it is set in one place and flows
through everything:

- `--out <name>.gguf` on `scripts/quantize_model.sh`, **and**
- `config/default.yaml` → `model_path` and `hf_filename`.

The default is `sabi-3b.Q4_K_M.gguf`. To brand it differently (e.g.
`godspower-coder-v1.Q4_K_M.gguf`), build with:

```bash
./scripts/quantize_model.sh --out godspower-coder-v1.Q4_K_M.gguf
```

then set the same name in `config/default.yaml`:

```yaml
model_path: models/godspower-coder-v1.Q4_K_M.gguf
hf_filename: godspower-coder-v1.Q4_K_M.gguf
```

Whatever name you pick, only that single file ends up in `models/`.

---

## Choosing a quant level (and the 7 GB ceiling)

| Quant | File size (3B) | Runtime RAM (4k ctx) | Verdict for ADTC |
|-------|----------------|----------------------|------------------|
| Q4_K_M | 4.68 GB | ~5.5–6.5 GB | ✅ recommended (tight but fits) |
| Q5_K_M | ~5.4 GB (often split into 2 files) | ~6.5–7+ GB | ⚠️ risky, may exceed ceiling |
| Q8_0   | ~8.1 GB | well over 7 GB | ❌ too big |

If even the 3B is tight in `sabi benchmark`, switch to a
smaller base model — the commands are identical:

```bash
./scripts/quantize_model.sh --hf Qwen/Qwen2.5-Coder-3B-Instruct \
                            --out sabi-3b.Q4_K_M.gguf
```

A 3B at Q4_K_M is ~2 GB and very comfortable under the ceiling.

---

## Upload to Hugging Face

```bash
huggingface-cli login
huggingface-cli upload Doctorgp1/sabi-v1 \
  models/sabi-3b.Q4_K_M.gguf sabi-3b.Q4_K_M.gguf
```

After uploading, anyone who clones the repo can fetch it with:

```bash
python scripts/download_model.py
```

(The repo id and filename are read from `config/default.yaml`:
`hf_repo_id` and `hf_filename`. Update them if you used different names.)

---

## Licensing reminder

Qwen2.5-Coder is Apache-2.0. If you redistribute a quantized derivative, keep
the upstream license and attribution in your Hugging Face model card.
