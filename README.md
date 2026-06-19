<div align="center">

# Sabi
### Offline Enterprise Knowledge Assistant for African SMEs

*"Sabi" — West African Pidgin for "to know."*

**Africa Deep Tech Challenge 2026 · Corporate / Enterprise track**

100% on-device · No cloud · No GPU · Runs comfortably on an 8 GB laptop

[`github.com/GodspowerUyanga/sabi-llm`](https://github.com/GodspowerUyanga/sabi-llm)

</div>

---

## Overview

**Sabi** is a private, fully offline AI assistant for small and medium businesses and the
people who run them, **created and trained by Godspower Uyanga**. It reads your own documents,
answers questions about your spreadsheets with **exact** figures, builds pivot tables and Excel
sheets, keeps a memory of useful facts, and remembers the conversation — all running locally on
a modest laptop, with nothing ever leaving the device.

The product interface is **Sabi v1**; the model is **Sabi-1**. Sabi pairs a compact,
quantized open-source language model — rebranded and behaviourally specialised by Godspower
Uyanga for enterprise knowledge work — with retrieval over your local files and a set of
**deterministic data tools**. The language model handles language; the tools handle the maths.
That separation is the heart of the design: it means the numbers are always right, regardless of
model size.

## Why a small model, done right, beats a big one

The accuracy-critical work in Sabi — *who is owing, how many debtors, the highest payment,
totals, averages, pivot tables, Excel exports* — is computed **in code from the actual cells**,
never inferred by the model. A language model is only asked to do what language models are good
at: understanding the question and writing the reply.

This is why Sabi ships on the **1.5B profile by default**. It is small (~1.1 GB), fast, and
light on memory, yet loses none of its accuracy on business data, because that accuracy comes
from the deterministic engine rather than the model. The result is a better balance across the
three things the competition measures: accuracy, speed, and efficiency.

## Capabilities

- **Talk to your documents** — upload **PDF, Word (.docx), Excel (.xlsx), CSV, text or Markdown**, then ask questions or request summaries. Every grounded answer cites the files it used.
- **Exact answers from spreadsheets** — *"who is currently owing?"*, *"how many debtors?"*, *"who made the highest payment?"*, totals, averages and counts are all computed deterministically from the real data and shown as clean tables. The model never does the arithmetic, so the figures are exact.
- **Pivot tables** — ask for *"a pivot table of revenue by region and month"* and Sabi builds a true cross-tab (rows × columns) with row totals, column totals and a grand total.
- **Create Excel sheets** — ask Sabi to *"create a Debtors sheet with names and balances"* and it writes a real, downloadable `.xlsx` built from your actual records — generated entirely on-device.
- **Persistent chat history** — the Sabi v1 interface keeps every conversation in a sidebar: start a new chat, switch between past chats, or delete them. History lives on your device.
- **Knowledge base** — tell Sabi *"remember this …"* and it stores the fact locally and recalls it later.
- **Conversation memory** — Sabi keeps recent turns as context for natural follow-up questions.
- **Compliance dashboard** — a one-glance view of every ADTC constraint, live (see below).
- **Languages** — clear English (the competition's primary evaluation language) and natural Nigerian Pidgin.

## How it fits the ADTC brief

| Requirement | How Sabi meets it |
|---|---|
| Runs on the ADTC Standard Laptop (8 GB RAM, integrated graphics, no GPU) | Quantized GGUF via llama.cpp on CPU, memory-mapped weights, small context window — typical peak RAM ~2–2.5 GB on the 1.5B profile |
| No cloud dependencies | Inference is 100% offline; the only network use is the one-time model download |
| One problem domain | Corporate / Enterprise — knowledge-work productivity for SMEs and operators |
| ≥1 cross-disciplinary integration | Knowledge work **×** quantitative reasoning: retrieval-grounded answers paired with a safe calculator, deterministic spreadsheet analysis, pivot tables and table queries |
| Accuracy (50%) | Document grounding plus deterministic computation — no hallucinated totals, counts or balances |
| Efficiency (20%) | Lazy-loaded, released embedder; mmap; small batch and context; live RAM gauge and profiler |
| Speed (30%) | Compact 1.5B model with token streaming for responsiveness on modest hardware |
| Language & African use case | English (primary) + Nigerian Pidgin, built for real African SME workflows |

See [`REPORT.md`](REPORT.md) for the full design rationale and benchmark methodology.

## The model: Sabi-1

- **Base:** Qwen2.5-Instruct (Apache-2.0), GGUF, `q4_k_m` quantization.
- **Default profile:** **1.5B** (~1.1 GB) — recommended for the target hardware. A **3B** profile (~2.1 GB) is available if you prefer richer free-text writing and can spare the RAM; data accuracy is identical either way.
- **Embeddings:** bge-small-en-v1.5 (GGUF) when present; otherwise a local lexical fallback (hashed character n-grams) so retrieval still works with zero extra downloads.
- **Customization:** metadata rebrand to `Sabi-1`, a fixed enterprise persona, retrieval grounding, a tool-calling protocol, and low-temperature decoding for consistent, factual answers. Honest scope is documented in `REPORT.md` §4.

## Quickstart

```bash
git clone https://github.com/GodspowerUyanga/sabi-llm.git
cd sabi-llm
./setup.sh            # installs dependencies, downloads the model, builds the index
./run.sh              # then open the printed http://127.0.0.1:8000
```

`setup.sh` fetches the model from the repository's **GitHub Releases** by default — one
reliable download, with no Hugging Face account or proxy required. To use Hugging Face instead,
run `./setup.sh hf 1.5b` (or `./setup.sh hf 3b`). After the single download, Sabi runs entirely
offline.

### How the model is provided

A 1–2 GB model cannot live inside a git repository, so the weights are attached to the repo's
**GitHub Releases** and pulled on demand by `scripts/download_model.py` (with resume support).

- **Using Sabi:** `./setup.sh` (or `python scripts/download_model.py`) downloads `sabi-1.gguf`
  into `models/`, ready to run with no internet thereafter.
- **Publishing the model (owner, one time):** see [`docs/PUBLISH_MODEL.md`](docs/PUBLISH_MODEL.md)
  for uploading the weights to a release — this is what makes the audit reproducible.

### Try it without the model first

Sabi ships with a deterministic **mock** backend so you can verify the whole pipeline —
retrieval, tools, streaming UI, benchmark, and the full test suite — before downloading anything:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m sabi index
pytest -q              # 48 passed
python -m sabi chat    # mock mode until the GGUF model is present
```

## Commands

| Command | What it does |
|---|---|
| `python -m sabi serve` | Web UI and streaming API at `http://127.0.0.1:8000` |
| `python -m sabi chat` | Interactive terminal chat |
| `python -m sabi index` | (Re)build the retrieval index from `data/corpus/` |
| `python -m sabi bench` | Local profiler: tokens/sec, peak RAM, efficiency score, thermal, OOM check |
| `python -m sabi compliance` | Print and save the ADTC compliance report (`COMPLIANCE.md`) |

Drop your own `.pdf`, `.docx`, `.xlsx`, `.csv`, `.md` or `.txt` files into `data/corpus/`,
re-run `python -m sabi index`, and Sabi answers from them — with the sources shown under each
answer.

## The Sabi v1 interface

- **Upload documents** with the 📎 button (**PDF, Word, Excel, CSV, text, Markdown**). Sabi
  extracts the text, indexes it on the spot, and you can immediately ask questions or click
  **Summarise it** — all locally; the file never leaves your machine.
- **Persistent chat history** in the sidebar — every conversation is saved on your device, with
  new-chat, switch and delete controls.
- **Manage documents** — a **🗂 Manage documents** panel lists every indexed file with its size and
  lets you delete any of them (or clear all); Sabi re-indexes instantly so it forgets what you remove.
- **Knows what it has** — ask *"what documents do you have?"* or *"do you have access to X?"* and Sabi
  answers from its actual index, instead of guessing.
- **Streaming answers** token-by-token for responsiveness on slow hardware.
- **Source citations** under every document-grounded answer.
- **Tool traces** — when Sabi computes a figure or builds a table, you see exactly what it ran.
- **Live RAM gauge** against the 7 GB budget, plus offline and private status badges.
- Fully self-contained: no external fonts, scripts or CDNs — the interface works offline too.

## Compliance dashboard — proof at a glance

Sabi makes its ADTC compliance **visible and verifiable** so reviewers don't have to check by
hand. Click **✓ ADTC Compliance** in the interface (or run `python -m sabi compliance`) to see
a live report drawn from the running machine:

- Model weight on disk and format (GGUF, `q4_k_m`)
- Peak RAM against the 7 GB budget, and the resulting efficiency score
- Generation speed (tokens/sec) and core temperature
- CPU and operating system
- The scoring formula and a pass/fail checklist against every published constraint

## Repository layout

```
sabi-llm/
├── src/sabi/
│   ├── config.py        # all tunables (loaded from config/sabi.yaml)
│   ├── model.py         # llama.cpp wrapper (LlamaChat) + MockChat, context budgeting
│   ├── embeddings.py    # llama.cpp embeddings + local lexical fallback
│   ├── rag.py           # chunking, numpy vector store, search
│   ├── tools.py         # calculator, spreadsheet analysis, pivot tables, table queries, Excel export
│   ├── agent.py         # retrieve → tool-loop → stream orchestrator (typed events)
│   ├── languages.py     # English + Nigerian Pidgin detection
│   ├── compliance.py    # telemetry + ADTC pass/fail checklist
│   ├── prompts.py       # Sabi-1 persona, tool protocol, grounding rules
│   ├── memory.py        # RAM monitor + ADTC efficiency scoring
│   └── app.py           # FastAPI server + CLI
├── scripts/
│   ├── download_model.py    # fetch Sabi-1 from GitHub Releases (Hugging Face fallback)
│   ├── customize_model.py   # rebrand base → Sabi-1
│   └── benchmark.py         # ADTC telemetry profiler
├── web/index.html       # single-page Sabi v1 interface (history, streaming, tools, RAM gauge)
├── data/corpus/         # sample SME documents (handbook, strategy, sales records)
├── config/sabi.yaml     # configuration
├── tests/               # 48 pytest tests (run without a model)
├── docs/                # PUBLISH_MODEL · VIDEO_SCRIPT · SUBMISSION_CHECKLIST
├── setup.sh  run.sh     # one-command setup and launch
└── REPORT.md            # the ADTC technical report
```

## Language scope — an honest, deliberate choice

Sabi focuses on **English** — the competition's primary evaluation language — and **Nigerian
Pidgin**, an African language spoken by roughly 100 million people (and the source of the name
"Sabi"). This is an accuracy-first decision: a small model spread across many languages produces
unreliable output, and accuracy is half of the total score. The ADTC FAQ confirms that
supporting a local language is a bonus rather than a requirement, and that English is the primary
language of evaluation. Full reasoning is in `REPORT.md` §6.

## Offline guarantee

After the one-time model download, you can disconnect entirely. The application makes **no**
network calls at inference — verified in the source (no runtime network imports, and the
interface loads no external resources). You can prove it by pulling the network cable or setting
`HF_HUB_OFFLINE=1`.

## Benchmarking on your hardware

Speed and efficiency are measured automatically on the target laptop. Run
`python -m sabi bench` on your machine to record real tokens/sec, peak RAM and the efficiency
score, then enter those figures in `REPORT.md` §9. These measured numbers — not estimates — are
what the ADTC profiler will also produce.

## License & attribution

Apache-2.0. **Sabi was created and trained by Godspower Uyanga.** It is built on open-source
components (Qwen2.5, bge-small, llama.cpp); see [`LICENSE`](LICENSE).
