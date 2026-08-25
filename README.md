<!-- markdownlint-disable MD033 MD041 -->
<div align="center">

# SABI — The Offline AI Coworker

**On-device AI for the hardware Africa actually has.**

SABI is a private, fully offline AI coworker. It runs a single quantized model
on a standard 8 GB laptop — no cloud, no API fees, no internet — staying under a
strict 7 GB memory budget. It chats, plans, writes code, and can **take real
action** (create files, run commands) through a Claude-Code-style agent, with
your approval. Use it from a polished terminal interface, a full-screen TUI, or
a ChatGPT-style web app.

[![Python](https://img.shields.io/badge/python-3.9%2B-1f8a8c)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-c8901f)](LICENSE)
[![Status](https://img.shields.io/badge/status-beta-16263d)](#)
[![ADTC 2026](https://img.shields.io/badge/ADTC-2026-1f8a8c)](#)

</div>

---

## Table of contents

- [What is SABI?](#what-is-sabi)
- [Quickstart](#quickstart)
- [Installation, in detail](#installation-in-detail)
  - [Optional extras explained](#optional-extras-explained)
- [Getting the model](#getting-the-model)
- [The three interfaces](#the-three-interfaces)
- [Command reference](#command-reference)
- [Configuration](#configuration)
- [How SABI works](#how-sabi-works)
- [Benchmarking & the 7 GB budget](#benchmarking--the-7-gb-budget)
- [Project structure](#project-structure)
- [Development](#development)
- [Hardware target (ADTC 2026)](#hardware-target-adtc-2026)
- [African localization](#african-localization)
- [FAQ](#faq)
- [License](#license)

---

## What is SABI?

SABI is an **offline AI coworker** built for constrained African hardware. It
runs fully on-device through a single quantized GGUF model and combines:

- **SABI THINK** — business analysis, planning, PRD/SOP generation, requirements
  and architecture design.
- **SABI CODE** — code generation, debugging, scaffolding and automation.
- **An agentic tool loop** — plan → act → verify, able to create folders/files
  and run commands, each gated by your permission.

It ships with **three ways to use it**: a simple terminal chat, a full-screen
opencode-style TUI, and a browser web app with chat history. Everything runs
locally; your data never leaves your machine.

> **Built for:** the ADTC 2026 Laptop LLM Challenge — private, low-cost,
> idea-to-execution AI that Africa can own, run, and trust.

### Highlights

- **100% offline** — no API keys, no internet on the critical path.
- **Acts, not just talks** — creates files and runs commands through tools.
- **Asks permission** — Allow once / Allow always → Confirm, like Claude Code.
- **Streams replies** — text appears as it is generated, so it feels fast.
- **Knows when to act vs. chat** — greetings and questions never touch your files.
- **Telemetry built in** — model size, tokens, speed and RAM vs the 7 GB budget.
- **Graceful** — starts and guides you even before the model is downloaded.

---

## Quickstart

No cloning, no source checkout — install the package like any CLI tool, then
run it from inside whatever project you want SABI to work on:

```bash
pip install "sabi-llm[full]"   # SABI + everything it needs to actually run
sabi download                  # pulls the model (~2 GB), once, into ~/.sabi/models/
cd ~/your-project              # cd into the codebase you want help with
sabi                           # start the offline AI coworker, scoped to this project
```

That's everything. The model downloads once into `~/.sabi/models/` — shared
across every project on the machine, not re-downloaded per project — and each
project gets its own `.sabi/` folder (created next to your code, like `.git/`)
for SABI's memory of that specific codebase. (Skip `sabi download` if you
like — the first `sabi run` offers to download it for you.)

Building from source instead (contributing, or the ADTC audit harness) still
works the usual way — see [Installation, in detail](#installation-in-detail).

**ADTC 2026 audit note.** For the challenge's audit harness, use
`./download_model.sh` instead of `sabi download` — same source, same
destination (`models/sabi-v1.Q4_K_M.gguf`), but it's the plain, dependency-free
entry point the [official submission template](https://github.com/Africa-Deep-Tech-Foundation/adtc-2026-submission-template)
expects, and its path matches `_runtime.model_path` in [`metadata.json`](metadata.json).

---

## Installation, in detail

SABI is a normal Python package. The steps below explain what each one does.

**1. Clone the repository.** The code is small; the large model is downloaded
separately (next section).

```bash
git clone https://github.com/godspoweruyanga/sabi-llm.git
cd sabi-llm
```

**2. Create a virtual environment.** This isolates SABI's dependencies from the
rest of your system.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# Windows PowerShell:  .venv\Scripts\Activate.ps1
```

**3. Install SABI.** `pip install -e .` installs the package in *editable* mode
and creates the `sabi` command. Add an extra in brackets to include optional
features:

```bash
pip install --upgrade pip
pip install -e ".[tui]"            # recommended: includes the full-screen UI
```

> If installing the inference backend (`llama-cpp-python`) fails, install build
> tools first — `sudo apt-get install -y build-essential cmake` on Ubuntu — then
> retry.

### Optional extras explained

Install any combination, e.g. `pip install -e ".[tui,serve]"`.

| Extra | Command | What it adds |
|-------|---------|--------------|
| *(none)* | `pip install -e .` | Core SABI + the simple terminal chat. Lightweight (rich, psutil, numpy, PyYAML). |
| `full` | `pip install -e ".[full]"` | Everything below in one shot — the one-line quickstart for an end-user laptop install. |
| `tui` | `pip install -e ".[tui]"` | The full-screen opencode-style interface (adds **textual**). **Recommended.** |
| `docs` | `pip install -e ".[docs]"` | Read PDFs, Word, Excel, PowerPoint, HTML & images (adds pypdf, python-docx, openpyxl, python-pptx, beautifulsoup4). |
| `serve` | `pip install -e ".[serve]"` | The browser web app with chat history (adds **flask**). |
| `inference` | `pip install -e ".[inference]"` | The GGUF model backend (**llama-cpp-python**) — needed to actually run the model. Usually pulled in by `requirements.txt`. |
| `hub` | `pip install -e ".[hub]"` | `huggingface_hub`, an optional fallback for model download (private repos / auth). Not required — direct download works without it. |
| `translate` | `pip install -e ".[translate]"` | sabi-yoruba-tts (African Alpha Bonus) — English<->Yoruba translation layer (adds **ctranslate2**, **transformers**). |
| `dev` | `pip install -e ".[dev]"` | Everything above plus test/lint tools (pytest, ruff). |

To install **everything**: `pip install -e ".[full,translate,dev]"`.

**Where things live.** `sabi download` puts the model at `~/.sabi/models/` and
uses that same home-anchored `~/.sabi/` for its own scratch workspace — so it
works from a real install with no writable repo directory in sight. The one
exception: if a `models/sabi-v1.Q4_K_M.gguf` already exists relative to where
`sabi` was installed from (a source checkout that ran `download_model.sh`),
SABI uses that file instead, so the ADTC audit flow above is unaffected.
Per-project memory (what SABI has learned about *this* codebase) always lives
in `.sabi/` inside the project you run `sabi` from, next to your code.

---

## Getting the model

The quantized model (~2 GB) is hosted on Hugging Face — too large for Git — so
it downloads on demand into `models/`. You don't need a Hugging Face account.

```bash
sabi download
```

This streams `qwen2.5-coder-3b-instruct-q4_k_m.gguf` from Qwen's official
[`Qwen/Qwen2.5-Coder-3B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF)
repo (verified 2.0 GB), saving it locally as `models/sabi-v1.Q4_K_M.gguf`, with a
progress bar, and creates the `models/` folder automatically. You can also just
start SABI — the first `sabi run` / `sabi chat` / `sabi tui` will offer to
download it for you.

> Our own `Doctorgp1/sabi-v1` Hugging Face repo currently still holds an older
> 7B build (4.68 GB, 7.07 GB peak RAM — over the 7 GB ceiling). We source
> directly from Qwen's repo for now; the maintainer note below covers
> re-quantizing and uploading our own 3B copy before Gate 2.

Verify it's ready (and see its size vs the 7 GB budget):

```bash
sabi doctor
```

Want to build your own quantized model from scratch, or use a different one? See
**[docs/MODEL.md](docs/MODEL.md)**.

> **Maintainers: building & uploading the 3B model.** SABI is configured for a
> **3B** GGUF (`sabi-v1.Q4_K_M.gguf`) to stay well under the 7 GB ceiling. To
> produce and publish it to the Hugging Face repo SABI downloads from:
>
> ```bash
> # 1) Build the 3B GGUF locally (needs build tools; see docs/MODEL.md)
> ./scripts/quantize_model.sh \
>     --hf Qwen/Qwen2.5-Coder-3B-Instruct \
>     --out sabi-v1.Q4_K_M.gguf --quant Q4_K_M
> #   -> models/sabi-v1.Q4_K_M.gguf  (~2 GB)
>
> # 2) Upload it to your Hugging Face repo (public)
> pip install "huggingface_hub[cli]"
> huggingface-cli login
> huggingface-cli upload Doctorgp1/sabi-v1 \
>     models/sabi-v1.Q4_K_M.gguf sabi-v1.Q4_K_M.gguf
>
> # 3) Verify the runtime footprint on the target laptop
> sabi doctor && sabi benchmark
> ```
>
> Until this 3B file is uploaded to our own repo, `sabi download` and
> `download_model.sh` source directly from Qwen's official
> `Qwen/Qwen2.5-Coder-3B-Instruct-GGUF` repo instead (see `config/default.yaml`
> → `hf_repo_id`/`hf_filename`) — this is a fully public, credential-free
> source, so it satisfies the ADTC audit requirements as-is. Uploading our own
> copy to `Doctorgp1/sabi-v1` is a nice-to-have before Gate 2, not a blocker.

---

## The three interfaces

SABI gives you three front-ends over the same engine — pick what suits you.

### 1. Full-screen TUI — `sabi tui` (or `sabi run`)

The richest experience: an always-visible input box with a **Send** button,
**streamed** replies, a live **Activity** feed ("creating folder…", "writing
file…"), and a **Session** sidebar showing tokens, context %, `$0.00` (offline),
plus your working directory and git branch. Needs the `tui` extra.

```bash
pip install -e ".[tui]"
sabi tui            # or just `sabi run` — it uses the TUI when installed
sabi run --simple   # force the plain line-by-line chat instead
```

### 2. Simple terminal chat — `sabi chat`

A lightweight line-by-line REPL — no extra dependencies. This is also where the
**step-by-step permission prompts** live (Allow once / Allow always / Confirm)
for users who want to approve each action individually.

```bash
sabi chat
```

### 3. Web app — `sabi serve`

A professional ChatGPT/Claude-style browser interface with a conversation
**history sidebar**, markdown + code rendering, a mode selector, **streamed
replies** (text appears as it generates), and a **file upload** button (📎) for
PDFs, Word, Excel, PowerPoint, CSV, HTML and images. Ask SABI to summarize or
analyze any uploaded file. Everything stays on your machine. Needs the `serve`
extra (and `docs` for file reading).

```bash
pip install -e ".[serve,docs]"
sabi serve                       # opens http://127.0.0.1:8765
sabi serve --port 9000 --no-browser
```

### Reading documents (any format)

With the `docs` extra, SABI reads almost any file — PDF, Word (`.docx`), Excel
(`.xlsx`), PowerPoint (`.pptx`), CSV/TSV, HTML, JSON, images (OCR if available),
and any text/code file — then summarizes, extracts, or acts on it. In the
terminal: *"summarize ~/Desktop/report.pdf"* or *"open sales.xlsx and give me the
top region."* In the web UI, click 📎 to upload.

---

## Command reference

Run `sabi --help` or `sabi <command> --help` any time. Add `--json` to `ask`,
`think`, `code`, `agent`, `benchmark` and `profile` for machine-readable output.

| Command | What it does |
|---------|--------------|
| `sabi` | Launch the runtime — opens the TUI if installed, otherwise the simple chat. |
| `sabi run` | Start the agentic coworker in the current folder (TUI by default). `--simple` forces the plain chat; `--yes` auto-approves actions; `--cwd PATH` sets the working directory. |
| `sabi tui` | Open the full-screen opencode-style interface explicitly (needs `tui`). |
| `sabi chat` | Open the simple terminal chat with step-by-step permission prompts. |
| `sabi serve` | Launch the web UI (needs `serve`). `--host`, `--port`, `--no-browser`. |
| `sabi ask "<text>"` | One-off question, auto-routed (THINK / CODE / CHAT), prints the answer and exits. |
| `sabi think "<text>"` | Force the planning/analysis engine (PRDs, SOPs, architecture). |
| `sabi code "<text>"` | Force the code engine (generates code as text). |
| `sabi agent "<text>"` | Run the acting agent once (can create files / run commands). `--yes`, `--cwd`. |
| `sabi download` | Download the model from Hugging Face into `models/`. `--repo`, `--file`, `--force`. |
| `sabi doctor` | Diagnose the environment: Python, dependencies, **model size vs the 7 GB budget**, RAM, workspace. |
| `sabi benchmark` | Run the local benchmark — accuracy, tokens/sec, peak RAM, thermals. `--limit N`. |
| `sabi profile` | Show live RAM / CPU / temperature telemetry. |
| `sabi workspace [info\|reset]` | Inspect or reset the `sabi_workspace/` folder. |
| `sabi version` | Print the version. |

### When does SABI take action vs. just chat?

SABI only touches your filesystem when your message clearly asks for an action —
phrases like *"create a folder"*, *"write a file"*, *"scan the project"*,
*"run the tests"*, *"scaffold a project"*. Greetings, questions, explanations and
*"write a function…"* (code shown as text) are answered conversationally and
**never** create files. In the simple `sabi chat`, every action also asks your
permission first (Allow once / Allow always → Confirm).

---

## Configuration

Defaults live in `config/default.yaml`. Every key can be overridden by an
environment variable named `SABI_<KEY>` (uppercase), or copy `.env.example` to
`.env`.

```bash
SABI_MODEL_PATH=models/sabi-v1.Q4_K_M.gguf
SABI_HF_REPO_ID=Qwen/Qwen2.5-Coder-3B-Instruct-GGUF
SABI_HF_FILENAME=qwen2.5-coder-3b-instruct-q4_k_m.gguf
SABI_TEMPERATURE=0.4
SABI_CONTEXT_LENGTH=4096
SABI_N_THREADS=0        # 0 = auto (use all physical cores)
SABI_N_GPU_LAYERS=0     # CPU-only target
SABI_LANGUAGE=en        # en | yo | ha | ig
```

---

## How SABI works

```text
Your message
   -> Action?  ── no ──>  Conversation (streamed reply, no file access)
        │ yes
        v
   Agent loop:  plan -> propose tool -> ask permission -> run -> observe -> repeat
        │
        v
   Tools: create_dir · write_file · read_file · list_dir · run_shell
```

| Layer | Responsibility |
|-------|----------------|
| Presentation | TUI, web UI, simple chat (`sabi/ui/`, `sabi/server.py`) |
| Application | Intent routing + agent controller (`router.py`, `agent.py`) |
| Reasoning | THINK / CODE engines (`engines/`) |
| Data | Local vector store + JSON memory + chat history (`rag/`, `memory/`, `conversations.py`) |
| Infrastructure | Quantized GGUF runtime via llama.cpp (`model.py`) |

---

## Benchmarking & the 7 GB budget

SABI is tuned for the ADTC 2026 scoring model (50% accuracy, 30% speed, 20%
efficiency, −10 thermal penalty).

```bash
sabi doctor                       # model size on disk + estimated RAM vs 7 GB ceiling
sabi benchmark                    # accuracy, tokens/sec, peak RAM, thermals
python scripts/run_benchmark.py   # writes benchmarks/report.json + report.md
```

`sabi doctor` reports, for example:

```
Model size on disk: 4.69 GB  (ADTC RAM budget: 7.0 GB)
Est. runtime RAM:  ~5.99 GB of 7.0 GB budget (86% of ceiling)
Headroom vs budget: 1.01 GB free under the 7.0 GB ceiling
```

Targets: peak RAM **< 6.5 GB**, startup **< 5 s**, **10–20 tok/s**, core temp
**< 85 °C**, **0 crashes**.

---

## Project structure

```text
sabi-llm/
├── metadata.json  download_model.sh   # ADTC 2026 submission template requirements
├── README.md  REPORT.md  LICENSE  CHANGELOG.md  CONTRIBUTING.md
├── pyproject.toml  requirements.txt  Makefile
├── config/        default.yaml
├── prompts/       system / think / code / router / agent templates
├── scripts/       download_model.py  quantize_model.sh  run_benchmark.py
├── docs/          MODEL.md  (how to build / pick a model)
├── models/        the GGUF model (downloaded from Hugging Face)
├── benchmarks/    prompts.jsonl + generated reports
├── tests/         pytest suite
└── sabi/
    ├── cli.py  runner.py  runtime.py  router.py  agent.py  permissions.py
    ├── model.py  downloader.py  config.py  conversations.py
    ├── project_scanner.py  workspace_manager.py  profiler.py  doctor.py  benchmark.py
    ├── server.py            (web UI backend)
    ├── engines/  (think.py, code.py)
    ├── tools/    (file, shell, workspace)
    ├── memory/   (json store)
    ├── rag/      (embeddings, vector_store, retriever)
    └── ui/       (console, chat, tui, web/)
```

---

## Development

```bash
pip install -e ".[dev]"     # everything + test/lint tools
pytest                      # run the test suite
ruff check sabi tests       # lint
make help                   # see all make targets
```

---

## Hardware target (ADTC 2026)

| Spec | Target |
|------|--------|
| CPU | Intel Core i5 (10–12th gen) / AMD Ryzen 5 3000–5000 (x86-64) |
| Memory | 8 GB DDR4 — peak RSS kept under the 7 GB ceiling |
| Graphics | Intel UHD / Iris Xe or AMD Radeon integrated — no discrete GPU |
| Storage | 256 GB SSD |
| OS | Ubuntu 22.04 LTS |

---

## African localization

SABI's base model (Qwen2.5-Coder-3B-Instruct) was never trained for Yoruba —
live testing showed its raw Yoruba output was degenerate, not fluent. Instead
of a prompt hack, SABI ships **sabi-yoruba-tts**: a dedicated English<->Yoruba
translation layer (`sabi/translate.py`, NLLB-200-distilled-600M as int8
CTranslate2) wrapped around the same coder model. A Yoruba turn is translated
to English, routed and answered normally, then the reply is translated back
to Yoruba — with fenced code blocks always left untouched, so code explained
in Yoruba is still real, runnable code. See `scripts/download_yoruba_model.py`
to fetch/convert the model, and REPORT.md §11 for verified examples and the
CC-BY-NC-4.0 license disclosure (non-commercial — unlike the rest of SABI's
MIT license). SABI claims the ADTC African Alpha Bonus (+15%) —
`metadata.json` (`african_alpha_claim: true`, `language_scope: ["en", "yo"]`).

`SABI_LANGUAGE=ha|ig` (Hausa, Igbo) remain reserved config values without an
implementation yet — a candidate for post-Gate-1 work.

---

## FAQ

**Do I have to download the model manually?** No. `sabi download` does it, and
`sabi run` offers to do it on first launch.

**Does it need internet?** Only once, to download the model. After that it is
fully offline.

**Will it edit my files without asking?** Inside the current project it acts
freely (read/write/create/run). When it needs to touch a folder **outside** the
project, or run a shell command, it asks first — **Allow once / Allow always /
Reject** — and "Allow always" sticks for the session (opencode-style). The simple
`sabi chat` asks for *every* action. SABI can write code in any language and
build whole projects; it will never claim it "can't access files" — it can.

**Why is the first reply a little slow?** The model loads into RAM on first use.
Replies then stream as they generate.

---

## License

Released under the [MIT License](LICENSE).

## Authors

- **Godspower Uyanga** — Lead Author · Senior Data Scientist / AI Engineer
- **Oreoluwa Akinwe** — Research Analyst

<div align="center">

*AI that Africa can own, run, and trust.*

</div>
