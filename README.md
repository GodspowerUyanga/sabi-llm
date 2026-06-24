<!-- markdownlint-disable MD033 MD041 -->
<div align="center">

# SABI — The Offline AI Coworker

**On-device AI for the hardware Africa actually has.**

Turn ideas into working software and structured business output — locally,
privately, with no cloud dependency, under a strict 7 GB memory ceiling.

[![Python](https://img.shields.io/badge/python-3.9%2B-1f8a8c)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-c8901f)](LICENSE)
[![Status](https://img.shields.io/badge/status-beta-16263d)](#)
[![ADTC 2026](https://img.shields.io/badge/ADTC-2026-1f8a8c)](#)

</div>

---

## What is SABI?

SABI is an **offline AI coworker** built for constrained African hardware. It
runs fully on-device through a single quantized GGUF model and pairs two
reasoning engines:

- **SABI THINK** — business analysis, planning, PRD/SOP generation, requirements
  and architecture design.
- **SABI CODE** — code generation, debugging, scaffolding, automation scripts
  and file generation.

An intelligent **agent loop** (plan → execute → verify) ties them together, with
local **memory**, a fully offline **RAG** layer, a sandboxed **tool layer**, and
automatic **project recognition**. SABI is engineered and benchmarked for the
ADTC 2026 *Standard Laptop*: an 8 GB machine with integrated graphics and no
discrete GPU.

> **Domain:** Corporate / Enterprise + Autonomous AI Agents.
> **Built for:** the ADTC 2026 Laptop LLM Challenge.

---

## Features

- **100% offline** — no API keys, no internet on the critical path.
- **Acts, not just talks** — SABI can actually create folders, write files and
  run commands through an agentic tool loop (like Claude Code / opencode), not
  just print code for you to copy.
- **Permission flow** — before any action it shows exactly what it will do and
  asks **Allow once / Allow always / Deny**; choosing *Allow always* then asks
  **Confirm / Cancel** before running.
- **Live status** — "SABI is thinking…", "SABI wants to make changes…", and the
  concrete action are shown as it works.
- **Single quantized model** — one GGUF (e.g. Qwen2.5-Coder 7B Instruct, Q4/Q5)
  prompted to behave as THINK, CODE or the agent, keeping RAM low.
- **Intent router** — fast keyword heuristic with optional model arbitration.
- **Local memory** — lightweight JSON history of turns and tasks.
- **Offline RAG** — pure-Python hashing embedder + JSON vector store, no network.
- **Project recognition** — detects Git, Python, Node.js, Next.js, package
  managers and virtual environments.
- **Telemetry built in** — `benchmark`, `profile` and `doctor` commands measure
  accuracy, tokens/sec, peak RAM and thermals.
- **Graceful degradation** — every subsystem starts and runs even before the
  model is downloaded, with clear guidance instead of crashes.

---

## Requirements

| Requirement | Recommended |
|-------------|-------------|
| OS | Ubuntu 22.04 LTS (Linux), macOS, or Windows (WSL2) |
| Python | 3.9 or newer |
| RAM | 8 GB (SABI keeps peak usage under 7 GB) |
| Disk | ~5–6 GB free for the quantized model |
| Build tools (for `llama-cpp-python`) | `build-essential`, `cmake` on Linux |
| Git | to clone the repository |

On Ubuntu, install the build tools once:

```bash
sudo apt-get update && sudo apt-get install -y build-essential cmake git python3-venv
```

---

## Quickstart

The model is **large**, so it is hosted on **Hugging Face** rather than in this
repository. The steps below clone the code, create a virtual environment,
install dependencies, download the model, and start SABI.

### TL;DR (copy-paste)

```bash
git clone https://github.com/godspoweruyanga/sabi-llm.git
cd sabi-llm
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
sabi download        # pulls sabi-v1.Q4_K_M.gguf (~4.7 GB) into ./models/
sabi run             # start the offline AI coworker
```

That's everything — the model downloads itself from Hugging Face into a new
`models/` folder, so there's nothing to fetch by hand. (If you skip
`sabi download`, the first `sabi run` will offer to download it for you.)

The step-by-step version follows.

### 1. Clone the repository

```bash
git clone https://github.com/godspoweruyanga/sabi-llm.git
cd sabi-llm
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
# .venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt   # core + llama-cpp-python + huggingface_hub
pip install -e .                  # installs the `sabi` command
```

> If `llama-cpp-python` fails to build, make sure `build-essential` and `cmake`
> are installed (see Requirements), then retry.

### 4. Download the model from Hugging Face

The model is hosted on Hugging Face (it's too large for Git), so it downloads
automatically. You don't need a Hugging Face account or any extra package — just
run:

```bash
sabi download
```

This streams `sabi-v1.Q4_K_M.gguf` (~4.7 GB) from
[`Doctorgp1/sabi-v1`](https://huggingface.co/Doctorgp1/sabi-v1) directly into a
new `models/` folder, with a progress bar.

You can also skip this step entirely: the first time you run `sabi run` or
`sabi chat`, if no model is found SABI offers to download it for you:

```text
No model found locally.
  Download it now from Hugging Face (~4.7 GB)?  [Y/n] > y
```

Advanced: override the source if you ever move the file:

```bash
sabi download --repo Doctorgp1/sabi-v1 --file sabi-v1.Q4_K_M.gguf
# or build your own from Qwen2.5-Coder — see docs/MODEL.md
```

The model is saved to `models/sabi-v1.Q4_K_M.gguf`.

### 5. Verify your environment

```bash
sabi doctor
```

You should see green checks for Python, dependencies, the model file, available
RAM and a writable workspace.

### 6. Start SABI

```bash
sabi run        # interactive runtime in the current project
# or
sabi chat       # the chat UI
```

That's it — you now have an offline AI coworker running on your laptop.

---

## Using SABI

SABI works as a CLI and as an interactive REPL. The MVP centres on
`run`, `ask`, `think` and `code`.

```bash
sabi                       # launch the runtime (same as `sabi run`)
sabi run                   # start the agentic runtime in the current project
sabi chat                  # launch the terminal chat interface
sabi serve                 # launch the web UI (browser, with chat history)
sabi ask "summarise this repo's structure"
sabi think "write a PRD for an offline invoicing tool"
sabi code  "write a Python function that validates an email address"
sabi agent "build a CLI todo app in Python"   # plan -> execute -> verify
sabi benchmark             # run the local benchmark
sabi profile               # RAM / CPU / thermal telemetry
sabi doctor                # diagnose the environment
sabi workspace info        # inspect the workspace
sabi workspace reset       # clear generated files (keeps memory)
sabi version
```

Add `--json` to `ask`, `think`, `code`, `agent`, `benchmark` and `profile` for
machine-readable output.

### Web UI (`sabi serve`)

For longer "thinking" work — planning, analysis, writing — SABI ships a
professional browser interface (ChatGPT/Claude-style) with persistent chat
history:

```bash
pip install "sabi-llm[serve]"   # one-time: installs Flask
sabi serve                      # opens http://127.0.0.1:8765 in your browser
sabi serve --port 9000 --no-browser
```

Features: a conversation history sidebar (new / switch / delete), a mode
selector (Auto / Think / Code / Agent), markdown + code rendering, and a live
"SABI is thinking…" indicator. Everything runs locally and your chats stay on
your machine. The web UI is conversational by default; switch to *Agent* mode to
let it create files / run commands (the browser auto-approves these, so for
step-by-step approval use the terminal `sabi run` instead).

### Taking real actions (agentic mode)

`sabi run` and `sabi chat` run every message through the agent, so SABI can
*do* things, not just describe them:

```text
you > create a folder called appfolder on my desktop

· SABI is thinking…
· SABI wants to make changes…

SABI wants to create a directory:  /home/you/Desktop/appfolder
  [1] Allow once     [2] Allow always     [3] Deny
choose 1/2/3 > 2
  Confirm — create a directory: /home/you/Desktop/appfolder?  [y] Yes  [n] Cancel > y
  done

SABI: Done — I created 'appfolder' on your Desktop.
```

The agent acts in the directory you launched it from (override with `--cwd`),
and `~` means your home directory. Trust a tool for the session with
*Allow always*; review what you've trusted with `/trust`. Use `--yes` to
auto-approve everything (handy for scripting, but it skips the prompts):

```bash
sabi run --yes
sabi agent "scaffold a python package called widgets" --cwd ~/projects
```

### Inside the chat UI

```text
you > /think design an offline knowledge base for a clinic
you > /code  parse a CSV and print row counts
you > /agent scaffold a FastAPI service with one health endpoint
you > /project      # show detected project context
you > /memory       # show memory stats
you > /help
you > /exit
```

Anything without a slash is auto-routed to THINK, CODE or CHAT.

---

## How it works

```text
User Input
   -> Intent Router            (heuristic + optional model arbitration)
        -> Planner (THINK)      (business reasoning, plans, PRDs/SOPs)
             -> Executor (CODE) (code generation, debugging, files)
                  -> Memory + Tool + RAG layers
                       -> Output Engine
```

Runtime initialisation follows a fixed, deterministic order so startup stays
fast: **load model → load prompts → init memory → init tools → start router →
activate THINK + CODE → start runtime.**

| Layer | Responsibility |
|-------|----------------|
| Presentation | CLI and chat UI (`sabi/cli.py`, `sabi/ui/`) |
| Application | Intent router + agent controller (`router.py`, `agent.py`) |
| Reasoning | THINK / CODE orchestration (`engines/`) |
| Data | Local vector store + JSON memory (`rag/`, `memory/`) |
| Infrastructure | Quantized GGUF runtime via llama.cpp (`model.py`) |

---

## Configuration

Defaults live in `config/default.yaml`. Every key can be overridden by an
environment variable named `SABI_<KEY>` (uppercase). Copy `.env.example` to
`.env` to customise.

```bash
SABI_MODEL_PATH=models/sabi-v1.Q4_K_M.gguf
SABI_HF_REPO_ID=Doctorgp1/sabi-v1
SABI_HF_FILENAME=sabi-v1.Q4_K_M.gguf
SABI_TEMPERATURE=0.4
SABI_CONTEXT_LENGTH=4096
SABI_N_THREADS=0        # 0 = auto
SABI_N_GPU_LAYERS=0     # CPU-only target
SABI_LANGUAGE=en        # en | yo | ha | ig
```

---

## Benchmarking

SABI ships with a local profiler aligned to the ADTC 2026 scoring model
(50% accuracy, 30% speed, 20% efficiency, −10 thermal penalty).

```bash
sabi benchmark                    # quick summary
python scripts/run_benchmark.py   # writes benchmarks/report.json + report.md
```

Performance targets: peak RAM **< 6.5 GB**, startup **< 5 s**, **10–20 tok/s**,
core temperature **< 85 °C**, **0 crashes**.

---

## Project structure

```text
sabi-llm/
├── README.md  LICENSE  CHANGELOG.md  CONTRIBUTING.md
├── pyproject.toml  requirements.txt  requirements-dev.txt  Makefile
├── config/        default.yaml
├── prompts/       system / think / code / router templates
├── scripts/       download_model.py  run_benchmark.py
├── models/        GGUF model (downloaded from Hugging Face)
├── benchmarks/    prompts.jsonl + generated reports
├── tests/         pytest suite
├── sabi_workspace/  generated projects + .sabi runtime state
└── sabi/
    ├── cli.py  runner.py  runtime.py  router.py  agent.py
    ├── model.py  config.py  project_scanner.py  workspace_manager.py
    ├── profiler.py  doctor.py  benchmark.py
    ├── engines/  (think.py, code.py)
    ├── tools/    (file, shell, workspace, base, registry)
    ├── memory/   (json store)
    ├── rag/      (embeddings, vector_store, retriever)
    └── ui/       (console, chat)
```

---

## Development

```bash
make install-dev     # venv + dev dependencies + editable install
make test            # run the test suite (pytest)
make lint            # ruff
make benchmark       # write a benchmark report
```

Or directly:

```bash
pip install -e ".[dev]"
pytest
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

Planned support for **Yoruba, Hausa, Igbo** and other African languages
(`SABI_LANGUAGE=yo|ha|ig`). Meaningful functionality in at least one African
language is eligible for the ADTC African Alpha Bonus (+15%).

---

## License

Released under the [MIT License](LICENSE).

## Authors

- **Godspower Uyanga** — Lead Author · Senior Data Scientist / AI Engineer
- **Oreoluwa Akinwe** — Research Analyst

<div align="center">

*AI that Africa can own, run, and trust.*

</div>
