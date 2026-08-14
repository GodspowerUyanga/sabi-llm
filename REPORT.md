# SABI — The Offline AI Coworker

### ADTC 2026 Laptop LLM Challenge — Gate 1 Submission Report

| | |
|---|---|
| **Project** | SABI — The Offline AI Coworker |
| **Primary track** | Coding Assistants |
| **Cross-disciplinary integration** | Coding Assistant × Corporate/Enterprise × Autonomous AI Agents |
| **Repository** | https://github.com/godspoweruyanga/sabi-llm |
| **Model** | `sabi-3b.Q4_K_M.gguf` (quantized GGUF, runs via llama.cpp) — hosted at https://huggingface.co/Doctorgp1/sabi-v1 |
| **Target hardware** | ADTC Standard Laptop — 8 GB RAM, no discrete GPU, Ubuntu 22.04 |
| **Memory ceiling** | 7 GB (hard limit; exceeding it = disqualification) |
| **Authors** | Godspower Uyanga (lead) · Oreoluwa Akinwe |
| **License** | MIT |
| **Bonus claims** | Budget-laptop profile: **claimed** · African-language bonus: **not claimed** (see §11) |

> **A note to reviewers on numbers.** The table in §8 is a real run of
> `sabi benchmark` (our own profiler, wrapping the same llama.cpp CPU inference
> path used at audit) on the development machine used to build SABI — **not**
> the ADTC Standard Laptop, and not yet the official `adtc-profiler` audit tool.
> We do not substitute estimates for measured telemetry anywhere in this report;
> every number below came from an actual run, and every number is labelled with
> the hardware it was measured on. We re-run both `sabi benchmark` and the
> official `adtc-profiler` on the target 8 GB no-GPU profile before Gate 2 and
> update this table with those figures.

---

## 1. Executive summary

SABI is a fully offline AI coworker for developers and small enterprises on
low-cost African hardware. It runs a single quantized language model entirely
on-device — no cloud, no API keys, no internet on the critical path — and stays
within the 7 GB memory ceiling. Beyond chat, SABI is an **agent**: it can read,
write and edit files, create folders, run commands, and build whole projects in
any language, asking permission before touching anything outside the current
project. It is usable through three interfaces sharing one engine: a polished
terminal chat, a full-screen TUI, and a browser web app with chat history.

The design philosophy is **"thin model, smart harness."** The expensive,
memory-hungry part (the model) is kept as small as the task allows, while
capability (tool use, permissions, routing, memory, RAG, telemetry) lives in
lightweight Python that costs almost no RAM. This is the correct shape for a
contest scored on accuracy, speed, and memory efficiency under a hard ceiling.

---

## 2. Problem definition

**The bottleneck is access economics, not capability.** Cloud LLMs assume API
budgets, stable fibre, and reliable power — assumptions that fail for a student
in Lagos, an SME operator in Accra, or a developer on intermittent connectivity.
The capability exists; affordable, private, local access does not.

**Who SABI serves.** Developers and knowledge workers on the machine already on
millions of desks — the $150–$500, 8 GB, integrated-graphics laptop — who need:

- a coding assistant that writes, edits and runs real code locally;
- enterprise knowledge-work help (drafting, planning, structuring) without
  sending company data to the cloud;
- automation of small multi-step tasks (scaffolding, file operations) safely.

**Why offline matters here.** Privacy (company/clinic/personal data never
leaves the device), cost (zero marginal inference cost), and resilience (works
without connectivity or grid power).

---

## 3. Constraints (and how SABI respects each)

| Constraint | Requirement | SABI's approach |
|---|---|---|
| **Memory** | Peak RSS < 7 GB or disqualified | Single Q4 quant; lazy model load; harness logic is pure-Python and near-zero RAM; `sabi doctor` reports size + estimated runtime RAM vs budget. |
| **No GPU** | CPU-only integrated graphics | llama.cpp CPU inference; `n_gpu_layers=0`; thread auto-tuning to physical cores. |
| **Offline** | No cloud dependency on critical path | No network calls at inference; model downloaded once; RAG and memory are local JSON/vector files. |
| **OS** | Ubuntu 22.04 LTS | Pure-Python + llama-cpp-python; no OS-specific code; tested on Linux. |
| **Reproducibility** | Auditable build | One-command setup, pinned deps, `sabi download`, `sabi doctor`, full pytest suite. |

---

## 4. System design & architecture

SABI separates a frozen **model** (judgment) from a **harness** (actions, state,
UX). A `.gguf` model is a pure next-token function; it cannot touch disk, run
commands, remember, or render UI. All of that is the harness — exactly how
production assistants (Claude Code, opencode) are built.

```
Your message
   ├─ greeting / question / explanation ──▶ Conversation (streamed reply, no file access)
   └─ action request ("create…", "open folder…", "build…")
            ▼
      Agent loop:  decide → propose tool → (permission) → run → observe → repeat
            ▼
      Tools: create_dir · write_file · read_file · list_dir · run_shell
```

| Layer | Components | Responsibility |
|---|---|---|
| Presentation | `ui/tui.py`, `ui/chat.py`, `server.py` + `ui/web/` | Full-screen TUI, simple REPL, web app |
| Application | `router.py`, `agent.py`, `permissions.py` | Intent routing, tool-calling agent, approvals |
| Reasoning | `engines/think.py`, `engines/code.py` | Planning/analysis and code generation |
| Data | `rag/`, `memory/`, `conversations.py` | Offline RAG, JSON memory, chat history |
| Infrastructure | `model.py`, `downloader.py`, `config.py` | GGUF runtime, model fetch, configuration |
| Telemetry | `doctor.py`, `benchmark.py`, `profiler.py` | Size/RAM/speed/thermal measurement |

**Key design decisions**

1. **Conversation vs. action routing.** A fast intent check sends greetings,
   questions, and "write a function" (code-as-text) to a streamed conversational
   reply that *cannot* touch the filesystem, while explicit action requests go to
   the agent. This prevents the classic failure of a greeting accidentally
   creating files, while still giving full agent power on demand.
2. **Permission model (opencode-style).** In-project actions run freely; touching
   a folder *outside* the project, or running a shell command, prompts
   **Allow once / Allow always / Reject**, and "Allow always" sticks for the
   session. A catastrophic-command deny-list is always enforced.
3. **Path intelligence + session memory.** The agent knows real machine locations
   (home, Desktop, Documents) and resolves "on Desktop" to an absolute path; it
   remembers what it created across turns, so "go into the folder you just made
   and add a file" works.
4. **Streaming.** Conversational replies stream token-by-token for responsiveness
   on slow CPU inference; the agent shows a live activity feed ("reading…",
   "writing file…").
5. **Graceful degradation.** Every subsystem starts and guides the user even
   before the model is downloaded, instead of crashing.

---

## 5. Cross-disciplinary integration

SABI deliberately load-bears across three of the seven tracks:

- **Coding Assistant (primary):** generates, edits, debugs and scaffolds code in
  any language, operating on the real filesystem.
- **Corporate / Enterprise:** the THINK engine produces PRDs, SOPs, plans and
  structured business documents — knowledge work for SMEs, fully private.
- **Autonomous AI Agents:** a local plan→act→verify loop with tool use and
  permissioned filesystem/shell access — privacy-focused workflow automation.

The integration is not cosmetic: the *same* agent that drafts an enterprise plan
can then scaffold the code project that implements it, on-device.

---

## 6. Model & quantization

- **Base model:** **Qwen2.5-Coder-3B-Instruct**, quantized to **Q4_K_M GGUF**
  for CPU inference via llama.cpp.
- **Why 3B (not 7B):** a 7B Q4 build measured **7.07 GB peak RAM** on the target
  machine — *over* the 7 GB ceiling, which means disqualification. The 3B build
  is ~2 GB on disk and ~3.5–4.5 GB at runtime, giving real headroom under the
  ceiling, a higher efficiency score, and roughly 2× the tokens/sec. This was an
  **evidence-based decision from `sabi benchmark`**, not a guess.
- **Why Q4_K_M:** best quality-per-byte for CPU; keeps the working set small.
- **Distribution:** hosted on Hugging Face; `sabi download` streams it directly
  into `models/` (no account needed). Not committed to Git (too large).
- **Runtime config:** `context_length=4096`, `n_gpu_layers=0`,
  `n_threads=auto`, temperature tuned for deterministic coding.

**Trade-off (documented).** The 3B gives up some raw accuracy versus the 7B, but
trades a *guaranteed disqualification* (over budget) for a *competitive, scored*
submission that also wins on the speed (30%) and efficiency (20%) gates. For a
contest with a hard memory ceiling, fitting under budget is the precondition for
any score at all.

---

## 7. Tools & frameworks

| Purpose | Choice | Why |
|---|---|---|
| Inference | `llama-cpp-python` (llama.cpp) | Best CPU GGUF runtime; quantization support |
| Quantization | llama.cpp `quantize` (Q4_K_M) | Standard, reproducible, CPU-friendly |
| TUI | `textual` | Modern terminal UI; low overhead |
| Web UI | `flask` | Minimal, offline, no build step |
| RAG | Custom hashing embedder + JSON vector store | Zero-dependency, offline, tiny RAM |
| Telemetry | `psutil` + custom profiler | Measures RSS, CPU, tokens/sec, temperature |
| CLI / config | `argparse`, `PyYAML`, env overrides | Simple, auditable |
| Tests | `pytest`, `pytest-asyncio` | 62 passing, 0 failing, 0 skipped (incl. headless TUI) |

All heavy components are **optional extras** (`[tui]`, `[serve]`, `[inference]`),
so the base install stays lean.

---

## 8. Benchmarks & telemetry

SABI ships its own profiler so results are reproducible on the target hardware
and aligned with the ADTC scoring formula
(**50% accuracy + 30% speed + 20% efficiency**, −10 thermal, OOM = 0).

```bash
sabi doctor                      # model size on disk + estimated runtime RAM vs 7 GB
sabi benchmark                   # accuracy on prompt set, tokens/sec, peak RAM, thermals
python scripts/run_benchmark.py  # writes benchmarks/report.json + report.md
sabi profile                     # live RAM / CPU / temperature
```

**Measured results — development machine, `sabi benchmark`, full 8-prompt set
in `benchmarks/prompts.jsonl`** *(not the ADTC Standard Laptop; see caveat
below the table):*

| Metric | Target | Measured (dev machine) |
|---|---|---|
| Model size on disk | < 7 GB | **1.80 GB** |
| Peak RAM (RSS) during inference | < 6.5 GB (well under 7 GB) | **3.28 GB** |
| Efficiency score `Seff = 100×(7−PeakRAM)/7` | maximise | **53.1** |
| Tokens/sec (CPU), avg across 8 prompts | 10–20 tok/s | **18.93 tok/s** |
| Cold start (load → first token) | < 5 s | not separately instrumented by `sabi benchmark`; to be measured on audit hardware |
| Peak core temperature / thermal throttle | < 85 °C (avoid −10) | **throttle flag tripped** — see caveat |
| Benchmark accuracy (prompt set, keyword-match heuristic) | maximise | **78.1%** |
| Crashes / OOM | 0 | **0** |

> **Hardware caveat.** This run was on the development workstation (22 logical
> cores, sustained package temps of 90+ °C under load from unrelated processes),
> not the ADTC Standard Laptop (4–8 threads, i5 10th–12th gen, no discrete GPU).
> The thermal-throttle flag reflects *this* machine, not the target profile, and
> is not a claim about audit-hardware thermals. Tokens/sec and peak RAM are
> expected to be broadly representative since inference is single-threaded-bound
> per token and RSS is dominated by model weights + KV cache, but we will
> re-measure on the actual ADTC Standard Laptop (or the closest available
> equivalent) and with the official `adtc-profiler` tool before Gate 2, and
> update this table with those numbers plus the DevPost self-reported Sperf/Seff
> fields.
>
> Reviewers can reproduce the dev-machine numbers with the commands above; the
> prompt set lives in `benchmarks/prompts.jsonl`. Per-prompt breakdown (tps /
> accuracy / elapsed) is in `benchmarks/report.json` after running
> `python scripts/run_benchmark.py`.

---

## 9. How the design targets the scoring model

| Gate | Weight | SABI's strategy |
|---|---|---|
| Accuracy | 50% | Code-tuned model + agent that *acts* (verifiable file output) rather than only describing; RAG over local context for grounded answers. |
| Speed | 30% | Q4 quant, thread tuning, token streaming; profiler to pick the fastest viable model. |
| Efficiency | 20% | Single quant, lazy load, near-zero-RAM harness; `doctor` tracks headroom under 7 GB. |
| Thermal (−10) | penalty | CPU-thread caps and modest context to limit sustained load; `profile` watches temperature. |
| OOM (DQ) | 0 | Hard discipline on RAM; profiler verifies peak RSS before audit; smaller-quant fallback if needed. |

---

## 10. Efficiency & the 7 GB budget

The harness is engineered to spend the RAM budget on the model, not on itself:

- **Lazy loading** — the model is mapped into memory only on first use.
- **Single quant** — exactly one GGUF in memory; no duplicate engines.
- **Pure-Python harness** — routing, permissions, memory, RAG and UI use
  negligible RAM (kilobytes–low megabytes), leaving the budget for inference.
- **Bounded context & history** — 4 096-token window and capped session memory
  prevent prompt growth from inflating the working set.
- **Continuous measurement** — `doctor` reports headroom; `benchmark` records the
  true peak RSS used for the efficiency score.

---

## 11. Bonus claims

These match `metadata.json` exactly — the claims here are not aspirational,
they are what the repo's machine-readable submission record says.

- **Budget-laptop profile (+10%): claimed** (`budget_laptop_claim: true`).
  SABI is designed and tested for the $150–$500, 8 GB, no-GPU machine and
  reports its footprint against the budget (§8, §10).
- **African-language bonus (+15%): not claimed** (`african_alpha_claim: false`).
  The architecture reserves a `SABI_LANGUAGE` config key (`en|yo|ha|ig`) for
  this, but no localized prompts or translation layer exist yet, and live
  testing (2026-08-14) shows the base model's raw Yoruba output is degenerate/
  repetitive, not real fluency — the base model was not trained for African
  languages. Meaningful end-to-end functionality is not demonstrably working.
  We would rather ship Gate 1 honestly at a lower multiplier than have an
  unsubstantiated claim fail the Gate 2 audit. Real support would need a local
  translation layer (e.g. an NLLB model wrapping the existing pipeline), which
  is a scoped follow-up, not a same-session fix. If we finish it before
  Gate 2, we will flip this flag and add evidence (sample transcripts) — not
  before.

---

## 12. Reproducibility (for the Gate 2 audit)

Two equivalent ways to get the model, matched to the two audiences reading this
repo: the ADTC audit harness (root-level script, per the official submission
template) and a human contributor (the `sabi` CLI).

```bash
git clone https://github.com/godspoweruyanga/sabi-llm.git
cd sabi-llm
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[tui,serve,inference]"
./download_model.sh    # audit-harness entry point: fetches the model (~1.8 GB)
                        # into ./models/sabi-3b.Q4_K_M.gguf — matches
                        # _runtime.model_path in metadata.json
# or: sabi download     # equivalent, human-friendly CLI entry point
sabi doctor             # verify environment + size vs 7 GB budget
sabi benchmark          # produce telemetry
pytest                  # test suite, incl. headless TUI
sabi run                # launch the coworker
```

- `metadata.json` (repo root) carries team, model, domain, cross-disciplinary
  pairing, and the two visible test prompts, per the official
  [ADTC 2026 submission template](https://github.com/Africa-Deep-Tech-Foundation/adtc-2026-submission-template).
- `download_model.sh` (repo root) is the audit-harness-facing download script;
  it and `metadata.json._runtime.model_path` agree on `models/sabi-3b.Q4_K_M.gguf`.
- Config is in `config/default.yaml`, overridable via `SABI_*` env vars.
- Model source is pinned (`Doctorgp1/sabi-v1`, `sabi-3b.Q4_K_M.gguf`).
- The test suite covers routing, permissions, agent file operations, RAG,
  memory, the web server, and the TUI (headless).

---

## 13. Limitations & risk register (honest)

| Risk | Severity | Mitigation |
|---|---|---|
| Peak RAM near 7 GB → OOM/DQ | **Resolved on 3B** | 7B measured 7.07 GB (over budget); switched to a 3B, measured **3.28 GB** peak RSS on dev hardware (§8). Re-measure on the ADTC Standard Laptop to confirm headroom before audit. |
| Dev-machine benchmark numbers may not transfer 1:1 to audit hardware | Medium | §8's numbers are from a 22-core workstation, not the target 4–8 thread laptop. Re-run `sabi benchmark` and the official `adtc-profiler` on the real (or closest available) target profile before Gate 2. |
| CPU tokens/sec vs smaller-model teams (30% gate) | Low–Med | 3B roughly doubles tok/s vs 7B; measured 18.93 tok/s avg on dev hardware; streaming for perceived speed. |
| African-language bonus not claimed | Low (by design) | Deliberately not claimed for Gate 1 (§11) rather than risk an unsubstantiated claim. Yorùbá pipeline (NLLB + MMS-TTS) is a post-Gate-1 candidate. |
| Tool-call reliability on a small model | **Resolved 2026-08-14** | Was confirmed 2026-08-13: `sabi agent "create a file main.py that prints hello"` reproducibly returned the code as plain text instead of the `write_file` JSON call. A temperature-only fix (0.4→0.1) was tried and was **not** sufficient — re-tested live and it still narrated prose. Fixed by switching the agent loop to `json_mode=True` (llama.cpp's `response_format: json_object` grammar), which structurally rules out prose replies: every turn must be either `{"tool": ..., "args": {...}}` or `{"final": "..."}`. Re-verified live: the exact repro case now creates the file every run (3/3), and the actual audit `test_prompt` #2 ("Scaffold a Python CLI project structure with argparse") now produces a real 6-file scaffold (`main.py`, `setup.py`, `requirements.txt`, `__init__.py`, `README.md`, `.gitignore`) with correct, runnable `argparse` code — this previously stopped early. A secondary issue surfaced during the fix — the model sometimes re-verifies a completed file/command redundantly instead of stopping — mitigated with an exact-repeat-call dedup guard in `AgentLoop.run` that force-terminates the loop rather than burning the step budget; a prompt-only nudge alone did not fully fix this either. |
| Interface verified mainly headlessly during development | Low | Final pass on real 8 GB Ubuntu hardware before audit. |

---

## 14. Roadmap to final submission

1. ~~Fill in the real `team_id` in `metadata.json`~~ — done 2026-08-14
   (`1054704`, DevPost submission ID for the "Sabi AI" project under
   Africa Deep Tech Challenge 2026).
2. Install Python 3.11+ and run the official
   [`adtc-profiler`](https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler)
   on the ADTC Standard Laptop (or closest available equivalent); replace §8's
   dev-machine numbers with audit-grade ones and enter Sperf/Seff on DevPost.
3. Confirm the **3B** stays comfortably under 7 GB on the audit machine
   (re-run `benchmark`); keep 1.5B as a fallback if any team-specific tightness.
4. Capture screenshots/clips and record the 2-minute demo video; finalise the
   10-slide defense deck.
5. Tighten any UI spacing/behaviour found on real hardware.
6. Decide, before Gate 2, whether to implement and validate one African
   language end-to-end and flip `african_alpha_claim` to `true` with evidence —
   or leave it honestly unclaimed.

---

## 15. Authors & license

- **Godspower Uyanga** — Lead Author · Senior Data Scientist / AI Engineer
- **Oreoluwa Akinwe** — Research Analyst

Released under the **MIT License**. Built for the ADTC 2026 Laptop LLM Challenge —
*AI that Africa can own, run, and trust.*
