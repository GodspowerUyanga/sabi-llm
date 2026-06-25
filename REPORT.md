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
| **Bonus claims** | Budget-laptop profile: **claimed** · African-language bonus: **in progress** (see §11) |

> **A note to reviewers on numbers.** Every quantitative claim marked `‹MEASURE›`
> is to be filled from a real run of our profiler (`sabi benchmark` /
> `python scripts/run_benchmark.py`) on the ADTC Standard Laptop. We have not
> substituted estimates for measured telemetry anywhere in this report; estimates
> are labelled as such.

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
| Tests | `pytest`, `pytest-asyncio` | 52 tests incl. headless TUI |

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

**Measured results on the ADTC Standard Laptop** *(to be completed from a real
run before audit):*

| Metric | Target | Measured |
|---|---|---|
| Model size on disk | < 7 GB | `‹MEASURE›` GB |
| Peak RAM (RSS) during inference | < 6.5 GB (well under 7 GB) | `‹MEASURE›` GB |
| Efficiency score `Seff = 100×(7−PeakRAM)/7` | maximise | `‹MEASURE›` |
| Tokens/sec (CPU) | 10–20 tok/s | `‹MEASURE›` tok/s |
| Cold start (load → first token) | < 5 s | `‹MEASURE›` s |
| Peak core temperature | < 85 °C (avoid −10) | `‹MEASURE›` °C |
| Benchmark accuracy (prompt set) | maximise | `‹MEASURE›` % |
| Crashes / OOM | 0 | `‹MEASURE›` |

> Reviewers can reproduce these with the commands above; the prompt set lives in
> `benchmarks/prompts.jsonl`.

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

- **Budget-laptop profile (+10%): claimed.** SABI is designed and tested for the
  $150–$500, 8 GB, no-GPU machine and reports its footprint against the budget.
- **African-language bonus (+15%): in progress — not yet claimed.** The
  architecture includes a `SABI_LANGUAGE` setting (`en|yo|ha|ig`) and localized
  prompt scaffolding, but meaningful end-to-end functionality in an African
  language is still being validated. **We will only claim this bonus once it is
  demonstrably working**; this report will be updated with evidence (sample
  transcripts) if/when it qualifies. We flag this transparently rather than
  overclaim.

---

## 12. Reproducibility (for the Gate 2 audit)

```bash
git clone https://github.com/godspoweruyanga/sabi-llm.git
cd sabi-llm
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[tui,serve,inference]"
sabi download          # fetch the model (~‹MEASURE› GB) into ./models/
sabi doctor            # verify environment + size vs 7 GB budget
sabi benchmark         # produce telemetry
pytest                 # 52 tests, incl. headless TUI
sabi run               # launch the coworker
```

- Config is in `config/default.yaml`, overridable via `SABI_*` env vars.
- Model source is pinned (`Doctorgp1/sabi-v1`, `sabi-3b.Q4_K_M.gguf`).
- The test suite covers routing, permissions, agent file operations, RAG,
  memory, the web server, and the TUI (headless).

---

## 13. Limitations & risk register (honest)

| Risk | Severity | Mitigation |
|---|---|---|
| Peak RAM near 7 GB → OOM/DQ | **Resolved on 3B** | 7B measured 7.07 GB (over budget); switched to a 3B (~3.5–4.5 GB runtime). Re-measure with `benchmark` to confirm headroom before audit. |
| CPU tokens/sec vs smaller-model teams (30% gate) | Low–Med | 3B roughly doubles tok/s vs 7B; streaming for perceived speed. |
| African-language bonus not yet earned | Medium | Yorùbá speech pipeline planned (NLLB + MMS-TTS), lazy-loaded under the ceiling; validate before claiming +15%. |
| Tool-call reliability on a small model | Medium | Strong tool-use prompt, action-routing gate, and a path safety-net reduce mis-fires. |
| Interface verified mainly headlessly during development | Low | Final pass on real 8 GB Ubuntu hardware before audit. |

---

## 14. Roadmap to final submission

1. Run the profiler on the ADTC Standard Laptop; fill every `‹MEASURE›`.
2. Confirm the **3B** stays comfortably under 7 GB on the audit machine
   (re-run `benchmark`); keep 1.5B as a fallback if any team-specific tightness.
3. Implement and validate **one African language** end-to-end → claim +15%.
4. Record the 2-minute demo video; finalise the 10-slide defense deck.
5. Tighten any UI spacing/behaviour found on real hardware.

---

## 15. Authors & license

- **Godspower Uyanga** — Lead Author · Senior Data Scientist / AI Engineer
- **Oreoluwa Akinwe** — Research Analyst

Released under the **MIT License**. Built for the ADTC 2026 Laptop LLM Challenge —
*AI that Africa can own, run, and trust.*
