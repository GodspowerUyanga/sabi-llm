# SABI — The Offline AI Coworker

### ADTC 2026 Laptop LLM Challenge — Gate 1 Submission Report

| | |
|---|---|
| **Project** | SABI — The Offline AI Coworker |
| **Primary track** | Coding Assistants |
| **Cross-disciplinary integration** | Coding Assistant × Corporate/Enterprise × Autonomous AI Agents |
| **Repository** | https://github.com/godspoweruyanga/sabi-llm |
| **Model** | Qwen2.5-Coder-3B-Instruct, Q4_K_M GGUF (~2.0 GB), runs via llama.cpp — downloaded automatically on first start from SABI's own repo (https://huggingface.co/Doctorgp1/sabi-v1), an unmodified mirror of Qwen's official release; saved locally as `sabi-v1.Q4_K_M.gguf` |
| **Target hardware** | ADTC Standard Laptop — 8 GB RAM, no discrete GPU, Ubuntu 22.04 |
| **Memory ceiling** | 7 GB (hard limit; exceeding it = disqualification) |
| **Authors** | Godspower Uyanga (lead) · Oreoluwa Akinwe |
| **License** | MIT (SABI's own code; the models it downloads carry their own separate licenses — see §13) |
| **Bonus claims** | Budget-laptop profile: **claimed** · African-language bonus: **claimed** — Yoruba via sabi-yoruba-tts (see §11) |

> **A note to reviewers on numbers.** The table in §8 is a real run of
> `sabi benchmark` **and** the official `adtc-profiler` (both run 2026-08-25),
> wrapping the same llama.cpp CPU inference path used at audit, on the
> development machine used to build SABI — **not** the ADTC Standard Laptop
> (see the profiler's own `environment` block below for the exact spec). We do
> not substitute estimates for measured telemetry anywhere in this report;
> every number below came from an actual run, and every number is labelled
> with the hardware it was measured on. We re-run both tools on the target
> 8 GB no-GPU profile before Gate 2 and update this table with those figures.

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

| Metric | Target | Measured (dev machine, 2026-08-25) |
|---|---|---|
| Model size on disk | < 7 GB | **1.96 GB** (`sabi-v1.Q4_K_M.gguf`, Qwen2.5-Coder-3B-Instruct Q4_K_M — verified against the file actually downloaded by `download_model.sh`) |
| Peak RAM (RSS) during inference | < 6.5 GB (well under 7 GB) | **3.45 GB** |
| Efficiency score `Seff = 100×(7−PeakRAM)/7` | maximise | **50.8** |
| Tokens/sec (CPU), avg across 8 prompts | 10–20 tok/s | **17.42 tok/s** |
| Cold start (load → first token) | < 5 s | not separately instrumented by `sabi benchmark`; to be measured on audit hardware |
| Peak core temperature / thermal throttle | < 85 °C (avoid −10) | **throttle flag tripped** — see caveat |
| Benchmark accuracy (prompt set, keyword-match heuristic) | maximise | **71.9%** |
| Crashes / OOM | 0 | **0** |

> **2026-08-25 re-measurement note.** These numbers replace an earlier table
> measured against a mismatched model file (claimed 1.80 GB; the file actually
> referenced by `download_model.sh` at the time 404'd, and our own
> `Doctorgp1/sabi-v1` HF upload turned out to be a stale 7B build at 4.68 GB —
> see §6 and §13). `download_model.sh` now sources the verified
> `Qwen/Qwen2.5-Coder-3B-Instruct-GGUF` file directly (2,104,932,800 bytes
> confirmed via the HF API), and every number above was re-measured against
> that exact file with `sabi doctor` + `sabi benchmark` +
> `python scripts/run_benchmark.py` on 2026-08-25.
>
> **2026-08-25, later same day.** `Doctorgp1/sabi-v1` on Hugging Face has
> been corrected: it now holds this exact verified `sabi-v1.Q4_K_M.gguf`
> (SHA256 `724fb256bec1ff062b2f65e4569e871ad2e95ab2a3989723d1769c54294730b7`,
> 2,104,932,800 bytes) with a model card, `LICENSE`, and `NOTICE` disclosing
> the Qwen Research License. The stale 4.68 GB build referenced above is no
> longer live.
>
> **2026-08-26.** `config/default.yaml`, `sabi/config.py`, and
> `download_model.sh` now source the model from `Doctorgp1/sabi-v1` by
> default instead of Qwen's upstream repo directly (same verified file — see
> the note above). More importantly, downloading is no longer a separate
> manual step at all: `sabi run` / `chat` / `tui` / `serve` all check for the
> model on startup and fetch it automatically (no `[Y/n]` prompt) if it's
> missing, and `sabi serve` (the web UI) does the same before it starts
> listening — so a judge only has to run one command and wait, not follow a
> multi-step setup. The Yoruba translation layer (§11) is fetched the same
> eager way, not lazily: every entry point above, `download_model.sh`, and
> the public `gradio_demo.py` share-link demo all download **both**
> `sabi-v1` and `sabi-yoruba-llm` up front (rather than only the coder model,
> or waiting for a Yoruba message to trigger the second download), so a
> judge who only follows the documented steps ends up with the full
> bilingual build ready before their first message, not just the English
> half.

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

**Official `adtc-profiler` results (participant mode, 2026-08-25):**

```bash
adtc-profiler run --submission . --mode participant --output submission.json --skip-accuracy
```

| Field (from `submission.json`) | Value |
|---|---|
| `environment.measured_on` | **`participant_laptop`** |
| `environment.cpu_model` | Intel(R) Core(TM) Ultra 7 165H (dev laptop — not the ADTC Standard Laptop's i5 10th–12th gen) |
| `environment.ram_gb` | 30.8 |
| `throughput.tokens_per_second_generation` | **19.36** |
| `throughput.first_token_latency_ms` | 8347.54 |
| `memory.peak_rss_mb` | **3453.76** (3.37 GB) |
| `memory.steady_state_rss_mb` | 3372.75 |
| `cpu_thermal.core_temp_c_peak` | 102.0 °C — `throttled: true` (dev-machine artifact, see caveat above) |
| `model_info.params_count` | 3,397,103,616 |
| `model_info.claimed_params_estimate` / `params_match` | `"3B"` / **`true`** — independently confirms this is genuinely the 3B build, not the mismatched 4.68 GB file `Doctorgp1/sabi-v1` previously held (§6, §13) |
| `reproducibility.git_commit_sha` | `f5fe9ce58b63` |

These figures corroborate `sabi benchmark`'s numbers (same peak RAM to within
~80 MB, tok/s within ~2). `accuracy` is empty because this run used
`--skip-accuracy`; the DevPost self-reported Sperf/Seff fields and a full
accuracy pass are still pending an actual ADTC Standard Laptop run before
Gate 2.

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
- **African-language bonus (+15%): claimed** (`african_alpha_claim: true`,
  `language_scope: ["en", "yo"]`). Live testing (2026-08-14) showed the base
  model's raw Yoruba output was degenerate/repetitive — Qwen2.5-Coder-3B was
  never trained for Yoruba, so we did not ship a same-session prompt hack for
  it. Instead we added **sabi-yoruba-tts**, a dedicated translation layer
  (`sabi/translate.py`) around the same English-speaking model: a Yoruba turn
  is translated to English (`Runtime._to_english`), routed and answered
  exactly like any other request, then the English reply is translated back
  to Yoruba (`Runtime._to_yoruba`) before it reaches the user — wired into
  both `Runtime.handle()` and `Runtime.agent()`. Backend: Meta's
  `facebook/nllb-200-distilled-600M` (NLLB-200), converted once to int8
  CTranslate2 via `scripts/download_yoruba_model.py` (`ctranslate2` +
  `transformers` at runtime; no `torch` needed after conversion). Fenced code
  blocks are never sent through the translator (`_CODE_FENCE_RE` in
  `translate.py`), so code explained in Yoruba still contains real, runnable
  code — verified directly:

  ```text
  >>> to_english("Ṣe o le ràn mí lọ́wọ́ pẹ̀lú kóòdù yìí?")
  "Can you help me with this code?"
  >>> to_yoruba("Here is a fix:\n\n```python\nprint(1)\n```\n\nDone.")
  "Ọ̀nà kan ni pé:\n\n```python\nprint(1)\n```\n\nA ti ṣe é."
  ```

  **License disclosure.** NLLB-200-distilled-600M is **CC-BY-NC-4.0
  (non-commercial)** — unlike the rest of SABI (MIT). This is appropriate for
  a non-commercial hackathon submission; a different translation model would
  be needed before any commercial use. Detection heuristic
  (`translate.looks_like_yoruba`): Yoruba diacritics (ẹ ọ ṣ etc.) or common
  romanized marker words (bawo, jowo, pele, ...) typed without diacritics.
  Model adds ~635 MB resident when loaded (lazy, first Yoruba turn only) —
  see §10 for how this stays under the 7 GB ceiling alongside the ~3.5–4.5 GB
  runtime footprint of the 3B coder model.

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
./download_model.sh    # audit-harness entry point: fetches sabi-v1 (~2.0 GB)
                        # into ./models/sabi-v1.Q4_K_M.gguf — matches
                        # _runtime.model_path in metadata.json — plus
                        # sabi-yoruba-llm (~635 MB) into
                        # ./models/sabi-yoruba-tts/
# or: sabi download     # equivalent, human-friendly CLI entry point for sabi-v1 only
sabi doctor             # verify environment + size vs 7 GB budget
sabi benchmark          # produce telemetry
pytest                  # test suite, incl. headless TUI
sabi run                # launch the coworker
```

- `metadata.json` (repo root) carries team, model, domain, cross-disciplinary
  pairing, and the two visible test prompts, per the official
  [ADTC 2026 submission template](https://github.com/Africa-Deep-Tech-Foundation/adtc-2026-submission-template).
- `download_model.sh` (repo root) is the audit-harness-facing download script;
  it and `metadata.json._runtime.model_path` agree on `models/sabi-v1.Q4_K_M.gguf`.
  It also fetches `sabi-yoruba-llm` into `models/sabi-yoruba-tts/` in the same
  run, so the audit flow ends with both models present, not just the coder model.
- Config is in `config/default.yaml`, overridable via `SABI_*` env vars.
- Model source is pinned (`Doctorgp1/sabi-v1`, `sabi-v1.Q4_K_M.gguf` — a verified, unmodified mirror of Qwen's official `Qwen2.5-Coder-3B-Instruct-GGUF` release, saved locally as `sabi-v1.Q4_K_M.gguf`).
- The test suite covers routing, permissions, agent file operations, RAG,
  memory, the web server, and the TUI (headless).

---

## 13. Limitations & risk register (honest)

| Risk | Severity | Mitigation |
|---|---|---|
| Peak RAM near 7 GB → OOM/DQ | **Resolved on 3B** | 7B measured 7.07 GB (over budget); switched to a 3B, measured **3.45 GB** peak RSS on dev hardware (§8). Re-measure on the ADTC Standard Laptop to confirm headroom before audit. |
| Dev-machine benchmark numbers may not transfer 1:1 to audit hardware | Medium | §8's numbers are from a 22-core workstation, not the target 4–8 thread laptop. Re-run `sabi benchmark` and the official `adtc-profiler` on the real (or closest available) target profile before Gate 2. |
| CPU tokens/sec vs smaller-model teams (30% gate) | Low–Med | 3B roughly doubles tok/s vs 7B; measured 17.42 tok/s avg on dev hardware; streaming for perceived speed. |
| sabi-yoruba-llm (NLLB-200) is CC-BY-NC-4.0, not MIT like the rest of SABI | Low, disclosed | Non-commercial license is appropriate for this non-commercial hackathon submission — disclosed in §11 rather than hidden. Would need a different translation model before any commercial use. |
| Qwen2.5-Coder-3B-Instruct (the base model, `Doctorgp1/sabi-v1`) is under the Qwen Research License (non-commercial), not Apache-2.0 or MIT | Low, disclosed | Only the 3B/1.5B/0.5B Qwen2.5-Coder variants carry this non-commercial term (7B+ are Apache-2.0); appropriate for this non-commercial hackathon submission. `LICENSE`/`NOTICE`/attribution shipped alongside the mirrored GGUF on Hugging Face per the license's redistribution terms. Would need a differently-licensed base model before any commercial use. |
| Combined resident footprint (3B coder model + Yoruba layer) vs 7 GB | Low | ~3.5–4.5 GB (coder, §8) + ~635 MB (NLLB int8, lazy-loaded only on a Yoruba turn) — real headroom under 7 GB even summed worst-case. Re-confirm on the ADTC Standard Laptop before Gate 2. |
| Tool-call reliability on a small model | **Resolved 2026-08-14** | Was confirmed 2026-08-13: `sabi agent "create a file main.py that prints hello"` reproducibly returned the code as plain text instead of the `write_file` JSON call. A temperature-only fix (0.4→0.1) was tried and was **not** sufficient — re-tested live and it still narrated prose. Fixed by switching the agent loop to `json_mode=True` (llama.cpp's `response_format: json_object` grammar), which structurally rules out prose replies: every turn must be either `{"tool": ..., "args": {...}}` or `{"final": "..."}`. Re-verified live: the exact repro case now creates the file every run (3/3), and the actual audit `test_prompt` #2 ("Scaffold a Python CLI project structure with argparse") now produces a real 6-file scaffold (`main.py`, `setup.py`, `requirements.txt`, `__init__.py`, `README.md`, `.gitignore`) with correct, runnable `argparse` code — this previously stopped early. A secondary issue surfaced during the fix — the model sometimes re-verifies a completed file/command redundantly instead of stopping — mitigated with an exact-repeat-call dedup guard in `AgentLoop.run` that force-terminates the loop rather than burning the step budget; a prompt-only nudge alone did not fully fix this either. |
| Model invents a nonexistent tool call after finishing a task | **Resolved 2026-08-25** | Observed 2026-08-25 (`demo/04`, `demo/05`): after correctly completing write→run→list, the model called one extra tool it invented itself (`delete_file`/`delete_dir`), rejected outright by `AgentLoop` with zero filesystem effect — the tool allowlist working as a safety net. Root-caused and fixed same day (see next row): `prompts/agent.txt` now carries an explicit no-deletion / stop-condition rule, and `run_shell` blocks `rm`/`rmdir`/`del`/`unlink`/`shred` at the code level regardless of what the model asks for. |
| A bare greeting reached the tool-calling agent loop and corrupted real files | **Resolved 2026-08-25** | Real incident, not a test: sending "hello" to `sabi chat`/`sabi tui` reached the full tool-access agent loop (by design — every message goes through it so the model can decide tool-vs-chat per turn) and the model invented and executed a tool call that overwrote `tests/test_yoruba_runtime.py`, `tests/test_memory.py`, and introduced a syntax error into `sabi/tools/workspace_tools.py` (the likely cause of an observed crash). Two compounding root causes fixed: (1) `prompts/agent.txt` was stale — missing `search_files`/`edit_file` entirely, and instructed a plain-text finish while `AgentLoop.run()` hardcodes `json_mode=True`, a direct contradiction with the model's actual output grammar; rebuilt from the correct JSON-mode-aware prompt that was already sitting unused as `agent.py`'s `DEFAULT_AGENT_PROMPT`. (2) Added `sabi.router.is_smalltalk` — a narrow allowlist (not a keyword search for "no action words," the exact heuristic already removed once for misclassifying real requests) that routes unambiguous greetings/pleasantries straight to `Runtime.handle()` (no tool access) in both `sabi/ui/chat.py` and `sabi/ui/tui.py`, so a greeting can no longer reach the tool loop at all. Verified live: `git status` is identical before/after sending "hello"; a real task ("create a file...") still completes normally and stops without extra actions. 5 new regression tests added (`tests/test_router.py`, `tests/test_agent.py`, `tests/test_tui.py`). |
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
6. ~~Implement and validate Yoruba end-to-end, flip `african_alpha_claim`~~ —
   done 2026-08-24: sabi-yoruba-tts (NLLB-200 int8) wired into
   `Runtime.handle()`/`Runtime.agent()`, verified round-trip including
   code-fence preservation (§11). Re-confirm the combined RAM footprint on the
   ADTC Standard Laptop before Gate 2.

---

## 15. Authors & license

- **Godspower Uyanga** — Lead Author · Senior Data Scientist / AI Engineer
- **Oreoluwa Akinwe** — Research Analyst

Released under the **MIT License**. Built for the ADTC 2026 Laptop LLM Challenge —
*AI that Africa can own, run, and trust.*
