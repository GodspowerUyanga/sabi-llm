# Sabi — Technical Report
### Africa Deep Tech Challenge 2026 · Corporate / Enterprise Track

**Submission:** Sabi, an offline enterprise knowledge assistant powered by the custom **Sabi-1** model.
**Domain:** Corporate / Enterprise — knowledge-work productivity for SMEs and operators.
**Cross-disciplinary integration:** Enterprise knowledge work **×** quantitative/data reasoning.

---

## 1. Problem definition

African SMEs and operators do knowledge work — answering questions from policy and
contract documents, drafting communications, summarising reports, and reasoning about
financial and operational numbers — on modest hardware, with unreliable connectivity,
and with strong reasons to keep commercial data private. Cloud assistants impose API
fees, depend on stable fibre and power, and send sensitive business data off-device.

**Sabi** removes those blockers. It is a fully offline assistant that runs on the
8 GB laptop an operator already owns. It grounds every answer in the company's own
documents, never fabricates figures (it computes them with a tool), and works in
English plus four African languages. Nothing leaves the device.

Concrete jobs Sabi does for an SME:
- *"What's our refund policy?"* → grounded answer from the company handbook.
- *"What's our total Q1 revenue, and is it on track for the ₦52M Q2 target?"* → reads the
  sales CSV, computes the exact sum with a tool, compares against the strategy doc.
- *"Draft a reminder to staff about expense receipts."* → short, professional draft.
- *"Abeg, how many leave days person fit take?"* → answered in Nigerian Pidgin.

## 2. Constraints (and how the design respects them)

The ADTC Standard Laptop: 8 GB RAM (**7 GB hard ceiling**), Intel i5 10–12th / Ryzen 5,
integrated graphics (**no GPU**), Ubuntu 22.04, x86-64. Exceeding 7 GB or crashing →
disqualification (Stotal = 0).

| Constraint | Design response |
|---|---|
| 7 GB RAM ceiling | `q4_k_m` quantization; **mmap** weights (low resident set); `n_ctx=4096`, `n_batch=256`; embedder loaded only when needed then **released**; live RAM guard. |
| No GPU | llama.cpp CPU inference (`n_gpu_layers=0`). |
| No cloud | All inference offline; only the one-time model download touches the network. |
| Disqualification on OOM | Profiler verifies peak RAM < budget before submission; `guard_budget()` refuses unsafe runs. |
| Thermal penalty (>85 °C) | Small model + bounded batch keep sustained load low; profiler reports core temperature. |

## 3. Design decisions

**Inference runtime — llama.cpp (via `llama-cpp-python`).** The de-facto standard for
CPU GGUF inference on constrained hardware: quantization support, mmap, deterministic,
no GPU needed, mature on Ubuntu.

**Model — Sabi-1, built on Qwen2.5-Instruct (`q4_k_m`).** Qwen2.5 is Apache-2.0, strong
at instruction-following and knowledge work at small sizes, and meaningfully multilingual
(including Swahili), which supports the African-language bonus. We default to **3B** for
accuracy and offer a **1.5B** profile for the budget-laptop bonus (lower RAM + higher TPS).

**RAG without a database.** A single NumPy matrix + JSON sidecar holds embeddings and
chunk metadata; search is brute-force cosine similarity. For an SME corpus (hundreds to a
few thousand chunks) this is fast and adds **zero** heavy dependencies (no FAISS, no DB) —
which keeps both RAM and the reproducibility surface small.

**Tool use over a text protocol.** Rather than relying on a model's native function-calling
(unreliable at 1.5–3B), Sabi uses a simple, robust `<tool_call>{json}</tool_call>` protocol
parsed by the agent. This works consistently across small models and is fully local.

**Low temperature (0.3).** Enterprise answers should be factual and repeatable, not creative.

## 4. Model customization — Sabi-1 (honest scope)

**Sabi AI was created and trained by Godspower Uyanga** as an offline assistant for African
SMEs. This is an *applied inference* contest ("not a training contest"), and fine-tuned
open-source bases are explicitly allowed. The customization is done at the levels that change
what the user actually experiences, and we are precise about what each step does:

1. **Identity.** Sabi presents itself only as **Sabi / Sabi AI / Sabi-1, created and trained by
   Godspower Uyanga**. A dedicated identity guard ensures questions about Sabi ("who are you",
   "who trained you", "what model are you") are answered *only* from this fixed persona and
   never from the user's uploaded documents — so an unrelated file can never redefine Sabi.
2. **Behavioural specialisation (the substance).** A fixed persona (`src/sabi/prompts.py`)
   gives Sabi-1 a consistent enterprise identity and a strict no-fabrication rule; RAG
   grounding ties answers to the user's documents; the tool protocol routes all arithmetic
   to a deterministic calculator; low-temperature decoding makes outputs stable.
3. **Optional adapter hook.** `customize_model.py --lora <path>` records a separately
   trained LoRA for teams that want to go further; the runtime can attach it via llama.cpp.
   We do **not** claim to have trained one — the submission's customization is the rebrand
   plus the behavioural contract, which is reproducible from this repo alone.

We deliberately avoid overclaiming. Sabi-1's value is systems engineering: grounding,
tool-augmented accuracy, multilingual routing, and an aggressively small memory footprint.

## 5. Cross-disciplinary integration (Best-Integration relevant)

**Knowledge work × quantitative reasoning.** Small LLMs are unreliable at arithmetic, which
is fatal for SME finance/operations. Sabi pairs language understanding with two deterministic,
offline tools the model calls when needed:

- **`calc`** — evaluates arithmetic / financial expressions through a safe AST evaluator
  (an allow-list of numeric operations; **never** `eval`/`exec`).
- **`aggregate` / deterministic analysis** — sum / mean / min / max / count over a column of a
  **CSV or Excel** file. Crucially, when the user asks a data question ("total revenue",
  "average units"), the agent detects it and computes the answer **deterministically in code**,
  returning an exact figure and a per-group breakdown table — the small model is never trusted
  to do arithmetic over a table (this eliminates hallucinated totals).
- **`search_docs`** — retrieval over the local corpus (PDF, Word, Excel, CSV, text, Markdown).
- **`pivot_table`** — builds a real cross-tab (rows × columns, e.g. revenue by region × month)
  with row totals, column totals and a grand total, all computed deterministically. Triggered by
  natural phrasing such as "pivot table of revenue by region and month".
- **`create_excel`** — on request ("create an Excel sheet of that summary") Sabi writes a genuine
  `.xlsx` to disk and returns a download link, generated fully offline.
- **knowledge base** — "remember this …" appends a fact to a local `knowledge_base.md` and
  indexes it for later recall, entirely on-device.

Example trace (real, from the mock pipeline; identical control flow with the real model):
```
user  : what is 0.15 * 52000000?
sabi  : <tool_call>{"name":"calc","arguments":{"expression":"0.15*52000000"}}</tool_call>
tool  : {"result": 7800000.0}
sabi  : Based on the computation, the result is 7800000.0.
```

## 6. African-language support (Alpha Bonus, +15%)

**Headline claim: English and Nigerian Pidgin (pcm).** Detection and reply-routing are
implemented for five languages — English, Nigerian Pidgin, Swahili (sw), Yoruba (yo),
Hausa (ha) — but we are precise about where *fluency* is dependable versus best-effort,
because this is exactly what an audit will test.

- **Detection** (`src/sabi/languages.py`): a lightweight, dependency-free marker-word
  heuristic — chosen over a heavy ML detector specifically to protect the RAM budget.
  It favours precision (defaulting to English when unsure) to avoid mis-routing.
- **Reply routing:** when a non-English language is detected, a directive plus a small
  **enterprise glossary** (invoice, revenue, report, customer, meeting…) is injected so
  business terms render correctly rather than as literal translations.
- **Where fluency is dependable:** the base model (Qwen2.5) does not officially list any
  African language among its ~29 supported languages. In practice it handles **Nigerian
  Pidgin** well (English-lexified) — which is also the language of the product name *Sabi*
  ("to know") — so we anchor the +15% claim there. **Swahili, Yoruba and Hausa are
  best-effort**: routing works, but output quality should be verified on the target
  hardware and is not claimed as fluent. Teams wanting strong Yoruba/Hausa can swap the
  base for a model with broader low-resource coverage (e.g. a small Qwen3 variant) via
  `config/sabi.yaml` — the rest of the system is unchanged.

This is deliberately conservative: a demonstrated Pidgin capability that survives testing
is worth more than an over-broad claim that doesn't.

## 7. Efficiency & memory strategy (Seff, 20%)

`Seff = 100 × (7 − PeakRAM) ÷ 7`. We minimise peak resident memory by:
- `q4_k_m` quantization and **mmap** (weights are paged, not all resident);
- a modest context window and prompt batch (`n_ctx=4096`, `n_batch=256`);
- loading the embedding model **only** during indexing/search and releasing it immediately,
  so it does not co-reside with the chat model during the measured workload;
- a brute-force NumPy index instead of an in-memory DB.

`src/sabi/memory.py` samples RSS (process + children) on a background thread to record the
true **peak**, computes `Seff`, and disqualifies any run that breaches the budget — mirroring
the audit so there are no surprises.

## 8. Reproducibility & audit instructions (Gate 2)

```bash
git clone https://github.com/GodspowerUyanga/sabi-llm.git && cd sabi-llm
./setup.sh                 # deps + model download (GitHub) + index
./setup.sh hf 1.5b         # alternative: Hugging Face base, budget profile
python -m sabi bench       # prints + saves benchmark_report.json
python -m sabi serve       # web app;  python -m sabi chat for terminal
pytest -q                  # 22 tests, no model required
```

**Model provisioning.** The weights are hosted as **GitHub Release assets** on the project
repo and fetched by `scripts/download_model.py` (with resume). This gives the audit a single,
reliable, owner-controlled source — no Hugging Face token or proxy required — and the app is
fully offline thereafter. See `docs/PUBLISH_MODEL.md`. Weights are never committed to git.

Everything else an auditor needs is fixed and visible: `config/sabi.yaml` (all run
parameters), pinned `requirements.txt`, the exact base-model sources in
`scripts/download_model.py`, and a `benchmark.py` that reproduces the telemetry. The pipeline
is verifiable **today** in mock mode (no download), and the real model is one command away.

## 9. Benchmark results

> The numbers below must be filled in from a run on the **ADTC Standard Laptop** (or your
> nearest equivalent: 8 GB, i5/Ryzen 5, no GPU, Ubuntu 22.04). Run `python -m sabi bench`,
> which also writes `benchmark_report.json`. Mock-mode speeds are **not** representative and
> are intentionally excluded.

| Metric | Sabi-1 (3B) | Sabi-1 (1.5B) | Notes |
|---|---|---|---|
| Avg generation speed (tok/s) | _fill_ | _fill_ | feeds Sperf = 100 × TPSact/TPSmax |
| Peak RAM (GB) | _fill_ (~2.8–3.4 est.) | _fill_ (~1.6–2.2 est.) | budget 7 GB |
| Efficiency Seff | _fill_ (~51–60 est.) | _fill_ (~69–77 est.) | 100×(7−peak)/7 |
| Max core temp (°C) | _fill_ | _fill_ | penalty if >85 |
| OOM / crash | None expected | None expected | within budget |

*Estimates are engineering expectations for guidance only; report measured values.*
The 1.5B profile is the recommended entry for the **budget-laptop bonus** and maximises the
Speed and Efficiency components; the 3B profile favours the 50%-weighted Accuracy component.

## 10. Bonus claims mapping

| Bonus | Claim | Evidence |
|---|---|---|
| African Language (+15%) | English + **Nigerian Pidgin** (dependable); Swahili/Yoruba/Hausa best-effort, auto-detected, glossary-grounded | `languages.py`, tests, UI chips, demo |
| Budget Laptop (+10%) | 1.5B profile lowers RAM and raises TPS for refurbished machines | `download_model.py --size 1.5b`, `config` |

## 11. Limitations & honesty

- Sabi-1 is a *customized inference system*, not a from-scratch trained model; customization
  is the metadata rebrand plus the behavioural/grounding/tooling contract (Section 4).
- Language detection is heuristic; it is tuned for short enterprise prompts and favours
  precision (defaulting to English when unsure) to avoid mis-routing.
- RAG quality depends on the documents provided; Sabi says when an answer is not in the
  corpus rather than guessing.
- Real benchmark numbers must come from the target hardware; we ship the profiler to make
  that one command.

## 12. Attribution

- **Qwen2.5-Instruct** — Alibaba/Qwen, Apache-2.0 (base chat model).
- **bge-small-en-v1.5** — BAAI, MIT (embeddings).
- **llama.cpp / llama-cpp-python** — MIT (runtime).
- FastAPI, NumPy, psutil, PyYAML, Pydantic, huggingface-hub, gguf — respective OSS licenses.

Sabi project code: Apache-2.0 (`LICENSE`).
