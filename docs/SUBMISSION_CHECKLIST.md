# Gate 1 Submission Checklist (due July 24, 2026)

Map each required deliverable to what's in this repo.

## Required

- [x] **Open-source GitHub repo** — https://github.com/GodspowerUyanga/sabi-llm. Apache-2.0.
- [ ] **Publish the model to GitHub Releases** — so `download_model.py` works for the audit.
      Follow `docs/PUBLISH_MODEL.md` (upload `sabi-1.gguf` + `embedding.gguf` to tag `v1.0`).
- [x] **`REPORT.md`** — problem definition, constraints, design decisions, model
      customization, cross-disciplinary integration, tools & benchmarks, reproducibility.
- [ ] **Screenshots / short clips of the model running** — capture:
      - the web UI answering a document question (RAG),
      - a `[used calc]` / `[used aggregate]` tool trace,
      - a Pidgin/Swahili reply,
      - `python -m sabi bench` output with peak RAM + Seff.
- [ ] **2-minute demo video** — follow `docs/VIDEO_SCRIPT.md`. Show the network unplugged
      and a RAM monitor on screen.
- [x] **Bonus claims** — African language (+15%) and budget laptop (+10%), documented in
      `REPORT.md` §10 and demonstrated in the app.

## Before you submit — verify on the target hardware

- [ ] `./setup.sh` completes on Ubuntu 22.04 (8 GB, no GPU).
- [ ] `python -m sabi bench` shows **peak RAM < 7 GB** and no OOM. Save `benchmark_report.json`.
- [ ] Fill the benchmark table in `REPORT.md` §9 with the measured numbers.
- [ ] Confirm inference works with the network disconnected.
- [ ] (Recommended) Run both `./setup.sh 3b` and `./setup.sh 1.5b`; report whichever profile
      you submit. 1.5B strengthens Speed/Efficiency + the budget bonus; 3B strengthens Accuracy.
- [ ] `pytest -q` → 22 passed.

## Nice-to-have polish

- [ ] Add a few of your own real (non-confidential) documents to `data/corpus/` for the demo.
- [ ] Take a screenshot of the GGUF reporting as `Sabi-1` (e.g. `python -c "from gguf import GGUFReader; ..."`).
- [ ] Record the model download step once so reviewers see the one-time online step.

## Gate 2 (audit) readiness — already covered
- Pinned `requirements.txt`, fixed `config/sabi.yaml`, exact base-model sources in
  `scripts/download_model.py`, and a reproducible `benchmark.py`.
- The mock backends let an auditor exercise the full pipeline without downloads.
