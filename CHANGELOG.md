# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project follows
semantic versioning.

## [1.0.0] — 2026-06-24

### Added
- Initial release of SABI, the offline AI coworker.
- Single quantized GGUF model wrapper via llama.cpp with graceful degradation
  when the model or backend is unavailable (`sabi/model.py`).
- Intent router with keyword heuristic and optional model arbitration
  (`sabi/router.py`).
- THINK and CODE reasoning engines (`sabi/engines/`).
- Agent loop: plan → execute → verify (`sabi/agent.py`).
- Local JSON memory store (`sabi/memory/`).
- Fully offline RAG layer: hashing embedder, JSON vector store, retriever
  (`sabi/rag/`).
- Sandboxed tool layer: file, shell (deny-listed), workspace scaffolding
  (`sabi/tools/`).
- Project recognition for Git, Python, Node.js, Next.js, package managers and
  virtual environments (`sabi/project_scanner.py`).
- Telemetry: `benchmark`, `profile` and `doctor` commands aligned to the
  ADTC 2026 scoring model.
- CLI with `run`, `chat`, `ask`, `think`, `code`, `agent`, `benchmark`,
  `profile`, `doctor`, `workspace`, `version` (`sabi/cli.py`).
- Configuration via `config/default.yaml` and `SABI_*` environment variables.
- Model download from Hugging Face (`scripts/download_model.py`).
- Benchmark runner producing JSON + Markdown reports
  (`scripts/run_benchmark.py`).
- Test suite (pytest) and Makefile for common tasks.

[1.0.0]: https://github.com/godspoweruyanga/sabi-llm/releases/tag/v1.0.0
