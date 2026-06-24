# Contributing to SABI

Thanks for your interest in improving SABI — the offline AI coworker. This guide
covers how to set up a development environment and the conventions we follow.

## Development setup

```bash
git clone https://github.com/godspoweruyanga/sabi-llm.git
cd sabi-llm
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Run the checks before opening a pull request:

```bash
pytest          # tests must pass
ruff check sabi tests scripts   # lint must be clean
```

## Principles

SABI is **offline-first** and **resource-constrained**. When contributing,
please keep these in mind:

1. **No network on the critical path.** Inference, RAG, memory and tools must
   work without internet. Only the model-download script may reach the network.
2. **Stay light.** Peak RAM must remain under the 7 GB ceiling. Avoid heavy
   dependencies; prefer the standard library and small, optional extras.
3. **Degrade gracefully.** The app must start and give actionable guidance even
   when the model or an optional dependency is missing. Never crash on a missing
   model — raise `ModelUnavailable` or return a clear result instead.
4. **Sandbox tools.** File and shell tools must stay within the workspace and
   respect the deny-list.

## Code style

- Target Python 3.9+.
- Format/lint with `ruff` (line length 100).
- Add or update tests for any behaviour change.
- Keep functions small and documented with concise docstrings.

## Commit / PR

- Write clear commit messages (imperative mood: "add", "fix", "refactor").
- Reference any related issue.
- Describe what changed and how you tested it in the PR body.

## Reporting issues

Open an issue at
<https://github.com/godspoweruyanga/sabi-llm/issues> with steps to reproduce,
your OS/Python version, and the output of `sabi doctor`.
