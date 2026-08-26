# Demo evidence (real, unedited transcripts)

`01`, `02`, `04`, `05` were re-recorded on 2026-08-25 against the current
build. `03` is the original 2026-08-14 capture — still accurate, untouched.
Nothing here is fabricated or trimmed for effect.

> **2026-08-25 history.** These transcripts went through three rounds of
> re-recording the same day, each one closing a real gap found by actually
> running the commands rather than assuming a fix worked:
>
> 1. `01_doctor.txt`/`02_benchmark_devmachine.json` originally showed numbers
>    measured against a mismatched model file (1.80 GB, 18.93 tok/s, 3.28 GB,
>    78.1%). Re-run against the correct model (1.96 GB, ~17-20 tok/s,
>    ~3.45 GB peak RSS, ~72% accuracy — see REPORT.md §8 and the official
>    `adtc-profiler` run). `01_doctor.txt` is now a literal terminal capture
>    from the maintainer's own machine, not a re-run by the assistant.
> 2. `04`'s original known issue — natural phrasing not reliably triggering
>    the agent's tool-call loop — was fixed, but re-testing surfaced a
>    *different* issue: after finishing, the model invented one extra tool
>    call (`delete_file`/`delete_dir`) that isn't in the registry. It failed
>    safely (nothing was deleted), but was real.
> 3. That, in turn, led to a genuine incident during further testing: a bare
>    **"hello" reached the same tool-calling loop** (both `sabi chat` and
>    `sabi tui` route everything through it by design) and the model invented
>    and *executed* a tool call that corrupted real repository files
>    (overwrote two test files, put a syntax error into
>    `sabi/tools/workspace_tools.py`). Root cause: `prompts/agent.txt` was
>    stale (missing tools) and self-contradictory (told the model to answer
>    in plain text while the code forces every reply through a JSON
>    grammar). Fixed by rebuilding the prompt from the correct JSON-mode-aware
>    version, adding `sabi.router.is_smalltalk` as a hard code gate so
>    greetings never reach the tool loop at all, and blocking `rm`/`rmdir`/
>    `del` through `run_shell` at the code level (deletion was never a
>    supported agent action). See REPORT.md §13 for the full writeup.
>
> `04` and `05` here are re-recorded a third time, against the fully fixed
> build: both now complete their task in the minimum number of steps and
> stop cleanly with no invented extra call.

| File | What it shows |
|---|---|
| `01_doctor.txt` | `sabi doctor` — environment check, model size, estimated runtime RAM vs the 7 GB ceiling. Literal capture from the maintainer's terminal. |
| `02_benchmark_devmachine.json` | `sabi benchmark --json` — full run over `benchmarks/prompts.jsonl` (8 prompts). Feeds REPORT.md §8. |
| `03_think_prd_example.txt` | `sabi think` — the Corporate/Enterprise reasoning engine producing a one-page PRD for an offline crop-price advisory tool. This is the input to the cross-disciplinary demo below. |
| `04_agent_natural_phrasing_known_issue.txt` | `sabi agent` given natural phrasing ("create a file main.py that prints hello" — the *exact example from the agent's own system prompt*). One tool call (write), then a proper final answer, then stops. |
| `05_agent_explicit_tool_success.txt` | The same intent plus "then run it". Two tool calls (write, run), then a proper final answer, then stops. |
| `gradio-live-demo-overview.png` | Screenshot of the public [live demo](https://8554ef3b3ec8bc6c1f.gradio.live) — an English exchange and a Yoruba-toggled reply in the same session. Referenced from `README.md` and `REPORT.md`. |

## Why this matters for the video / live demo

Both the natural-phrasing gap and the invented-tool-call/greeting-corruption
issues are fixed and covered by 5 new regression tests (`tests/test_router.py`,
`tests/test_agent.py`, `tests/test_tui.py`). Worth a quick live sanity check
before recording rather than assuming — don't discover a regression live on
camera — but there's no known open issue to work around at this point.

## Live demo

**[8554ef3b3ec8bc6c1f.gradio.live](https://8554ef3b3ec8bc6c1f.gradio.live)** —
a hosted, click-and-test instance of SABI's Coding Assistant (the same
`sabi-v1` model + `sabi-yoruba-llm` layer as everywhere else in this repo),
public with no install needed. Text-only (no file/shell access) since the
link is open to the internet — see `README.md`'s "Live demo" section for the
full writeup and `REPORT.md`'s header note for the reviewer-facing version.
It's a temporary tunnel (best-effort up to ~1 week) from the author's own
machine, not a permanent deployment.

`gradio-live-demo-overview.png` (table above) is a real, unedited screenshot
of it — an English exchange followed by a Yoruba-toggled reply in the same
session, showing the chat history sidebar and document-upload panel too.

## Screenshots / video — still needed

The transcripts above are text, not the images/video Gate 1 asks for (the
live-demo screenshot covers the web UI, but not the terminal/TUI). You still
need to:
1. Open a real terminal on your machine, run the same commands, and take actual
   screenshots (or a short screen recording) — text logs aren't a substitute.
2. Record the 2-minute demo video — see `demo/VIDEO_SCRIPT.md`.
