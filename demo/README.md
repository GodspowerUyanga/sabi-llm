# Demo evidence (real, unedited transcripts)

`01`, `02`, `04`, `05` were re-recorded on 2026-08-25 against the current
build (correct model, merged agent/runtime fixes). `03` is the original
2026-08-14 capture — still accurate, untouched. Nothing here is fabricated
or trimmed for effect, including the ones that show a real (non-blocking)
quirk.

> **2026-08-25 re-recording note.** `01_doctor.txt` and
> `02_benchmark_devmachine.json` previously showed numbers measured against
> a model file mismatched with what the repo actually downloads (1.80 GB,
> 18.93 tok/s, 3.28 GB, 78.1%). Both are now fresh runs against the correct
> model (1.96 GB, ~17-20 tok/s, ~3.45 GB peak RSS, ~72% accuracy — see
> REPORT.md §8 for the full numbers plus the official `adtc-profiler` run).
>
> `04`'s original known issue — natural phrasing not reliably triggering the
> agent's tool-call loop — **is fixed**: both the natural phrasing in `04`
> and the explicit-tool-name phrasing in `05` now correctly write the file,
> run it, and list the directory. Re-testing live surfaced a *different*,
> smaller issue in both runs: after finishing the task, the model attempts
> one more tool call it invents itself (`delete_file` in `04`, `delete_dir`
> in `05`) that isn't in the tool registry. The call fails cleanly —
> `Unknown tool: ...` — and nothing is deleted; both transcripts confirm the
> file/directory survive intact. This is worth being upfront about rather
> than re-recording until it disappears: it demonstrates the tool allowlist
> working as a safety net (an invented destructive call cannot execute),
> but it's also a real small-model quirk (over-continuing past task
> completion) that a future session could tighten with an explicit stop
> condition once the model reports all steps done.

| File | What it shows |
|---|---|
| `01_doctor.txt` | `sabi doctor` — environment check, model size, estimated runtime RAM vs the 7 GB ceiling. |
| `02_benchmark_devmachine.json` | `sabi benchmark --json` — full run over `benchmarks/prompts.jsonl` (8 prompts). Feeds REPORT.md §8. |
| `03_think_prd_example.txt` | `sabi think` — the Corporate/Enterprise reasoning engine producing a one-page PRD for an offline crop-price advisory tool. This is the input to the cross-disciplinary demo below. |
| `04_agent_natural_phrasing_known_issue.txt` | `sabi agent` given natural phrasing ("create a file main.py that prints hello" — the *exact example from the agent's own system prompt*). Core task (write → run → list) succeeds; ends with a safely-rejected invented `delete_file` call (see note above). |
| `05_agent_explicit_tool_success.txt` | The same intent, phrased with an explicit tool-name cue. Same result: core task succeeds, then a safely-rejected invented `delete_dir` call. |

## Why this matters for the video / live demo

The natural-phrasing gap documented in `04` is fixed — either phrasing style
works for the actual task. The invented-tool-call quirk at the end of both
runs is real and worth mentioning honestly if asked in Gate 2/3 Q&A, but it
doesn't block the demo: the task completes correctly before it happens, and
the failed call has zero effect on the filesystem.

## Screenshots / video — still needed

These are text transcripts, not the images/video Gate 1 asks for. You still
need to:
1. Open a real terminal on your machine, run the same commands, and take actual
   screenshots (or a short screen recording) — text logs aren't a substitute.
2. Record the 2-minute demo video — see `demo/VIDEO_SCRIPT.md`.
