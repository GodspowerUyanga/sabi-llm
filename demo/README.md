# Demo evidence (real, unedited transcripts)

These are raw terminal transcripts captured on 2026-08-14 on the development
machine (not the ADTC Standard Laptop — see REPORT.md §8 for the hardware
caveat). Nothing here is fabricated or trimmed for effect, including the one
that shows a known limitation.

> **2026-08-25 update.** The specific numbers in `01_doctor.txt` and
> `02_benchmark_devmachine.json` (1.80 GB model, 18.93 tok/s, 3.28 GB peak
> RSS, 78.1% accuracy) were measured against a model file that turned out to
> be mismatched with what the repo now downloads — see REPORT.md §8 for the
> corrected, re-measured numbers (1.96 GB, 17.42 tok/s, 3.45 GB, 71.9%) plus
> an official `adtc-profiler` run. The **known issue in `04`** — natural
> phrasing not reliably triggering the agent's tool-call loop — has since
> been fixed: the agent now always routes through the tool-calling loop
> (`sabi/ui/tui.py`), and gained `search_files`/`edit_file`/`read_file` with
> offset, so both `04`'s exact repro and `05`'s explicit-tool-name phrasing
> should now succeed identically. Re-recording these transcripts against the
> current build is worth doing before the demo video, but the underlying gap
> they document is resolved, not just re-labeled.

| File | What it shows |
|---|---|
| `01_doctor.txt` | `sabi doctor` — environment check, model size, estimated runtime RAM vs the 7 GB ceiling. |
| `02_benchmark_devmachine.json` | `sabi benchmark --json` — full run over `benchmarks/prompts.jsonl` (8 prompts). Feeds REPORT.md §8 (numbers since re-measured, see note above). |
| `03_think_prd_example.txt` | `sabi think` — the Corporate/Enterprise reasoning engine producing a one-page PRD for an offline crop-price advisory tool. This is the input to the cross-disciplinary demo below. |
| `04_agent_natural_phrasing_known_issue.txt` | `sabi agent` given natural phrasing ("create a file main.py that prints hello" — the *exact example from the agent's own system prompt*). At the time, the model replied with the code as plain text instead of the `write_file` tool call — reproduced twice, not a fluke. **Fixed 2026-08-25** (see note above). |
| `05_agent_explicit_tool_success.txt` | The same intent, phrased with an explicit tool-name cue ("use the write_file tool to create main.py..."). Completes a full four-step loop: write file → run it → list dir → read it back. |

## Why this matters for the video / live demo

The natural-phrasing gap documented in `04` is now fixed, so either phrasing
style should work for the demo video or Gate 2/3 live defense. Still worth a
quick live sanity check before recording rather than assuming — don't
discover a regression live on camera.

## Screenshots / video — still needed

These are text transcripts, not the images/video Gate 1 asks for. You still
need to:
1. Open a real terminal on your machine, run the same commands, and take actual
   screenshots (or a short screen recording) — text logs aren't a substitute.
2. Record the 2-minute demo video — see `demo/VIDEO_SCRIPT.md`.
