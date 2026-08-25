# 2-minute demo video — script

Gate 1 asks for "your solution and development journey," not just a feature
tour. Structure below targets ~2:00 and leads with the judged criteria
(accuracy/speed/efficiency, cross-disciplinary integration, offline-ness).

Record with the terminal font size large enough to read on a shared screen.
Use `sabi run` (TUI) for the visual parts — it shows tokens/context/RAM in the
sidebar live, which is free evidence of the efficiency story. Use the phrasing
pattern from `demo/05_agent_explicit_tool_success.txt`, not `04`'s, for any
agent action you show live.

---

**0:00–0:15 — The problem (voice over a static slide or your face)**
"Cloud LLMs assume API budgets and stable fibre. SABI is a fully offline AI
coworker that runs entirely on the laptop already on millions of desks in
Africa — 8 GB RAM, no discrete GPU, under $500 — with zero cloud dependency."

**0:15–0:35 — Prove it's offline and under budget**
- Disconnect / show Wi-Fi off, or simply state it and don't touch the network.
- Run `sabi doctor` on screen: point at "Model size on disk: 1.80 GB" and
  "Est. runtime RAM: ~3.10 GB of 7.0 GB budget" — well under the 7 GB ceiling
  that causes automatic disqualification.

**0:35–1:10 — The cross-disciplinary integration (your differentiator)**
- `sabi think "Write a one-page PRD for an offline crop-price advisory tool
  for smallholder farmers."` — let it stream a few lines on screen, don't wait
  for the full thing.
- Cut to: `sabi run` (TUI) or `sabi agent --yes "use the write_file tool to
  create main.py with content ..."` scaffolding part of that same idea into
  real code, in the same tool, on the same machine.
- Voice over: "The same agent that just wrote a business plan can build the
  project that implements it — on-device, with your permission for every
  action outside the sandbox."

**1:10–1:35 — Speed + the permission model**
- Show a short code-generation exchange streaming (tokens/sec visible in the
  TUI sidebar).
- Show one permission prompt (Allow once / Allow always / Reject) to
  demonstrate the safety model — this is worth showing, judges care about
  trust as well as capability.

**1:35–1:55 — Benchmarks, honestly**
- `sabi benchmark` (or show `demo/02_benchmark_devmachine.json`) —
  "18.9 tokens/sec, 3.3 GB peak RAM, on this development machine; we're
  re-running on the ADTC Standard Laptop with the official profiler before
  Gate 2."
- One sentence on what's NOT claimed: "We don't yet claim the African-language
  bonus — it's not working end-to-end, so we're not claiming it."

**1:55–2:00 — Close**
"SABI — an AI coworker Africa can own, run, and trust, on the hardware it
already has." Repo URL on screen.

---

## Do NOT do live on camera
- Don't type the exact phrase from `demo/04_agent_natural_phrasing_known_issue.txt`
  ("create a file main.py that prints hello") — it's a confirmed failure case.
  Use `05`'s explicit-tool phrasing instead, or fix the underlying issue first
  (see REPORT.md §13).
- Don't wait out a full 45-second generation in real time — cut/speed up, or
  talk over it.
