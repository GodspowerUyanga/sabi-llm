# Sabi — 2-Minute Demo Video Script

Total: ~120 seconds. Record at 1080p. Show a RAM monitor (e.g. `htop`) in a
corner the whole time to make the "offline + low memory" claims visible.

---

**[0:00–0:15] Hook + problem**
> "This is Sabi — an AI assistant for African small businesses that runs entirely
> on this 8 GB laptop. No internet, no GPU, no cloud bill. Watch the network: I'm
> pulling the cable now."
*(Unplug Wi-Fi / show airplane mode. Keep it visible.)*

**[0:15–0:35] It runs offline, and it's grounded**
> "Sabi reads the company's own documents. Let me ask about our refund policy."
*(Type: "What's our refund policy?" — answer streams in, cites the handbook.)*
> "That came straight from our handbook — and notice the RAM gauge: we're well under
> the 7 GB ceiling."

**[0:35–0:60] The cross-disciplinary integration — exact numbers**
> "Small models are bad at maths, which is dangerous for money. Sabi doesn't guess —
> it computes."
*(Type: "What's our total Q1 revenue across all regions?" — show the `[used aggregate]`
trace, then the exact figure. Then: "If that grows 15%, what's Q2?" — show `[used calc]`.)*
> "It read the sales CSV, summed it with a tool, and did the growth maths exactly."

**[0:60–0:80] African language support**
> "And it speaks the languages our operators actually use."
*(Type in Pidgin: "Abeg, how many leave days person fit take?" — answer comes back in Pidgin.)*
> "English, Nigerian Pidgin, Swahili, Yoruba, Hausa — detected automatically."

**[0:80–1:00] The custom model + efficiency**
> "The model is Sabi-1 — a quantized open-source base, rebranded and specialised for
> enterprise work, grounded in your files and wired to tools."
*(Show terminal: `python -m sabi bench` running; point at peak RAM and Seff in the output.)*

**[1:00–1:50] Development journey**
> "We built for the constraint first: q4 quantization, memory-mapped weights, an
> embedder we load and release so it never inflates peak RAM, and a database-free
> NumPy index. The whole pipeline is testable without even downloading the model —
> twenty tests, all green — so reproducibility is trivial for the audit."
*(Show `pytest -q` passing; show the project structure briefly.)*

**[1:50–2:00] Close**
> "Sabi: AI that an African business can own, run offline, and trust. Thank you."

---

### Shot list / b-roll
- The RAM gauge in the Sabi sidebar and `htop` agreeing.
- The `<tool_call>` / `[used calc]` traces appearing inline.
- Cable unplugged / airplane mode icon.
- `benchmark_report.json` opened to show the telemetry.
- `pytest` green.
