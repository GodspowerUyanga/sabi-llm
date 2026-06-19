#!/usr/bin/env python3
"""
Sabi local profiler — reproduces the ADTC telemetry on your own hardware.

It measures, on the ADTC Standard Laptop profile:
  - Speed:       tokens/second (feeds Sperf = 100 * TPSact / TPSmax)
  - Efficiency:  peak RAM vs the 7 GB budget (Seff = 100 * (7 - peak)/7)
  - Thermal:     core temperature; flags the -10 penalty if > 85 °C
  - OOM/crash:   verifies the run never exceeds the budget (else Stotal = 0)

Run it before submitting so there are no surprises in the official audit.

Usage:
    python scripts/benchmark.py
    python -m sabi bench
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sabi.config import load_config            # noqa: E402
from sabi.memory import measure_peak, current_rss_gb  # noqa: E402
from sabi.model import load_chat               # noqa: E402

DEFAULT_PROMPTS = [
    "Summarise our refund policy in two sentences.",
    "What is 0.15 * 52000000?",
    "List three risks from our 2026 growth strategy.",
    "Draft a one-line reminder to staff about submitting expense receipts.",
    "Abeg, how many days person fit take as annual leave?",
]


def read_temperature_c() -> float | None:
    """Best-effort CPU temperature in Celsius (Linux)."""
    try:
        import psutil
        temps = psutil.sensors_temperatures()
        for entries in temps.values():
            for e in entries:
                if e.current and e.current > 0:
                    return float(e.current)
    except Exception:
        pass
    # Fallback: /sys thermal zones
    try:
        for zone in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
            val = int(zone.read_text().strip())
            return val / 1000.0 if val > 1000 else float(val)
    except Exception:
        pass
    return None


def count_tokens(text: str) -> int:
    # Approximate if no tokenizer is available: ~4 chars/token is a safe proxy
    # for English; the audit uses the model's own tokenizer, but this is a close
    # local estimate for relative speed.
    return max(1, len(text) // 4)


def run_benchmark(config_path=None, prompts_path=None, output="benchmark_report.json"):
    cfg = load_config(config_path)
    budget = cfg.model.ram_budget_gb

    prompts = DEFAULT_PROMPTS
    if prompts_path and Path(prompts_path).exists():
        prompts = [l.strip() for l in Path(prompts_path).read_text().splitlines() if l.strip()]

    print("\n" + "=" * 58)
    print("  SABI LOCAL PROFILER  ·  ADTC Standard Laptop telemetry")
    print("=" * 58)

    chat = load_chat(cfg)
    is_mock = "mock" in getattr(chat, "name", "").lower()
    if is_mock:
        print("  ⚠ No GGUF present — running in MOCK mode (speeds are not")
        print("    representative). Download the model for real telemetry.\n")

    results = []
    max_temp = read_temperature_c() or 0.0
    oom = False

    with measure_peak(budget_gb=budget) as sampler:
        for i, prompt in enumerate(prompts, 1):
            messages = [{"role": "system", "content": "You are Sabi-1."},
                        {"role": "user", "content": prompt}]
            t0 = time.perf_counter()
            out = ""
            try:
                for delta in chat.chat(messages, temperature=cfg.model.temperature,
                                       top_p=cfg.model.top_p, max_tokens=cfg.model.max_tokens):
                    out += delta
            except MemoryError:
                oom = True
                break
            dt = time.perf_counter() - t0
            toks = count_tokens(out)
            tps = toks / dt if dt > 0 else 0.0
            t = read_temperature_c()
            if t:
                max_temp = max(max_temp, t)
            results.append({"prompt": prompt, "tokens": toks, "seconds": round(dt, 3),
                            "tps": round(tps, 2)})
            print(f"  [{i}/{len(prompts)}] {tps:6.1f} tok/s  ·  {dt:5.2f}s  ·  {prompt[:42]}")

    mem = sampler.report  # type: ignore[attr-defined]

    avg_tps = round(sum(r["tps"] for r in results) / len(results), 2) if results else 0.0
    thermal_penalty = 10 if max_temp and max_temp > 85 else 0
    disqualified = oom or not mem.within_budget

    report = {
        "model": getattr(chat, "name", "unknown"),
        "mock_mode": is_mock,
        "config": {"n_ctx": cfg.model.n_ctx, "n_threads": cfg.model.n_threads,
                   "n_batch": cfg.model.n_batch},
        "speed": {"avg_tps": avg_tps, "per_prompt": results},
        "efficiency": {"peak_ram_gb": mem.peak_gb, "budget_gb": budget,
                       "Seff": mem.efficiency_score, "within_budget": mem.within_budget},
        "thermal": {"max_temp_c": round(max_temp, 1) if max_temp else None,
                    "penalty": thermal_penalty},
        "oom_or_crash": oom,
        "disqualified": disqualified,
    }

    print("-" * 58)
    print(f"  Avg speed      : {avg_tps} tok/s")
    print(f"  Peak RAM       : {mem.peak_gb} / {budget} GB   (Seff = {mem.efficiency_score})")
    print(f"  Max temp       : {report['thermal']['max_temp_c']} °C   (penalty {thermal_penalty})")
    print(f"  Within budget  : {'YES' if mem.within_budget else 'NO — DISQUALIFIED'}")
    print("=" * 58)
    print(f"  Note: Sperf depends on TPSmax across ALL teams, so it cannot be")
    print(f"  computed locally. Efficiency and thermal are final here.\n")

    Path(output).write_text(json.dumps(report, indent=2))
    print(f"  Saved → {output}\n")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--output", default="benchmark_report.json")
    a = ap.parse_args()
    run_benchmark(a.config, a.prompts, a.output)
