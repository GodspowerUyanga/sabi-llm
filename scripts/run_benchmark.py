#!/usr/bin/env python3
"""Run the SABI benchmark and write a JSON + Markdown report.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --limit 3 --out benchmarks/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sabi.config import load_config       # noqa: E402
from sabi import benchmark as bench        # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SABI benchmark.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=str(ROOT / "benchmarks" / "report.json"))
    args = parser.parse_args()

    cfg = load_config(root=ROOT)
    report = bench.run(cfg, limit=args.limit)
    data = report.to_dict()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")

    md = out.with_suffix(".md")
    lines = [
        "# SABI Benchmark Report", "",
        f"- Model available: **{data['model_available']}**",
        f"- Average accuracy: **{data['avg_accuracy']:.1%}**",
        f"- Average speed: **{data['avg_tps']:.2f} tok/s**",
        f"- Peak RAM: **{data['peak_rss_gb']:.3f} GB** (budget {cfg.ram_ceiling_gb} GB)",
        f"- Efficiency score: **{data['efficiency_score']:.1f} / 100**",
        f"- Thermal throttle: **{data['thermal_throttle']}**", "",
        "## Cases", "",
        "| Case | Intent | tok/s | Accuracy | OK |",
        "|------|--------|------:|---------:|----|",
    ]
    for c in data["cases"]:
        lines.append(
            f"| {c['name']} | {c['intent']} | {c['tps']:.1f} | "
            f"{c['accuracy']:.0%} | {'yes' if c['ok'] else 'no'} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {out} and {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
