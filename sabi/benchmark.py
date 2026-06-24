"""Benchmark harness.

Runs the local benchmark prompt set and reports the three ADTC 2026 scoring
dimensions: accuracy (keyword-coverage proxy), speed (tokens/sec) and
efficiency (peak RAM vs the 7 GB budget), plus a thermal check.

When the model is unavailable, the harness still runs and reports the
methodology with N/A figures, so the command never crashes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .config import Config
from .model import LLMModel, ModelUnavailable
from . import profiler


@dataclass
class CaseResult:
    name: str
    intent: str
    tps: float
    accuracy: float
    elapsed_s: float
    ok: bool
    note: str = ""


@dataclass
class BenchmarkReport:
    cases: List[CaseResult] = field(default_factory=list)
    peak_rss_gb: float = 0.0
    avg_tps: float = 0.0
    avg_accuracy: float = 0.0
    efficiency_score: float = 0.0
    thermal_throttle: bool = False
    model_available: bool = False

    def to_dict(self):
        return {
            "model_available": self.model_available,
            "avg_tps": round(self.avg_tps, 2),
            "avg_accuracy": round(self.avg_accuracy, 3),
            "peak_rss_gb": round(self.peak_rss_gb, 3),
            "efficiency_score": round(self.efficiency_score, 1),
            "thermal_throttle": self.thermal_throttle,
            "cases": [c.__dict__ for c in self.cases],
        }


def _accuracy(expected_keywords: List[str], output: str) -> float:
    if not expected_keywords:
        return 1.0
    low = output.lower()
    hit = sum(1 for kw in expected_keywords if kw.lower() in low)
    return hit / len(expected_keywords)


def load_cases(path: Path) -> List[dict]:
    cases = []
    if not path.exists():
        return cases
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                cases.append(json.loads(line))
            except Exception:
                continue
    return cases


def run(config: Config, prompts_file: Optional[Path] = None, limit: Optional[int] = None) -> BenchmarkReport:
    prompts_file = prompts_file or (config.root / "benchmarks" / "prompts.jsonl")
    cases = load_cases(prompts_file)
    if limit:
        cases = cases[:limit]

    model = LLMModel(config)
    report = BenchmarkReport(model_available=model.is_available())
    tracker = profiler.PeakTracker()

    tps_list, acc_list = [], []
    for case in cases:
        name = case.get("name", "case")
        intent = case.get("intent", "CHAT")
        prompt = case.get("prompt", "")
        keywords = case.get("expected_keywords", [])

        tracker.sample()
        if not report.model_available:
            report.cases.append(
                CaseResult(name, intent, 0.0, 0.0, 0.0, False, "model unavailable")
            )
            continue
        try:
            gen = model.generate(prompt, max_tokens=config.max_tokens)
            tracker.sample()
            acc = _accuracy(keywords, gen.text)
            tps_list.append(gen.tokens_per_second)
            acc_list.append(acc)
            report.cases.append(
                CaseResult(name, intent, round(gen.tokens_per_second, 2),
                           round(acc, 3), round(gen.elapsed_s, 2), True)
            )
        except ModelUnavailable as exc:
            report.cases.append(CaseResult(name, intent, 0.0, 0.0, 0.0, False, str(exc)))

    snap = profiler.snapshot(config.thermal_ceiling_c)
    report.peak_rss_gb = max(tracker.peak_rss_gb, snap.rss_gb)
    report.thermal_throttle = snap.thermal_throttle
    report.avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0
    report.avg_accuracy = sum(acc_list) / len(acc_list) if acc_list else 0.0
    budget = config.ram_ceiling_gb
    report.efficiency_score = max(0.0, 100.0 * (budget - report.peak_rss_gb) / budget)
    return report
