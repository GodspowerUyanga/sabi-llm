"""
Memory monitoring for Sabi.

The ADTC scoring model rewards low peak RAM (Efficiency, 20%) and disqualifies
any run that exceeds the 7 GB budget (OOM -> Stotal = 0). This module gives us:

- a live reading of process resident memory (RSS),
- a background sampler that records *peak* RSS during a workload,
- a hard guard that can refuse to proceed if memory is dangerously high.

We measure RSS of this process plus its children (llama.cpp may spawn threads
within the same process, but we include children to be safe).
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

try:
    import psutil
except Exception:  # pragma: no cover - psutil is a hard dependency at runtime
    psutil = None  # type: ignore

_GB = 1024 ** 3


def _rss_bytes() -> int:
    """Resident memory of this process + children, in bytes."""
    if psutil is None:
        return 0
    proc = psutil.Process()
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except Exception:
            pass
    return total


def current_rss_gb() -> float:
    return _rss_bytes() / _GB


def available_gb() -> float:
    if psutil is None:
        return 0.0
    return psutil.virtual_memory().available / _GB


@dataclass
class MemoryReport:
    peak_gb: float
    budget_gb: float

    @property
    def within_budget(self) -> bool:
        return self.peak_gb <= self.budget_gb

    @property
    def efficiency_score(self) -> float:
        """ADTC Seff = 100 * (budget - peak) / budget, floored at 0."""
        score = 100.0 * (self.budget_gb - self.peak_gb) / self.budget_gb
        return max(0.0, round(score, 2))


class PeakSampler:
    """Background thread that records peak RSS while active."""

    def __init__(self, interval: float = 0.1, budget_gb: float = 7.0):
        self.interval = interval
        self.budget_gb = budget_gb
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, _rss_bytes())
            time.sleep(self.interval)

    def start(self) -> "PeakSampler":
        self.peak_bytes = _rss_bytes()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> MemoryReport:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self.peak_bytes = max(self.peak_bytes, _rss_bytes())
        return MemoryReport(peak_gb=round(self.peak_bytes / _GB, 3), budget_gb=self.budget_gb)


@contextmanager
def measure_peak(budget_gb: float = 7.0, interval: float = 0.1):
    """Context manager yielding a sampler; returns a MemoryReport on exit.

    Usage:
        with measure_peak(7.0) as sampler:
            ...run workload...
        report = sampler.report
    """
    sampler = PeakSampler(interval=interval, budget_gb=budget_gb).start()
    try:
        yield sampler
    finally:
        sampler.report = sampler.stop()  # type: ignore[attr-defined]


def guard_budget(budget_gb: float, headroom_gb: float = 0.3) -> None:
    """Raise MemoryError if current RSS is within *headroom* of the budget."""
    rss = current_rss_gb()
    if rss > (budget_gb - headroom_gb):
        raise MemoryError(
            f"RSS {rss:.2f} GB is too close to the {budget_gb} GB budget. "
            "Reduce n_ctx / n_batch or use a smaller model."
        )
