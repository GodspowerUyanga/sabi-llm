"""Runtime profiler.

Measures the telemetry that the ADTC 2026 scoring model cares about:
peak RAM (RSS), CPU load, and core temperature / thermal throttling. Uses
``psutil`` when available, with graceful degradation otherwise.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

try:
    import psutil  # type: ignore

    _HAS_PSUTIL = True
except Exception:  # pragma: no cover
    psutil = None  # type: ignore
    _HAS_PSUTIL = False


@dataclass
class Snapshot:
    rss_gb: float
    available_gb: float
    cpu_percent: float
    temperature_c: Optional[float]
    thermal_throttle: bool


def _process_rss_gb() -> float:
    if _HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    # Fallback: read VmRSS from /proc on Linux
    try:
        with open(f"/proc/{os.getpid()}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    kb = float(line.split()[1])
                    return kb / (1024 ** 2)
    except Exception:
        pass
    return 0.0


def _available_gb() -> float:
    if _HAS_PSUTIL:
        return psutil.virtual_memory().available / (1024 ** 3)
    return 0.0


def _temperature() -> Optional[float]:
    if not _HAS_PSUTIL or not hasattr(psutil, "sensors_temperatures"):
        return None
    try:
        temps = psutil.sensors_temperatures()
    except Exception:
        return None
    if not temps:
        return None
    readings = []
    for entries in temps.values():
        for entry in entries:
            if entry.current:
                readings.append(entry.current)
    return max(readings) if readings else None


def snapshot(thermal_ceiling_c: float = 85.0) -> Snapshot:
    temp = _temperature()
    return Snapshot(
        rss_gb=round(_process_rss_gb(), 3),
        available_gb=round(_available_gb(), 3),
        cpu_percent=psutil.cpu_percent(interval=0.1) if _HAS_PSUTIL else 0.0,
        temperature_c=round(temp, 1) if temp is not None else None,
        thermal_throttle=bool(temp is not None and temp > thermal_ceiling_c),
    )


class PeakTracker:
    """Sample RSS over the life of an operation to capture the peak."""

    def __init__(self):
        self.peak_rss_gb = 0.0
        self._start = time.perf_counter()

    def sample(self) -> float:
        rss = _process_rss_gb()
        self.peak_rss_gb = max(self.peak_rss_gb, rss)
        return rss

    def elapsed(self) -> float:
        return time.perf_counter() - self._start
