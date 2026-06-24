"""Environment diagnostics (`sabi doctor`).

Verifies that the host meets the requirements to run SABI and reports anything
that needs attention before the model can be used.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from typing import List

from .config import Config
from .model import LLMModel
from . import profiler


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def run_checks(config: Config) -> List[Check]:
    checks: List[Check] = []

    # Python version
    py_ok = sys.version_info >= (3, 9)
    checks.append(Check("Python >= 3.9", py_ok, platform.python_version()))

    # OS
    checks.append(Check("Operating system", True, f"{platform.system()} {platform.release()}"))

    # Optional dependencies
    for mod, label in (
        ("llama_cpp", "llama-cpp-python (inference)"),
        ("psutil", "psutil (telemetry)"),
        ("rich", "rich (UI)"),
        ("numpy", "numpy (RAG speed)"),
        ("yaml", "PyYAML (config)"),
        ("huggingface_hub", "huggingface_hub (model download)"),
    ):
        try:
            __import__(mod)
            checks.append(Check(label, True, "installed"))
        except Exception:
            optional = mod not in {"llama_cpp"}
            checks.append(
                Check(label, optional, "missing" + (" (optional)" if optional else " (required for inference)"))
            )

    # Model file + size vs the ADTC RAM budget
    model = LLMModel(config)
    mf = model.model_file
    if mf.exists():
        size_gb = mf.stat().st_size / (1024 ** 3)
        budget = config.ram_ceiling_gb
        # On disk the file must obviously be smaller than the RAM ceiling; at
        # runtime llama.cpp uses roughly file size + KV cache + buffers.
        under = size_gb < budget
        checks.append(Check("Model file", True, str(mf)))
        checks.append(Check(
            "Model size on disk", under,
            f"{size_gb:.2f} GB  (ADTC RAM budget: {budget:.1f} GB)"
            + ("" if under else "  — exceeds budget!"),
        ))
        # Rough runtime estimate: file + ~1.0-1.5 GB overhead for 4k context.
        est_runtime = size_gb + 1.3
        est_ok = est_runtime < budget
        checks.append(Check(
            "Est. runtime RAM", est_ok,
            f"~{est_runtime:.2f} GB of {budget:.1f} GB budget "
            f"({est_runtime / budget * 100:.0f}% of ceiling)"
            + ("" if est_ok else "  — likely too large, try a smaller quant/model"),
        ))
        checks.append(Check(
            "Headroom vs budget", est_ok,
            f"{budget - est_runtime:.2f} GB free under the {budget:.1f} GB ceiling"
            + ("  (tight)" if est_ok and (budget - est_runtime) < 1.0 else ""),
        ))
        checks.append(Check(
            "Measured peak RAM", True,
            "run `sabi benchmark` to measure actual peak RSS during inference",
        ))
    else:
        checks.append(Check("Model file", False, f"{mf}  (not downloaded — run `sabi download`)"))
        checks.append(Check(
            "Model size on disk", False,
            f"not downloaded  (ADTC RAM budget: {config.ram_ceiling_gb:.1f} GB)",
        ))

    # Memory headroom
    snap = profiler.snapshot(config.thermal_ceiling_c)
    if snap.available_gb:
        mem_ok = snap.available_gb >= 1.0
        checks.append(Check("Available RAM", mem_ok, f"{snap.available_gb:.2f} GB free"))
    else:
        checks.append(Check("Available RAM", True, "unknown (psutil not installed)"))

    # Workspace writable
    ws = config.abs_workspace()
    try:
        ws.mkdir(parents=True, exist_ok=True)
        probe = ws / ".sabi_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks.append(Check("Workspace writable", True, str(ws)))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("Workspace writable", False, str(exc)))

    return checks


def all_ok(checks: List[Check]) -> bool:
    return all(c.ok for c in checks)
