"""Configuration management for SABI.

Loads settings from (in order of precedence):
  1. Environment variables (SABI_*)
  2. A user config file (config/default.yaml or path passed in)
  3. Built-in defaults

Designed to work even when PyYAML is not installed, by falling back to the
built-in defaults so that ``sabi`` always starts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict

# PyYAML is optional. If missing, we silently use built-in defaults.
try:  # pragma: no cover - trivial import guard
    import yaml  # type: ignore

    _HAS_YAML = True
except Exception:  # pragma: no cover
    yaml = None  # type: ignore
    _HAS_YAML = False


# Project root = the directory that contains the `sabi` package. Used only to
# locate read-only assets shipped WITH the package (prompts/, config/) — this
# is site-packages/ for a real `pip install sabi-llm`, which is where those
# ship (verified against a real wheel build/install, not just the source tree).
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

# User data root for anything SABI writes on its own (downloaded model,
# internal scratch workspace) — deliberately NOT under PROJECT_ROOT. A real
# install's site-packages is often read-only for a normal user (system-wide
# installs) and is the wrong place for a multi-GB model file regardless: it's
# shared user state, not part of the package, so it belongs where every other
# CLI tool puts it — the user's home directory — and survives a reinstall.
USER_DATA_ROOT = Path.home() / ".sabi"


DEFAULTS: Dict[str, Any] = {
    # --- Model ---
    "model_path": "models/sabi-v1.Q4_K_M.gguf",
    "hf_repo_id": "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
    "hf_filename": "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
    "hf_revision": "main",
    "context_length": 4096,
    "max_tokens": 1024,
    "temperature": 0.4,
    "top_p": 0.9,
    "n_threads": 0,          # 0 = auto (use all physical cores)
    "n_gpu_layers": 0,       # 0 = CPU-only (challenge target has no discrete GPU)
    # --- Runtime ---
    "workspace_dir": "sabi_workspace",
    "prompts_dir": "prompts",
    "memory_file": ".sabi/memory.json",
    "vector_store_file": ".sabi/vector_store.json",
    # --- Limits (ADTC 2026 hardware ceiling) ---
    "ram_ceiling_gb": 7.0,
    "ram_target_gb": 6.5,
    "thermal_ceiling_c": 85.0,
    # --- UX ---
    "ui": "terminal",        # "terminal" or "web"
    "language": "en",        # en, yo (Yoruba), ha (Hausa), ig (Igbo)
    "verbose": False,
    # --- sabi-yoruba-tts (African Alpha Bonus): NLLB-200-distilled-600M,
    # int8 CTranslate2. See scripts/download_yoruba_model.py and sabi/translate.py.
    "yoruba_enabled": True,
    "yoruba_model_path": "models/sabi-yoruba-tts",
}


def _coerce(value: str) -> Any:
    """Coerce an environment-variable string into bool/int/float when possible."""
    low = value.lower()
    if low in {"true", "yes", "1", "on"}:
        return True
    if low in {"false", "no", "0", "off"}:
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


@dataclass
class Config:
    """Resolved runtime configuration."""

    model_path: str = DEFAULTS["model_path"]
    hf_repo_id: str = DEFAULTS["hf_repo_id"]
    hf_filename: str = DEFAULTS["hf_filename"]
    hf_revision: str = DEFAULTS["hf_revision"]
    context_length: int = DEFAULTS["context_length"]
    max_tokens: int = DEFAULTS["max_tokens"]
    temperature: float = DEFAULTS["temperature"]
    top_p: float = DEFAULTS["top_p"]
    n_threads: int = DEFAULTS["n_threads"]
    n_gpu_layers: int = DEFAULTS["n_gpu_layers"]
    workspace_dir: str = DEFAULTS["workspace_dir"]
    prompts_dir: str = DEFAULTS["prompts_dir"]
    memory_file: str = DEFAULTS["memory_file"]
    vector_store_file: str = DEFAULTS["vector_store_file"]
    ram_ceiling_gb: float = DEFAULTS["ram_ceiling_gb"]
    ram_target_gb: float = DEFAULTS["ram_target_gb"]
    thermal_ceiling_c: float = DEFAULTS["thermal_ceiling_c"]
    ui: str = DEFAULTS["ui"]
    language: str = DEFAULTS["language"]
    verbose: bool = DEFAULTS["verbose"]
    yoruba_enabled: bool = DEFAULTS["yoruba_enabled"]
    yoruba_model_path: str = DEFAULTS["yoruba_model_path"]

    # Absolute root the runtime operates from (set at load time).
    root: Path = field(default_factory=lambda: PROJECT_ROOT)

    # ----- Derived absolute paths -----
    def abs_model_path(self) -> Path:
        # The ADTC 2026 submission contract (metadata.json's _runtime.model_path,
        # download_model.sh) hardcodes "models/<file>" relative to the repo —
        # the audit harness checks that exact path. If it's already there
        # (repo checkout + download_model.sh), use it as-is so the audited
        # flow never changes. Otherwise this is a real end-user install with
        # no repo at all, so fall back to the home-anchored user data dir.
        legacy = self._resolve(self.model_path)
        if legacy.exists():
            return legacy
        return self._resolve_data(self.model_path)

    def abs_workspace(self) -> Path:
        # User data (scratch/sandbox): same reasoning as the model.
        return self._resolve_data(self.workspace_dir)

    def abs_prompts(self) -> Path:
        # Read-only asset shipped WITH the package: install-anchored.
        return self._resolve(self.prompts_dir)

    def abs_memory(self) -> Path:
        return self.abs_workspace() / self.memory_file

    def abs_vector_store(self) -> Path:
        return self.abs_workspace() / self.vector_store_file

    def abs_yoruba_model_path(self) -> Path:
        return self._resolve(self.yoruba_model_path)

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (self.root / path)

    def _resolve_data(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (USER_DATA_ROOT / path)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["root"] = str(self.root)
        return d


def load_config(path: str | os.PathLike | None = None, root: Path | None = None) -> Config:
    """Build a :class:`Config` from defaults, an optional YAML file, and env vars."""
    data: Dict[str, Any] = dict(DEFAULTS)

    root = Path(root).resolve() if root else PROJECT_ROOT

    # 1) YAML file (if present and PyYAML is available)
    cfg_path = Path(path) if path else (root / "config" / "default.yaml")
    if _HAS_YAML and cfg_path.exists():
        try:
            loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                data.update({k: v for k, v in loaded.items() if k in DEFAULTS})
        except Exception:
            pass  # malformed config should never block startup

    # 2) Environment overrides (SABI_MODEL_PATH, SABI_TEMPERATURE, ...)
    for key in DEFAULTS:
        env_key = "SABI_" + key.upper()
        if env_key in os.environ:
            data[key] = _coerce(os.environ[env_key])

    data["root"] = root
    return Config(**data)
