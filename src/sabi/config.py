"""
Central configuration for Sabi.

All tunable parameters live here and are loaded from ``config/sabi.yaml``.
Keeping configuration in one place is part of the reproducibility story for the
ADTC audit: a reviewer can read this file and know exactly how the model is run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

# Project root = three levels up from this file (src/sabi/config.py -> project root)
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelConfig:
    """Inference settings for the Sabi-1 chat model (llama.cpp / GGUF)."""

    # Display name used everywhere in the UI and in the model's self-identity.
    name: str = "Sabi-1"
    # Path to the quantised GGUF chat model on disk.
    path: str = "models/sabi-1.gguf"
    # Context window. Kept modest to protect the 7 GB RAM ceiling.
    n_ctx: int = 8192
    # CPU threads. 0 => auto-detect at runtime.
    n_threads: int = 0
    # Prompt batch size. Smaller = lower peak RAM, slightly slower prefill.
    n_batch: int = 256
    # Memory-map the model file instead of loading it into RAM. Critical for
    # keeping resident memory low on an 8 GB machine.
    use_mmap: bool = True
    # Do NOT lock pages into RAM (mlock would inflate RSS).
    use_mlock: bool = False
    # Sampling defaults.
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 640
    repeat_penalty: float = 1.18  # curbs repetition/runaway loops on small models
    # No GPU on the ADTC Standard Laptop.
    n_gpu_layers: int = 0
    # Hard safety valve: refuse to run if measured RSS would blow the budget.
    ram_budget_gb: float = 7.0


@dataclass
class EmbeddingConfig:
    """Settings for the local embedding model used by the RAG pipeline."""

    name: str = "bge-small-en-v1.5"
    path: str = "models/embedding.gguf"
    n_ctx: int = 512
    # Embeddings are computed at index time and briefly at query time, then the
    # embedder can be released so it does not count against peak RAM.
    lazy: bool = True


@dataclass
class RagConfig:
    """Retrieval-augmented generation over the local enterprise corpus."""

    corpus_dir: str = "data/corpus"
    index_dir: str = "data/index"
    chunk_size: int = 700        # characters per chunk (approx.)
    chunk_overlap: int = 120
    top_k: int = 4               # chunks injected into context
    min_score: float = 0.35      # cosine similarity floor
    enabled: bool = True


@dataclass
class AgentConfig:
    """Local tool-using agent loop (privacy-focused, fully offline)."""

    max_steps: int = 4           # max tool round-trips per user turn
    enable_tools: bool = True


@dataclass
class LanguageConfig:
    """African-language support (ADTC Alpha Bonus)."""

    # Default reply language when none is detected.
    default: str = "en"
    # Languages Sabi is explicitly tuned to handle.
    supported: list[str] = field(
        default_factory=lambda: ["en", "pcm"]
    )
    # Auto-detect the user's language and reply in kind.
    auto_detect: bool = True


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    # ---- path helpers (resolve relative paths against the project root) ----
    def abs_path(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (ROOT / path)

    @property
    def model_path(self) -> Path:
        return self.abs_path(self.model.path)

    @property
    def embedding_path(self) -> Path:
        return self.abs_path(self.embedding.path)

    @property
    def corpus_dir(self) -> Path:
        return self.abs_path(self.rag.corpus_dir)

    @property
    def index_dir(self) -> Path:
        return self.abs_path(self.rag.index_dir)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(dc: Any, data: dict[str, Any]) -> None:
    """Recursively overlay a plain dict onto a dataclass instance."""
    for key, value in (data or {}).items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(dc, key, value)


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load configuration, overlaying ``config/sabi.yaml`` if present.

    Environment variable ``SABI_CONFIG`` can override the path.
    """
    cfg = Config()
    cfg_path = Path(path or os.environ.get("SABI_CONFIG", ROOT / "config" / "sabi.yaml"))
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        _merge(cfg, data)
    # Resolve auto thread count.
    if cfg.model.n_threads == 0:
        cfg.model.n_threads = max(1, (os.cpu_count() or 4))
    return cfg
