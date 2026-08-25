"""Quantized GGUF model wrapper.

SABI runs a single quantized GGUF model (e.g. Qwen2.5-Coder 3B Instruct, Q4_K_M)
through llama.cpp. The same model is prompted to behave as THINK or CODE
depending on how the router classifies the task.

The wrapper is defensive: if ``llama-cpp-python`` is not installed or the model
file has not been downloaded yet, SABI still starts. Generation then raises
:class:`ModelUnavailable` with actionable guidance instead of crashing, so the
router, tools, memory, RAG and project scanner all remain usable offline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .config import Config


class ModelUnavailable(RuntimeError):
    """Raised when generation is requested but no usable model is loaded."""


@dataclass
class Generation:
    """Result of a single generation, with timing telemetry."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_s <= 0:
            return 0.0
        return self.completion_tokens / self.elapsed_s


def _llama_available() -> bool:
    try:
        import llama_cpp  # noqa: F401

        return True
    except Exception:
        return False


def _auto_threads() -> int:
    """Physical core count, not logical/hyperthread count.

    llama.cpp's own "auto" (passing n_threads=None) defaults to
    logical_cpu_count // 2, which underutilizes a machine with hybrid
    P/E cores or hyperthreading — e.g. 11 threads on a 22-logical/16-physical
    core CPU, measurably slower than using all 16 physical cores for this
    compute-bound (matrix-multiply-heavy) workload, where hyperthreads add
    little since both threads on a core share the same execution units.
    """
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:
        pass
    import os
    return max(os.cpu_count() or 4, 1)


class LLMModel:
    """Lazy wrapper around a llama.cpp model."""

    def __init__(self, config: Config):
        self.config = config
        self._llm = None
        self._loaded = False
        self._load_error: Optional[str] = None

    # --------------------------------------------------------------- status
    @property
    def model_file(self) -> Path:
        return self.config.abs_model_path()

    def is_available(self) -> bool:
        """True if llama.cpp is importable AND the model file exists."""
        return _llama_available() and self.model_file.exists()

    def status(self) -> str:
        if not _llama_available():
            return "llama-cpp-python not installed"
        if not self.model_file.exists():
            return f"model file missing: {self.model_file}"
        if self._loaded:
            return "loaded"
        return "ready (not yet loaded)"

    # ----------------------------------------------------------------- load
    def load(self) -> bool:
        """Load the model into memory. Returns True on success."""
        if self._loaded:
            return True
        if not _llama_available():
            self._load_error = (
                "llama-cpp-python is not installed. Install it with:\n"
                "    pip install llama-cpp-python"
            )
            return False
        if not self.model_file.exists():
            self._load_error = (
                f"Model file not found at {self.model_file}.\n"
                "Download it from Hugging Face with:\n"
                "    sabi download"
            )
            return False

        from llama_cpp import Llama  # local import keeps startup fast

        n_threads = self.config.n_threads or _auto_threads()
        self._llm = Llama(
            model_path=str(self.model_file),
            n_ctx=self.config.context_length,
            n_threads=n_threads,
            n_batch=self.config.n_batch,
            n_gpu_layers=self.config.n_gpu_layers,
            verbose=self.config.verbose,
        )
        self._loaded = True
        self._load_error = None
        return True

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    # ------------------------------------------------------------- generate
    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> Generation:
        """Generate a completion. Raises :class:`ModelUnavailable` if no model."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, max_tokens=max_tokens, temperature=temperature, stop=stop)

    def chat(
        self,
        messages: List[dict],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        json_mode: bool = False,
    ) -> Generation:
        """Multi-turn chat completion over a list of role/content messages.

        ``json_mode`` forces the backend's JSON-object grammar so the reply is
        always syntactically valid JSON (used by the agent loop, where a small
        model otherwise tends to narrate in prose instead of emitting a tool
        call — see AgentLoop).
        """
        if not self._loaded and not self.load():
            raise ModelUnavailable(self._load_error or "model not available")

        t0 = time.perf_counter()
        kwargs = dict(
            messages=messages,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature if temperature is None else temperature,
            top_p=self.config.top_p,
            repeat_penalty=self.config.repeat_penalty,
            stop=stop or [],
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        result = self._llm.create_chat_completion(**kwargs)
        elapsed = time.perf_counter() - t0

        choice = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        return Generation(
            text=(choice or "").strip(),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            elapsed_s=elapsed,
        )

    def chat_stream(
        self,
        messages: List[dict],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        json_mode: bool = False,
    ):
        """Yield text deltas as they are generated (for a live, fast feel).

        ``json_mode`` grammar-constrains the stream to a valid JSON object,
        same as :meth:`chat` — the constraint applies per-token, so streaming
        still works; used by AgentLoop to stream a restricted turn's final
        answer live while it's still inside a ``{"final": "..."}`` wrapper.
        """
        if not self._loaded and not self.load():
            raise ModelUnavailable(self._load_error or "model not available")
        kwargs = dict(
            messages=messages,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature if temperature is None else temperature,
            top_p=self.config.top_p,
            repeat_penalty=self.config.repeat_penalty,
            stop=stop or [],
            stream=True,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        stream = self._llm.create_chat_completion(**kwargs)
        for chunk in stream:
            try:
                delta = chunk["choices"][0]["delta"].get("content")
            except Exception:
                delta = None
            if delta:
                yield delta

    def count_tokens(self, text: str) -> int:
        """Best-effort token count for the given text."""
        if self._loaded and self._llm is not None:
            try:
                return len(self._llm.tokenize(text.encode("utf-8")))
            except Exception:
                pass
        return max(1, len(text) // 4)

    def embed(self, text: str) -> Optional[List[float]]:
        """Return an embedding vector if the backend supports it, else None."""
        if not self._loaded and not self.load():
            return None
        try:
            return self._llm.embed(text)
        except Exception:
            return None
