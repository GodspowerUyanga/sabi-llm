"""
Embedding backends for the RAG pipeline.

Two implementations behind one interface:

- ``LlamaEmbedder``  : real, offline embeddings via a small GGUF model
                       (default bge-small-en-v1.5) using llama-cpp-python.
- ``MockEmbedder``   : deterministic, dependency-free hashing embeddings used
                       for tests and CI so the whole pipeline is verifiable
                       without downloading any model.

The embedder is created lazily and can be released after indexing so it does
not inflate peak RAM during the chat workload that the audit measures.
"""
from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return vec / norm


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between query vector *a* (d,) and matrix *b* (n, d)."""
    a = normalize(a.reshape(1, -1))
    b = normalize(b)
    return (b @ a.T).ravel()


class Embedder(Protocol):
    dim: int
    def embed(self, texts: list[str]) -> np.ndarray: ...
    def close(self) -> None: ...


class MockEmbedder:
    """Local lexical embedder — hashed char n-grams + word unigrams.

    No external model required. This is the offline fallback used when no
    neural embedding GGUF is present. Char n-grams make it robust to short
    queries, morphology, and the mixed business vocabulary in SME documents —
    meaningfully better than plain word hashing, while staying dependency-free
    and using negligible RAM. (The neural LlamaEmbedder is preferred when the
    embedding.gguf model is available.)
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    @staticmethod
    def _tokens(text: str) -> list[str]:
        clean = "".join(c.lower() if c.isalnum() else " " for c in text)
        words = clean.split()
        grams: list[str] = list(words)
        for w in words:
            padded = f"#{w}#"
            for n in (3, 4):
                grams += [padded[i:i + n] for i in range(len(padded) - n + 1)]
        return grams

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in self._tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._embed_one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def close(self) -> None:  # nothing to release
        pass


class LlamaEmbedder:
    """Real offline embeddings via llama-cpp-python and a small GGUF model."""

    def __init__(self, model_path: str, n_ctx: int = 512, n_threads: int | None = None):
        from llama_cpp import Llama  # imported lazily so tests don't require it

        self._llama = Llama(
            model_path=str(model_path),
            embedding=True,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
        # Probe dimensionality once.
        probe = self._llama.create_embedding("dimension probe")
        self.dim = len(self._embedding_vector(probe))

    @staticmethod
    def _embedding_vector(resp) -> list[float]:
        emb = resp["data"][0]["embedding"]
        # llama.cpp may return token-level embeddings (list of lists); mean-pool.
        if emb and isinstance(emb[0], list):
            arr = np.asarray(emb, dtype=np.float32)
            return arr.mean(axis=0).tolist()
        return emb

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            resp = self._llama.create_embedding(t)
            out.append(self._embedding_vector(resp))
        return np.asarray(out, dtype=np.float32) if out else np.zeros((0, self.dim), np.float32)

    def close(self) -> None:
        try:
            del self._llama
        except Exception:
            pass


def make_embedder(model_path: str | None, n_ctx: int = 512, n_threads: int | None = None) -> Embedder:
    """Return a LlamaEmbedder if the GGUF exists, else a MockEmbedder."""
    from pathlib import Path

    if model_path and Path(model_path).exists():
        try:
            return LlamaEmbedder(model_path, n_ctx=n_ctx, n_threads=n_threads)
        except Exception as exc:  # pragma: no cover
            print(f"[sabi] Falling back to MockEmbedder ({exc})")
    return MockEmbedder()
