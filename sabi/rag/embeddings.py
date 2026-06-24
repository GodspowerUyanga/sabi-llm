"""Lightweight offline embedder.

Implements a hashing TF vectorizer with sublinear scaling. No external model
or network needed, which keeps the RAG layer usable on a bare install and well
inside the memory budget. NumPy is used when present for speed, with a pure
Python fallback.
"""

from __future__ import annotations

import math
import re
from typing import List

try:  # optional acceleration
    import numpy as _np

    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _np = None
    _HAS_NUMPY = False

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class HashingEmbedder:
    """Maps text to a fixed-size sparse-ish vector via feature hashing."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = tokenize(text)
        if not tokens:
            return vec
        counts: dict[int, float] = {}
        for tok in tokens:
            idx = (hash(tok) % self.dim + self.dim) % self.dim
            counts[idx] = counts.get(idx, 0.0) + 1.0
        # sublinear term frequency + L2 normalisation
        for idx, c in counts.items():
            vec[idx] = 1.0 + math.log(c)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def cosine(a: List[float], b: List[float]) -> float:
        if _HAS_NUMPY:
            va, vb = _np.asarray(a), _np.asarray(b)
            denom = (_np.linalg.norm(va) * _np.linalg.norm(vb)) or 1.0
            return float(va.dot(vb) / denom)
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)
