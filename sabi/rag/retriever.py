"""Retriever: index local documents and fetch the most relevant chunks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Tuple

from .embeddings import HashingEmbedder
from .vector_store import VectorStore


def _chunk(text: str, size: int = 800, overlap: int = 120) -> List[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


class Retriever:
    def __init__(self, store: VectorStore, embedder: HashingEmbedder | None = None):
        self.store = store
        self.embedder = embedder or HashingEmbedder()

    # ------------------------------------------------------------- indexing
    def add_text(self, text: str, source: str = "inline") -> int:
        added = 0
        for chunk in _chunk(text):
            doc_id = hashlib.sha1((source + chunk).encode()).hexdigest()[:12]
            vector = self.embedder.embed(chunk)
            self.store.add(doc_id, chunk, vector, source)
            added += 1
        self.store.save()
        return added

    def add_file(self, path: Path) -> int:
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        return self.add_text(text, source=str(path))

    def index_directory(self, directory: Path, patterns=(".md", ".txt", ".py")) -> int:
        directory = Path(directory)
        total = 0
        for p in directory.rglob("*"):
            if p.is_file() and p.suffix in patterns:
                try:
                    total += self.add_file(p)
                except Exception:
                    continue
        return total

    # ------------------------------------------------------------- query
    def query(self, text: str, k: int = 3) -> List[Tuple[float, str, str]]:
        if not len(self.store):
            return []
        qv = self.embedder.embed(text)
        scored = []
        for rec in self.store.records:
            score = self.embedder.cosine(qv, rec["vector"])
            scored.append((score, rec["text"], rec.get("source", "")))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]

    def context(self, text: str, k: int = 3) -> str:
        hits = self.query(text, k=k)
        if not hits:
            return ""
        return "\n\n".join(f"[{src}] {chunk}" for _, chunk, src in hits)
