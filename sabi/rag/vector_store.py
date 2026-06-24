"""A minimal JSON-backed vector store for the local RAG layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class VectorStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.records: List[Dict] = []
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                self.records = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.records = []
        else:
            self.records = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records), encoding="utf-8")

    def add(self, doc_id: str, text: str, vector: List[float], source: str = "") -> None:
        self.records.append({"id": doc_id, "text": text, "vector": vector, "source": source})

    def clear(self) -> None:
        self.records = []
        self.save()

    def __len__(self) -> int:
        return len(self.records)
