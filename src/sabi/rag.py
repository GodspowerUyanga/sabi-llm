"""
Retrieval-augmented generation over the local enterprise corpus.

Design choices (all in service of the 7 GB ceiling and reproducibility):

- Vector store is a single NumPy ``.npy`` matrix + a JSON sidecar of metadata.
  No database, no FAISS dependency. For an SME knowledge base (hundreds to a
  few thousand chunks) brute-force cosine search is fast and uses trivial RAM.
- Chunks are character-windowed with overlap; simple, language-agnostic, and
  robust for mixed business documents (.md, .txt, .csv).
- The embedder is loaded only while building or querying, then released.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import Embedder, cosine_sim, make_embedder
from .ingest import extract_text

SUPPORTED_SUFFIXES = {".md", ".txt", ".csv", ".pdf", ".docx", ".xlsx", ".xls"}


@dataclass
class Chunk:
    text: str
    source: str
    idx: int


def _read_text(path: Path) -> str:
    try:
        return extract_text(path)
    except Exception:
        return ""


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    step = max(1, size - overlap)
    while start < len(text):
        end = start + size
        # try to break on a paragraph/sentence boundary near the window edge
        window = text[start:end]
        if end < len(text):
            for sep in ("\n\n", "\n", ". ", " "):
                cut = window.rfind(sep)
                if cut > size * 0.5:
                    window = window[: cut + len(sep)]
                    break
        chunks.append(window.strip())
        start += max(step, len(window) - overlap) if len(window) > overlap else step
    return [c for c in chunks if c]


class RagIndex:
    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)
        self.vectors: np.ndarray | None = None
        self.chunks: list[Chunk] = []

    # ---------------------------------------------------------------- build
    def build(self, corpus_dir: str | Path, embedder: Embedder,
              chunk_size: int, overlap: int) -> int:
        corpus_dir = Path(corpus_dir)
        chunks: list[Chunk] = []
        for path in sorted(corpus_dir.rglob("*")):
            if path.suffix.lower() not in SUPPORTED_SUFFIXES or not path.is_file():
                continue
            raw = _read_text(path)
            for i, piece in enumerate(chunk_text(raw, chunk_size, overlap)):
                chunks.append(Chunk(text=piece, source=path.name, idx=i))

        if not chunks:
            self.vectors, self.chunks = np.zeros((0, embedder.dim), np.float32), []
            return 0

        vecs = embedder.embed([c.text for c in chunks])
        self.vectors, self.chunks = vecs.astype(np.float32), chunks
        self.save()
        return len(chunks)

    def add_file(self, path: str | Path, embedder: Embedder,
                 chunk_size: int, overlap: int) -> int:
        """Index a single file and append it to the store (incremental upload).

        If a file with the same name was indexed before, its old chunks are
        replaced. Returns the number of chunks added.
        """
        path = Path(path)
        name = path.name
        pieces = chunk_text(_read_text(path), chunk_size, overlap)
        new_chunks = [Chunk(text=p, source=name, idx=i) for i, p in enumerate(pieces)]
        if not new_chunks:
            return 0
        vecs = embedder.embed([c.text for c in new_chunks]).astype(np.float32)

        # Drop any previous chunks from the same source (re-upload / update).
        if self.vectors is not None and self.chunks:
            keep = [i for i, c in enumerate(self.chunks) if c.source != name]
            if len(keep) != len(self.chunks):
                self.vectors = self.vectors[keep] if keep else np.zeros((0, vecs.shape[1]), np.float32)
                self.chunks = [self.chunks[i] for i in keep]

        if self.vectors is None or len(self.chunks) == 0:
            self.vectors, self.chunks = vecs, list(new_chunks)
        else:
            if vecs.shape[1] != self.vectors.shape[1]:
                raise ValueError("embedding dimension mismatch — rebuild the index "
                                 "(python -m sabi index) after changing the embedder")
            self.vectors = np.vstack([self.vectors, vecs])
            self.chunks = self.chunks + new_chunks
        self.save()
        return len(new_chunks)

    # --------------------------------------------------------------- persist
    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.index_dir / "vectors.npy", self.vectors)
        meta = [{"text": c.text, "source": c.source, "idx": c.idx} for c in self.chunks]
        (self.index_dir / "chunks.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )

    def load(self) -> bool:
        vpath, cpath = self.index_dir / "vectors.npy", self.index_dir / "chunks.json"
        if not (vpath.exists() and cpath.exists()):
            return False
        self.vectors = np.load(vpath)
        meta = json.loads(cpath.read_text(encoding="utf-8"))
        self.chunks = [Chunk(**m) for m in meta]
        return True

    # ---------------------------------------------------------------- search
    def search(self, query_vec: np.ndarray, top_k: int, min_score: float) -> list[tuple[Chunk, float]]:
        if self.vectors is None or len(self.chunks) == 0:
            return []
        sims = cosine_sim(query_vec, self.vectors)
        order = np.argsort(-sims)[:top_k]
        results = [(self.chunks[i], float(sims[i])) for i in order if sims[i] >= min_score]
        return results

    @property
    def size(self) -> int:
        return len(self.chunks)


def format_context(results: list[tuple[Chunk, float]]) -> str:
    """Render retrieved chunks for injection into the system prompt."""
    if not results:
        return ""
    blocks = []
    for chunk, score in results:
        blocks.append(f"[source: {chunk.source}  (relevance {score:.2f})]\n{chunk.text}")
    return "\n\n".join(blocks)
