"""Tests for the offline RAG layer."""

from sabi.rag import HashingEmbedder, VectorStore, Retriever


def test_embedding_is_normalised():
    emb = HashingEmbedder(dim=128)
    v = emb.embed("hello world hello")
    norm = sum(x * x for x in v) ** 0.5
    assert abs(norm - 1.0) < 1e-6 or norm == 0.0


def test_retrieval_finds_relevant_chunk(tmp_path):
    store = VectorStore(tmp_path / "vs.json")
    retr = Retriever(store, HashingEmbedder(dim=256))
    retr.add_text("The capital of France is Paris.", source="geo")
    retr.add_text("Python is a programming language.", source="tech")
    hits = retr.query("Which city is the capital of France?", k=1)
    assert hits
    assert "paris" in hits[0][1].lower()


def test_empty_store_returns_nothing(tmp_path):
    store = VectorStore(tmp_path / "vs.json")
    retr = Retriever(store, HashingEmbedder())
    assert retr.query("anything") == []
