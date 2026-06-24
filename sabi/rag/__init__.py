"""Local Retrieval-Augmented Generation (RAG) layer.

A dependency-light, fully-offline retriever: documents are chunked, embedded
with a built-in hashing TF-IDF vectorizer (pure Python, optional NumPy
acceleration), stored in a local JSON vector store, and retrieved by cosine
similarity. If the GGUF model exposes embeddings, the retriever can use those
instead.
"""

from .embeddings import HashingEmbedder
from .vector_store import VectorStore
from .retriever import Retriever

__all__ = ["HashingEmbedder", "VectorStore", "Retriever"]
