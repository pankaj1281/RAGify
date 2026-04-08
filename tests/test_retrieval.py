"""Tests for the FAISS vector store and retriever."""

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.documents import Document

from ingestion.embedder import EmbeddingModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(text: str, source: str = "test.txt") -> Document:
    return Document(page_content=text, metadata={"source": source, "chunk_index": 0})


def _make_mock_embedder(dim: int = 384) -> MagicMock:
    """Return a mock EmbeddingModel that returns random unit vectors."""
    mock = MagicMock(spec=EmbeddingModel)
    mock.embedding_dim = dim

    def embed_texts(texts, **kwargs):
        vecs = np.random.randn(len(texts), dim).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs

    def embed_query(query, **kwargs):
        vec = np.random.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    mock.embed_texts.side_effect = embed_texts
    mock.embed_query.side_effect = embed_query
    return mock


# ---------------------------------------------------------------------------
# FAISSVectorStore tests
# ---------------------------------------------------------------------------


class TestFAISSVectorStore:
    """Unit tests for vectorstore/faiss_store.py."""

    @pytest.fixture
    def store(self, tmp_path: Path):
        """Return a fresh FAISSVectorStore with a mock embedder."""
        from vectorstore.faiss_store import FAISSVectorStore

        mock_embedder = _make_mock_embedder()
        return FAISSVectorStore(
            embedding_model=mock_embedder,
            index_path=str(tmp_path / "faiss"),
        )

    def test_add_and_count(self, store) -> None:
        """add_documents should increase document count."""
        docs = [_make_doc(f"Document {i}") for i in range(5)]
        store.add_documents(docs)
        assert store.document_count == 5

    def test_add_empty_raises(self, store) -> None:
        """add_documents should raise ValueError for empty list."""
        with pytest.raises(ValueError, match="empty"):
            store.add_documents([])

    def test_similarity_search_returns_results(self, store) -> None:
        """similarity_search should return (doc, score) tuples."""
        docs = [_make_doc(f"Text chunk {i}") for i in range(10)]
        store.add_documents(docs)

        results = store.similarity_search("some query", k=3)
        assert len(results) == 3
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    def test_similarity_search_empty_store_raises(self, store) -> None:
        """similarity_search should raise ValueError when store is empty."""
        with pytest.raises(ValueError, match="empty"):
            store.similarity_search("query")

    def test_similarity_search_blank_query_raises(self, store) -> None:
        """similarity_search should raise ValueError for blank query."""
        docs = [_make_doc("Some content")]
        store.add_documents(docs)
        with pytest.raises(ValueError):
            store.similarity_search("   ")

    def test_save_and_load(self, store, tmp_path: Path) -> None:
        """save and load should persist and restore the index."""
        docs = [_make_doc(f"Persistent doc {i}") for i in range(4)]
        store.add_documents(docs)
        store.save()

        # Create a new store and load from disk
        from vectorstore.faiss_store import FAISSVectorStore

        mock_embedder = _make_mock_embedder()
        store2 = FAISSVectorStore(
            embedding_model=mock_embedder,
            index_path=str(tmp_path / "faiss"),
        )
        loaded = store2.load()
        assert loaded is True
        assert store2.document_count == 4

    def test_load_nonexistent_returns_false(self, tmp_path: Path) -> None:
        """load should return False if no persisted index exists."""
        from vectorstore.faiss_store import FAISSVectorStore

        store = FAISSVectorStore(
            embedding_model=_make_mock_embedder(),
            index_path=str(tmp_path / "empty_faiss"),
        )
        assert store.load() is False

    def test_clear_resets_store(self, store) -> None:
        """clear should reset document count to zero."""
        store.add_documents([_make_doc("content")])
        assert store.document_count == 1
        store.clear()
        assert store.document_count == 0


# ---------------------------------------------------------------------------
# Retriever tests
# ---------------------------------------------------------------------------


class TestRetriever:
    """Unit tests for rag/retriever.py."""

    @pytest.fixture
    def retriever(self, tmp_path: Path):
        """Return a Retriever backed by a mock vector store."""
        from rag.retriever import Retriever
        from vectorstore.faiss_store import FAISSVectorStore

        mock_embedder = _make_mock_embedder()
        store = FAISSVectorStore(
            embedding_model=mock_embedder,
            index_path=str(tmp_path / "faiss"),
        )
        docs = [_make_doc(f"Chunk {i}", source="doc.txt") for i in range(10)]
        store.add_documents(docs)
        return Retriever(vector_store=store, top_k=3)

    def test_retrieve_returns_documents(self, retriever) -> None:
        """retrieve should return a list of Document objects."""
        docs = retriever.retrieve("test query")
        assert isinstance(docs, list)
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_retrieve_respects_k(self, retriever) -> None:
        """retrieve should respect the k parameter."""
        docs = retriever.retrieve("query", k=2)
        assert len(docs) == 2

    def test_retrieve_blank_query_raises(self, retriever) -> None:
        """retrieve should raise ValueError for a blank query."""
        with pytest.raises(ValueError):
            retriever.retrieve("   ")

    def test_retrieve_with_scores(self, retriever) -> None:
        """retrieve_with_scores should return (Document, float) tuples."""
        results = retriever.retrieve_with_scores("query")
        assert all(isinstance(score, float) for _, score in results)
