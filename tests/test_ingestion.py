"""Tests for the ingestion pipeline (loader, chunking, embedder)."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


class TestLoader:
    """Unit tests for ingestion/loader.py."""

    def test_load_txt_file(self, tmp_path: Path) -> None:
        """load_document should parse a plain-text file."""
        txt_file = tmp_path / "sample.txt"
        txt_file.write_text("Hello, this is a test document.\nSecond line.", encoding="utf-8")

        from ingestion.loader import load_document

        docs = load_document(str(txt_file))
        assert len(docs) >= 1
        assert "Hello" in docs[0].page_content
        assert docs[0].metadata["source"] == "sample.txt"

    def test_load_nonexistent_file_raises(self) -> None:
        """load_document should raise FileNotFoundError for missing files."""
        from ingestion.loader import load_document

        with pytest.raises(FileNotFoundError):
            load_document("/nonexistent/path/file.txt")

    def test_load_unsupported_format_raises(self, tmp_path: Path) -> None:
        """load_document should raise ValueError for unsupported extensions."""
        bad_file = tmp_path / "file.csv"
        bad_file.write_text("a,b,c")

        from ingestion.loader import load_document

        with pytest.raises(ValueError, match="Unsupported file format"):
            load_document(str(bad_file))

    def test_load_directory_empty(self, tmp_path: Path) -> None:
        """load_directory should return an empty list for an empty directory."""
        from ingestion.loader import load_directory

        docs = load_directory(str(tmp_path))
        assert docs == []

    def test_load_directory_missing_raises(self) -> None:
        """load_directory should raise FileNotFoundError for missing dirs."""
        from ingestion.loader import load_directory

        with pytest.raises(FileNotFoundError):
            load_directory("/nonexistent/dir")

    def test_preprocess_text(self) -> None:
        """_preprocess_text should collapse whitespace."""
        from ingestion.loader import _preprocess_text

        raw = "Hello   world\n\n\n\nNext paragraph"
        cleaned = _preprocess_text(raw)
        assert "   " not in cleaned
        assert "\n\n\n" not in cleaned


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------


class TestChunking:
    """Unit tests for ingestion/chunking.py."""

    def _make_doc(self, text: str, source: str = "test.txt") -> Document:
        return Document(page_content=text, metadata={"source": source})

    def test_basic_chunking(self) -> None:
        """chunk_documents should split a long document into multiple chunks."""
        from ingestion.chunking import chunk_documents

        long_text = "This is a sentence. " * 200  # ~4000 chars
        docs = [self._make_doc(long_text)]
        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) <= 600  # some tolerance for splitter

    def test_metadata_preserved(self) -> None:
        """chunk_documents should preserve source metadata."""
        from ingestion.chunking import chunk_documents

        docs = [self._make_doc("Short text.", source="myfile.pdf")]
        chunks = chunk_documents(docs)

        assert all(c.metadata["source"] == "myfile.pdf" for c in chunks)

    def test_chunk_index_added(self) -> None:
        """chunk_documents should add a chunk_index metadata field."""
        from ingestion.chunking import chunk_documents

        text = "Word " * 300
        docs = [self._make_doc(text)]
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)

        assert all("chunk_index" in c.metadata for c in chunks)

    def test_invalid_chunk_size_raises(self) -> None:
        """chunk_documents should raise ValueError for invalid chunk_size."""
        from ingestion.chunking import chunk_documents

        with pytest.raises(ValueError):
            chunk_documents([self._make_doc("text")], chunk_size=-1)

    def test_overlap_ge_chunk_size_raises(self) -> None:
        """chunk_documents should raise ValueError when overlap >= chunk_size."""
        from ingestion.chunking import chunk_documents

        with pytest.raises(ValueError):
            chunk_documents(
                [self._make_doc("text")], chunk_size=100, chunk_overlap=100
            )


# ---------------------------------------------------------------------------
# Embedder tests (mocked – no network access needed)
# ---------------------------------------------------------------------------


def _make_mock_sentence_transformer(dim: int = 384):
    """Build a mock SentenceTransformer that avoids network downloads."""
    mock_st = MagicMock()
    mock_st.get_sentence_embedding_dimension.return_value = dim

    def encode(texts, **kwargs):
        vecs = np.random.randn(len(texts), dim).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs

    mock_st.encode.side_effect = encode
    return mock_st


class TestEmbeddingModel:
    """Unit tests for ingestion/embedder.py (model download mocked)."""

    @pytest.fixture
    def model(self, tmp_path: Path):
        """Return an EmbeddingModel with a mocked SentenceTransformer."""
        mock_st = _make_mock_sentence_transformer(dim=384)
        with patch("ingestion.embedder.SentenceTransformer", return_value=mock_st):
            from ingestion.embedder import EmbeddingModel

            return EmbeddingModel(cache_dir=str(tmp_path))

    def test_embed_texts_returns_array(self, model) -> None:
        """embed_texts should return a 2-D numpy array."""
        vectors = model.embed_texts(["Hello world", "RAG is cool"])
        assert vectors.shape == (2, model.embedding_dim)

    def test_embed_query_returns_1d(self, model) -> None:
        """embed_query should return a 1-D numpy array."""
        vec = model.embed_query("What is RAG?")
        assert vec.ndim == 1
        assert vec.shape[0] == model.embedding_dim

    def test_embed_empty_raises(self, model) -> None:
        """embed_texts should raise ValueError for empty input."""
        with pytest.raises(ValueError):
            model.embed_texts([])

    def test_embed_blank_query_raises(self, model) -> None:
        """embed_query should raise ValueError for blank query."""
        with pytest.raises(ValueError):
            model.embed_query("   ")

    def test_normalized_embeddings(self, model) -> None:
        """Embeddings should be L2-normalized (norm ≈ 1.0)."""
        vec = model.embed_query("normalisation check")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5
