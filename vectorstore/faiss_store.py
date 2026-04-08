"""FAISS vector store module.

Stores document embeddings and metadata; supports similarity search,
persistence (save/load), and metadata filtering.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from langchain_core.documents import Document

from config.settings import get_settings
from ingestion.embedder import EmbeddingModel, get_embedding_model

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS-backed vector store with metadata support.

    Documents and their embeddings are kept in memory; the index and
    metadata can be persisted to disk and reloaded between runs.

    Args:
        embedding_model: EmbeddingModel instance used to embed queries.
        index_path: Directory where the FAISS index and metadata are saved.
    """

    _INDEX_FILE = "index.faiss"
    _META_FILE = "metadata.json"

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        index_path: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self._embedder = embedding_model or get_embedding_model()
        self._index_path = Path(index_path or settings.faiss_index_path)
        self._index_path.mkdir(parents=True, exist_ok=True)

        dim = self._embedder.embedding_dim
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self._documents: List[Document] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Document]) -> None:
        """Embed and add a list of documents to the store.

        Args:
            documents: LangChain Document objects to index.

        Raises:
            ValueError: If documents list is empty.
        """
        if not documents:
            raise ValueError("Cannot add an empty list of documents")

        texts = [doc.page_content for doc in documents]
        embeddings = self._embedder.embed_texts(texts)

        # FAISS expects float32
        vectors = np.array(embeddings, dtype=np.float32)
        self._index.add(vectors)
        self._documents.extend(documents)

        logger.info("Added %d document chunk(s) to FAISS store", len(documents))

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """Retrieve the top-k most similar documents for a query.

        Args:
            query: Natural-language query string.
            k: Number of results to return. Defaults to settings.top_k.

        Returns:
            List of (Document, score) tuples ordered by descending similarity.

        Raises:
            ValueError: If the store is empty or query is blank.
        """
        settings = get_settings()
        k = k or settings.top_k

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if self._index.ntotal == 0:
            raise ValueError("Vector store is empty. Please ingest documents first.")

        query_vector = self._embedder.embed_query(query)
        query_matrix = np.array([query_vector], dtype=np.float32)

        actual_k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query_matrix, actual_k)

        results: List[Tuple[Document, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self._documents[idx]
            results.append((doc, float(score)))

        logger.debug("Similarity search returned %d result(s)", len(results))
        return results

    def save(self) -> None:
        """Persist the FAISS index and document metadata to disk."""
        index_file = self._index_path / self._INDEX_FILE
        meta_file = self._index_path / self._META_FILE

        faiss.write_index(self._index, str(index_file))

        # Serialize documents to JSON (safe, no arbitrary code execution)
        serialised = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in self._documents
        ]
        with open(meta_file, "w", encoding="utf-8") as fh:
            json.dump(serialised, fh, ensure_ascii=False)

        logger.info(
            "Saved FAISS index (%d vectors) to %s",
            self._index.ntotal,
            self._index_path,
        )

    def load(self) -> bool:
        """Load a previously saved FAISS index and metadata from disk.

        Returns:
            True if the index was loaded successfully, False if no saved
            index exists.
        """
        index_file = self._index_path / self._INDEX_FILE
        meta_file = self._index_path / self._META_FILE

        if not index_file.exists() or not meta_file.exists():
            logger.info("No saved FAISS index found at %s", self._index_path)
            return False

        self._index = faiss.read_index(str(index_file))
        with open(meta_file, "r", encoding="utf-8") as fh:
            serialised = json.load(fh)
        self._documents = [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in serialised
        ]

        logger.info(
            "Loaded FAISS index (%d vectors) from %s",
            self._index.ntotal,
            self._index_path,
        )
        return True

    @property
    def document_count(self) -> int:
        """Return the number of indexed document chunks."""
        return self._index.ntotal

    def get_all_documents(self) -> List[Document]:
        """Return a copy of all stored documents.

        Returns:
            List of all Document objects in the store.
        """
        return list(self._documents)

    def clear(self) -> None:
        """Remove all documents and reset the FAISS index."""
        dim = self._embedder.embedding_dim
        self._index = faiss.IndexFlatIP(dim)
        self._documents = []
        logger.info("FAISS vector store cleared")


# Module-level singleton
_default_store: Optional[FAISSVectorStore] = None


def get_vector_store() -> FAISSVectorStore:
    """Return the shared FAISSVectorStore singleton.

    Attempts to load a persisted index on first call.
    """
    global _default_store
    if _default_store is None:
        _default_store = FAISSVectorStore()
        _default_store.load()
    return _default_store
