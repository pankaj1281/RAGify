"""Retriever module.

Implements basic similarity search and optional hybrid BM25 + vector search.
"""

import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from config.settings import get_settings
from vectorstore.faiss_store import FAISSVectorStore, get_vector_store

logger = logging.getLogger(__name__)


class Retriever:
    """Document retriever backed by a FAISS vector store.

    Supports:
    * Basic dense similarity search (default)
    * Hybrid search combining BM25 sparse retrieval with dense reranking

    Args:
        vector_store: FAISSVectorStore instance to search.
        top_k: Default number of documents to retrieve per query.
    """

    def __init__(
        self,
        vector_store: Optional[FAISSVectorStore] = None,
        top_k: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self._store = vector_store or get_vector_store()
        self._top_k = top_k or settings.top_k

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Document]:
        """Retrieve the most relevant documents for a query.

        Uses pure dense similarity search.

        Args:
            query: The user query string.
            k: Number of documents to return. Defaults to self._top_k.

        Returns:
            List of relevant Document objects ordered by relevance.

        Raises:
            ValueError: If query is empty or the store has no documents.
        """
        k = k or self._top_k
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        results: List[Tuple[Document, float]] = self._store.similarity_search(
            query, k=k
        )
        documents = [doc for doc, _score in results]
        logger.info(
            "Retrieved %d document(s) for query: %.80s", len(documents), query
        )
        return documents

    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """Retrieve documents along with their similarity scores.

        Args:
            query: The user query string.
            k: Number of documents to return.

        Returns:
            List of (Document, score) tuples ordered by descending score.
        """
        k = k or self._top_k
        return self._store.similarity_search(query, k=k)

    def hybrid_retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
    ) -> List[Document]:
        """Hybrid retrieval combining BM25 sparse and dense vector scores.

        BM25 scores are computed in-memory over the stored document corpus.
        Final scores are a weighted linear combination of normalised BM25
        and cosine similarity scores.

        Args:
            query: The user query string.
            k: Number of documents to return.
            bm25_weight: Weight assigned to BM25 scores (0–1).
            vector_weight: Weight assigned to vector similarity scores (0–1).

        Returns:
            List of Document objects ordered by combined relevance score.
        """
        k = k or self._top_k
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            from rank_bm25 import BM25Okapi  # optional dependency
        except ImportError:
            logger.warning(
                "rank_bm25 not installed – falling back to pure vector search"
            )
            return self.retrieve(query, k=k)

        documents = self._store.get_all_documents()
        if not documents:
            raise ValueError("Vector store is empty. Please ingest documents first.")

        # BM25 scoring
        tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)

        # Vector similarity scores
        vector_results = self._store.similarity_search(query, k=len(documents))
        vector_score_map = {
            id(doc): score for doc, score in vector_results
        }

        # Normalise BM25 scores to [0, 1]
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        bm25_scores_norm = bm25_scores / max_bm25

        # Combine scores
        combined: List[Tuple[Document, float]] = []
        for i, doc in enumerate(documents):
            v_score = vector_score_map.get(id(doc), 0.0)
            combined_score = (
                bm25_weight * bm25_scores_norm[i] + vector_weight * v_score
            )
            combined.append((doc, combined_score))

        combined.sort(key=lambda x: x[1], reverse=True)
        logger.info(
            "Hybrid retrieval returned %d document(s) for query: %.80s",
            min(k, len(combined)),
            query,
        )
        return [doc for doc, _ in combined[:k]]
