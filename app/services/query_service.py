"""Query service.

Handles the business logic for answering user queries.
"""

import logging
from typing import Any, Dict, Optional

from app.core.exceptions import DocumentNotFoundError, EmptyQueryError
from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class QueryService:
    """Wraps the RAG pipeline and adds request-level validation and logging.

    Args:
        pipeline: RAGPipeline instance to delegate queries to.
    """

    def __init__(self, pipeline: RAGPipeline) -> None:
        self._pipeline = pipeline

    def answer(
        self,
        question: str,
        k: Optional[int] = None,
        rewrite: bool = False,
        use_hybrid: bool = False,
    ) -> Dict[str, Any]:
        """Answer a user question using the RAG pipeline.

        Args:
            question: The user's natural-language question.
            k: Number of context documents to retrieve.
            rewrite: Whether to rewrite the query before retrieval.
            use_hybrid: Whether to use hybrid BM25+vector retrieval.

        Returns:
            Result dictionary from the RAG pipeline.

        Raises:
            EmptyQueryError: If the question is blank.
            DocumentNotFoundError: If the vector store is empty.
        """
        question = question.strip()
        if not question:
            raise EmptyQueryError("Question cannot be empty")

        try:
            result = self._pipeline.query(
                question, k=k, rewrite=rewrite, use_hybrid=use_hybrid
            )
        except ValueError as exc:
            msg = str(exc)
            if "empty" in msg.lower():
                raise DocumentNotFoundError(
                    "No documents have been ingested yet. "
                    "Please upload documents via POST /ingest first."
                ) from exc
            raise

        logger.info(
            "Query answered | latency=%.1fms | docs=%d | question=%.80s",
            result.get("latency_ms", 0),
            result.get("retrieved_docs", 0),
            question,
        )
        return result
