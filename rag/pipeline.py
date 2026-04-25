"""RAG pipeline module.

Orchestrates the full Retrieval-Augmented Generation workflow:
query rewriting → retrieval → optional reranking → generation.
"""

import logging
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from config.settings import get_settings
from rag.generator import Generator
from rag.retriever import Retriever
from vectorstore.faiss_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline combining retrieval and generation.

    Features:
    * LRU-cached query results (configurable cache size)
    * Optional query rewriting via LLM to improve retrieval recall
    * Optional cross-encoder reranking of retrieved documents
    * Response citations from source metadata

    Args:
        retriever: Retriever instance (uses shared store by default).
        generator: Generator instance (uses default settings by default).
        use_hybrid: Whether to use hybrid BM25+vector retrieval.
        use_reranker: Whether to apply cross-encoder reranking.
        cache_size: Maximum number of cached query responses.
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        generator: Optional[Generator] = None,
        use_hybrid: bool = False,
        use_reranker: bool = False,
        cache_size: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self._retriever = retriever or Retriever()
        self._generator = generator or Generator()
        self._use_hybrid = use_hybrid
        self._use_reranker = use_reranker
        self._cache_size = cache_size or settings.query_cache_size
        self._cache: Dict[str, Dict[str, Any]] = {}

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        rewrite: bool = False,
        use_hybrid: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Process a user question through the full RAG pipeline.

        Args:
            question: The user's natural-language question.
            k: Number of documents to retrieve.
            rewrite: Whether to rewrite the query using LLM before retrieval.
            use_hybrid: Override the instance-level hybrid retrieval setting
                for this single call. If None, falls back to
                ``self._use_hybrid``.

        Returns:
            Dictionary with keys:
            * ``answer`` – generated answer string.
            * ``sources`` – list of citation dicts (source, page, chunk).
            * ``latency_ms`` – total pipeline latency in milliseconds.
            * ``retrieved_docs`` – count of retrieved document chunks.
            * ``rewritten_query`` – the (possibly rewritten) query used.

        Raises:
            ValueError: If the question is empty.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        hybrid = use_hybrid if use_hybrid is not None else self._use_hybrid
        start = time.perf_counter()

        # LRU-style cache check
        cache_key = self._cache_key(question, k, rewrite)
        if cache_key in self._cache:
            logger.debug("Cache hit for query: %.80s", question)
            return self._cache[cache_key]

        # Optional query rewriting
        effective_query = question
        if rewrite:
            effective_query = self._rewrite_query(question)
            logger.info(
                "Query rewritten: '%.80s' → '%.80s'", question, effective_query
            )

        # Retrieval
        if hybrid:
            docs = self._retriever.hybrid_retrieve(effective_query, k=k)
        else:
            docs = self._retriever.retrieve(effective_query, k=k)

        # Optional reranking
        if self._use_reranker and docs:
            docs = self._rerank(effective_query, docs)

        # Generation
        gen_result = self._generator.generate(question, docs)
        total_latency = round((time.perf_counter() - start) * 1000, 2)

        result: Dict[str, Any] = {
            "answer": gen_result["answer"],
            "sources": gen_result["sources"],
            "latency_ms": total_latency,
            "retrieved_docs": len(docs),
            "rewritten_query": effective_query,
        }

        # Store in cache (simple bounded dict)
        if len(self._cache) >= self._cache_size:
            # Evict the oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[cache_key] = result

        logger.info(
            "RAG pipeline completed in %.1fms "
            "(retrieved=%d, answer_len=%d)",
            total_latency,
            len(docs),
            len(gen_result["answer"]),
        )
        return result

    def ingest(
        self,
        documents: List[Document],
        save: bool = True,
    ) -> int:
        """Add documents to the vector store and optionally persist the index.

        Args:
            documents: List of chunked Document objects to index.
            save: Whether to save the updated index to disk.

        Returns:
            Total number of indexed document chunks after ingestion.
        """
        self._retriever._store.add_documents(documents)
        if save:
            self._retriever._store.save()
        # Clear query cache since store changed
        self._cache.clear()
        return self._retriever._store.document_count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rewrite_query(self, question: str) -> str:
        """Rewrite a user query to improve retrieval using the active LLM."""
        try:
            settings = get_settings()
            provider = settings.llm_provider.lower()

            if provider == "groq":
                try:
                    from groq import Groq
                except ImportError:
                    logger.warning("groq package not installed; skipping query rewrite")
                    return question
                client = Groq(api_key=self._generator._api_key)
                response = client.chat.completions.create(
                    model=self._generator._model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Rewrite the following question to make it more precise "
                                "and retrieval-friendly. Return ONLY the rewritten question."
                            ),
                        },
                        {"role": "user", "content": question},
                    ],
                    max_tokens=128,
                    temperature=0.0,
                )
                rewritten = response.choices[0].message.content.strip()
                return rewritten if rewritten else question

            # openai or nvidia use the OpenAI-compatible client
            from openai import OpenAI

            kwargs: dict = {"api_key": self._generator._api_key}
            if self._generator._base_url:
                kwargs["base_url"] = self._generator._base_url
            client = OpenAI(**kwargs)
            response = client.chat.completions.create(
                model=self._generator._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the following question to make it more precise "
                            "and retrieval-friendly. Return ONLY the rewritten question."
                        ),
                    },
                    {"role": "user", "content": question},
                ],
                max_tokens=128,
                temperature=0.0,
            )
            rewritten = response.choices[0].message.content.strip()
            return rewritten if rewritten else question
        except Exception as exc:
            logger.warning("Query rewrite failed (%s); using original query", exc)
            return question

    def _rerank(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Document]:
        """Rerank documents using a cross-encoder model (optional dependency).

        Falls back to the original order if the cross-encoder is unavailable.
        """
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [(query, doc.page_content) for doc in documents]
            scores = model.predict(pairs)
            ranked = sorted(
                zip(documents, scores), key=lambda x: x[1], reverse=True
            )
            logger.debug("Reranked %d document(s)", len(documents))
            return [doc for doc, _ in ranked]
        except ImportError:
            logger.warning(
                "CrossEncoder not available – skipping reranking"
            )
            return documents
        except Exception as exc:
            logger.warning("Reranking failed (%s) – using original order", exc)
            return documents

    @staticmethod
    def _cache_key(question: str, k: Optional[int], rewrite: bool) -> str:
        """Build a deterministic cache key from query parameters."""
        return f"{question}|k={k}|rewrite={rewrite}"
