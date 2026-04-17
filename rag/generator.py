"""LLM response generator module.

Builds structured prompts from retrieved context and calls the OpenAI API
to produce a grounded, citation-aware answer.
"""

import logging
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Prompt template used for all RAG responses
_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer ONLY using the context below.\n"
    "If the answer is not present in the context, say 'I don't know'.\n"
    "Always be concise and factual."
)

_USER_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

_FALLBACK_CONTEXT_PREVIEW_CHARS = 500
_FALLBACK_UNAVAILABLE_MESSAGE = (
    "Generated answer is unavailable, so returning retrieved context preview."
)


class Generator:
    """Generates LLM answers from retrieved document context.

    Uses the OpenAI ChatCompletion API by default. Falls back to a simple
    context-extraction heuristic when no API key is configured (useful for
    testing).

    Args:
        model: OpenAI model name (e.g. ``"gpt-3.5-turbo"``).
        max_tokens: Maximum tokens for the generated answer.
        temperature: Sampling temperature (lower = more deterministic).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.openai_model
        self._max_tokens = max_tokens or settings.openai_max_tokens
        self._temperature = (
            temperature if temperature is not None else settings.openai_temperature
        )
        self._api_key = settings.openai_api_key

    def generate(
        self,
        question: str,
        context_docs: List[Document],
    ) -> Dict[str, Any]:
        """Generate an answer grounded in the provided context documents.

        Args:
            question: The user's question.
            context_docs: Retrieved documents used as context.

        Returns:
            Dictionary with keys:
            * ``answer`` – the generated answer string.
            * ``sources`` – list of source metadata dicts for citations.
            * ``latency_ms`` – generation latency in milliseconds.

        Raises:
            ValueError: If question or context_docs is empty.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        if not context_docs:
            raise ValueError("No context documents provided")

        context = self._build_context(context_docs)
        sources = self._extract_sources(context_docs)

        start = time.perf_counter()
        if self._api_key:
            try:
                answer = self._call_openai(question, context)
            except Exception as exc:
                if self._is_openai_api_error(exc):
                    logger.warning(
                        "OpenAI generation failed (%s); using fallback response", exc
                    )
                    answer = self._fallback_answer(
                        question=question,
                        context=context,
                        reason=f"OpenAI request failed: {exc}",
                    )
                else:
                    raise
        else:
            logger.warning(
                "No OpenAI API key configured – using fallback stub response"
            )
            answer = self._fallback_answer(
                question=question,
                context=context,
                reason="No OpenAI API key is configured.",
            )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "Generated answer in %.1fms for question: %.80s", latency_ms, question
        )
        return {
            "answer": answer,
            "sources": sources,
            "latency_ms": latency_ms,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_context(self, documents: List[Document]) -> str:
        """Concatenate document chunks into a single numbered context block."""
        parts = []
        for i, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[{i}] (source: {source})\n{doc.page_content}")
        return "\n\n".join(parts)

    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Build a list of source citation dicts from document metadata."""
        seen: set = set()
        sources = []
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            if source not in seen:
                seen.add(source)
                sources.append(
                    {
                        "source": source,
                        "page": doc.metadata.get("page"),
                        "chunk_index": doc.metadata.get("chunk_index"),
                    }
                )
        return sources

    def _call_openai(self, question: str, context: str) -> str:
        """Call the OpenAI ChatCompletion API and return the answer text."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self._api_key)
            user_message = _USER_TEMPLATE.format(
                context=context, question=question
            )
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("OpenAI API error: %s", exc)
            raise

    @staticmethod
    def _is_openai_api_error(exc: Exception) -> bool:
        """Return True for OpenAI client/API errors, False if types are unavailable."""
        return isinstance(exc, Generator._openai_error_types())

    @staticmethod
    @lru_cache(maxsize=1)
    def _openai_error_types() -> tuple[type[Exception], ...]:
        """Load and cache OpenAI exception classes used for fallback decisions."""
        try:
            from openai import (
                APIConnectionError,
                APIError,
                APITimeoutError,
                AuthenticationError,
                BadRequestError,
                ConflictError,
                InternalServerError,
                NotFoundError,
                PermissionDeniedError,
                RateLimitError,
                UnprocessableEntityError,
            )
        except (ImportError, ModuleNotFoundError):
            return tuple()
        return (
            APIConnectionError,
            APIError,
            APITimeoutError,
            AuthenticationError,
            BadRequestError,
            ConflictError,
            InternalServerError,
            NotFoundError,
            PermissionDeniedError,
            RateLimitError,
            UnprocessableEntityError,
        )

    @staticmethod
    def _fallback_answer(
        question: str, context: str, reason: str = "OpenAI response unavailable."
    ) -> str:
        """Return a stub answer without calling an LLM (used when no API key)."""
        return (
            f"{reason}\n"
            f"{_FALLBACK_UNAVAILABLE_MESSAGE}\n"
            f"Question: {question}\n\n"
            f"Context preview:\n{context[:_FALLBACK_CONTEXT_PREVIEW_CHARS]}"
        )
