"""LLM response generator module.

Builds structured prompts from retrieved context and calls the configured LLM
provider to produce a grounded, citation-aware answer.

Supported providers
-------------------
* ``ollama``  – Free, local LLMs via Ollama (https://ollama.com).
                No API key required. Default provider.
* ``groq``    – Free cloud inference via Groq (https://console.groq.com).
                Requires ``GROQ_API_KEY``.
* ``nvidia``  – Free NVIDIA cloud API (https://build.nvidia.com).
                Requires ``NVIDIA_API_KEY``. OpenAI-compatible endpoint.
* ``openai``  – OpenAI ChatCompletion (https://platform.openai.com).
                Requires ``OPENAI_API_KEY``.
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

    Selects the LLM backend from the ``LLM_PROVIDER`` setting:

    * ``ollama``  – free local Ollama server (default, no API key needed).
    * ``groq``    – free Groq cloud API (requires ``GROQ_API_KEY``).
    * ``nvidia``  – free NVIDIA cloud API (requires ``NVIDIA_API_KEY``).
    * ``openai``  – paid OpenAI API (requires ``OPENAI_API_KEY``).

    Falls back to a simple context-extraction stub when the selected provider
    is unreachable or has no credentials configured.

    Args:
        model: Model name override (provider-specific).
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
        self._provider = settings.llm_provider.lower()
        self._max_tokens = max_tokens or settings.openai_max_tokens
        self._temperature = (
            temperature if temperature is not None else settings.openai_temperature
        )

        # Resolve the model name and credentials per provider
        if self._provider == "ollama":
            self._model = model or settings.ollama_model
            self._api_key = "ollama"  # Ollama accepts any non-empty string
            self._base_url = settings.ollama_base_url
        elif self._provider == "groq":
            self._model = model or settings.groq_model
            self._api_key = settings.groq_api_key
            self._base_url = None
        elif self._provider == "nvidia":
            self._model = model or settings.nvidia_model
            self._api_key = settings.nvidia_api_key
            self._base_url = settings.nvidia_base_url
        else:  # openai (default fallback)
            self._provider = "openai"
            self._model = model or settings.openai_model
            self._api_key = settings.openai_api_key
            self._base_url = None

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

        try:
            answer = self._call_llm(question, context)
        except Exception as exc:
            logger.warning(
                "LLM generation failed (%s); using fallback response", exc
            )
            answer = self._fallback_answer(
                question=question,
                context=context,
                reason=f"LLM request failed ({self._provider}): {exc}",
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

    def _call_llm(self, question: str, context: str) -> str:
        """Dispatch to the configured LLM provider and return the answer."""
        if self._provider == "groq":
            return self._call_groq(question, context)
        if self._provider == "nvidia":
            return self._call_nvidia(question, context)
        # Both "ollama" and "openai" use the OpenAI-compatible client
        return self._call_openai_compatible(question, context)

    def _call_openai_compatible(self, question: str, context: str) -> str:
        """Call an OpenAI-compatible API (OpenAI or Ollama) and return the answer."""
        from openai import OpenAI

        kwargs: Dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url

        client = OpenAI(**kwargs)
        user_message = _USER_TEMPLATE.format(context=context, question=question)
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

    def _call_groq(self, question: str, context: str) -> str:
        """Call the Groq API and return the answer."""
        if not self._api_key or self._api_key == "ollama":
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com"
            )
        try:
            from groq import Groq
        except ImportError as exc:
            raise ImportError(
                "The 'groq' package is required for Groq provider. "
                "Install it with: pip install groq"
            ) from exc

        client = Groq(api_key=self._api_key)
        user_message = _USER_TEMPLATE.format(context=context, question=question)
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

    def _call_nvidia(self, question: str, context: str) -> str:
        """Call the NVIDIA API (OpenAI-compatible) and return the answer."""
        if not self._api_key:
            raise ValueError(
                "NVIDIA_API_KEY is not set. "
                "Get a free key at https://build.nvidia.com"
            )
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        user_message = _USER_TEMPLATE.format(context=context, question=question)
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

    @staticmethod
    def _fallback_answer(
        question: str, context: str, reason: str = "LLM response unavailable."
    ) -> str:
        """Return a stub answer without calling an LLM (used on failure)."""
        return (
            f"{reason}\n"
            f"{_FALLBACK_UNAVAILABLE_MESSAGE}\n"
            f"Question: {question}\n\n"
            f"Context preview:\n{context[:_FALLBACK_CONTEXT_PREVIEW_CHARS]}"
        )
