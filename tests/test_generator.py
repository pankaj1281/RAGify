"""Tests for rag/generator.py."""

from unittest.mock import MagicMock

from langchain_core.documents import Document

from rag.generator import Generator


def _make_docs() -> list[Document]:
    return [
        Document(
            page_content="RAG stands for Retrieval-Augmented Generation.",
            metadata={"source": "notes.txt", "chunk_index": 0},
        )
    ]


class TestGenerator:
    """Unit tests for generation fallback behavior."""

    def test_generate_returns_llm_answer_when_call_succeeds(self) -> None:
        """Should return the model answer when the LLM call succeeds."""
        generator = Generator()
        generator._call_llm = MagicMock(return_value="Direct model answer.")  # type: ignore[method-assign]

        result = generator.generate("What is RAG?", _make_docs())

        assert result["answer"] == "Direct model answer."
        generator._call_llm.assert_called_once()

    def test_generate_falls_back_when_llm_call_fails(self) -> None:
        """Should return a fallback answer when the LLM request errors out."""
        generator = Generator()
        generator._call_llm = MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("connection refused")
        )

        result = generator.generate("What is RAG?", _make_docs())

        assert "Generated answer is unavailable" in result["answer"]
        assert "LLM request failed" in result["answer"]
        assert result["sources"][0]["source"] == "notes.txt"
        assert result["latency_ms"] >= 0

    def test_generate_falls_back_without_api_key(self) -> None:
        """Should return a fallback answer when the provider call fails due to missing key."""
        generator = Generator()
        generator._call_llm = MagicMock(  # type: ignore[method-assign]
            side_effect=ValueError("API key not set")
        )

        result = generator.generate("What is RAG?", _make_docs())

        assert "Generated answer is unavailable" in result["answer"]
        assert result["sources"][0]["source"] == "notes.txt"
