"""Tests for rag/generator.py."""

from unittest.mock import MagicMock, patch

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

    def test_generate_uses_openai_answer_when_available(self) -> None:
        """Should return the model answer when OpenAI call succeeds."""
        generator = Generator()
        generator._api_key = "test-key"
        generator._call_openai = MagicMock(return_value="Direct model answer.")  # type: ignore[method-assign]

        result = generator.generate("What is RAG?", _make_docs())

        assert result["answer"] == "Direct model answer."
        generator._call_openai.assert_called_once()

    def test_generate_falls_back_when_openai_call_fails(self) -> None:
        """Should return a fallback answer when OpenAI request errors out."""
        generator = Generator()
        generator._api_key = "test-key"
        generator._call_openai = MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("insufficient_quota")
        )
        with patch("rag.generator.Generator._openai_error_types", return_value=(RuntimeError,)):
            result = generator.generate("What is RAG?", _make_docs())

        assert "Generated answer is unavailable" in result["answer"]
        assert "OpenAI request failed: insufficient_quota" in result["answer"]
        assert result["sources"][0]["source"] == "notes.txt"
        assert result["latency_ms"] >= 0

    def test_generate_falls_back_without_api_key(self) -> None:
        """Should return a fallback answer when no API key is configured."""
        generator = Generator()
        generator._api_key = ""

        result = generator.generate("What is RAG?", _make_docs())

        assert "No OpenAI API key is configured." in result["answer"]
