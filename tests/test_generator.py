"""Tests for rag/generator.py."""

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

    def test_generate_falls_back_when_openai_call_fails(self) -> None:
        """Should return a fallback answer when OpenAI request errors out."""
        generator = Generator()
        generator._api_key = "test-key"
        generator._call_openai = lambda *_: (_ for _ in ()).throw(  # type: ignore[method-assign]
            RuntimeError("insufficient_quota")
        )

        result = generator.generate("What is RAG?", _make_docs())

        assert result["answer"]
        assert "OpenAI request failed: insufficient_quota" in result["answer"]
        assert result["sources"][0]["source"] == "notes.txt"
        assert result["latency_ms"] >= 0

    def test_generate_falls_back_without_api_key(self) -> None:
        """Should return a fallback answer when no API key is configured."""
        generator = Generator()
        generator._api_key = ""

        result = generator.generate("What is RAG?", _make_docs())

        assert "No OpenAI API key is configured." in result["answer"]
