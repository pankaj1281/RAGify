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

    def test_nvidia_provider_initializes_with_correct_settings(self, monkeypatch) -> None:
        """Generator should use NVIDIA base URL and model when provider is 'nvidia'."""
        monkeypatch.setenv("LLM_PROVIDER", "nvidia")
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key")
        monkeypatch.setenv("NVIDIA_MODEL", "meta/llama3-70b-instruct")
        monkeypatch.setenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

        # Clear lru_cache so env vars are re-read
        from config.settings import get_settings
        get_settings.cache_clear()

        generator = Generator()

        assert generator._provider == "nvidia"
        assert generator._api_key == "nvapi-test-key"
        assert generator._base_url == "https://integrate.api.nvidia.com/v1"
        assert generator._model == "meta/llama3-70b-instruct"

        # Clean up cache
        get_settings.cache_clear()

    def test_nvidia_raises_without_api_key(self, monkeypatch) -> None:
        """_call_nvidia should raise ValueError when NVIDIA_API_KEY is missing."""
        monkeypatch.setenv("LLM_PROVIDER", "nvidia")
        monkeypatch.setenv("NVIDIA_API_KEY", "")

        from config.settings import get_settings
        get_settings.cache_clear()

        generator = Generator()
        result = generator.generate("What is RAG?", _make_docs())

        assert "Generated answer is unavailable" in result["answer"]
        assert "LLM request failed" in result["answer"]

        get_settings.cache_clear()
