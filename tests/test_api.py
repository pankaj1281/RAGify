"""Tests for the FastAPI endpoints."""

import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_vector_store(doc_count: int = 0) -> MagicMock:
    store = MagicMock()
    store.document_count = doc_count
    return store


def _make_mock_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline._use_hybrid = False
    pipeline.query.return_value = {
        "answer": "The answer is 42.",
        "sources": [{"source": "doc.txt", "page": None, "chunk_index": 0}],
        "retrieved_docs": 3,
        "latency_ms": 100.0,
        "rewritten_query": "What is the answer?",
    }
    pipeline.ingest.return_value = 10
    return pipeline


def _make_mock_ingestion_service() -> MagicMock:
    from app.services.ingestion_service import IngestionService

    svc = MagicMock(spec=IngestionService)
    svc.ingest_file.return_value = 5
    return svc


def _make_mock_query_service(answer_return: dict | None = None) -> MagicMock:
    from app.services.query_service import QueryService

    svc = MagicMock(spec=QueryService)
    svc.answer.return_value = answer_return or {
        "answer": "The answer is 42.",
        "sources": [{"source": "doc.txt", "page": None, "chunk_index": 0}],
        "retrieved_docs": 3,
        "latency_ms": 100.0,
        "rewritten_query": "What is the answer?",
    }
    return svc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> Generator:
    """Return a TestClient with a mocked lifespan (no real model loading)."""
    import app.main as main_module

    mock_vector_store = _make_mock_vector_store(doc_count=0)
    mock_pipeline = _make_mock_pipeline()
    mock_ingestion_svc = _make_mock_ingestion_service()
    mock_query_svc = _make_mock_query_service()

    @asynccontextmanager
    async def mock_lifespan(app) -> AsyncGenerator[None, None]:
        """Lifespan that injects mocks without loading real models."""
        main_module._vector_store = mock_vector_store
        main_module._pipeline = mock_pipeline
        main_module._ingestion_service = mock_ingestion_svc
        main_module._query_service = mock_query_svc
        yield
        main_module._vector_store = None
        main_module._pipeline = None
        main_module._ingestion_service = None
        main_module._query_service = None

    from fastapi import FastAPI
    from app.core.exceptions import (
        DocumentNotFoundError,
        document_not_found_handler,
        generic_error_handler,
        value_error_handler,
    )
    from app.routes import health_router, ingest_router, query_router, ui_router
    from config.settings import get_settings

    settings = get_settings()
    test_app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=mock_lifespan,
    )
    test_app.add_exception_handler(ValueError, value_error_handler)  # type: ignore[arg-type]
    test_app.add_exception_handler(DocumentNotFoundError, document_not_found_handler)  # type: ignore[arg-type]
    test_app.add_exception_handler(Exception, generic_error_handler)
    test_app.include_router(ui_router)
    test_app.include_router(health_router)
    test_app.include_router(ingest_router)
    test_app.include_router(query_router)

    with TestClient(test_app, raise_server_exceptions=False) as c:
        yield c, mock_vector_store, mock_query_svc, mock_ingestion_svc


# ---------------------------------------------------------------------------
# UI endpoint
# ---------------------------------------------------------------------------


class TestUIEndpoint:
    """Tests for GET /."""

    def test_home_has_upload_option(self, client) -> None:
        """Home page should include a file upload input."""
        c, _, __, ___ = client
        response = c.get("/")
        assert response.status_code == 200
        text = response.text
        assert "type=\"file\"" in text
        assert "name=\"files\"" in text


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for GET /health/."""

    def test_health_ok(self, client) -> None:
        """Health endpoint should return status ok."""
        c, mock_store, _, __ = client
        mock_store.document_count = 42
        response = c.get("/health/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "indexed_documents" in data


# ---------------------------------------------------------------------------
# Ingest endpoint
# ---------------------------------------------------------------------------


class TestIngestEndpoint:
    """Tests for POST /ingest/."""

    def test_ingest_txt_file(self, client) -> None:
        """Should accept a TXT file and return ingestion stats."""
        c, _, __, mock_ingest = client
        content = b"This is a test document with some content."
        response = c.post(
            "/ingest/",
            files={"files": ("test.txt", io.BytesIO(content), "text/plain")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["files_processed"] == 1
        assert data["chunks_indexed"] >= 0

    def test_ingest_unsupported_format(self, client) -> None:
        """Should reject unsupported file formats."""
        c, _, __, ___ = client
        response = c.post(
            "/ingest/",
            files={"files": ("data.csv", io.BytesIO(b"a,b,c"), "text/csv")},
        )
        assert response.status_code == 422

    def test_ingest_no_files(self, client) -> None:
        """Should return 422 when no files are provided."""
        c, _, __, ___ = client
        response = c.post("/ingest/")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------


class TestQueryEndpoint:
    """Tests for GET /query/."""

    def test_query_returns_answer(self, client) -> None:
        """Should return an answer for a valid question."""
        c, _, __, ___ = client
        response = c.get("/query/", params={"q": "What is the answer?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "latency_ms" in data
        assert "retrieved_docs" in data

    def test_query_empty_string_returns_422(self, client) -> None:
        """FastAPI validation should reject a missing q parameter."""
        c, _, __, ___ = client
        response = c.get("/query/")
        assert response.status_code == 422

    def test_query_no_documents_raises_404(self, client) -> None:
        """Should return 404 when the vector store is empty."""
        from app.core.exceptions import DocumentNotFoundError

        c, _, mock_query, __ = client
        mock_query.answer.side_effect = DocumentNotFoundError("No documents ingested")
        response = c.get("/query/", params={"q": "anything"})
        assert response.status_code == 404
        # Reset
        mock_query.answer.side_effect = None
        mock_query.answer.return_value = {
            "answer": "The answer is 42.",
            "sources": [{"source": "doc.txt", "page": None, "chunk_index": 0}],
            "retrieved_docs": 3,
            "latency_ms": 100.0,
            "rewritten_query": "What is the answer?",
        }

    def test_query_with_k_parameter(self, client) -> None:
        """Should accept an optional k parameter."""
        c, _, __, ___ = client
        response = c.get("/query/", params={"q": "test question", "k": 3})
        assert response.status_code == 200

    def test_query_k_out_of_range(self, client) -> None:
        """Should reject k values outside 1-20."""
        c, _, __, ___ = client
        response = c.get("/query/", params={"q": "test", "k": 25})
        assert response.status_code == 422
