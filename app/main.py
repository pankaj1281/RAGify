"""FastAPI application entrypoint for RAGify.

Wires together all routes, middleware, exception handlers, and
shared service instances.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.exceptions import (
    DocumentNotFoundError,
    document_not_found_handler,
    generic_error_handler,
    value_error_handler,
)
from app.core.logging import configure_logging
from app.routes import health_router, ingest_router, query_router, ui_router
from app.services.ingestion_service import IngestionService
from app.services.query_service import QueryService
from config.settings import get_settings
from rag.pipeline import RAGPipeline
from vectorstore.faiss_store import FAISSVectorStore, get_vector_store

# ---------------------------------------------------------------------------
# Shared singletons (initialized during startup)
# ---------------------------------------------------------------------------

_pipeline: RAGPipeline | None = None
_ingestion_service: IngestionService | None = None
_query_service: QueryService | None = None
_vector_store: FAISSVectorStore | None = None


def get_pipeline() -> RAGPipeline:
    """Return the shared RAGPipeline singleton."""
    if _pipeline is None:
        raise RuntimeError("Application has not been started yet")
    return _pipeline


def get_ingestion_service() -> IngestionService:
    """Return the shared IngestionService singleton."""
    if _ingestion_service is None:
        raise RuntimeError("Application has not been started yet")
    return _ingestion_service


def get_query_service() -> QueryService:
    """Return the shared QueryService singleton."""
    if _query_service is None:
        raise RuntimeError("Application has not been started yet")
    return _query_service


def get_vector_store_instance() -> FAISSVectorStore:
    """Return the shared FAISSVectorStore singleton."""
    if _vector_store is None:
        raise RuntimeError("Application has not been started yet")
    return _vector_store


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize shared resources on startup and clean up on shutdown."""
    global _pipeline, _ingestion_service, _query_service, _vector_store

    configure_logging()
    logger = logging.getLogger(__name__)

    settings = get_settings()

    # Ensure required directories exist
    os.makedirs(settings.data_dir, exist_ok=True)
    os.makedirs(settings.upload_dir, exist_ok=True)

    logger.info("Starting RAGify v%s", settings.app_version)

    _vector_store = get_vector_store()
    _pipeline = RAGPipeline()
    _ingestion_service = IngestionService(pipeline=_pipeline)
    _query_service = QueryService(pipeline=_pipeline)

    logger.info(
        "RAGify ready – %d document chunks indexed", _vector_store.document_count
    )

    yield

    logger.info("RAGify shutting down")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production-grade Retrieval-Augmented Generation (RAG) API. "
            "Upload documents and query them with an LLM."
        ),
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    app.add_exception_handler(ValueError, value_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(DocumentNotFoundError, document_not_found_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, generic_error_handler)

    # Routers
    app.include_router(ui_router)
    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(query_router)

    return app


app = create_app()
