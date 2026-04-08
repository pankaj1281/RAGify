"""Pydantic request / response schemas for the API layer."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class IngestResponse(BaseModel):
    """Response returned after document ingestion."""

    message: str = Field(..., description="Human-readable status message")
    files_processed: int = Field(..., description="Number of files successfully processed")
    chunks_indexed: int = Field(..., description="Total document chunks added to the vector store")


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class SourceCitation(BaseModel):
    """Citation metadata for a retrieved document chunk."""

    source: str = Field(..., description="Source filename")
    page: Optional[int] = Field(None, description="Page number (PDF only)")
    chunk_index: Optional[int] = Field(None, description="Chunk index within the document")


class QueryResponse(BaseModel):
    """Response returned for a user query."""

    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceCitation] = Field(default_factory=list, description="Source citations")
    retrieved_docs: int = Field(..., description="Number of retrieved document chunks")
    latency_ms: float = Field(..., description="End-to-end pipeline latency in milliseconds")
    rewritten_query: Optional[str] = Field(None, description="Rewritten query (if rewriting was enabled)")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response returned by the health-check endpoint."""

    status: str = Field(..., description="'ok' when the service is healthy")
    version: str = Field(..., description="Application version string")
    indexed_documents: int = Field(..., description="Number of document chunks in the vector store")
