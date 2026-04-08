"""Application-wide exception classes and HTTP error handlers."""

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


class DocumentNotFoundError(Exception):
    """Raised when no relevant documents are found in the vector store."""


class EmptyQueryError(ValueError):
    """Raised when the user submits a blank query."""


class IngestError(RuntimeError):
    """Raised when document ingestion fails."""


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle ValueError as HTTP 400 Bad Request."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


async def document_not_found_handler(
    request: Request, exc: DocumentNotFoundError
) -> JSONResponse:
    """Handle DocumentNotFoundError as HTTP 404."""
    return JSONResponse(status_code=404, content={"detail": str(exc)})


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unexpected errors."""
    import logging

    logger = logging.getLogger(__name__)
    logger.error("Unhandled error on %s: %s", request.url, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
    )
