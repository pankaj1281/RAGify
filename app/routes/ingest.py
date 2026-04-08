"""Ingest API router.

POST /ingest – upload one or more documents and index them into the vector store.
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.services.ingestion_service import IngestionService
from app.utils.schemas import IngestResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

# Allowed MIME types / extensions
_ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
_ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def _get_ingestion_service() -> IngestionService:
    """Dependency: return the application-level IngestionService."""
    from app.main import get_ingestion_service  # avoid circular import

    return get_ingestion_service()


@router.post("/", response_model=IngestResponse)
async def ingest_documents(
    files: List[UploadFile] = File(..., description="Documents to ingest (PDF, TXT, DOCX)"),
    service: IngestionService = Depends(_get_ingestion_service),
) -> IngestResponse:
    """Upload and index one or more documents.

    Accepts PDF, TXT, and DOCX files. Each file is loaded, chunked, embedded,
    and stored in the FAISS vector index.

    Returns the number of files processed and total chunks indexed.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    total_chunks = 0
    processed = 0
    errors: List[str] = []

    for upload in files:
        filename = upload.filename or "unknown"
        suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if suffix not in _ALLOWED_EXTENSIONS:
            errors.append(f"'{filename}' has unsupported format '{suffix}'")
            continue

        try:
            content = await upload.read()
            chunks = service.ingest_file(filename, content)
            total_chunks += chunks
            processed += 1
            logger.info("Ingested file: %s (%d chunks)", filename, chunks)
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", filename, exc)
            errors.append(f"'{filename}': {exc}")

    if errors and processed == 0:
        raise HTTPException(
            status_code=422,
            detail={"message": "All files failed to ingest", "errors": errors},
        )

    msg = f"Successfully ingested {processed} file(s)"
    if errors:
        msg += f"; {len(errors)} file(s) failed"

    return IngestResponse(
        message=msg,
        files_processed=processed,
        chunks_indexed=total_chunks,
    )
