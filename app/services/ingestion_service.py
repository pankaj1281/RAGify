"""Ingestion service.

Handles the business logic for uploading and indexing documents.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import List

from app.core.exceptions import IngestError
from config.settings import get_settings
from ingestion.chunking import chunk_documents
from ingestion.loader import load_document
from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class IngestionService:
    """Coordinates document loading, chunking, and vector store indexing.

    Args:
        pipeline: RAGPipeline instance used for ingestion.
        upload_dir: Directory where uploaded files are temporarily stored.
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        upload_dir: str | None = None,
    ) -> None:
        settings = get_settings()
        self._pipeline = pipeline
        self._upload_dir = Path(upload_dir or settings.upload_dir)
        self._upload_dir.mkdir(parents=True, exist_ok=True)

    def ingest_file(self, filename: str, file_bytes: bytes) -> int:
        """Save an uploaded file and ingest it into the vector store.

        Args:
            filename: Original filename (used to determine format).
            file_bytes: Raw bytes of the uploaded file.

        Returns:
            Number of chunks indexed from this file.

        Raises:
            IngestError: If loading or indexing fails.
        """
        dest = self._upload_dir / filename
        try:
            dest.write_bytes(file_bytes)
            documents = load_document(str(dest))
            chunks = chunk_documents(documents)
            if not chunks:
                logger.warning("No chunks produced from %s", filename)
                return 0
            self._pipeline.ingest(chunks, save=True)
            logger.info(
                "Ingested %d chunks from %s", len(chunks), filename
            )
            return len(chunks)
        except Exception as exc:
            logger.error("Ingestion failed for %s: %s", filename, exc)
            raise IngestError(f"Failed to ingest '{filename}': {exc}") from exc
