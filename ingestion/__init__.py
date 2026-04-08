"""Ingestion package."""

from .chunking import chunk_documents
from .embedder import EmbeddingModel, get_embedding_model
from .loader import load_directory, load_document

__all__ = [
    "load_document",
    "load_directory",
    "chunk_documents",
    "EmbeddingModel",
    "get_embedding_model",
]
