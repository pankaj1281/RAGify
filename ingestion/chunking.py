"""Text chunking module.

Splits documents into overlapping chunks using token-aware splitting.
"""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    """Split a list of documents into smaller overlapping chunks.

    Args:
        documents: List of LangChain Document objects to split.
        chunk_size: Maximum number of characters per chunk. Defaults to
            the value from application settings (500 tokens ≈ chars).
        chunk_overlap: Number of overlapping characters between adjacent
            chunks. Defaults to application settings value.

    Returns:
        List of chunked Document objects with preserved metadata and an
        additional ``chunk_index`` metadata field.

    Raises:
        ValueError: If documents list is empty or chunk_size is invalid.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Document] = []
    for doc in documents:
        doc_chunks = splitter.split_documents([doc])
        for idx, chunk in enumerate(doc_chunks):
            chunk.metadata["chunk_index"] = idx
            chunks.append(chunk)

    logger.info(
        "Chunked %d document(s) into %d chunk(s) "
        "(size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
