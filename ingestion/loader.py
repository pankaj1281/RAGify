"""Document loader module.

Supports loading PDF, TXT, and DOCX files with preprocessing.
"""

import logging
import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def load_document(file_path: str) -> List[Document]:
    """Load a single document from the given file path.

    Supports PDF, TXT, and DOCX file formats.

    Args:
        file_path: Absolute or relative path to the document file.

    Returns:
        A list of LangChain Document objects with page content and metadata.

    Raises:
        ValueError: If the file format is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{extension}'. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    logger.info("Loading document: %s (type=%s)", file_path, extension)

    loader_map = {
        ".pdf": lambda p: PyPDFLoader(str(p)),
        ".txt": lambda p: TextLoader(str(p), encoding="utf-8"),
        ".docx": lambda p: Docx2txtLoader(str(p)),
    }

    loader = loader_map[extension](path)
    documents = loader.load()

    # Enrich metadata with source filename
    for doc in documents:
        doc.metadata["source"] = path.name
        doc.metadata["file_path"] = str(path.resolve())
        doc.page_content = _preprocess_text(doc.page_content)

    logger.info("Loaded %d page(s) from %s", len(documents), path.name)
    return documents


def load_directory(directory: str) -> List[Document]:
    """Load all supported documents from a directory.

    Args:
        directory: Path to the directory containing documents.

    Returns:
        Combined list of LangChain Document objects from all supported files.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    all_documents: List[Document] = []
    files = [
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        logger.warning("No supported documents found in %s", directory)
        return all_documents

    for file_path in files:
        try:
            docs = load_document(str(file_path))
            all_documents.extend(docs)
        except Exception as exc:
            logger.error("Failed to load %s: %s", file_path.name, exc)

    logger.info(
        "Loaded %d document(s) from %d file(s) in %s",
        len(all_documents),
        len(files),
        directory,
    )
    return all_documents


def _preprocess_text(text: str) -> str:
    """Clean and normalize document text.

    Removes excessive whitespace and normalizes line breaks.

    Args:
        text: Raw text content.

    Returns:
        Cleaned text string.
    """
    if not text:
        return text
    # Collapse multiple newlines into two
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
