"""Embedding module.

Generates dense vector embeddings using SentenceTransformers with local caching.
"""

import logging
import os
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import get_settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Reusable wrapper around SentenceTransformer for generating embeddings.

    Caches the model in memory after first load. Embeddings are generated
    in batches for efficiency.

    Args:
        model_name: HuggingFace model name or local path.
        cache_dir: Directory for caching the downloaded model weights.
    """

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.embedding_model
        self._cache_dir = cache_dir or settings.embedding_cache_dir

        os.makedirs(self._cache_dir, exist_ok=True)
        logger.info("Loading embedding model: %s", self._model_name)
        self._model = SentenceTransformer(
            self._model_name, cache_folder=self._cache_dir
        )
        logger.info("Embedding model loaded successfully")

    @property
    def model_name(self) -> str:
        """Return the name of the underlying embedding model."""
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        return self._model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of text strings.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts to process per batch.

        Returns:
            NumPy array of shape (len(texts), embedding_dim).

        Raises:
            ValueError: If texts list is empty.
        """
        if not texts:
            raise ValueError("Cannot embed empty list of texts")

        logger.debug("Embedding %d text(s) (batch_size=%d)", len(texts), batch_size)
        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate an embedding for a single query string.

        Args:
            query: The query text to embed.

        Returns:
            1-D NumPy array of shape (embedding_dim,).

        Raises:
            ValueError: If query is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty")

        return self.embed_texts([query])[0]


# Module-level singleton (lazy-initialized)
_default_model: EmbeddingModel | None = None


def get_embedding_model() -> EmbeddingModel:
    """Return the shared EmbeddingModel singleton.

    Creates the model on the first call; subsequent calls return the cached instance.
    """
    global _default_model
    if _default_model is None:
        _default_model = EmbeddingModel()
    return _default_model
