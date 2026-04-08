"""Logging configuration for the RAGify application."""

import logging
import sys
from typing import Optional

from config.settings import get_settings


def configure_logging(level: Optional[str] = None) -> None:
    """Configure root logger with a structured format.

    Args:
        level: Optional log level override (e.g. ``"DEBUG"``).
               Falls back to the application settings value.
    """
    settings = get_settings()
    log_level = level or settings.log_level

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    # Avoid duplicate handlers when called multiple times
    if not root_logger.handlers:
        root_logger.addHandler(handler)
