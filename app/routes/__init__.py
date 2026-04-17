"""App routes package."""

from .health import router as health_router
from .ingest import router as ingest_router
from .query import router as query_router
from .ui import router as ui_router

__all__ = ["ingest_router", "query_router", "health_router", "ui_router"]
