"""Health-check router.

GET /health – returns service status and basic metrics.
"""

from fastapi import APIRouter

from app.utils.schemas import HealthResponse
from config.settings import get_settings

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return the health status of the RAGify service.

    Includes the application version and the current count of indexed
    document chunks.
    """
    from app.main import get_vector_store_instance

    settings = get_settings()
    store = get_vector_store_instance()

    return HealthResponse(
        status="ok",
        version=settings.app_version,
        indexed_documents=store.document_count,
    )
