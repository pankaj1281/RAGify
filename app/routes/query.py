"""Query API router.

GET /query?q=<question> – answer a question using the RAG pipeline.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.exceptions import DocumentNotFoundError, EmptyQueryError
from app.services.query_service import QueryService
from app.utils.schemas import QueryResponse, SourceCitation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


def _get_query_service() -> QueryService:
    """Dependency: return the application-level QueryService."""
    from app.main import get_query_service  # avoid circular import

    return get_query_service()


@router.get("/", response_model=QueryResponse)
async def query_documents(
    q: str = Query(..., min_length=1, description="Natural-language question"),
    k: Optional[int] = Query(None, ge=1, le=20, description="Number of context documents"),
    rewrite: bool = Query(False, description="Rewrite query using LLM before retrieval"),
    hybrid: bool = Query(False, description="Use hybrid BM25+vector retrieval"),
    service: QueryService = Depends(_get_query_service),
) -> QueryResponse:
    """Query the indexed documents using the RAG pipeline.

    Returns a generated answer grounded in the retrieved context, along with
    source citations and latency metadata.
    """
    try:
        result = service.answer(
            question=q,
            k=k,
            rewrite=rewrite,
            use_hybrid=hybrid,
        )
    except EmptyQueryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Query endpoint error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    sources = [SourceCitation(**s) for s in result.get("sources", [])]

    return QueryResponse(
        question=q,
        answer=result["answer"],
        sources=sources,
        retrieved_docs=result["retrieved_docs"],
        latency_ms=result["latency_ms"],
        rewritten_query=result.get("rewritten_query"),
    )
