import logging

from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    assistant = request.app.state.assistant
    try:
        answer = await assistant.answer_query(body.question, session_id=body.session_id)
        return QueryResponse(answer=answer)
    except Exception:
        logger.exception("Failed to process query")
        raise HTTPException(status_code=500, detail="Failed to process query")
