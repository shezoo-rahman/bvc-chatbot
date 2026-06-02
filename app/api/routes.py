from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/api")


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest):
    assistant = request.app.state.assistant
    try:
        answer = await assistant.answer_query(body.question, session_id=body.session_id)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process query: {e}")
