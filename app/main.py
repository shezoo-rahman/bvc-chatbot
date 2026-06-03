import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

from app.api.routes import router
from app.config import settings
from app.services.ai_assistant import AIAssistant
from app.services.stock_data import StockDataService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    http_client = httpx.AsyncClient(timeout=30.0)
    stock_service = StockDataService(api_key=settings.finnhub_api_key, http_client=http_client)
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    assistant = AIAssistant(
        openai_client=openai_client,
        stock_service=stock_service,
        model=settings.openai_model,
    )

    app.state.http_client = http_client
    app.state.stock_service = stock_service
    app.state.assistant = assistant

    logger.info("Stock Insights Assistant ready at http://localhost:8000")

    yield

    await http_client.aclose()


app = FastAPI(title="Stock Insights Assistant", lifespan=lifespan)
app.include_router(router)


@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if request.url.path.startswith("/api"):
        logger.info(
            "%s %s %d %.0fms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
    return response


static_dir = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root() -> Response:
    return FileResponse(str(static_dir / "index.html"))
