from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from app.api.routes import router


@pytest.fixture
def client():
    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    test_app = FastAPI(lifespan=noop_lifespan)
    test_app.include_router(router)

    static_dir = Path(__file__).parent.parent / "static"
    test_app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @test_app.get("/")
    async def root():
        return FileResponse(str(static_dir / "index.html"))

    test_app.state.assistant = AsyncMock()
    test_app.state.assistant.answer_query = AsyncMock(return_value="AAPL is at $178.50")

    with TestClient(test_app, raise_server_exceptions=False) as c:
        yield c


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_success(client):
    response = client.post("/api/query", json={"question": "How is AAPL doing?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "178.50" in data["answer"]


def test_query_empty_question(client):
    response = client.post("/api/query", json={"question": ""})
    assert response.status_code == 422


def test_query_missing_question(client):
    response = client.post("/api/query", json={})
    assert response.status_code == 422


def test_query_error(client):
    client.app.state.assistant.answer_query = AsyncMock(side_effect=Exception("API error"))
    response = client.post("/api/query", json={"question": "test"})
    assert response.status_code == 500


def test_root_serves_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Stock Insights Assistant" in response.text
