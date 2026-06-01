from unittest.mock import AsyncMock

import pytest

from app.models.schemas import CompanyNews, CompanyProfile, StockQuote
from app.services.ai_assistant import AIAssistant
from app.services.stock_data import StockDataService


@pytest.fixture
def sample_quote():
    return StockQuote(
        symbol="AAPL",
        current_price=178.50,
        change=2.30,
        percent_change=1.31,
        high=179.20,
        low=176.10,
        open=176.80,
        previous_close=176.20,
    )


@pytest.fixture
def sample_profile():
    return CompanyProfile(
        symbol="AAPL",
        name="Apple Inc",
        industry="Technology",
        market_cap=2800000,
        country="US",
        currency="USD",
        exchange="NASDAQ",
        ipo_date="1980-12-12",
        logo="https://example.com/logo.png",
        web_url="https://apple.com",
    )


@pytest.fixture
def sample_news():
    return [
        CompanyNews(
            headline="Apple releases new product",
            summary="Apple announced a new product line today.",
            source="Reuters",
            url="https://example.com/news/1",
            datetime=1700000000,
        )
    ]


@pytest.fixture
def mock_http_client():
    return AsyncMock()


@pytest.fixture
def mock_stock_service(sample_quote, sample_profile, sample_news):
    service = AsyncMock(spec=StockDataService)
    service.get_quote = AsyncMock(return_value=sample_quote)
    service.get_company_profile = AsyncMock(return_value=sample_profile)
    service.get_company_news = AsyncMock(return_value=sample_news)
    service.format_quote = StockDataService.format_quote
    service.format_profile = StockDataService.format_profile
    service.format_news = StockDataService.format_news
    return service


@pytest.fixture
def mock_openai_client():
    return AsyncMock()


@pytest.fixture
def assistant(mock_openai_client, mock_stock_service):
    return AIAssistant(
        openai_client=mock_openai_client,
        stock_service=mock_stock_service,
        model="gpt-4o",
    )
