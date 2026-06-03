import json
from unittest.mock import AsyncMock

import httpx
import pytest

from app.models.schemas import CompanyNews, CompanyProfile, StockQuote, SymbolSearchResult
from app.services.stock_data import (
    RateLimitError,
    StockDataService,
    SymbolNotFoundError,
)


def make_response(data: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=data,
        request=httpx.Request("GET", "https://finnhub.io/api/v1/test"),
    )


@pytest.fixture
def service():
    http_client = AsyncMock()
    return StockDataService(api_key="test-key", http_client=http_client)


@pytest.mark.asyncio
async def test_get_quote_success(service):
    service.http_client.get.return_value = make_response(
        {
            "c": 178.50,
            "d": 2.30,
            "dp": 1.31,
            "h": 179.20,
            "l": 176.10,
            "o": 176.80,
            "pc": 176.20,
            "t": 1700000000,
        }
    )
    quote = await service.get_quote("AAPL")
    assert quote.symbol == "AAPL"
    assert quote.current_price == 178.50
    assert quote.change == 2.30
    assert quote.percent_change == 1.31


@pytest.mark.asyncio
async def test_get_quote_invalid_symbol(service):
    service.http_client.get.return_value = make_response(
        {
            "c": 0,
            "d": None,
            "dp": None,
            "h": 0,
            "l": 0,
            "o": 0,
            "pc": 0,
            "t": 0,
        }
    )
    with pytest.raises(SymbolNotFoundError):
        await service.get_quote("INVALID")


@pytest.mark.asyncio
async def test_get_quote_rate_limit(service):
    service.http_client.get.return_value = make_response({}, status_code=429)
    with pytest.raises(RateLimitError):
        await service.get_quote("AAPL")


@pytest.mark.asyncio
async def test_get_company_profile_success(service):
    service.http_client.get.return_value = make_response(
        {
            "name": "Apple Inc",
            "finnhubIndustry": "Technology",
            "marketCapitalization": 2800000,
            "country": "US",
            "currency": "USD",
            "exchange": "NASDAQ",
            "ipo": "1980-12-12",
            "logo": "https://example.com/logo.png",
            "weburl": "https://apple.com",
            "ticker": "AAPL",
        }
    )
    profile = await service.get_company_profile("AAPL")
    assert profile.name == "Apple Inc"
    assert profile.industry == "Technology"
    assert profile.market_cap == 2800000


@pytest.mark.asyncio
async def test_get_company_profile_not_found(service):
    service.http_client.get.return_value = make_response({})
    with pytest.raises(SymbolNotFoundError):
        await service.get_company_profile("INVALID")


@pytest.mark.asyncio
async def test_get_company_news_success(service):
    service.http_client.get.return_value = make_response(
        [
            {
                "headline": "Test headline",
                "summary": "Test summary",
                "source": "Reuters",
                "url": "https://example.com",
                "datetime": 1700000000,
            }
        ]
    )
    news = await service.get_company_news("AAPL")
    assert len(news) == 1
    assert news[0].headline == "Test headline"


@pytest.mark.asyncio
async def test_get_company_news_empty(service):
    service.http_client.get.return_value = make_response([])
    news = await service.get_company_news("AAPL")
    assert news == []


@pytest.mark.asyncio
async def test_search_symbol_success(service):
    service.http_client.get.return_value = make_response(
        {
            "count": 2,
            "result": [
                {"symbol": "AAPL", "description": "Apple Inc", "type": "Common Stock"},
                {"symbol": "APLE", "description": "Apple Hospitality REIT", "type": "Common Stock"},
            ],
        }
    )
    results = await service.search_symbol("Apple")
    assert len(results) == 2
    assert results[0].symbol == "AAPL"
    assert results[1].symbol == "APLE"


@pytest.mark.asyncio
async def test_search_symbol_no_results(service):
    service.http_client.get.return_value = make_response({"count": 0, "result": []})
    results = await service.search_symbol("XYZRANDOM123")
    assert results == []


@pytest.mark.asyncio
async def test_search_symbol_filters_international_keeps_share_classes(service):
    service.http_client.get.return_value = make_response(
        {
            "count": 4,
            "result": [
                {"symbol": "BRK.A", "description": "Berkshire Hathaway", "type": "Common Stock"},
                {"symbol": "BRK.B", "description": "Berkshire Hathaway", "type": "Common Stock"},
                {"symbol": "AMC.TO", "description": "Arizona Metals", "type": "Common Stock"},
                {"symbol": "AMC.F", "description": "Albemarle Corp", "type": "Common Stock"},
            ],
        }
    )
    results = await service.search_symbol("Berkshire")
    assert len(results) == 2
    assert results[0].symbol == "BRK.A"
    assert results[1].symbol == "BRK.B"


@pytest.mark.asyncio
async def test_symbol_uppercased(service):
    service.http_client.get.return_value = make_response(
        {
            "c": 100,
            "d": 1,
            "dp": 1,
            "h": 101,
            "l": 99,
            "o": 100,
            "pc": 99,
            "t": 0,
        }
    )
    quote = await service.get_quote("aapl")
    assert quote.symbol == "AAPL"


def test_format_quote():
    quote = StockQuote(
        symbol="AAPL",
        current_price=178.5,
        change=2.3,
        percent_change=1.31,
        high=179.2,
        low=176.1,
        open=176.8,
        previous_close=176.2,
    )
    result = json.loads(StockDataService.format_quote(quote))
    assert result["symbol"] == "AAPL"
    assert result["current_price"] == 178.5


def test_format_profile():
    profile = CompanyProfile(
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
    result = json.loads(StockDataService.format_profile(profile))
    assert result["name"] == "Apple Inc"
    assert result["market_cap"] == 2800000


def test_format_news():
    news = [
        CompanyNews(
            headline="Test",
            summary="Summary",
            source="Reuters",
            url="https://example.com",
            datetime=1700000000,
        )
    ]
    result = json.loads(StockDataService.format_news(news))
    assert len(result) == 1
    assert result[0]["headline"] == "Test"


def test_format_search_results():
    results = [SymbolSearchResult(symbol="AAPL", description="Apple Inc")]
    result = json.loads(StockDataService.format_search_results(results))
    assert len(result) == 1
    assert result[0]["symbol"] == "AAPL"


def test_format_search_results_empty():
    result = json.loads(StockDataService.format_search_results([]))
    assert result["message"] == "No matching symbols found."
