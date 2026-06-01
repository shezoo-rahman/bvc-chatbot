import json
from datetime import datetime, timedelta

import httpx

from app.models.schemas import CompanyNews, CompanyProfile, StockQuote


class StockDataError(Exception):
    pass


class SymbolNotFoundError(StockDataError):
    pass


class RateLimitError(StockDataError):
    pass


class StockDataService:
    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str, http_client: httpx.AsyncClient):
        self.api_key = api_key
        self.http_client = http_client

    async def get_quote(self, symbol: str) -> StockQuote:
        symbol = symbol.upper().strip()
        response = await self.http_client.get(
            f"{self.BASE_URL}/quote",
            params={"symbol": symbol, "token": self.api_key},
        )
        self._check_rate_limit(response)
        data = response.json()

        if data.get("c", 0) == 0 and data.get("h", 0) == 0:
            raise SymbolNotFoundError(f"No data found for symbol '{symbol}'")

        return StockQuote(
            symbol=symbol,
            current_price=data["c"],
            change=data["d"],
            percent_change=data["dp"],
            high=data["h"],
            low=data["l"],
            open=data["o"],
            previous_close=data["pc"],
        )

    async def get_company_profile(self, symbol: str) -> CompanyProfile:
        symbol = symbol.upper().strip()
        response = await self.http_client.get(
            f"{self.BASE_URL}/stock/profile2",
            params={"symbol": symbol, "token": self.api_key},
        )
        self._check_rate_limit(response)
        data = response.json()

        if not data or not data.get("name"):
            raise SymbolNotFoundError(f"No profile found for symbol '{symbol}'")

        return CompanyProfile(
            symbol=symbol,
            name=data.get("name", ""),
            industry=data.get("finnhubIndustry", ""),
            market_cap=data.get("marketCapitalization", 0),
            country=data.get("country", ""),
            currency=data.get("currency", ""),
            exchange=data.get("exchange", ""),
            ipo_date=data.get("ipo", ""),
            logo=data.get("logo", ""),
            web_url=data.get("weburl", ""),
        )

    async def get_company_news(self, symbol: str, max_articles: int = 5) -> list[CompanyNews]:
        symbol = symbol.upper().strip()
        today = datetime.now().strftime("%Y-%m-%d")
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        response = await self.http_client.get(
            f"{self.BASE_URL}/company-news",
            params={
                "symbol": symbol,
                "from": week_ago,
                "to": today,
                "token": self.api_key,
            },
        )
        self._check_rate_limit(response)
        data = response.json()

        if not isinstance(data, list):
            return []

        articles = []
        for item in data[:max_articles]:
            articles.append(
                CompanyNews(
                    headline=item.get("headline", ""),
                    summary=item.get("summary", ""),
                    source=item.get("source", ""),
                    url=item.get("url", ""),
                    datetime=item.get("datetime", 0),
                )
            )
        return articles

    def _check_rate_limit(self, response: httpx.Response) -> None:
        if response.status_code == 429:
            raise RateLimitError("Finnhub API rate limit exceeded. Please try again shortly.")
        response.raise_for_status()

    @staticmethod
    def format_quote(quote: StockQuote) -> str:
        return json.dumps(quote.model_dump(), indent=2)

    @staticmethod
    def format_profile(profile: CompanyProfile) -> str:
        return json.dumps(profile.model_dump(), indent=2)

    @staticmethod
    def format_news(news: list[CompanyNews]) -> str:
        return json.dumps([n.model_dump() for n in news], indent=2)
