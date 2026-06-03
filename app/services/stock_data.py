import json
from datetime import datetime, timedelta

import httpx

from app.models.schemas import CompanyNews, CompanyProfile, StockQuote, SymbolSearchResult


class StockDataError(Exception):
    pass


class SymbolNotFoundError(StockDataError):
    pass


class RateLimitError(StockDataError):
    pass


class StockDataService:
    """Async client for the Finnhub API.

    Provides stock quotes, company profiles, news, and symbol search.
    Raises SymbolNotFoundError or RateLimitError on expected failures
    so the AI assistant can relay errors to the user naturally.
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str, http_client: httpx.AsyncClient) -> None:
        self.api_key: str = api_key
        self.http_client: httpx.AsyncClient = http_client

    async def get_quote(self, symbol: str) -> StockQuote:
        symbol = symbol.upper().strip()
        response: httpx.Response = await self.http_client.get(
            f"{self.BASE_URL}/quote",
            params={"symbol": symbol, "token": self.api_key},
        )
        self._check_rate_limit(response)
        data: dict = response.json()

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
        response: httpx.Response = await self.http_client.get(
            f"{self.BASE_URL}/stock/profile2",
            params={"symbol": symbol, "token": self.api_key},
        )
        self._check_rate_limit(response)
        data: dict = response.json()

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
        today: str = datetime.now().strftime("%Y-%m-%d")
        week_ago: str = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        response: httpx.Response = await self.http_client.get(
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

        articles: list[CompanyNews] = []
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

    async def search_symbol(self, query: str, max_results: int = 5) -> list[SymbolSearchResult]:
        response: httpx.Response = await self.http_client.get(
            f"{self.BASE_URL}/search",
            params={"q": query, "token": self.api_key},
        )
        self._check_rate_limit(response)
        data: dict = response.json()

        results: list[SymbolSearchResult] = []
        for item in data.get("result", []):
            symbol: str = item.get("symbol", "")
            if item.get("type") != "Common Stock":
                continue
            # Keep US share classes (.A, .B) but skip all exchange suffixes
            if "." in symbol:
                suffix = symbol.split(".")[-1]
                if suffix not in ("A", "B"):
                    continue
            results.append(
                SymbolSearchResult(
                    symbol=symbol,
                    description=item.get("description", ""),
                )
            )
            if len(results) >= max_results:
                break
        return results

    @staticmethod
    def format_search_results(results: list[SymbolSearchResult]) -> str:
        if not results:
            return json.dumps({"message": "No matching symbols found."})
        return json.dumps([r.model_dump() for r in results], indent=2)

    def _check_rate_limit(self, response: httpx.Response) -> None:
        if response.status_code == 429:
            raise RateLimitError("Finnhub API rate limit exceeded. Please try again shortly.")
        if response.status_code == 403:
            raise StockDataError(
                "This stock is not available on the free data plan. "
                "Only US-listed stocks are supported."
            )
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
