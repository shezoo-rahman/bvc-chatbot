from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class QueryResponse(BaseModel):
    answer: str


class StockQuote(BaseModel):
    symbol: str
    current_price: float
    change: float
    percent_change: float
    high: float
    low: float
    open: float
    previous_close: float


class CompanyProfile(BaseModel):
    symbol: str
    name: str
    industry: str
    market_cap: float
    country: str
    currency: str
    exchange: str
    ipo_date: str
    logo: str
    web_url: str


class CompanyNews(BaseModel):
    headline: str
    summary: str
    source: str
    url: str
    datetime: int
