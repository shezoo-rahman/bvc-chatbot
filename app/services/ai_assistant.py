import asyncio
import json
import logging
import time

from openai import AsyncOpenAI

from app.services.stock_data import (
    RateLimitError,
    StockDataError,
    StockDataService,
    SymbolNotFoundError,
)
from app.services.validation import validate_response

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a helpful stock market analyst assistant. You provide clear, \
accurate information about stocks using real-time data from your tools.

RULES:
- You ONLY answer questions about stocks, companies, and financial markets. \
If the user asks about anything else (weather, recipes, travel, coding, etc.), \
reply: "I can only help with stock and market data. Try asking about a stock \
quote, company profile, or market news."
- You can ONLY access US-listed stocks. If the user asks about international \
stocks (e.g. TSCO.L, BMW.DE), tell them only US stocks are supported.
- You only have current-day data (price, change, high, low, open). You do NOT \
have historical price data or charts. If asked about past performance or trends, \
explain this limitation.
- ONLY cite numbers that come directly from tool results. NEVER invent prices, \
percentages, or financial data.
- If a tool call fails, tell the user. Do NOT make up numbers.
- ALWAYS present stock data in a markdown table. For a single stock use two \
columns (Metric | Value). For comparisons use columns per stock \
(Metric | AAPL | MSFT). Include price, change, percent change, high, low, \
open, and previous close.
- When presenting news, briefly assess relevance. If an article only mentions \
the company in passing, note that. Prioritise articles that are directly about \
the company.
- NEVER include images or logos in responses. Ignore any logo URLs from tool data.
- When a user asks about a company (e.g. "tell me about X"), ALWAYS provide all \
three: (1) a brief company summary from the profile, (2) current stock data in \
a table, and (3) recent news. Call get_company_profile, get_stock_quote, and \
get_company_news together.
- Keep responses concise but insightful. After presenting data for any stock \
(single or comparison), always include a brief 1-2 sentence analyst-style \
takeaway. Focus on what matters: is it up or down, by how much, and any \
notable signals (e.g. trading near the high/low, large gap from previous \
close). No filler or generic statements.

SYMBOL LOOKUP:
- When the user gives a company name (e.g. "Apple", "Berkshire Hathaway") \
instead of a ticker, call search_symbol to find the right ticker first.
- When search_symbol returns 2+ results, list them and ask the user to pick. \
Do NOT silently choose one. Example:
  User: "AMC stock" → search returns AMC and AMCX → you reply:
  "I found multiple matches: **AMC** (AMC Entertainment) and \
**AMCX** (AMC Networks). Which one would you like?"
- CRITICAL: Once you have presented options and the user replies with a ticker \
or company name from those options, call get_stock_quote with that ticker \
immediately. NEVER call search_symbol a second time in the same conversation.
- For clear, unambiguous tickers (AAPL, MSFT, TSLA, AMZN, GOOGL), skip search \
and fetch data directly.
- For stocks with multiple share classes (e.g. Berkshire Hathaway has BRK.A \
and BRK.B), ALWAYS call search_symbol and present all classes to the user. \
Never silently pick one — the price difference can be enormous."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_symbol",
            "description": (
                "Search for stock ticker symbols by company name or partial symbol. "
                "Use when the user provides a company name instead of a ticker, "
                "or when the symbol could be ambiguous."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Company name or partial symbol (e.g. 'Apple', 'AMC')",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_quote",
            "description": (
                "Get the current stock quote for a given ticker symbol, "
                "including price, change, high, low, and volume."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., AAPL, TSLA, GOOGL)",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_profile",
            "description": (
                "Get company profile information including name, industry, "
                "market cap, country, and exchange."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., AAPL, TSLA, GOOGL)",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_news",
            "description": (
                "Get recent news articles for a company. "
                "Returns up to 5 recent headlines with summaries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., AAPL, TSLA, GOOGL)",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
]

MAX_ITERATIONS = 5
MAX_HISTORY_MESSAGES = 20


class AIAssistant:
    def __init__(self, openai_client: AsyncOpenAI, stock_service: StockDataService, model: str):
        self.client = openai_client
        self.stock_service = stock_service
        self.model = model
        self._sessions: dict[str, list[dict]] = {}

    async def answer_query(self, question: str, session_id: str = "") -> str:
        logger.info("Query received: %s (session=%s)", question[:80], session_id[:8] or "none")
        start = time.perf_counter()

        history = self._sessions.get(session_id, []) if session_id else []

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history[-MAX_HISTORY_MESSAGES:],
            {"role": "user", "content": question},
        ]
        tool_results: list[str] = []

        for iteration in range(MAX_ITERATIONS):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            message = response.choices[0].message

            if not message.tool_calls:
                final_text = message.content or "I wasn't able to generate a response."
                validated_text, was_valid = validate_response(final_text, tool_results)
                if not was_valid:
                    logger.warning("Validation flagged ungrounded numbers in response")
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.info("Query completed in %.0fms (%d iterations)", elapsed_ms, iteration + 1)
                self._save_turn(session_id, question, validated_text)
                return validated_text

            messages.append(message)
            tool_names = [tc.function.name for tc in message.tool_calls]
            logger.info("Tool calls requested: %s", ", ".join(tool_names))

            tasks = []
            for tool_call in message.tool_calls:
                tasks.append(self._execute_tool(tool_call))

            results = await asyncio.gather(*tasks)

            for tool_call, result in zip(message.tool_calls, results):
                tool_results.append(result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

        logger.warning("Max iterations reached for query: %s", question[:80])
        fallback = (
            "I gathered some data but couldn't complete the analysis. "
            "Please try a simpler question."
        )
        self._save_turn(session_id, question, fallback)
        return fallback

    def _save_turn(self, session_id: str, question: str, answer: str) -> None:
        if not session_id:
            return
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": "user", "content": question})
        self._sessions[session_id].append({"role": "assistant", "content": answer})

    async def _execute_tool(self, tool_call) -> str:
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON arguments for tool %s", name)
            return json.dumps({"error": "Invalid tool arguments"})

        start = time.perf_counter()
        try:
            if name == "search_symbol":
                results = await self.stock_service.search_symbol(args["query"])
                return StockDataService.format_search_results(results)
            elif name == "get_stock_quote":
                quote = await self.stock_service.get_quote(args["symbol"])
                return StockDataService.format_quote(quote)
            elif name == "get_company_profile":
                profile = await self.stock_service.get_company_profile(args["symbol"])
                return StockDataService.format_profile(profile)
            elif name == "get_company_news":
                news = await self.stock_service.get_company_news(args["symbol"])
                return StockDataService.format_news(news)
            else:
                return json.dumps({"error": f"Unknown tool: {name}"})
        except SymbolNotFoundError as e:
            logger.info("Symbol not found: %s", e)
            return json.dumps({"error": str(e)})
        except RateLimitError as e:
            logger.warning("Rate limit hit: %s", e)
            return json.dumps({"error": str(e)})
        except StockDataError as e:
            logger.error("Stock data error in %s: %s", name, e)
            return json.dumps({"error": f"Failed to fetch data: {e}"})
        except Exception as e:
            logger.exception("Unexpected error executing tool %s", name)
            return json.dumps({"error": f"An unexpected error occurred: {e}"})
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info("Tool %s(%s) completed in %.0fms", name, args, elapsed_ms)
