import asyncio
import json
import logging

from openai import AsyncOpenAI

from app.services.stock_data import (
    RateLimitError,
    StockDataError,
    StockDataService,
    SymbolNotFoundError,
)
from app.services.validation import validate_response

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (  # noqa: E501
    "You are a helpful stock market analyst assistant. "
    "You provide clear, accurate information about stocks "
    "using real-time data from your available tools.\n\n"
    "CRITICAL RULES:\n"
    "- ONLY cite numbers, prices, percentages, and financial data "
    "that come directly from tool results.\n"
    "- NEVER invent, estimate, or hallucinate prices, percentages, "
    "market caps, or any numerical financial data.\n"
    "- If a tool call fails or returns an error, clearly tell the user "
    "you couldn't fetch that data. Do NOT make up numbers instead.\n"
    "- Use standard ticker symbols "
    "(e.g., AAPL for Apple, TSLA for Tesla, GOOGL for Alphabet).\n"
    "- When comparing stocks, present data in a clear, structured format.\n"
    "- Keep responses concise and actionable.\n\n"
    "EXAMPLE INTERACTION:\n"
    'User: "How is AAPL doing today?"\n'
    '[You call get_stock_quote with symbol "AAPL"]\n'
    "[Tool returns: current_price: 178.50, change: 2.30, "
    "percent_change: 1.31, high: 179.20, low: 176.10]\n"
    'Response: "Apple (AAPL) is trading at **$178.50**, '
    "up **$2.30 (+1.31%)** today. The stock has ranged between "
    'a low of $176.10 and a high of $179.20 during the session."\n\n'
    "Note how the response ONLY uses numbers from the tool result. "
    "Always follow this pattern."
)

TOOLS = [
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


class AIAssistant:
    def __init__(self, openai_client: AsyncOpenAI, stock_service: StockDataService, model: str):
        self.client = openai_client
        self.stock_service = stock_service
        self.model = model

    async def answer_query(self, question: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        tool_results: list[str] = []

        for _ in range(MAX_ITERATIONS):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            message = response.choices[0].message

            if not message.tool_calls:
                final_text = message.content or "I wasn't able to generate a response."
                validated_text, _ = validate_response(final_text, tool_results)
                return validated_text

            messages.append(message)

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

        return (
            "I gathered some data but couldn't complete the analysis. "
            "Please try a simpler question."
        )

    async def _execute_tool(self, tool_call) -> str:
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid tool arguments"})

        try:
            if name == "get_stock_quote":
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
            return json.dumps({"error": str(e)})
        except RateLimitError as e:
            return json.dumps({"error": str(e)})
        except StockDataError as e:
            return json.dumps({"error": f"Failed to fetch data: {e}"})
        except Exception as e:
            logger.exception("Unexpected error executing tool %s", name)
            return json.dumps({"error": f"An unexpected error occurred: {e}"})
