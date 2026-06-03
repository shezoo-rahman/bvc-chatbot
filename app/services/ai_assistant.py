import asyncio
import json
import logging
import time
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageToolCall

from app.services.prompts import SYSTEM_PROMPT, TOOLS
from app.services.stock_data import (
    RateLimitError,
    StockDataError,
    StockDataService,
    SymbolNotFoundError,
)
from app.services.validation import validate_response

logger: logging.Logger = logging.getLogger(__name__)

MAX_ITERATIONS: int = 5
MAX_HISTORY_MESSAGES: int = 20


class AIAssistant:
    """Orchestrates OpenAI function calling with Finnhub stock data tools.

    Manages multi-turn conversations per session and validates LLM responses
    against tool results to prevent hallucinated financial data.
    """

    def __init__(
        self, openai_client: AsyncOpenAI, stock_service: StockDataService, model: str
    ) -> None:
        self.client: AsyncOpenAI = openai_client
        self.stock_service: StockDataService = stock_service
        self.model: str = model
        self._sessions: dict[str, list[dict[str, str]]] = {}

    async def answer_query(self, question: str, session_id: str = "") -> str:
        logger.info("Query received: %s (session=%s)", question[:80], session_id[:8] or "none")
        start = time.perf_counter()

        history: list[dict[str, str]] = self._sessions.get(session_id, []) if session_id else []

        messages: list[dict[str, Any]] = [
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
                final_text: str = message.content or "I wasn't able to generate a response."
                validated_text, was_valid = validate_response(final_text, tool_results)
                if not was_valid:
                    logger.warning("Validation flagged ungrounded numbers in response")
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.info("Query completed in %.0fms (%d iterations)", elapsed_ms, iteration + 1)
                self._save_turn(session_id, question, validated_text)
                return validated_text

            messages.append(message)
            tool_names: list[str] = [tc.function.name for tc in message.tool_calls]
            logger.info("Tool calls requested: %s", ", ".join(tool_names))

            tasks: list[asyncio.Task[str]] = [
                self._execute_tool(tool_call) for tool_call in message.tool_calls
            ]

            results: list[str] = await asyncio.gather(*tasks)

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
        fallback: str = (
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

    async def _execute_tool(self, tool_call: ChatCompletionMessageToolCall) -> str:
        name: str = tool_call.function.name
        try:
            args: dict[str, Any] = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON arguments for tool %s", name)
            return json.dumps({"error": "Invalid tool arguments"})

        start: float = time.perf_counter()
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
            elapsed_ms: float = (time.perf_counter() - start) * 1000
            logger.info("Tool %s(%s) completed in %.0fms", name, args, elapsed_ms)
