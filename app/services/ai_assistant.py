import asyncio
import json
import logging
import time

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

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5


class AIAssistant:
    """Orchestrates OpenAI function calling with Finnhub stock data tools.

    Manages multi-turn conversations per session and validates LLM responses
    against tool results to prevent hallucinated financial data.
    """

    def __init__(
        self, openai_client: AsyncOpenAI, stock_service: StockDataService, model: str
    ) -> None:
        self.client = openai_client
        self.stock_service = stock_service
        self.model = model
        self._sessions: dict[str, list[dict[str, str]]] = {}

    async def answer_query(self, question: str, session_id: str = "") -> str:
        logger.info("Query received: %s (session=%s)", question[:80], session_id[:8] or "none")
        start = time.perf_counter()

        history = self._sessions.get(session_id, []) if session_id else []

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history[-20:],
            {"role": "user", "content": question},
        ]
        tool_results = []

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

            results = await asyncio.gather(*[self._execute_tool(tc) for tc in message.tool_calls])

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
            if len(self._sessions) >= 200:
                oldest = next(iter(self._sessions))
                del self._sessions[oldest]
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": "user", "content": question})
        self._sessions[session_id].append({"role": "assistant", "content": answer})

    async def _execute_tool(self, tool_call: ChatCompletionMessageToolCall) -> str:
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON arguments for tool %s", name)
            return json.dumps({"error": "Invalid tool arguments"})

        start = time.perf_counter()
        try:
            result = await self._dispatch_tool(name, args)
            return result
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

    async def _dispatch_tool(self, name: str, args: dict) -> str:
        if name == "search_symbol":
            results = await self.stock_service.search_symbol(args["query"])
            if not results:
                return json.dumps({"message": "No matching symbols found."})
            return json.dumps([r.model_dump() for r in results], indent=2)
        elif name == "get_stock_quote":
            quote = await self.stock_service.get_quote(args["symbol"])
            return quote.model_dump_json(indent=2)
        elif name == "get_company_profile":
            profile = await self.stock_service.get_company_profile(args["symbol"])
            return profile.model_dump_json(indent=2)
        elif name == "get_company_news":
            news = await self.stock_service.get_company_news(args["symbol"])
            return json.dumps([n.model_dump() for n in news], indent=2)
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})
