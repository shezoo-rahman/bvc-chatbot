import json
from unittest.mock import AsyncMock, MagicMock

import pytest


def make_text_response(content: str):
    """Create a mock OpenAI response with text content."""
    message = MagicMock()
    message.tool_calls = None
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def make_tool_call_response(tool_calls: list[dict]):
    """Create a mock OpenAI response with tool calls."""
    mock_tool_calls = []
    for tc in tool_calls:
        mock_tc = MagicMock()
        mock_tc.id = tc["id"]
        mock_tc.function.name = tc["name"]
        mock_tc.function.arguments = json.dumps(tc["arguments"])
        mock_tool_calls.append(mock_tc)

    message = MagicMock()
    message.tool_calls = mock_tool_calls
    message.content = None
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.mark.asyncio
async def test_simple_text_response(assistant, mock_openai_client):
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=make_text_response("Hello! How can I help you with stocks?")
    )
    result = await assistant.answer_query("Hello")
    assert "Hello" in result or "help" in result


@pytest.mark.asyncio
async def test_tool_call_and_response(assistant, mock_openai_client, mock_stock_service):
    tool_response = make_tool_call_response(
        [{"id": "call_1", "name": "get_stock_quote", "arguments": {"symbol": "AAPL"}}]
    )
    text_response = make_text_response(
        "Apple (AAPL) is trading at **$178.50**, up **$2.30 (+1.31%)** today."
    )

    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=[tool_response, text_response]
    )

    result = await assistant.answer_query("How is AAPL doing?")
    assert "178.50" in result
    mock_stock_service.get_quote.assert_called_once_with("AAPL")


@pytest.mark.asyncio
async def test_multiple_tool_calls(assistant, mock_openai_client, mock_stock_service):
    tool_response = make_tool_call_response(
        [
            {"id": "call_1", "name": "get_stock_quote", "arguments": {"symbol": "AAPL"}},
            {"id": "call_2", "name": "get_company_profile", "arguments": {"symbol": "AAPL"}},
        ]
    )
    text_response = make_text_response("Apple is doing well at $178.50.")

    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=[tool_response, text_response]
    )

    result = await assistant.answer_query("Tell me about AAPL")
    assert "178.50" in result
    mock_stock_service.get_quote.assert_called_once()
    mock_stock_service.get_company_profile.assert_called_once()


@pytest.mark.asyncio
async def test_max_iterations_safeguard(assistant, mock_openai_client):
    """Ensure we don't loop infinitely if OpenAI keeps requesting tools."""
    tool_response = make_tool_call_response(
        [{"id": "call_1", "name": "get_stock_quote", "arguments": {"symbol": "AAPL"}}]
    )
    mock_openai_client.chat.completions.create = AsyncMock(return_value=tool_response)

    result = await assistant.answer_query("AAPL")
    assert "couldn't complete" in result


@pytest.mark.asyncio
async def test_tool_error_handling(assistant, mock_openai_client, mock_stock_service):
    from app.services.stock_data import SymbolNotFoundError

    mock_stock_service.get_quote = AsyncMock(
        side_effect=SymbolNotFoundError("No data found for symbol 'INVALID'")
    )

    tool_response = make_tool_call_response(
        [{"id": "call_1", "name": "get_stock_quote", "arguments": {"symbol": "INVALID"}}]
    )
    text_response = make_text_response(
        "I couldn't find data for the symbol 'INVALID'. Please check the ticker symbol."
    )

    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=[tool_response, text_response]
    )

    result = await assistant.answer_query("How is INVALID doing?")
    assert "couldn't find" in result.lower() or "invalid" in result.lower()
