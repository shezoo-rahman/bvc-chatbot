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
async def test_search_then_quote(assistant, mock_openai_client, mock_stock_service):
    search_response = make_tool_call_response(
        [{"id": "call_1", "name": "search_symbol", "arguments": {"query": "Apple"}}]
    )
    quote_response = make_tool_call_response(
        [{"id": "call_2", "name": "get_stock_quote", "arguments": {"symbol": "AAPL"}}]
    )
    text_response = make_text_response(
        "Apple (AAPL) is trading at **$178.50**, up **$2.30 (+1.31%)** today."
    )
    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=[search_response, quote_response, text_response]
    )
    result = await assistant.answer_query("How is Apple doing?")
    assert "178.50" in result
    mock_stock_service.search_symbol.assert_called_once_with("Apple")


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


@pytest.mark.asyncio
async def test_conversation_history_preserved(assistant, mock_openai_client):
    """Messages from previous turns are included when using a session_id."""
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=make_text_response("AAPL is at $178.50.")
    )
    await assistant.answer_query("How is AAPL?", session_id="sess-1")

    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=make_text_response("The high today is $179.20.")
    )
    await assistant.answer_query("What about the high?", session_id="sess-1")

    # Check that the second call included prior conversation history
    call_args = mock_openai_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    roles = [m["role"] for m in messages]
    # system, prior-user, prior-assistant, current-user
    assert roles == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "How is AAPL?"
    assert messages[3]["content"] == "What about the high?"


@pytest.mark.asyncio
async def test_no_history_without_session_id(assistant, mock_openai_client):
    """Without a session_id, each call is stateless."""
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=make_text_response("AAPL is at $178.50.")
    )
    await assistant.answer_query("How is AAPL?")
    await assistant.answer_query("What about the high?")

    call_args = mock_openai_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    # Only system + current user, no history
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "What about the high?"


@pytest.mark.asyncio
async def test_malformed_tool_arguments(assistant, mock_openai_client):
    """Malformed JSON from the LLM is handled gracefully."""
    mock_tc = MagicMock()
    mock_tc.id = "call_bad"
    mock_tc.function.name = "get_stock_quote"
    mock_tc.function.arguments = "not valid json{{"

    message = MagicMock()
    message.tool_calls = [mock_tc]
    message.content = None
    choice = MagicMock()
    choice.message = message
    tool_response = MagicMock()
    tool_response.choices = [choice]

    text_response = make_text_response("Sorry, something went wrong.")

    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=[tool_response, text_response]
    )

    result = await assistant.answer_query("AAPL")
    assert "sorry" in result.lower() or "wrong" in result.lower()


@pytest.mark.asyncio
async def test_unknown_tool_name(assistant, mock_openai_client):
    """An unknown tool name from the LLM returns an error to the model."""
    tool_response = make_tool_call_response(
        [{"id": "call_1", "name": "nonexistent_tool", "arguments": {"foo": "bar"}}]
    )
    text_response = make_text_response("I encountered an issue.")

    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=[tool_response, text_response]
    )

    result = await assistant.answer_query("test")
    assert "issue" in result.lower() or "error" in result.lower()
