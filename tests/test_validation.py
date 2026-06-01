import json

from app.services.validation import extract_numbers, validate_response


def test_extract_numbers_basic():
    numbers = extract_numbers("The price is $178.50 and it changed 1.31%")
    assert 178.50 in numbers
    assert 1.31 in numbers


def test_extract_numbers_with_commas():
    numbers = extract_numbers("Market cap is $2,800,000")
    assert 2800000 in numbers


def test_extract_numbers_empty():
    numbers = extract_numbers("No numbers here")
    assert len(numbers) == 0


def test_validate_correct_numbers():
    tool_results = [json.dumps({"current_price": 178.50, "change": 2.30})]
    text = "The stock is at $178.50, up $2.30."
    result, valid = validate_response(text, tool_results)
    assert valid is True
    assert "Note" not in result


def test_validate_fabricated_number():
    tool_results = [json.dumps({"current_price": 178.50})]
    text = "The stock is at $178.50, and could reach $250.00 soon."
    result, valid = validate_response(text, tool_results)
    assert valid is False
    assert "Note" in result


def test_validate_no_numbers_passes():
    tool_results = [json.dumps({"current_price": 178.50})]
    text = "Apple is a great company in the technology sector."
    result, valid = validate_response(text, tool_results)
    assert valid is True


def test_validate_no_tool_results_with_numbers():
    text = "The price is $500.00 today."
    result, valid = validate_response(text, [])
    assert valid is False
    assert "Note" in result


def test_validate_no_tool_results_no_numbers():
    text = "I can help you with stock information."
    result, valid = validate_response(text, [])
    assert valid is True
