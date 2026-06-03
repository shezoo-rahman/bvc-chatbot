import re


def extract_numbers(text: str) -> set[float]:
    """Extract numeric values from text, including dollar amounts and percentages."""
    patterns = [
        r"[\$]?([\d,]+\.?\d*)",
        r"([\d,]+\.?\d*)\s*%",
    ]
    numbers = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            try:
                value = float(match.group(1).replace(",", ""))
                if value != 0:
                    numbers.add(value)
            except ValueError:
                continue
    return numbers


def validate_response(response_text: str, tool_results: list[str]) -> tuple[str, bool]:
    """Check if numbers in the response are grounded in tool results.

    Returns (possibly_modified_response, was_valid).
    """
    if not tool_results:
        response_numbers = extract_numbers(response_text)
        if response_numbers:
            return (
                response_text + "\n\n_Note: Some data may not reflect the latest values._",
                False,
            )
        return response_text, True

    tool_numbers = set()
    for result in tool_results:
        tool_numbers.update(extract_numbers(result))

    response_numbers = extract_numbers(response_text)
    if not response_numbers:
        return response_text, True

    # Check if response numbers are grounded in tool data (with tolerance for rounding)
    ungrounded = set()
    for num in response_numbers:
        grounded = False
        for tool_num in tool_numbers:
            if tool_num == 0:
                continue
            if abs(num - tool_num) / max(abs(tool_num), 1) < 0.01:
                grounded = True
                break
        if not grounded:
            ungrounded.add(num)

    if ungrounded:
        return (
            response_text + "\n\n_Note: Some data may not reflect the latest values._",
            False,
        )

    return response_text, True
