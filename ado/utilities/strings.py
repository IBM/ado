# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT


def normalize_and_truncate_at_period(text: str) -> str:
    """
    Normalize whitespace and return the string up to the first period.

    The string is first stripped of leading/trailing whitespace, then all newlines
    are replaced with spaces, multiple consecutive spaces are normalized to single spaces,
    and finally the string is truncated at the first period (including the period).
    This handles triple-quoted strings gracefully by removing excess whitespace that
    may have been added for alignment.

    Args:
        text: The input string to process

    Returns:
        The processed string up to and including the first period. If no period is found,
        returns the entire processed string.

    Examples:
        >>> normalize_and_truncate_at_period("Hello world. More text")
        'Hello world.'
        >>> normalize_and_truncate_at_period("  Hello\\nworld. More text  ")
        'Hello world.'
        >>> normalize_and_truncate_at_period("First sentence.\\nSecond sentence.")
        'First sentence.'
        >>> normalize_and_truncate_at_period("No period here")
        'No period here'
        >>> normalize_and_truncate_at_period(\"\"\"This is a
        ...     triple quoted string
        ...     with alignment spaces. More text\"\"\")
        'This is a triple quoted string with alignment spaces.'
    """
    import re

    # Strip leading/trailing whitespace
    text = text.strip()

    # Replace newlines with spaces
    text = text.replace("\n", " ")

    # Normalize multiple spaces to single space
    text = re.sub(r"\s+", " ", text)

    # Find the first period
    period_index = text.find(".")

    # If period found, return up to and including it
    if period_index != -1:
        return text[: period_index + 1]

    # If no period found, return the entire processed string
    return text
