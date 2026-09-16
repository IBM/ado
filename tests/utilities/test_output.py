# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pytest

from ado.utilities.output import format_elided_list


def test_format_elided_list_empty() -> None:
    """An empty list renders as empty brackets."""
    assert format_elided_list([], None) == "[]"


def test_format_elided_list_no_truncation_when_max_items_none() -> None:
    """None as the item limit disables truncation."""
    items = list(range(20))
    result = format_elided_list(items, None)
    assert result == "[" + ", ".join(repr(v) for v in items) + "]"


def test_format_elided_list_no_truncation_when_at_limit() -> None:
    """A list at the item limit is not truncated."""
    items = [1, 2, 3]
    assert format_elided_list(items, 3) == "[1, 2, 3]"


def test_format_elided_list_truncated_even_max_items() -> None:
    """An even item limit retains equal-sized head and tail sections."""
    items = list("ABCDEF")
    result = format_elided_list(items, 4)
    assert result == "['A', 'B', ..., 'E', 'F'] (+2 more)"


def test_format_elided_list_truncated_odd_max_items() -> None:
    """An odd item limit retains one more head item than tail item."""
    items = list("ABCDE")
    result = format_elided_list(items, 3)
    assert result == "['A', 'B', ..., 'E'] (+2 more)"


def test_format_elided_list_truncated_min_max_items() -> None:
    """The minimum item limit retains the first and last items."""
    items = list("ABCDE")
    result = format_elided_list(items, 2)
    assert result == "['A', ..., 'E'] (+3 more)"


@pytest.mark.parametrize("max_items", [0, 1])
def test_format_elided_list_raises_when_max_items_less_than_two(
    max_items: int,
) -> None:
    """An item limit below two raises a ValueError."""
    with pytest.raises(ValueError, match="max_items must be at least 2"):
        format_elided_list([1, 2, 3], max_items)
