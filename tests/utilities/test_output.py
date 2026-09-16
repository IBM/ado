# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pytest

from ado.utilities.output import format_elided_list


def test_format_elided_list_empty() -> None:
    assert format_elided_list([], None) == "[]"


def test_format_elided_list_no_truncation_when_max_items_none() -> None:
    items = list(range(20))
    result = format_elided_list(items, None)
    assert result == "[" + ", ".join(repr(v) for v in items) + "]"


def test_format_elided_list_no_truncation_when_within_limit() -> None:
    items = [1, 2, 3]
    assert format_elided_list(items, 3) == "[1, 2, 3]"


def test_format_elided_list_no_truncation_when_below_limit() -> None:
    items = [1, 2]
    assert format_elided_list(items, 5) == "[1, 2]"


def test_format_elided_list_truncated_even_max_items() -> None:
    # max_items=4: head=2, tail=2; items 2..7 (6 items) → omits 2
    items = list("ABCDEF")
    result = format_elided_list(items, 4)
    assert result == "['A', 'B', ..., 'E', 'F'] (+2 more)"


def test_format_elided_list_truncated_odd_max_items() -> None:
    # max_items=3: head=ceil(3/2)=2, tail=1; items 5 → omits 2
    items = list("ABCDE")
    result = format_elided_list(items, 3)
    assert result == "['A', 'B', ..., 'E'] (+2 more)"


def test_format_elided_list_truncated_min_max_items() -> None:
    # max_items=2: head=1, tail=1
    items = list("ABCDE")
    result = format_elided_list(items, 2)
    assert result == "['A', ..., 'E'] (+3 more)"


def test_format_elided_list_string_items() -> None:
    items = [f"item-{i}" for i in range(10)]
    result = format_elided_list(items, 4)
    assert result == "['item-0', 'item-1', ..., 'item-8', 'item-9'] (+6 more)"


def test_format_elided_list_raises_when_max_items_less_than_2() -> None:
    with pytest.raises(ValueError, match="max_items must be at least 2"):
        format_elided_list([1, 2, 3], 1)


def test_format_elided_list_raises_when_max_items_zero() -> None:
    with pytest.raises(ValueError, match="max_items must be at least 2"):
        format_elided_list([1, 2, 3], 0)
