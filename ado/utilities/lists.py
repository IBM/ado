# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from typing import TypeVar

T = TypeVar("T")


def as_list(value: T | list[T]) -> list[T]:
    """Return *value* as a list.

    Args:
        value: A single item, or a list of items.

    Returns:
        *value* when it is a list, otherwise a one-element list holding *value*.
    """
    if isinstance(value, list):
        return value
    return [value]
