# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for annotation helpers."""

from typing import Annotated

from ado.utilities.annotations import unwrap_annotated


def test_unwrap_annotated_returns_the_underlying_type() -> None:
    """Annotated[T, ...] unwraps to T."""
    assert unwrap_annotated(Annotated[int, "meta"]) is int


def test_unwrap_annotated_returns_a_plain_type_unchanged() -> None:
    """A type that is not Annotated is returned as itself."""
    assert unwrap_annotated(int) is int
