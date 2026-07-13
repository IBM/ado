# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import enum
from collections.abc import Callable
from typing import TypeVar

import typer

from ado.cli.utils.resources.mappings import (
    cli_shorthands_to_cli_names,
)

T = TypeVar("T", bound=enum.Enum)


def parse_key_value_pairs(
    pairs: list[str] | None,
    separator: str = "=",
    allow_only_key: bool = False,
    invert_key_value: bool = False,
) -> list[dict[str, str | None]]:
    """
    Converts a list of key-value pairs into a list of dictionaries.

    Args:
        pairs (Optional[List[str]]): A list of strings representing key-value pairs.
        separator (str): The separator character used to split the key-value pairs. Defaults to "="
        allow_only_key (bool): Whether to allow only keys without values. Defaults to False.
        invert_key_value (bool): Whether to invert the key-value pairs. Defaults to False.

    Returns:
        list[dict[str, Optional[str]]]: A list of dictionaries containing the key-value pairs.
    """
    result = []

    if not pairs:
        return result

    for pair in pairs:
        split_result = pair.split(sep=separator)
        if len(split_result) != 2:  # noqa: PLR2004
            # There are instances where we want to allow just one element
            if allow_only_key and len(split_result) == 1:
                result.append({split_result[0]: None})
                continue

            # If we don't, we raise an exception
            raise ValueError(f"Key/Value pairs must be in form key{separator}value")

        if invert_key_value:
            result.append({split_result[1]: split_result[0]})
        else:
            result.append({split_result[0]: split_result[1]})

    return result


def resource_shorthands_to_full_names(value: str) -> str:
    """
    Resolves a resource shorthand to its full CLI name.

    Args:
        value (str): The shorthand or full CLI name of a resource.

    Returns:
        str: The full CLI name corresponding to the shorthand if found;
        otherwise, returns the original value.
    """
    return cli_shorthands_to_cli_names.get(value, value)


def enum_choice_parser(
    enum_type: type[T],
    case_sensitive: bool = True,
    resolve_shorthands: bool = False,
    handle_plurals: bool = False,
) -> Callable[[str], str]:
    """
    Create a parser function for enum choices with optional transformations.

    Typer will convert the returned string to the enum type based on the type annotation.

    Args:
        enum_type: The enum class to create choices from
        case_sensitive: Whether the choice matching should be case sensitive
        resolve_shorthands: Whether to resolve CLI shorthands to full names
        handle_plurals: Whether to strip trailing 's' for plural forms

    Returns:
        A parser function that validates input and returns the string value
    """
    choices = [member.value for member in enum_type]

    def parser(value: str) -> str:
        resolved_value = value

        # Apply transformations in order
        if handle_plurals:
            resolved_value = resolved_value.removesuffix("s")

        if resolve_shorthands:
            resolved_value = resource_shorthands_to_full_names(resolved_value)

        # Case-insensitive matching
        if not case_sensitive:
            resolved_lower = resolved_value.lower()
            for choice in choices:
                if choice.lower() == resolved_lower:
                    return choice

        # Exact match
        if resolved_value in choices:
            return resolved_value

        # Build error message
        error_msg = f"Invalid choice: '{value}'"
        if resolved_value != value:
            error_msg += f" (resolved to '{resolved_value}')"
        error_msg += f". Choose from: {', '.join(choices)}"

        raise typer.BadParameter(error_msg)

    return parser


def enum_choice_with_shorthand_parser(
    enum_type: type[T], case_sensitive: bool = True
) -> Callable[[str], str]:
    """
    Create a parser function that resolves shorthands and returns the validated string value.

    Typer will convert the returned string to the enum type based on the type annotation.

    Args:
        enum_type: The enum class to create choices from
        case_sensitive: Whether the choice matching should be case sensitive

    Returns:
        A parser function that resolves shorthands and returns the string value
    """
    return enum_choice_parser(enum_type, case_sensitive, resolve_shorthands=True)


def enum_choice_with_plural_parser(
    enum_type: type[T], case_sensitive: bool = True
) -> Callable[[str], str]:
    """
    Create a parser function that handles plurals, resolves shorthands, and returns the validated string value.

    Typer will convert the returned string to the enum type based on the type annotation.

    Args:
        enum_type: The enum class to create choices from
        case_sensitive: Whether the choice matching should be case sensitive

    Returns:
        A parser function that handles plurals and returns the string value
    """
    return enum_choice_parser(
        enum_type, case_sensitive, resolve_shorthands=True, handle_plurals=True
    )
