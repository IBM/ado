# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
from orchestrator.cli.utils.resources.mappings import cli_names_to_resource_kinds


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


def parse_core_resource_kinds(value: str) -> str:
    """
    Parses the core resource kind from a CLI name and returns its enum value.

    Args:
        value (str): The CLI name of the CoreResourceKind.

    Returns:
        str: The enum value associated with the CLI name.

    Raises:
        ValueError: If ``value`` is not a valid CoreResourceKind name.
    """
    if value not in cli_names_to_resource_kinds:
        raise ValueError(f"{value} is not a valid CoreResourceKind")
    return cli_names_to_resource_kinds[value].value
