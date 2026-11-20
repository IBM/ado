# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
"""Shared helpers for ado create tests."""

import pathlib


def raw_operation(
    name: str,
    operation_type: str,
    inputs: dict | None = None,
) -> dict:
    """Return a raw operation configuration for the operator *name*.

    Args:
        name: Operator name.
        operation_type: Operation type of the operator.
        inputs: Value for the ``inputs`` field, omitted when ``None``.

    Returns:
        The raw operation configuration.
    """
    raw: dict = {
        "operation": {
            "module": {
                "operatorName": name,
                "operationType": operation_type,
            },
            "parameters": {},
        }
    }
    if inputs is not None:
        raw["inputs"] = inputs
    return raw


def write_operation(
    path: pathlib.Path,
    name: str,
    operation_type: str,
    inputs_yaml: str = "",
) -> None:
    """Write an operation YAML for the operator *name* to *path*.

    Args:
        path: File to write.
        name: Operator name.
        operation_type: Operation type of the operator.
        inputs_yaml: YAML text appended after the operation section.
    """
    path.write_text(
        "operation:\n"
        "  module:\n"
        f"    operatorName: {name}\n"
        f"    operationType: {operation_type}\n"
        "  parameters: {}\n"
        f"{inputs_yaml}"
    )
