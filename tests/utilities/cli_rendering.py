# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
"""Utility functions for rendering resources to CLI output format in tests."""

import pandas as pd
import rich.box

from orchestrator.cli.utils.resources.formatters import (
    most_important_status_update,
    time_since_timestamp,
    timedelta_to_string,
)
from orchestrator.core import OperationResource
from orchestrator.utilities.rich import dataframe_to_rich_table, render_to_string


def render_operations_to_cli_output(
    operations: OperationResource | list[OperationResource],
    show_index: bool = True,
) -> str:
    """Render operation(s) to CLI output format as a string.

    This utility function creates the expected CLI output format for operations,
    matching the format used by the 'ado get operations' command. It's useful
    for testing CLI output in unit and integration tests.

    Args:
        operations: A single OperationResource or list of OperationResource objects
            to render
        show_index: Whether to show the row index in the output table. Default is True.

    Returns:
        A string containing the rendered table output that can be compared against
        CLI command output in tests

    Example:
        >>> operation = OperationResource.model_validate(yaml.safe_load(...))
        >>> expected_output = render_operations_to_cli_output(operation)
        >>> assert expected_output in result.output
    """
    # Normalize input to list
    if not isinstance(operations, list):
        operations = [operations]

    # Build DataFrame with operation data
    data = {
        "IDENTIFIER": [],
        "NAME": [],
        "STATUS": [],
        "EXIT_STATE": [],
        "AGE": [],
    }

    for operation in operations:
        data["IDENTIFIER"].append(operation.identifier)
        data["NAME"].append(operation.config.metadata.name or "")

        status_update = most_important_status_update(operation.status)
        data["STATUS"].append(status_update.event.value)
        data["EXIT_STATE"].append(
            status_update.exit_state.value if status_update.exit_state else "N/A"
        )

        data["AGE"].append(
            timedelta_to_string(time_since_timestamp(operation.created).total_seconds())
        )

    df = pd.DataFrame(data)

    # Render to rich table and convert to string
    return render_to_string(
        dataframe_to_rich_table(
            df, show_edge=True, box=rich.box.SQUARE, show_index=show_index
        )
    )
