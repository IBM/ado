# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
"""Utility functions for rendering resources to CLI output format in tests."""

import rich.box

from orchestrator.cli.utils.resources.formatters import (
    format_default_ado_get_single_resource,
)
from orchestrator.core import ADOResource
from orchestrator.utilities.rich import dataframe_to_rich_table, render_to_string


def render_ado_resources_to_cli_output(
    resources: ADOResource | list[ADOResource],
    show_index: bool = True,
) -> str:
    """Render ADO resource(s) to CLI output format as a string.

    This utility function creates the expected CLI output format for ADO resources,
    matching the format used by the 'ado get' commands. It's useful for testing
    CLI output in unit and integration tests.

    Args:
        resources: A single ADOResource or list of ADOResource objects to render.
            This includes OperationResource, SampleStoreResource, DiscoverySpaceResource,
            DataContainerResource, and ActuatorConfigurationResource.
        show_index: Whether to show the row index in the output table. Default is True.

    Returns:
        A string containing the rendered table output that can be compared against
        CLI command output in tests

    Example:
        >>> operation = OperationResource.model_validate(yaml.safe_load(...))
        >>> expected_output = render_ado_resources_to_cli_output(operation)
        >>> assert expected_output in result.output

        >>> sample_store = SampleStoreResource.model_validate(yaml.safe_load(...))
        >>> expected_output = render_ado_resources_to_cli_output(sample_store)
        >>> assert expected_output in result.output
    """
    # Normalize input to list
    if not isinstance(resources, list):
        resources = [resources]

    # Use the default formatter for each resource and concatenate DataFrames
    import pandas as pd

    dfs = []
    for resource in resources:
        df = format_default_ado_get_single_resource(
            resource=resource, show_details=False
        )
        dfs.append(df)

    # Concatenate all DataFrames and reset index
    combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # Render to rich table and convert to string
    return render_to_string(
        dataframe_to_rich_table(
            combined_df, show_edge=True, box=rich.box.SQUARE, show_index=show_index
        )
    )
