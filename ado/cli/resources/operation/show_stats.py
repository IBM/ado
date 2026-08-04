# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typer
from rich.status import Status

from ado.cli.models.parameters import AdoShowStatsCommandParameters
from ado.cli.utils.generic.wrappers import get_sql_store
from ado.cli.utils.output.prints import (
    ADO_INFO_EMPTY_DATAFRAME,
    ADO_SPINNER_GETTING_OUTPUT_READY,
    ADO_SPINNER_QUERYING_DB,
    ERROR,
    console_print,
)
from ado.cli.utils.output.stats import render_stats_dataframe
from ado.cli.utils.resources.formatters import (
    build_resource_listing_dataframe,
    format_ado_get_stats_for_operations,
    format_default_ado_get_multiple_resources,
)
from ado.core.resources import CoreResourceKinds


def show_operation_stats(parameters: AdoShowStatsCommandParameters) -> None:
    """Show statistics for one or more operations.

    Outputs all standard ``ado get`` table columns (IDENTIFIER, NAME, AGE,
    SPACE, STATUS, EXIT_STATE) plus result-level stats columns
    (TOTAL_RESULTS, SUCCESSFUL_RESULTS, FAILED_RESULTS, MEASURED_ENTITIES
    — includes failed measurements)
    and request-level stats columns (TOTAL_REQUESTS, FAILED_REQUESTS,
    SUCCESSFUL_REQUESTS).

    Args:
        parameters: Command parameters including resource IDs, output format,
            output file, query filters, and render flag.
    """
    if parameters.filters and parameters.resource_ids:
        console_print(
            f"{ERROR}You cannot specify operation ids and filters/labels at the same time",
            stderr=True,
        )
        raise typer.Exit(1)

    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )
    with Status(ADO_SPINNER_QUERYING_DB) as status:
        if parameters.resource_ids:
            operation_resources = sql_store.getResources(parameters.resource_ids)
            base_df = format_default_ado_get_multiple_resources(
                resources=build_resource_listing_dataframe(
                    resources=operation_resources,
                    resource_kind=CoreResourceKinds.OPERATION,
                    show_details=parameters.show_details,
                ),
                resource_kind=CoreResourceKinds.OPERATION,
            )
        else:
            operation_resources = sql_store.getResourceIdentifiersOfKind(
                kind=CoreResourceKinds.OPERATION.value,
                field_selectors=parameters.filters,
                details=parameters.show_details,
            )
            base_df = format_default_ado_get_multiple_resources(
                resources=operation_resources,
                resource_kind=CoreResourceKinds.OPERATION,
            )

        if base_df.empty:
            if parameters.filters:
                console_print(
                    f"{ERROR}The filter/labels provided did not match any operation.",
                    stderr=True,
                )
                raise typer.Exit(1)
            if parameters.resource_ids:
                console_print(
                    f"{ERROR}No data was retrieved for any of the operations: {parameters.resource_ids}",
                    stderr=True,
                )
                raise typer.Exit(1)
            console_print(ADO_INFO_EMPTY_DATAFRAME, stderr=True)
            return

        status.update(ADO_SPINNER_GETTING_OUTPUT_READY)
        df = format_ado_get_stats_for_operations(
            base_df,
            sql_store,
            spinner=status,
            include_request_columns=True,
        )

    render_stats_dataframe(
        df=df,
        output_format=parameters.output_format,
        output_file=parameters.output_file,
        render_output=parameters.render_output,
    )
